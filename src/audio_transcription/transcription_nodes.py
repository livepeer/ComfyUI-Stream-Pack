"""
Real-time Audio Transcription Node for ComfyUI.

This node buffers audio segments and performs transcription using faster-whisper,
with controlled output timing to prevent message flooding.
"""

import asyncio
import json
import logging
import time
import tempfile
import os
import numpy as np
from typing import Optional, List, Deque, Dict, Any
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class CriticalTranscriptionError(Exception):
    """Raised for unrecoverable errors where the stream should stop."""
    pass


@dataclass
class TranscriptionSegment:
    """Represents a transcribed segment with timing information."""
    start: float  # Start time in seconds (buffer-relative)
    end: float    # End time in seconds (buffer-relative)
    text: str     # Transcribed text
    confidence: float = 0.0  # Confidence score


class AudioTranscriptionNode:
    """
    Real-time audio transcription node that buffers audio and outputs 
    transcribed text on a controlled schedule to prevent message flooding.
    """
    
    # ------------------------------------------------------------------
    # Configurable class attributes (easy future exposure via UI):
    #   MAX_TOTAL_BUFFER_SECONDS : Hard cap on rolling audio kept in memory
    #   OVERLAP_RATIO            : Portion of previous buffer kept for next pass (0 < r < 1)
    #   TRANSCRIPTION_QUEUE_MAXLEN: Max queued outputs waiting for timing window
    #   WORD_TIMESTAMPS / BEAM_SIZE / BEST_OF : Whisper inference params
    #   DEFAULT_VAD              : Default Voice Activity Detection toggle
    # ------------------------------------------------------------------
    MAX_TOTAL_BUFFER_SECONDS = 60
    OVERLAP_RATIO = 0.25 # 25% overlap by default
    TRANSCRIPTION_QUEUE_MAXLEN = 100
    WORD_TIMESTAMPS = True
    BEAM_SIZE = 1
    BEST_OF = 1
    DEFAULT_VAD = True

    CATEGORY = "audio_utils"
    RETURN_TYPES = ("STRING",)
    
    def __init__(self):
        # Audio buffering
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_duration = 0.0  # Duration in seconds
        self.sample_rate = None
        self.buffer_samples = None
        self.max_total_buffer_samples = None  # derived once sample_rate known
        
        # Whisper model (per-instance lazy load)
        self._whisper_model = None
        
        # Transcription queue for controlled output
        self.transcription_queue: Deque[str] = deque(maxlen=self.TRANSCRIPTION_QUEUE_MAXLEN)
        self.last_output_time = 0.0
        
        # Processing state
        self.total_audio_processed = 0.0  # Total audio time processed
        self.transcription_count = 0
        

        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("WAVEFORM",),  # Input from LoadAudioTensor
                "transcription_interval": ("FLOAT", {
                    "default": 2.0, 
                    "min": 1.0, 
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Minimum seconds between transcription outputs"
                }),
                "buffer_duration": ("FLOAT", {
                    "default": 4.0,
                    "min": 4.0, 
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Audio buffer duration in seconds for transcription"
                }),
                "whisper_model": (["tiny", "base", "small", "medium", "large-v2"], {
                    "default": "base",
                    "tooltip": "Whisper model size (larger = more accurate but slower)"
                }),
                "language": (["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"], {
                    "default": "auto",
                    "tooltip": "Language for transcription (auto = auto-detect)"
                }),
                "enable_vad": ("BOOLEAN", {
                    "default": cls.DEFAULT_VAD,
                    "tooltip": "Enable Voice Activity Detection to filter silence"
                })
            },
            "optional": {
                "output_format": (["text", "json_segments", "json_words"], {
                    "default": "json_segments",
                    "tooltip": "Output format: text (simple), json_segments (with timing), json_words (word-level timing)"
                })
            }
        }
    
    FUNCTION = "execute"

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")
    


    def _load_whisper_model_now(self, model_size: str):
        """Load the Whisper model using faster-whisper's automatic download."""
        # Use CPU for compatibility, can be changed to CUDA if available
        device = "cpu"
        try:
            # Try CUDA first if available
            import torch  # type: ignore
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                compute_type = "int8"
        except Exception:
            compute_type = "int8"

        logger.info(f"Loading Whisper model via faster-whisper: {model_size}")
        try:
            self._whisper_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            # Attach size name for change detection
            self._whisper_model.model_size_name = model_size  # type: ignore[attr-defined]
            logger.info(f"Whisper model '{model_size}' loaded successfully on {device}")
        except (MemoryError, RuntimeError) as e:
            logger.error(f"Critical: Failed to load Whisper model '{model_size}' due to resource issue: {e}")
            raise CriticalTranscriptionError(f"Model load failed (resources): {e}") from e
        except OSError as e:
            logger.error(f"Critical: Filesystem/model download issue for '{model_size}': {e}")
            raise CriticalTranscriptionError(f"Filesystem/model download failure: {e}") from e
        except Exception as e:
            logger.error(f"Critical: Unexpected error loading model '{model_size}': {e}")
            raise CriticalTranscriptionError(f"Unexpected model load error: {e}") from e

    def _ensure_model_loaded(self, model_size: str) -> bool:
        """Ensure the Whisper model is loaded. Returns True if a fresh load occurred."""
        if (self._whisper_model is None or 
            getattr(self._whisper_model, 'model_size_name', '') != model_size):
            logger.info(f"Loading Whisper model: {model_size}")
            self._load_whisper_model_now(model_size)
            return True
        return False
    
    def _buffer_audio(self, audio_input: np.ndarray, sample_rate: int, buffer_duration: float):
        """
        Efficiently buffer audio data optimized for Whisper transcription.
        
        This is the primary buffering system - no duplicate buffering from LoadAudioTensor.
        """
        # Initialize buffer parameters
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            self.buffer_samples = int(self.sample_rate * buffer_duration)
            self.max_total_buffer_samples = int(self.sample_rate * self.MAX_TOTAL_BUFFER_SECONDS)
            logger.info(f"Initialized Whisper-optimized audio buffer: {buffer_duration}s at {sample_rate}Hz = {self.buffer_samples} samples")
        
        # Skip empty frames (from streaming loader)
        if audio_input is None or audio_input.size == 0:
            return False
        
        audio_input = self._normalize_audio_for_whisper(audio_input)
        
        # Add to buffer - concatenate efficiently
        if self.audio_buffer.size == 0:
            self.audio_buffer = audio_input.copy()
        else:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
        
        # Enforce maximum total buffer size (keep most recent samples)
        if self.max_total_buffer_samples and len(self.audio_buffer) > self.max_total_buffer_samples:
            self.audio_buffer = self.audio_buffer[-self.max_total_buffer_samples:]
        
        # Update buffer duration
        self.buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # Check if we have enough audio for transcription
        if self.buffer_samples is None:
            return False
        ready = len(self.audio_buffer) >= self.buffer_samples
        
        if ready:
            logger.debug(f"Audio buffer ready for transcription: {self.buffer_duration:.2f}s ({len(self.audio_buffer)} samples)")
        
        return ready
    
    def _normalize_audio_for_whisper(self, audio_input: np.ndarray) -> np.ndarray:
        """
        Minimal audio normalization to avoid artifacts.
        Assumes LoadAudioTensorStream has already done most processing.
        """
        if audio_input is None or audio_input.size == 0:
            return np.array([], dtype=np.int16)
        
        # Ensure numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)
        
        # If it's already int16 and 1D, return as-is
        if audio_input.dtype == np.int16 and audio_input.ndim == 1:
            return audio_input
        
        # Handle multi-channel by taking first channel only
        if audio_input.ndim > 1:
            audio_input = audio_input.flatten() if audio_input.size < 1000 else audio_input[:, 0] if audio_input.shape[1] > audio_input.shape[0] else audio_input[0, :]
        
        # Convert to int16 only if necessary
        if audio_input.dtype != np.int16:
            if audio_input.dtype in [np.float32, np.float64]:
                # Very conservative conversion to avoid clipping artifacts
                audio_input = np.clip(audio_input, -0.99, 0.99)  # Leave headroom
                audio_input = (audio_input * 32000).astype(np.int16)  # Slightly lower scale
            else:
                audio_input = audio_input.astype(np.int16)
        
        return audio_input
    

    
    def _transcribe_audio_buffer(self, model_size: str, language: str, enable_vad: bool, batch_size: int, output_format: str = "json_segments") -> Optional[str]:
        """Transcribe the current audio buffer and return combined text."""
        if self.buffer_samples is None or len(self.audio_buffer) < self.buffer_samples:
            return None
        try:
            # Ensure model is loaded; if it was freshly loaded, emit a single warmup sentinel
            send_warmup_sentinel = self._ensure_model_loaded(model_size)
            # Emit a single warmup sentinel immediately after model loading completes
            if send_warmup_sentinel:
                if output_format == "text":
                    result = "__WARMUP_SENTINEL__"
                elif output_format == "json_segments":
                    result = json.dumps([{ "start": 0.0, "end": 1.0, "text": "__WARMUP_SENTINEL__"}])
                elif output_format == "json_words":
                    result = json.dumps([{ "start": 0.0, "end": 1.0, "word": "__WARMUP_SENTINEL__", "probability": 1.0}])
                else:
                    result = "__WARMUP_SENTINEL__"
                logger.debug("Warmup sentinel emitted once after model load")
                return result

            # Extract audio chunk for transcription
            audio_chunk = self.audio_buffer[:self.buffer_samples]
            
            # Validate audio chunk quality before transcription
            if len(audio_chunk) != self.buffer_samples:
                logger.warning(f"Audio chunk size mismatch: expected {self.buffer_samples}, got {len(audio_chunk)}")
                return None
            
            # Check for audio artifacts (all zeros, extreme values, etc.)
            if np.all(audio_chunk == 0):
                return ""
            
            # Check for clipping or extreme values that might cause hallucinations
            if np.abs(audio_chunk).max() < 100:  # Very quiet audio
                return ""

            # Save audio to temporary WAV file for whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Write clean WAV file to avoid format conversion artifacts
                try:
                    from scipy.io.wavfile import write  # type: ignore
                    # Ensure audio is properly scaled for WAV format
                    clean_audio = np.copy(audio_chunk)
                    write(temp_path, int(self.sample_rate or 16000), clean_audio)
                except ImportError:
                    # Fallback to manual WAV writing
                    self._write_wav_file(temp_path, audio_chunk, int(self.sample_rate or 16000))

                language_code = None if language == "auto" else language
                segments, info = self._whisper_model.transcribe(  # type: ignore[attr-defined]
                    temp_path,
                    language=language_code,
                    word_timestamps=self.WORD_TIMESTAMPS,
                    vad_filter=enable_vad,
                    beam_size=self.BEAM_SIZE,
                    best_of=self.BEST_OF
                )

                # Simple format output without complex deduplication (causes accuracy issues)
                if output_format == "text":
                    texts = [s.text.strip() for s in segments if s.text and s.text.strip()]
                    result = " ".join(texts)
                elif output_format == "json_segments":
                    seg_list = [
                        {"start": s.start, "end": s.end, "text": s.text.strip()}
                        for s in segments if s.text and s.text.strip()
                    ]
                    result = json.dumps(seg_list, ensure_ascii=False)
                elif output_format == "json_words":
                    words = []
                    for s in segments:
                        s_words = getattr(s, 'words', None)
                        if s_words:
                            for w in s_words:
                                if w.word.strip():
                                    words.append({
                                        "start": w.start,
                                        "end": w.end,
                                        "word": w.word.strip(),
                                        "probability": getattr(w, 'probability', 1.0)
                                    })
                        elif s.text and s.text.strip():
                            words.append({
                                "start": s.start,
                                "end": s.end,
                                "word": s.text.strip(),
                                "probability": 1.0
                            })
                    result = json.dumps(words, ensure_ascii=False)
                else:
                    result = None

                if not result or (output_format in ["json_segments", "json_words"] and result == "[]"):
                    result = ""  # Return empty string for ComfyUI compatibility
                    logger.debug(f"No transcription produced, returning empty")


                # Metrics
                if self.sample_rate and self.buffer_samples:
                    self.total_audio_processed += self.buffer_samples / self.sample_rate
                self.transcription_count += 1

                # Advance buffer with configurable overlap
                if self.buffer_samples:
                    overlap_samples = int(self.buffer_samples * self.OVERLAP_RATIO)
                    if overlap_samples < 0:
                        overlap_samples = 0
                    if overlap_samples >= self.buffer_samples:
                        overlap_samples = self.buffer_samples - 1
                    advance_samples = self.buffer_samples - overlap_samples
                    self.audio_buffer = self.audio_buffer[advance_samples:]

                if result:
                    logger.debug(
                        f"Transcribed chunk {self.transcription_count} ({output_format}): '{str(result)[:50]}...' ({len(str(result))} chars)"
                    )
                else:
                    logger.debug(f"Transcribed chunk {self.transcription_count} produced no result")
                return result
            finally:
                try:
                    os.unlink(temp_path)
                except FileNotFoundError:
                    logger.debug(f"Temp transcription file already removed: {temp_path}")
                except PermissionError as e:
                    logger.warning(f"Permission error removing temp file {temp_path}: {e}")
                except OSError as e:
                    logger.warning(f"OS error removing temp file {temp_path}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error removing temp file {temp_path}: {e}")
        except CriticalTranscriptionError:
            # Propagate critical to stop workflow
            raise
        except (MemoryError, OSError) as e:
            logger.error(f"Critical transcription failure: {e}")
            raise CriticalTranscriptionError(f"Critical transcription failure: {e}") from e
        except Exception as e:
            logger.warning(f"Non-critical transcription error, skipping chunk: {e}")
            # Advance half buffer to avoid being stuck
            if self.buffer_samples:
                self.audio_buffer = self.audio_buffer[self.buffer_samples // 2:]
            return None
    
    def _write_wav_file(self, filename: str, audio_data: np.ndarray, sample_rate: int):
        """Write WAV file manually if scipy is not available."""
        import struct
        import wave
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
    
    def _should_output_transcription(self, transcription_interval: float) -> bool:
        """Check if we should output transcription based on queue or timing."""
        # For real-time streaming, prioritize queue content
        if len(self.transcription_queue) > 0:
            # logger.debug(f"Queue has {len(self.transcription_queue)} items - outputting immediately")  # Too frequent
            return True
            
        # Fallback to timing interval only if queue is empty
        current_time = time.time()
        
        effective_interval = transcription_interval
            
        return (current_time - self.last_output_time) >= effective_interval
    
    def _get_queued_transcription(self) -> Optional[str]:
        """Get transcription from queue if available."""
        try:
            return self.transcription_queue.popleft()
        except IndexError:
            return None
    
    def execute(self, audio, transcription_interval=8.0, buffer_duration=8.0, 
                whisper_model="base", language="auto", enable_vad=True, output_format="json_segments"):
        """
        Execute transcription on streaming audio input.
        
        Args:
            audio: Audio input from LoadAudioTensorStream - tuple of (audio_data, sample_rate)
            transcription_interval: Minimum seconds between outputs (prevents message flooding)
            buffer_duration: Audio buffer duration for optimal Whisper transcription
            whisper_model: Whisper model size (tiny/base/small/medium/large-v2)
            language: Language for transcription (auto for auto-detection)
            enable_vad: Enable voice activity detection to filter silence
            output_format: Output format (text/json_segments/json_words)
            
        Returns:
            Tuple containing transcribed text (empty string if not ready to output)
        """
        try:
            # Parse audio input - expect (audio_data, sample_rate) from LoadAudioTensorStream
            if isinstance(audio, tuple) and len(audio) == 2:
                audio_data, sample_rate = audio
            elif hasattr(audio, 'shape'):
                # Fallback for direct numpy array (backward compatibility)
                audio_data = audio
                sample_rate = 16000  # Default Whisper-optimized rate
                logger.debug("Using fallback audio format detection")
            else:
                logger.warning(f"Unexpected audio format: {type(audio)}, returning empty")
                return ("",)
            
            # Validate sample rate for Whisper optimization
            if sample_rate != 16000:
                logger.warning(f"Non-optimal sample rate {sample_rate}Hz for Whisper (16kHz recommended)")
            
            # Ensure audio_data is a numpy array before buffering
            if not isinstance(audio_data, np.ndarray):
                logger.debug(f"Converting audio_data from {type(audio_data)} to numpy array.")
                audio_data = np.array(audio_data, dtype=np.float32)

            
            # Buffer the audio (primary buffering system - no duplicate buffering)
            ready_for_transcription = self._buffer_audio(audio_data, sample_rate, buffer_duration)
            
            # Transcribe if buffer has enough audio
            if ready_for_transcription:
                transcription = self._transcribe_audio_buffer(whisper_model, language, enable_vad, 16, output_format)
                if transcription and transcription.strip() and len(transcription.strip()) > 5:
                    # Add to queue for controlled output timing (only substantial content or warmup sentinels)
                    self._queue_transcription(transcription)
                    is_sentinel = "__WARMUP_SENTINEL__" in transcription
                    if is_sentinel:
                        logger.debug(f"Queued warmup sentinel ({output_format})")
                    else:
                        logger.debug(f"Queued transcription ({output_format}): '{transcription[:50]}...' (length: {len(transcription)})")
                elif transcription and transcription != "":
                    logger.debug(f"Skipping minimal transcription: '{transcription}' (too short)")
                elif transcription == "":
                    logger.debug(f"Empty transcription result (post-warmup silence)")
                else:
                    logger.debug(f"No transcription result returned")
            
            # Output transcriptions immediately when available (real-time streaming)
            if self._should_output_transcription(transcription_interval):
                output_text = self._get_queued_transcription()
                if output_text and output_text.strip():
                    self.last_output_time = time.time()
                    
                    # Log differently for sentinel vs regular transcription
                    if "__WARMUP_SENTINEL__" in output_text:
                        logger.info(f"Outputting warmup sentinel immediately ({output_format}, {len(output_text)} chars): '{output_text[:100]}...'")
                    else:
                        logger.info(f"Outputting transcription immediately ({output_format}, {len(output_text)} chars): '{output_text[:100]}...' (queue: {len(self.transcription_queue)})")
                    
                    return (output_text,)
            
            # Return empty string if not ready to output (SaveTextTensor will filter empty content)
            return ("",)
            
        except CriticalTranscriptionError as e:
            logger.error(f"Critical error in transcription node: {e}")
            raise
        except Exception as e:
            logger.error(f"Recoverable execute error: {e}")
            return ("",)
    
    def _queue_transcription(self, transcription: str):
        """Safely queue transcription for controlled output."""
        if len(self.transcription_queue) == self.transcription_queue.maxlen:
            logger.debug("Transcription queue is full, oldest entry will be replaced.")
        self.transcription_queue.append(transcription)

