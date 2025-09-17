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
    #   TRANSCRIPTION_QUEUE_MAXLEN: Max queued outputs waiting for timing window
    #   WORD_TIMESTAMPS / BEAM_SIZE / BEST_OF : Whisper inference params
    #   DEFAULT_VAD              : Default Voice Activity Detection toggle
    #   MAX_BUFFER_DURATION      : Safety limit to prevent memory exhaustion
    #   MAX_BUFFER_SAMPLES       : Hard limit on buffer size in samples
    #   
    #   RECOMMENDED CONFIGURATIONS:
    #   - Fast response: accumulation_duration=2.0, audio_chunk_size_ms=1000.0
    #   - Balanced: accumulation_duration=4.0, audio_chunk_size_ms=2000.0  
    #   - High quality: accumulation_duration=8.0, audio_chunk_size_ms=4000.0
    # ------------------------------------------------------------------
    TRANSCRIPTION_QUEUE_MAXLEN = 100
    WORD_TIMESTAMPS = True
    BEAM_SIZE = 1
    BEST_OF = 1
    DEFAULT_VAD = True
    
    # Safety limits to prevent unbounded buffer growth
    MAX_BUFFER_DURATION = 60.0  # Maximum 60 seconds of audio in buffer
    MAX_BUFFER_SAMPLES = 16000 * 60  # 60 seconds at 16kHz (960k samples)

    CATEGORY = "audio_utils"
    RETURN_TYPES = ("STRING",)
    
    def __init__(self):
        # Smart accumulation buffer for better transcription quality (optimized for real-time)
        self.sample_rate = None
        self.accumulation_buffer = np.empty(0, dtype=np.int16)
        self.accumulation_duration = 0.0  # Current buffer duration in seconds
        self.target_accumulation_duration = 3.0  # Target duration for optimal real-time Whisper results
        
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
                "sample_rate": ("INT",),  # Sample rate from LoadAudioTensor
                "transcription_interval": ("FLOAT", {
                    "default": 2.0, 
                    "min": 1.0, 
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Minimum seconds between transcription outputs (optimized for real-time)"
                }),
                "accumulation_duration": ("FLOAT", {
                    "default": 3.0,
                    "min": 2.0,
                    "max": 10.0,
                    "step": 0.5,
                    "tooltip": "Audio accumulation duration for optimal Whisper transcription (reduced for real-time: shorter = faster output, longer = better quality)"
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
    
    def _process_audio_chunk(self, audio_input: np.ndarray, sample_rate: int, accumulation_duration: float):
        """
        Process a single audio chunk and accumulate it for optimal Whisper transcription.
        Returns True when we have enough audio for transcription.
        """
        # Initialize sample rate on first call
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            self.target_accumulation_duration = accumulation_duration
            logger.info(f"Initialized transcription node with sample rate: {sample_rate}Hz, target accumulation: {accumulation_duration}s")
        
        # Skip empty frames
        if audio_input is None or audio_input.size == 0:
            return False
        
        # Normalize audio for Whisper
        normalized_audio = self._normalize_audio_for_whisper(audio_input)
        
        # Add to accumulation buffer
        if self.accumulation_buffer.size == 0:
            self.accumulation_buffer = normalized_audio.copy()
        else:
            self.accumulation_buffer = np.concatenate([self.accumulation_buffer, normalized_audio])
        
        # Update accumulation duration
        self.accumulation_duration = len(self.accumulation_buffer) / self.sample_rate
        
        # Safety check: enforce buffer limits to prevent memory exhaustion
        self._enforce_buffer_limits()
        
        # Check if we have enough audio for transcription
        ready = self.accumulation_duration >= self.target_accumulation_duration
        
        if ready:
            logger.debug(f"Audio accumulation ready: {self.accumulation_duration:.2f}s >= {self.target_accumulation_duration}s ({len(self.accumulation_buffer)} samples)")
        
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
    
    def _enforce_buffer_limits(self):
        """
        Enforce safety limits on the accumulation buffer to prevent memory exhaustion.
        Trims buffer from the beginning if limits are exceeded.
        """
        if self.sample_rate is None or self.accumulation_buffer.size == 0:
            return
        
        # Check duration limit
        if self.accumulation_duration > self.MAX_BUFFER_DURATION:
            # Calculate how many samples to keep (keep the most recent audio)
            samples_to_keep = int(self.MAX_BUFFER_DURATION * self.sample_rate)
            if samples_to_keep < len(self.accumulation_buffer):
                # Trim from the beginning, keep the end
                self.accumulation_buffer = self.accumulation_buffer[-samples_to_keep:]
                self.accumulation_duration = len(self.accumulation_buffer) / self.sample_rate
                logger.warning(f"Buffer duration limit exceeded, trimmed to {self.accumulation_duration:.2f}s (kept most recent {self.MAX_BUFFER_DURATION}s)")
        
        # Check sample count limit (secondary safety check)
        if len(self.accumulation_buffer) > self.MAX_BUFFER_SAMPLES:
            self.accumulation_buffer = self.accumulation_buffer[-self.MAX_BUFFER_SAMPLES:]
            self.accumulation_duration = len(self.accumulation_buffer) / self.sample_rate
            logger.warning(f"Buffer sample limit exceeded, trimmed to {len(self.accumulation_buffer)} samples ({self.accumulation_duration:.2f}s)")

    
    def _transcribe_accumulated_audio(self, model_size: str, language: str, enable_vad: bool, batch_size: int, output_format: str = "json_segments") -> Optional[str]:
        """Transcribe the accumulated audio buffer and return combined text."""
        if self.accumulation_buffer.size == 0:
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

            # Use the accumulated audio buffer
            audio_chunk = self.accumulation_buffer
            
            # Validate audio buffer quality before transcription
            if len(audio_chunk) == 0:
                logger.warning("Audio buffer is empty")
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
                if self.sample_rate and self.accumulation_buffer.size > 0:
                    self.total_audio_processed += len(self.accumulation_buffer) / self.sample_rate
                self.transcription_count += 1

                # Clear the accumulation buffer after processing (no overlap to prevent duplication)
                self.accumulation_buffer = np.empty(0, dtype=np.int16)
                self.accumulation_duration = 0.0
                logger.debug("Cleared accumulation buffer for next cycle")

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
    
    def execute(self, audio, sample_rate, transcription_interval, accumulation_duration,
                whisper_model="base", language="auto", enable_vad=True, output_format="json_segments"):
        """
        Execute transcription on streaming audio input.
        
        Args:
            audio: Audio input from LoadAudioTensor - numpy array of audio data
            sample_rate: Sample rate of the audio (from LoadAudioTensor)
            transcription_interval: Minimum seconds between outputs (prevents message flooding) - optimized for real-time
            accumulation_duration: Audio accumulation duration for optimal Whisper results - reduced for real-time
            whisper_model: Whisper model size (tiny/base/small/medium/large-v2)
            language: Language for transcription (auto for auto-detection)
            enable_vad: Enable voice activity detection to filter silence
            output_format: Output format (text/json_segments/json_words)
            
        Returns:
            Tuple containing transcribed text (empty string if not ready to output)
        """
        try:
            # Direct audio processing - sample_rate is now a separate parameter from ComfyUI
            audio_data = audio
            
            # Validate inputs
            if audio_data is None:
                logger.warning("Received None audio data, returning empty")
                return ("",)
                
            if not hasattr(audio_data, 'shape') and not isinstance(audio_data, (list, tuple, np.ndarray)):
                logger.warning(f"Unexpected audio format: {type(audio_data)}, returning empty")
                return ("",)
            
            # Validate sample rate for Whisper optimization
            if sample_rate != 16000:
                logger.warning(f"Non-optimal sample rate {sample_rate}Hz for Whisper (16kHz recommended)")
            
            # Ensure audio_data is a numpy array before buffering
            if not isinstance(audio_data, np.ndarray):
                logger.debug(f"Converting audio_data from {type(audio_data)} to numpy array.")
                audio_data = np.array(audio_data, dtype=np.float32)

            
            # Process the audio chunk and accumulate for optimal transcription
            ready_for_transcription = self._process_audio_chunk(audio_data, sample_rate, accumulation_duration)
            
            # Transcribe if we have accumulated enough audio
            if ready_for_transcription:
                transcription = self._transcribe_accumulated_audio(whisper_model, language, enable_vad, 16, output_format)
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
    


