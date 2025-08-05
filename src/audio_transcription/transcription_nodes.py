"""
Real-time Audio Transcription Node for ComfyUI.

This node buffers audio segments and performs transcription using faster-whisper,
with controlled output timing to prevent message flooding.
"""

import json
import logging
import time
import tempfile
import os
import numpy as np
from typing import Optional, List, Deque, Dict, Any
from collections import deque
from dataclasses import dataclass
import threading
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


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
    
    def __init__(self):
        # Audio buffering
        self.audio_buffer = np.empty(0, dtype=np.int16)
        self.buffer_duration = 0.0  # Duration in seconds
        self.sample_rate: Optional[int] = None
        self.buffer_samples: Optional[int] = None
        
        # Whisper model (shared across instances)
        self._whisper_model = None
        self._model_lock = threading.Lock()
        self._model_size_name = ""  # Track model size separately
        
        # Transcription queue for controlled output
        self.transcription_queue: Deque[str] = deque(maxlen=50)
        self.last_output_time = 0.0
        
        # Processing state
        self.total_audio_processed = 0.0  # Total audio time processed
        self.transcription_count = 0
        
        # Warmup state - only generate sentinels during initial warmup phase
        self.warmup_phase = True
        self.successful_transcriptions = 0
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("WAVEFORM",),  # Input from LoadAudioTensor
                "transcription_interval": ("FLOAT", {
                    "default": 4.0, 
                    "min": 1.0, 
                    "max": 30.0,
                    "step": 0.5,
                    "tooltip": "Minimum seconds between transcription outputs"
                }),
                "buffer_duration": ("FLOAT", {
                    "default": 4.0,
                    "min": 2.0, 
                    "max": 15.0,
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
                    "default": True,
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
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcription-text",)
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "StreamPack/AudioTranscription"

    def _initialize_whisper_model(self, model_size: str):
        """Initialize the Whisper model if not already loaded."""
        with self._model_lock:
            # Load model if not already loaded or if different size requested
            if (self._whisper_model is None or 
                self._model_size_name != model_size):
                
                logger.info(f"Loading Whisper model: {model_size}")
                self._load_whisper_model_now(model_size)



    def _load_whisper_model_now(self, model_size: str):
        """Load the Whisper model using faster-whisper's automatic download."""
        # Use CPU for compatibility, can be changed to CUDA if available
        device = "cpu"
        try:
            # Try CUDA first if available
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
            else:
                compute_type = "int8"
        except:
            compute_type = "int8"
        
        # Let faster-whisper handle model downloading and caching automatically
        # This ensures we get the correct file structure (vocabulary.txt, etc.)
        logger.info(f"Loading Whisper model via faster-whisper: {model_size}")
        
        self._whisper_model = WhisperModel(
            model_size, 
            device=device, 
            compute_type=compute_type
        )
        self._model_size_name = model_size
        logger.info(f"Whisper model '{model_size}' loaded successfully on {device}")

    def _ensure_model_loaded(self, model_size: str):
        """Ensure the Whisper model is loaded and ready for transcription."""
        with self._model_lock:
            # Check if we need to load or switch models
            if (self._whisper_model is None or 
                self._model_size_name != model_size):
                
                logger.info(f"Loading Whisper model: {model_size}")
                self._load_whisper_model_now(model_size)
    

    
    def _buffer_audio(self, audio_input: np.ndarray, sample_rate: int, buffer_duration: float):
        """
        Efficiently buffer audio data optimized for Whisper transcription.
        
        This is the primary buffering system - no duplicate buffering from LoadAudioTensor.
        """
        # Initialize buffer parameters
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            self.buffer_samples = int(self.sample_rate * buffer_duration)
            logger.info(f"Initialized Whisper-optimized audio buffer: {buffer_duration}s at {sample_rate}Hz = {self.buffer_samples} samples")
        
        # Skip empty frames (from streaming loader)
        if audio_input is None or audio_input.size == 0:
            return False
        
        # Ensure audio is in the right format for Whisper
        audio_input = self._normalize_audio_for_whisper(audio_input)
        
        # Add to buffer - concatenate efficiently
        if self.audio_buffer.size == 0:
            self.audio_buffer = audio_input.copy()
        else:
            self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
        
        # Update buffer duration
        if self.sample_rate is not None:
            self.buffer_duration = len(self.audio_buffer) / self.sample_rate
        
        # Check if we have enough audio for transcription
        if self.buffer_samples is not None:
            ready = len(self.audio_buffer) >= self.buffer_samples
            
            if ready:
                logger.debug(f"Audio buffer ready for transcription: {self.buffer_duration:.2f}s ({len(self.audio_buffer)} samples)")
            
            return ready
        
        return False
    
    def _normalize_audio_for_whisper(self, audio_input: np.ndarray) -> np.ndarray:
        """
        Normalize audio specifically for Whisper requirements.
        
        Whisper expects:
        - 16-bit PCM audio (int16)
        - Mono channel
        - 16 kHz sample rate (handled by LoadAudioTensorStream)
        """
        if audio_input is None or audio_input.size == 0:
            return np.array([], dtype=np.int16)
        
        # Ensure numpy array
        if not isinstance(audio_input, np.ndarray):
            audio_input = np.array(audio_input)
        
        # Handle multi-channel audio (take first channel for transcription)
        if audio_input.ndim > 1:
            if audio_input.shape[1] > audio_input.shape[0]:
                # Shape is (samples, channels) - take first channel
                audio_input = audio_input[:, 0]
            else:
                # Shape is (channels, samples) - take first channel  
                audio_input = audio_input[0, :]
        
        # Convert to int16 format for Whisper
        if audio_input.dtype != np.int16:
            if audio_input.dtype in [np.float32, np.float64]:
                # Convert from float [-1, 1] to int16 [-32768, 32767]
                audio_input = np.clip(audio_input, -1.0, 1.0)
                audio_input = (audio_input * 32767).astype(np.int16)
            else:
                audio_input = audio_input.astype(np.int16)
        
        return audio_input
    
    def _transcribe_audio_buffer(self, model_size: str, language: str, enable_vad: bool, output_format: str = "json_segments") -> Optional[str]:
        """Transcribe the current audio buffer and return combined text."""
        if self.buffer_samples is None or len(self.audio_buffer) < self.buffer_samples:
            return None
        
        try:
            # Ensure model is loaded (handles deferred loading from warmup)
            self._ensure_model_loaded(model_size)
            
            if self._whisper_model is None:
                logger.error("Whisper model not loaded")
                return None
            
            # Extract audio chunk for transcription
            audio_chunk = self.audio_buffer[:self.buffer_samples]
            
            # Save audio to temporary WAV file for whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Write WAV file using scipy.io.wavfile
                try:
                    from scipy.io.wavfile import write
                    if self.sample_rate is not None:
                        write(temp_path, self.sample_rate, audio_chunk)
                    else:
                        logger.error("Sample rate not initialized")
                        return None
                except ImportError:
                    # Fallback to manual WAV writing if scipy not available
                    if self.sample_rate is not None:
                        self._write_wav_file(temp_path, audio_chunk, self.sample_rate)
                    else:
                        logger.error("Sample rate not initialized")
                        return None
                
                # Transcribe using whisper
                language_code = None if language == "auto" else language
                segments, info = self._whisper_model.transcribe(
                    temp_path,
                                    language=language_code,
                word_timestamps=True,  # Enable for SRT generation
                vad_filter=enable_vad,
                    beam_size=1,  # Faster transcription
                    best_of=1     # Faster transcription
                )
                
                # Format output based on requested format
                if output_format == "text":
                    # Simple text output (original behavior)
                    transcribed_texts = []
                    for segment in segments:
                        if segment.text.strip():
                            transcribed_texts.append(segment.text.strip())
                    result = " ".join(transcribed_texts)
                    
                elif output_format == "json_segments":
                    # Segment-level JSON output (like project-transcript)
                    segments_data = []
                    for segment in segments:
                        if segment.text.strip():
                            segments_data.append({
                                "start": segment.start,
                                "end": segment.end,
                                "text": segment.text.strip()
                            })
                    result = json.dumps(segments_data, ensure_ascii=False)
                    
                elif output_format == "json_words":
                    # Word-level JSON output (detailed timing)
                    words_data = []
                    for segment in segments:
                        if hasattr(segment, 'words') and segment.words:
                            for word in segment.words:
                                if word.word.strip():
                                    words_data.append({
                                        "start": word.start,
                                        "end": word.end, 
                                        "word": word.word.strip(),
                                        "probability": getattr(word, 'probability', 1.0)
                                    })
                        elif segment.text.strip():
                            # Fallback if word-level timing not available
                            words_data.append({
                                "start": segment.start,
                                "end": segment.end,
                                "word": segment.text.strip(),
                                "probability": 1.0
                            })
                    result = json.dumps(words_data, ensure_ascii=False)
                else:
                    result = None
                
                # Only generate sentinel values during warmup phase
                if not result or (output_format in ["json_segments", "json_words"] and result == "[]"):
                    if self.warmup_phase:
                        # Return sentinel value to indicate the pipeline is working during warmup
                        if output_format == "text":
                            result = "__WARMUP_SENTINEL__"
                        elif output_format == "json_segments":
                            result = json.dumps([{"start": 0.0, "end": 1.0, "text": "__WARMUP_SENTINEL__"}])
                        elif output_format == "json_words":
                            result = json.dumps([{"start": 0.0, "end": 1.0, "word": "__WARMUP_SENTINEL__", "probability": 1.0}])
                        logger.debug(f"No transcription produced during warmup, returning sentinel value")
                    else:
                        # During normal operation, return None for empty transcriptions
                        result = ""  # Return empty string for ComfyUI compatibility 
                        logger.debug(f"No transcription produced during normal operation, returning empty")
                
                # Track successful transcriptions and exit warmup phase
                if result and not ("__WARMUP_SENTINEL__" in result):
                    self.successful_transcriptions += 1
                    if self.successful_transcriptions >= 2:
                        self.warmup_phase = False
                        logger.debug("Exited warmup phase - future empty transcriptions will return None")
                
                # Update processing metrics
                if self.buffer_samples is not None and self.sample_rate is not None:
                    self.total_audio_processed += self.buffer_samples / self.sample_rate
                self.transcription_count += 1
                
                # Advance buffer (keep some overlap for context)
                if self.buffer_samples is not None:
                    overlap_samples = self.buffer_samples // 4  # 25% overlap
                    advance_samples = self.buffer_samples - overlap_samples
                    self.audio_buffer = self.audio_buffer[advance_samples:]
                
                if result:
                    logger.debug(f"Transcribed chunk {self.transcription_count} ({output_format}): '{str(result)[:50]}...' ({len(str(result))} chars)")
                else:
                    logger.debug(f"Transcribed chunk {self.transcription_count} ({output_format}): No result")
                
                return result
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            # Still advance buffer to prevent getting stuck
            if self.buffer_samples is not None:
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
        
        # During warmup phase, use shorter intervals to speed up warmup detection
        if self.warmup_phase:
            effective_interval = min(transcription_interval, 2.0)  # Max 2 seconds during warmup
            logger.debug(f"Warmup phase - using {effective_interval}s interval instead of {transcription_interval}s")
        else:
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
            
            # Ensure audio_data is a numpy array
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data)
            
            # Buffer the audio (primary buffering system - no duplicate buffering)
            ready_for_transcription = self._buffer_audio(audio_data, sample_rate, buffer_duration)
            
            # Transcribe if buffer has enough audio
            if ready_for_transcription:
                transcription = self._transcribe_audio_buffer(whisper_model, language, enable_vad, output_format)
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
            
        except Exception as e:
            logger.error(f"Error in audio transcription execute: {e}")
            return ("",)
    
    def _queue_transcription(self, transcription: str):
        """Safely queue transcription for controlled output."""
        try:
            self.transcription_queue.append(transcription)
        except:
            # Queue full, remove oldest and add new
            try:
                self.transcription_queue.popleft()
                self.transcription_queue.append(transcription)
                logger.debug("Transcription queue full, replaced oldest entry")
            except:
                logger.warning("Failed to queue transcription")


"""
SRT subtitle generation node for ComfyStream.
Based on project-transcript SRT generation capabilities.
"""

import logging
from datetime import timedelta
from typing import List, Dict, Any, Union
import json

logger = logging.getLogger(__name__)


class SRTGeneratorNode:
    """
    Generate SRT subtitle files from transcription with timing information.
    Compatible with AudioTranscriptionNode output when word timestamps are enabled.
    """
    
    def __init__(self):
        self.subtitle_counter = 1
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcription_data": ("STRING", {
                    "tooltip": "Transcription text or JSON with timing data from AudioTranscriptionNode"
                }),
                "segment_start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 86400.0,  # 24 hours max
                    "step": 0.001,
                    "tooltip": "Start time of this segment in seconds (for absolute timing)"
                }),
                "segment_duration": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.1,
                    "max": 60.0,
                    "step": 0.1,
                    "tooltip": "Duration of this segment in seconds"
                }),
                "use_absolute_time": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use absolute timing (True) or segment-relative timing (False)"
                }),
                "minimum_duration": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Minimum subtitle duration in seconds"
                })
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("srt-content",)
    FUNCTION = "generate_srt"
    OUTPUT_NODE = True
    CATEGORY = "StreamPack/SRTGeneration"    
    
    def generate_srt(self, 
                    transcription_data: str,
                    segment_start_time: float = 0.0,
                    segment_duration: float = 3.0,
                    use_absolute_time: bool = False,
                    minimum_duration: float = 1.0) -> tuple:
        """
        Generate SRT subtitle content from transcription data.
        
        Args:
            transcription_data: Text transcription or JSON with timing data
            segment_start_time: Start time of this segment (for absolute timing)
            segment_duration: Duration of this segment
            use_absolute_time: Whether to use absolute or segment-relative timing
            minimum_duration: Minimum subtitle duration
            
        Returns:
            Tuple containing SRT formatted content
        """
        try:
            if not transcription_data or not transcription_data.strip() or len(transcription_data.strip()) < 5:
                logger.debug(f"Empty or minimal transcription data: '{transcription_data}'")
                return ("",)
            
            # Pass through warmup sentinel values (they'll be filtered later in the pipeline)
            if "__WARMUP_SENTINEL__" in transcription_data:
                logger.debug("Passing through warmup sentinel for pipeline detection")
                return (transcription_data,)
            
            # Try to parse as JSON first (if AudioTranscriptionNode supports structured output)
            segments = self._parse_transcription_data(transcription_data)
            
            if not segments:
                logger.debug("No valid segments found in transcription data")
                return ("",)
            
            # Generate SRT content
            srt_content = self._generate_srt_from_segments(
                segments, 
                segment_start_time,
                segment_duration,
                use_absolute_time,
                minimum_duration
            )
            
            logger.debug(f"Generated SRT with {len(segments)} segments")
            return (srt_content,)
            
        except Exception as e:
            logger.error(f"Error generating SRT: {e}")
            return ("",)
    
    def _parse_transcription_data(self, data: str) -> List[Dict[str, Any]]:
        """
        Parse transcription data into segments.
        Enhanced to handle AudioTranscriptionNode JSON output formats.
        """
        segments = []
        
        try:
            # Try parsing as JSON first
            parsed_data = json.loads(data)
            
            if isinstance(parsed_data, list):
                # Handle both segment-level and word-level JSON arrays
                for item in parsed_data:
                    if isinstance(item, dict):
                        # Check if it's word-level data (has 'word' field) or segment-level (has 'text' field)
                        if 'word' in item:
                            # Word-level JSON from AudioTranscriptionNode (json_words format)
                            segments.append({
                                'start': item.get('start', 0.0),
                                'end': item.get('end', 1.0),
                                'text': item.get('word', '').strip()
                            })
                        elif 'text' in item:
                            # Segment-level JSON from AudioTranscriptionNode (json_segments format)
                            segments.append({
                                'start': item.get('start', 0.0),
                                'end': item.get('end', 1.0),
                                'text': item.get('text', '').strip()
                            })
                            
            elif isinstance(parsed_data, dict):
                # Single segment or word
                if 'word' in parsed_data:
                    segments.append({
                        'start': parsed_data.get('start', 0.0),
                        'end': parsed_data.get('end', 1.0),
                        'text': parsed_data.get('word', '').strip()
                    })
                elif 'text' in parsed_data:
                    segments.append({
                        'start': parsed_data.get('start', 0.0),
                        'end': parsed_data.get('end', 1.0),
                        'text': parsed_data.get('text', '').strip()
                    })
                    
        except (json.JSONDecodeError, TypeError):
            # Fallback to plain text - create a single segment
            if data.strip():
                segments.append({
                    'start': 0.0,
                    'end': 3.0,  # Default 3-second duration
                    'text': data.strip()
                })
        
        return [seg for seg in segments if seg['text']]  # Filter empty text
    
    def _generate_srt_from_segments(self, 
                                   segments: List[Dict[str, Any]],
                                   segment_start_time: float,
                                   segment_duration: float,
                                   use_absolute_time: bool,
                                   minimum_duration: float) -> str:
        """
        Generate SRT content from parsed segments.
        """
        srt_lines = []
        
        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            if not text:
                continue
            
            # Calculate timing
            if use_absolute_time:
                # Absolute timing: add segment start time
                start_time = segment_start_time + segment['start']
                end_time = segment_start_time + segment['end']
            else:
                # Segment-relative timing
                start_time = segment['start']
                end_time = segment['end']
            
            # Ensure minimum duration
            if end_time - start_time < minimum_duration:
                end_time = start_time + minimum_duration
            
            # Format timing
            start_srt = self._seconds_to_srt_time(start_time)
            end_srt = self._seconds_to_srt_time(end_time)
            
            # Add SRT entry
            srt_lines.append(str(self.subtitle_counter))
            srt_lines.append(f"{start_srt} --> {end_srt}")
            srt_lines.append(text)
            srt_lines.append("")  # Blank line
            
            self.subtitle_counter += 1
        
        return "\n".join(srt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """
        Convert seconds to SRT time format (HH:MM:SS,mmm).
        Compatible with project-transcript format.
        """
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        milliseconds = int((seconds - total_seconds) * 1000)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
