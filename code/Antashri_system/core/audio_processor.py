# core/audio_processor.py
"""
Audio processing module for real-time speech capture and transcription
Uses OpenAI Whisper for speech-to-text conversion
"""

import os
import wave
import tempfile
import threading
import queue
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
from loguru import logger
import torch

class AudioProcessor:
    """Handles audio capture, processing, and transcription"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 whisper_model: str = "base"):
        """
        Initialize audio processor
        
        Args:
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono, 2 for stereo)
            chunk_size: Size of audio chunks for processing
            whisper_model: Whisper model size (tiny, base, small, medium, large)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        
        # Initialize Whisper
        logger.info(f"Loading Whisper model: {whisper_model}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisper.load_model(whisper_model, device=self.device)
        logger.success(f"Whisper loaded on {self.device}")
        
        # Audio stream management
        self.stream = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # Silence detection parameters
        self.silence_threshold = int(os.getenv('SILENCE_THRESHOLD', 500))
        self.silence_duration = 0
        self.max_silence = 2.0  # seconds
        
        # Debugging
        self.debug = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
        
    def list_audio_devices(self) -> List[dict]:
        """List available audio input devices"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
                if self.debug:
                    logger.debug(f"Audio device {i}: {device['name']}")
        return devices
    
    def get_default_input_device(self) -> Optional[int]:
        """Get the default audio input device ID"""
        try:
            default = sd.query_devices(kind='input')
            return sd.default.device[0]
        except Exception as e:
            logger.error(f"Error getting default input device: {e}")
            return None
    
    def start_stream(self, device_id: Optional[int] = None):
        """Start audio input stream"""
        try:
            if self.stream is not None:
                self.stop_stream()
            
            # Use default device if not specified
            if device_id is None:
                device_id = self.get_default_input_device()
            
            logger.info(f"Starting audio stream on device {device_id}")
            
            # Create input stream
            self.stream = sd.InputStream(
                device=device_id,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            )
            
            self.stream.start()
            self.is_recording = True
            logger.success("Audio stream started")
            
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def stop_stream(self):
        """Stop audio input stream"""
        if self.stream is not None:
            logger.info("Stopping audio stream...")
            self.is_recording = False
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.success("Audio stream stopped")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        if self.is_recording:
            # Put audio data in queue for processing
            self.audio_queue.put(indata.copy())
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get audio chunk from queue"""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def detect_speech(self, audio: np.ndarray) -> bool:
        """
        Detect if audio contains speech using energy-based VAD
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            True if speech detected, False otherwise
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(max(rms, 1e-10))
        
        # Simple energy-based VAD
        is_speech = rms_db > -40  # Threshold in dB
        
        if self.debug:
            logger.debug(f"Audio RMS: {rms_db:.2f} dB, Speech: {is_speech}")
        
        return is_speech
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        Transcribe audio using Whisper
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Transcribed text or None if no speech detected
        """
        try:
            # Check if audio contains speech
            if not self.detect_speech(audio):
                logger.debug("No speech detected in audio")
                return None
            
            # Ensure audio is float32 and normalized
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Normalize audio to [-1, 1] range
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()
            
            # Save audio to temporary file (Whisper requirement)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, self.sample_rate)
            
            # Transcribe with Whisper
            logger.debug("Transcribing audio...")
            result = self.whisper_model.transcribe(
                tmp_path,
                language='en',
                fp16=self.device == 'cuda'
            )
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            text = result['text'].strip()
            
            if text:
                logger.info(f"Transcription: {text}")
                return text
            else:
                logger.debug("Empty transcription")
                return None
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            if self.debug:
                import traceback
                logger.error(traceback.format_exc())
            return None
    
    def process_audio_buffer(self, 
                           audio_buffer: List[np.ndarray], 
                           max_duration: float = 5.0) -> Optional[np.ndarray]:
        """
        Process audio buffer and combine chunks
        
        Args:
            audio_buffer: List of audio chunks
            max_duration: Maximum duration in seconds
            
        Returns:
            Combined audio array or None
        """
        if not audio_buffer:
            return None
        
        # Combine audio chunks
        combined = np.concatenate(audio_buffer, axis=0)
        
        # Limit duration
        max_samples = int(max_duration * self.sample_rate)
        if len(combined) > max_samples:
            combined = combined[-max_samples:]
        
        # Flatten if multi-channel
        if len(combined.shape) > 1:
            combined = combined.mean(axis=1)
        
        return combined
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply basic noise reduction to audio
        
        Args:
            audio: Input audio array
            
        Returns:
            Denoised audio array
        """
        try:
            # Simple spectral subtraction noise reduction
            from scipy import signal
            
            # Estimate noise from first 0.5 seconds
            noise_samples = int(0.5 * self.sample_rate)
            if len(audio) > noise_samples:
                noise_profile = audio[:noise_samples]
                noise_power = np.mean(noise_profile ** 2)
                
                # Apply spectral subtraction
                f, t, Zxx = signal.stft(audio, fs=self.sample_rate)
                magnitude = np.abs(Zxx)
                phase = np.angle(Zxx)
                
                # Subtract noise
                magnitude_clean = magnitude - noise_power
                magnitude_clean = np.maximum(magnitude_clean, 0)
                
                # Reconstruct signal
                Zxx_clean = magnitude_clean * np.exp(1j * phase)
                _, audio_clean = signal.istft(Zxx_clean, fs=self.sample_rate)
                
                return audio_clean.astype(np.float32)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def save_audio(self, audio: np.ndarray, filepath: str):
        """Save audio to file for debugging"""
        try:
            sf.write(filepath, audio, self.sample_rate)
            logger.debug(f"Audio saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.stream is not None:
            self.stop_stream()


class AudioRecorder:
    """Simple audio recorder for testing"""
    
    def __init__(self, processor: AudioProcessor):
        self.processor = processor
        self.recording = []
        self.is_recording = False
        
    def start_recording(self):
        """Start recording audio"""
        self.recording = []
        self.is_recording = True
        self.processor.start_stream()
        
        # Start recording thread
        self.record_thread = threading.Thread(target=self._record_loop)
        self.record_thread.start()
        
        logger.info("Recording started")
    
    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return audio"""
        self.is_recording = False
        self.processor.stop_stream()
        
        if self.record_thread:
            self.record_thread.join()
        
        logger.info("Recording stopped")
        
        if self.recording:
            return self.processor.process_audio_buffer(self.recording)
        return None
    
    def _record_loop(self):
        """Recording loop"""
        while self.is_recording:
            chunk = self.processor.get_audio_chunk(timeout=0.1)
            if chunk is not None:
                self.recording.append(chunk)


# Test function for debugging
def test_audio_processor():
    """Test audio processor functionality"""
    logger.info("Testing audio processor...")
    
    processor = AudioProcessor(whisper_model="tiny")
    
    # List devices
    devices = processor.list_audio_devices()
    logger.info(f"Found {len(devices)} input devices")
    
    # Test recording
    recorder = AudioRecorder(processor)
    
    logger.info("Recording for 5 seconds...")
    recorder.start_recording()
    time.sleep(5)
    audio = recorder.stop_recording()
    
    if audio is not None:
        logger.info(f"Recorded {len(audio)} samples")
        
        # Test transcription
        text = processor.transcribe(audio)
        if text:
            logger.success(f"Transcription: {text}")
        else:
            logger.warning("No transcription obtained")