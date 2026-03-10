"""Utility functions for SpeechPulse.

This module provides audio loading and processing utilities using only Python
standard library (no numpy/scipy/librosa). This is essential for the Lite tier
to maintain zero ML dependencies.
"""

import wave
import struct
import math
from array import array
from pathlib import Path
from typing import Tuple, Union, List, Optional


# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".wave"}


def validate_audio_path(path: str) -> bool:
    """Validate audio file path for security.
    
    Performs checks to prevent directory traversal attacks and ensure
    the file exists and has a supported format.
    
    Args:
        path: Path to audio file
        
    Returns:
        True if path is valid
        
    Raises:
        ValueError: If path contains suspicious patterns
        FileNotFoundError: If file does not exist
    """
    # Check for directory traversal attempts
    normalized = Path(path).resolve()
    
    # Check if file exists
    if not normalized.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    
    if not normalized.is_file():
        raise ValueError(f"Path is not a file: {path}")
    
    # Check extension
    ext = normalized.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {ext}. "
            f"Supported formats: {SUPPORTED_FORMATS}"
        )
    
    return True


def get_audio_format(path: str) -> str:
    """Get audio file format from extension.
    
    Args:
        path: Path to audio file
        
    Returns:
        Format string (e.g., 'wav', 'mp3')
    """
    return Path(path).suffix.lower().lstrip(".")


def load_audio_stdlib(path: str) -> Tuple[array, int]:
    """Load audio file using only Python standard library.
    
    Currently supports WAV files with 16-bit and 24-bit PCM encoding.
    Returns audio samples normalized to float range [-1.0, 1.0].
    
    Args:
        path: Path to WAV audio file
        
    Returns:
        Tuple of (audio_samples, sample_rate)
        - audio_samples: array.array('f') of float values in [-1.0, 1.0]
        - sample_rate: Sample rate in Hz
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If format is unsupported or invalid
    """
    validate_audio_path(path)
    
    with wave.open(path, "rb") as wav_file:
        # Get audio parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Read raw audio data
        raw_data = wav_file.readframes(n_frames)
        
        # Convert based on sample width
        if sample_width == 1:
            # 8-bit unsigned PCM
            samples = array("B", raw_data)
            # Convert to signed and normalize
            float_samples = array("f", [(s - 128) / 128.0 for s in samples])
            
        elif sample_width == 2:
            # 16-bit signed PCM
            n_samples = len(raw_data) // 2
            format_char = "h"  # signed short
            samples = struct.unpack(f"<{n_samples}{format_char}", raw_data)
            float_samples = array("f", [s / 32768.0 for s in samples])
            
        elif sample_width == 3:
            # 24-bit signed PCM (packed into 3 bytes)
            n_samples = len(raw_data) // 3
            samples = []
            for i in range(n_samples):
                # Read 3 bytes and convert to signed 24-bit int
                b = raw_data[i * 3:(i + 1) * 3]
                value = b[0] | (b[1] << 8) | (b[2] << 16)
                # Sign extend if negative
                if value & 0x800000:
                    value -= 0x1000000
                samples.append(value)
            float_samples = array("f", [s / 8388608.0 for s in samples])
            
        elif sample_width == 4:
            # 32-bit signed PCM
            n_samples = len(raw_data) // 4
            samples = struct.unpack(f"<{n_samples}i", raw_data)
            float_samples = array("f", [s / 2147483648.0 for s in samples])
            
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes")
        
        # Convert to mono if stereo
        if n_channels == 2:
            float_samples = _stereo_to_mono(float_samples)
        elif n_channels > 2:
            raise ValueError(f"Unsupported number of channels: {n_channels}")
    
    return float_samples, sample_rate


def _stereo_to_mono(stereo_samples: array) -> array:
    """Convert stereo audio to mono by averaging channels.
    
    Args:
        stereo_samples: Array of interleaved stereo samples
        
    Returns:
        Array of mono samples
    """
    mono_samples = array("f")
    for i in range(0, len(stereo_samples), 2):
        left = stereo_samples[i]
        right = stereo_samples[i + 1]
        mono_samples.append((left + right) / 2.0)
    return mono_samples


def resample_audio(
    audio: Union[array, List[float]],
    orig_sr: int,
    target_sr: int
) -> array:
    """Resample audio to target sample rate using linear interpolation.
    
    This is a simple resampling implementation using linear interpolation.
    For production use with higher quality requirements, consider using
    proper resampling algorithms (e.g., sinc interpolation).
    
    Args:
        audio: Audio samples as array.array or List[float]
        orig_sr: Original sample rate in Hz
        target_sr: Target sample rate in Hz
        
    Returns:
        Resampled audio as array.array('f')
    """
    if orig_sr == target_sr:
        return audio if isinstance(audio, array) else array("f", audio)
    
    # Calculate resampling ratio
    ratio = target_sr / orig_sr
    orig_len = len(audio)
    target_len = int(orig_len * ratio)
    
    resampled = array("f")
    
    for i in range(target_len):
        # Find position in original audio
        orig_pos = i / ratio
        
        # Get surrounding samples for linear interpolation
        idx_low = int(math.floor(orig_pos))
        idx_high = min(idx_low + 1, orig_len - 1)
        frac = orig_pos - idx_low
        
        # Linear interpolation
        sample_low = audio[idx_low]
        sample_high = audio[idx_high]
        interpolated = sample_low + frac * (sample_high - sample_low)
        resampled.append(interpolated)
    
    return resampled


def ensure_16khz(audio: Union[array, List[float]], sr: int) -> Tuple[array, int]:
    """Ensure audio is at 16kHz sample rate.
    
    Args:
        audio: Audio samples
        sr: Current sample rate in Hz
        
    Returns:
        Tuple of (audio at 16kHz, 16000)
    """
    if sr != 16000:
        return resample_audio(audio, sr, 16000), 16000
    return audio if isinstance(audio, array) else array("f", audio), sr


def calculate_rms(samples: Union[array, List[float]]) -> float:
    """Calculate Root Mean Square (RMS) of audio samples.
    
    Args:
        samples: Audio samples
        
    Returns:
        RMS value
    """
    if len(samples) == 0:
        return 0.0
    
    sum_squares = sum(s * s for s in samples)
    return math.sqrt(sum_squares / len(samples))


def calculate_dbfs(rms: float) -> float:
    """Convert RMS to dBFS (decibels relative to full scale).
    
    Args:
        rms: RMS amplitude (0.0 to 1.0)
        
    Returns:
        dBFS value (typically negative, with 0 being full scale)
    """
    if rms <= 0:
        return -float("inf")
    return 20 * math.log10(rms)


def frame_audio(
    audio: Union[array, List[float]],
    frame_size: int,
    hop_size: int
) -> List[array]:
    """Split audio into overlapping frames.
    
    Args:
        audio: Audio samples
        frame_size: Number of samples per frame
        hop_size: Number of samples between frame starts
        
    Returns:
        List of frames as array.array('f')
    """
    frames = []
    audio_len = len(audio)
    
    for start in range(0, audio_len - frame_size + 1, hop_size):
        frame = array("f", audio[start:start + frame_size])
        frames.append(frame)
    
    return frames


def apply_hamming_window(frame: Union[array, List[float]]) -> array:
    """Apply Hamming window to a frame.
    
    Args:
        frame: Audio frame samples
        
    Returns:
        Windowed frame as array.array('f')
    """
    n = len(frame)
    windowed = array("f")
    
    for i, sample in enumerate(frame):
        # Hamming window formula: 0.54 - 0.46 * cos(2 * pi * i / (n - 1))
        window_val = 0.54 - 0.46 * math.cos(2 * math.pi * i / (n - 1))
        windowed.append(sample * window_val)
    
    return windowed


def get_audio_duration(path: str) -> float:
    """Get audio file duration in seconds.
    
    Args:
        path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    validate_audio_path(path)
    
    with wave.open(path, "rb") as wav_file:
        n_frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        return n_frames / sample_rate


def normalize_audio(audio: Union[array, List[float]], target_peak: float = 0.95) -> array:
    """Normalize audio to target peak amplitude.
    
    Args:
        audio: Audio samples
        target_peak: Target peak amplitude (default 0.95 to avoid clipping)
        
    Returns:
        Normalized audio as array.array('f')
    """
    if len(audio) == 0:
        return array("f")
    
    # Find current peak
    current_peak = max(abs(s) for s in audio)
    
    if current_peak == 0:
        return array("f", audio) if not isinstance(audio, array) else audio
    
    # Calculate gain
    gain = target_peak / current_peak
    
    # Apply gain
    normalized = array("f", [s * gain for s in audio])
    return normalized


def trim_silence(
    audio: Union[array, List[float]],
    threshold_db: float = -40.0,
    min_silence_ms: float = 100.0,
    sr: int = 16000
) -> array:
    """Trim silence from beginning and end of audio.
    
    Args:
        audio: Audio samples
        threshold_db: Silence threshold in dBFS
        min_silence_ms: Minimum silence duration to trim in milliseconds
        sr: Sample rate in Hz
        
    Returns:
        Trimmed audio as array.array('f')
    """
    if len(audio) == 0:
        return array("f")
    
    # Convert threshold from dB to linear
    threshold_linear = 10 ** (threshold_db / 20)
    
    # Calculate minimum silence samples
    min_silence_samples = int(min_silence_ms * sr / 1000)
    
    # Find start (first sample above threshold)
    start = 0
    for i, sample in enumerate(audio):
        if abs(sample) > threshold_linear:
            start = max(0, i - min_silence_samples // 2)
            break
    
    # Find end (last sample above threshold)
    end = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) > threshold_linear:
            end = min(len(audio), i + min_silence_samples // 2)
            break
    
    return array("f", audio[start:end])
