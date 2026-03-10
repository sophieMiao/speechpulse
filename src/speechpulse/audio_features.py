"""Audio feature extraction using pure Python standard library.

This module extracts prosodic features from audio signals without using
numpy, scipy, or librosa. This is essential for the Lite tier to maintain
zero ML dependencies.

Features extracted:
- Duration
- Energy (mean and std)
- Zero crossing rate
- Pitch (fundamental frequency using autocorrelation)
- Silence ratio
"""

import math
import statistics
from array import array
from typing import List, Tuple, Union

from .types import AudioFeatures
from .utils import (
    load_audio_stdlib,
    ensure_16khz,
    frame_audio,
    apply_hamming_window,
    calculate_rms,
)


# Constants for feature extraction
TARGET_SR = 16000  # Target sample rate
FRAME_SIZE = 512  # Frame size in samples (32ms at 16kHz)
HOP_SIZE = 256  # Hop size in samples (50% overlap)
SILENCE_THRESHOLD = 0.01  # Energy threshold for silence detection
MIN_PITCH_HZ = 50  # Minimum pitch in Hz (for voiced speech)
MAX_PITCH_HZ = 500  # Maximum pitch in Hz (for voiced speech)


class AudioFeatureExtractor:
    """Extract prosodic features from audio signals.
    
    This class provides methods to extract various audio features using
    only Python standard library functions.
    
    Example:
        extractor = AudioFeatureExtractor()
        features = extractor.extract("path/to/audio.wav")
    """
    
    def __init__(
        self,
        target_sr: int = TARGET_SR,
        frame_size: int = FRAME_SIZE,
        hop_size: int = HOP_SIZE,
    ):
        """Initialize the feature extractor.
        
        Args:
            target_sr: Target sample rate in Hz
            frame_size: Frame size in samples
            hop_size: Hop size in samples
        """
        self.target_sr = target_sr
        self.frame_size = frame_size
        self.hop_size = hop_size
    
    def extract(self, audio_path: str) -> AudioFeatures:
        """Extract all features from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        # Load audio
        audio, sr = load_audio_stdlib(audio_path)
        
        # Ensure 16kHz sample rate
        audio, sr = ensure_16khz(audio, sr)
        
        # Calculate duration
        duration_sec = len(audio) / sr
        
        # Extract frame-level features
        frame_energies = self._extract_frame_energies(audio)
        frame_pitches = self._extract_frame_pitches(audio, sr)
        
        # Calculate statistics
        energy_mean = statistics.mean(frame_energies) if frame_energies else 0.0
        energy_std = statistics.stdev(frame_energies) if len(frame_energies) > 1 else 0.0
        
        pitch_mean = statistics.mean(frame_pitches) if frame_pitches else 0.0
        pitch_std = statistics.stdev(frame_pitches) if len(frame_pitches) > 1 else 0.0
        
        # Calculate zero crossing rate
        zcr = self._calculate_zcr(audio)
        
        # Calculate silence ratio
        silence_ratio = self._calculate_silence_ratio(frame_energies)
        
        return AudioFeatures(
            duration_sec=duration_sec,
            sample_rate=sr,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zero_crossing_rate=zcr,
            silence_ratio=silence_ratio,
            mfcc=None,  # MFCC not available in Lite tier
        )
    
    def extract_from_array(
        self,
        audio: Union[array, List[float]],
        sr: int,
    ) -> AudioFeatures:
        """Extract features from an audio array.
        
        Args:
            audio: Audio samples
            sr: Sample rate in Hz
            
        Returns:
            AudioFeatures object containing all extracted features
        """
        # Ensure 16kHz sample rate
        audio, sr = ensure_16khz(audio, sr)
        
        # Calculate duration
        duration_sec = len(audio) / sr
        
        # Extract frame-level features
        frame_energies = self._extract_frame_energies(audio)
        frame_pitches = self._extract_frame_pitches(audio, sr)
        
        # Calculate statistics
        energy_mean = statistics.mean(frame_energies) if frame_energies else 0.0
        energy_std = statistics.stdev(frame_energies) if len(frame_energies) > 1 else 0.0
        
        pitch_mean = statistics.mean(frame_pitches) if frame_pitches else 0.0
        pitch_std = statistics.stdev(frame_pitches) if len(frame_pitches) > 1 else 0.0
        
        # Calculate zero crossing rate
        zcr = self._calculate_zcr(audio)
        
        # Calculate silence ratio
        silence_ratio = self._calculate_silence_ratio(frame_energies)
        
        return AudioFeatures(
            duration_sec=duration_sec,
            sample_rate=sr,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zero_crossing_rate=zcr,
            silence_ratio=silence_ratio,
            mfcc=None,
        )
    
    def _extract_frame_energies(self, audio: Union[array, List[float]]) -> List[float]:
        """Extract energy for each frame.
        
        Args:
            audio: Audio samples
            
        Returns:
            List of frame energies
        """
        frames = frame_audio(audio, self.frame_size, self.hop_size)
        energies = []
        
        for frame in frames:
            # Apply Hamming window
            windowed = apply_hamming_window(frame)
            # Calculate RMS energy
            energy = calculate_rms(windowed)
            energies.append(energy)
        
        return energies
    
    def _extract_frame_pitches(
        self,
        audio: Union[array, List[float]],
        sr: int,
    ) -> List[float]:
        """Extract pitch for each frame using autocorrelation.
        
        Args:
            audio: Audio samples
            sr: Sample rate in Hz
            
        Returns:
            List of frame pitches (Hz), 0 for unvoiced frames
        """
        frames = frame_audio(audio, self.frame_size, self.hop_size)
        pitches = []
        
        for frame in frames:
            pitch = self._estimate_pitch_autocorrelation(frame, sr)
            if pitch > 0:
                pitches.append(pitch)
        
        return pitches
    
    def _estimate_pitch_autocorrelation(
        self,
        frame: Union[array, List[float]],
        sr: int,
    ) -> float:
        """Estimate pitch using autocorrelation method.
        
        This implements the autocorrelation method for pitch detection.
        It finds the lag with maximum autocorrelation within the valid
        pitch range.
        
        Args:
            frame: Audio frame samples
            sr: Sample rate in Hz
            
        Returns:
            Estimated pitch in Hz, 0 if unvoiced
        """
        # Apply Hamming window
        windowed = apply_hamming_window(frame)
        
        # Calculate autocorrelation
        n = len(windowed)
        
        # Calculate valid lag range for pitch detection
        min_lag = int(sr / MAX_PITCH_HZ)  # Minimum lag (maximum pitch)
        max_lag = int(sr / MIN_PITCH_HZ)  # Maximum lag (minimum pitch)
        max_lag = min(max_lag, n // 2)  # Don't exceed half the frame
        
        if min_lag >= max_lag:
            return 0.0
        
        # Calculate autocorrelation for each lag
        max_corr = -float("inf")
        best_lag = 0
        
        for lag in range(min_lag, max_lag):
            corr = 0.0
            for i in range(n - lag):
                corr += windowed[i] * windowed[i + lag]
            
            # Normalize by the number of samples
            corr /= (n - lag)
            
            if corr > max_corr:
                max_corr = corr
                best_lag = lag
        
        # Check if the correlation is strong enough (voiced detection)
        # Calculate zero-lag autocorrelation for normalization
        zero_lag_corr = sum(x * x for x in windowed) / n
        
        if zero_lag_corr == 0:
            return 0.0
        
        # Voicing threshold: correlation should be at least 30% of zero-lag
        if max_corr < 0.3 * zero_lag_corr:
            return 0.0
        
        # Convert lag to frequency
        if best_lag > 0:
            pitch = sr / best_lag
            return pitch
        
        return 0.0
    
    def _calculate_zcr(self, audio: Union[array, List[float]]) -> float:
        """Calculate zero crossing rate.
        
        Args:
            audio: Audio samples
            
        Returns:
            Zero crossing rate (0 to 1)
        """
        if len(audio) < 2:
            return 0.0
        
        crossings = 0
        for i in range(1, len(audio)):
            # Check if sign changed
            if (audio[i] >= 0) != (audio[i - 1] >= 0):
                crossings += 1
        
        return crossings / (len(audio) - 1)
    
    def _calculate_silence_ratio(self, frame_energies: List[float]) -> float:
        """Calculate the ratio of silent frames.
        
        Args:
            frame_energies: List of frame energy values
            
        Returns:
            Ratio of silent frames (0 to 1)
        """
        if not frame_energies:
            return 0.0
        
        silent_frames = sum(1 for e in frame_energies if e < SILENCE_THRESHOLD)
        return silent_frames / len(frame_energies)
    
    def get_frame_level_features(
        self,
        audio_path: str,
    ) -> Tuple[List[float], List[float]]:
        """Get frame-level energy and pitch features.
        
        This is useful for emotion analysis that needs access to
        frame-level statistics for z-score calculation.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (frame_energies, frame_pitches)
        """
        # Load audio
        audio, sr = load_audio_stdlib(audio_path)
        audio, sr = ensure_16khz(audio, sr)
        
        # Extract frame-level features
        frame_energies = self._extract_frame_energies(audio)
        frame_pitches = self._extract_frame_pitches(audio, sr)
        
        return frame_energies, frame_pitches
    
    def extract_all(
        self,
        audio_path: str,
    ) -> Tuple[AudioFeatures, List[float], List[float]]:
        """Extract all features and frame-level data in one pass.
        
        This method loads the audio only once and returns both the
        aggregated features and the frame-level data needed for
        emotion analysis. This avoids the performance issue of
        loading the audio twice.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (AudioFeatures, frame_energies, frame_pitches)
        """
        # Load audio once
        audio, sr = load_audio_stdlib(audio_path)
        audio, sr = ensure_16khz(audio, sr)
        
        # Calculate duration
        duration_sec = len(audio) / sr
        
        # Extract frame-level features
        frame_energies = self._extract_frame_energies(audio)
        frame_pitches = self._extract_frame_pitches(audio, sr)
        
        # Calculate statistics
        energy_mean = statistics.mean(frame_energies) if frame_energies else 0.0
        energy_std = statistics.stdev(frame_energies) if len(frame_energies) > 1 else 0.0
        
        pitch_mean = statistics.mean(frame_pitches) if frame_pitches else 0.0
        pitch_std = statistics.stdev(frame_pitches) if len(frame_pitches) > 1 else 0.0
        
        # Calculate zero crossing rate
        zcr = self._calculate_zcr(audio)
        
        # Calculate silence ratio
        silence_ratio = self._calculate_silence_ratio(frame_energies)
        
        features = AudioFeatures(
            duration_sec=duration_sec,
            sample_rate=sr,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            energy_mean=energy_mean,
            energy_std=energy_std,
            zero_crossing_rate=zcr,
            silence_ratio=silence_ratio,
            mfcc=None,  # MFCC not available in Lite tier
        )
        
        return features, frame_energies, frame_pitches


def extract_features(audio_path: str) -> AudioFeatures:
    """Convenience function to extract features from an audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        AudioFeatures object
    """
    extractor = AudioFeatureExtractor()
    return extractor.extract(audio_path)


def extract_features_from_array(
    audio: Union[array, List[float]],
    sr: int,
) -> AudioFeatures:
    """Convenience function to extract features from an audio array.
    
    Args:
        audio: Audio samples
        sr: Sample rate in Hz
        
    Returns:
        AudioFeatures object
    """
    extractor = AudioFeatureExtractor()
    return extractor.extract_from_array(audio, sr)
