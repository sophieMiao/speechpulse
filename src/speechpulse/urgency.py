"""Urgency assessment module for SpeechPulse.

This module implements urgency level assessment based on audio prosodic features.
It analyzes speaking rate, volume, pitch variation, pause patterns, and energy
trends to determine the urgency level of speech.

Urgency levels:
- low (0.0 - 0.3): Normal, relaxed speech
- medium (0.3 - 0.6): Slightly elevated urgency
- high (0.6 - 0.8): Urgent speech
- critical (0.8 - 1.0): Highly urgent/emergency speech
"""

import math
from typing import List, Optional
from dataclasses import dataclass

from .types import AudioFeatures, UrgencyResult
from .audio_features import AudioFeatureExtractor


# Urgency scoring factors and their weights
URGENCY_FACTORS = {
    "speaking_rate": {
        "fast": 0.3,      # Speaking rate > 4 chars/sec (Chinese) or > 150 wpm (English)
        "normal": 0.0,
        "slow": -0.1,
    },
    "volume_level": {
        "high": 0.25,     # energy_mean > 0.6
        "normal": 0.0,
        "low": -0.05,
    },
    "pitch_variation": {
        "high": 0.2,      # pitch_std > 50
        "normal": 0.0,
        "low": -0.05,
    },
    "pause_pattern": {
        "few_pauses": 0.15,   # silence_ratio < 0.15
        "normal": 0.0,
        "many_pauses": -0.1,
    },
    "energy_trend": {
        "increasing": 0.1,
        "stable": 0.0,
        "decreasing": -0.05,
    },
}


@dataclass
class UrgencyFeatures:
    """Features used for urgency assessment."""
    speaking_rate_factor: float  # Based on audio duration vs expected
    volume_level: str  # "high", "normal", "low"
    pitch_variation: str  # "high", "normal", "low"
    pause_pattern: str  # "few_pauses", "normal", "many_pauses"
    energy_trend: str  # "increasing", "stable", "decreasing"


def sigmoid(x: float) -> float:
    """Apply sigmoid function to normalize score to 0-1 range.
    
    Args:
        x: Input value
        
    Returns:
        Sigmoid output in range (0, 1)
    """
    # Use a scaled sigmoid: output approaches 1 as x increases
    return 1.0 / (1.0 + math.exp(-x * 3))


def score_to_level(score: float) -> str:
    """Convert urgency score to categorical level.
    
    Args:
        score: Urgency score (0-1)
        
    Returns:
        Level string: "low", "medium", "high", or "critical"
    """
    if score < 0.3:
        return "low"
    elif score < 0.6:
        return "medium"
    elif score < 0.8:
        return "high"
    else:
        return "critical"


def get_recommended_action(level: str) -> str:
    """Get recommended action based on urgency level.
    
    Args:
        level: Urgency level string
        
    Returns:
        Recommended action string
    """
    actions = {
        "low": "正常处理",
        "medium": "关注并适时跟进",
        "high": "优先处理",
        "critical": "立即处理",
    }
    return actions.get(level, "正常处理")


def assess_speaking_rate(
    features: AudioFeatures,
    text: Optional[str] = None,
) -> float:
    """Assess speaking rate factor.
    
    If text is provided, calculates actual speaking rate.
    Otherwise uses heuristics based on audio features.
    
    Args:
        features: Audio features
        text: Optional transcription text
        
    Returns:
        Speaking rate factor (-1 to 1, higher is faster)
    """
    if text and len(text) > 0:
        # Calculate actual speaking rate
        chars_per_sec = len(text) / features.duration_sec if features.duration_sec > 0 else 0
        # Normal speaking rate is about 3-4 chars/sec for Chinese
        if chars_per_sec > 5:
            return 1.0  # Very fast
        elif chars_per_sec > 4:
            return 0.5  # Fast
        elif chars_per_sec < 2:
            return -0.5  # Slow
        else:
            return 0.0  # Normal
    else:
        # Use energy variation as proxy for speaking rate
        # Higher variation often indicates faster, more animated speech
        if features.energy_std > 0.15:
            return 0.3
        elif features.energy_std < 0.05:
            return -0.2
        return 0.0


def assess_volume_level(features: AudioFeatures) -> str:
    """Assess volume level from energy.
    
    Args:
        features: Audio features
        
    Returns:
        Volume level: "high", "normal", or "low"
    """
    if features.energy_mean > 0.6:
        return "high"
    elif features.energy_mean < 0.2:
        return "low"
    return "normal"


def assess_pitch_variation(features: AudioFeatures) -> str:
    """Assess pitch variation level.
    
    Args:
        features: Audio features
        
    Returns:
        Variation level: "high", "normal", or "low"
    """
    if features.pitch_std > 50:
        return "high"
    elif features.pitch_std < 15:
        return "low"
    return "normal"


def assess_pause_pattern(features: AudioFeatures) -> str:
    """Assess pause pattern from silence ratio.
    
    Args:
        features: Audio features
        
    Returns:
        Pause pattern: "few_pauses", "normal", or "many_pauses"
    """
    if features.silence_ratio < 0.15:
        return "few_pauses"
    elif features.silence_ratio > 0.35:
        return "many_pauses"
    return "normal"


def assess_energy_trend(frame_energies: List[float]) -> str:
    """Assess energy trend from frame-level energies.
    
    Args:
        frame_energies: List of frame energy values
        
    Returns:
        Trend: "increasing", "stable", or "decreasing"
    """
    if len(frame_energies) < 4:
        return "stable"
    
    # Split into first and second half
    mid = len(frame_energies) // 2
    first_half = frame_energies[:mid]
    second_half = frame_energies[mid:]
    
    # Calculate means
    first_mean = sum(first_half) / len(first_half) if first_half else 0
    second_mean = sum(second_half) / len(second_half) if second_half else 0
    
    # Determine trend
    diff = second_mean - first_mean
    threshold = first_mean * 0.2 if first_mean > 0 else 0.05
    
    if diff > threshold:
        return "increasing"
    elif diff < -threshold:
        return "decreasing"
    return "stable"


def generate_reasoning(
    urgency_features: UrgencyFeatures,
    features: AudioFeatures,
) -> List[str]:
    """Generate human-readable reasoning for urgency assessment.
    
    Args:
        urgency_features: Urgency-specific features
        features: Audio features
        
    Returns:
        List of reasoning strings
    """
    reasoning = []
    
    # Speaking rate
    if urgency_features.speaking_rate_factor > 0.5:
        reasoning.append("语速明显加快")
    elif urgency_features.speaking_rate_factor > 0:
        reasoning.append("语速偏快")
    elif urgency_features.speaking_rate_factor < -0.2:
        reasoning.append("语速较慢")
    
    # Volume
    if urgency_features.volume_level == "high":
        reasoning.append("音量持续偏高")
    elif urgency_features.volume_level == "low":
        reasoning.append("音量偏低")
    
    # Pitch variation
    if urgency_features.pitch_variation == "high":
        reasoning.append("音调变化剧烈")
    elif urgency_features.pitch_variation == "low":
        reasoning.append("音调较为平稳")
    
    # Pause pattern
    if urgency_features.pause_pattern == "few_pauses":
        reasoning.append("停顿时间极短，连续说话")
    elif urgency_features.pause_pattern == "many_pauses":
        reasoning.append("停顿较多")
    
    # Energy trend
    if urgency_features.energy_trend == "increasing":
        reasoning.append("音量逐渐增大")
    elif urgency_features.energy_trend == "decreasing":
        reasoning.append("音量逐渐减小")
    
    # Add specific metrics if no reasoning yet
    if not reasoning:
        if features.energy_mean > 0.5:
            reasoning.append("整体音量较高")
        if features.pitch_std > 30:
            reasoning.append("语调有明显起伏")
    
    return reasoning if reasoning else ["语音特征正常"]


class UrgencyAssessor:
    """Urgency level assessor based on audio prosodic features.
    
    This assessor analyzes various acoustic features to determine the
    urgency level of speech, useful for prioritizing responses in
    customer service or emergency scenarios.
    
    Example:
        assessor = UrgencyAssessor()
        result = assessor.assess("path/to/audio.wav")
        print(result.level)  # "high"
        print(result.recommended_action)  # "优先处理"
    """
    
    def __init__(self):
        """Initialize the urgency assessor."""
        self.feature_extractor = AudioFeatureExtractor()
    
    def assess(
        self,
        audio_path: str,
        text: Optional[str] = None,
    ) -> UrgencyResult:
        """Assess urgency level from an audio file.
        
        Args:
            audio_path: Path to the audio file
            text: Optional transcription text for more accurate assessment
            
        Returns:
            UrgencyResult with score, level, and reasoning
        """
        # Extract features
        features = self.feature_extractor.extract(audio_path)
        
        # Get frame-level features
        frame_energies, _ = self.feature_extractor.get_frame_level_features(audio_path)
        
        return self._assess_from_features(features, frame_energies, text)
    
    def assess_from_features(
        self,
        features: AudioFeatures,
        frame_energies: List[float],
        text: Optional[str] = None,
    ) -> UrgencyResult:
        """Assess urgency from pre-extracted features.
        
        Args:
            features: Audio features
            frame_energies: Frame-level energy values
            text: Optional transcription text
            
        Returns:
            UrgencyResult
        """
        return self._assess_from_features(features, frame_energies, text)
    
    def _assess_from_features(
        self,
        features: AudioFeatures,
        frame_energies: List[float],
        text: Optional[str] = None,
    ) -> UrgencyResult:
        """Internal method to assess urgency from features.
        
        Args:
            features: Audio features
            frame_energies: Frame-level energy values
            text: Optional transcription text
            
        Returns:
            UrgencyResult
        """
        # Extract urgency-specific features
        speaking_rate_factor = assess_speaking_rate(features, text)
        volume_level = assess_volume_level(features)
        pitch_variation = assess_pitch_variation(features)
        pause_pattern = assess_pause_pattern(features)
        energy_trend = assess_energy_trend(frame_energies)
        
        urgency_features = UrgencyFeatures(
            speaking_rate_factor=speaking_rate_factor,
            volume_level=volume_level,
            pitch_variation=pitch_variation,
            pause_pattern=pause_pattern,
            energy_trend=energy_trend,
        )
        
        # Calculate urgency score
        raw_score = 0.0
        
        # Speaking rate contribution
        if speaking_rate_factor > 0.5:
            raw_score += URGENCY_FACTORS["speaking_rate"]["fast"]
        elif speaking_rate_factor < -0.2:
            raw_score += URGENCY_FACTORS["speaking_rate"]["slow"]
        
        # Volume contribution
        raw_score += URGENCY_FACTORS["volume_level"][volume_level]
        
        # Pitch variation contribution
        raw_score += URGENCY_FACTORS["pitch_variation"][pitch_variation]
        
        # Pause pattern contribution
        raw_score += URGENCY_FACTORS["pause_pattern"][pause_pattern]
        
        # Energy trend contribution
        raw_score += URGENCY_FACTORS["energy_trend"][energy_trend]
        
        # Normalize to 0-1 using sigmoid
        urgency_score = sigmoid(raw_score)
        
        # Determine level
        level = score_to_level(urgency_score)
        
        # Generate reasoning
        reasoning = generate_reasoning(urgency_features, features)
        
        # Get recommended action
        recommended_action = get_recommended_action(level)
        
        return UrgencyResult(
            score=urgency_score,
            level=level,
            reasoning=reasoning,
            recommended_action=recommended_action,
        )


def assess_urgency(
    audio_path: str,
    text: Optional[str] = None,
) -> UrgencyResult:
    """Convenience function to assess urgency from an audio file.
    
    Args:
        audio_path: Path to the audio file
        text: Optional transcription text
        
    Returns:
        UrgencyResult
    """
    assessor = UrgencyAssessor()
    return assessor.assess(audio_path, text)
