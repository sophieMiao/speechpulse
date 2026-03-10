"""Core data types for SpeechPulse.

This module defines all core data classes used across the SpeechPulse package.
Centralizing type definitions avoids circular imports and ensures consistency.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class AudioFeatures:
    """Audio prosodic features extracted from waveform.
    
    All features are computed using pure Python standard library (no numpy).
    This allows the Lite tier to run with zero ML dependencies.
    
    Attributes:
        duration_sec: Audio duration in seconds
        sample_rate: Sample rate in Hz (typically 16000 after resampling)
        pitch_mean: Mean fundamental frequency in Hz
        pitch_std: Standard deviation of pitch (variation measure)
        energy_mean: Mean energy level (normalized 0-1)
        energy_std: Standard deviation of energy
        zero_crossing_rate: Rate of zero crossings (voice quality indicator)
        silence_ratio: Proportion of silent frames
        mfcc: Optional MFCC features (Standard/Pro tier only)
    """
    duration_sec: float
    sample_rate: int
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    zero_crossing_rate: float
    silence_ratio: float
    mfcc: Optional[List[float]] = None
    
    def __repr__(self) -> str:
        return (
            f"AudioFeatures("
            f"duration={self.duration_sec:.2f}s, "
            f"pitch={self.pitch_mean:.1f}Hz, "
            f"energy={self.energy_mean:.2f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "pitch_mean": self.pitch_mean,
            "pitch_std": self.pitch_std,
            "energy_mean": self.energy_mean,
            "energy_std": self.energy_std,
            "zero_crossing_rate": self.zero_crossing_rate,
            "silence_ratio": self.silence_ratio,
            "mfcc": self.mfcc,
        }


@dataclass
class EmotionResult:
    """Emotion analysis result.
    
    Attributes:
        primary: Primary emotion label (happy, sad, angry, anxious, neutral, excited, tired)
        confidence: Confidence score (0-1)
        secondary: Optional secondary emotion label
        scores: Dictionary of all emotion scores for detailed analysis
    """
    primary: str
    confidence: float
    secondary: Optional[str] = None
    scores: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if self.secondary:
            return (
                f"EmotionResult({self.primary}+{self.secondary}, "
                f"confidence={self.confidence:.2f})"
            )
        return f"EmotionResult({self.primary}, confidence={self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "primary": self.primary,
            "confidence": self.confidence,
            "scores": self.scores,
        }
        if self.secondary:
            result["secondary"] = self.secondary
        return result


@dataclass
class UrgencyResult:
    """Urgency assessment result.
    
    Attributes:
        score: Urgency score from 0 to 1
        level: Categorical level (low, medium, high, critical)
        reasoning: List of human-readable reasoning strings
        recommended_action: Suggested action based on urgency level
    """
    score: float
    level: str
    reasoning: List[str] = field(default_factory=list)
    recommended_action: str = ""
    
    def __repr__(self) -> str:
        return f"UrgencyResult({self.level}, score={self.score:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": self.score,
            "level": self.level,
            "reasoning": self.reasoning,
            "recommended_action": self.recommended_action,
        }


@dataclass
class SarcasmResult:
    """Sarcasm detection result.
    
    Attributes:
        is_sarcastic: Boolean indicating if sarcasm is detected
        confidence: Confidence score (0-1)
        indicators: List of detection indicators
        text_emotion: Optional text sentiment polarity
        audio_emotion: Optional audio emotion detected
    """
    is_sarcastic: bool
    confidence: float
    indicators: List[str] = field(default_factory=list)
    text_emotion: Optional[str] = None
    audio_emotion: Optional[str] = None
    
    def __repr__(self) -> str:
        return f"SarcasmResult(is_sarcastic={self.is_sarcastic}, confidence={self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "is_sarcastic": self.is_sarcastic,
            "confidence": self.confidence,
            "indicators": self.indicators,
        }
        if self.text_emotion:
            result["text_emotion"] = self.text_emotion
        if self.audio_emotion:
            result["audio_emotion"] = self.audio_emotion
        return result


@dataclass
class AnalysisResult:
    """Complete analysis result combining all components.
    
    This is the return type for full_analysis() method.
    
    Attributes:
        summary: Human-readable summary of the analysis
        transcription: Optional transcription text (Standard/Pro tier)
        emotion_analysis: Emotion analysis result
        urgency_assessment: Urgency assessment result
        sarcasm_detection: Sarcasm detection result
        raw_features: Raw audio features extracted
    """
    summary: str
    transcription: Optional[str]
    emotion_analysis: EmotionResult
    urgency_assessment: UrgencyResult
    sarcasm_detection: SarcasmResult
    raw_features: AudioFeatures
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": self.summary,
            "transcription": self.transcription,
            "emotion_analysis": self.emotion_analysis.to_dict(),
            "urgency_assessment": self.urgency_assessment.to_dict(),
            "sarcasm_detection": self.sarcasm_detection.to_dict(),
            "raw_features": self.raw_features.to_dict(),
        }
