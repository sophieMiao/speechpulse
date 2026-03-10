"""Emotion analysis engine using rule-based approach with variation coefficients.

This module implements a rule-based emotion recognition system that uses
variation coefficients (CV = std/mean) and absolute thresholds. This approach
avoids the z-score problem where comparing mean to itself always yields 0.

Emotions supported:
- happy: Elevated pitch with moderate-high energy and variation
- excited: Very high pitch and energy with large variation
- angry: High energy with harsh voice quality and variation
- sad: Low pitch, low energy, monotone
- tired: Low energy with increased pauses
- anxious: Unstable pitch and energy, rushed speech
- neutral: All features near baseline
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .types import AudioFeatures, EmotionResult
from .audio_features import AudioFeatureExtractor


# Emotion rule definitions using variation coefficients and absolute thresholds
# CV = coefficient of variation (std/mean), eliminates gender differences
EMOTION_RULES = {
    "happy": {
        "conditions": [
            ("pitch_cv", ">", 0.15),       # Pitch variation coefficient (lively)
            ("energy_mean", ">", 0.3),      # Medium-high energy
            ("energy_cv", ">", 0.2),        # Energy has variation
        ],
        "weight": 0.8,
        "description": "Elevated pitch with moderate-high energy and variation",
    },
    "excited": {
        "conditions": [
            ("pitch_cv", ">", 0.25),        # Very high pitch variation
            ("energy_mean", ">", 0.4),      # High energy
            ("energy_cv", ">", 0.3),        # Large energy fluctuation
        ],
        "weight": 0.85,
        "description": "Very high pitch and energy with large variation",
    },
    "angry": {
        "conditions": [
            ("energy_mean", ">", 0.5),      # Very high energy
            ("pitch_cv", ">", 0.15),        # Pitch has variation
            ("zcr", ">", 0.1),             # High ZCR (rough voice quality)
        ],
        "weight": 0.9,
        "description": "High energy with harsh voice quality and variation",
    },
    "sad": {
        "conditions": [
            ("energy_mean", "<", 0.25),     # Low energy
            ("pitch_cv", "<", 0.1),         # Flat pitch
            ("energy_cv", "<", 0.15),       # Flat energy
        ],
        "weight": 0.85,
        "description": "Low pitch, low energy, monotone",
    },
    "tired": {
        "conditions": [
            ("energy_mean", "<", 0.2),      # Very low energy
            ("silence_ratio", ">", 0.3),    # Many pauses
            ("pitch_cv", "<", 0.08),        # Extremely flat
        ],
        "weight": 0.8,
        "description": "Low energy with increased pauses",
    },
    "anxious": {
        "conditions": [
            ("pitch_cv", ">", 0.25),        # Unstable pitch
            ("energy_cv", ">", 0.3),        # Unstable energy
            ("silence_ratio", "<", 0.1),    # Almost no pauses (fast speech)
        ],
        "weight": 0.75,
        "description": "Unstable pitch and energy, rushed speech",
    },
    "neutral": {
        "conditions": [
            ("pitch_cv", "<", 0.18),
            ("energy_cv", "<", 0.25),
            ("energy_mean", "in", (0.15, 0.45)),  # Medium energy
        ],
        "weight": 0.5,  # Lowest weight for neutral, let other emotions take priority
        "description": "All features near baseline",
    },
}


@dataclass
class AnalysisFeatures:
    """Analysis features for emotion detection using variation coefficients."""
    pitch_mean: float
    energy_mean: float
    pitch_cv: float        # Coefficient of variation (std/mean)
    energy_cv: float       # Coefficient of variation
    zcr: float            # Zero crossing rate
    silence_ratio: float


def compute_analysis_features(
    features: AudioFeatures,
    frame_pitches: List[float],
    frame_energies: List[float],
) -> AnalysisFeatures:
    """Compute analysis features from raw audio features.
    
    Uses variation coefficients (CV = std/mean) to eliminate gender differences.
    CV is dimensionless and comparable across different speakers.
    
    Args:
        features: Extracted audio features
        frame_pitches: Frame-level pitch values
        frame_energies: Frame-level energy values
        
    Returns:
        AnalysisFeatures with computed values
    """
    # Calculate variation coefficients (CV = std/mean)
    # This normalizes variation by the mean, making it comparable across speakers
    pitch_cv = features.pitch_std / features.pitch_mean if features.pitch_mean > 0 else 0.0
    energy_cv = features.energy_std / features.energy_mean if features.energy_mean > 0 else 0.0
    
    return AnalysisFeatures(
        pitch_mean=features.pitch_mean,
        energy_mean=features.energy_mean,
        pitch_cv=pitch_cv,
        energy_cv=energy_cv,
        zcr=features.zero_crossing_rate,
        silence_ratio=features.silence_ratio,
    )


def check_condition(analysis_features: AnalysisFeatures, condition: Tuple) -> bool:
    """Check if a single condition is met.
    
    Args:
        analysis_features: Analysis features
        condition: Tuple of (feature_name, operator, threshold)
        
    Returns:
        True if condition is satisfied
    """
    feature_name, operator, threshold = condition
    
    # Get feature value
    value = getattr(analysis_features, feature_name, 0.0)
    
    # Check condition
    if operator == ">":
        return value > threshold
    elif operator == "<":
        return value < threshold
    elif operator == ">=":
        return value >= threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == "==":
        return abs(value - threshold) < 0.01
    elif operator == "in":
        # Range check: threshold is a tuple (min, max)
        min_val, max_val = threshold
        return min_val <= value <= max_val
    
    return False


def evaluate_emotion_rule(
    emotion: str,
    analysis_features: AnalysisFeatures,
) -> float:
    """Evaluate how well an emotion rule matches the features.
    
    Args:
        emotion: Emotion label
        analysis_features: Analysis features
        
    Returns:
        Match score (0 to 1)
    """
    if emotion not in EMOTION_RULES:
        return 0.0
    
    rule = EMOTION_RULES[emotion]
    conditions = rule["conditions"]
    weight = rule["weight"]
    
    if not conditions:
        return 0.0
    
    # Count satisfied conditions
    satisfied = sum(1 for cond in conditions if check_condition(analysis_features, cond))
    
    # Calculate score based on proportion of satisfied conditions
    base_score = satisfied / len(conditions)
    
    # Apply emotion-specific weight
    return base_score * weight


class EmotionAnalyzer:
    """Rule-based emotion analyzer using variation coefficients.
    
    This analyzer uses variation coefficients (CV = std/mean) to eliminate
    gender differences while maintaining discriminative power for emotions.
    
    Example:
        analyzer = EmotionAnalyzer()
        result = analyzer.analyze("path/to/audio.wav")
        print(result.primary)  # "happy"
    """
    
    def __init__(self):
        """Initialize the emotion analyzer."""
        self.feature_extractor = AudioFeatureExtractor()
    
    def analyze(self, audio_path: str) -> EmotionResult:
        """Analyze emotion from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            EmotionResult with primary and secondary emotions
        """
        # Extract features and frame-level data in one pass
        features, frame_energies, frame_pitches = self.feature_extractor.extract_all(
            audio_path
        )
        
        return self._analyze_from_features(features, frame_energies, frame_pitches)
    
    def analyze_from_features(
        self,
        features: AudioFeatures,
        frame_energies: List[float],
        frame_pitches: List[float],
    ) -> EmotionResult:
        """Analyze emotion from pre-extracted features.
        
        Args:
            features: Extracted audio features
            frame_energies: Frame-level energy values
            frame_pitches: Frame-level pitch values
            
        Returns:
            EmotionResult with primary and secondary emotions
        """
        return self._analyze_from_features(features, frame_energies, frame_pitches)
    
    def _analyze_from_features(
        self,
        features: AudioFeatures,
        frame_energies: List[float],
        frame_pitches: List[float],
    ) -> EmotionResult:
        """Internal method to analyze emotion from features.
        
        Args:
            features: Extracted audio features
            frame_energies: Frame-level energy values
            frame_pitches: Frame-level pitch values
            
        Returns:
            EmotionResult with primary and secondary emotions
        """
        # Compute analysis features (using CV instead of z-scores)
        analysis_features = compute_analysis_features(features, frame_pitches, frame_energies)
        
        # Evaluate all emotion rules
        scores: Dict[str, float] = {}
        for emotion in EMOTION_RULES:
            scores[emotion] = evaluate_emotion_rule(emotion, analysis_features)
        
        # Sort by score
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get primary emotion
        primary_emotion, primary_score = sorted_emotions[0]
        
        # Get secondary emotion (must have at least 70% of primary score)
        secondary_emotion = None
        if len(sorted_emotions) > 1:
            secondary, secondary_score = sorted_emotions[1]
            if secondary_score > 0 and secondary_score >= primary_score * 0.7:
                secondary_emotion = secondary
        
        # Calculate confidence (normalize to 0-1 range)
        # Use sigmoid-like transformation
        confidence = min(1.0, primary_score * 1.5)
        
        return EmotionResult(
            primary=primary_emotion,
            confidence=confidence,
            secondary=secondary_emotion,
            scores=scores,
        )
    
    def get_emotion_scores(self, audio_path: str) -> Dict[str, float]:
        """Get scores for all emotions.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Dictionary mapping emotion labels to scores
        """
        result = self.analyze(audio_path)
        return result.scores


def analyze_emotion(audio_path: str) -> EmotionResult:
    """Convenience function to analyze emotion from an audio file.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        EmotionResult
    """
    analyzer = EmotionAnalyzer()
    return analyzer.analyze(audio_path)
