"""Sarcasm detection module for SpeechPulse.

This module implements sarcasm detection by comparing text sentiment polarity
with audio emotion. The core principle is that sarcasm often involves a mismatch
between what is said (positive text) and how it's said (negative/neutral tone).

Lite Tier Logic:
- Requires user-provided text (no ASR capability)
- Performs simple keyword-based sentiment analysis on text
- Compares with audio emotion detected from speech
- Detects sarcasm when: text positive + audio negative, or flat tone + exaggerated text

Standard/Pro Tier:
- Can automatically transcribe audio for text analysis
- Uses more sophisticated sentiment analysis models
"""

from typing import List, Optional, Dict
from dataclasses import dataclass

from .types import AudioFeatures, SarcasmResult, EmotionResult
from .audio_features import AudioFeatureExtractor
from .emotion import EmotionAnalyzer


# Positive sentiment keywords (simplified for Lite tier)
POSITIVE_KEYWORDS = {
    "好", "棒", "优秀", "完美", "喜欢", "爱", "开心", "高兴", "快乐",
    "幸福", "满意", "赞", "厉害", "出色", "精彩", "美好", "愉快",
    "good", "great", "excellent", "perfect", "love", "like", "happy",
    "wonderful", "amazing", "awesome", "fantastic", "nice", "best",
}

# Negative sentiment keywords (simplified for Lite tier)
NEGATIVE_KEYWORDS = {
    "坏", "差", "糟糕", "讨厌", "恨", "难过", "伤心", "痛苦", "失望",
    "生气", "愤怒", "烦", "累", "糟", "烂", "恶心", "郁闷", "糟糕",
    "bad", "terrible", "awful", "hate", "sad", "angry", "disappointed",
    "annoying", "worst", "horrible", "disgusting", "frustrated",
}

# Exaggeration indicators
EXAGGERATION_KEYWORDS = {
    "太", "超级", "非常", "极其", "绝对", "完全", "真的", "最",
    "too", "super", "extremely", "absolutely", "totally", "really",
    "definitely", "completely", "utterly", "so", "very",
}


@dataclass
class TextSentiment:
    """Text sentiment analysis result."""
    polarity: str  # "positive", "negative", "neutral"
    confidence: float
    has_exaggeration: bool


def analyze_text_sentiment(text: str) -> TextSentiment:
    """Analyze text sentiment using keyword matching (Lite tier).
    
    This is a simple keyword-based approach suitable for the Lite tier.
    Standard/Pro tiers would use more sophisticated models.
    
    Args:
        text: Input text to analyze
        
    Returns:
        TextSentiment with polarity and confidence
    """
    if not text:
        return TextSentiment(polarity="neutral", confidence=0.0, has_exaggeration=False)
    
    text_lower = text.lower()
    words = set(text_lower.split())
    
    # Count positive and negative keywords
    positive_count = len(words & POSITIVE_KEYWORDS)
    negative_count = len(words & NEGATIVE_KEYWORDS)
    exaggeration_count = len(words & EXAGGERATION_KEYWORDS)
    
    # Determine polarity
    if positive_count > negative_count:
        polarity = "positive"
        confidence = min(1.0, (positive_count - negative_count) * 0.3 + 0.3)
    elif negative_count > positive_count:
        polarity = "negative"
        confidence = min(1.0, (negative_count - positive_count) * 0.3 + 0.3)
    else:
        polarity = "neutral"
        confidence = 0.5
    
    has_exaggeration = exaggeration_count > 0
    
    return TextSentiment(
        polarity=polarity,
        confidence=confidence,
        has_exaggeration=has_exaggeration,
    )


def is_negative_emotion(emotion: str) -> bool:
    """Check if emotion is negative.
    
    Args:
        emotion: Emotion label
        
    Returns:
        True if emotion is negative
    """
    negative_emotions = {"sad", "angry", "tired", "anxious"}
    return emotion in negative_emotions


def is_positive_emotion(emotion: str) -> bool:
    """Check if emotion is positive.
    
    Args:
        emotion: Emotion label
        
    Returns:
        True if emotion is positive
    """
    positive_emotions = {"happy", "excited"}
    return emotion in positive_emotions


def is_neutral_emotion(emotion: str) -> bool:
    """Check if emotion is neutral.
    
    Args:
        emotion: Emotion label
        
    Returns:
        True if emotion is neutral
    """
    return emotion == "neutral"


def is_flat_tone(features: AudioFeatures) -> bool:
    """Check if audio has a flat/monotone tone.
    
    Args:
        features: Audio features
        
    Returns:
        True if tone is flat
    """
    # Low pitch variation indicates flat tone
    return features.pitch_std < 20


def calculate_sarcasm_confidence(
    text_sentiment: TextSentiment,
    audio_emotion: EmotionResult,
    features: AudioFeatures,
) -> float:
    """Calculate sarcasm confidence score.
    
    Args:
        text_sentiment: Analyzed text sentiment
        audio_emotion: Detected audio emotion
        features: Audio features
        
    Returns:
        Confidence score (0-1)
    """
    indicators = []
    
    # Indicator 1: Positive text + negative audio emotion
    if text_sentiment.polarity == "positive" and is_negative_emotion(audio_emotion.primary):
        indicators.append(("positive_text_negative_audio", 0.8))
    
    # Indicator 2: Negative text + positive audio emotion
    elif text_sentiment.polarity == "negative" and is_positive_emotion(audio_emotion.primary):
        indicators.append(("negative_text_positive_audio", 0.6))
    
    # Indicator 3: Exaggerated text + flat tone
    if text_sentiment.has_exaggeration and is_flat_tone(features):
        indicators.append(("exaggerated_text_flat_tone", 0.7))
    
    # Indicator 4: Neutral audio emotion with emotional text
    if text_sentiment.polarity in ("positive", "negative") and is_neutral_emotion(audio_emotion.primary):
        indicators.append(("emotional_text_neutral_audio", 0.5))
    
    # Calculate overall confidence
    if not indicators:
        return 0.0
    
    # Combine indicators (simple average for now)
    total_confidence = sum(conf for _, conf in indicators)
    return min(1.0, total_confidence * 0.8)


def generate_sarcasm_indicators(
    text_sentiment: TextSentiment,
    audio_emotion: EmotionResult,
    features: AudioFeatures,
) -> List[str]:
    """Generate human-readable sarcasm indicators.
    
    Args:
        text_sentiment: Analyzed text sentiment
        audio_emotion: Detected audio emotion
        features: Audio features
        
    Returns:
        List of indicator strings
    """
    indicators = []
    
    # Check for polarity mismatch
    if text_sentiment.polarity == "positive" and is_negative_emotion(audio_emotion.primary):
        indicators.append(f"文本情感极性为正面，语音情感为负面({audio_emotion.primary})")
    elif text_sentiment.polarity == "negative" and is_positive_emotion(audio_emotion.primary):
        indicators.append(f"文本情感极性为负面，语音情感为正面({audio_emotion.primary})")
    
    # Check for exaggerated text + flat tone
    if text_sentiment.has_exaggeration and is_flat_tone(features):
        indicators.append("文本使用夸张表达，但语调异常平坦")
    
    # Check for emotional text + neutral audio
    if text_sentiment.polarity in ("positive", "negative") and is_neutral_emotion(audio_emotion.primary):
        indicators.append(f"文本带有情感色彩，但语音语调平淡")
    
    # Add specific feature observations
    if features.pitch_std < 15:
        indicators.append("语调缺乏变化（单调）")
    
    if text_sentiment.has_exaggeration:
        indicators.append("文本包含夸张词汇")
    
    return indicators if indicators else ["未检测到明显的讽刺特征"]


class SarcasmDetector:
    """Sarcasm detector comparing text and audio emotion.
    
    This detector identifies sarcasm by detecting mismatches between
    the sentiment expressed in text and the emotion conveyed through
    speech prosody.
    
    Lite Tier Requirements:
    - Requires user-provided text parameter (no ASR)
    
    Example:
        detector = SarcasmDetector()
        result = detector.detect("path/to/audio.wav", text="这真是太棒了")
        print(result.is_sarcastic)  # True
    """
    
    def __init__(self):
        """Initialize the sarcasm detector."""
        self.feature_extractor = AudioFeatureExtractor()
        self.emotion_analyzer = EmotionAnalyzer()
    
    def detect(
        self,
        audio_path: str,
        text: Optional[str] = None,
    ) -> SarcasmResult:
        """Detect sarcasm from audio and optional text.
        
        Args:
            audio_path: Path to the audio file
            text: Optional transcription text (required for Lite tier)
            
        Returns:
            SarcasmResult with detection result
            
        Raises:
            ValueError: If text is not provided (Lite tier requirement)
        """
        # Lite tier requires text parameter
        if text is None:
            return SarcasmResult(
                is_sarcastic=False,
                confidence=0.0,
                indicators=["Lite tier requires 'text' parameter for sarcasm detection"],
                text_emotion=None,
                audio_emotion=None,
            )
        
        # Extract features
        features = self.feature_extractor.extract(audio_path)
        
        # Get frame-level features for emotion analysis
        frame_energies, frame_pitches = self.feature_extractor.get_frame_level_features(
            audio_path
        )
        
        # Analyze audio emotion
        audio_emotion = self.emotion_analyzer.analyze_from_features(
            features, frame_energies, frame_pitches
        )
        
        return self._detect_from_analysis(text, audio_emotion, features)
    
    def detect_from_analysis(
        self,
        text: str,
        audio_emotion: EmotionResult,
        features: AudioFeatures,
    ) -> SarcasmResult:
        """Detect sarcasm from pre-analyzed components.
        
        Args:
            text: Transcription text
            audio_emotion: Detected audio emotion
            features: Audio features
            
        Returns:
            SarcasmResult
        """
        return self._detect_from_analysis(text, audio_emotion, features)
    
    def _detect_from_analysis(
        self,
        text: str,
        audio_emotion: EmotionResult,
        features: AudioFeatures,
    ) -> SarcasmResult:
        """Internal method to detect sarcasm.
        
        Args:
            text: Transcription text
            audio_emotion: Detected audio emotion
            features: Audio features
            
        Returns:
            SarcasmResult
        """
        # Analyze text sentiment
        text_sentiment = analyze_text_sentiment(text)
        
        # Calculate sarcasm confidence
        confidence = calculate_sarcasm_confidence(text_sentiment, audio_emotion, features)
        
        # Determine if sarcastic (threshold at 0.6)
        is_sarcastic = confidence >= 0.6
        
        # Generate indicators
        indicators = generate_sarcasm_indicators(text_sentiment, audio_emotion, features)
        
        return SarcasmResult(
            is_sarcastic=is_sarcastic,
            confidence=confidence,
            indicators=indicators,
            text_emotion=text_sentiment.polarity,
            audio_emotion=audio_emotion.primary,
        )


def detect_sarcasm(
    audio_path: str,
    text: Optional[str] = None,
) -> SarcasmResult:
    """Convenience function to detect sarcasm from audio file.
    
    Args:
        audio_path: Path to the audio file
        text: Optional transcription text (required for Lite tier)
        
    Returns:
        SarcasmResult
    """
    detector = SarcasmDetector()
    return detector.detect(audio_path, text)
