"""Complete analysis pipeline for SpeechPulse.

This module provides a unified interface for speech analysis, combining
emotion detection, urgency assessment, and sarcasm detection into a single
coherent pipeline.

The SpeechAnalyzer class serves as the main entry point for all analysis
operations, coordinating between different analysis modules and providing
a consistent API.
"""

from typing import Dict, List, Optional, Any

from .types import (
    AudioFeatures,
    EmotionResult,
    UrgencyResult,
    SarcasmResult,
    AnalysisResult,
)
from .audio_features import AudioFeatureExtractor
from .emotion import EmotionAnalyzer
from .urgency import UrgencyAssessor
from .sarcasm import SarcasmDetector


class SpeechAnalyzer:
    """Complete speech analysis pipeline.
    
    This class provides a unified interface for analyzing speech audio,
    combining emotion recognition, urgency assessment, and sarcasm detection.
    
    For Lite tier, the text parameter is optional for emotion and urgency
    analysis, but required for sarcasm detection.
    
    Example:
        analyzer = SpeechAnalyzer()
        
        # Analyze emotion only
        emotion_result = analyzer.analyze("path/to/audio.wav")
        
        # Assess urgency
        urgency_result = analyzer.assess_urgency("path/to/audio.wav")
        
        # Detect sarcasm (requires text in Lite tier)
        sarcasm_result = analyzer.detect_sarcasm(
            "path/to/audio.wav",
            text="这真是太棒了"
        )
        
        # Full analysis
        full_result = analyzer.full_analysis(
            "path/to/audio.wav",
            text="这真是太棒了"
        )
    """
    
    def __init__(self):
        """Initialize the speech analyzer with all sub-analyzers."""
        self.feature_extractor = AudioFeatureExtractor()
        self.emotion_analyzer = EmotionAnalyzer()
        self.urgency_assessor = UrgencyAssessor()
        self.sarcasm_detector = SarcasmDetector()
    
    def analyze(self, audio_path: str, text: Optional[str] = None) -> Dict[str, Any]:
        """Analyze audio for emotion and basic features.
        
        Args:
            audio_path: Path to the audio file
            text: Optional transcription text
            
        Returns:
            Dictionary containing:
            - transcription: Text transcription (None for Lite tier)
            - emotion: EmotionResult
            - speaker_state: Dict with energy_level and stress_indicator
            - features: AudioFeatures
        """
        # Extract features and frame-level data in one pass
        features, frame_energies, frame_pitches = self.feature_extractor.extract_all(
            audio_path
        )
        
        # Analyze emotion
        emotion = self.emotion_analyzer.analyze_from_features(
            features, frame_energies, frame_pitches
        )
        
        # Determine speaker state
        speaker_state = self._determine_speaker_state(features, emotion)
        
        return {
            "transcription": None,  # Lite tier doesn't include ASR
            "note": "Lite tier does not include ASR. Use 'text' param to provide transcription, or upgrade to Standard/Pro tier.",
            "emotion": emotion.to_dict(),
            "speaker_state": speaker_state,
            "features": features.to_dict(),
        }
    
    def assess_urgency(
        self,
        audio_path: str,
        text: Optional[str] = None,
    ) -> UrgencyResult:
        """Assess urgency level from audio.
        
        Args:
            audio_path: Path to the audio file
            text: Optional transcription text for more accurate assessment
            
        Returns:
            UrgencyResult with score, level, and reasoning
        """
        return self.urgency_assessor.assess(audio_path, text)
    
    def detect_sarcasm(
        self,
        audio_path: str,
        text: Optional[str] = None,
    ) -> SarcasmResult:
        """Detect sarcasm by comparing text and audio emotion.
        
        Args:
            audio_path: Path to the audio file
            text: Transcription text (required for Lite tier)
            
        Returns:
            SarcasmResult with detection result
        """
        return self.sarcasm_detector.detect(audio_path, text)
    
    def full_analysis(
        self,
        audio_path: str,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform full analysis including emotion, urgency, and sarcasm.
        
        Args:
            audio_path: Path to the audio file
            text: Optional transcription text (recommended for complete analysis)
            
        Returns:
            Dictionary containing all analysis results
        """
        # Extract features and frame-level data in one pass
        features, frame_energies, frame_pitches = self.feature_extractor.extract_all(
            audio_path
        )
        
        # Perform all analyses
        emotion = self.emotion_analyzer.analyze_from_features(
            features, frame_energies, frame_pitches
        )
        
        urgency = self.urgency_assessor.assess_from_features(
            features, frame_energies, text
        )
        
        # Sarcasm detection (requires text)
        if text:
            sarcasm = self.sarcasm_detector.detect_from_analysis(
                text, emotion, features
            )
        else:
            sarcasm = SarcasmResult(
                is_sarcastic=False,
                confidence=0.0,
                indicators=["Sarcasm detection requires 'text' parameter in Lite tier"],
                text_emotion=None,
                audio_emotion=None,
            )
        
        # Generate summary
        summary = self._generate_summary(emotion, urgency, sarcasm, text)
        
        # Build result
        result = {
            "summary": summary,
            "transcription": None,  # Lite tier doesn't include ASR
            "note": "Lite tier does not include ASR. Provide 'text' param for transcription and sarcasm detection.",
            "emotion_analysis": emotion.to_dict(),
            "urgency_assessment": urgency.to_dict(),
            "sarcasm_detection": sarcasm.to_dict(),
            "raw_features": features.to_dict(),
        }
        
        # Add interpretation if we have text
        if text:
            result["interpretation"] = self._generate_interpretation(
                emotion, urgency, text
            )
        
        return result
    
    def _determine_speaker_state(
        self,
        features: AudioFeatures,
        emotion: EmotionResult,
    ) -> Dict[str, str]:
        """Determine speaker state from features and emotion.
        
        Args:
            features: Audio features
            emotion: Emotion result
            
        Returns:
            Dictionary with energy_level and stress_indicator
        """
        # Determine energy level
        if features.energy_mean > 0.5:
            energy_level = "high"
        elif features.energy_mean < 0.2:
            energy_level = "low"
        else:
            energy_level = "medium"
        
        # Determine stress indicator based on emotion and variation
        stressed_emotions = {"anxious", "angry", "excited"}
        if emotion.primary in stressed_emotions:
            if features.pitch_std > 40:
                stress_indicator = "high"
            else:
                stress_indicator = "medium"
        elif emotion.primary in {"sad", "tired"}:
            stress_indicator = "low"
        else:
            stress_indicator = "low"
        
        return {
            "energy_level": energy_level,
            "stress_indicator": stress_indicator,
        }
    
    def _generate_summary(
        self,
        emotion: EmotionResult,
        urgency: UrgencyResult,
        sarcasm: SarcasmResult,
        text: Optional[str],
    ) -> str:
        """Generate a human-readable summary of the analysis.
        
        Args:
            emotion: Emotion result
            urgency: Urgency result
            sarcasm: Sarcasm result
            text: Optional transcription text
            
        Returns:
            Summary string
        """
        parts = []
        
        # Emotion description
        emotion_desc = {
            "happy": "开心",
            "excited": "兴奋",
            "angry": "愤怒",
            "sad": "悲伤",
            "tired": "疲惫",
            "anxious": "焦虑",
            "neutral": "平静",
        }
        emotion_text = emotion_desc.get(emotion.primary, emotion.primary)
        parts.append(f"说话者表现出{emotion_text}的情绪")
        
        # Urgency description
        if urgency.level in {"high", "critical"}:
            parts.append(f"带有明显的紧迫感（{urgency.level}级别）")
        
        # Sarcasm description
        if sarcasm.is_sarcastic:
            parts.append("语气中可能带有讽刺意味")
        
        # Combine
        summary = "。".join(parts)
        if not summary.endswith("。"):
            summary += "。"
        
        return summary
    
    def _generate_interpretation(
        self,
        emotion: EmotionResult,
        urgency: UrgencyResult,
        text: str,
    ) -> str:
        """Generate interpretation based on emotion, urgency, and text.
        
        Args:
            emotion: Emotion result
            urgency: Urgency result
            text: Transcription text
            
        Returns:
            Interpretation string
        """
        interpretations = []
        
        # Emotion-based interpretation
        if emotion.primary == "anxious":
            interpretations.append("用户语气急促且带有焦虑情绪")
        elif emotion.primary == "angry":
            interpretations.append("用户表现出明显的不满情绪")
        elif emotion.primary == "sad":
            interpretations.append("用户情绪低落")
        
        # Urgency-based interpretation
        if urgency.level == "critical":
            interpretations.append("情况紧急，建议立即处理")
        elif urgency.level == "high":
            interpretations.append("建议尽快联系处理")
        
        # Combine
        if interpretations:
            return "；".join(interpretations) + "。"
        
        return "用户语气正常，建议按常规流程处理。"


def analyze_speech(
    audio_path: str,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for basic speech analysis.
    
    Args:
        audio_path: Path to the audio file
        text: Optional transcription text
        
    Returns:
        Analysis results dictionary
    """
    analyzer = SpeechAnalyzer()
    return analyzer.analyze(audio_path, text)


def full_speech_analysis(
    audio_path: str,
    text: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function for complete speech analysis.
    
    Args:
        audio_path: Path to the audio file
        text: Optional transcription text
        
    Returns:
        Complete analysis results dictionary
    """
    analyzer = SpeechAnalyzer()
    return analyzer.full_analysis(audio_path, text)
