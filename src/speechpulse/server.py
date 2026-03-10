"""MCP Server implementation for SpeechPulse.

This module implements the Model Context Protocol (MCP) server using FastMCP.
It exposes speech analysis capabilities as MCP tools that can be used by
MCP clients like Claude Desktop.

Tools exposed:
- analyze_audio: Analyze emotion from audio
- assess_urgency: Assess urgency level from audio
- detect_sarcasm: Detect sarcasm from audio and text
- full_analysis: Perform complete analysis (emotion + urgency + sarcasm)
"""

import logging
from typing import Dict, Any, Optional

from mcp.server import FastMCP

from .analyzer import SpeechAnalyzer
from .types import EmotionResult, UrgencyResult, SarcasmResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# Initialize FastMCP server
mcp = FastMCP("speechpulse")

# Initialize analyzer
analyzer = SpeechAnalyzer()


@mcp.tool()
def analyze_audio(audio_path: str, text: Optional[str] = None) -> Dict[str, Any]:
    """Analyze audio for emotion and basic features.
    
    This tool analyzes speech audio to detect the speaker's emotional state
    and extract basic audio features. For Lite tier, ASR is not included,
    so provide the 'text' parameter if you have a transcription.
    
    Args:
        audio_path: Path to the audio file (WAV format supported)
        text: Optional transcription text for context
        
    Returns:
        Dictionary containing:
        - transcription: None for Lite tier (ASR not included)
        - note: Information about Lite tier limitations
        - emotion: Object with primary emotion, confidence, secondary emotion, scores
        - speaker_state: Object with energy_level and stress_indicator
        - features: Raw audio features (duration, pitch, energy, etc.)
        
    Example:
        {
            "transcription": null,
            "note": "Lite tier does not include ASR...",
            "emotion": {
                "primary": "happy",
                "confidence": 0.85,
                "secondary": "excited",
                "scores": {"happy": 0.8, "excited": 0.6, ...}
            },
            "speaker_state": {
                "energy_level": "high",
                "stress_indicator": "low"
            },
            "features": {...}
        }
    """
    logger.info(f"Analyzing audio: {audio_path}")
    
    try:
        result = analyzer.analyze(audio_path, text)
        logger.info(f"Analysis complete. Primary emotion: {result['emotion']['primary']}")
        return result
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise


@mcp.tool()
def assess_urgency(audio_path: str, text: Optional[str] = None) -> Dict[str, Any]:
    """Assess urgency level from audio.
    
    This tool evaluates the urgency level of speech based on prosodic features
    like speaking rate, volume, pitch variation, and pause patterns.
    
    Args:
        audio_path: Path to the audio file (WAV format supported)
        text: Optional transcription text for keyword-based urgency detection
        
    Returns:
        Dictionary containing:
        - score: Urgency score (0.0 to 1.0)
        - level: Urgency level ("low", "medium", "high", "critical")
        - reasoning: List of factors contributing to the urgency assessment
        - factors: Detailed breakdown of contributing factors
        
    Example:
        {
            "score": 0.75,
            "level": "high",
            "reasoning": ["Fast speaking rate detected", "High volume variation"],
            "factors": {
                "speaking_rate": "fast",
                "volume_level": "high",
                "pitch_variation": "high",
                "pause_pattern": "few_pauses"
            }
        }
    """
    logger.info(f"Assessing urgency: {audio_path}")
    
    try:
        result = analyzer.assess_urgency(audio_path, text)
        logger.info(f"Urgency assessment complete. Level: {result.level}")
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error assessing urgency: {e}")
        raise


@mcp.tool()
def detect_sarcasm(audio_path: str, text: Optional[str] = None) -> Dict[str, Any]:
    """Detect sarcasm by comparing text sentiment with audio emotion.
    
    This tool detects sarcasm by analyzing the mismatch between the sentiment
    of the text and the emotional tone of the audio. For Lite tier, the 'text'
    parameter is required for accurate detection.
    
    Args:
        audio_path: Path to the audio file (WAV format supported)
        text: Transcription text (recommended for Lite tier)
        
    Returns:
        Dictionary containing:
        - is_sarcastic: Boolean indicating sarcasm detection
        - confidence: Confidence score (0.0 to 1.0)
        - indicators: List of indicators that suggest sarcasm
        - text_emotion: Detected emotion from text (if available)
        - audio_emotion: Detected emotion from audio
        
    Example:
        {
            "is_sarcastic": true,
            "confidence": 0.82,
            "indicators": ["Positive text with negative audio tone"],
            "text_emotion": "positive",
            "audio_emotion": "sad"
        }
    """
    logger.info(f"Detecting sarcasm: {audio_path}")
    
    try:
        result = analyzer.detect_sarcasm(audio_path, text)
        logger.info(f"Sarcasm detection complete. Is sarcastic: {result.is_sarcastic}")
        return result.to_dict()
    except Exception as e:
        logger.error(f"Error detecting sarcasm: {e}")
        raise


@mcp.tool()
def full_analysis(audio_path: str, text: Optional[str] = None) -> Dict[str, Any]:
    """Perform complete analysis including emotion, urgency, and sarcasm.
    
    This tool performs a comprehensive analysis of speech audio, combining
    emotion recognition, urgency assessment, and sarcasm detection into a
    single coherent result.
    
    Args:
        audio_path: Path to the audio file (WAV format supported)
        text: Optional transcription text (recommended for complete analysis)
        
    Returns:
        Dictionary containing:
        - summary: Human-readable summary of the analysis
        - transcription: None for Lite tier (ASR not included)
        - note: Information about Lite tier limitations
        - emotion_analysis: Complete emotion analysis results
        - urgency_assessment: Complete urgency assessment results
        - sarcasm_detection: Complete sarcasm detection results
        - raw_features: Raw audio features extracted
        - interpretation: Contextual interpretation (if text provided)
        
    Example:
        {
            "summary": "说话者表现出开心的情绪。带有明显的紧迫感（high级别）。",
            "transcription": null,
            "note": "Lite tier does not include ASR...",
            "emotion_analysis": {...},
            "urgency_assessment": {...},
            "sarcasm_detection": {...},
            "raw_features": {...},
            "interpretation": "用户语气急促且带有焦虑情绪；建议尽快联系处理。"
        }
    """
    logger.info(f"Performing full analysis: {audio_path}")
    
    try:
        result = analyzer.full_analysis(audio_path, text)
        logger.info(f"Full analysis complete.")
        return result
    except Exception as e:
        logger.error(f"Error in full analysis: {e}")
        raise


@mcp.tool()
def health_check() -> Dict[str, Any]:
    """Check server health status.
    
    This tool can be used to verify that the SpeechPulse MCP server
    is running and functioning correctly.
    
    Returns:
        Dictionary containing:
        - status: "healthy" or "unhealthy"
        - version: Server version
        - tier: Current tier ("lite", "standard", or "pro")
        - capabilities: List of available capabilities
        
    Example:
        {
            "status": "healthy",
            "version": "0.1.0",
            "tier": "lite",
            "capabilities": [
                "emotion_analysis",
                "urgency_assessment",
                "sarcasm_detection"
            ]
        }
    """
    from . import __version__
    
    return {
        "status": "healthy",
        "version": __version__,
        "tier": "lite",
        "capabilities": [
            "emotion_analysis",
            "urgency_assessment",
            "sarcasm_detection",
        ],
    }


def create_server() -> FastMCP:
    """Create and configure the MCP server.
    
    Returns:
        Configured FastMCP instance
    """
    return mcp
