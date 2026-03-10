"""SpeechPulse - Voice emotion understanding MCP Server.

SpeechPulse analyzes speech audio to detect emotions, assess urgency,
and detect sarcasm using prosodic features (pitch, energy, rhythm).

Example:
    >>> from speechpulse import SpeechAnalyzer
    >>> analyzer = SpeechAnalyzer()
    >>> result = analyzer.analyze("path/to/audio.wav")
    >>> print(result['emotion']['primary'])
    'happy'

Lite Tier Features:
    - Emotion detection (7 categories)
    - Urgency assessment (4 levels)
    - Sarcasm detection
    - Pure Python standard library (no ML dependencies)

For more information, see: https://github.com/yourusername/speechpulse
"""

from .types import AudioFeatures, EmotionResult, UrgencyResult, SarcasmResult
from .analyzer import SpeechAnalyzer
from .emotion import EmotionAnalyzer
from .urgency import UrgencyAssessor
from .sarcasm import SarcasmDetector
from .config import Config
from .audio_features import AudioFeatureExtractor

__version__ = "0.1.0"
__author__ = "SpeechPulse Team"
__license__ = "MIT"

__all__ = [
    # Version
    "__version__",
    # Types
    "AudioFeatures",
    "EmotionResult",
    "UrgencyResult",
    "SarcasmResult",
    # Main analyzer
    "SpeechAnalyzer",
    # Sub-analyzers
    "EmotionAnalyzer",
    "UrgencyAssessor",
    "SarcasmDetector",
    # Utilities
    "Config",
    "AudioFeatureExtractor",
]
