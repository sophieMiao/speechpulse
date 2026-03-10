"""Automatic Speech Recognition (ASR) module - Standard/Pro Tier Stub.

This module is a placeholder for ASR functionality in Standard and Pro tiers.
Lite tier does not include ASR capabilities.

Planned implementations:
- Standard Tier: Use faster-whisper or similar lightweight ASR
- Pro Tier: Use Qwen2-Audio for end-to-end audio understanding

For now, this module provides stub functions that raise NotImplementedError
when ASR is requested in non-Lite tiers.
"""

from typing import Optional, Dict, Any


class ASRProvider:
    """Base class for ASR providers.
    
    This is a stub implementation. Actual ASR will be implemented
    in Standard and Pro tiers.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """Initialize ASR provider.
        
        Args:
            model_name: Name of the ASR model to use
        """
        self.model_name = model_name
        self._initialized = False
    
    def transcribe(self, audio_path: str, language: Optional[str] = None) -> str:
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'zh', 'en')
            
        Returns:
            Transcribed text
            
        Raises:
            NotImplementedError: ASR is not available in Lite tier
        """
        raise NotImplementedError(
            "ASR is not available in Lite tier. "
            "Please upgrade to Standard or Pro tier for ASR capabilities."
        )
    
    def transcribe_with_timestamps(
        self,
        audio_path: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe audio with word-level timestamps.
        
        Args:
            audio_path: Path to audio file
            language: Optional language code
            
        Returns:
            Dictionary with transcription and timestamps
            
        Raises:
            NotImplementedError: ASR is not available in Lite tier
        """
        raise NotImplementedError(
            "ASR with timestamps is not available in Lite tier. "
            "Please upgrade to Standard or Pro tier for ASR capabilities."
        )


class WhisperASR(ASRProvider):
    """Whisper-based ASR provider - Standard Tier.
    
    This is a stub. Actual implementation will use faster-whisper.
    """
    
    def __init__(self, model_size: str = "base"):
        """Initialize Whisper ASR.
        
        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        super().__init__(f"whisper-{model_size}")
        self.model_size = model_size


class QwenAudioASR(ASRProvider):
    """Qwen2-Audio based ASR provider - Pro Tier.
    
    This is a stub. Actual implementation will use Qwen2-Audio
    for end-to-end audio understanding.
    """
    
    def __init__(self, model_name: str = "Qwen/Qwen2-Audio-7B"):
        """Initialize Qwen2-Audio ASR.
        
        Args:
            model_name: HuggingFace model name
        """
        super().__init__(model_name)


def transcribe_audio(
    audio_path: str,
    provider: str = "whisper",
    language: Optional[str] = None,
) -> str:
    """Convenience function to transcribe audio.
    
    Args:
        audio_path: Path to audio file
        provider: ASR provider to use ("whisper" or "qwen")
        language: Optional language code
        
    Returns:
        Transcribed text
        
    Raises:
        NotImplementedError: ASR is not available in Lite tier
    """
    raise NotImplementedError(
        "ASR is not available in Lite tier. "
        "Please upgrade to Standard or Pro tier for ASR capabilities."
    )
