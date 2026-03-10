"""Configuration management for SpeechPulse.

Supports loading configuration from environment variables and JSON config files.
Provides a global configuration singleton for easy access across the package.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1


@dataclass
class ModelConfig:
    """Model configuration for different tiers."""
    whisper_model: str = "base"
    whisper_language: str = "zh"
    emotion2vec_model: str = "iic/emotion2vec_base"
    emotion2vec_granularity: str = "utterance"
    qwen_model: str = "Qwen/Qwen2-Audio-7B-Instruct"
    qwen_quantization: str = "4bit"
    qwen_max_length: int = 512


@dataclass
class AnalysisConfig:
    """Analysis configuration settings."""
    default_language: str = "zh"
    min_audio_duration: float = 0.5
    max_audio_duration: float = 300.0
    max_file_size_mb: float = 50.0


@dataclass
class Config:
    """Global configuration for SpeechPulse.
    
    Attributes:
        tier: Analysis tier (lite, standard, pro)
        frame_size: Analysis frame size in samples (default: 512)
        hop_size: Frame hop size in samples (default: 256)
        server: Server configuration
        models: Model configuration
        analysis: Analysis configuration
        use_gpu: Whether to use GPU for inference
        gpu_device: GPU device index
        dashscope_api_key: API key for DashScope (Pro tier)
    """
    tier: str = "lite"
    frame_size: int = 512
    hop_size: int = 256
    server: ServerConfig = field(default_factory=ServerConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    use_gpu: bool = False
    gpu_device: int = 0
    dashscope_api_key: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        valid_tiers = ["lite", "standard", "pro"]
        if self.tier not in valid_tiers:
            raise ValueError(f"Invalid tier: {self.tier}. Must be one of {valid_tiers}")
        
        if self.server.port < 1 or self.server.port > 65535:
            raise ValueError(f"Invalid port: {self.server.port}")
        
        if self.analysis.min_audio_duration < 0:
            raise ValueError("min_audio_duration must be positive")
        
        if self.analysis.max_audio_duration <= self.analysis.min_audio_duration:
            raise ValueError("max_audio_duration must be greater than min_audio_duration")
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.
        
        Returns:
            Config instance with values from environment variables
        """
        config = cls()
        
        # Load from environment variables
        if "SPEECHPULSE_TIER" in os.environ:
            config.tier = os.environ["SPEECHPULSE_TIER"]
        if "SPEECHPULSE_FRAME_SIZE" in os.environ:
            config.frame_size = int(os.environ["SPEECHPULSE_FRAME_SIZE"])
        if "SPEECHPULSE_HOP_SIZE" in os.environ:
            config.hop_size = int(os.environ["SPEECHPULSE_HOP_SIZE"])
        
        return config


class ConfigManager:
    """Configuration manager supporting multiple sources.
    
    Priority order (highest to lowest):
    1. Environment variables
    2. Config file
    3. Default values
    """
    
    _instance: Optional[Config] = None
    
    @classmethod
    def get_config(cls, config_path: Optional[str] = None) -> Config:
        """Get or create global configuration singleton.
        
        Args:
            config_path: Optional path to JSON config file
            
        Returns:
            Config instance
        """
        if cls._instance is None:
            cls._instance = cls._load_config(config_path)
        return cls._instance
    
    @classmethod
    def reset_config(cls) -> None:
        """Reset the configuration singleton."""
        cls._instance = None
    
    @classmethod
    def _load_config(cls, config_path: Optional[str] = None) -> Config:
        """Load configuration from all sources.
        
        Args:
            config_path: Optional path to JSON config file
            
        Returns:
            Config instance with merged settings
        """
        # Start with defaults
        config = Config()
        
        # Load from config file if provided
        if config_path:
            config = cls._load_from_file(config, config_path)
        
        # Override with environment variables
        config = cls._load_from_env(config)
        
        return config
    
    @classmethod
    def _load_from_file(cls, config: Config, path: str) -> Config:
        """Load configuration from JSON file.
        
        Args:
            config: Existing config to update
            path: Path to JSON config file
            
        Returns:
            Updated Config instance
        """
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Update server config
        if "server" in data:
            server_data = data["server"]
            if "host" in server_data:
                config.server.host = server_data["host"]
            if "port" in server_data:
                config.server.port = server_data["port"]
            if "workers" in server_data:
                config.server.workers = server_data["workers"]
        
        # Update tier
        if "tier" in data:
            config.tier = data["tier"]
        
        # Update models config
        if "models" in data:
            models_data = data["models"]
            if "whisper" in models_data:
                whisper = models_data["whisper"]
                if "model_size" in whisper:
                    config.models.whisper_model = whisper["model_size"]
                if "language" in whisper:
                    config.models.whisper_language = whisper["language"]
            
            if "emotion2vec" in models_data:
                emo_data = models_data["emotion2vec"]
                if "model_name" in emo_data:
                    config.models.emotion2vec_model = emo_data["model_name"]
                if "granularity" in emo_data:
                    config.models.emotion2vec_granularity = emo_data["granularity"]
            
            if "qwen" in models_data:
                qwen_data = models_data["qwen"]
                if "model_path" in qwen_data:
                    config.models.qwen_model = qwen_data["model_path"]
                if "quantization" in qwen_data:
                    config.models.qwen_quantization = qwen_data["quantization"]
                if "max_length" in qwen_data:
                    config.models.qwen_max_length = qwen_data["max_length"]
        
        # Update analysis config
        if "analysis" in data:
            analysis_data = data["analysis"]
            if "default_language" in analysis_data:
                config.analysis.default_language = analysis_data["default_language"]
            if "min_audio_duration" in analysis_data:
                config.analysis.min_audio_duration = analysis_data["min_audio_duration"]
            if "max_audio_duration" in analysis_data:
                config.analysis.max_audio_duration = analysis_data["max_audio_duration"]
        
        return config
    
    @classmethod
    def _load_from_env(cls, config: Config) -> Config:
        """Load configuration from environment variables.
        
        Args:
            config: Existing config to update
            
        Returns:
            Updated Config instance
        """
        # Server config
        if "SPEECHPULSE_HOST" in os.environ:
            config.server.host = os.environ["SPEECHPULSE_HOST"]
        if "SPEECHPULSE_PORT" in os.environ:
            config.server.port = int(os.environ["SPEECHPULSE_PORT"])
        
        # Tier
        if "SPEECHPULSE_TIER" in os.environ:
            config.tier = os.environ["SPEECHPULSE_TIER"]
        
        # GPU config
        if "SPEECHPULSE_USE_GPU" in os.environ:
            config.use_gpu = os.environ["SPEECHPULSE_USE_GPU"].lower() in ("true", "1", "yes")
        if "SPEECHPULSE_GPU_DEVICE" in os.environ:
            config.gpu_device = int(os.environ["SPEECHPULSE_GPU_DEVICE"])
        
        # Model paths
        if "SPEECHPULSE_WHISPER_MODEL" in os.environ:
            config.models.whisper_model = os.environ["SPEECHPULSE_WHISPER_MODEL"]
        if "SPEECHPULSE_EMOTION2VEC_MODEL" in os.environ:
            config.models.emotion2vec_model = os.environ["SPEECHPULSE_EMOTION2VEC_MODEL"]
        if "SPEECHPULSE_QWEN_MODEL" in os.environ:
            config.models.qwen_model = os.environ["SPEECHPULSE_QWEN_MODEL"]
        
        # API keys
        if "DASHSCOPE_API_KEY" in os.environ:
            config.dashscope_api_key = os.environ["DASHSCOPE_API_KEY"]
        
        return config
    
    @classmethod
    def save_config(cls, config: Config, path: str) -> None:
        """Save configuration to JSON file.
        
        Args:
            config: Config instance to save
            path: Path to save JSON config file
        """
        data = {
            "tier": config.tier,
            "server": {
                "host": config.server.host,
                "port": config.server.port,
                "workers": config.server.workers,
            },
            "models": {
                "whisper": {
                    "model_size": config.models.whisper_model,
                    "language": config.models.whisper_language,
                },
                "emotion2vec": {
                    "model_name": config.models.emotion2vec_model,
                    "granularity": config.models.emotion2vec_granularity,
                },
                "qwen": {
                    "model_path": config.models.qwen_model,
                    "quantization": config.models.qwen_quantization,
                    "max_length": config.models.qwen_max_length,
                },
            },
            "analysis": {
                "default_language": config.analysis.default_language,
                "min_audio_duration": config.analysis.min_audio_duration,
                "max_audio_duration": config.analysis.max_audio_duration,
            },
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


# Convenience function for getting global config
def get_config(config_path: Optional[str] = None) -> Config:
    """Get global configuration singleton.
    
    Args:
        config_path: Optional path to JSON config file
        
    Returns:
        Config instance
    """
    return ConfigManager.get_config(config_path)
