# Changelog

All notable changes to SpeechPulse will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-11

### Added

- Initial release of SpeechPulse
- **Lite Tier Features**:
  - Emotion detection with 7 categories (happy, excited, angry, sad, tired, anxious, neutral)
  - Z-score relative threshold-based emotion rules to avoid gender bias
  - Urgency assessment with 4 levels (low, medium, high, critical)
  - Sarcasm detection by comparing text sentiment with audio emotion
  - Pure Python standard library implementation (no numpy/scipy/librosa)
- **Audio Processing**:
  - WAV file support (8/16/24/32-bit PCM)
  - Automatic resampling to 16kHz
  - Autocorrelation-based pitch detection
  - Frame-based feature extraction with Hamming window
- **MCP Server**:
  - FastMCP-based MCP server implementation
  - 5 MCP tools: analyze_audio, assess_urgency, detect_sarcasm, full_analysis, health_check
  - Support for stdio and SSE transports
- **Testing**:
  - Comprehensive unit tests
  - Integration tests
  - Demo script with synthetic audio generation
- **Documentation**:
  - Complete README with usage examples
  - OpenClaw Skill definition (SKILL.md)
  - Code documentation and docstrings

### Technical Details

- Uses prosodic features: pitch (F0), energy (RMS), zero crossing rate, silence ratio
- Implements z-score normalization relative to audio baseline
- Supports stereo to mono conversion
- Frame size: 512 samples (32ms at 16kHz)
- Hop size: 256 samples (50% overlap)

### Limitations (Lite Tier)

- No ASR (Automatic Speech Recognition) - text must be provided manually
- WAV format only
- Rule-based emotion recognition only (no ML models)
- Optimized for Chinese and English

[0.1.0]: https://github.com/yourusername/speechpulse/releases/tag/v0.1.0
