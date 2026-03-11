# SpeechPulse

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

**Voice Emotion Understanding MCP Server**

SpeechPulse analyzes speech audio to detect emotions, assess urgency, and detect sarcasm using prosodic features (pitch, energy, rhythm). Built with pure Python standard library for zero ML dependencies in the Lite tier.

## Features

- **Emotion Detection**: Recognizes 7 emotions (happy, excited, angry, sad, tired, anxious, neutral) using coefficient of variation (CV) thresholds
- **Urgency Assessment**: 4-level urgency detection (low, medium, high, critical) based on speaking patterns
- **Sarcasm Detection**: Identifies sarcasm by comparing text sentiment with audio emotion
- **Zero ML Dependencies**: Lite tier uses pure Python standard library (no numpy/scipy/librosa)
- **MCP Compatible**: Exposes tools via Model Context Protocol for integration with Claude Desktop and other MCP clients

## Installation

### From PyPI (when published)

```bash
pip install speechpulse
```

### From Source

```bash
git clone https://github.com/sophieMiao/speechpulse.git
cd speechpulse
pip install -e ".[dev]"
```

## Quick Start

### As MCP Server

Add to your MCP client configuration (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "speechpulse": {
      "command": "python",
      "args": ["-m", "speechpulse"],
      "env": {
        "SPEECHPULSE_TIER": "lite"
      }
    }
  }
}
```

### As Python Library

```python
from speechpulse.analyzer import SpeechAnalyzer

# Initialize analyzer
analyzer = SpeechAnalyzer()

# Analyze emotion
result = analyzer.analyze("path/to/audio.wav")
print(f"Primary emotion: {result['emotion']['primary']}")

# Assess urgency
urgency = analyzer.assess_urgency("path/to/audio.wav")
print(f"Urgency level: {urgency.level}")

# Detect sarcasm (requires text in Lite tier)
sarcasm = analyzer.detect_sarcasm(
    "path/to/audio.wav",
    text="这真是太棒了"
)
print(f"Is sarcastic: {sarcasm.is_sarcastic}")

# Full analysis
full = analyzer.full_analysis("path/to/audio.wav", text="我受够了！")
print(full['summary'])
print(full['interpretation'])
```

### CLI Usage

```bash
# Start MCP server with stdio transport (default)
python -m speechpulse

# Start with SSE transport
python -m speechpulse --transport sse --port 8080

# Enable verbose logging
python -m speechpulse -v
```

## MCP Tools

### analyze_audio

Analyze audio for emotion and basic features.

**Parameters:**
- `audio_path` (string, required): Path to WAV audio file
- `text` (string, optional): Transcription text for context

**Returns:** Emotion detection results, speaker state, and raw audio features

### assess_urgency

Assess urgency level from audio prosody.

**Parameters:**
- `audio_path` (string, required): Path to audio file
- `text` (string, optional): Text for keyword-based urgency detection

**Returns:** Urgency score, level, and reasoning

### detect_sarcasm

Detect sarcasm by comparing text sentiment with audio emotion.

**Parameters:**
- `audio_path` (string, required): Path to audio file
- `text` (string, optional): Transcription text (recommended)

**Returns:** Sarcasm detection result with confidence and indicators

### full_analysis

Perform complete analysis (emotion + urgency + sarcasm).

**Parameters:**
- `audio_path` (string, required): Path to audio file
- `text` (string, optional): Transcription text

**Returns:** Complete analysis with summary and interpretation

### health_check

Check server health and capabilities.

**Returns:** Status, version, tier, and available capabilities

## Architecture

```
speechpulse/
├── types.py           # Core data types (AudioFeatures, EmotionResult, etc.)
├── config.py          # Configuration management
├── utils.py           # Audio loading and processing utilities
├── audio_features.py  # Feature extraction (pitch, energy, etc.)
├── emotion.py         # CV-based emotion rule engine
├── urgency.py         # Urgency assessment logic
├── sarcasm.py         # Sarcasm detection
├── analyzer.py        # Main analysis pipeline
├── server.py          # MCP server implementation
├── asr.py             # ASR stub (Standard/Pro tier)
└── ml_emotion.py      # ML emotion stub (Pro tier)
```

## Technical Details

### Audio Processing

- **Pure Python**: Uses only `wave`, `struct`, `math`, and `array` modules
- **Format Support**: WAV files with 8/16/24/32-bit PCM
- **Resampling**: Linear interpolation to 16kHz
- **Framing**: 32ms frames with 50% overlap, Hamming window

### Feature Extraction

- **Pitch**: Autocorrelation-based F0 detection (50-500 Hz range)
- **Energy**: RMS energy per frame
- **Zero Crossing Rate**: Voice/unvoiced discrimination
- **Silence Ratio**: Pause pattern analysis

### Emotion Recognition

Uses coefficient of variation (CV = std/mean) to avoid gender bias while maintaining discriminative power:

```python
# Example: Happy emotion rule (using coefficient of variation)
"happy": {
    "conditions": [
        ("pitch_cv", ">", 0.15),       # High pitch variation (lively)
        ("energy_mean", ">", 0.3),      # Moderate-high energy
        ("energy_cv", ">", 0.2),        # Energy fluctuation
    ],
    "weight": 0.8,
}
```

### Urgency Assessment

Based on 5 factors:
- Speaking rate (fast/medium/slow)
- Volume level (high/medium/low)
- Pitch variation (high/medium/low)
- Pause pattern (few/normal/many pauses)
- Keyword detection (when text provided)

## Tiers

### Lite Tier (Current)

- ✅ Rule-based emotion recognition
- ✅ Prosodic urgency assessment
- ✅ Keyword-based sarcasm detection
- ✅ Pure Python (no ML dependencies)
- ❌ No ASR (provide text manually)
- ❌ WAV format only

### Standard Tier (Planned)

- ASR with faster-whisper
- Additional audio formats (MP3, FLAC, etc.)
- Speaker diarization

### Pro Tier (Planned)

- Qwen2-Audio integration
- Context-aware emotion analysis
- Nuanced emotion detection
- Real-time streaming

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/sophieMiao/speechpulse.git
cd speechpulse

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python tests/test_all.py

# Run integration tests
python tests/test_integration.py
```

### Demo

```bash
# Run demo script
python examples/demo.py
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SPEECHPULSE_TIER` | `lite` | Service tier (lite/standard/pro) |
| `SPEECHPULSE_SAMPLE_RATE` | `16000` | Target sample rate |
| `SPEECHPULSE_FRAME_SIZE` | `512` | Analysis frame size |
| `SPEECHPULSE_HOP_SIZE` | `256` | Frame hop size |

## Limitations

1. **Lite tier requires text for sarcasm detection**: Provide transcription via `text` parameter
2. **WAV format only**: Convert other formats to WAV before analysis
3. **Rule-based emotions**: ML-based nuanced emotion detection in Pro tier
4. **Optimized for Chinese/English**: Full multilingual support in Pro tier

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MCP SDK](https://github.com/modelcontextprotocol/python-sdk)
- Inspired by prosodic analysis research in speech emotion recognition
- CV approach based on gender-fair emotion recognition research

## Support

- GitHub Issues: [https://github.com/sophieMiao/speechpulse/issues](https://github.com/sophieMiao/speechpulse/issues)

---

**Made with ❤️ for voice emotion understanding**
