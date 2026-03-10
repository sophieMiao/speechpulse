# SpeechPulse Skill Definition

## Overview

**SpeechPulse** is a voice emotion understanding MCP Server that analyzes speech audio to detect emotions, assess urgency, and detect sarcasm.

## Metadata

```yaml
name: speechpulse
version: 0.1.0
description: Voice emotion understanding with prosodic analysis
tier: lite
author: SpeechPulse Team
license: MIT
```

## Capabilities

### Emotion Analysis
- Detects 7 emotions: happy, excited, angry, sad, tired, anxious, neutral
- Uses z-score relative thresholds to avoid gender bias
- Returns primary emotion, confidence score, and secondary emotion

### Urgency Assessment
- 4 urgency levels: low, medium, high, critical
- Based on speaking rate, volume, pitch variation, and pause patterns
- Keyword-based enhancement (when text provided)

### Sarcasm Detection
- Detects sarcasm by comparing text sentiment with audio emotion
- Keyword-based sentiment analysis for Lite tier

## MCP Tools

### analyze_audio

Analyze audio for emotion and basic features.

**Parameters:**
- `audio_path` (string, required): Path to the audio file (WAV format)
- `text` (string, optional): Transcription text for context

**Returns:**
```json
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
  "features": {
    "duration_sec": 5.0,
    "sample_rate": 16000,
    "pitch_mean": 150.0,
    "pitch_std": 20.0,
    "energy_mean": 0.5,
    "energy_std": 0.1,
    "zero_crossing_rate": 0.05,
    "silence_ratio": 0.2
  }
}
```

### assess_urgency

Assess urgency level from audio.

**Parameters:**
- `audio_path` (string, required): Path to the audio file
- `text` (string, optional): Transcription text for keyword detection

**Returns:**
```json
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
```

### detect_sarcasm

Detect sarcasm by comparing text sentiment with audio emotion.

**Parameters:**
- `audio_path` (string, required): Path to the audio file
- `text` (string, optional): Transcription text (recommended for Lite tier)

**Returns:**
```json
{
  "is_sarcastic": true,
  "confidence": 0.82,
  "indicators": ["Positive text with negative audio tone"],
  "text_emotion": "positive",
  "audio_emotion": "sad"
}
```

### full_analysis

Perform complete analysis including emotion, urgency, and sarcasm.

**Parameters:**
- `audio_path` (string, required): Path to the audio file
- `text` (string, optional): Transcription text (recommended for complete analysis)

**Returns:**
```json
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
```

### health_check

Check server health status.

**Parameters:** None

**Returns:**
```json
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
```

## Usage Examples

### Basic Emotion Analysis

```python
# Using MCP client
result = await client.call_tool("analyze_audio", {
    "audio_path": "/path/to/audio.wav"
})
print(f"Primary emotion: {result['emotion']['primary']}")
```

### Full Analysis with Text

```python
result = await client.call_tool("full_analysis", {
    "audio_path": "/path/to/audio.wav",
    "text": "这真是太棒了"
})
print(result['summary'])
print(result['interpretation'])
```

### Urgency Assessment

```python
result = await client.call_tool("assess_urgency", {
    "audio_path": "/path/to/audio.wav",
    "text": "紧急情况，请立即处理！"
})
if result['level'] in ['high', 'critical']:
    print("Urgent response required!")
```

## Configuration

### Environment Variables

- `SPEECHPULSE_TIER`: Service tier (lite, standard, pro) - default: lite
- `SPEECHPULSE_SAMPLE_RATE`: Target sample rate - default: 16000
- `SPEECHPULSE_FRAME_SIZE`: Analysis frame size - default: 512
- `SPEECHPULSE_HOP_SIZE`: Frame hop size - default: 256

### MCP Server Configuration

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

## Technical Details

### Audio Processing
- Pure Python standard library (no numpy/scipy/librosa)
- Supports WAV files with 8/16/24/32-bit PCM
- Automatic resampling to 16kHz
- Frame-based analysis with Hamming window

### Feature Extraction
- Pitch: Autocorrelation-based fundamental frequency detection
- Energy: RMS energy per frame
- Zero Crossing Rate: Voice/unvoiced detection
- Silence Ratio: Pause pattern analysis

### Emotion Rules (Z-Score Based)
- Avoids absolute thresholds to prevent gender bias
- Relative to audio's own baseline
- 7 emotion categories with weighted rule matching

## Limitations (Lite Tier)

1. **No ASR**: Automatic Speech Recognition not included
   - Use `text` parameter to provide transcriptions
   - ASR available in Standard/Pro tiers

2. **Rule-Based Only**: ML-based emotion recognition not included
   - Uses z-score relative thresholds
   - ML models available in Pro tier

3. **WAV Only**: Only WAV format supported
   - Other formats in Standard/Pro tiers

4. **Single Language**: Optimized for Chinese and English
   - Full multilingual support in Pro tier

## Roadmap

### Standard Tier (Planned)
- ASR with faster-whisper
- Additional audio format support (MP3, FLAC, etc.)
- Speaker diarization

### Pro Tier (Planned)
- Qwen2-Audio integration for end-to-end understanding
- Context-aware emotion analysis
- Nuanced emotion detection (sarcasm, passive-aggressive, etc.)
- Real-time streaming analysis

## Support

- GitHub Issues: https://github.com/yourusername/speechpulse/issues
- Documentation: https://speechpulse.readthedocs.io
