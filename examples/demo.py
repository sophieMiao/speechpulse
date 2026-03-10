"""Demo script for SpeechPulse.

This script demonstrates the capabilities of SpeechPulse by:
1. Creating synthetic test audio files with different emotional characteristics
2. Analyzing them for emotion, urgency, and sarcasm
3. Displaying the results

Usage:
    python demo.py

Requirements:
    - speechpulse package installed
    - No external audio files needed (creates synthetic data)
"""

import wave
import os
import tempfile
from array import array
from typing import Tuple

from speechpulse.analyzer import SpeechAnalyzer
from speechpulse.types import AudioFeatures, EmotionResult, UrgencyResult, SarcasmResult


def create_emotional_audio(
    path: str,
    emotion: str,
    duration: float = 3.0,
    sample_rate: int = 16000,
) -> str:
    """Create a synthetic audio file with characteristics of a specific emotion.
    
    Args:
        path: Output file path
        emotion: Emotion type ("happy", "sad", "angry", "anxious", "neutral")
        duration: Audio duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Path to created audio file
    """
    n_samples = int(sample_rate * duration)
    samples = array('h')
    
    import math
    import random
    
    if emotion == "happy":
        # High energy sine wave + frequency modulation (simulates pitch variation)
        base_freq = 220
        for i in range(n_samples):
            t = i / sample_rate
            # Frequency modulation for lively variation
            freq = base_freq + 40 * math.sin(2 * math.pi * 4 * t)
            # Amplitude modulation for energy variation
            amp = 0.8 + 0.2 * math.sin(2 * math.pi * 2 * t)
            value = int(28000 * amp * math.sin(2 * math.pi * freq * t))
            samples.append(value)
    
    elif emotion == "sad":
        # Low energy sine wave + almost no modulation (flat)
        base_freq = 150
        for i in range(n_samples):
            t = i / sample_rate
            # Very slight modulation
            freq = base_freq + 5 * math.sin(2 * math.pi * 0.5 * t)
            # Low, stable amplitude
            value = int(12000 * math.sin(2 * math.pi * freq * t))
            samples.append(value)
    
    elif emotion == "angry":
        # High energy + noise (high ZCR) + frequency modulation
        base_freq = 180
        for i in range(n_samples):
            t = i / sample_rate
            # Frequency modulation
            freq = base_freq + 35 * math.sin(2 * math.pi * 5 * t)
            # Base sine wave
            value = 24000 * math.sin(2 * math.pi * freq * t)
            # Add noise for high ZCR (rough voice quality)
            noise = 8000 * (random.random() * 2 - 1)
            # Add harmonics for harshness
            harmonic = 8000 * math.sin(2 * math.pi * freq * 2 * t)
            final_value = int(value + noise + harmonic)
            # Clip to valid range
            final_value = max(-32768, min(32767, final_value))
            samples.append(final_value)
    
    elif emotion == "anxious":
        # Medium energy + large frequency/amplitude modulation + no silence
        base_freq = 200
        for i in range(n_samples):
            t = i / sample_rate
            # Large frequency modulation (unstable pitch)
            freq = base_freq + 60 * math.sin(2 * math.pi * 7 * t)
            # Large amplitude modulation (unstable energy)
            amp = 0.6 + 0.3 * math.sin(2 * math.pi * 6 * t)
            value = int(20000 * amp * math.sin(2 * math.pi * freq * t))
            samples.append(value)
    
    else:  # neutral
        # Medium energy + slight modulation
        base_freq = 170
        for i in range(n_samples):
            t = i / sample_rate
            # Slight frequency modulation
            freq = base_freq + 15 * math.sin(2 * math.pi * 2 * t)
            # Slight amplitude modulation
            amp = 0.7 + 0.1 * math.sin(2 * math.pi * 1.5 * t)
            value = int(20000 * amp * math.sin(2 * math.pi * freq * t))
            samples.append(value)
    
    # Write WAV file
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    
    return path


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_emotion_result(result: dict):
    """Print emotion analysis result."""
    emotion = result.get("emotion", {})
    print(f"  Primary Emotion: {emotion.get('primary', 'N/A')}")
    print(f"  Confidence: {emotion.get('confidence', 0):.2f}")
    if emotion.get('secondary'):
        print(f"  Secondary Emotion: {emotion.get('secondary')}")
    print(f"  All Scores:")
    for emotion_name, score in emotion.get('scores', {}).items():
        print(f"    - {emotion_name}: {score:.2f}")


def print_urgency_result(result: dict):
    """Print urgency assessment result."""
    print(f"  Urgency Level: {result.get('level', 'N/A')}")
    print(f"  Score: {result.get('score', 0):.2f}")
    print(f"  Reasoning:")
    for reason in result.get('reasoning', []):
        print(f"    - {reason}")


def print_sarcasm_result(result: dict):
    """Print sarcasm detection result."""
    print(f"  Is Sarcastic: {result.get('is_sarcastic', False)}")
    print(f"  Confidence: {result.get('confidence', 0):.2f}")
    print(f"  Indicators:")
    for indicator in result.get('indicators', []):
        print(f"    - {indicator}")


def demo_basic_analysis(analyzer: SpeechAnalyzer):
    """Demo basic emotion analysis."""
    print_section("Demo 1: Basic Emotion Analysis")
    
    # Create test audio files with different emotions
    emotions = ["happy", "sad", "angry", "anxious", "neutral"]
    
    for emotion in emotions:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            audio_path = f.name
        
        try:
            create_emotional_audio(audio_path, emotion)
            result = analyzer.analyze(audio_path)
            
            print(f"\n  [Expected: {emotion}]")
            print_emotion_result(result)
            
        finally:
            os.unlink(audio_path)


def demo_urgency_assessment(analyzer: SpeechAnalyzer):
    """Demo urgency assessment."""
    print_section("Demo 2: Urgency Assessment")
    
    # Create anxious audio (high urgency)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        anxious_path = f.name
    
    # Create neutral audio (low urgency)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        neutral_path = f.name
    
    try:
        create_emotional_audio(anxious_path, "anxious")
        create_emotional_audio(neutral_path, "neutral")
        
        print("\n  [Anxious Audio - Expected: High Urgency]")
        result = analyzer.assess_urgency(anxious_path)
        print_urgency_result(result.to_dict())
        
        print("\n  [Neutral Audio - Expected: Low Urgency]")
        result = analyzer.assess_urgency(neutral_path)
        print_urgency_result(result.to_dict())
        
    finally:
        os.unlink(anxious_path)
        os.unlink(neutral_path)


def demo_sarcasm_detection(analyzer: SpeechAnalyzer):
    """Demo sarcasm detection."""
    print_section("Demo 3: Sarcasm Detection")
    
    # Create sad audio with positive text (sarcastic)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    
    try:
        create_emotional_audio(audio_path, "sad")
        
        print("\n  [Sad Audio + Positive Text '这真是太棒了' - Expected: Sarcastic]")
        result = analyzer.detect_sarcasm(audio_path, text="这真是太棒了")
        print_sarcasm_result(result.to_dict())
        
        print("\n  [Sad Audio + Negative Text '这太糟糕了' - Expected: Not Sarcastic]")
        result = analyzer.detect_sarcasm(audio_path, text="这太糟糕了")
        print_sarcasm_result(result.to_dict())
        
    finally:
        os.unlink(audio_path)


def demo_full_analysis(analyzer: SpeechAnalyzer):
    """Demo full analysis with all features."""
    print_section("Demo 4: Full Analysis")
    
    # Create angry audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_path = f.name
    
    try:
        create_emotional_audio(audio_path, "angry")
        
        print("\n  [Angry Audio + Text '我受够了！' - Full Analysis]")
        result = analyzer.full_analysis(audio_path, text="我受够了！")
        
        print(f"\n  Summary: {result.get('summary', 'N/A')}")
        print(f"\n  Interpretation: {result.get('interpretation', 'N/A')}")
        
        print("\n  Emotion Analysis:")
        print_emotion_result({"emotion": result.get("emotion_analysis", {})})
        
        print("\n  Urgency Assessment:")
        print_urgency_result(result.get("urgency_assessment", {}))
        
        print("\n  Sarcasm Detection:")
        print_sarcasm_result(result.get("sarcasm_detection", {}))
        
    finally:
        os.unlink(audio_path)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("  SpeechPulse Demo")
    print("  Voice Emotion Understanding - Lite Tier")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SpeechAnalyzer()
    
    # Run demos
    demo_basic_analysis(analyzer)
    demo_urgency_assessment(analyzer)
    demo_sarcasm_detection(analyzer)
    demo_full_analysis(analyzer)
    
    print_section("Demo Complete")
    print("\n  Thank you for trying SpeechPulse!")
    print("  For more information, see the README.md file.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
