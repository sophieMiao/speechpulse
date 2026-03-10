"""Unit tests for SpeechPulse.

This module contains comprehensive unit tests for all SpeechPulse modules.
Tests use synthetic audio data to avoid external dependencies.
"""

import unittest
import wave
import struct
import tempfile
import os
from array import array

from speechpulse.types import AudioFeatures, EmotionResult, UrgencyResult, SarcasmResult
from speechpulse.config import Config
from speechpulse.utils import (
    load_audio_stdlib,
    resample_audio,
    ensure_16khz,
    calculate_rms,
    frame_audio,
    apply_hamming_window,
)
from speechpulse.audio_features import AudioFeatureExtractor
from speechpulse.emotion import EmotionAnalyzer, compute_analysis_features, evaluate_emotion_rule, AnalysisFeatures
from speechpulse.urgency import UrgencyAssessor
from speechpulse.sarcasm import SarcasmDetector
from speechpulse.analyzer import SpeechAnalyzer


class TestTypes(unittest.TestCase):
    """Tests for types module."""
    
    def test_audio_features_to_dict(self):
        """Test AudioFeatures to_dict conversion."""
        features = AudioFeatures(
            duration_sec=5.0,
            sample_rate=16000,
            pitch_mean=150.0,
            pitch_std=20.0,
            energy_mean=0.5,
            energy_std=0.1,
            zero_crossing_rate=0.05,
            silence_ratio=0.2,
        )
        d = features.to_dict()
        self.assertEqual(d["duration_sec"], 5.0)
        self.assertEqual(d["sample_rate"], 16000)
        self.assertEqual(d["pitch_mean"], 150.0)
    
    def test_emotion_result_to_dict(self):
        """Test EmotionResult to_dict conversion."""
        result = EmotionResult(
            primary="happy",
            confidence=0.85,
            secondary="excited",
            scores={"happy": 0.8, "excited": 0.6, "neutral": 0.3},
        )
        d = result.to_dict()
        self.assertEqual(d["primary"], "happy")
        self.assertEqual(d["confidence"], 0.85)
        self.assertEqual(d["secondary"], "excited")


class TestConfig(unittest.TestCase):
    """Tests for config module."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()
        self.assertEqual(config.tier, "lite")
        self.assertEqual(config.frame_size, 512)
        self.assertEqual(config.hop_size, 256)
    
    def test_config_from_env(self):
        """Test creating config from environment variables."""
        import os
        os.environ["SPEECHPULSE_TIER"] = "standard"
        os.environ["SPEECHPULSE_FRAME_SIZE"] = "1024"
        
        config = Config.from_env()
        self.assertEqual(config.tier, "standard")
        self.assertEqual(config.frame_size, 1024)
        
        # Clean up
        del os.environ["SPEECHPULSE_TIER"]
        del os.environ["SPEECHPULSE_FRAME_SIZE"]


class TestUtils(unittest.TestCase):
    """Tests for utils module."""
    
    def setUp(self):
        """Create temporary test audio file."""
        # Windows: use mkstemp and close immediately to avoid file lock
        fd, self.temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._create_test_wav(self.temp_path)
    
    def tearDown(self):
        """Clean up temporary file."""
        try:
            os.unlink(self.temp_path)
        except (PermissionError, OSError):
            pass  # Windows: file may still be locked, ignore
    
    def _create_test_wav(self, path: str, duration: float = 1.0, freq: float = 440.0):
        """Create a test WAV file with a sine wave."""
        sample_rate = 16000
        n_samples = int(sample_rate * duration)
        
        # Generate sine wave
        import math
        samples = array('h')
        for i in range(n_samples):
            t = i / sample_rate
            value = int(10000 * math.sin(2 * math.pi * 440 * t))
            samples.append(value)
        
        # Write WAV file
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
    
    def test_load_audio_stdlib(self):
        """Test loading audio with stdlib."""
        audio, sr = load_audio_stdlib(self.temp_path)
        self.assertEqual(sr, 16000)
        self.assertGreater(len(audio), 0)
        # Check normalization to [-1, 1]
        self.assertTrue(all(-1.0 <= s <= 1.0 for s in audio))
    
    def test_resample_audio(self):
        """Test audio resampling."""
        audio = array('f', [0.0, 0.5, 1.0, 0.5, 0.0])
        resampled = resample_audio(audio, 16000, 8000)
        # Should have approximately half the samples
        self.assertLess(len(resampled), len(audio) * 0.6)
    
    def test_ensure_16khz(self):
        """Test ensuring 16kHz sample rate."""
        audio = array('f', [0.0, 0.5, 1.0, 0.5, 0.0])
        result, sr = ensure_16khz(audio, 16000)
        self.assertEqual(sr, 16000)
        self.assertEqual(len(result), len(audio))
    
    def test_calculate_rms(self):
        """Test RMS calculation."""
        samples = [1.0, -1.0, 1.0, -1.0]
        rms = calculate_rms(samples)
        self.assertAlmostEqual(rms, 1.0, places=5)
    
    def test_frame_audio(self):
        """Test audio framing."""
        audio = array('f', list(range(100)))
        frames = frame_audio(audio, frame_size=32, hop_size=16)
        self.assertGreater(len(frames), 0)
        self.assertEqual(len(frames[0]), 32)
    
    def test_apply_hamming_window(self):
        """Test Hamming window application."""
        frame = array('f', [1.0] * 10)
        windowed = apply_hamming_window(frame)
        self.assertEqual(len(windowed), len(frame))
        # Window should attenuate edges
        self.assertLess(windowed[0], windowed[len(frame)//2])


class TestAudioFeatures(unittest.TestCase):
    """Tests for audio_features module."""
    
    def setUp(self):
        """Create temporary test audio file."""
        # Windows: use mkstemp and close immediately
        fd, self.temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._create_test_wav(self.temp_path)
        self.extractor = AudioFeatureExtractor()
    
    def tearDown(self):
        """Clean up temporary file."""
        try:
            os.unlink(self.temp_path)
        except (PermissionError, OSError):
            pass
    
    def _create_test_wav(self, path: str, duration: float = 2.0):
        """Create a test WAV file."""
        import math
        sample_rate = 16000
        n_samples = int(sample_rate * duration)
        
        samples = array('h')
        for i in range(n_samples):
            # Create varying signal
            value = int(10000 * math.sin(2 * math.pi * 200 * i / sample_rate))
            samples.append(value)
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
    
    def test_extract(self):
        """Test feature extraction."""
        features = self.extractor.extract(self.temp_path)
        self.assertIsInstance(features, AudioFeatures)
        self.assertGreater(features.duration_sec, 0)
        self.assertEqual(features.sample_rate, 16000)
        self.assertIsNone(features.mfcc)  # MFCC not in Lite tier
    
    def test_extract_all(self):
        """Test extract_all method."""
        features, frame_energies, frame_pitches = self.extractor.extract_all(
            self.temp_path
        )
        self.assertIsInstance(features, AudioFeatures)
        self.assertIsInstance(frame_energies, list)
        self.assertIsInstance(frame_pitches, list)


class TestEmotion(unittest.TestCase):
    """Tests for emotion module."""
    
    def test_compute_analysis_features(self):
        """Test analysis feature computation."""
        features = AudioFeatures(
            duration_sec=5.0,
            sample_rate=16000,
            pitch_mean=150.0,
            pitch_std=20.0,  # CV = 20/150 = 0.133
            energy_mean=0.5,
            energy_std=0.1,  # CV = 0.1/0.5 = 0.2
            zero_crossing_rate=0.05,
            silence_ratio=0.2,
        )
        frame_pitches = [140.0, 150.0, 160.0]
        frame_energies = [0.4, 0.5, 0.6]
        
        analysis = compute_analysis_features(features, frame_pitches, frame_energies)
        
        # Check CV calculations
        self.assertAlmostEqual(analysis.pitch_cv, 20.0/150.0, places=3)
        self.assertAlmostEqual(analysis.energy_cv, 0.1/0.5, places=3)
        self.assertEqual(analysis.pitch_mean, 150.0)
        self.assertEqual(analysis.energy_mean, 0.5)
        self.assertEqual(analysis.zcr, 0.05)
        self.assertEqual(analysis.silence_ratio, 0.2)
    
    def test_evaluate_emotion_rule(self):
        """Test emotion rule evaluation."""
        # Happy: pitch_cv > 0.15, energy_mean > 0.3, energy_cv > 0.2
        analysis = AnalysisFeatures(
            pitch_mean=200.0,
            energy_mean=0.5,
            pitch_cv=0.20,  # > 0.15
            energy_cv=0.25,  # > 0.2
            zcr=0.05,
            silence_ratio=0.1,
        )
        
        # Should match happy
        score = evaluate_emotion_rule("happy", analysis)
        self.assertGreater(score, 0)
    
    def test_analyze_from_features(self):
        """Test emotion analysis from features."""
        analyzer = EmotionAnalyzer()
        
        # Create features that should match "excited"
        # excited: pitch_cv > 0.25, energy_mean > 0.4, energy_cv > 0.3
        features = AudioFeatures(
            duration_sec=5.0,
            sample_rate=16000,
            pitch_mean=200.0,
            pitch_std=60.0,    # CV = 0.3 > 0.25
            energy_mean=0.6,   # > 0.4
            energy_std=0.25,   # CV = 0.417 > 0.3
            zero_crossing_rate=0.05,
            silence_ratio=0.05,
        )
        frame_pitches = [140.0, 200.0, 260.0]
        frame_energies = [0.4, 0.6, 0.8]
        
        result = analyzer.analyze_from_features(features, frame_energies, frame_pitches)
        
        self.assertIsInstance(result, EmotionResult)
        self.assertIn(result.primary, ["happy", "excited", "angry", "sad", "tired", "anxious", "neutral"])
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestUrgency(unittest.TestCase):
    """Tests for urgency module."""
    
    def test_assess_from_features(self):
        """Test urgency assessment from features."""
        assessor = UrgencyAssessor()
        
        features = AudioFeatures(
            duration_sec=3.0,
            sample_rate=16000,
            pitch_mean=200.0,
            pitch_std=60.0,    # High variation = high urgency
            energy_mean=0.8,   # High energy
            energy_std=0.3,
            zero_crossing_rate=0.1,
            silence_ratio=0.05,  # Few pauses
        )
        frame_energies = [0.7, 0.8, 0.9, 0.85, 0.75]
        
        result = assessor.assess_from_features(features, frame_energies)
        
        self.assertIsInstance(result, UrgencyResult)
        self.assertIn(result.level, ["low", "medium", "high", "critical"])
        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 1.0)


class TestSarcasm(unittest.TestCase):
    """Tests for sarcasm module."""
    
    def test_detect_from_analysis(self):
        """Test sarcasm detection from analysis."""
        detector = SarcasmDetector()
        
        emotion = EmotionResult(
            primary="sad",
            confidence=0.8,
            secondary=None,
            scores={"sad": 0.8, "neutral": 0.2},
        )
        
        features = AudioFeatures(
            duration_sec=3.0,
            sample_rate=16000,
            pitch_mean=120.0,
            pitch_std=15.0,
            energy_mean=0.3,
            energy_std=0.05,
            zero_crossing_rate=0.05,
            silence_ratio=0.2,
        )
        
        # Positive text with negative emotion should indicate sarcasm
        result = detector.detect_from_analysis(
            "这真是太棒了",  # "This is really great" in Chinese
            emotion,
            features,
        )
        
        self.assertIsInstance(result, SarcasmResult)
        self.assertIsInstance(result.is_sarcastic, bool)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


class TestAnalyzer(unittest.TestCase):
    """Tests for analyzer module."""
    
    def setUp(self):
        """Create temporary test audio file."""
        # Windows: use mkstemp and close immediately
        fd, self.temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self._create_test_wav(self.temp_path)
        self.analyzer = SpeechAnalyzer()
    
    def tearDown(self):
        """Clean up temporary file."""
        try:
            os.unlink(self.temp_path)
        except (PermissionError, OSError):
            pass
    
    def _create_test_wav(self, path: str, duration: float = 2.0):
        """Create a test WAV file."""
        import math
        sample_rate = 16000
        n_samples = int(sample_rate * duration)
        
        samples = array('h')
        for i in range(n_samples):
            value = int(10000 * math.sin(2 * math.pi * 200 * i / sample_rate))
            samples.append(value)
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
    
    def test_analyze(self):
        """Test basic analysis."""
        result = self.analyzer.analyze(self.temp_path)
        
        self.assertIn("emotion", result)
        self.assertIn("features", result)
        self.assertIn("speaker_state", result)
        self.assertIsNone(result["transcription"])  # Lite tier
    
    def test_assess_urgency(self):
        """Test urgency assessment."""
        result = self.analyzer.assess_urgency(self.temp_path)
        
        self.assertIsInstance(result, UrgencyResult)
        self.assertIn(result.level, ["low", "medium", "high", "critical"])
    
    def test_detect_sarcasm(self):
        """Test sarcasm detection."""
        result = self.analyzer.detect_sarcasm(
            self.temp_path,
            text="这真是太棒了",
        )
        
        self.assertIsInstance(result, SarcasmResult)
    
    def test_full_analysis(self):
        """Test full analysis."""
        result = self.analyzer.full_analysis(
            self.temp_path,
            text="这真是太棒了",
        )
        
        self.assertIn("summary", result)
        self.assertIn("emotion_analysis", result)
        self.assertIn("urgency_assessment", result)
        self.assertIn("sarcasm_detection", result)
        self.assertIn("interpretation", result)


def create_test_suite():
    """Create a test suite with all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioFeatures))
    suite.addTests(loader.loadTestsFromTestCase(TestEmotion))
    suite.addTests(loader.loadTestsFromTestCase(TestUrgency))
    suite.addTests(loader.loadTestsFromTestCase(TestSarcasm))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalyzer))
    
    return suite


if __name__ == "__main__":
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(create_test_suite())
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
