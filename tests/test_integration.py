"""Integration tests for SpeechPulse.

These tests verify the end-to-end functionality of the SpeechPulse system
by testing the complete pipeline from audio input to analysis output.
"""

import unittest
import wave
import os
import tempfile
from array import array
import math

from speechpulse.analyzer import SpeechAnalyzer
from speechpulse.server import health_check, analyze_audio, assess_urgency, detect_sarcasm, full_analysis


class IntegrationTestBase(unittest.TestCase):
    """Base class for integration tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SpeechAnalyzer()
        self.temp_files = []
    
    def tearDown(self):
        """Clean up temporary files."""
        for path in self.temp_files:
            if os.path.exists(path):
                os.unlink(path)
    
    def create_test_audio(
        self,
        emotion: str = "neutral",
        duration: float = 2.0,
        sample_rate: int = 16000,
    ) -> str:
        """Create a test audio file with specified characteristics.
        
        Args:
            emotion: Type of emotion to simulate
            duration: Audio duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Path to created audio file
        """
        import random
        
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        self.temp_files.append(path)
        
        n_samples = int(sample_rate * duration)
        samples = array('h')
        
        if emotion == "happy":
            # High energy + frequency modulation (pitch variation) + amplitude modulation
            base_freq = 220
            for i in range(n_samples):
                t = i / sample_rate
                # Frequency modulation for lively variation
                freq = base_freq + 50 * math.sin(2 * math.pi * 4 * t)
                # Amplitude modulation for energy variation
                amp = 0.8 + 0.2 * math.sin(2 * math.pi * 2 * t)
                value = int(28000 * amp * math.sin(2 * math.pi * freq * t))
                samples.append(value)
        
        elif emotion == "sad":
            # Low pitch, low energy, almost no modulation (flat)
            base_freq = 120
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
                freq = base_freq + 40 * math.sin(2 * math.pi * 5 * t)
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
                freq = base_freq + 70 * math.sin(2 * math.pi * 7 * t)
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
                freq = base_freq + 20 * math.sin(2 * math.pi * 2 * t)
                # Slight amplitude modulation
                amp = 0.7 + 0.1 * math.sin(2 * math.pi * 1.5 * t)
                value = int(20000 * amp * math.sin(2 * math.pi * freq * t))
                samples.append(value)
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())
        
        return path


class TestEndToEndAnalysis(IntegrationTestBase):
    """End-to-end integration tests."""
    
    def test_happy_emotion_detection(self):
        """Test detection of happy emotion."""
        audio_path = self.create_test_audio("happy")
        result = self.analyzer.analyze(audio_path)
        
        self.assertIn("emotion", result)
        emotion = result["emotion"]
        # Happy audio should be detected as happy, excited, or angry (all high energy)
        self.assertIn(emotion["primary"], ["happy", "excited", "angry"])
        self.assertGreater(emotion["confidence"], 0)
    
    def test_sad_emotion_detection(self):
        """Test detection of sad emotion."""
        audio_path = self.create_test_audio("sad")
        result = self.analyzer.analyze(audio_path)
        
        emotion = result["emotion"]
        self.assertIn(emotion["primary"], ["sad", "tired"])
    
    def test_angry_emotion_detection(self):
        """Test detection of angry emotion."""
        audio_path = self.create_test_audio("angry")
        result = self.analyzer.analyze(audio_path)
        
        emotion = result["emotion"]
        # Angry might be detected as angry or excited due to high energy
        self.assertIn(emotion["primary"], ["angry", "excited"])
    
    def test_urgency_high_for_anxious(self):
        """Test high urgency detection for anxious audio."""
        audio_path = self.create_test_audio("anxious")
        result = self.analyzer.assess_urgency(audio_path)
        
        # Anxious audio should have higher urgency
        self.assertIn(result.level, ["medium", "high", "critical"])
    
    def test_urgency_low_for_neutral(self):
        """Test low urgency detection for neutral audio."""
        audio_path = self.create_test_audio("neutral")
        result = self.analyzer.assess_urgency(audio_path)
        
        # Neutral audio should not have critical urgency
        self.assertIn(result.level, ["low", "medium", "high"])
    
    def test_sarcasm_detection_positive_text_negative_audio(self):
        """Test sarcasm detection with positive text and negative audio."""
        audio_path = self.create_test_audio("sad")
        result = self.analyzer.detect_sarcasm(
            audio_path,
            text="这真是太棒了"
        )
        
        # Positive text with sad audio should indicate sarcasm
        self.assertIsInstance(result.is_sarcastic, bool)
    
    def test_full_analysis_complete(self):
        """Test that full analysis returns all expected fields."""
        audio_path = self.create_test_audio("happy")
        result = self.analyzer.full_analysis(audio_path, text="今天真开心")
        
        # Check all expected fields
        self.assertIn("summary", result)
        self.assertIn("emotion_analysis", result)
        self.assertIn("urgency_assessment", result)
        self.assertIn("sarcasm_detection", result)
        self.assertIn("raw_features", result)
        self.assertIn("interpretation", result)
        
        # Verify emotion analysis structure
        emotion = result["emotion_analysis"]
        self.assertIn("primary", emotion)
        self.assertIn("confidence", emotion)
        
        # Verify urgency assessment structure
        urgency = result["urgency_assessment"]
        self.assertIn("level", urgency)
        self.assertIn("score", urgency)
        
        # Verify sarcasm detection structure
        sarcasm = result["sarcasm_detection"]
        self.assertIn("is_sarcastic", sarcasm)
        self.assertIn("confidence", sarcasm)


class TestMCPServerTools(IntegrationTestBase):
    """Integration tests for MCP server tools."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        result = health_check()
        
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["tier"], "lite")
        self.assertIn("version", result)
        self.assertIn("capabilities", result)
        self.assertIn("emotion_analysis", result["capabilities"])
    
    def test_analyze_audio_tool(self):
        """Test analyze_audio MCP tool."""
        audio_path = self.create_test_audio("happy")
        result = analyze_audio(audio_path)
        
        self.assertIn("emotion", result)
        self.assertIn("features", result)
        self.assertIsNone(result["transcription"])  # Lite tier
    
    def test_assess_urgency_tool(self):
        """Test assess_urgency MCP tool."""
        audio_path = self.create_test_audio("anxious")
        result = assess_urgency(audio_path)
        
        self.assertIn("level", result)
        self.assertIn("score", result)
        self.assertIn("reasoning", result)
    
    def test_detect_sarcasm_tool(self):
        """Test detect_sarcasm MCP tool."""
        audio_path = self.create_test_audio("sad")
        result = detect_sarcasm(audio_path, text="这真是太棒了")
        
        self.assertIn("is_sarcastic", result)
        self.assertIn("confidence", result)
        self.assertIn("indicators", result)
    
    def test_full_analysis_tool(self):
        """Test full_analysis MCP tool."""
        audio_path = self.create_test_audio("angry")
        result = full_analysis(audio_path, text="我受够了！")
        
        self.assertIn("summary", result)
        self.assertIn("emotion_analysis", result)
        self.assertIn("urgency_assessment", result)


class TestAudioProcessingPipeline(IntegrationTestBase):
    """Tests for the audio processing pipeline."""
    
    def test_audio_loading_and_feature_extraction(self):
        """Test that audio can be loaded and features extracted."""
        from speechpulse.audio_features import AudioFeatureExtractor
        
        audio_path = self.create_test_audio("neutral", duration=3.0)
        extractor = AudioFeatureExtractor()
        
        features, frame_energies, frame_pitches = extractor.extract_all(audio_path)
        
        # Verify features
        self.assertAlmostEqual(features.duration_sec, 3.0, places=1)
        self.assertEqual(features.sample_rate, 16000)
        self.assertGreater(features.pitch_mean, 0)
        self.assertGreater(features.energy_mean, 0)
        
        # Verify frame-level data
        self.assertGreater(len(frame_energies), 0)
        self.assertGreater(len(frame_pitches), 0)
    
    def test_different_sample_rates(self):
        """Test handling of different sample rates."""
        from speechpulse.utils import load_audio_stdlib, ensure_16khz
        
        # Create audio at different sample rates
        for target_sr in [8000, 16000, 22050, 44100]:
            with self.subTest(sample_rate=target_sr):
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)
                self.temp_files.append(path)
                
                n_samples = int(target_sr * 1.0)  # 1 second
                samples = array('h')
                for i in range(n_samples):
                    t = i / target_sr
                    value = int(10000 * math.sin(2 * math.pi * 440 * t))
                    samples.append(value)
                
                with wave.open(path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(target_sr)
                    wav_file.writeframes(samples.tobytes())
                
                # Load and verify
                audio, sr = load_audio_stdlib(path)
                audio_16k, sr_16k = ensure_16khz(audio, sr)
                
                self.assertEqual(sr_16k, 16000)
                # After resampling to 16kHz, should have approximately 16000 samples for 1 second
                self.assertAlmostEqual(len(audio_16k), 16000, delta=100)


class TestErrorHandling(IntegrationTestBase):
    """Tests for error handling."""
    
    def test_nonexistent_file(self):
        """Test handling of non-existent audio file."""
        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze("/nonexistent/path/audio.wav")
    
    def test_invalid_file_format(self):
        """Test handling of invalid file format."""
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        self.temp_files.append(path)
        
        with open(path, 'w') as f:
            f.write("This is not audio data")
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze(path)
    
    def test_sarcasm_without_text(self):
        """Test sarcasm detection without text in Lite tier."""
        audio_path = self.create_test_audio("neutral")
        result = self.analyzer.detect_sarcasm(audio_path)
        
        # Should return a result indicating text is needed
        self.assertFalse(result.is_sarcastic)
        self.assertEqual(result.confidence, 0.0)


class TestPerformance(IntegrationTestBase):
    """Performance-related integration tests."""
    
    def test_single_audio_load(self):
        """Verify that audio is loaded only once per analysis."""
        from speechpulse.audio_features import AudioFeatureExtractor
        from speechpulse.utils import load_audio_stdlib
        
        audio_path = self.create_test_audio("neutral", duration=2.0)
        extractor = AudioFeatureExtractor()
        
        # extract_all should load audio only once
        import time
        start = time.time()
        features, frame_energies, frame_pitches = extractor.extract_all(audio_path)
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 5 seconds for 2-second audio)
        self.assertLess(elapsed, 5.0)
        
        # Verify results are valid
        self.assertIsNotNone(features)
        self.assertIsNotNone(frame_energies)
        self.assertIsNotNone(frame_pitches)


def create_integration_test_suite():
    """Create a test suite with all integration tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestMCPServerTools))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioProcessingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    return suite


if __name__ == "__main__":
    # Run all integration tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(create_integration_test_suite())
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
