import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from reference_audio import ReferenceSelector, select_aligned_segment


class ReferenceTests(unittest.TestCase):
    def test_long_audio_selects_complete_sentences_with_matching_text(self):
        text = "The gate is open. The road is safe. The night is cold."
        words = [SimpleNamespace(word=" " + word, start=i, end=i + 0.9)
                 for i, word in enumerate(text.split())]
        wav = np.arange(12000, dtype=np.float32)
        segment, transcript = select_aligned_segment(wav, 1000, text, words)
        self.assertTrue(3 <= len(segment) / 1000 <= 10)
        self.assertEqual(transcript, "The gate is open. The road is safe.")
        self.assertEqual(segment[0], wav[0])
        self.assertLess(segment[-1], 8000)

    def test_alignment_with_missing_text_or_zero_duration_is_rejected(self):
        wav = np.zeros(20000)
        with self.assertRaises(ValueError):
            select_aligned_segment(wav, 1000, "Hello there.", [SimpleNamespace(word="Hello", start=0, end=4)])
        with self.assertRaises(ValueError):
            select_aligned_segment(wav, 1000, "Hello.", [SimpleNamespace(word="Hello.", start=0, end=0)])

    def test_suitable_reference_keeps_exact_audio_and_text(self):
        wav = np.arange(4000)
        selected, text, source = ReferenceSelector().select(wav, 1000, "Exact text!", "English")
        np.testing.assert_array_equal(selected, wav)
        self.assertEqual((text, source), ("Exact text!", "dialogue"))

    def test_short_and_truncated_dialogue_require_extension(self):
        with self.assertRaises(ValueError):
            ReferenceSelector().select(np.zeros(1000), 1000, "Hello", "English")
        with self.assertRaises(ValueError):
            ReferenceSelector().select(np.zeros(4000), 1000, "Incomplete text", "English", may_be_truncated=True)

    def test_auto_language_is_detected_before_long_dialogue_alignment(self):
        text = "The gate is open. The road is safe. The night is cold."
        words = [SimpleNamespace(word=" " + word, start=i, end=i + 0.9)
                 for i, word in enumerate(text.split())]
        selector = ReferenceSelector()
        selector.model = Mock()
        selector.model.dims.n_mels = 80
        selector.model.detect_language.return_value = (None, {"en": 0.9, "de": 0.1})
        selector.model.align.return_value.all_words.return_value = words
        whisper = SimpleNamespace(pad_or_trim=lambda audio: audio,
                                  log_mel_spectrogram=lambda audio, **kwargs: audio)
        with patch("reference_audio.ALIGNMENT_MODEL") as checkpoint, patch.dict("sys.modules", whisper=whisper):
            checkpoint.is_file.return_value = True
            _, transcript, source = selector.select(np.zeros(12000), 1000, text, "Auto")
        self.assertEqual(selector.model.align.call_args.kwargs["language"], "en")
        self.assertEqual(source, "aligned_dialogue")
        self.assertIn(transcript, text)
