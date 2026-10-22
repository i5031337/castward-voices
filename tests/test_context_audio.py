from types import SimpleNamespace
import unittest
import numpy as np
from context_audio import contextual_cut

def token(text,start,end):
    return SimpleNamespace(text=text,start_ts=start,end_ts=end)

class ContextTests(unittest.TestCase):
    def test_middle_span_extraction_keeps_target_and_excludes_neighbors(self):
            words = [token("Hello", 0.05, 0.2), token("he", 0.3, 0.4),
                     token("said", 0.4, 0.6), token("Goodbye", 0.8, 0.9)]
            waveform = np.linspace(0.01, 0.5, 24000, dtype=np.float32)
            cut = contextual_cut(waveform, words, '"Hello," ', "he said.", ' "Goodbye."')
            self.assertEqual(len(cut), round(0.68 * 24000) - round(0.26 * 24000))
            np.testing.assert_array_equal(cut[120:-120], waveform[round(0.26 * 24000) + 120:round(0.68 * 24000) - 120])
            self.assertEqual(float(cut[0]), 0)
            self.assertEqual(float(cut[-1]), 0)

    def test_repeated_words_use_offsets_not_a_text_search(self):
            words = [token("said", 0.1, 0.2), token("said", 0.4, 0.6), token("said", 0.8, 0.9)]
            waveform = np.ones(24000, dtype=np.float32)
            cut = contextual_cut(waveform, words, "said ", "said", " said")
            self.assertEqual(len(cut), round(0.68 * 24000) - round(0.36 * 24000))

    def test_unsafe_native_timings_and_token_coverage_are_rejected(self):
            waveform = np.ones(24000, dtype=np.float32)
            for words in ([], [token("wrong", 0.1, 0.3)], [token("hello", 0.1, 0.1)],
                          [token("hello", None, None)], [token("hello", 0.9, 1.2)],
                          [token("hello", 0.1, float("nan"))]):
                with self.assertRaises(ValueError):
                    contextual_cut(waveform, words, "", "hello", "")
            with self.assertRaises(ValueError):
                contextual_cut(waveform, [token("hellohe", 0.1, 0.4)], "hello", "he", "")
