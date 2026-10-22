import unittest

import torch

from narrator_rng import NarratorRandomMode


class NarratorRandomTests(unittest.TestCase):
    def test_narrator_noise_does_not_consume_or_reset_qwen_rng(self):
        previous = torch.get_rng_state()
        try:
            generator = torch.Generator().manual_seed(1234)
            torch.manual_seed(42)
            expected = torch.rand(10)
            torch.manual_seed(42)
            before = torch.get_rng_state().clone()
            with NarratorRandomMode(generator):
                uniform = torch.rand(10)
                normal = torch.randn_like(torch.ones(10))
                torch.randn(10)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertTrue(torch.equal(expected, torch.rand(10)))
            # Reseeding Qwen does not rewind Kokoro's independent generator.
            torch.manual_seed(42)
            with NarratorRandomMode(generator):
                self.assertFalse(torch.equal(uniform, torch.rand(10)))
                self.assertFalse(torch.equal(normal, torch.randn_like(torch.ones(10))))
        finally:
            torch.set_rng_state(previous)
