import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import fast_tts


class NarratorMemoryTests(unittest.TestCase):
    def test_overlap_workspace_can_move_qwen_decoder_to_cpu(self):
        embedding, decoder = object(), object()
        config = SimpleNamespace(num_hidden_layers=1, num_key_value_heads=1, head_dim=64)
        model = SimpleNamespace(config=SimpleNamespace(talker_config=config),
                                talker=SimpleNamespace(model=SimpleNamespace(text_embedding=embedding)),
                                speech_tokenizer=SimpleNamespace(model=decoder))
        base = SimpleNamespace(model=model)
        def nbytes(module, dtype=None):
            return (1 if module is embedding else 0.25 if module is decoder else 3) * 2**30
        with patch.object(fast_tts, "_nbytes", nbytes), \
                patch("torch.cuda.empty_cache"), \
                patch("torch.cuda.mem_get_info", return_value=(4 * 2**30, 8 * 2**30)), \
                patch("torch.cuda.get_per_process_memory_fraction", return_value=1), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=8 * 2**30)), \
                patch("torch.cuda.memory_reserved", return_value=0):
            ordinary = fast_tts._plan(base, "cuda:0", torch.bfloat16, 128, False, None, None)
            overlap = fast_tts._plan(base, "cuda:0", torch.bfloat16, 128, False, None, None,
                                     memory_reserve_bytes=2**30)
        self.assertTrue(ordinary.decoder_on_gpu)
        self.assertFalse(overlap.decoder_on_gpu)
