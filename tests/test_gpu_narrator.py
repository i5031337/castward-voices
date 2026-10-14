import asyncio
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from narrator_tts import NarratorEngine
from server import create_app
from test_server import FakeTTS


class GpuNarratorTests(unittest.TestCase):
    def test_model_swap_cannot_change_dtype_during_cpu_narration(self):
        async def exercise():
            loading, release, inference = threading.Event(), threading.Event(), threading.Event()
            loads = []
            def loader(**kwargs):
                loads.append(True)
                if len(loads) > 1:
                    previous = torch.get_default_dtype()
                    try:
                        torch.set_default_dtype(torch.bfloat16)
                        loading.set()
                        if not release.wait(5):
                            raise RuntimeError("Test timed out")
                    finally:
                        torch.set_default_dtype(previous)
                return FakeTTS()
            narrator = NarratorEngine.__new__(NarratorEngine)
            narrator.validate = lambda body: "bm_george"
            narrator.packs = {"bm_george": None}
            def pipeline(*args, **kwargs):
                inference.set()
                yield SimpleNamespace(audio=torch.ones(2400) * 0.1)
            narrator.pipelines = {"b": pipeline}
            with tempfile.TemporaryDirectory() as storage:
                app = create_app(device="cpu", clone_model=None, voices_dir=storage,
                                 loader=loader, kokoro_loader=lambda **kw: narrator)
                async with app.router.lifespan_context(app):
                    loop = asyncio.get_running_loop()
                    app.state.engine.kind = None
                    swap = loop.run_in_executor(app.state.executor, app.state.engine.get, "design")
                    try:
                        self.assertTrue(await asyncio.to_thread(loading.wait, 2))
                        iterator = narrator.chunks(SimpleNamespace(input="Hello.", speed=1))
                        narration = loop.run_in_executor(app.state.narrator_executor, next, iterator)
                        await asyncio.sleep(0.02)
                        self.assertFalse(inference.is_set())
                    finally:
                        release.set()
                    await swap
                    self.assertEqual((await narration).dtype.name, "float32")
                    await loop.run_in_executor(app.state.narrator_executor, iterator.close)
        asyncio.run(exercise())

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.empty_cache")
    def test_cuda_narrator_loads_before_qwen_on_same_worker(self, *_):
        async def exercise():
            calls = []
            def narrator_loader(**kwargs):
                calls.append(("kokoro", kwargs["device"], threading.get_ident()))
                return object()
            def qwen_loader(**kwargs):
                calls.append(("qwen", kwargs["device"], threading.get_ident()))
                return FakeTTS()
            with tempfile.TemporaryDirectory() as storage:
                app = create_app(device="cuda:0", kokoro_device="cuda:0", clone_model=None,
                                 voices_dir=storage, loader=qwen_loader, kokoro_loader=narrator_loader)
                async with app.router.lifespan_context(app):
                    self.assertEqual([row[:2] for row in calls],
                                     [("kokoro", "cuda:0"), ("qwen", "cuda:0")])
                    self.assertEqual(calls[0][2], calls[1][2])
                    self.assertIs(app.state.executor, app.state.narrator_executor)
                    entered, release = threading.Event(), threading.Event()
                    def capture():
                        entered.set()
                        if not release.wait(5):
                            raise RuntimeError("Test timed out")
                    loop = asyncio.get_running_loop()
                    task = loop.run_in_executor(app.state.executor, capture)
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                        narration = loop.run_in_executor(app.state.narrator_executor, lambda: "audio")
                        await asyncio.sleep(0.02)
                        self.assertFalse(narration.done(), "Kokoro ran during graph capture")
                    finally:
                        release.set()
                    await task
                    self.assertEqual(await narration, "audio")
        asyncio.run(exercise())

    @patch("torch.cuda.set_device")
    @patch("torch.cuda.empty_cache")
    def test_cpu_narration_keeps_independent_worker(self, *_):
        async def exercise():
            with tempfile.TemporaryDirectory() as storage:
                app = create_app(device="cuda:0", kokoro_device="cpu", clone_model=None,
                                 voices_dir=storage, loader=lambda **kw: FakeTTS(),
                                 kokoro_loader=lambda **kw: object())
                async with app.router.lifespan_context(app):
                    self.assertIsNot(app.state.executor, app.state.narrator_executor)
                    self.assertFalse(app.state.shared_cuda_worker)
        asyncio.run(exercise())
