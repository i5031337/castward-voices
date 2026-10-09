import asyncio
import hashlib
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from server import SpeechRequest, create_app


class FakeModel:
    plan = "test"

    def __init__(self, kind, calls):
        self.kind = kind
        self.calls = calls

    def generate_voice_design(self, **kwargs):
        self.calls.append(("design", kwargs))
        length = 12000 if kwargs["text"] == "short" else 96000
        return [np.sin(np.arange(length) * 0.1).astype(np.float32) * 0.1], 24000

    def create_voice_clone_prompt(self, **kwargs):
        import torch
        self.calls.append(("extract", kwargs))
        xvec = kwargs.get("x_vector_only_mode", False)
        return [SimpleNamespace(ref_code=None if xvec else torch.ones((48, 16), dtype=torch.long),
                                ref_spk_embedding=torch.ones(1024), ref_text=kwargs.get("ref_text"),
                                x_vector_only_mode=xvec, icl_mode=not xvec)]

    def _prompt_items_to_voice_clone_prompt(self, items):
        return {key: [getattr(item, key) for item in items]
                for key in ("ref_code", "ref_spk_embedding", "x_vector_only_mode", "icl_mode")}

    def generate_voice_clone(self, **kwargs):
        self.calls.append(("clone", kwargs))
        length = 96000 if kwargs["text"].startswith("Welcome, traveler.") else 2400
        return [np.sin(np.arange(length) * 0.1).astype(np.float32) * 0.1], 24000


class NPCTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.calls = []
        self.loads = []

        def loader(**kwargs):
            self.loads.append(kwargs["model_id"])
            kind = "clone" if kwargs["model_id"] == "fake-clone" else "design"
            return FakeModel(kind, self.calls)

        self.loader = loader
        self.start()

    def start(self):
        self.client = TestClient(create_app(device="cpu", clone_model="fake-clone", loader=self.loader, kokoro_enabled=False,
                                            voices_dir=self.storage.name))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.storage.cleanup()

    def request(self, **kwargs):
        body = {"model": "qwen3-tts", "input": "Hello", "response_format": "wav", "npc_id": "guard-1"}
        body.update(kwargs)
        return self.client.post("/v1/audio/speech", json=body)

    def test_create_reuse_and_restart_without_reextracting(self):
        self.assertEqual(self.request(instructions="A gruff guard").status_code, 200)
        folder = Path(self.storage.name) / hashlib.sha256(b"guard-1").hexdigest()
        original = (folder / "reference.wav").read_bytes()
        self.assertEqual(self.request(instructions="Changed description").status_code, 200)
        self.assertEqual((folder / "reference.wav").read_bytes(), original)
        self.assertEqual(json.loads((folder / "metadata.json").read_text())["description"], "A gruff guard")
        self.client.__exit__(None, None, None)
        self.start()
        self.assertEqual(self.request(model="qwen3-tts-0.6b").status_code, 200)
        self.assertEqual([kind for kind, _ in self.calls].count("design"), 1)
        self.assertEqual([kind for kind, _ in self.calls].count("extract"), 1)
        self.assertEqual([kind for kind, _ in self.calls].count("clone"), 2)
        self.assertEqual(self.loads, ["fake-clone", None, "fake-clone", "fake-clone"])
        self.assertEqual(json.loads((folder / "metadata.json").read_text())["ref_text"], "Hello")

    def test_concurrent_creation_only_creates_one_reference(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(executor.map(lambda _: self.request(instructions="A guard"), range(3)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual([kind for kind, _ in self.calls].count("design"), 1)
        self.assertEqual([kind for kind, _ in self.calls].count("extract"), 1)

    def test_unknown_voice_and_invalid_ids(self):
        self.assertEqual(self.request().status_code, 400)
        self.assertEqual(self.request(npc_id="../../outside", instructions="A guard").status_code, 400)
        self.assertEqual(self.client.post("/v1/audio/speech", json={"model": "qwen3-tts-0.6b", "input": "Hello"}).status_code, 400)
        self.assertEqual(list(Path(self.storage.name).iterdir()), [])

    def test_failed_extraction_preserves_first_audio_and_can_retry(self):
        original_loader = self.loader

        def bad_loader(**kwargs):
            model = original_loader(**kwargs)
            if kwargs["model_id"] == "fake-clone":
                def fail(**kwargs):
                    raise RuntimeError("extraction failed")
                model.create_voice_clone_prompt = fail
            return model

        self.client.__exit__(None, None, None)
        self.loader = bad_loader
        self.start()
        response = self.request(instructions="A guard")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-voice-status"], "preparing")
        folder = Path(self.storage.name) / hashlib.sha256(b"guard-1").hexdigest()
        original = (folder / "dialogue.wav").read_bytes()
        self.assertEqual(self.client.get("/v1/voices/guard-1").json()["status"], "failed")
        self.assertFalse((folder / "prompt.npz").exists())
        self.client.__exit__(None, None, None)
        self.loader = original_loader
        self.start()
        self.assertEqual(self.request().status_code, 200)
        self.assertEqual((folder / "dialogue.wav").read_bytes(), original)
        self.assertEqual(self.client.get("/v1/voices/guard-1").json()["status"], "ready")
        self.assertEqual([kind for kind, _ in self.calls].count("design"), 1)

    def test_short_dialogue_is_extended_in_background_using_its_speaker(self):
        response = self.request(input="short", instructions="A guard")
        self.assertEqual(response.status_code, 200)
        folder = Path(self.storage.name) / hashlib.sha256(b"guard-1").hexdigest()
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(metadata["reference_source"], "extended_dialogue")
        self.assertEqual(metadata["dialogue_text"], "short")
        self.assertGreaterEqual(metadata["reference_seconds"], 3)
        self.assertEqual([kind for kind, _ in self.calls].count("design"), 1)
        extractions = [call for kind, call in self.calls if kind == "extract"]
        self.assertTrue(extractions[0]["x_vector_only_mode"])
        self.assertFalse(extractions[1]["x_vector_only_mode"])

    def test_audio_body_is_sent_before_clone_extraction(self):
        # TestClient waits for response background tasks. Capture ASGI messages to
        # prove the actual HTTP body is sent before preparation starts.
        async def exercise():
            calls, sent = [], []

            def loader(**kwargs):
                model = FakeModel("test", calls)
                original = model.create_voice_clone_prompt
                def extract(**kwargs):
                    self.assertTrue(sent, "Extraction started before the audio body was sent")
                    return original(**kwargs)
                model.create_voice_clone_prompt = extract
                return model

            with tempfile.TemporaryDirectory() as directory:
                app = create_app(device="cpu", clone_model="fake-clone", loader=loader, voices_dir=directory,
                                 kokoro_enabled=False)
                async with app.router.lifespan_context(app):
                    body = json.dumps({"model": "qwen3-tts", "input": "Hello", "npc_id": "first",
                                       "instructions": "A guard", "response_format": "wav"}).encode()
                    async def receive():
                        return {"type": "http.request", "body": body, "more_body": False}
                    async def send(message):
                        if message["type"] == "http.response.body":
                            self.assertTrue(message["body"].startswith(b"RIFF"))
                            self.assertEqual([kind for kind, _ in calls], ["design"])
                            self.assertEqual(app.state.engine.status("first"), "preparing")
                            sent.append(True)
                    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
                             "method": "POST", "scheme": "http", "path": "/v1/audio/speech",
                             "raw_path": b"/v1/audio/speech", "query_string": b"",
                             "headers": [(b"content-type", b"application/json")],
                             "server": ("test", 80), "client": ("test", 1), "root_path": ""}
                    await app(scope, receive, send)
                    self.assertEqual(app.state.engine.status("first"), "ready")
                    self.assertEqual([kind for kind, _ in calls], ["design", "extract"])
        asyncio.run(exercise())

    def test_restart_resumes_unfinished_reference_without_redesign(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as directory:
                def app():
                    return create_app(device="cpu", clone_model="fake-clone", loader=self.loader, kokoro_enabled=False,
                                      voices_dir=directory)
                first = app()
                async with first.router.lifespan_context(first):
                    body = SpeechRequest(model="qwen3-tts", input="Hello", npc_id="pending")
                    await asyncio.get_running_loop().run_in_executor(
                        first.state.executor, first.state.engine.generate, body, "A guard")
                    self.assertEqual(first.state.engine.status("pending"), "preparing")
                resumed = app()
                async with resumed.router.lifespan_context(resumed):
                    await asyncio.gather(*list(resumed.state.preparations.values()))
                    self.assertEqual(resumed.state.engine.status("pending"), "ready")
                self.assertEqual([kind for kind, _ in self.calls].count("design"), 1)
                self.assertEqual([kind for kind, _ in self.calls].count("extract"), 1)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
