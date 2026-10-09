import asyncio
import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from narrator_tts import NarratorEngine, text_chunks
from server import create_app
from test_server import FakeTTS


class FakeNarrator:
    voices = ["af_alloy", "bm_george"]
    validate = NarratorEngine.validate

    def __init__(self):
        self.calls = []
        self.active = 0
        self.max_active = 0
        self.first = np.ones(2400, dtype=np.float32) * 0.1
        self.second = np.ones(4800, dtype=np.float32) * -0.1

    def chunks(self, body):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append(body)
        try:
            yield self.first
            yield self.second
        finally:
            self.active -= 1

    def generate(self, body):
        return np.concatenate(list(self.chunks(body))), 24000


class NarratorTests(unittest.TestCase):
    def setUp(self):
        self.narrator, self.qwen = FakeNarrator(), FakeTTS()
        self.storage = tempfile.TemporaryDirectory()
        self.app = create_app(device="cpu", loader=lambda **kwargs: self.qwen, clone_model=None,
                              voices_dir=self.storage.name, kokoro_loader=lambda **kwargs: self.narrator)
        self.client = TestClient(self.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.storage.cleanup()

    def request(self, **kwargs):
        body = {"model": "kokoro", "voice": "narrator", "input": "A long story. Another sentence.",
                "response_format": "wav"}
        return self.client.post("/v1/audio/speech", json={**body, **kwargs})

    def test_routing_formats_speed_and_model_discovery(self):
        for fmt in ("wav", "flac", "mp3", "pcm", "opus", "aac"):
            response = self.request(response_format=fmt, speed=1.5)
            self.assertEqual(response.status_code, 200, response.text if response.status_code != 200 else "")
            if fmt == "wav":
                wav, sr = sf.read(io.BytesIO(response.content))
                self.assertEqual((len(wav), sr), (7200, 24000))  # Speed isn't applied twice.
        self.assertTrue(all(body.speed == 1.5 for body in self.narrator.calls))
        self.assertEqual(self.qwen.calls, [])
        self.assertIn("kokoro", [model["id"] for model in self.client.get("/v1/models").json()["data"]])
        self.assertEqual(len(self.client.get("/v1/voices").json()["data"]), 2)

    def test_invalid_narration_and_streaming_requests(self):
        for kwargs in ({"voice": "../../evil.pt"}, {"instructions": "A pirate"}, {"npc_id": "guard"},
                       {"language": "Japanese"}, {"stream": True}, {"input": "x" * 241}):
            self.assertEqual(self.request(**kwargs).status_code, 400)
        self.assertEqual(self.request(model="qwen3-tts", stream=True).status_code, 400)
        self.assertEqual(self.narrator.calls, [])

    def test_long_input_routes_only_to_narrator(self):
        text = "The road is long. " * 300
        self.assertGreater(len(text), 4096)
        self.assertEqual(self.request(input=text).status_code, 200)
        self.assertEqual(self.request(input=text, model="qwen3-tts").status_code, 400)

    def test_narration_does_not_wait_for_qwen_worker(self):
        entered, release = threading.Event(), threading.Event()
        original = self.qwen.generate_voice_design
        def blocked(**kwargs):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Qwen test timed out")
            return original(**kwargs)
        self.qwen.generate_voice_design = blocked
        with ThreadPoolExecutor(max_workers=1) as executor:
            qwen = executor.submit(self.request, model="qwen3-tts")
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.request().status_code, 200)
                self.assertFalse(qwen.done())
            finally:
                release.set()
            self.assertEqual(qwen.result().status_code, 200)

    def test_pcm_body_is_sent_before_next_chunk_and_generator_closes(self):
        async def exercise(disconnect=False):
            sent, closed = [], []
            def chunks(body):
                try:
                    yield self.narrator.first
                    self.assertTrue(sent, "Second inference started before first audio delivery")
                    yield self.narrator.second
                finally:
                    closed.append(True)
            self.narrator.chunks = chunks
            app = create_app(device="cpu", clone_model=None, loader=lambda **kwargs: self.qwen,
                             voices_dir=self.storage.name, kokoro_loader=lambda **kwargs: self.narrator)
            async with app.router.lifespan_context(app):
                body = json.dumps({"model": "kokoro", "input": "First. Second.", "voice": "narrator",
                                   "stream": True, "response_format": "pcm"}).encode()
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                async def send(message):
                    if message["type"] == "http.response.body" and message["body"]:
                        sent.append(message["body"])
                        if disconnect:
                            raise OSError("Client disconnected")
                scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                         "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/audio/speech",
                         "raw_path": b"/v1/audio/speech", "query_string": b"", "root_path": "",
                         "headers": [(b"content-type", b"application/json")],
                         "server": ("test", 80), "client": ("test", 1)}
                if disconnect:
                    with self.assertRaises(ClientDisconnect):
                        await app(scope, receive, send)
                else:
                    await app(scope, receive, send)
            self.assertTrue(closed)
            if disconnect:
                self.assertEqual(len(sent), 1)
                return
            self.assertEqual([len(chunk) for chunk in sent], [4800, 9600])
            pcm = np.frombuffer(b"".join(sent), dtype="<i2") / 32768
            np.testing.assert_allclose(pcm, np.concatenate([self.narrator.first, self.narrator.second]), atol=1/32768)
        asyncio.run(exercise())
        asyncio.run(exercise(disconnect=True))

    def test_chunking_preserves_every_word_and_bounds_unpunctuated_text(self):
        text = "First sentence. " + "long " * 300 + "Last sentence."
        chunks = list(text_chunks(text))
        self.assertEqual(" ".join(chunks).split(), text.split())
        self.assertTrue(all(len(chunk) <= 240 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
