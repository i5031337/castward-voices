import io
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from server import create_app, encode_audio


class FakeTTS:
    plan = "test"

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = []
        self.thread_ids = set()

    def generate_voice_design(self, **kwargs):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.thread_ids.add(threading.get_ident())
        try:
            time.sleep(0.01)
            self.calls.append(kwargs)
            if kwargs["text"] == "fail":
                raise RuntimeError("test failure")
            return [np.sin(np.arange(2400) * 0.1).astype(np.float32) * 0.1], 24000
        finally:
            self.active -= 1


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.tts = FakeTTS()
        self.loads = []

        def loader(**kwargs):
            self.loads.append((kwargs, threading.get_ident()))
            return self.tts

        self.client = TestClient(create_app(device="cpu", api_key="test-key", loader=loader))
        self.client.__enter__()
        self.headers = {"Authorization": "Bearer test-key"}

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def request(self, **kwargs):
        body = {"model": "qwen3-tts", "input": "Hello", "voice": "alloy", "response_format": "wav"}
        body.update(kwargs)
        return self.client.post("/v1/audio/speech", json=body, headers=self.headers)

    def test_speech_contract_and_voice_design(self):
        response = self.request(instructions="A gruff pirate", language="English", seed=42)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        audio, sr = sf.read(io.BytesIO(response.content))
        self.assertEqual(sr, 24000)
        self.assertEqual(len(audio), 2400)
        self.assertEqual(self.tts.calls[-1]["instruct"], "A gruff pirate")
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.tts.thread_ids, {self.loads[0][1]})

    def test_openai_defaults_and_model_discovery(self):
        response = self.client.post("/v1/audio/speech", headers=self.headers,
                                    json={"model": "tts-1", "input": "Hello", "voice": "alloy"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/mpeg")
        info = sf.info(io.BytesIO(response.content))
        self.assertEqual(info.format, "MP3")
        models = self.client.get("/v1/models", headers=self.headers).json()
        self.assertEqual(models["data"][0]["id"], "qwen3-tts")

    def test_validation_and_auth(self):
        for override in ({"model": "missing"}, {"input": " "}, {"input": "x" * 4097},
                         {"speed": 0}, {"stream_format": "sse"}, {"language": "missing"},
                         {"response_format": "invalid"}):
            with self.subTest(override=override):
                response = self.request(**override)
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["error"]["type"], "invalid_request_error")
        self.assertEqual(self.client.get("/v1/models").status_code, 401)
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.tts.calls, [])

    def test_concurrent_requests_are_serialized_and_failures_recover(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(executor.map(lambda _: self.request(), range(3)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(self.tts.max_active, 1)
        self.assertEqual(len(self.loads), 1)
        self.assertEqual(self.request(input="fail").status_code, 500)
        self.assertEqual(self.request().status_code, 200)

    def test_lossless_formats_and_pcm(self):
        wav = np.sin(np.arange(24000) * 0.1).astype(np.float32) * 0.1
        for fmt in ("wav", "flac"):
            decoded, sr = sf.read(io.BytesIO(encode_audio(wav, 24000, fmt, 1)))
            self.assertEqual(sr, 24000)
            np.testing.assert_allclose(decoded, wav, atol=1 / 32768)
        pcm = np.frombuffer(encode_audio(wav, 24000, "pcm", 1), dtype="<i2")
        self.assertEqual(len(pcm), len(wav))
        np.testing.assert_allclose(pcm / 32768, wav, atol=1 / 32768)


if __name__ == "__main__":
    unittest.main()
