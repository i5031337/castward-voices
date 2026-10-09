"""Test real speech generation against a running local server."""
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf


def main():
    output = Path("out/api")
    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url="http://127.0.0.1:8001", timeout=120, trust_env=False) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/v1/models").json()["data"][0]["id"] == "qwen3-tts"
        body = {"model": "qwen3-tts", "input": "Hello. The speech model is ready.",
                "voice": "alloy", "instructions": "A warm, clear narrator",
                "seed": 42, "max_new_tokens": 80}
        for fmt in ("wav", "mp3", "flac", "pcm", "opus", "aac"):
            started = time.perf_counter()
            response = client.post("/v1/audio/speech", json={**body, "response_format": fmt})
            response.raise_for_status()
            assert response.content
            (output / f"speech.{fmt}").write_bytes(response.content)
            if fmt in {"wav", "mp3", "flac"}:
                audio, sr = sf.read(io.BytesIO(response.content))
                assert sr == 24000 and len(audio) > 0 and np.isfinite(audio).all()
                assert np.abs(audio).max() > 0.001
            print(json.dumps({"format": fmt, "bytes": len(response.content),
                              "seconds": round(time.perf_counter() - started, 3)}), flush=True)
        response = client.post("/v1/audio/speech", json={**body, "response_format": "wav", "speed": 2})
        response.raise_for_status()
        faster, sr = sf.read(io.BytesIO(response.content))
        original, _ = sf.read(output / "speech.wav")
        assert 0.4 < len(faster) / len(original) < 0.6
        (output / "speech_fast.wav").write_bytes(response.content)
        for bad in ({"model": "missing"}, {"stream_format": "sse"}, {"max_new_tokens": 4096},
                    {"input": "x" * 4097}):
            response = client.post("/v1/audio/speech", json={**body, **bad})
            assert response.status_code == 400, response.text
            assert "error" in response.json()
        assert client.post("/v1/audio/speech", json={**body, "response_format": "wav"}).status_code == 200
    print("Live endpoint checks passed.")


if __name__ == "__main__":
    main()
