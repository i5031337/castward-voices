"""Measure real CPU narration, streamed first audio, and simultaneous NPC speech."""
import argparse
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

TEXT = (
    "Welcome, traveler. The road ahead is long, but there is always a place for you by the fire. "
    "The northern gate closes at sunset, so you should leave before the bells begin to ring. "
    "If you meet the old keeper near the bridge, tell him that I sent you. "
    "He knows the mountain paths better than anyone in this town. "
    "Take some bread and water for the journey, and keep your lantern close at hand."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--npc-id", default="benchmark-guard-001", help="Previously prepared NPC for concurrent test")
    args = parser.parse_args()
    output = Path("out/narrator")
    output.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {os.environ['TTS_API_KEY']}"} if os.environ.get("TTS_API_KEY") else {}
    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=180,
                      trust_env=False, headers=headers) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert "kokoro" in [m["id"] for m in client.get("/v1/models").json()["data"]]
        assert "bm_george" in [v["id"] for v in client.get("/v1/voices").json()["data"]]
        measurements = []
        body = {"model": "kokoro", "voice": "narrator", "input": TEXT, "response_format": "wav"}
        # First run includes phonemizer/model warmup; later runs measure steady state.
        for index in range(3):
            started = time.perf_counter()
            response = client.post("/v1/audio/speech", json=body)
            response.raise_for_status()
            elapsed = time.perf_counter() - started
            audio, sr = sf.read(io.BytesIO(response.content))
            assert sr == 24000 and np.isfinite(audio).all() and np.abs(audio).max() > 0.001
            (output / f"paragraph_{index}.wav").write_bytes(response.content)
            measurements.append({"run": index, "audio_seconds": len(audio) / sr,
                                 "elapsed_seconds": elapsed, "rtf": elapsed / (len(audio) / sr)})
            print(json.dumps(measurements[-1]), flush=True)
        started = time.perf_counter()
        first_audio, chunks = None, []
        with client.stream("POST", "/v1/audio/speech", json={**body, "stream": True, "response_format": "pcm"}) as response:
            response.raise_for_status()
            assert response.headers["x-audio-sample-rate"] == "24000"
            for data in response.iter_bytes():
                if data:
                    if first_audio is None:
                        first_audio = time.perf_counter() - started
                    chunks.append(data)
        streamed = np.frombuffer(b"".join(chunks), dtype="<i2") / 32768
        assert np.abs(streamed).max() > 0.001
        assert abs(len(streamed) / 24000 - measurements[-1]["audio_seconds"]) < 0.1
        elapsed = time.perf_counter() - started
        assert first_audio is not None and first_audio < elapsed
        stream_measurement = {"stream_first_audio_seconds": first_audio, "stream_total_seconds": elapsed,
                              "audio_seconds": len(streamed) / 24000}
        print(json.dumps(stream_measurement), flush=True)
        sf.write(output / "paragraph_streamed.wav", streamed, 24000)
        # Native Kokoro speed applies once, including for non-streaming responses.
        fast = client.post("/v1/audio/speech", json={**body, "speed": 2})
        fast.raise_for_status()
        faster, _ = sf.read(io.BytesIO(fast.content))
        assert 0.4 < len(faster) / len(audio) < 0.65
        for fmt in ("mp3", "flac", "opus", "aac"):
            encoded = client.post("/v1/audio/speech", json={**body, "input": "The story continues.", "response_format": fmt})
            encoded.raise_for_status()
            assert encoded.content
            (output / f"sample.{fmt}").write_bytes(encoded.content)
        concurrent = None
        if client.get(f"/v1/voices/{args.npc_id}").status_code == 200:
            with ThreadPoolExecutor(max_workers=2) as executor:
                started = time.perf_counter()
                npc = executor.submit(client.post, "/v1/audio/speech", json={"model": "qwen3-tts-0.6b",
                    "npc_id": args.npc_id, "input": "Hello, traveler. You are welcome here.",
                    "response_format": "wav", "max_new_tokens": 128})
                narration = executor.submit(client.post, "/v1/audio/speech", json=body)
                npc_response = npc.result()
                npc_response.raise_for_status()
                npc_elapsed = time.perf_counter() - started
                narration_response = narration.result()
                narration_response.raise_for_status()
                concurrent = {"concurrent_npc_seconds": npc_elapsed,
                              "concurrent_total_seconds": time.perf_counter() - started}
                print(json.dumps(concurrent), flush=True)
        # Ending a stream early must not prevent the next request from completing.
        with client.stream("POST", "/v1/audio/speech", json={**body, "stream": True, "response_format": "pcm"}) as response:
            response.raise_for_status()
            next(response.iter_bytes())
        retry = client.post("/v1/audio/speech", json={**body, "input": "The connection has recovered."})
        retry.raise_for_status()
        (output / "timings.json").write_text(json.dumps({"runs": measurements, "stream": stream_measurement,
                                                        "concurrent": concurrent}, indent=2) + "\n")
    print("Narrator generation, streaming, formats, speed, and disconnect checks passed.")


if __name__ == "__main__":
    main()
