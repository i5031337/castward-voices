"""Exercise persistent NPC creation and repeated cloning against the live server."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--npc-id", default="benchmark-guard-001")
    parser.add_argument("--case", choices=["short", "suitable", "long"], default="short")
    parser.add_argument("--language", default="English")
    args = parser.parse_args()
    output = Path("out/npc")
    output.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=f"http://127.0.0.1:{args.port}", timeout=180, trust_env=False) as client:
        texts = {
            "short": "Hello, traveler. You are welcome here.",
            "suitable": "Welcome, traveler. The road ahead is long, but there is always a place for you by the fire.",
            "long": ("Welcome, traveler. The road ahead is long, but there is always a place for you by the fire. "
                     "The northern gate closes at sunset, so you should leave before the bells begin to ring. "
                     "If you meet the old keeper near the bridge, tell him that I sent you. "
                     "He knows the mountain paths better than anyone in this town. "
                     "Take some bread and water for the journey, and keep your lantern close at hand."),
        }
        body = {"model": "qwen3-tts", "npc_id": args.npc_id,
                "instructions": "A warm, clear middle-aged male narrator with a calm, deep voice.",
                "input": texts[args.case], "response_format": "wav",
                "language": args.language,
                "max_new_tokens": 768 if args.case == "long" else 256, "seed": 42}
        timings = []
        record = Path("voices") / hashlib.sha256(args.npc_id.encode()).hexdigest()
        reference_hash = None
        new_voice = client.get(f"/v1/voices/{args.npc_id}").status_code == 404
        for index in range(4):
            started = time.perf_counter()
            response = client.post("/v1/audio/speech", json=body)
            response.raise_for_status()
            elapsed = time.perf_counter() - started
            audio, sr = sf.read(io.BytesIO(response.content))
            assert sr == 24000 and len(audio) > 0 and np.isfinite(audio).all()
            assert np.abs(audio).max() > 0.001
            (output / f"{args.npc_id}_{index}.wav").write_bytes(response.content)
            measurement = {"request": index, "audio_seconds": len(audio) / sr,
                           "elapsed_seconds": elapsed, "rtf": elapsed / (len(audio) / sr),
                           "voice_status_at_response": response.headers.get("x-voice-status")}
            print(json.dumps(measurement), flush=True)
            if index == 0 and new_voice:
                assert response.headers.get("x-voice-status") == "preparing"
                stored, _ = sf.read(record / "dialogue.wav")
                np.testing.assert_array_equal(stored, audio)
            deadline = time.perf_counter() + 180
            while True:
                status = client.get(f"/v1/voices/{args.npc_id}").json()
                assert status["status"] != "failed", status
                if status["status"] == "ready":
                    break
                assert time.perf_counter() < deadline, "Reference preparation timed out"
                time.sleep(0.25)
            measurement["ready_seconds_from_request"] = time.perf_counter() - started
            metadata = json.loads((record / "metadata.json").read_text())
            if new_voice:
                assert 3 <= metadata["reference_seconds"] <= 10
                if args.case == "long":
                    assert metadata["reference_source"] == "aligned_dialogue", metadata
                    assert metadata["ref_text"] in texts["long"]
                    if index == 0:
                        assert metadata["reference_seconds"] < measurement["audio_seconds"]
            current_hash = hashlib.sha256((record / "reference.wav").read_bytes()).hexdigest()
            if reference_hash is not None:
                assert current_hash == reference_hash, "Original voice reference changed"
            reference_hash = current_hash
            timings.append(measurement)
            print(json.dumps({"prepared_reference_seconds": status["reference_seconds"],
                              "reference_source": metadata.get("reference_source"),
                              "ready_seconds_from_request": measurement["ready_seconds_from_request"]}), flush=True)
            body.pop("instructions", None)  # Existing NPCs only need their ID.
            body["model"] = "qwen3-tts-0.6b"
            body["input"] = "Hello, traveler. You are welcome here."
            body["max_new_tokens"] = 128
        assert (record / "prompt.npz").is_file()
        assert json.loads((record / "metadata.json").read_text())["npc_id"] == args.npc_id
        bad = client.post("/v1/audio/speech", json={**body, "npc_id": "not-created-yet"})
        assert bad.status_code == 400
        (output / f"{args.npc_id}_timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    print("NPC voice creation, reuse, and stored-reference checks passed.")


if __name__ == "__main__":
    main()
