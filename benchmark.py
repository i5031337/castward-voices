"""Measure warmed-up design or saved-NPC cloning speed and allocated VRAM."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import fast_tts
from npc_tts import LOCAL_CLONE_MODEL, SpeechEngine
from server import SpeechRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kind", choices=["clone", "design"], default="clone")
    parser.add_argument("--npc-id", default="benchmark-guard-001")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()
    output = Path("out/bench")
    output.mkdir(parents=True, exist_ok=True)
    gpu = args.device.startswith("cuda")
    started = time.perf_counter()
    engine = SpeechEngine(loader=fast_tts.load, device=args.device, design_model=None,
                          clone_model=str(LOCAL_CLONE_MODEL) if args.kind == "clone" else None,
                          max_seq_len=args.max_seq_len, voices_dir="voices")
    load_seconds = time.perf_counter() - started
    body = SpeechRequest(model="qwen3-tts", input="Hello, traveler. You are welcome here.",
                         npc_id=args.npc_id if args.kind == "clone" else None, max_new_tokens=128, seed=42)
    if args.kind == "clone" and not (engine.voice_path(args.npc_id) / "metadata.json").is_file():
        parser.error("Create the NPC through the server first, so this measures cloning only.")
    description = "A warm, clear middle-aged male narrator with a calm, deep voice."
    measurements = []
    for index in range(args.runs + 1):
        if gpu:
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        wavs, sr = engine.generate(body, description)
        if gpu:
            torch.cuda.synchronize(args.device)
        seconds = time.perf_counter() - started
        wav = np.asarray(wavs[0])
        assert len(wav) > 0 and np.isfinite(wav).all() and np.abs(wav).max() > 0.001
        record = {"run": index, "audio_seconds": len(wav) / sr, "generation_seconds": seconds,
                  "rtf": seconds / (len(wav) / sr),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(args.device) / 2**30 if gpu else 0}
        print(json.dumps(record), flush=True)
        if index:  # Exclude first generation and its one-time kernel setup.
            measurements.append(record)
        sf.write(output / f"{args.kind}_{args.device.replace(':', '_')}_{index}.wav", wav, sr)
    summary = {"kind": args.kind, "device": args.device, "max_seq_len": args.max_seq_len,
               "load_seconds": load_seconds, "measurements": measurements,
               "median_rtf": statistics.median(row["rtf"] for row in measurements),
               "median_generation_seconds": statistics.median(row["generation_seconds"] for row in measurements),
               "peak_allocated_gib": max(row["peak_allocated_gib"] for row in measurements)}
    path = output / f"{args.kind}_{args.device.replace(':', '_')}.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
