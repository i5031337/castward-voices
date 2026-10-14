"""A/B benchmark: serial queue versus automatic parallel GPU narration.

Does not change the server's worker policy. Measures inference/PCM readiness,
excluding HTTP overhead. Qwen remains serial; both models preload before overlap.
"""
import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

from server import SpeechRequest, create_app, encode_audio
from smoke_narrator import TEXT


async def benchmark(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with GPU access")
    device = args.device
    app = create_app(device=device, kokoro_device=device, keep_models_loaded=True,
                     max_seq_len=args.max_seq_len)
    narration = SpeechRequest(model="kokoro", voice="george", input=TEXT, response_format="pcm")
    dialogue = SpeechRequest(model="qwen3-tts", input="Hello, traveler. You are welcome here.",
                             seed=42, max_new_tokens=128)
    description = "A warm, clear middle-aged male narrator with a calm, deep voice."
    records = []
    async with app.router.lifespan_context(app):
        if app.state.engine.status(args.npc_id) != "ready":
            raise ValueError("--npc-id must name a ready saved voice")
        torch.cuda.synchronize(device)
        loop = asyncio.get_running_loop()
        stream = app.state.narrator.stream

        def audio_info(wav, sr):
            wav = np.asarray(wav)
            if not wav.size or not np.isfinite(wav).all() or np.abs(wav).max() <= 0.001:
                raise RuntimeError("Invalid or silent benchmark audio")
            return len(wav) / sr, hashlib.sha256(wav.tobytes()).hexdigest()

        def generate_qwen(body):
            wavs, sr = app.state.engine.generate(body, description)
            return (*audio_info(wavs[0], sr), torch.cuda.current_stream(device).cuda_stream)

        async def qwen(body):
            started = time.perf_counter()
            duration, digest, cuda_stream = await loop.run_in_executor(app.state.executor, generate_qwen, body)
            elapsed = time.perf_counter() - started
            return {"seconds": elapsed, "audio_seconds": duration, "rtf": elapsed / duration,
                    "audio_sha256": digest, "cuda_stream": cuda_stream}

        async def narrate(worker):
            started = time.perf_counter()
            iterator = app.state.narrator.chunks(narration)
            first, duration = None, 0
            def next_chunk():
                wav = next(iterator, None)
                if wav is None:
                    return None
                pcm = encode_audio(wav, 24000, "pcm", 1)
                return len(pcm) / (24000 * 2), audio_info(wav, 24000)
            try:
                while True:
                    chunk = await loop.run_in_executor(worker, next_chunk)
                    if chunk is None:
                        break
                    if first is None:
                        first = time.perf_counter() - started
                    duration += chunk[0]
            finally:
                await loop.run_in_executor(worker, iterator.close)
            return {"seconds": time.perf_counter() - started,
                    "audio_seconds": duration, "first_pcm_seconds": first}

        with torch.cuda.device(device):
            overlap_worker = app.state.narrator_executor
            for kind in ("clone", "design"):
                body = dialogue.model_copy(update={"npc_id": args.npc_id if kind == "clone" else None})
                # Warm both backends/streams before measuring paired workloads.
                await qwen(body)
                await narrate(app.state.executor)
                await narrate(overlap_worker)
                for scenario in ("together", "dialogue_inflight"):
                    for run in range(args.runs + 1):
                        # Alternate order to reduce clock/temperature drift bias.
                        modes = ("serial", "overlap") if run % 2 == 0 else ("overlap", "serial")
                        for mode in modes:
                            torch.cuda.synchronize(device)
                            torch.cuda.reset_peak_memory_stats(device)
                            started = time.perf_counter()
                            worker = app.state.executor if mode == "serial" else overlap_worker
                            if scenario == "together":
                                narration_result, qwen_result = await asyncio.gather(narrate(worker), qwen(body))
                            else:
                                task = asyncio.create_task(qwen(body))
                                await asyncio.sleep(0.1)
                                narration_result = await narrate(worker)
                                qwen_result = await task
                            torch.cuda.synchronize(device)
                            elapsed = time.perf_counter() - started
                            row = {"kind": kind, "scenario": scenario, "mode": mode, "run": run,
                                   "warmup": run == 0, "wall_seconds": elapsed,
                                   "audio_seconds_per_wall_second":
                                       (narration_result["audio_seconds"] + qwen_result["audio_seconds"]) / elapsed,
                                   "narrator": narration_result, "qwen": qwen_result,
                                   "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                                   "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30}
                            records.append(row)
                            print(json.dumps(row), flush=True)
                print(f"Completed {kind} comparison", flush=True)

        summaries = []
        for kind in ("clone", "design"):
            for scenario in ("together", "dialogue_inflight"):
                for mode in ("serial", "overlap"):
                    rows = [r for r in records if (r["kind"], r["scenario"], r["mode"]) ==
                            (kind, scenario, mode) and not r["warmup"]]
                    summaries.append({"kind": kind, "scenario": scenario, "mode": mode,
                        "median_wall_seconds": statistics.median(r["wall_seconds"] for r in rows),
                        "median_qwen_seconds": statistics.median(r["qwen"]["seconds"] for r in rows),
                        "median_qwen_rtf": statistics.median(r["qwen"]["rtf"] for r in rows),
                        "median_narrator_seconds": statistics.median(r["narrator"]["seconds"] for r in rows),
                        "median_first_pcm_seconds": statistics.median(r["narrator"]["first_pcm_seconds"] for r in rows),
                        "median_audio_throughput": statistics.median(r["audio_seconds_per_wall_second"] for r in rows),
                        "qwen_audio_seconds": sorted(set(r["qwen"]["audio_seconds"] for r in rows)),
                        "qwen_unique_audio_hashes": len(set(r["qwen"]["audio_sha256"] for r in rows)),
                        "peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
                        "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows)})
        comparisons = []
        for kind in ("clone", "design"):
            for scenario in ("together", "dialogue_inflight"):
                pairs = [{r["mode"]: r for r in records if r["kind"] == kind and
                          r["scenario"] == scenario and r["run"] == run}
                         for run in range(1, args.runs + 1)]
                comparisons.append({"kind": kind, "scenario": scenario,
                    "median_overlap_to_serial_wall_ratio": statistics.median(
                        p["overlap"]["wall_seconds"] / p["serial"]["wall_seconds"] for p in pairs),
                    "median_overlap_to_serial_qwen_ratio": statistics.median(
                        p["overlap"]["qwen"]["seconds"] / p["serial"]["qwen"]["seconds"] for p in pairs),
                    "identical_qwen_audio_pairs": sum(p["serial"]["qwen"]["audio_sha256"] ==
                        p["overlap"]["qwen"]["audio_sha256"] for p in pairs)})
        output = {"gpu": torch.cuda.get_device_name(device), "device": device,
                  "max_seq_len": args.max_seq_len, "runs": args.runs,
                  "threads": torch.get_num_threads(), "keep_models_loaded": True,
                  "isolated_kokoro_rng": True,
                  "kokoro_stream": stream.cuda_stream,
                  "comparisons": comparisons, "summaries": summaries, "measurements": records}
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(json.dumps(summaries, indent=2), flush=True)
        print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--npc-id", default="benchmark-guard-001")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--output", default="out/bench/gpu_overlap.json")
    args = parser.parse_args()
    if args.runs < 1 or args.max_seq_len < 128 or not args.device.startswith("cuda"):
        parser.error("requires CUDA, positive runs, and --max-seq-len >= 128")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
