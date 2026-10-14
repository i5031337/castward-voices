"""Benchmark Kokoro latency/VRAM and queued or parallel Qwen workloads.

Uses the server's real lifespan and workers, without HTTP/network overhead.
Streaming latency measures the first synthesized PCM chunk, not socket delivery.
"""
import argparse
import asyncio
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
    gpu = args.device.startswith("cuda") or args.kokoro_device.startswith("cuda")
    cuda_device = args.device if args.device.startswith("cuda") else args.kokoro_device
    if gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run with GPU access")
    if args.vram_limit_gib:
        if not gpu:
            raise ValueError("--vram-limit-gib requires CUDA")
        total = torch.cuda.get_device_properties(cuda_device).total_memory
        torch.cuda.set_per_process_memory_fraction(args.vram_limit_gib * 2**30 / total, cuda_device)

    def synchronize():
        if gpu:
            torch.cuda.synchronize(cuda_device)

    def memory():
        return {"peak_allocated_gib": torch.cuda.max_memory_allocated(cuda_device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(cuda_device) / 2**30} if gpu else {}

    narration = SpeechRequest(model="kokoro", voice="george", input=TEXT, response_format="pcm")
    long_text = " ".join(TEXT.replace(".", "").split())[:240].rsplit(" ", 1)[0]
    long_narration = narration.model_copy(update={"input": long_text})
    clone = SpeechRequest(model="qwen3-tts-0.6b", npc_id=args.npc_id,
                          input="Hello, traveler. You are welcome here.", seed=42, max_new_tokens=128)
    design = SpeechRequest(model="qwen3-tts", input=clone.input, seed=42, max_new_tokens=128)
    description = "A warm, clear middle-aged male narrator with a calm, deep voice."
    app = create_app(device=args.device, kokoro_device=args.kokoro_device,
                     max_seq_len=args.max_seq_len, keep_models_loaded=args.keep_models_loaded)
    started = time.perf_counter()
    async with app.router.lifespan_context(app):
        load_seconds = time.perf_counter() - started
        if app.state.engine.status(args.npc_id) != "ready":
            raise ValueError("Create and prepare --npc-id through the server before benchmarking")
        synchronize()
        startup_memory = memory()
        loop = asyncio.get_running_loop()

        def validate(wav, sr):
            wav = np.asarray(wav)
            if not wav.size or not np.isfinite(wav).all() or np.abs(wav).max() <= 0.001:
                raise RuntimeError("Invalid or silent benchmark audio")
            return len(wav) / sr

        async def narrate(stream=False, body=narration):
            started = time.perf_counter()
            if stream:
                iterator = app.state.narrator.chunks(body)
                durations, first = [], None
                def next_chunk():
                    wav = next(iterator, None)
                    if wav is None:
                        return None
                    return encode_audio(wav, 24000, "pcm", 1), validate(wav, 24000)
                try:
                    while True:
                        chunk = await loop.run_in_executor(app.state.narrator_executor, next_chunk)
                        if chunk is None:
                            break
                        if first is None:
                            first = time.perf_counter() - started
                        durations.append(chunk[1])
                finally:
                    await loop.run_in_executor(app.state.narrator_executor, iterator.close)
                duration = sum(durations)
            else:
                wav, sr = await loop.run_in_executor(app.state.narrator_executor,
                                                    app.state.narrator.generate, body)
                duration, first = validate(wav, sr), None
            elapsed = time.perf_counter() - started
            return {"generation_seconds": elapsed, "audio_seconds": duration,
                    "rtf": elapsed / duration, "first_pcm_seconds": first}

        async def qwen(body):
            started = time.perf_counter()
            wavs, sr = await loop.run_in_executor(app.state.executor, app.state.engine.generate,
                                                 body, description)
            duration = validate(wavs[0], sr)
            elapsed = time.perf_counter() - started
            return {"generation_seconds": elapsed, "audio_seconds": duration, "rtf": elapsed / duration}

        async def mixed(body):
            # Submit streamed narration and dialogue on their independent workers.
            narrator_result, qwen_result = await asyncio.gather(narrate(True), qwen(body))
            return {"narrator": narrator_result, "qwen": qwen_result}

        async def measure(label, operation):
            records = []
            warmup = None
            for index in range(args.runs + 1):
                synchronize()
                if gpu:
                    torch.cuda.reset_peak_memory_stats(cuda_device)
                started = time.perf_counter()
                result = await operation()
                synchronize()
                row = {"run": index, "wall_seconds": time.perf_counter() - started,
                       **result, **memory()}
                print(json.dumps({"workload": label, **row}), flush=True)
                if index:
                    records.append(row)
                else:
                    warmup = row
            summary = {"warmup": warmup, "measurements": records,
                       "qwen_plan": str(app.state.engine.plan),
                       "median_wall_seconds": statistics.median(r["wall_seconds"] for r in records)}
            for key in ("generation_seconds", "rtf", "first_pcm_seconds"):
                if key in records[0] and records[0][key] is not None:
                    summary["median_" + key] = statistics.median(r[key] for r in records)
            if gpu:
                summary.update({key: max(r[key] for r in [warmup, *records])
                                for key in ("peak_allocated_gib", "peak_reserved_gib")})
            return summary

        results = {}
        results["narration"] = await measure("narration", narrate)
        results["stream"] = await measure("stream", lambda: narrate(True))
        results["clone"] = await measure("clone", lambda: qwen(clone))
        results["mixed_clone"] = await measure("mixed_clone", lambda: mixed(clone))
        if not args.skip_design:
            results["swap_to_design"] = await measure("swap_to_design", lambda: mixed(design))
            results["design"] = await measure("design", lambda: qwen(design))
            results["mixed_design"] = await measure("mixed_design", lambda: mixed(design))
            results["long_chunk_with_design_resident"] = await measure(
                "long_chunk_with_design_resident", lambda: narrate(body=long_narration))
            # Exercise the load/capture guard while narration is scheduled.
            results["swap_back"] = await measure("swap_back", lambda: mixed(clone))
        summary = {"device": args.device, "kokoro_device": args.kokoro_device,
                   "gpu": torch.cuda.get_device_name(cuda_device) if gpu else None,
                   "threads": torch.get_num_threads(), "max_seq_len": args.max_seq_len,
                   "vram_limit_gib": args.vram_limit_gib,
                   "keep_models_loaded": args.keep_models_loaded,
                   "shared_cuda_worker": app.state.shared_cuda_worker,
                   "load_seconds": load_seconds, "startup_memory": startup_memory,
                   "results": results}
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kokoro-device", default="cpu")
    parser.add_argument("--npc-id", default="benchmark-guard-001")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--skip-design", action="store_true")
    parser.add_argument("--keep-models-loaded", action="store_true")
    parser.add_argument("--vram-limit-gib", type=float,
                        help="PyTorch allocator cap; excludes driver/display memory, not a real smaller GPU")
    parser.add_argument("--output", default="out/bench/narrator.json")
    args = parser.parse_args()
    if args.runs < 1 or (args.vram_limit_gib is not None and args.vram_limit_gib <= 0):
        parser.error("runs and VRAM limit must be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
