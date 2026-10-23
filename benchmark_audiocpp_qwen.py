"""Measure native Qwen or Ming Q8 design/cloning latency and process VRAM.

Uses the existing local server and its normal two-entry LRU cache. Kokoro requests
evict character models between scenarios; no configuration or saved NPC is changed.
Run while other clients are idle. GPU telemetry requires host nvidia-smi access.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time

import httpx
import numpy as np
import soundfile as sf


REFERENCE_TEXT = "The morning light spilled across the garden. Shall we take a walk before the rain arrives?"
STYLE = "A warm, clear female voice with a neutral American accent and a natural conversational pace."
TEXTS = {
    "short": "Welcome back. Your next adventure is waiting just around the corner.",
    "long": "Beyond the quiet village, a narrow path climbed through the forest toward the hills. We followed it slowly, listening to the birds and watching the sunlight move between the trees. By noon, we could see the ocean in the distance.",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:7862")
    parser.add_argument("--server-pid", required=True, type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--suite", choices=("qwen", "ming"), default="qwen")
    parser.add_argument("--reference", type=Path, help="Reuse an exact reference WAV (required for Ming comparisons)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.suite == "ming" and args.reference is None:
        parser.error("--suite ming requires --reference from the Qwen run")
    if args.output is None:
        args.output = Path(f"out/bench/audiocpp_{args.suite}.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = httpx.Client(base_url=args.url, timeout=240)
    samples = []
    stop = threading.Event()
    telemetry_errors = []

    def observe():
        while not stop.is_set():
            result = subprocess.run([
                "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True)
            if result.returncode:
                telemetry_errors.append(result.stderr.strip())
                return
            values = {}
            for line in result.stdout.splitlines():
                pid, gpu, memory = (part.strip() for part in line.split(","))
                if pid == str(args.server_pid):
                    values[gpu] = values.get(gpu, 0) + int(memory)
            rss = None
            try:
                for line in Path(f"/proc/{args.server_pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) / 1024
            except OSError:
                pass
            samples.append({"time": time.time(), "gpu_mib": values, "rss_mib": rss})
            stop.wait(0.1)

    def memory_since(start):
        selected = [sample for sample in samples if sample["time"] >= start]
        gpus = {gpu for sample in selected for gpu in sample["gpu_mib"]}
        return {
            "peak_gpu_mib": {gpu: max(s["gpu_mib"].get(gpu, 0) for s in selected) for gpu in gpus},
            "last_gpu_mib": selected[-1]["gpu_mib"] if selected else {},
            "peak_rss_mib": max((s["rss_mib"] or 0 for s in selected), default=None),
            "sample_count": len(selected),
        }

    records = []
    result = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "url": args.url, "server_pid": args.server_pid, "repeats": args.repeats,
        "suite": args.suite,
        "texts": TEXTS, "reference_text": REFERENCE_TEXT, "style": STYLE,
        "options": {"language": "en", "max_tokens": 512},
        "records": records, "memory_samples": samples, "telemetry_errors": telemetry_errors,
        "notes": [
            "Native Q8 checkpoints; default sampling with seed 1895.",
            "Cold means first request after LRU eviction, not a fresh process or cold OS file cache.",
            "Warm medians exclude the cold request and an additional short warmup.",
            "Each clone request includes the same WAV and exact transcript; includes native reference processing.",
            "Native wall timing excludes model loading; client timing includes loading, HTTP and WAV transfer.",
            "Process VRAM sampled approximately every 100 ms plus nvidia-smi overhead; brief peaks may be missed.",
            "Memory includes CUDA context and all resident native entries; GPU UUIDs distinguish devices.",
        ],
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def run(model, text, phase, *, reference=None, length=None, task=None):
        options = dict(result["options"])
        payload = {"model": model, "input": text, "seed": "1895", "response_format": "wav", "options": options}
        if model.startswith("kokoro"):
            options.clear()
            payload["voice"] = "af_heart"
        if model == "qwen-design-1.7b":
            options["instruct"] = STYLE
        if model == "ming":
            options["template_name"] = "voice_design" if task == "design" else "tts"
            if task == "design":
                options["instruct"] = STYLE
        if reference is not None:
            payload["voice_ref"] = {"type": "base64", "data": base64.b64encode(reference).decode()}
            options["reference_text"] = REFERENCE_TEXT
        start = time.time()
        clock = time.perf_counter()
        response = client.post("/v1/audio/speech", json=payload)
        elapsed = time.perf_counter() - clock
        if response.is_error:
            raise RuntimeError(f"{model}: HTTP {response.status_code}: {response.text[:1000]}")
        audio, rate = sf.read(io.BytesIO(response.content))
        if len(audio) < rate // 2 or not np.isfinite(audio).all() or np.max(np.abs(audio)) < 0.001:
            raise RuntimeError(f"Invalid audio from {model}")
        duration = len(audio) / rate
        native = float(response.headers["X-AudioCPP-Wall-Ms"]) / 1000
        # Allow the monitor to observe memory retained at request completion.
        time.sleep(0.2)
        row = {"model": model, "phase": phase, "length": length,
               "task": task, "options": options,
               "client_seconds": elapsed, "native_seconds": native,
               "audio_seconds": duration, "sample_rate": rate,
               "native_rtf": native / duration, "client_rtf": elapsed / duration,
               **memory_since(start)}
        row["loaded_models"] = [m["id"] for m in client.get("/v1/models").json()["data"] if m["loaded"]]
        records.append(row)
        if phase == "warm":
            audio_path = args.output.parent / f"{args.output.stem}-{model}-{task}-{length}.wav"
            audio_path.write_bytes(response.content)
        save()
        print(json.dumps(row), flush=True)
        return response.content

    def clear_character_models():
        # Both Kokoro entries reside on GPU 0, character models on GPU 1.
        # Two entries replace them under this project's max_loaded_models=2.
        for model in ("kokoro", "kokoro-stream"):
            run(model, "This is a brief narration.", "eviction")
        loaded = [m["id"] for m in client.get("/v1/models").json()["data"] if m["loaded"]]
        if set(loaded) != {"kokoro", "kokoro-stream"}:
            raise RuntimeError(f"Expected two-entry cache; found {loaded}. Keep other clients idle.")

    watcher = threading.Thread(target=observe, daemon=True)
    watcher.start()
    try:
        health = client.get("/health")
        health.raise_for_status()
        result["health"] = health.json()
        result["gpu_info"] = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,driver_version", "--format=csv",
        ], text=True)
        result["config"] = json.loads(Path("audiocpp-server.local.json").read_text())
        native_root = Path("../Ming/audio.cpp")
        if native_root.is_dir():
            result["native_source_revision"] = subprocess.check_output(
                ["git", "-C", str(native_root), "rev-parse", "HEAD"], text=True).strip()
        result["checkpoint_gib"] = {
            m["id"]: Path(m["path"]).stat().st_size / 1024**3
            for m in result["config"]["models"] if m["id"].startswith("qwen") or m["id"] == "ming"
        }
        reference = args.reference.read_bytes() if args.reference else run("qwen-design-1.7b", REFERENCE_TEXT, "reference")
        import hashlib
        result["reference_sha256"] = hashlib.sha256(reference).hexdigest()
        args.output.with_suffix(".reference.wav").write_bytes(reference)
        cases = ([("ming", "design"), ("ming", "clone")] if args.suite == "ming" else
                 [("qwen-design-1.7b", "design"), ("qwen-base-0.6b", "clone"), ("qwen-base-1.7b", "clone")])
        for model, task in cases:
            clear_character_models()
            ref = reference if task == "clone" else None
            run(model, TEXTS["short"], "cold", reference=ref, length="short", task=task)
            run(model, TEXTS["short"], "warmup", reference=ref, length="short", task=task)
            for length, text in TEXTS.items():
                for _ in range(args.repeats):
                    run(model, text, "warm", reference=ref, length=length, task=task)
        for clone in (("qwen-base-0.6b", "qwen-base-1.7b") if args.suite == "qwen" else ()):
            run("qwen-design-1.7b", TEXTS["short"], "pair_load")
            run(clone, TEXTS["short"], "pair", reference=reference)
        result["summary"] = []
        for model, task in cases:
            for length in TEXTS:
                rows = [r for r in records if r["model"] == model and r["task"] == task and r["length"] == length and r["phase"] == "warm"]
                result["summary"].append({"model": model, "task": task, "length": length, **{
                    key: statistics.median(r[key] for r in rows)
                    for key in ("client_seconds", "native_seconds", "audio_seconds", "native_rtf", "client_rtf")}})
        save()
    finally:
        stop.set()
        watcher.join()
        client.close()
        save()


if __name__ == "__main__":
    main()
