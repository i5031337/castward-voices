# System Architecture & Technical Specifications (AGENTS.md)

This document provides technical reference for developers, maintainers, and AI coding agents working on the `qwen3-tts` engine.

---

## 1. The 3-Model Hybrid Architecture

The server unifies three models into a single OpenAI-compatible HTTP service, each specialized for its performance and cost characteristics:

| Model | Hardware | Primary Role | Lifecycle |
|---|---|---|---|
| **Kokoro-82M** | CPU by default; optional GPU | Long-form narrator speech & real-time PCM streaming | Stays permanently resident on its own worker |
| **Qwen3-TTS 1.7B VoiceDesign** | GPU (or CPU) | Zero-shot NPC creation from natural language text | Loaded on-demand when creating new character voices |
| **Qwen3-TTS 0.6B Base** | GPU (or CPU) | High-speed dialogue generation using cloned NPC voices | Default resident Qwen model; executes via CUDA graphs |

```
                              ┌──────────────────────────────────┐
                              │     Client Request (/speech)     │
                              └─────────────────┬────────────────┘
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     ┌───────────────────────┐                                     ┌───────────────────────┐
     │   model == "kokoro"   │                                     │     model == Qwen     │
     └───────────┬───────────┘                                     └───────────┬───────────┘
                 │                                                             │
                 ▼                                                             ▼
     ┌───────────────────────┐                                     ┌───────────────────────┐
     │  Narrator Executor    │                                     │   Qwen GPU Executor   │
     │  (Dedicated CPU Thrd) │                                     │  (Single-Thrd Serial) │
     └───────────┬───────────┘                                     └───────────┬───────────┘
                 │                                                             │
        ┌────────┴────────┐                                           ┌────────┴────────┐
        ▼                 ▼                                           ▼                 ▼
 ┌─────────────┐   ┌─────────────┐                             ┌─────────────┐   ┌─────────────┐
 │ Full Audio  │   │  Real-Time  │                             │  New Voice  │   │ Existing NPC│
 │ (WAV/MP3/…) │   │  PCM Stream │                             │ (1.7B Des.) │   │ (0.6B Base) │
 └─────────────┘   └─────────────┘                             └─────────────┘   └─────────────┘
```

### Concurrency & Thread Isolation
CUDA graphs and static decoding KV caches cannot be accessed concurrently. The server enforces strict thread isolation using two separate `ThreadPoolExecutor` workers:
* **GPU Worker (`tts`, max_workers=1)**: Serializes all Qwen model loads, CUDA graph warmups, and voice design/cloning inferences.
* **Narrator Worker (`narrator`, max_workers=1)**: Dedicated to Kokoro G2P and CPU/GPU neural inference. Narration runs alongside Qwen automatically; GPU inference uses a private CUDA stream and can contend for GPU resources.

`--kokoro-device cuda:0` moves Kokoro to GPU in FP32. CPU and GPU narration always use an independent worker, with no parallel-mode option. A process-wide `MODEL_LOAD_LOCK` protects the entire Qwen model transition (unload, garbage collection, load, and CUDA graph capture) against Kokoro inference. Kokoro holds this lock for each inference chunk, CPU transfer, and GPU stream drain, including error paths; it releases the lock before yielding audio. Thus no narration CUDA work remains in flight when graph capture begins, and Transformers' temporary default-dtype changes cannot break CPU narration. Ordinary Qwen inference does not hold this lock, so both backends run concurrently with either swapping or resident Qwen models.

`NarratorRandomMode` intercepts Kokoro's `rand`, `randn`, and `randn_like` noise operations on its inference thread, using a private CPU/CUDA `torch.Generator`. It never swaps or reseeds the global RNG, preserving seeded Qwen output during overlap. GPU Kokoro owns a private stream that waits for model initialization before use and synchronizes at chunk boundaries before the transition lock releases.

Kokoro loads before Qwen so its resident GPU weights are accounted for by the adaptive VRAM planner. On the same GPU, the server passes an additional 1 GiB Kokoro workspace reserve into Qwen's plan, allowing CPU offloading when overlapping inference would otherwise exceed the budget. This is planner headroom, not a preallocated tensor. The planner also respects PyTorch's per-process allocator memory cap.

### Dynamic VRAM Model Swapping (Default)
To keep peak VRAM under 4–6 GiB, the server holds only one Qwen model in GPU memory at a time:
1. **Startup**: Loads `0.6B Base` and captures CUDA graphs.
2. **New NPC (`instructions` provided)**: Drops `0.6B Base`, invokes garbage collection, frees PyTorch CUDA cache, and loads `1.7B VoiceDesign` to generate the initial dialogue line.
3. **Return to Dialogue**: When subsequent requests arrive for saved NPCs, the server swaps back to `0.6B Base`, restoring the CUDA-graph decoding pipeline.

### Optional Persistent Residency

`--keep-models-loaded` preloads both Qwen models and captures their independent CUDA graphs during startup. `SpeechEngine.loaded_models` retains each pipeline; `get(kind)` selects the cached pipeline and its device without unloading, garbage collection, or graph recapture. Base remains the initial active model. Kokoro is permanently resident on its configured CPU/GPU device, so adding `--kokoro-device cuda:0` puts all three on GPU. The second Qwen loader sees the first model's live memory when computing its adaptive placement plan. Qwen inference remains serial while Kokoro runs alongside it on its own worker. With cloning disabled, only VoiceDesign is preloaded. Default behavior still swaps Qwen models.

The initial serial GPU measurement on an RTX 5060 Ti with all three on GPU and a 4096-token cache peaked at **7.75 GiB allocated / 8.28 GiB reserved**, including startup, mixed narration/dialogue, Qwen switching, and a near-240-character Kokoro chunk. The production automatic parallel comparison peaked at **8.46 GiB reserved**. Allow approximately **9 GiB with headroom**; a 12 GB GPU is a comfortable target. These figures exclude driver/display and other processes; longer audio, slower speech, other voices, and larger caches can increase memory use. CPU offloading remains adaptive on smaller devices. Reproduce full workloads with `benchmark_narrator.py --kokoro-device cuda:0 --keep-models-loaded --output out/bench/all_models_gpu.json`.

---

## 2. Hardware Adaptation (`fast_tts.py`)

The loader inspects GPU capabilities and free VRAM to compute an optimal execution `Plan` before capturing CUDA graphs.

### Execution Plan Heuristics
* **Dtype Selection**:
  * Compute capability $\ge 8.0$ (Ampere / Ada / Blackwell, e.g. RTX 30xx, 40xx, 50xx): Uses native `torch.bfloat16`.
  * Compute capability $< 8.0$ (Turing, GTX 16-series, Volta): Uses `torch.float16`.
* **`GemvLinear` Optimization**:
  On GPUs without tensor cores (e.g. GTX 1650 / 1660 Ti), standard cuBLAS FP16 GEMM kernels fall back to pathologically slow paths (~25 GB/s memory bandwidth). `GemvLinear` transposes weights contiguously at load time to trigger fast GEMV kernels (~130 GB/s). It microbenchmarks a representative linear layer on startup to verify if pre-transposition provides $>20\%$ speedup.
* **Text Embedding Offloading (`OffloadedEmbedding`)**:
  The 311M-parameter embedding table is only indexed during prompt prefill. Moving it to CPU saves ~0.6 GiB VRAM with unmeasurable impact on latency.
* **Speech Decoder Placement**:
  The speech tokenizer's convolutional decoder overflows to NaN in standard FP16. When VRAM permits, it runs on GPU in `bfloat16` or `float32`. If VRAM is constrained, it offloads to CPU in `float32`.
* **Windows WDDM Slack**:
  Under Windows WDDM, `cudaMemGetInfo` underreports available VRAM before driver shared memory spill occurs. A 0.4 GiB slack buffer is added on Windows hosts (`os.name == "nt"`).

---

## 3. Persistent NPC Voice Lifecycle (`npc_tts.py`)

Character voice identities are created zero-shot and cloned without manual audio recording.

```
       First Request: npc_id + instructions
                        │
                        ▼
            Generate 1.7B VoiceDesign Audio
                        │
                        ├──────────────────────────┐
                        ▼                          ▼
            Return First Audio to Client   Background Task
            (X-Voice-Status: preparing)            │
                                                   ▼
                                        Align Dialogue (Whisper tiny)
                                        Target: 3–10s clean audio
                                                   │
                                        Extract Acoustic Codes & Embed
                                                   │
                                        Save prompt.npz & metadata.json
                                        (X-Voice-Status: ready)
```

### Storage Directory Structure
Each NPC record is persisted under `voices/<sha256(npc_id)>/`:
* `dialogue.wav`: Raw uncompressed 24 kHz WAV of the first spoken utterance.
* `reference.wav`: Bounded 3–10s reference audio extracted for cloning.
* `metadata.json`: Stores `status`, `npc_id`, `description`, `seed`, exact transcript, and source.
* `prompt.npz`: NumPy archive containing reference speech codes and speaker embedding (serialized securely without Python pickle).

### Alignment & Fallback Reference Logic (`reference_audio.py`)
1. **Suitable First Line (3–10s)**: Reused directly as the reference clip without modification.
2. **Long First Line (>10s)**: Aligned against the exact spoken transcript using local Whisper `tiny` on CPU. The selector extracts complete sentences fitting within the 3–10s window.
3. **Short or Ambiguous First Line (<3s or uncertain alignment)**: Uses the initial utterance's speaker embedding to synthesize an extended reference clip on CPU/GPU (`extend_reference`). This preserves the identical acoustic identity without redesigning a new voice.

### State Transitions & Crash Recovery
* **States**: `missing` $\rightarrow$ `preparing` $\rightarrow$ `ready` (or `failed`).
* **Interrupted Jobs**: Server lifespan checks `pending_ids()` on startup and re-queues any records left in the `preparing` state.
* **Failure Retries**: If extraction fails, the original dialogue audio remains preserved on disk; subsequent speech requests automatically retry preparation.

---

## 4. Kokoro Narrator Backend (`narrator_tts.py`)

* **G2P & Accents**: Initializes distinct `KPipeline` instances for American (`'a'`) and British (`'b'`) English phonemization, sharing a single `KModel` on the configured CPU/GPU device. G2P remains on CPU.
* **Local Pack Loading**: Tensor packs are loaded directly into memory (`self.packs`) on initialization, preventing Hugging Face Hub downloads during inference.
* **Voice Aliases & Shortcuts**: Maps short names (`george`, `heart`, `alloy`, `echo`, etc.) to full voice IDs (`bm_george`, `af_heart`, etc.). Dynamically derives shortcuts from all installed `[ab][fm]_[a-z]+.pt` voice packs.
* **Text Chunking**: `text_chunks()` splits text preferentially on sentence punctuation (`[.!?]`), sub-splitting words only if individual sentences exceed 240 characters.
* **PCM Streaming Protocol**: Yields raw 24 kHz mono signed 16-bit little-endian PCM (`response_format=pcm`, `stream=True`). Disconnections gracefully terminate generator iterators to release CPU inference resources.

---

## 5. Audio Encoding Pipeline (`server.py`)

* **Direct Libsndfile Encoding**: Clean WAV, RAW PCM, FLAC, and MP3 are generated in-memory via `soundfile`.
* **FFmpeg Subprocess Transcoding**:
  * Required for Opus (`-c:a libopus -f ogg`) and AAC (`-c:a aac -f adts`).
  * Used for Qwen speed modifications: chained `atempo` filters preserve pitch across $0.25\times$ to $4.0\times$ playback.
  * Kokoro handles speed natively during G2P/synthesis; FFmpeg is bypassed for Kokoro WAV/MP3/FLAC/PCM.

---

## 6. SillyNPC Bridge Protocol (`sillynpc_bridge.py`)

When interfacing with SillyTavern through standard OpenAI-compatible proxies that only pass the `voice` parameter, the client encodes metadata into a URL-safe base64 payload:
```
sillynpc-v1:<base64url_json>
```
Decoded JSON structure:
```json
{
  "npc_id": "guard_83a9f2",
  "instructions": "A gruff elderly guard with a low, gravelly voice",
  "max_new_tokens": 128
}
```
The server transparently intercepts and unwraps this format prior to request validation.

---

## 7. Performance Benchmarks

### RTX 5060 Ti GPU (CUDA graphs, 4096-token sequence budget)
| Model | Audio Duration | Median Generation Time | RTF (Compute / Audio) | Peak VRAM |
|---|---:|---:|---:|---:|
| **1.7B VoiceDesign** | 2.40 s | 1.29 s | 0.536 | 4.46 GiB |
| **0.6B Base (Cloned NPC)** | 2.72 s | 1.32 s | 0.487 | 2.78 GiB |

### CPU Narration (Ryzen 5800X3D, 4 Threads)
* **Steady State RTF**: ~0.22 (27.5 s audio generated in ~6.1 s).
* **PCM Stream First Audio**: Delivered within ~0.48 s.
* **Concurrent Generation**: GPU NPC dialogue generates in 1.85 s while CPU Kokoro narration streams concurrently.

### Optional GPU Kokoro (RTX 5060 Ti, FP32; 2026-10-13)

Measured using `benchmark_narrator.py`, four CPU threads, 4096-token Qwen cache, three warmed runs. Narration is the same 27.5-second paragraph in both modes. These timings exclude HTTP overhead; PCM latency measures the first synthesized chunk.

This initial comparison used the shared GPU worker. Current automatic parallel measurements are recorded below.

| Workload | CPU Kokoro | GPU Kokoro |
|---|---:|---:|
| Full paragraph generation | 5.22 s | 0.319 s |
| First PCM chunk | 0.412 s | 0.037 s |
| Combined streamed paragraph + cloned dialogue | ~5.5 s | 1.66 s |
| Base + Kokoro peak reserved VRAM | 3.12 GiB | 3.71 GiB |
| VoiceDesign + Kokoro peak reserved VRAM | 4.73 GiB | 5.38 GiB |

GPU Kokoro is approximately 16 times faster for this paragraph. Its incremental warmed GPU footprint is about 0.54 GiB resident and 0.66 GiB during narration with Base resident. Allocated peaks are 3.32 GiB with Base and 5.07 GiB during mixed VoiceDesign/narration; reserved memory is the more conservative planning figure. CUDA context, desktop applications, longer utterances, slower speech speeds, and other voices can increase the actual budget. The benchmark hardware has 16 GB VRAM; these figures do not prove every workload fits on a physical 6 GB card.

A reduced-budget run with `--max-seq-len 1024 --vram-limit-gib 5` passed both model swaps while streaming, plus a near-240-character unpunctuated chunk (14.675 seconds of audio) with VoiceDesign resident. Peak reserved memory across all workloads, including warmup, was **4.96 GiB**. The adaptive plan kept VoiceDesign's text embedding on GPU and moved its speech decoder to CPU FP32. Kokoro's warmed paragraph median remained **0.329 s**; combined narration/cloned-dialogue median was **1.27 s**. This caps only PyTorch allocations on a 16 GB RTX 5060 Ti and leaves nominal headroom within a 6 GiB budget; it does not emulate the throughput of a physical 6 GB card.

Raw local results: `out/bench/kokoro_cpu.json`, `out/bench/kokoro_gpu.json`, and `out/bench/kokoro_gpu_5gib.json`.

### Experimental GPU Overlap

The initial `benchmark_gpu_overlap.py` experiment tested resident Kokoro on a dedicated single-thread worker/CUDA stream alongside serial Qwen on the default CUDA stream. All models and graphs loaded before testing. The initial experiment did not change production scheduling; automatic parallel narration has since been implemented as described below. The current comparison script uses the production narrator worker/stream and RNG isolation, with no isolation flag.

Measured on the RTX 5060 Ti with a 4096-token cache and four CPU threads. Each pair generates the same 27.5-second narrated paragraph plus a short NPC line, using either Base or VoiceDesign. The script alternates measurement order to reduce drift and excludes warmup from latency medians. Seven measured pairs per scenario used normal shared RNG; five pairs per scenario used a private Kokoro CUDA random generator. This controlled run preserves normal Qwen sampling and produced identical Qwen waveform hashes in all 20 serial/overlap pairs. Timings exclude HTTP overhead.

| Controlled scenario | Serial total | Overlap total | Serial dialogue | Overlap dialogue | Serial first PCM | Overlap first PCM |
|---|---:|---:|---:|---:|---:|---:|
| Base, simultaneous arrivals | 1.689 s | 1.514 s | 1.382 s | 1.514 s | 46 ms | 66 ms |
| Base, narration 100 ms after dialogue starts | 1.687 s | 1.489 s | 1.333 s | 1.489 s | 1279 ms | 49 ms |
| VoiceDesign, simultaneous arrivals | 1.647 s | 1.467 s | 1.339 s | 1.467 s | 45 ms | 66 ms |
| VoiceDesign, narration 100 ms after dialogue starts | 1.649 s | 1.453 s | 1.291 s | 1.453 s | 1237 ms | 47 ms |

Across paired runs, overlap reduced total completion time by **10–12%**, increased audio throughput by approximately **11–14%**, and increased dialogue latency by **9–12%**. Narration completed in about **0.39–0.42 s** rather than waiting roughly **1.55–1.69 s** for the shared worker. First narration audio improves dramatically when Qwen is already running, but becomes slower for simultaneous arrivals due to contention. The combined controlled A/B run peaked at **8.57 GiB reserved**; caches from both modes persist in the shared allocator, so this is a combined high-water mark rather than a measurement of extra parallel-mode memory.

The normal shared-RNG experiment changed seeded Qwen output under overlap: Base produced different durations (2.56–3.44 s versus a stable serial 2.72 s), and VoiceDesign output changed too. Kokoro's random excitation consumed the same default CUDA generator used by Qwen sampling. The controlled experiment isolated Kokoro's CUDA `rand`/`randn_like` calls using a private generator. This isolation is now automatic in production through `NarratorRandomMode`, on CPU as well as GPU. The Qwen backend retains its upstream device-wide `torch.cuda.synchronize()` calls, which limit overlap; neither the benchmark nor the server patches those.

These measurements motivated automatic parallel narration with independent RNG and exclusion around model transitions. Qwen requests still cannot share one graph/cache concurrently. Only one GPU, one narrator voice, a fixed paragraph, and a short dialogue line were tested. Historical experiment results: `out/bench/gpu_overlap.json` and `out/bench/gpu_overlap_isolated_rng.json`.

### Automatic Parallel Narration

Production CPU/GPU narration now runs alongside Qwen without an additional option. Both resident and swapping modes use the same independent narrator worker. GPU stream draining and the model-transition lock pause narration around loading/capture, while private RNG preserves seeded Qwen generation.

The production GPU swapping benchmark completed streamed narration during both model transitions, full/PCM narration, Base and VoiceDesign inference, and a near-240-character narration chunk. The unrestricted 4096-token run peaked at **5.89 GiB reserved**; use a smaller cache/CPU offloading for 6 GiB devices. With a **5 GiB allocator cap and 1024-token cache**, the automatic overlap workspace budget selected CPU VoiceDesign decoder and text embedding and all workloads passed at **4.42 GiB peak reserved**. This is a capped RTX 5060 Ti run, not a physical 6 GB GPU performance measurement. Results: `out/bench/parallel_swapping_gpu.json` and `out/bench/parallel_gpu_5gib.json`.

The production resident-model A/B comparison completed three measured pairs for each model/arrival scenario, with identical Qwen waveform hashes in all **12 pairs**, using the actual narrator worker, private CUDA stream, and private RNG. It peaked at **8.46 GiB reserved**. Results: `out/bench/automatic_gpu_overlap.json`.
