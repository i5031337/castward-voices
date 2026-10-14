# System Architecture & Technical Specifications (AGENTS.md)

This document provides technical reference for developers, maintainers, and AI coding agents working on the `qwen3-tts` engine.

---

## 1. The 3-Model Hybrid Architecture

The server unifies three models into a single OpenAI-compatible HTTP service, each specialized for its performance and cost characteristics:

| Model | Hardware | Primary Role | Lifecycle |
|---|---|---|---|
| **Kokoro-82M** | CPU by default; optional GPU | Long-form narrator speech & real-time PCM streaming | Stays permanently resident; CUDA mode shares Qwen's worker |
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
* **CPU Worker (`narrator`, max_workers=1)**: Dedicated to Kokoro G2P phonemization and neural inference. Long narration requests run completely in parallel with GPU NPC dialogue without contention.

`--kokoro-device cuda:0` moves Kokoro to GPU in FP32. CUDA Kokoro and any CUDA Qwen device use the same serial `tts` worker, preventing Kokoro kernels/allocations from interfering with global CUDA graph capture during model swaps. PCM streams submit one chunk at a time so queued Qwen requests can execute between chunks. Non-streaming narration occupies the shared worker through generation and encoding. CPU Kokoro keeps the independent worker. A process-wide `MODEL_LOAD_LOCK` protects Kokoro inference from the temporary default-dtype changes made by Transformers during Qwen loading; loaded-model CPU/GPU inference still runs in parallel.

Kokoro loads before Qwen so its resident GPU weights are accounted for by the adaptive VRAM planner. The planner also respects PyTorch's per-process allocator memory cap.

### Dynamic VRAM Model Swapping (Default)
To keep peak VRAM under 4–6 GiB, the server holds only one Qwen model in GPU memory at a time:
1. **Startup**: Loads `0.6B Base` and captures CUDA graphs.
2. **New NPC (`instructions` provided)**: Drops `0.6B Base`, invokes garbage collection, frees PyTorch CUDA cache, and loads `1.7B VoiceDesign` to generate the initial dialogue line.
3. **Return to Dialogue**: When subsequent requests arrive for saved NPCs, the server swaps back to `0.6B Base`, restoring the CUDA-graph decoding pipeline.

### Optional Persistent Residency

`--keep-models-loaded` preloads both Qwen models and captures their independent CUDA graphs during startup. `SpeechEngine.loaded_models` retains each pipeline; `get(kind)` selects the cached pipeline and its device without unloading, garbage collection, or graph recapture. Base remains the initial active model. Kokoro is permanently resident on its configured CPU/GPU device, so adding `--kokoro-device cuda:0` puts all three on GPU. The second Qwen loader sees the first model's live memory when computing its adaptive placement plan. Qwen inference remains serial, and CUDA Kokoro continues to share that worker. With cloning disabled, only VoiceDesign is preloaded. Default behavior still swaps Qwen models.

Measured on an RTX 5060 Ti with all three on GPU and a 4096-token cache: **7.75 GiB peak allocated / 8.28 GiB peak reserved**, including startup, mixed narration/dialogue, switching between Qwen models, and a near-240-character Kokoro chunk. Allow approximately **9 GiB with headroom**; a 12 GB GPU is a comfortable target. These figures exclude driver/display and other processes; longer audio, slower speech, other voices, and larger caches can increase memory use. CPU offloading remains adaptive on smaller devices. Reproduce with `benchmark_narrator.py --kokoro-device cuda:0 --keep-models-loaded --output out/bench/all_models_gpu.json`.

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
