# System Architecture & Technical Specifications (AGENTS.md)

This document provides technical reference for developers, maintainers, and AI coding agents working on the `qwen3-tts` engine.

---

## 1. The 3-Model Hybrid Architecture

The server unifies three models into a single OpenAI-compatible HTTP service, each specialized for its performance and cost characteristics:

| Model | Hardware | Primary Role | Lifecycle |
|---|---|---|---|
| **Kokoro-82M** | CPU | Long-form narrator speech & real-time PCM streaming | Stays permanently resident on dedicated CPU thread |
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

### Dynamic VRAM Model Swapping
To keep peak VRAM under 4–6 GiB, the server holds only one Qwen model in GPU memory at a time:
1. **Startup**: Loads `0.6B Base` and captures CUDA graphs.
2. **New NPC (`instructions` provided)**: Drops `0.6B Base`, invokes garbage collection, frees PyTorch CUDA cache, and loads `1.7B VoiceDesign` to generate the initial dialogue line.
3. **Return to Dialogue**: When subsequent requests arrive for saved NPCs, the server swaps back to `0.6B Base`, restoring the CUDA-graph decoding pipeline.

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

* **G2P & Accents**: Initializes distinct `KPipeline` instances for American (`'a'`) and British (`'b'`) English phonemization, sharing a single CPU `KModel`.
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
