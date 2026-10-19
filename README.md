# Qwen3-TTS Speech Server

A local, OpenAI-compatible text-to-speech server built around a **3-model hybrid workflow**. It pairs lightweight CPU narration with GPU-accelerated zero-shot voice design and fast character dialogue cloning, fitting comfortably on GPUs with 4–6 GB VRAM (such as the GTX 1650/1660 Ti and RTX 30/40/50-series).

---

## The 3-Model Workflow

Most voice applications need two very different things: long, cheap narrator reading and distinct, consistent character voices. This server solves that by orchestrating three specialized models behind a single OpenAI-compatible `/v1/audio/speech` endpoint:

```
                            ┌─────────────────────────────────┐
                            │    OpenAI-Compatible Client     │
                            └────────────────┬────────────────┘
                                             │
             ┌───────────────────────────────┴───────────────────────────────┐
             ▼                                                               ▼
   ┌────────────────────┐                                          ┌────────────────────┐
   │    1. Narrator     │                                          │  Character Dialogue│
   │    (Kokoro-82M)    │                                          │    (Qwen3-TTS)     │
   ├────────────────────┤                                          └─────────┬──────────┘
   │ • CPU-resident     │                                                    │
   │ • 28 voice presets │                               ┌────────────────────┴────────────────────┐
   │ • Real-time stream │                               ▼                                         ▼
   │ • Up to 16k chars  │                     ┌────────────────────┐                    ┌────────────────────┐
   └────────────────────┘                     │  2. Voice Designer │                    │  3. Dialogue Engine│
                                              │    (1.7B Design)   │                    │    (0.6B Base)     │
                                              ├────────────────────┤                    ├────────────────────┤
                                              │ • Zero-shot design │                    │ • Fast cloned NPC  │
                                              │ • First dialogue   │                    │ • CUDA graphs      │
                                              │ • Background align │                    │ • Low VRAM footprint│
                                              └────────────────────┘                    └────────────────────┘
```

1. **The Narrator — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (CPU)**
   * Permanently resident on a dedicated CPU worker; runs completely in parallel with GPU tasks without blocking dialogue.
   * Handles long texts (up to 16,384 characters) with sentence-boundary chunking.
   * Includes 28 English voice presets with clean shortcuts (e.g. `george`, `heart`, `alloy`, `echo`).
   * Supports low-latency, real-time 24 kHz mono PCM audio streaming (`stream=true`).

2. **The Voice Designer — [Qwen3-TTS 1.7B VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) (GPU)**
   * Designs brand-new character voices on-demand from plain text descriptions (e.g. `"A gruff old pirate with a gravelly voice"`).
   * Speaks the NPC's introductory line immediately—no sample audio required.
   * Hands off dialogue to the background pipeline to extract bounded reference features for cloning.

3. **The Dialogue Engine — [Qwen3-TTS 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (GPU)**
   * The default resident GPU model, running ultra-fast inference with captured CUDA graphs via [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts).
   * Generates subsequent character dialogue using saved voice profiles (`npc_id`).
   * Only one Qwen model is active in VRAM at a time, switching models dynamically to stay well within low-VRAM budgets.

---

## Quick Setup

### 1. Requirements & Dependencies
* Python 3.11 with [uv](https://astral.sh/uv) package manager.
* NVIDIA GPU with driver supporting CUDA 12.8+ (CPU fallback supported via `--device cpu`).
* FFmpeg on `PATH` (used for Opus, AAC, and tempo changes).

```bash
# Create virtual environment and install locked dependencies
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

### 2. Download Model Weights
Run these commands once to download weights and voices into the local directories expected by the server:

```bash
# Kokoro model and 28 English voice packs (~320 MB)
.venv/bin/hf download hexgrad/Kokoro-82M config.json kokoro-v1_0.pth --local-dir models/kokoro
.venv/bin/hf download hexgrad/Kokoro-82M --include 'voices/[ab]*.pt' --local-dir models/kokoro

# Qwen3-TTS 0.6B Base cloning model (~1.8 GB)
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base

# Whisper alignment checkpoint for background reference processing (~72 MB)
.venv/bin/python download_alignment.py
```

*(Note: Qwen3-TTS 1.7B VoiceDesign (~4.3 GB) automatically downloads to the Hugging Face cache on its first use, or you can pre-fetch it using `.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`).*

---

## Running the Server

Start the server on port 8001:

```bash
OMP_NUM_THREADS=4 .venv/bin/python server.py --device cuda:0 --port 8001
```

* **GPU selection**: Default is `--device cuda:0`. Use `--device cpu` for standard CPU generation.
* **GPU Kokoro**: Add `--kokoro-device cuda:0` to keep the narrator on GPU in FP32. CPU remains the default. Kokoro loads before Qwen so its resident weights count toward Qwen's VRAM plan.
* **Keep all models loaded**: Add `--keep-models-loaded` to preload and retain both Qwen models and their CUDA graphs, avoiding reload/capture delays when switching between voice creation and dialogue. Kokoro stays on its configured device. With `--kokoro-device cuda:0` and the default 4096-token cache, all three use roughly **8–9 GiB reserved VRAM** in the benchmarks; the automatic parallel comparison peaked at **8.46 GiB**. Budget roughly **9 GiB with headroom**; a 12 GB GPU is a comfortable target. Other applications and longer/slower audio can require more memory. The adaptive CPU offloading plan still applies; the default remains one resident Qwen model with swapping.
* **Low-VRAM GPUs (e.g. 4 GB cards)**: Add `--max-seq-len 1024` or `--clone-device cpu`.
* **6 GB starting configuration**: Use `--kokoro-device cuda:0 --max-seq-len 1024`. The planner can move Qwen's speech decoder and text embedding to CPU when needed. A 5 GiB allocator-cap benchmark on an RTX 5060 Ti passed automatic parallel narration, dialogue, model swaps, and a near-240-character Kokoro chunk, peaking at **4.42 GiB reserved**. This leaves room for driver/display memory within a 6 GiB budget; slower speech or other workloads may need more memory. See [benchmark results](AGENTS.md#automatic-parallel-narration).
* **Narration only**: Pass `--design-only` to disable NPC cloning while keeping Kokoro available.
* **API Security**: Set `export TTS_API_KEY="your-secret-key"` to require Bearer token authentication.

Narration automatically runs alongside Qwen on its own worker, whether Kokoro uses CPU or GPU. GPU Kokoro owns a separate CUDA stream; both CPU and GPU Kokoro use a private random generator so narration does not change seeded Qwen output. Each backend still queues its own additional requests, and background NPC preparation shares Qwen's queue. Narration pauses during Qwen model loading and graph capture, then resumes alongside inference. This works with swapping or `--keep-models-loaded` and requires no parallel-mode flag. The memory planner automatically reserves 1 GiB of workspace for GPU narration on the same device, in addition to its resident weights. `/health` reports `kokoro_device`, `keep_models_loaded`, and `shared_cuda_worker=false`.

For ample-VRAM GPUs, keep all three models resident on GPU:

```bash
OMP_NUM_THREADS=4 .venv/bin/python server.py --device cuda:0 --kokoro-device cuda:0 --keep-models-loaded
```

Resident mode removes model-switch loading and graph-capture overhead. Qwen requests remain serial while narration runs alongside them. Startup takes longer because both Qwen models initialize before the server accepts requests. With `--design-only`, the flag retains only VoiceDesign because Base is disabled.

To benchmark the same paragraph, first-chunk latency, simultaneous streamed narration/dialogue, and model swaps on your hardware (requires a ready saved NPC):

```bash
OMP_NUM_THREADS=4 .venv/bin/python benchmark_narrator.py --kokoro-device cpu --output out/bench/kokoro_cpu.json
OMP_NUM_THREADS=4 .venv/bin/python benchmark_narrator.py --kokoro-device cuda:0 --output out/bench/kokoro_gpu.json
OMP_NUM_THREADS=4 .venv/bin/python benchmark_narrator.py --kokoro-device cuda:0 --max-seq-len 1024 --vram-limit-gib 5 --output out/bench/kokoro_gpu_5gib.json
OMP_NUM_THREADS=4 .venv/bin/python benchmark_narrator.py --kokoro-device cuda:0 --keep-models-loaded --output out/bench/all_models_gpu.json
```

Each workload saves its first warmup and three measured runs. Median timings exclude warmup; peak PyTorch allocated and reserved VRAM include it. Timings use the server's workers directly and exclude HTTP overhead; first PCM latency measures synthesized audio readiness. The benchmark also tests a near-240-character narration chunk with VoiceDesign resident. `--max-seq-len 1024` tests a smaller Qwen cache. `--vram-limit-gib 5` caps PyTorch's allocator to leave headroom within a 6 GiB budget; the planner respects this cap. This excludes CUDA context/display memory and does not reproduce a physical 6 GB GPU's speed or memory bandwidth.

A comparison benchmark measures the serial queue against automatic parallel narration with both Qwen models resident:

```bash
.venv/bin/python benchmark_gpu_overlap.py --output out/bench/gpu_overlap.json
```

It preloads all three models, keeps Qwen requests serial, alternates serial/overlap measurement order, and tests simultaneous arrivals plus narration arriving 100 ms after dialogue starts. It validates generated audio and records output hashes, latency, throughput, and VRAM. Both modes use Kokoro's automatic RNG isolation. See the [GPU overlap results](AGENTS.md#experimental-gpu-overlap).

---

## Usage Examples

### 1. Narrator Speech (Kokoro)

Use `model="kokoro"` with any of the 28 English voice presets. Convenient short names are supported without language prefixes:

* **British Voices**: `george` (default narrator), `fable`, `daniel`, `lewis`, `emma`, `alice`, `lily`, `isabella`
* **American Voices**: `heart`, `alloy`, `echo`, `onyx`, `nova`, `bella`, `nicole`, `sarah`, `sky`, `adam`, etc.

#### Complete Audio File (curl)
```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "kokoro",
    "voice": "george",
    "input": "The ancient gates swing open. Beyond them, a forgotten kingdom waits.",
    "response_format": "wav"
  }' \
  --output narration.wav
```

#### Real-Time PCM Streaming (OpenAI Python SDK)
Kokoro can stream raw 24 kHz mono signed 16-bit little-endian PCM chunks during synthesis before the full text finishes:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="local")

with client.audio.speech.with_streaming_response.create(
    model="kokoro",
    voice="george",
    input="The bells began to toll across the valley, warning everyone of the approaching storm.",
    response_format="pcm",
    extra_body={"stream": True},
) as response:
    for chunk in response.iter_bytes():
        # Stream raw PCM audio directly to your audio playback buffer
        pass
```

---

### 2. Character Voices & NPC Dialogue (Qwen3)

#### Step 1: Create a New Character Voice (1.7B VoiceDesign)
Provide a unique `npc_id` and character description in `instructions`. The server generates the introductory line with VoiceDesign and prepares the voice in the background:

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts",
    "npc_id": "guard_01",
    "instructions": "A gruff elderly guard with a low, gravelly voice.",
    "input": "Halt! State your business.",
    "response_format": "wav",
    "max_new_tokens": 128
  }' \
  --output intro.wav
```
The response header includes `X-Voice-Status: preparing`. You can check readiness via `GET /v1/voices/guard_01`.

#### Step 2: Continue Dialogue (0.6B Base Cloning)
For subsequent lines, simply pass `npc_id`. The server reuses the persistent voice with the fast 0.6B Base model:

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts-0.6b",
    "npc_id": "guard_01",
    "input": "You may pass. Keep out of trouble.",
    "response_format": "wav",
    "max_new_tokens": 128
  }' \
  --output dialogue.wav
```

---

### 3. Standalone Voice Design CLI (`demo.py`)

Generate samples directly from the terminal without starting the server:

```bash
.venv/bin/python demo.py \
  -v "A cheerful shopkeeper with an upbeat, welcoming tone" \
  -t "Take a look around! We have the finest wares in the realm."
```

---

### 4. SillyTavern Integration (SillyNPC)

Point SillyTavern's **OpenAI Compatible TTS** provider to `http://127.0.0.1:8001/v1`. 

When used with the **SillyNPC** extension, voice descriptions, NPC IDs, and token budgets are bundled transparently into the `voice` parameter (`sillynpc-v1:<payload>`). The server automatically decodes these requests, routes narration to Kokoro, and preserves character voices across conversations.

---

## API Summary

| Endpoint | Method | Description |
|---|---|---|
| `/v1/audio/speech` | POST | Synthesize speech (OpenAI format). Supports WAV, MP3, FLAC, PCM, Opus, AAC. |
| `/v1/models` | GET | List enabled backends (`kokoro`, `qwen3-tts`, `qwen3-tts-0.6b`). |
| `/v1/voices` | GET | List installed Kokoro voice presets with accent information. |
| `/v1/voices/{npc_id}` | GET | Check preparation status (`ready`, `preparing`, `failed`) and reference audio length for an NPC. |
| `/health` | GET | Server status and active device. |

For short Kokoro narration fragments, optional `context_before` and
`context_after` strings provide sentence context while the response contains
only `input`. The combined text must fit within 240 characters; streaming is
unsupported for contextual requests. Native token timings validate the cut;
unsafe extraction falls back to standalone synthesis. `X-Narration-Context`
reports `applied` or `fallback`.

---

## Technical Details & Architecture

For deep-dive technical documentation—including CUDA graph management, hardware adaptation heuristics (`GemvLinear`), memory offloading plans, background alignment algorithms, and performance benchmarks—see **[AGENTS.md](AGENTS.md)**.

---

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). Downloaded model weights and third-party packages retain their respective licenses.
