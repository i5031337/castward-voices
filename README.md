# Castward Voices

<p><img src="webui/branding/castward-voices-logo.svg" width="640" alt="Castward Voices"></p>

The local voice companion for the **Castward SillyTavern RPG extension**: persistent, designed character voices for short dialogue, with lightweight Kokoro narration for the longer story. **Ming and Qwen** design and clone character voices through audio.cpp, behind an OpenAI-compatible speech API. **Castward Voice Studio**, the web UI at `/`, provides voice design, cloning, and preset synthesis. Ming is the default character backend; Kokoro handles narration.

```bash
./run-local.sh --port 8001
```

Open <http://127.0.0.1:8001/>. See [setup, configuration, and Castward Voice Studio](AUDIOCPP.md) for dependencies and model profiles.

**For a 4 GB GPU with Ming, or a 6 GB GPU with Qwen, keep narration on CPU:**

```bash
./run-local.sh --kokoro-cpu --port 8001
```

This starts CPU Kokoro and CUDA character synthesis on **GPU 0**, with **one resident GPU model**. Qwen automatically swaps between design and cloning checkpoints. Use `--gpu-device 1` to select another GPU. Budget about **3 GiB available for Ming or 5 GiB for Qwen**, leaving the rest for your desktop; these are workload-based recommendations rather than physical 4/6 GB card benchmarks.

## Current performance: Ming vs Qwen

Measured with **Q8 models on an RTX 5060 Ti**, CUDA, four CPU threads, and three measured runs per task/passage. All cloning tests used the **same 4.72-second reference recording and transcript**. Design used the same voice description. The passages contain 68 and 229 characters; generation settings use native defaults with seed 1895 and `max_tokens=512`.

| Model / task | Short generation | Paragraph generation | RTF, short / paragraph ↓ | VRAM, short → paragraph |
|---|---:|---:|---:|---:|
| **Ming 0.5B — design** | 0.91 s | 3.28 s | 0.259 / 0.256 | **1.48 → 1.52 GiB** |
| **Ming 0.5B — clone** | 0.95 s | 2.95 s | 0.272 / 0.254 | **2.05 → 2.08 GiB** |
| Qwen 1.7B VoiceDesign | 0.70 s | 2.99 s | 0.179 / 0.192 | 2.59 → 3.77 GiB |
| Qwen 0.6B Base — clone | **0.64 s** | **2.19 s** | **0.159 / 0.154** | 2.21 → 3.20 GiB |
| Qwen 1.7B Base — clone | 0.81 s | 2.53 s | 0.196 / 0.189 | 3.07 → 3.96 GiB |

**Ming uses less VRAM and one checkpoint for both design and cloning. Qwen generates audio faster in these tests.** RTF is generation time divided by output audio duration; lower is faster. Outputs vary by model: short lines lasted 3.5–4.2 seconds and paragraphs 11.6–15.6 seconds. These are warm native generation times; cold requests after eviction took about **2 seconds for Ming and 2.4–4 seconds for Qwen**. Quality was not ranked.

VRAM is the native server's sampled process allocation on the synthesis GPU, including context and retained buffers, **excluding Kokoro on the other GPU**. It is not model-file size or total card usage. [Full methodology and results](AUDIOCPP.md#native-ming-comparison-and-server-vram-budget-2026-10-22) include native sample rates, loading overhead, memory scope, and reproduction commands.

## Recommended minimum available VRAM

**Use `--kokoro-cpu` for smaller GPUs: 3 GiB available for Ming or 5 GiB for Qwen.** The default GPU-narration workflow needs 6 GiB available for Ming + Kokoro. “Available” means the budget left for this server after the desktop and other applications, rather than the card's advertised capacity.

| Intended use on one GPU | Recommended available VRAM |
|---|---:|
| Ming design/cloning + CPU Kokoro (`--kokoro-cpu`) | **3 GiB** |
| Qwen design/cloning + CPU Kokoro, one GPU checkpoint resident (`--kokoro-cpu`) | **5 GiB** |
| Default Ming design/cloning + GPU Kokoro narration | **6 GiB** |
| One Qwen model + GPU Kokoro narration | **8 GiB** |
| Qwen VoiceDesign + 1.7B Base cached together | **9 GiB** |

These are planning recommendations with headroom, not tested minimum card sizes. Concurrent Ming cloning and paragraph narration measured **4.81 GiB combined across two GPUs**: 2.08 GiB for Ming and 2.73 GiB for Kokoro. Qwen + Kokoro's 8 GiB recommendation combines separately observed allocations; resident Qwen pairs reached 6.36 GiB with mixed short/long buffers and can grow further. Longer outputs and different workloads can require more.

Without `--kokoro-cpu`, the supplied configuration uses **GPU 1 for Ming/Qwen and GPU 0 for Kokoro**. For that split, budget at least **3 GiB available on GPU 1 for Ming and 4 GiB on GPU 0 for Kokoro**; Qwen needs more on GPU 1. For GPU narration on a single GPU, change every `device` in `audiocpp-server.local.json` to its index. The default two-entry cache counts all models across devices and can evict models as you switch; memory on different cards cannot be pooled for one model. CPU-narration mode instead gives the GPU process a one-entry cache and keeps Kokoro in a separate CPU process.

**Kokoro is usable on CPU.** Audio.cpp Q8 on a Ryzen 5800X3D with four threads generated 13.53 seconds of narration in **4.88 seconds** (RTF **0.36**, about **2.8× faster than playback**), with no GPU synthesis allocation. Streaming with `options: {"text_chunk_size": 80}` delivered first PCM in **0.68 seconds**; the GPU mode's default 240-character budget delivered this whole paragraph on CPU in one event after **4.86 seconds**. These are three-run warm medians. `--kokoro-cpu` supplies the 80-character streaming default unless your profile already sets one. See [CPU measurements and deployment](AUDIOCPP.md#kokoro-on-cpu-2026-10-22).

## Historical Python runtime

The architecture, commands, flags, and measurements below describe the previous Python synthesis runtime. Its helpers remain available for historical experiments; use [AUDIOCPP.md](AUDIOCPP.md) for the current server.

## Historical Python Runtime: The 3-Model Workflow

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

The current server uses **audio.cpp** for all synthesis. Python provides the OpenAI-compatible API and Castward Voice Studio web UI. The workflow above describes the previous Python runtime.

For a fresh Python environment:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

Install FFmpeg separately for Opus, AAC, and tempo changes. The native backend needs a built `audiocpp_server` and GGUF model files; the previous Python model weights are not used by this server. See [the audio.cpp guide](AUDIOCPP.md#run-locally) for native configuration and optional CPU word alignment.

## Running the Server

From the repository directory, run:

```bash
./run-local.sh --port 8001
```

Then open **Castward Voice Studio** at <http://127.0.0.1:8001/>. The launcher starts audio.cpp on port 7862 if needed, then starts the compatibility API on port 8001. It stops only the processes it started when you exit.

The supplied local configuration expects the native binary at `../Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server`, with Ming and Qwen on GPU 1 and Kokoro on GPU 0. Edit model paths and devices in `audiocpp-server.local.json` to match your machine. Public model profiles and the native service URL are configured in `backend.json`. Launcher overrides and further details are in [AUDIOCPP.md](AUDIOCPP.md#run-locally); native startup logs are written to `audiocpp-backend.log`.

The default host is `127.0.0.1`. To serve trusted devices on your local network, add `--host 0.0.0.0`. Set `TTS_API_KEY` to require bearer authentication.

If audio.cpp is already running separately at the URL configured in `backend.json`, start just the API with:

```bash
.venv/bin/python server.py --config backend.json --port 8001
```

GPU selection, model residency, and native threads are configured in the audio.cpp JSON file. The old `--device`, `--kokoro-device`, `--keep-models-loaded`, and `--max-seq-len` server flags are no longer supported.

## Historical Python Runtime Benchmarks

The following benchmarks use the previous Python synthesis helpers and require their legacy dependencies; they do not benchmark the current audio.cpp server.

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

### 4. SillyTavern Integration (ST-Castward)

Point SillyTavern's **OpenAI Compatible TTS** provider to `http://127.0.0.1:8001/v1`. 

When used with the **CASTWARD** extension, voice descriptions, NPC IDs, and token budgets are bundled transparently into the `voice` parameter (`sillynpc-v1:<payload>`). The server automatically decodes these requests, routes narration to Kokoro, and preserves character voices across conversations.

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
reports `applied` or `fallback`. The audiobook example selects context
automatically and records its use in `narration_context.json`.

---

## Audiobook Example

The [audiobook prototype](examples/audiobook/README.md) uses a local llama-server
to cast characters and attribute dialogue in *Beyond Lies the Wub*, then renders
an editable manifest with Kokoro narration and cloned Qwen character voices.
It supports a single-scene trial, full-story rendering, and reusable audio caches.

---

## Technical Details & Architecture

For deep-dive technical documentation—including CUDA graph management, hardware adaptation heuristics (`GemvLinear`), memory offloading plans, background alignment algorithms, and performance benchmarks—see **[AGENTS.md](AGENTS.md)**.

---

## Tests

Run the maintained unit tests with:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

---

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). Downloaded model weights and third-party packages retain their respective licenses.
