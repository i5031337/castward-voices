# qwen3-tts

Voice design with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign):
describe a voice in plain text, get speech in that voice, with no reference audio needed. Uses
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (CUDA-graph decoding) with a loader
that adapts to the GPU, down to 4 GB cards like the GTX 1650 Ti.

## Setup

The fast path requires an NVIDIA GPU (Turing / GTX 16-series or newer) with a driver supporting CUDA 12.8 (R570+).
CPU generation is also available with `--device cpu`, using the standard Qwen3-TTS implementation.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```

`requirements.lock.txt` records the exact versions installed and tested on this Linux machine.
To reproduce them, use `uv pip install --python .venv/bin/python -r requirements.lock.txt`.

If `model/model.safetensors` exists, the loader uses the local `model/` directory automatically.
Otherwise the model (~4.3 GB) downloads to the Hugging Face cache on first run. For offline machines, pre-fetch with
`hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` and set `HF_HUB_OFFLINE=1`.

## Usage

```bash
.venv/bin/python demo.py          # Windows: .venv/Scripts/python
.venv/bin/python demo.py -v "A gruff old pirate, gravelly and slow" -t "Arr, the tide waits for no one."
```

Options: `-l` language (English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish,
Italian, Auto), `--seed`, `--temperature`, `--max-seq-len` (prompt + audio frames at 12/s; lower is faster).
Output goes to `out/`.

If other applications leave too little free VRAM for this model, run on CPU (slower):

```bash
OMP_NUM_THREADS=4 .venv/bin/python demo.py --device cpu --max-new-tokens 80 --seed 42 \
  -v "A warm, clear narrator" -t "Hello. The speech model is ready."
```

With enough VRAM free (allow at least 4 GiB for short clips with CPU offloading), select
the RTX 5060 Ti with `--device cuda:0`:

```bash
.venv/bin/python demo.py --device cuda:0 --max-seq-len 256 --seed 42 \
  -v "A warm, clear narrator" -t "Hello. The speech model is ready."
```

Verified on the RTX 5060 Ti with all components on GPU in bfloat16: a short sample
generated 2.2 seconds of audio in 1.1 seconds, with 4.02 GiB peak allocated VRAM.
Model loading and CUDA graph warmup took 2.3 seconds. The CPU test generated 2.48 seconds
of audio in 14.5 seconds with four threads. These are single short samples, not a full benchmark.

Use `--model /path/to/model` to choose another local model directory or a Hugging Face model ID.
`--max-new-tokens` limits generated audio frames; raise it for longer speech. `--max-seq-len` applies to CUDA only.
Python 3.11 is used for the local virtual environment; the default system Python 3.14 is not used.

From Python:

```python
import fast_tts
tts = fast_tts.load()
wavs, sr = tts.generate_voice_design(text="Hello there.", instruct="A warm elderly woman", language="English")
```

## OpenAI-compatible speech endpoint

Start the server (port 8001 avoids another service already using 8000 on this machine):

```bash
OMP_NUM_THREADS=4 .venv/bin/python server.py --device cuda:0 --port 8001
```

The server loads the 0.6B Base model and captures CUDA graphs at startup. It switches to
VoiceDesign when a request needs a new voice, then switches back to the smaller model for NPC
dialogue. Only one model remains loaded at a time. Requests queue on one generation thread
so the GPU's shared caches cannot overlap. The server binds to localhost by default.
Use `--device cpu` for CPU generation, or `--model /path/to/model` for another model directory.
The default CUDA sequence budget is 4096; adjust with `--max-seq-len`.
Use `--design-only` to run the original VoiceDesign endpoint without the smaller model.

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","input":"Hello. The speech model is ready.","voice":"alloy","instructions":"A warm, clear narrator","response_format":"wav","seed":42,"max_new_tokens":80}' \
  --output speech.wav
```

An OpenAI Python SDK client can use `base_url="http://127.0.0.1:8001/v1"`:

```python
from openai import OpenAI  # install the client separately: pip install openai

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="local")
with client.audio.speech.with_streaming_response.create(
    model="qwen3-tts",
    voice="alloy",
    input="Hello. The speech model is ready.",
    instructions="A warm, clear narrator",
    response_format="wav",
    extra_body={"seed": 42, "max_new_tokens": 80},
) as response:
    response.stream_to_file("speech.wav")
```

The request follows the [OpenAI speech API format](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create),
with these local behaviors:

- `POST /v1/audio/speech` returns binary audio; `GET /v1/models` lists `qwen3-tts`.
- Formats: WAV, MP3 (the API default), FLAC, PCM, Opus, and AAC. PCM is raw little-endian
  16-bit mono at 24 kHz. Opus, AAC, and speed changes require `ffmpeg`, installed on this machine.
- `instructions` supplies the complete voice description. Without it, `voice` selects a local
  description preset, or can itself be a free-form description. Names such as `alloy` are
  convenience presets, not replicas of OpenAI voices or fixed character identities.
- `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` are accepted as aliases for the local Qwen model.
- `speed` supports 0.25–4.0 through pitch-preserving audio processing.
- Qwen extensions: `language`, `seed`, `temperature`, and `max_new_tokens` (default 2048 frames).
  Input and instructions each accept up to 4096 characters. On CUDA, prompt tokens plus
  `max_new_tokens` must fit within `--max-seq-len`; requests exceeding this return HTTP 400.
- Audio is returned after generation finishes. The SDK's streaming response helper works for
  downloading it, but this server does not yet stream audio during generation or support SSE.
- Optional authentication: set `TTS_API_KEY` before starting the server and send that key as a
  Bearer token. Without it, no key is required; SDK clients can use a placeholder such as `local`.

`GET /health` reports readiness after model loading. Interactive API documentation is at `/docs`.
Run request, encoding, authentication, and concurrency tests with
`.venv/bin/python -m unittest discover -s tests -v`.
With a GPU server running on port 8001, `.venv/bin/python tests/smoke_endpoint.py` verifies real
HTTP generation in all six formats, speed changes, and request validation.

## Persistent NPC voices

The local cloning model is [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).
Download it into the directory expected by the server:

```bash
.venv/bin/hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/0.6B-Base
.venv/bin/python download_alignment.py
```

Send the NPC's hash in `npc_id` and its voice description in `instructions`. The first request
speaks the requested dialogue directly with 1.7B VoiceDesign and returns it immediately after
generation. After sending the audio, the server prepares a reference and clone features in the
background for subsequent requests using 0.6B Base.

The reference stays between 3 and 10 seconds. Suitable first dialogue is reused intact. Longer
dialogue is aligned against its exact text using a local Whisper tiny model on CPU; the server
selects complete sentences and saves only their matching transcript. `download_alignment.py`
fetches that 72 MiB checkpoint once, enabling offline alignment. For very short dialogue, a
sentence longer than the reference budget, uncertain alignment, or a potentially truncated
generation, the server uses the original speaker embedding to synthesize a bounded reference
in the background. This preserves the original voice as the cloning source; it does not design
another voice. An absent alignment checkpoint also uses this fallback.

The first audio response includes `X-Voice-Status: preparing`; subsequent responses report
`ready`. Poll `GET /v1/voices/<npc_id>` for `status`, `reference_seconds`, and any preparation
error. Follow-up speech requests wait for pending preparation on the same generation worker.
The first dialogue still waits for its full generation; audio generation itself is not streamed.

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts","npc_id":"guard_83a9f2","instructions":"A gruff elderly guard with a low, gravelly voice","input":"Halt! State your business.","response_format":"wav","max_new_tokens":128}' \
  --output guard.wav
```

Subsequent requests need only the same `npc_id` and new dialogue:

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tts-0.6b","npc_id":"guard_83a9f2","input":"You may pass. Keep out of trouble.","response_format":"wav","max_new_tokens":128}' \
  --output guard_next.wav
```

For an OpenAI SDK client, add `extra_body={"npc_id": npc_hash, "max_new_tokens": 128}` to the
speech call. Both `qwen3-tts` and `qwen3-tts-0.6b` support this NPC lifecycle.
The Base model needs a reference voice; `qwen3-tts-0.6b` rejects requests without `npc_id`.
An unknown NPC requires a description in `instructions`. IDs accept 1–128 letters, digits,
underscores, or hyphens.

Each NPC's record is stored in `voices/<sha256-of-npc-id>/`:

- `dialogue.wav`: the original first dialogue before output-format or playback-speed conversion.
- `reference.wav`: the bounded reference, preserved across later dialogue requests.
- `metadata.json`: preparation status, NPC ID, original description, exact dialogue/reference
  transcripts, seed, reference source, and model identity.
- `prompt.npz`: reference speech codes and speaker embedding, saved without Python pickle.

Records survive server restarts, and saved clone features skip repeated reference extraction.
Requests for the same new NPC are serialized and produce one reference. Changed instructions
on an existing NPC do not replace its voice; use a new versioned ID to create a different voice.
The first dialogue and pending metadata are persisted before returning audio. Clone features
are published before marking the record ready. If background preparation fails, the already
returned audio remains valid and the status becomes `failed`; a later speech request retries
preparation from the saved original dialogue. Server startup resumes records left `preparing`.
Previously saved voices remain supported and do not require regeneration.
The in-memory prompt cache holds up to 128 NPCs; all records remain on disk.
Models, voices, virtual environments, and output audio are excluded from Git.

Use `--clone-model /path/to/0.6B-Base` or `--voices-dir /path/to/voices` to change storage paths.
`--device cpu` also supports saved NPC voices with the standard Qwen implementation.
`--design-only` disables the NPC workflow.

Measured on the RTX 5060 Ti, with a 4096-token sequence budget, the same dialogue text,
and three warmed-up runs per model:

| Model | Audio duration | Median generation time | Time per second of audio | Peak allocated VRAM |
|---|---:|---:|---:|---:|
| 1.7B VoiceDesign | 2.40 s | 1.29 s | 0.536 s | 4.46 GiB |
| 0.6B Base, saved NPC | 2.72 s | 1.32 s | 0.487 s | 2.78 GiB |

This short test used about 38% less allocated VRAM and about 9% less generation time per second
of audio for the clone model. It compares design against cloning, so it is not a controlled
model-quality comparison. The previous synchronous workflow took about 19 seconds for first
NPC creation, including reference generation, model swaps, and initial feature extraction.
With background preparation, a new NPC's 2.40-second first line returned in 3.64 seconds;
its 4.72-second extended reference was ready at 7.82 seconds from the request. Follow-up lines
took about 1.21 seconds. A longer first dialogue returned 24.64 seconds of audio in 14.94 seconds,
then its 7.92-second aligned reference was ready at 18.57 seconds. These measurements include
HTTP handling and model switches on this machine; voice descriptions and text affect timing.

`.venv/bin/python tests/smoke_npc.py --npc-id new-test-id --case short` tests live creation,
background readiness, and reference reuse. Use `--case suitable` or `--case long` to exercise
other reference lengths, with a fresh ID for each creation test.
Reproduce the GPU measurements with:

```bash
OMP_NUM_THREADS=4 .venv/bin/python benchmark.py --kind clone --device cuda:0
OMP_NUM_THREADS=4 .venv/bin/python benchmark.py --kind design --device cuda:0
```

The benchmark uses `benchmark-guard-001`, created by `smoke_npc.py`; override with `--npc-id`.
Audio and measurement JSON files are written to `out/bench/`.
CPU cloning with four threads also passed: one warmed-up run generated 2.56 seconds of audio
in 12.61 seconds (4.93 seconds of compute per second of audio). GPU cloning is the recommended
default on this machine.

## Hardware adaptation

`fast_tts.load()` picks a plan at load time and prints it (also on `tts.plan`):

| Choice | Rule |
|---|---|
| dtype | bf16 on compute capability ≥ 8.0 (RTX 30xx+), else fp16 |
| `gemv_linear` | microbenchmark: pre-transposed GEMV linears, enabled where cuBLAS half-precision GEMMs are slow (GPUs without tensor cores) |
| decoder | on GPU if VRAM allows (bf16 or fp32), else CPU fp32; fp16 overflows to NaN |
| text embedding | on CPU only if VRAM is still short (~0.6 GiB saved, negligible cost) |

Override any of them: `fast_tts.load(dtype=..., gemv=..., offload_embedding=..., decoder_on_gpu=...)`.

Reference: GTX 1650 Ti 4 GB → fp16, gemv on, embedding and decoder on CPU; 3.2 GiB peak VRAM, ~1.6× real time.

## Notes

- Pascal (GTX 10xx) and older aren't in the cu128 PyTorch build; use the cu126 index instead.
- `transformers` is pinned to 5.15.1: 5.19 breaks `qwen-tts-hf`. `fast_tts.py` relies on
  faster-qwen3-tts internals, so keep its pin too.
- Seeds don't reproduce the same voice across GPUs or library versions. For a persistent character voice,
  save a generated clip and clone it with a Qwen3-TTS Base model.
- On Windows, exceeding VRAM silently spills into slow shared memory instead of raising OOM.
- The SoX and flash-attn warnings at startup are harmless.
