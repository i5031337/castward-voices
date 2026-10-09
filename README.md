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

The model loads and captures CUDA graphs once at startup. Requests queue on one generation
thread so the GPU's shared caches cannot overlap. The server binds to localhost by default.
Use `--device cpu` for CPU generation, or `--model /path/to/model` for another model directory.
The default CUDA sequence budget is 4096; adjust with `--max-seq-len`.

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
