# qwen3-tts

Voice design with [Qwen3-TTS-12Hz-1.7B-VoiceDesign](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign):
describe a voice in plain text, get speech in that voice, with no reference audio needed. Uses
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (CUDA-graph decoding) with a loader
that adapts to the GPU, down to 4 GB cards like the GTX 1650 Ti.

## Setup

Requires an NVIDIA GPU (Turing / GTX 16-series or newer) with a driver supporting CUDA 12.8 (R570+).

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
```

The model (~4.3 GB) downloads to the Hugging Face cache on first run. For offline machines, pre-fetch with
`hf download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` and set `HF_HUB_OFFLINE=1`.

## Usage

```bash
.venv/Scripts/python demo.py          # Windows; use .venv/bin/python on Linux/macOS
.venv/Scripts/python demo.py -v "A gruff old pirate, gravelly and slow" -t "Arr, the tide waits for no one."
```

Options: `-l` language (English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish,
Italian, Auto), `--seed`, `--temperature`, `--max-seq-len` (prompt + audio frames at 12/s; lower is faster).
Output goes to `out/`.

From Python:

```python
import fast_tts
tts = fast_tts.load()
wavs, sr = tts.generate_voice_design(text="Hello there.", instruct="A warm elderly woman", language="English")
```

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
