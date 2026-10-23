# Castward Voices

<p><img src="webui/branding/castward-voices-logo.svg" width="640" alt="Castward Voices"></p>

Local voice design, cloning, and narration for the **Castward SillyTavern RPG extension**, with an OpenAI-compatible speech API and **Castward Voice Studio** web UI. Ming is the default character voice model, Qwen offers alternative design and cloning profiles, and Kokoro provides preset narrator voices. All synthesis runs through audio.cpp.

## Setup

You need Python 3.11, [uv](https://docs.astral.sh/uv/), a built `audiocpp_server`, and GGUF model files. The supplied native configuration uses CUDA; build audio.cpp with `ming3_tts`, `kokoro_tts`, and `qwen3_tts` support to use all included profiles. Install FFmpeg for Opus, AAC, and playback tempo changes.

From the repository directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

Configure the native backend for your machine before starting:

- In [audiocpp-server.local.json](audiocpp-server.local.json), set model `path` values to your GGUF files. Configure GPU devices and CPU threads there as needed.
- The launcher expects the binary at `../Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server`. Set `AUDIOCPP_SERVER` to use a different path.
- [backend.json](backend.json) defines the public model profiles and native service URL. See the [native configuration guide](AUDIOCPP.md#replace-models-through-configuration) for custom profiles.

Start with CPU narration and character synthesis on GPU 0:

```bash
./run-local.sh --kokoro-cpu --port 8001
```

Open **<http://127.0.0.1:8001/>** for Castward Voice Studio. Use `--gpu-device 1` to select another GPU in this mode. The launcher manages the native services and stops them when you exit.

For GPU narration, run `./run-local.sh --port 8001`. The supplied configuration places Ming/Qwen on GPU 1 and Kokoro on GPU 0; for a single GPU, set every model's `device` to `0` in `audiocpp-server.local.json`.

### Hardware guidance

Allow this much **available VRAM**, after the desktop and other applications:

| Workflow | Recommended available VRAM |
|---|---:|
| Ming + CPU Kokoro (`--kokoro-cpu`) | 3 GiB |
| Qwen + CPU Kokoro (`--kokoro-cpu`) | 5 GiB |
| Ming + GPU Kokoro on one GPU | 6 GiB |
| One Qwen checkpoint + GPU Kokoro on one GPU | 8 GiB |

These are planning budgets based on Q8 workloads, not guaranteed minimum card sizes. Longer outputs can need more memory. CPU narration mode keeps one GPU checkpoint resident and swaps models as needed. See the [hardware measurements and residency guidance](AUDIOCPP.md#native-ming-comparison-and-server-vram-budget-2026-10-22) for details.

## Use Castward Voice Studio

Choose a model, then a supported mode:

- **Design:** describe a voice and enter the text it should speak.
- **Clone:** upload a reference recording and its transcript, or select **Use as reference** on a generated take. A clean 3–10 second clip is a useful starting point.
- **Presets:** choose a built-in speaker and enter text to synthesize.

| Model profile | Use |
|---|---|
| Ming (`npc`) | Voice design, cloning, and built-in speakers |
| Qwen (`qwen3-tts`) | 1.7B voice design and 0.6B cloning |
| Qwen (`qwen3-tts-1.7b`) | 1.7B voice design and 1.7B cloning |
| Qwen CustomVoice (`qwen3-customvoice`) | Nine preset speakers |
| Kokoro (`narrator`, alias `kokoro`) | Preset narration and PCM streaming |

The Studio provides playback, waveforms, downloads, and model-specific generation controls. Takes stay in browser memory: **download anything you want to keep before refreshing**. Studio takes do not create saved NPC voices. See the [Studio guide](AUDIOCPP.md#castward-voice-studio-web-ui) for reference limits and advanced controls.

## Use the speech API

The API base URL is `http://127.0.0.1:8001/v1`. Supported output formats are WAV, MP3, FLAC, PCM, Opus, and AAC.

### Narration

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

For Kokoro streaming, set `"stream": true` and `"response_format": "pcm"`. The response is raw 24 kHz mono signed 16-bit little-endian PCM.

### Persistent character voices

Create a character by supplying a unique `npc_id` and voice description:

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "npc",
    "npc_id": "guard_01",
    "instructions": "A gruff elderly guard with a low, gravelly voice.",
    "input": "Halt! State your business.",
    "response_format": "wav"
  }' \
  --output intro.wav
```

The server returns the first line and prepares a reusable reference in the background. Check `GET /v1/voices/guard_01` until its status is `ready`, then reuse the same model and `npc_id` without `instructions`:

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "npc",
    "npc_id": "guard_01",
    "input": "You may pass. Keep out of trouble.",
    "response_format": "wav"
  }' \
  --output dialogue.wav
```

Saved character voices live in `voices/`; back up this directory to keep them.

| Endpoint | Purpose |
|---|---|
| `POST /v1/audio/speech` | Generate speech |
| `GET /v1/models` | List configured model profiles |
| `GET /v1/voices` | Discover available voices |
| `GET /v1/voices/{npc_id}` | Check a saved character's preparation status |
| `GET /health` | Check server health |

## Connect SillyTavern

Set SillyTavern's **OpenAI Compatible TTS** provider URL to `http://127.0.0.1:8001/v1`. The Castward extension sends character descriptions and NPC IDs automatically, routes narration to Kokoro, and reuses saved character voices.

The server listens on `127.0.0.1` by default. To serve trusted devices on your local network, add `--host 0.0.0.0` and use the server machine's address in your client. Set `TTS_API_KEY` to require bearer authentication; enter that key in your client or the Studio connection settings.

See [AUDIOCPP.md](AUDIOCPP.md) for advanced configuration, separately hosted native services, and troubleshooting logs.

## License

This project's code is licensed under the [Apache License 2.0](LICENSE). Downloaded model weights and third-party packages retain their respective licenses.
