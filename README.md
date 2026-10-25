# Castward Voices

<p><img src="webui/branding/castward-voices-logo.svg" width="640" alt="Castward Voices"></p>

Local voice design, cloning, and narration for the **Castward SillyTavern RPG extension**, with an OpenAI-compatible speech API and **Castward Voice Studio** web UI. Ming is the default character voice model, Qwen and FireRed offer alternative design and cloning profiles, Kokoro provides preset narrator voices, and PocketTTS 100M offers English presets and reference cloning. All synthesis runs through audio.cpp.

## Setup

Install [uv](https://docs.astral.sh/uv/), Git, a C/C++ compiler, and CMake 3.20 or
newer. For NVIDIA GPU synthesis, also install the CUDA toolkit and a working GPU
driver. FFmpeg is needed for Opus, AAC, and playback tempo changes. The commands
below are for Linux; run them from this repository's directory.

```bash
./setup.sh
./run-local.sh --kokoro-cpu
```

Open **<http://127.0.0.1:8001/>** for Castward Voice Studio.

Setup installs Python dependencies, obtains the pinned audio.cpp source if needed,
builds one shared native runtime, downloads the default **Ming + Kokoro** Q8 models,
and generates configs with the correct paths. The first CUDA build can take several
minutes; rerunning setup reuses completed compilation and downloads. No manual
JSON editing is required.

### Optional models

There is one server and one Studio. Add models to the existing installation:

```bash
./setup.sh --add pocket
# Stop the running server, then restart it:
./run-local.sh --kokoro-cpu
```

Select **PocketTTS 100M · English** in Studio's **Presets** or **Clone** tab.
PocketTTS cloning needs reference audio without a transcript. Its weights require
[Kyutai attribution](THIRD_PARTY_NOTICES.md).

| Setup name | Model | Studio modes |
|---|---|---|
| `ming` | Ming 0.5B (default) | Design, clone, presets |
| `kokoro` | Kokoro 82M (default) | Narrator presets |
| `pocket` | PocketTTS 100M English | Presets, clone |
| `firered` | FireRedTTS3 Instruct | Design, clone |
| `qwen` | Qwen3-TTS VoiceDesign, Base, CustomVoice | Design, clone, presets |

```bash
./setup.sh --add firered,qwen       # Add several models
./setup.sh --models all            # Install all supported models
./setup.sh --models ming,kokoro,pocket  # Choose the exact enabled set
./setup.sh --list-models           # List available models
./setup.sh --add pocket --plan     # Preview installation commands
```

`--add` keeps your existing selection. `--models` replaces the enabled selection;
downloaded weights remain on disk. Only enabled profiles appear in Studio.
`setup-firered.sh` and `setup-pocket.sh` are compatibility shortcuts to this same
installer.

### Hardware and configuration

`--kokoro-cpu` runs Kokoro and PocketTTS on CPU while other models use GPU 0, with
one character checkpoint resident at a time. Use `--gpu-device 1` to select another
GPU. This mode needs an enabled CPU model and an enabled character model.

For synthesis on GPU throughout, launch with `./run-local.sh`. Setup configures all
models for GPU 0 by default; `./setup.sh --device 1` selects another GPU. For a
machine without CUDA, use `./setup.sh --cpu` and then `./run-local.sh`.

Generated configs and the saved model selection live in `.cache/install/`.
Setup manages these files and rewrites them when selection changes. For custom
settings, copy `native.json` and `backend.json` elsewhere and select them with
`AUDIOCPP_CONFIG` and `SPEECH_CONFIG`. Existing installations without generated
configs still use `audiocpp-server.local.json` and `backend.json`.

The installer reuses `../Ming/audio.cpp` if present; otherwise it clones into
`.cache/audio.cpp`. `AUDIOCPP_SOURCE`, `AUDIOCPP_BUILD_DIR`, and `BUILD_JOBS` override
source location, build location, and compilation parallelism. `AUDIOCPP_SERVER`
overrides the runtime binary when launching. See [AUDIOCPP.md](AUDIOCPP.md) for
advanced settings and [adding future models](AUDIOCPP.md#adding-a-model-to-the-shared-installer).

**Restart after adding models or rebuilding.** The launcher reuses an already
healthy native service. If you see `unsupported model family hint: pocket_tts`,
run `./setup.sh --add pocket`, stop the old server, and restart it.

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
- **Clone:** upload a reference recording and its transcript (PocketTTS needs only the audio), or select **Use as reference** on a generated take. A clean 3–10 second clip is a useful starting point.
- **Presets:** choose a built-in speaker and enter text to synthesize.

| Model profile | Use |
|---|---|
| Ming (`npc`) | Voice design, cloning, and built-in speakers |
| Qwen (`qwen3-tts`) | 1.7B voice design and 0.6B cloning |
| Qwen (`qwen3-tts-1.7b`) | 1.7B voice design and 1.7B cloning |
| Qwen CustomVoice (`qwen3-customvoice`) | Nine preset speakers |
| FireRed (`fireredtts3`) | Instruct voice design and reference cloning; compatibility presets design fresh voices |
| Kokoro (`narrator`, alias `kokoro`) | Preset narration and PCM streaming |
| PocketTTS (`pocket-tts`) | English preset voices and reference cloning |

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
