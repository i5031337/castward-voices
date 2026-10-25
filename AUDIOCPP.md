# Castward Voices · audio.cpp server

Castward Voices is the local speech companion for the Castward RPG extension. All synthesis runs in audio.cpp. The Python API handles NPC persistence, reference selection, the SillyNPC bridge, audio encoding, and client compatibility. It does not import the Qwen or Kokoro Python synthesis runtimes.

## Run locally

The existing virtual environment can run the new server. A fresh Python 3.11 environment needs `uv pip install -r requirements.lock.txt`; install FFmpeg separately for tempo changes, AAC, and Opus. Optional CPU word alignment needs `uv pip install -r requirements-alignment.txt` and the local Whisper checkpoint (`python download_alignment.py`).

### FireRedTTS3 in the standard server

The normal `backend.json` and `audiocpp-server.local.json` include the public `fireredtts3` profile. It uses one FireRedTTS3 Instruct checkpoint for design and cloning, loaded on demand on GPU 1 alongside the other character models. Ming remains the default. CPU narration mode routes FireRed to the selected GPU with the same one-model residency limit as Ming and Qwen.

Add FireRed to the shared installation:

```bash
./setup.sh --add firered
# Stop the running server, then restart it:
./run-local.sh --kokoro-cpu
```

The common installer builds all cataloged native families, downloads the selected
Q8 packages, and generates native/API configs containing only enabled models.
Existing selections are retained by `--add`. `./setup-firered.sh` is a compatibility
shortcut for the same command. Source/build overrides, CPU-only setup, and the
initial install are covered in the [README](README.md#setup).

Select **FireRedTTS3 Instruct** in Studio for Design or Clone, or send `model: "fireredtts3"` to `/v1/audio/speech`. NPC creation, background reference preparation, recovery, saved-reference reuse, the SillyNPC bridge, output formats, and FFmpeg tempo adjustment use the existing server paths. Compatibility voices such as `alloy`, `echo`, and `nova` design a fresh recording from their descriptions; save an NPC or reuse a reference to keep a voice identity.

Design uses `template_name=voice_design`; Clone uses `instruct_tts`. The profile's `mode_conditions` restrict each mode to its compatible template in both the adapter and Studio. Semantic/acoustic editing requires native input-audio fields that this speech UI/API does not expose. Cloning accepts the reference transcript; complete short NPC recordings also supply their known transcript during extension. Cropped or possibly truncated recordings do not receive a guessed transcript. FireRed has no native streaming entry; PCM downloads are supported, while `stream: true` is rejected.

Native discovery supplies the flow-step, guidance, stop-threshold, and text-chunk controls. Untouched controls retain native defaults, including ten inference steps. Seeds use `0..4294967295`. Language names retain FireRed's capitalization; `Auto` maps to English, and native dialect tags can be entered directly. See the [native FireRed guide](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/fireredtts3.md) for supported tags and model limitations. The earlier Ming/Qwen memory budgets do not establish a FireRed Q8 budget.

Reproduce the live design/clone/preset/NPC checks against a running FireRed-capable native service:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/smoke_firered.py --backend-url http://127.0.0.1:7862
```

This uses temporary NPC storage and saves recordings and observations under `out/bench/firered-smoke/`. It checks finite, non-silent audio and NPC readiness; it does not assess voice quality. API elapsed times include reference preparation for the first NPC request because the test waits for its background task.

The [2026-10-23 CPU integration run](benchmarks/firered-integration-2026-10-23.json) passed discovery, design, cloning, compatibility presets, short-NPC extension, and saved-reference reuse with native Q8, four Ryzen 5800X3D threads, and four inference steps. The 0.96-second first NPC recording was extended to a 5.12-second reference and reached `ready`. These are single smoke observations; CUDA and GPU memory were not measured because the NVIDIA driver was unavailable in that session.

### PocketTTS 100M: English presets and reference cloning

The public `pocket-tts` profile uses Kyutai's English PocketTTS 100M Q8 checkpoint
through native `pocket_tts`. It supports packaged voices (default `alba`) and
ordinary reference-WAV cloning in Studio. Ming remains the default profile.
PocketTTS does not design voices from descriptions. Reference transcripts and
language controls are omitted; another language needs its own checkpoint/profile.
This initial integration exposes offline synthesis; native streaming and persistent
prepared voice-state exports are not exposed by this profile.

Add PocketTTS to the shared native server and download the pinned English Q8
package, including its packaged `embeddings/*.safetensors` voices:

```bash
./setup.sh --add pocket
# Stop the running server, then restart it:
./run-local.sh --kokoro-cpu --port 8001
```

The common installer keeps the existing model selection, adds PocketTTS, and
updates the same shared native runtime. `./setup-pocket.sh` is a compatibility
shortcut. The catalog pins both source and download revisions; the English
package includes all 26 packaged `embeddings/*.safetensors` voices. Keep the
embeddings directory beside its GGUF.

The generated configs enable native `pocket-english` and public `pocket-tts`.
With `--kokoro-cpu`, PocketTTS joins Kokoro in the CPU worker process; other
character models retain the one-model CUDA residency limit. Plain
`./run-local.sh` uses the backend/device selected during setup. Existing local
configs without setup-generated configs retain their configured devices.

If Studio reports `unsupported model family hint: pocket_tts`, the running native
binary was built without PocketTTS. Run `./setup.sh --add pocket`, stop the running
server, and restart it. The launcher reuses an already healthy native service,
so rebuilding alone does not replace that running process.

Select **PocketTTS 100M · English** in Studio's Presets or Clone tab, or send:

```json
{"model":"pocket-tts","input":"Welcome back, traveler.","voice":"alba","response_format":"wav"}
```

Studio accepts uploaded reference audio without a transcript. Existing saved NPC
references can be reused by sending `model: "pocket-tts"` with their `npc_id`;
creating a new NPC still requires a design-capable profile. Cloning uses the saved
WAV, including the existing reference selection/extension and recovery paths, and
does not save Pocket-specific prepared states. Native controls such as
`text_chunk_tokens`, `max_frames`, and `temperature` come from discovery. Prefer
`options.max_frames` for output limits: the native legacy `max_tokens` (also reached
via `max_new_tokens`) means input tokens per text chunk for this family. Offline
speed changes use FFmpeg. Outputs retain the facade's default 24 kHz mono format.

The weights are **CC-BY-4.0**, attributed to **Kyutai**; see
[third-party attribution](THIRD_PARTY_NOTICES.md), the
[upstream model card](https://huggingface.co/kyutai/pocket-tts), and
[native PocketTTS documentation](https://github.com/0xShug0/audio.cpp/blob/main/docs/tts.md#pockettts).
Q8 is a GGUF quantization of the original weights. No PocketTTS CPU speed or memory
budget is inferred from the Ming, Qwen, or Kokoro measurements below.

Reproduce discovery, packaged synthesis, uploaded-reference cloning, and temporary
saved-NPC reuse against the running CPU native service:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tests/smoke_pocket.py
```

Recordings and elapsed/audio timings go to `out/bench/pocket-smoke/`; the check
validates finite, non-silent audio rather than comparative voice quality.

The [2026-10-24 CPU smoke run](benchmarks/pocket-integration-2026-10-24.json)
passed all four checks with four threads: preset 0.72 s for 3.36 s audio, reference
clone 0.74 s for 2.72 s audio, and saved-NPC reuse 0.25 s for 1.36 s audio. These
are single facade observations (the preset includes first load), not warmed
medians, memory measurements, text-coverage checks, or voice-quality ratings.

### Windows FireRedTTS3 pilot

`audiocpp-server.firered.json` and `backend-firered.json` isolate FireRedTTS3 Instruct from the older machine's native model paths. `run-firered.ps1` starts the native server and Castward API together, then stops only those child processes when the launcher exits. Open <http://127.0.0.1:8001/> after starting it. The native model is lazy-loaded on the first synthesis request.

The pilot uses the official [audio.cpp v0.9.2 Windows CUDA 13.3 binary](https://github.com/0xShug0/audio.cpp/releases/tag/v0.9.2) extracted into `.cache/audio.cpp-v0.9.2/runtime/`, and the [FireRedTTS3 Instruct Q8 GGUF](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/ef1ab2598f6d4d101486e4c5b9e605c687f20642/FireRedTTS3-Instruct-GGUF) under `models/audiocpp-extra/`. The supplied native config runs a locally converted **Q4_K Instruct** file on GPU 0. Both directories are ignored by Git. The conversion preserves audio.cpp's eight embedded sidecars and model spec. The original Q8 is 4.15 GB and does not fit this machine's 4 GB GPU; the converted Q4_K is 3.19 GB on disk. This lower-bit FireRed variant has not been quality-validated against Q8 beyond smoke synthesis.

From PowerShell in the repository directory, with `hf` and `uv` installed:

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.lock.txt
New-Item -ItemType Directory -Force .cache\audio.cpp-v0.9.2 | Out-Null
curl.exe -L --fail --output .cache\audio.cpp-v0.9.2\audio-v0.9.2-bin-windows-x64-cuda13.3.zip https://github.com/0xShug0/audio.cpp/releases/download/v0.9.2/audio-v0.9.2-bin-windows-x64-cuda13.3.zip
Expand-Archive -LiteralPath .cache\audio.cpp-v0.9.2\audio-v0.9.2-bin-windows-x64-cuda13.3.zip -DestinationPath .cache\audio.cpp-v0.9.2\runtime
hf download audio-cpp/audio.cpp-gguf FireRedTTS3-Instruct-GGUF/fireredtts3-instruct-q8_0.gguf --revision ef1ab2598f6d4d101486e4c5b9e605c687f20642 --local-dir models\audiocpp-extra
.\.cache\audio.cpp-v0.9.2\runtime\audiocpp_gguf.exe --input models\audiocpp-extra\FireRedTTS3-Instruct-GGUF\fireredtts3-instruct-q8_0.gguf --output models\audiocpp-extra\FireRedTTS3-Instruct-GGUF\fireredtts3-instruct-q4_k.gguf --type q4_k --family fireredtts3
.\run-firered.ps1
```

The public `fireredtts3` profile maps `voice_design` and `instruct_tts` templates to the same native Instruct checkpoint. Preset-style synthesis uses the configured `alloy` description to design a fresh voice; NPC requests generate a first designed recording and then clone from its prepared reference. FireRed's [model guide](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/fireredtts3.md) documents the reference transcript, language tags, and uint32 seed limit.

Local smoke results on the GTX 1650 Ti, with `num_inference_steps=4` and seed 1895: CPU Q8 designed 2.88 seconds of audio in 49.67 seconds (RTF 17.25) and cloned 2.56 seconds in 19.39 seconds (RTF 7.57). The Q4_K GPU facade designed 2.88 seconds in 10.06 seconds including first model load, cloned 3.68 seconds in 3.61 seconds (RTF 0.98), and cloned a 236-character paragraph into 18.08 seconds in 14.36 seconds (RTF 0.79). After that paragraph, `nvidia-smi` showed 3,649 MiB used and 287 MiB free. A short new NPC line reached `ready` after reference extension to 5.76 seconds; the next saved-NPC request succeeded in 3.88 seconds end to end. These are single observations, not medians or peak VRAM measurements. The Q4 output needs listening review and broader text/voice tests before changing the default profile.

The separate [CrispASR Q4_K package](https://huggingface.co/cstr/fireredtts3-GGUF) is a 2.4 GB **Base** model with a separate RedAE file. It does not include Instruct voice design, and its GGUF layout is not documented as compatible with audio.cpp. The local audio.cpp-native conversion above retains the Instruct workflow.

```bash
./run-local.sh --port 8001
```

This starts audio.cpp on port 7862 if it is not already healthy, then the compatibility API on port 8001. The launcher stops only processes it started. It uses `../Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server`, with character models on GPU 1 and Kokoro Q8 on GPU 0. Offline and streaming Kokoro are separate native model entries and can consume separate memory. Change paths, devices, residency, and threads in `audiocpp-server.local.json` for other machines. Build audio.cpp with `ming3_tts`, `kokoro_tts`, `qwen3_tts`, `fireredtts3`, and `pocket_tts` enabled.

Environment overrides: `PYTHON`, `AUDIOCPP_SERVER`, `AUDIOCPP_CONFIG`, `AUDIOCPP_CUDA_LIB`, and `SPEECH_CONFIG`. Native logs go to `audiocpp-backend.log`. `TTS_API_KEY` still enables bearer authentication on the Python API. Separately hosted native services can be configured with `backend_url`, optional native-model-ID `backend_routes`, and `backend_api_key_env` in `backend.json`; it can also be used directly with `python server.py --config backend.json`.

## CPU narration with GPU character speech

```bash
./run-local.sh --kokoro-cpu --port 8001
# Select another GPU:
./run-local.sh --kokoro-cpu --gpu-device 1 --port 8001
```

This mode targets **4 GB cards for Ming** and **6 GB cards for Qwen**, with approximately **3 GiB / 5 GiB available VRAM** respectively after accounting for the desktop and other applications. The default device in this mode is GPU 0, so one GPU is sufficient. Measurements used Q8 models on a 16 GB card; these are planning budgets, not physical 4/6 GB card performance guarantees. Longer outputs can increase memory requirements.

The launcher creates temporary configs in `.cache/launch.*` and starts two native processes: **CUDA on port 7864**, containing the other character models with **one resident model**, and **CPU on port 7863**, containing offline/streaming Kokoro and PocketTTS. The GPU process swaps checkpoints when you switch between Ming, Qwen design, and Qwen cloning. This is necessary to avoid keeping two large Qwen checkpoints resident on a 6 GB card. The CPU process can keep both Kokoro entries and PocketTTS resident in host RAM. No source config is rewritten.

Native-ID `backend_routes` send both Kokoro entries and PocketTTS to CPU for their supported speech and voice-discovery paths. Model/parameter discovery merges both native services, so Castward Voice Studio and existing clients use the same public profiles. CPU streaming defaults to `text_chunk_size=80` for earlier first audio, unless the profile already sets a chunk budget. Native thread counts come from `audiocpp-server.local.json` (four in the supplied config). Authentication settings are preserved.

Stop the previous launcher before switching modes to release its GPU allocations. CPU mode starts its own services rather than reusing an existing CUDA service whose residency settings may exceed the intended budget. Override the private native ports with `AUDIOCPP_GPU_PORT` and `AUDIOCPP_CPU_PORT` if needed; occupied ports fail before starting either service. Logs are `audiocpp-backend-cpu-mode-gpu.log` and `audiocpp-backend-cpu.log`. The launcher stops only its own child processes and removes temporary configs when the API or either native child exits, including startup failures. `AUDIOCPP_CONFIG`, `SPEECH_CONFIG` (or `--config`), and `AUDIOCPP_SERVER` remain supported. Custom configs already using `backend_routes` should launch their services manually.

Without `--kokoro-cpu`, startup and the existing GPU narration configuration are unchanged.

Live validation covered UI discovery, offline CPU narration, CPU PCM streaming alongside Ming design/cloning and Qwen design/both cloning sizes, and checkpoint eviction. The CUDA process peaked at **4.05 GiB** across those workloads; the CPU process showed **zero GPU allocation**. Terminating a native child verified cleanup of both native processes, the facade, and temporary configs while the original server remained running. See [`benchmarks/cpu-kokoro-launch-validation-2026-10-22.json`](benchmarks/cpu-kokoro-launch-validation-2026-10-22.json).

## Replace models through configuration

### Future model candidates (2026-10-23)

These are candidates for separate experiments unless an integration is linked below. Compare each against Ming, Qwen, and Kokoro with the same text and reference recordings before changing defaults.

| Candidate | Reason to evaluate | Integration consideration |
|---|---|---|
| PocketTTS 100M | [Integrated English presets and ordinary reference cloning](#pockettts-100m-english-presets-and-reference-cloning), including CPU launch | Prepared-state persistence remains future work. Weights are CC-BY-4.0; see THIRD_PARTY_NOTICES.md. |
| VoxCPM2 | Design, controllable cloning, 48 kHz output, and native streaming | Design instructions are prepended to the input text, so the generic `instruction_option` mapping is insufficient without adapter work. Measure VRAM and latency before considering it for the shared GPU. Weights are Apache-2.0. |
| VibeVoice 1.5B | Long-form, multi-speaker audiobook scenes | Requires speaker-labeled scripts and multiple ordered voice references. This is a scene-rendering feature, not a normal single-voice profile. Weights are MIT. |
| Qwen native streaming | Earlier first audio from checkpoints already installed | Add separate streaming native entries and test the PCM/SSE path; Base streaming needs reference audio, while the current public streaming request path does not supply one. VoiceDesign or CustomVoice may be simpler first tests. |
| OmniVoice | One family offers design, cloning, broad language coverage, and chunked streaming | Its streaming is chunked pseudo-streaming. Original OmniVoice weights are non-commercial, so treat it as a research comparison unless licensing changes. |
| Fish Audio S2 Pro | Expressive cloning and multi-reference, speaker-tagged dialogue | Multi-reference conditioning needs a new UI/API input path. Its research license requires separate permission for commercial use. |

Source documentation: [audio.cpp TTS models](https://github.com/0xShug0/audio.cpp/blob/main/docs/tts.md), [Qwen streaming](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/qwen3_tts.md), and [model licenses](https://github.com/0xShug0/audio.cpp/blob/main/docs/model_licenses.md).

`backend.json` maps public profile IDs to native IDs. The supplied `npc` profile uses Ming for design and cloning; `narrator` uses Kokoro for narration and streaming. Old client model IDs are optional aliases in this JSON file. Neither model names nor Qwen generation defaults are embedded in the Python routing code.

To change architectures, add or replace the model entries in audio.cpp's configuration, then edit the public profile:

* `backend_model`, `design_model`, `clone_model`, and `streaming_model` select native entries and capabilities. Omit unsupported capabilities.
* `instruction_option`, `reference_text_option`, `language_option`, and `language_map` translate profile-specific option names. Set `language_option` to null if the architecture does not accept it.
* `options`, `design_options`, `clone_options`, and `streaming_options` supply native generation options. Request `options` can override them. Explicit `temperature` and `max_new_tokens` map to native `temperature` and `max_tokens`; omitted fields leave native defaults intact.
* `voice_aliases`, `voice_descriptions`, and `default_voice` define compatibility voices. Descriptions request voice design; presets pass a native voice ID.
* `native_speed` selects native speed control. Otherwise the facade uses FFmpeg for non-streaming tempo changes. Streaming speed changes require native support.
* `streaming_sample_rate` and `streaming_channels` describe native PCM. They must match the native model; WAV sample rates are read from their headers.
* `max_input_chars`, `context_max_chars`, and reference duration settings are configurable rather than inferred from model names.

`GET /v1/backend/models` exposes native model/parameter discovery. `GET /v1/models` lists public profiles and aliases; `GET /v1/voices` combines configured voices with native discovery. Native parameter names and architecture limits remain audio.cpp's responsibility. For long English Ming requests, `options: {"text_chunk_size": 140}` can improve text coverage; an appropriate chunk budget depends on the language and workload.

## Preserved client behavior

`POST /v1/audio/speech` retains WAV, MP3, FLAC, PCM, Opus, AAC, speed, NPC fields, and the encoded `sillynpc-v1:` bridge. The default output remains mono 24 kHz for the existing audiobook client, even when a backend synthesizes at another rate. `output_sample_rate` controls this conversion.

Streaming requests use `stream: true` and `response_format: "pcm"` with a streaming-enabled profile. The API translates native SSE audio events into raw signed 16-bit little-endian PCM. It checks the first native audio event before returning success, and closes the native stream when the client disconnects. Mid-stream failures terminate the response rather than produce a complete success signal.

NPC storage keeps the existing hashed directories, `dialogue.wav`, `reference.wav`, and transcripts. Existing Qwen `prompt.npz` files are left untouched and ignored. Reuse means cloning from the saved recording; a replacement architecture may render that voice differently. New records contain generic backend/profile metadata. The first utterance is returned before reference preparation runs. Short first utterances are extended using their own audio as the speaker reference; failed preparations retain the original dialogue and retry on a later request. Preparing records recover on startup. Back up `voices/` before changing production models.

Context-aware narration remains available on profiles with `context: true`. The facade synthesizes the combined context in audio.cpp, then optionally aligns words on CPU and cuts the target. Native Kokoro timestamps describe phoneme groups, so they are not used as written-word timings. Missing alignment support or unsafe boundaries trigger standalone synthesis with `X-Narration-Context: fallback`; successful cuts return `applied`. This CPU alignment adds latency compared with the previous Python Kokoro timing path. Ordinary narration and suitable-duration NPC references need no Python alignment model.

NPC requests are serialized separately from narration. Native calls sharing a model ID are locked in the adapter; audio.cpp owns model loading and inference. The previous Qwen loader flags, CUDA graph setup, and VRAM planner are no longer server options.

## Validation and historical tools

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

The regression suite uses fake native responses to check model mapping, reference reuse/recovery, response ordering, streaming errors/disconnects, sample rates, alignment boundaries, and authentication. Live validation also exercised native Ming design/cloning, Kokoro offline/streaming, concurrent narration/dialogue, and CPU contextual cuts.

The old direct-runtime demos, benchmarks, and Qwen/Kokoro helpers remain for historical experiments. Their dependencies are not part of the new server requirements. Previous runtime-specific tests and requirements are preserved in `legacy_tests/` with filenames excluded from normal test discovery. The earlier architecture notes below the notices in README and AGENTS describe that historical runtime.


## Castward Voice Studio web UI

The studio uses Castward’s teal, violet, and pink palette. Self-contained SVG assets in `webui/branding/` adapt the Castward character speech bubbles and wordmark with a speaker badge for the Voices companion.

Start the server with `./run-local.sh --port 8001`, then open <http://127.0.0.1:8001/>. The UI is served by the same Python API; it needs no Node build, CDN, or external assets. API bearer keys can be entered through the connection settings and are held in this tab's session storage. API routes continue to require authentication when configured, including the studio routes.

Choose a model and **Design**, **Clone**, or **Presets**. Capabilities come from the configured public profiles, and the active mode's native model publishes its generation controls through discovery. Unsupported modes are disabled. Every advertised native request option is available through a typed control or its dedicated language, instruction, transcript, seed, or speed field. Overrides are opt-in: untouched fields retain configured/native defaults. Additional JSON options can override controls and pass through options added by newer backends. Session/load options remain deployment settings in the audio.cpp configuration. The Preset voice dropdown lists the selected model's discovered voices, configured aliases, and native voice IDs advertised in enum metadata. Choose **Model default** to retain the configured/native default.

Each take has a waveform, player, download, saved request settings, generation time, audio duration, sample-rate information, and real-time throughput. **Use as reference** moves the original native-rate WAV and its transcript into Clone mode; choose new text and generate. References can be reused across clone-capable profiles. Output format and playback tempo do not degrade the source recording kept for cloning. A replacement architecture can render the same reference differently.

You can also drop or upload WAV, FLAC, Ogg, or MP3 audio. References are limited to 5 MiB, two minutes, and eight source channels, with an additional bound on decoded PCM size. Individual models can have smaller limits; a clean 3–10 second clip is a useful starting point. Edit the reference transcript if a designed take skipped or changed words; leaving it blank requests speaker-only conditioning where supported.

The workbench keeps up to eight takes in browser memory, with independent design/clone option drafts. Download recordings you want to keep before refreshing. These experiments do not create or modify saved NPC records. WAV previews play regardless of download format. Context-aware preset narration is available on profiles that enable it.

Implementation: `voice_studio.py`, `webui/`, and a registration hook in `server.py`. `GET /ui/config` combines profiles and native metadata; `POST /ui/generate` validates capabilities, decodes references, and uses the existing serialized workers and native adapter. The OpenAI-compatible speech and NPC endpoints are unchanged.


## Qwen models in Castward Voice Studio

The audio.cpp build also enables `qwen3_tts`. The local configuration includes four native Q8 checkpoints on GPU 1, loaded on demand with a two-model cache shared across native entries. This bounds residency as you switch between architectures; it can cause a model to reload after other models run. Ming remains the default `npc` profile, and Kokoro remains `narrator`.

| Public model / UI choice | Design | Clone | Presets |
|---|---|---|---|
| `qwen3-tts` | 1.7B VoiceDesign | 0.6B Base | Compatibility descriptions use VoiceDesign |
| `qwen3-tts-1.7b` | 1.7B VoiceDesign | 1.7B Base | Compatibility descriptions use VoiceDesign |
| `qwen3-customvoice` | — | — | 1.7B CustomVoice, nine native speakers |

`qwen3-tts-0.6b` aliases `qwen3-tts`. The former Qwen aliases that pointed to Ming now route to actual Qwen models. `npc`, `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` continue to use Ming. Both Qwen design/clone profiles share the same VoiceDesign checkpoint and differ in the Base checkpoint used for cloning. Use **Use as reference** on a take to enter Clone mode, or switch to the other Qwen profile to compare cloning sizes with the same reference. CustomVoice itself does not clone; its takes can be handed to a compatible Qwen Base profile, or another clone-capable profile you select.

The UI discovers Qwen's native sampling controls, including main/subtalker temperature, top-k/top-p, repetition penalty, greedy/sample toggles, chunk size, and reference mode. Qwen seeds are limited to `0..4294967295`. The generic profile `seed_max` enforces that limit; derived NPC seeds and legacy reference-extension seeds are reduced to the selected profile's range. The UI publishes the limit as a decimal string to preserve full precision for other models that accept uint64 seeds.

With an exact reference transcript, Qwen uses its normal reference-code conditioning. Without one, the configured `speaker_only_option: "x_vector_only_mode"` automatically enables speaker-only conditioning. An explicit request option can override this behavior, in which case the native model validates the combination. This also allows the NPC workflow to extend a short first utterance without an invented transcript. Qwen Base does not offer the same instruction-based style control as VoiceDesign or CustomVoice; native option descriptions identify which variant uses each field.

The GGUF files are in `../Ming/models/audiocpp-extra/`. They are pinned to `audio-cpp/audio.cpp-gguf` revision `d81ea87066e0b48ddfc7ef3fc7254ad230d45faa`; `../Ming/setup-qwen-audiocpp.sh` rebuilds the native targets and fetches these four files. Existing original Qwen safetensors and saved voices are untouched. Live validation covered design, both cloning sizes, no-transcript cloning, CustomVoice, return to Ming, and short-NPC reference extension/reuse; browser checks covered model selection, native controls, seed bounds, and reference handoff.


### Native Qwen speed and memory (2026-10-22)

Measured through the running audio.cpp HTTP server with the configured Q8 checkpoints on GPU 1 (RTX 5060 Ti, 16 GB), CUDA backend, four CPU threads, and the normal two-entry model cache. Source checkout revision: `bd7032cba3a83144de4480456f7f1e5e7e2dccd9`. These measurements supersede the historical Python estimates for this configuration; they do not measure Ming or Kokoro narration performance.

Both Base models received the same 4.72-second VoiceDesign recording, exact reference transcript, and target text. Sampling used native defaults, seed 1895, and `max_tokens=512`. The short passage has 68 characters; the long passage has 229. Each table entry is the median of three requests after model loading and an additional short warmup. Different models produce different audio lengths, so RTF (generation time / audio duration) is the useful throughput comparison.

| Native Q8 checkpoint | Short: generation / audio | Long: generation / audio | RTF, short / long | Process VRAM on Qwen GPU, short / long |
|---|---:|---:|---:|---:|
| 1.7B VoiceDesign | 0.70 s / 3.92 s | 2.99 s / 15.60 s | 0.179 / 0.192 | 2.59 / 3.77 GiB |
| 0.6B Base, cloning | 0.64 s / 4.00 s | 2.19 s / 14.24 s | 0.159 / 0.154 | 2.21 / 3.20 GiB |
| 1.7B Base, cloning | 0.81 s / 4.16 s | 2.53 s / 13.36 s | 0.196 / 0.189 | 3.07 / 3.96 GiB |

The 0.6B Base delivered about **23% more audio per second** than 1.7B Base on both passages and used **0.76–0.86 GiB less VRAM**. For identical text, client completion time was 0.64 vs 0.82 seconds (short) and 2.20 vs 2.54 seconds (long). Both cloning sizes generated roughly five to six-and-a-half seconds of audio per second. This establishes a modest speed and memory advantage for 0.6B here; it does not compare voice quality.

Native generation time comes from `X-AudioCPP-Wall-Ms`, excluding loading and HTTP/WAV transfer. Warm client overhead was about 2–8 ms. First short requests after model eviction took **2.42 s for VoiceDesign, 2.66 s for 0.6B Base, and 4.02 s for 1.7B Base** end to end. These are individual observations with warm OS file caches, not fresh-process startup estimates. Native cloning includes reference processing; the facade's resampling, output encoding, NPC reference preparation, and optional alignment are outside this benchmark. The first long request also allocated/warmed additional buffers and took about 0.3–0.45 s longer than subsequent long requests.

VRAM values are sampled **server-process allocations on GPU 1**, including CUDA context and retained buffers, with one Qwen entry resident. A Kokoro entry remained on GPU 0 and is excluded from that column. The sampler uses `nvidia-smi` at approximately 100-ms intervals plus query overhead; brief peaks can be missed. These figures are neither PyTorch allocated/reserved counters nor whole-card consumption. Other applications and the display consume additional memory. The same model's retained allocation grew by about 0.9–1.2 GiB after the longer output.

With both Qwen entries resident after short synthesis, VoiceDesign + 0.6B Base occupied **4.61 GiB**, versus **5.46 GiB** for VoiceDesign + 1.7B Base. A mixed case with 1.7B Base's long-output buffers retained and VoiceDesign's short-output buffers reached **6.36 GiB**. Both models retaining long-output buffers, longer passages, and concurrent inference were not tested; leave headroom above these observations. The two-entry limit counts models across devices: loading Kokoro or Ming can evict a Qwen entry and introduce reload latency. Model size on disk is also different from runtime memory: the Q8 files are 2.62 GiB (VoiceDesign), 1.85 GiB (0.6B Base), and 2.51 GiB (1.7B Base).

Reproduce with an idle client session and the project's two-entry cache configuration:

```bash
# Get the native server PID from nvidia-smi's process list.
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
.venv/bin/python benchmark_audiocpp_qwen.py --server-pid <PID>
```

The benchmark makes ordinary synthesis requests and uses the two Kokoro entries to evict Qwen between scenarios. It changes the resident cache, leaves saved NPCs and configuration untouched, and requires access to GPU telemetry. It validates generated waveforms, saves a reusable reference, all request timings, and memory samples under `out/bench/audiocpp_qwen.*`. A compact record of this run is checked in at [`benchmarks/audiocpp-qwen-2026-10-22.json`](benchmarks/audiocpp-qwen-2026-10-22.json). Other applications were resident on the GPUs; this was not an exclusive-device benchmark. Measurements cover one voice, two English passages, and one seed, not a universal hardware or workload guarantee.


### Native Ming comparison and server VRAM budget (2026-10-22)

Ming 3.0 TTS 0.5B Q8 was measured on the same RTX 5060 Ti, CUDA backend, four CPU threads, 68/229-character passages, seed 1895, and `max_tokens=512` as Qwen above. Design used the same description via `template_name=voice_design`; cloning used `template_name=tts` and the **identical Qwen-generated 4.72-second WAV and exact transcript**. Other generation and chunking settings retained native defaults. One cold request and a short warmup preceded three measured runs per passage/task. Ming outputs were 44.1 kHz; Qwen outputs were 24 kHz. These native measurements exclude facade resampling, encoding, and optional alignment.

| Ming Q8 task | Short: generation / audio | Long: generation / audio | RTF, short / long | Process VRAM on synthesis GPU, short / long |
|---|---:|---:|---:|---:|
| Voice design | 0.91 s / 3.50 s | 3.28 s / 12.80 s | 0.259 / 0.256 | 1.48 / 1.52 GiB |
| Voice cloning | 0.95 s / 3.50 s | 2.95 s / 11.60 s | 0.272 / 0.254 | 2.05 / 2.08 GiB |

Ming generated about 3.7–3.9 seconds of audio per second. Qwen was faster in these workloads, while Ming used less GPU memory and shares one checkpoint for design and cloning. First requests after eviction took 2.03 seconds for design and 2.09 seconds for cloning end to end. The Q8 file is 1.62 GiB on disk. All four final Ming clips were transcribed with local Whisper tiny on CPU; the transcripts cover the complete target text. This checks coverage, not comparative voice quality.

**Budget for narration too.** The short Kokoro eviction clip occupied only 0.79 GiB on GPU 0, but synthesizing the full 229-character paragraph raised that entry to 2.63 GiB. One concurrent Ming clone + Kokoro narration check reached **2.08 GiB on GPU 1 plus 2.73 GiB on GPU 0: 4.81 GiB combined**. Ming design/cloning reused the same model entry. These are process allocations with contexts and retained buffers; the experiment used two GPUs, so a shared-GPU budget is a planning estimate, not a single-GPU fit test.

Recommended available VRAM, after allowing for the desktop and other applications:

| Workload on one GPU | Recommended available VRAM | Basis |
|---|---:|---|
| Ming + CPU Kokoro (`--kokoro-cpu`) | **3 GiB** | About 2.08 GiB Ming observed, plus headroom |
| One Qwen checkpoint + CPU Kokoro (`--kokoro-cpu`) | **5 GiB** | 4.05 GiB observed in mixed-launch validation, plus headroom; one-entry GPU cache |
| Default Ming design/cloning + one GPU Kokoro narrator entry | **6 GiB** | 4.81 GiB measured across two devices, plus headroom |
| One Qwen entry + Kokoro narration | **8 GiB** | Up to 3.96 GiB Qwen + 2.73 GiB Kokoro observed separately, plus headroom |
| VoiceDesign + 1.7B Base cached together | **9 GiB** | 6.36 GiB mixed-output pair observed; extra room for both models' longer-output buffers |

These are recommended starting budgets for the tested passages, not hard limits or guarantees for arbitrary output lengths. For the supplied split placement, budget **3 GiB available on GPU 1 for Ming and 4 GiB on GPU 0 for Kokoro**; Qwen and its resident pairs need larger budgets on GPU 1. To use one GPU, set every model's `device` to its index. The native cache permits two entries total, so switching among Ming, Qwen, offline narration, and streaming narration can cause eviction and reloads. Available memory on different cards cannot be pooled for a model.

Reproduce the matched Ming test after the Qwen benchmark has saved its reference:

```bash
.venv/bin/python benchmark_audiocpp_qwen.py --suite ming \
  --reference out/bench/audiocpp_qwen.reference.wav --server-pid <PID>
```

Results, reference SHA-256, coverage transcripts, and the additional concurrent memory check are recorded in [`benchmarks/audiocpp-ming-2026-10-22.json`](benchmarks/audiocpp-ming-2026-10-22.json). Raw request/memory samples and generated WAVs remain under `out/bench/audiocpp_ming*`. The concurrent check was a single memory workload, not a repeated latency comparison. As with Qwen, sampling may miss brief peaks and other applications were resident on the GPUs.

### Kokoro on CPU (2026-10-22)

**Native CPU Kokoro remains faster than playback.** A separate audio.cpp process with `backend=cpu`, four threads, the same Kokoro Q8 file, `af_heart`, and seed 1895 was tested on the Ryzen 7 5800X3D. The process had only offline and streaming Kokoro entries and `max_loaded_models=1`. Each workload used one warmup and three measured runs; other benchmark/Whisper work had finished before timing.

| CPU workload | Native generation | Audio duration | RTF |
|---|---:|---:|---:|
| Same 68-character short line | 1.56 s | 4.50 s | 0.348 |
| Same 229-character paragraph | 4.88 s | 13.53 s | 0.361 |

That is about **2.8× faster than playback** for the paragraph. First native PCM arrived in **4.86 seconds** with the default 240-character text chunk budget, because this paragraph fit in one synthesis chunk. Setting `options: {"text_chunk_size": 80}` reduced median first PCM to **0.68 seconds**; the complete 14.48-second stream finished in **5.04 seconds**. Splitting changes the generated durations and phrasing. These are native HTTP/SSE timings without facade overhead and use full stream consumption, not cancellation after the first event.

CPU Kokoro does not need GPU model/workspace allocations, so it can spare the roughly 2.73 GiB observed for GPU paragraph narration. **Use `./run-local.sh --kokoro-cpu` for CPU narration alongside CUDA character synthesis.** The launcher starts separate native CPU/CUDA processes and configures `backend_routes` automatically. Budget 3 GiB available for Ming or 5 GiB for one Qwen checkpoint in this mode. Changing the existing native config's global `backend` to `cpu` instead moves all character synthesis to CPU; use the launch flag to keep it on GPU.

To reproduce, launch a separate native instance on port 17863 using a copy of the local config with `backend=cpu`, `threads=4`, only the two Kokoro entries, no per-entry GPU device overrides, and `max_loaded_models=1`. Then run:

```bash
.venv/bin/python benchmark_audiocpp_kokoro_cpu.py --url http://127.0.0.1:17863
```

Recorded requests and summaries: [`benchmarks/audiocpp-kokoro-cpu-2026-10-22.json`](benchmarks/audiocpp-kokoro-cpu-2026-10-22.json). This measures the native CPU path; the older Python Kokoro timing and first-audio figures are historical.

### Model labels and template-specific controls

The UI labels Qwen profiles by both roles: **design 1.7B / clone 0.6B**, **design 1.7B / clone 1.7B**, and **CustomVoice 1.7B (presets)**. The first two share one VoiceDesign checkpoint; only their Base cloning checkpoint changes. Labels and descriptions live in each profile's `display_name` and `description`, while HTTP model IDs stay unchanged.

Ming's native controls belong to particular templates: `voice_id` uses `ip`; categorical `speed`, `pitch`, and `volume` use `basic`; `emotion` uses `emotion`; and `dialect` uses `dialect`. The `voice_design` template uses the voice description instead. A built-in voice ID is not an extra attribute to combine with a newly designed voice.

Use the Ming **Presets** tab and enter a built-in voice such as `US_Emma` or `UK_George`, or enable its Voice ID override. The preset defaults select the `ip` template and map the selected speaker into native `voice_id`. Advanced users can explicitly choose `ip` in the template control; the voice description is then hidden and omitted because that template uses structured controls. Changing templates disables incompatible overrides rather than forwarding them. Incompatible additional JSON options receive a clear client error, and the Python adapter also rejects them with HTTP 400 before calling audio.cpp.

These rules are configuration-driven: `preset_options` supplies preset-mode defaults, `voice_option` maps preset speakers, `option_dependencies` declares allowed combinations, and `instruction_conditions` identifies templates that accept descriptions. They apply to both the UI and adapter without a hardcoded model-family branch.

## Adding a model to the shared installer

The installer reads [model_catalog.json](model_catalog.json). Add an entry with
its native `family`, public `profiles`, pinned download `repo`/`revision`,
`local_dir`, native-ID-to-file `weights`, and download `include`/`exclude` patterns.
`prefer_cpu` makes `--kokoro-cpu` route that family to the CPU worker. Add the
corresponding native entries in `audiocpp-server.local.json` and public profiles
in `backend.json`. If the family needs newer native code, update the catalog's
pinned source revision as well. Existing custom source checkouts are not reset
by setup; update those deliberately or select a fresh source location.

The shared installer derives compiled families, downloads, profile filtering,
paths, aliases, and CPU routing from this catalog. Do not add a model-specific
installer or launch command. Verify every catalog selection with
`tests/test_setup_models.py`, preview with `./setup.sh --add NAME --plan`, then
validate live synthesis for that family. Future models with additional sidecar
files should include them in the catalog's download patterns.

Setup-owned configs are under `.cache/install/`; hand-maintained configs should
be copied outside that directory and selected through `AUDIOCPP_CONFIG` and
`SPEECH_CONFIG`. Rerunning setup may regenerate setup-owned configs. Downloads
and builds complete before new configs are written, so a failed setup keeps the
previous enabled selection.
