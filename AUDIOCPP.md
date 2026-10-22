# audio.cpp speech server

All synthesis runs in audio.cpp. The Python API handles NPC persistence, reference selection, the SillyNPC bridge, audio encoding, and client compatibility. It does not import the Qwen or Kokoro Python synthesis runtimes.

## Run locally

The existing virtual environment can run the new server. A fresh Python 3.11 environment needs `uv pip install -r requirements.lock.txt`; install FFmpeg separately for tempo changes, AAC, and Opus. Optional CPU word alignment needs `uv pip install -r requirements-alignment.txt` and the local Whisper checkpoint (`python download_alignment.py`).

```bash
./run-local.sh --port 8001
```

This starts audio.cpp on port 7862 if it is not already healthy, then the compatibility API on port 8001. The launcher stops only processes it started. It uses the native binary built in `../Ming/audio.cpp/build/ming-cuda/bin/audiocpp_server`, with Ming Q8 on GPU 1 and Kokoro Q8 on GPU 0. Offline and streaming Kokoro are separate native model entries and can consume separate memory. Change paths, devices, residency, and threads in `audiocpp-server.local.json` for other machines. Build audio.cpp with both `ming3_tts` and `kokoro_tts` enabled.

Environment overrides: `PYTHON`, `AUDIOCPP_SERVER`, `AUDIOCPP_CONFIG`, `AUDIOCPP_CUDA_LIB`, and `SPEECH_CONFIG`. Native logs go to `audiocpp-backend.log`. `TTS_API_KEY` still enables bearer authentication on the Python API. A separately hosted native service can be configured with `backend_url` and `backend_api_key_env` in `backend.json`; it can also be used directly with `python server.py --config backend.json`.

## Replace models through configuration

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


## Voice Atelier web UI

Start the server with `./run-local.sh --port 8001`, then open <http://127.0.0.1:8001/>. The UI is served by the same Python API; it needs no Node build, CDN, or external assets. API bearer keys can be entered through the connection settings and are held in this tab's session storage. API routes continue to require authentication when configured, including the studio routes.

Choose a model and **Design**, **Clone**, or **Presets**. Capabilities come from the configured public profiles, and the active mode's native model publishes its generation controls through discovery. Unsupported modes are disabled. Every advertised native request option is available through a typed control or its dedicated language, instruction, transcript, seed, or speed field. Overrides are opt-in: untouched fields retain configured/native defaults. Additional JSON options can override controls and pass through options added by newer backends. Session/load options remain deployment settings in the audio.cpp configuration. Preset fields accept native voice IDs, including all IDs advertised in native enum metadata.

Each take has a waveform, player, download, saved request settings, generation time, audio duration, sample-rate information, and real-time throughput. **Use as reference** moves the original native-rate WAV and its transcript into Clone mode; choose new text and generate. References can be reused across clone-capable profiles. Output format and playback tempo do not degrade the source recording kept for cloning. A replacement architecture can render the same reference differently.

You can also drop or upload WAV, FLAC, Ogg, or MP3 audio. References are limited to 5 MiB, two minutes, and eight source channels, with an additional bound on decoded PCM size. Individual models can have smaller limits; a clean 3–10 second clip is a useful starting point. Edit the reference transcript if a designed take skipped or changed words; leaving it blank requests speaker-only conditioning where supported.

The workbench keeps up to eight takes in browser memory, with independent design/clone option drafts. Download recordings you want to keep before refreshing. These experiments do not create or modify saved NPC records. WAV previews play regardless of download format. Context-aware preset narration is available on profiles that enable it.

Implementation: `voice_studio.py`, `webui/`, and a registration hook in `server.py`. `GET /ui/config` combines profiles and native metadata; `POST /ui/generate` validates capabilities, decodes references, and uses the existing serialized workers and native adapter. The OpenAI-compatible speech and NPC endpoints are unchanged.


## Qwen models in Voice Atelier

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


### Model labels and template-specific controls

The UI labels Qwen profiles by both roles: **design 1.7B / clone 0.6B**, **design 1.7B / clone 1.7B**, and **CustomVoice 1.7B (presets)**. The first two share one VoiceDesign checkpoint; only their Base cloning checkpoint changes. Labels and descriptions live in each profile's `display_name` and `description`, while HTTP model IDs stay unchanged.

Ming's native controls belong to particular templates: `voice_id` uses `ip`; categorical `speed`, `pitch`, and `volume` use `basic`; `emotion` uses `emotion`; and `dialect` uses `dialect`. The `voice_design` template uses the voice description instead. A built-in voice ID is not an extra attribute to combine with a newly designed voice.

Use the Ming **Presets** tab and enter a built-in voice such as `US_Emma` or `UK_George`, or enable its Voice ID override. The preset defaults select the `ip` template and map the selected speaker into native `voice_id`. Advanced users can explicitly choose `ip` in the template control; the voice description is then hidden and omitted because that template uses structured controls. Changing templates disables incompatible overrides rather than forwarding them. Incompatible additional JSON options receive a clear client error, and the Python adapter also rejects them with HTTP 400 before calling audio.cpp.

These rules are configuration-driven: `preset_options` supplies preset-mode defaults, `voice_option` maps preset speakers, `option_dependencies` declares allowed combinations, and `instruction_conditions` identifies templates that accept descriptions. They apply to both the UI and adapter without a hardcoded model-family branch.
