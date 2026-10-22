"""Model-independent HTTP adapter; inference and model residency belong to audio.cpp."""
import base64
import io
import json
import os
import threading

import httpx
import numpy as np
import soundfile as sf
from audio_utils import mono_audio, resample


class AudioCppBackend:
    def __init__(self, config, *, transport=None):
        self.config = config
        headers = {}
        if config.backend_api_key_env:
            key = os.environ.get(config.backend_api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        self.client = httpx.Client(base_url=config.backend_url.rstrip("/"), timeout=config.backend_timeout,
                                   headers=headers, transport=transport)
        self._locks = {}
        self._locks_guard = threading.Lock()

    def close(self):
        self.client.close()

    def _lock(self, model):
        with self._locks_guard:
            return self._locks.setdefault(model, threading.RLock())

    @staticmethod
    def _check(response):
        if response.is_success:
            return
        try:
            message = response.json().get("error", "audio.cpp request failed")
            if isinstance(message, dict):
                message = message.get("message", str(message))
        except (ValueError, AttributeError):
            message = "audio.cpp request failed"
        if 400 <= response.status_code < 500 and response.status_code not in {401, 403, 429}:
            raise ValueError(str(message))
        raise RuntimeError(f"audio.cpp returned HTTP {response.status_code}: {message}")

    def discover(self):
        response = self.client.get("/v1/models", params={"include_params": "true"})
        self._check(response)
        return response.json().get("data", [])

    def voices(self, profile):
        response = self.client.get("/v1/audio/voices", params={"model": profile.backend_model})
        self._check(response)
        return response.json().get("voices", [])

    def payload(self, profile, body, *, instruction=None, reference=None, reference_text=None,
                mode="speech", text=None, seed=None, options=None):
        model = {"design": profile.design_model, "clone": profile.clone_model,
                 "stream": profile.streaming_model}.get(mode, profile.backend_model)
        if not model:
            raise ValueError(f"This model does not support {mode}")
        opts = profile.options_for(mode)
        language = body.language
        language = next((value for key, value in profile.language_map.items()
                         if key.casefold() == language.casefold()), language)
        if profile.language_option and language is not None:
            opts[profile.language_option] = language
        if instruction:
            if not profile.instruction_option:
                raise ValueError("This model does not accept voice instructions")
            opts[profile.instruction_option] = instruction
        opts.update(body.options)
        if options:
            opts.update(options)
        # Only explicit sampling parameters are sent; each model keeps its defaults.
        for attribute, key in (("temperature", "temperature"), ("max_new_tokens", "max_tokens")):
            value = getattr(body, attribute, None)
            if value is not None:
                opts[key] = value
        request = {"model": model, "input": body.input if text is None else text,
                   "response_format": "wav", "options": opts}
        if profile.native_speed:
            request["speed"] = body.speed
        chosen_seed = body.seed if seed is None else seed
        if chosen_seed is not None:
            if not 0 <= chosen_seed <= profile.seed_max:
                raise ValueError(f"Seed must be between 0 and {profile.seed_max} for this profile")
            # Native JSON numbers use doubles; preserve all supported uint64 bits.
            request["seed"] = str(chosen_seed)
        voice = body.voice or (profile.default_voice if mode in {"speech", "stream"} else None)
        if voice and not instruction and reference is None:
            if profile.voice_option:
                opts.setdefault(profile.voice_option, profile.voice_aliases.get(voice, voice))
            else:
                request["voice"] = profile.voice_aliases.get(voice, voice)
        if reference is not None:
            waveform, rate = reference
            stream = io.BytesIO()
            sf.write(stream, waveform, rate, format="WAV", subtype="PCM_16")
            request["voice_ref"] = {"type": "base64", "data": base64.b64encode(stream.getvalue()).decode("ascii")}
            if reference_text:
                opts[profile.reference_text_option] = reference_text
            elif profile.speaker_only_option:
                opts.setdefault(profile.speaker_only_option, True)
        profile.validate_options(opts)
        return request

    def generate(self, profile, body, **kwargs):
        request = self.payload(profile, body, **kwargs)
        with self._lock(request["model"]):
            response = self.client.post("/v1/audio/speech", json=request)
            self._check(response)
            audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        return mono_audio(audio, rate), rate

    def stream(self, profile, body):
        if not profile.native_speed and body.speed != 1:
            raise ValueError("This streaming profile does not support speed changes")
        request = self.payload(profile, body, mode="stream")
        request.update(response_format="pcm", stream_format="sse", speed=body.speed)
        with self._lock(request["model"]), self.client.stream("POST", "/v1/audio/speech", json=request) as response:
            if not response.is_success:
                response.read()
                self._check(response)
            data = []
            for line in response.iter_lines():
                if line.startswith("data:"):
                    data.append(line[5:].lstrip())
                    continue
                if line or not data:
                    continue
                event_text = "\n".join(data)
                data.clear()
                if event_text == "[DONE]":
                    return
                event = json.loads(event_text)
                if event.get("type") == "error" or "error" in event:
                    raise RuntimeError(f"audio.cpp stream failed: {event.get('error', event)}")
                if event.get("type") == "speech.audio.delta":
                    raw = base64.b64decode(event["audio"], validate=True)
                    if len(raw) % (2 * profile.streaming_channels):
                        raise RuntimeError("audio.cpp returned incomplete PCM samples")
                    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
                    if profile.streaming_channels > 1:
                        audio = audio.reshape(-1, profile.streaming_channels)
                    yield resample(audio, profile.streaming_sample_rate, self.config.output_sample_rate)
            raise RuntimeError("audio.cpp stream ended before [DONE]")
