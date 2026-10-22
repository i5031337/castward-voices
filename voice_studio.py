"""Same-origin voice workbench. Inference stays in the configured native backend."""
import asyncio
import base64
import binascii
import io
import logging
import shutil
import time
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from audio_utils import mono_audio, resample

MAX_REFERENCE_BYTES = 5 * 1024 * 1024
log = logging.getLogger("speech.studio")


def decode_reference(encoded):
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_REFERENCE_BYTES:
            raise ValueError("Reference audio must be between 1 byte and 5 MiB")
        with sf.SoundFile(io.BytesIO(raw)) as source:
            # Bound decoded memory as well as compressed uploads.
            if source.frames > source.samplerate * 120 or source.channels > 8:
                raise ValueError("Reference audio must be at most two minutes and eight channels")
            if source.frames * 2 + 128 > MAX_REFERENCE_BYTES:
                raise ValueError("Decoded reference is too large; use a shorter clip")
            rate = source.samplerate
            audio = source.read(dtype="float32")
        audio = mono_audio(audio, rate)
        if np.max(np.abs(audio)) < 0.001:
            raise ValueError("Reference audio is silent")
        # The adapter sends PCM WAV to native, whose decoded upload limit is 5 MiB.
        if len(audio) * 2 + 128 > MAX_REFERENCE_BYTES:
            raise ValueError("Decoded reference is too large; use a shorter clip")
        return audio, rate
    except (binascii.Error, sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("Cannot read reference audio; use a valid WAV, FLAC, Ogg, or MP3 clip") from exc


def register_studio(app, settings, authenticate, SpeechRequest, encode_audio):
    assets = Path(__file__).with_name("webui")
    app.mount("/studio-assets", StaticFiles(directory=assets), name="studio-assets")

    @app.get("/", include_in_schema=False)
    async def studio_page():
        return FileResponse(assets / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/ui/config", dependencies=[Depends(authenticate)])
    async def studio_config():
        def discover():
            native = app.state.backend.discover()
            profiles = []
            for name, profile in settings.models.items():
                data = profile.model_dump()
                data["seed_max"] = str(profile.seed_max)
                data["id"] = name
                data["voices"] = app.state.backend.voices(profile)
                profiles.append(data)
            return {"profiles": profiles, "native_models": native,
                    "default_model": settings.default_model,
                    "output_sample_rate": settings.output_sample_rate,
                    "max_reference_bytes": MAX_REFERENCE_BYTES}
        try:
            return await asyncio.get_running_loop().run_in_executor(app.state.speech_executor, discover)
        except Exception as exc:
            log.exception("Studio discovery failed")
            raise HTTPException(502, "Cannot discover audio.cpp models; see server logs") from exc

    class StudioRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        mode: Literal["speech", "design", "clone"]
        speech: SpeechRequest
        reference_audio: str | None = Field(default=None, max_length=4 * ((MAX_REFERENCE_BYTES + 2) // 3))
        reference_text: str = Field(default="", max_length=16384)

    @app.post("/ui/generate", dependencies=[Depends(authenticate)])
    async def studio_generate(body: StudioRequest):
        request = body.speech
        try:
            model_id, profile = settings.resolve(request.model)
            if ("voice_id" in request.options or any(key in request.options for key in profile.instruction_conditions)) and not request.voice:
                profile = profile.model_copy(update={"default_voice": None})
            effective = profile.options_for(body.mode, request.options)
            if request.instructions and profile.instruction_option:
                effective[profile.instruction_option] = request.instructions
            profile.validate_options(effective)
            if request.npc_id or request.stream:
                raise ValueError("The studio uses standalone clips; NPC persistence and streaming use /v1/audio/speech")
            if len(request.input) > profile.max_input_chars:
                raise ValueError(f"Input exceeds the profile's {profile.max_input_chars}-character limit")
            if body.mode == "design" and not profile.design_model:
                raise ValueError("This profile does not support voice design")
            if body.mode == "design" and profile.accepts_instruction(effective) and (not request.instructions or not request.instructions.strip()):
                raise ValueError("Voice design needs a description for this template")
            if body.mode == "clone" and (not profile.clone_model or not body.reference_audio):
                raise ValueError("Voice cloning needs a clone-capable model and reference audio")
            if body.mode != "clone" and body.reference_audio:
                raise ValueError("Reference audio is only used in clone mode")
            contextual = bool(request.context_before or request.context_after)
            if contextual and (body.mode != "speech" or not profile.context):
                raise ValueError("Context is only available for a context-enabled speech profile")
            if contextual and len(request.context_before + request.input + request.context_after) > profile.context_max_chars:
                raise ValueError("Context plus input exceeds the profile's configured context limit")
            if ((request.speed != 1 and not profile.native_speed) or request.response_format in {"opus", "aac"}) and not shutil.which("ffmpeg"):
                raise ValueError("This format or speed requires ffmpeg on PATH")

            def generate():
                started = time.perf_counter()
                kwargs = {"mode": body.mode}
                if body.mode == "design":
                    kwargs["instruction"] = request.instructions
                elif body.mode == "clone":
                    kwargs.update(reference=decode_reference(body.reference_audio), reference_text=body.reference_text)
                    if request.instructions:
                        kwargs["instruction"] = request.instructions
                context_status = None
                if contextual:
                    try:
                        audio, rate = app.state.backend.generate(profile, request,
                            text=request.context_before + request.input + request.context_after, **kwargs)
                        audio = app.state.context.cut(audio, rate, request)
                        context_status = "applied"
                    except (ValueError, RuntimeError, ImportError):
                        audio, rate = app.state.backend.generate(profile, request, **kwargs)
                        context_status = "fallback"
                elif body.mode == "speech":
                    audio, rate = app.state.engine.generate(request, model_id, profile)
                else:
                    audio, rate = app.state.backend.generate(profile, request, **kwargs)
                native_duration = len(audio) / rate
                reference = encode_audio(audio, rate, "wav", 1)
                converted = resample(audio, rate, settings.output_sample_rate)
                tempo = 1 if profile.native_speed else request.speed
                preview = encode_audio(converted, settings.output_sample_rate, "wav", tempo)
                download = preview if request.response_format == "wav" else encode_audio(converted, settings.output_sample_rate, request.response_format, tempo)
                elapsed = time.perf_counter() - started
                return {"audio": base64.b64encode(preview).decode(),
                        "download": base64.b64encode(download).decode(), "format": request.response_format,
                        "reference_audio": base64.b64encode(reference).decode(),
                        "transcript": request.input, "model": request.model, "mode": body.mode,
                        "duration": native_duration / tempo, "elapsed": elapsed,
                        "realtime_factor": elapsed / (native_duration / tempo),
                        "native_sample_rate": rate, "sample_rate": settings.output_sample_rate,
                        "context_status": context_status}

            executor = app.state.executor if profile.clone_model else app.state.speech_executor
            return await asyncio.get_running_loop().run_in_executor(executor, generate)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("Studio generation failed")
            raise HTTPException(502, "Speech backend failed; see server logs") from exc
