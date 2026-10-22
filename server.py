"""OpenAI-compatible speech facade backed entirely by audio.cpp."""
import argparse
import asyncio
import io
import logging
import os
import re
import secrets
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal
import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.background import BackgroundTask
from audiocpp_backend import AudioCppBackend
from audio_utils import resample
from backend_config import ServerConfig, load_config
from npc_tts import SpeechEngine
from sillynpc_bridge import decode_sillynpc_voice

log = logging.getLogger("speech.server")
MEDIA_TYPES = {"wav":"audio/wav", "mp3":"audio/mpeg", "flac":"audio/flac", "pcm":"application/octet-stream", "opus":"audio/ogg", "aac":"audio/aac"}

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str
    input: str = Field(min_length=1, max_length=16384)
    voice: str | None = Field(default=None, max_length=4096)
    instructions: str | None = Field(default=None, max_length=4096)
    response_format: Literal["wav","mp3","flac","pcm","opus","aac"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["audio"] = "audio"
    stream: bool = False
    context_before: str = Field(default="", max_length=4096)
    context_after: str = Field(default="", max_length=4096)
    language: str = Field(default="English", min_length=1, max_length=64)
    seed: int | None = Field(default=None, ge=0, le=2**64-1)
    temperature: float | None = Field(default=None, ge=0)
    max_new_tokens: int | None = Field(default=None, ge=1)
    npc_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input", "voice")
    @classmethod
    def not_blank(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Must not be blank")
        return value

    @field_validator("options")
    @classmethod
    def finite_options(cls, value):
        import json
        json.dumps(value, allow_nan=False)
        return value

def encode_audio(wav: np.ndarray, sample_rate: int, fmt: str, speed: float) -> bytes:
    if speed != 1.0 or fmt in {"opus", "aac"}:
        # ffmpeg applies tempo changes without shifting pitch and encodes formats
        # that are not consistently supported by libsndfile.
        source = io.BytesIO()
        sf.write(source, wav, sample_rate, format="WAV", subtype="PCM_16")
        codec, container = {
            "wav": ("pcm_s16le", "wav"), "pcm": ("pcm_s16le", "s16le"),
            "flac": ("flac", "flac"), "mp3": ("libmp3lame", "mp3"),
            "opus": ("libopus", "ogg"), "aac": ("aac", "adts"),
        }[fmt]
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0"]
        if speed != 1.0:
            # Each stage stays in [0.5, 2], including speed=0.25 and speed=4.
            stage = speed**0.5
            command += ["-af", f"atempo={stage},atempo={stage}"]
        command += ["-ac", "1", "-ar", str(sample_rate), "-c:a", codec, "-f", container, "pipe:1"]
        return subprocess.run(command, input=source.getvalue(), capture_output=True,
                              check=True, timeout=60).stdout
    output = io.BytesIO()
    sf.write(output, wav, sample_rate, format="RAW" if fmt == "pcm" else fmt.upper(),
             subtype="MPEG_LAYER_III" if fmt == "mp3" else "PCM_16",
             **({"endian": "LITTLE"} if fmt == "pcm" else {}))
    return output.getvalue()

def create_app(*, config=None, config_path=None, backend=None, voices_dir="voices", api_key=None, selector=None):
    settings = config if isinstance(config, ServerConfig) else (ServerConfig.model_validate(config) if config is not None else load_config(config_path or Path(__file__).with_name("backend.json")))
    owns_backend = backend is None

    def prepare_worker(engine, npc_id):
        try:
            engine.prepare(npc_id)
        except Exception:
            log.exception("Background reference preparation failed for NPC %s", npc_id)

    def queue_preparation(npc_id):
        future = app.state.preparations.get(npc_id)
        if future is None or future.done():
            future = asyncio.get_running_loop().run_in_executor(app.state.executor, prepare_worker, app.state.engine, npc_id)
            app.state.preparations[npc_id] = future
            future.add_done_callback(lambda done: app.state.preparations.pop(npc_id, None)
                                     if app.state.preparations.get(npc_id) is done else None)
        return future

    async def prepare_after_response(npc_id):
        await asyncio.shield(queue_preparation(npc_id))

    @asynccontextmanager
    async def lifespan(app):
        from context_audio import ContextAudio
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="npc") as executor, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="speech") as speech_executor:
            app.state.executor, app.state.speech_executor = executor, speech_executor
            app.state.backend = backend or AudioCppBackend(settings)
            app.state.engine = SpeechEngine(app.state.backend, settings, voices_dir, selector)
            app.state.context = ContextAudio(settings.alignment_model)
            app.state.preparations = {}
            try:
                native = await asyncio.get_running_loop().run_in_executor(speech_executor, app.state.backend.discover)
                ids = {entry["id"] for entry in native}
                required = {value for profile in settings.models.values() for value in
                            (profile.backend_model, profile.design_model, profile.clone_model, profile.streaming_model) if value}
                if required - ids:
                    raise RuntimeError(f"Models missing from audio.cpp: {sorted(required - ids)}")
                for npc_id in app.state.engine.pending_ids():
                    queue_preparation(npc_id)
                yield
            finally:
                await asyncio.gather(*list(app.state.preparations.values()), return_exceptions=True)
                if owns_backend:
                    app.state.backend.close()

    app = FastAPI(title="Local Speech", lifespan=lifespan)

    async def authenticate(request: Request):
        if api_key and not secrets.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {api_key}".encode()):
            raise HTTPException(401, "Invalid API key")

    from voice_studio import register_studio
    register_studio(app, settings, authenticate, SpeechRequest, encode_audio)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"error": {
            "message": str(exc.detail), "type": "invalid_request_error" if exc.status_code < 500 else "server_error", "param": None, "code": None}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        return JSONResponse(status_code=400, content={"error": {
            "message": error["msg"], "type": "invalid_request_error", "param": ".".join(str(p) for p in error["loc"][1:]), "code": None}})

    @app.get("/health")
    async def health():
        return {"status": "ok", "backend": "audio.cpp", "backend_url": settings.backend_url,
                "output_sample_rate": settings.output_sample_rate}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "local"}
                for name in (*settings.models, *settings.aliases)]}

    @app.get("/v1/backend/models", dependencies=[Depends(authenticate)])
    async def backend_models():
        data = await asyncio.get_running_loop().run_in_executor(app.state.speech_executor, app.state.backend.discover)
        return {"object": "list", "data": data}

    @app.get("/v1/voices", dependencies=[Depends(authenticate)])
    async def voices():
        def discover():
            data = []
            for name, profile in settings.models.items():
                available = set(profile.voice_aliases.values()) | set(profile.voice_descriptions)
                available.update(app.state.backend.voices(profile))
                data.extend({"id": voice, "model": name} for voice in sorted(available))
            return data
        return {"object": "list", "data": await asyncio.get_running_loop().run_in_executor(app.state.speech_executor, discover)}

    @app.get("/v1/voices/{npc_id}", dependencies=[Depends(authenticate)])
    async def voice_status(npc_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", npc_id):
            raise HTTPException(400, "Invalid NPC ID")
        _, metadata = app.state.engine.record(npc_id)
        if metadata is None:
            raise HTTPException(404, "Unknown NPC")
        return {"npc_id": npc_id, "status": app.state.engine.status(npc_id),
                "reference_seconds": metadata.get("reference_seconds"), "error": metadata.get("error")}

    def synthesize(body, model_id, profile):
        context_status = None
        if body.context_before or body.context_after:
            text = body.context_before + body.input + body.context_after
            try:
                audio, rate = app.state.backend.generate(profile, body, text=text)
                audio = app.state.context.cut(audio, rate, body)
                context_status = "applied"
            except (ValueError, RuntimeError, ImportError):
                log.warning("Contextual speech fell back to standalone", exc_info=True)
                audio, rate = app.state.backend.generate(profile, body)
                context_status = "fallback"
        else:
            audio, rate = app.state.engine.generate(body, model_id, profile)
        audio = resample(audio, rate, settings.output_sample_rate)
        encoded = encode_audio(audio, settings.output_sample_rate, body.response_format, 1.0 if profile.native_speed else body.speed)
        status = app.state.engine.status(body.npc_id) if body.npc_id else None
        return encoded, status, context_status

    def next_pcm(iterator):
        audio = next(iterator, None)
        return None if audio is None else encode_audio(audio, settings.output_sample_rate, "pcm", 1.0)

    async def stream_response(body, profile, executor):
        iterator = app.state.backend.stream(profile, body)
        loop = asyncio.get_running_loop()
        try:
            first = await loop.run_in_executor(executor, next_pcm, iterator)
            if first is None:
                raise ValueError("Backend produced no audio")
        except BaseException:
            await loop.run_in_executor(executor, iterator.close)
            raise
        async def chunks():
            yield first
            while True:
                chunk = await loop.run_in_executor(executor, next_pcm, iterator)
                if chunk is None:
                    return
                yield chunk
        class AudioStream(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    await self.body_iterator.aclose()
                    await loop.run_in_executor(executor, iterator.close)
        return AudioStream(chunks(), media_type=MEDIA_TYPES["pcm"], headers={
            "X-Audio-Sample-Rate": str(settings.output_sample_rate), "X-Audio-Channels": "1"})

    @app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
    async def speech(body: SpeechRequest):
        try:
            body = decode_sillynpc_voice(body)
            model_id, profile = settings.resolve(body.model)
            if len(body.input) > profile.max_input_chars:
                raise ValueError(f"Input exceeds this profile's {profile.max_input_chars}-character limit")
            contextual = bool(body.context_before or body.context_after)
            if contextual and (not profile.context or body.stream or body.npc_id):
                raise ValueError("Context requires a non-streaming context-enabled profile without npc_id")
            if contextual and len(body.context_before + body.input + body.context_after) > profile.context_max_chars:
                raise ValueError("Context plus input exceeds the configured context limit")
            if body.npc_id and not profile.clone_model:
                raise ValueError("This profile does not support NPC cloning")
            if body.stream and (not profile.streaming_model or body.npc_id or body.response_format != "pcm"):
                raise ValueError("Streaming requires PCM, a streaming-enabled profile, and no npc_id")
            if ((body.speed != 1 and not profile.native_speed) or body.response_format in {"opus", "aac"}) and not shutil.which("ffmpeg"):
                raise ValueError("This format or speed requires ffmpeg on PATH")
            executor = app.state.executor if profile.clone_model else app.state.speech_executor
            if body.stream:
                return await stream_response(body, profile, executor)
            audio, status, context_status = await asyncio.get_running_loop().run_in_executor(executor, synthesize, body, model_id, profile)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        except Exception as error:
            log.exception("Speech generation failed")
            raise HTTPException(502, "Speech backend failed; see server logs") from error
        headers = {}
        if status:
            headers["X-Voice-Status"] = status
        if context_status:
            headers["X-Narration-Context"] = context_status
        return Response(audio, media_type=MEDIA_TYPES[body.response_format], headers=headers,
                        background=BackgroundTask(prepare_after_response, body.npc_id) if status == "preparing" else None)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--config", default=str(Path(__file__).with_name("backend.json")))
    parser.add_argument("--voices-dir", default="voices")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    uvicorn.run(create_app(config_path=args.config, voices_dir=args.voices_dir, api_key=os.environ.get("TTS_API_KEY")), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
