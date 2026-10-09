"""Local OpenAI-compatible Qwen3-TTS speech server."""
import argparse
import asyncio
import io
import logging
import os
import secrets
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_NAMES = {"qwen3-tts", MODEL_ID, "tts-1", "tts-1-hd", "gpt-4o-mini-tts"}
VOICES = {
    "alloy": "A warm, clear narrator with a neutral American accent.",
    "ash": "A calm male voice with a clear, conversational delivery.",
    "ballad": "A gentle male storyteller with expressive, measured delivery.",
    "coral": "A bright, friendly female voice with an expressive delivery.",
    "echo": "A smooth male voice with a relaxed conversational delivery.",
    "fable": "An expressive storyteller with a British accent.",
    "onyx": "A deep male narrator with a rich baritone and measured delivery.",
    "nova": "An energetic young woman with a clear, upbeat delivery.",
    "sage": "A calm, gentle voice with a thoughtful, measured delivery.",
    "shimmer": "A soft, warm female voice with a clear delivery.",
    "verse": "An expressive, lively voice with a conversational delivery.",
    "marin": "A warm female narrator with a clear, natural delivery.",
    "cedar": "A warm male narrator with a clear, natural delivery.",
}
MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac",
               "pcm": "application/octet-stream", "opus": "audio/ogg", "aac": "audio/aac"}
LANGUAGES = {"auto", "english", "chinese", "japanese", "korean", "german", "french",
             "russian", "portuguese", "spanish", "italian"}
log = logging.getLogger("qwen3-tts.server")


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    input: str = Field(min_length=1, max_length=4096)
    voice: str = Field(default="alloy", min_length=1, max_length=4096)
    instructions: str | None = Field(default=None, max_length=4096)
    response_format: Literal["wav", "mp3", "flac", "pcm", "opus", "aac"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["audio"] = "audio"
    # Qwen-specific extensions; pass through extra_body with the OpenAI SDK.
    language: str = "English"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    temperature: float = Field(default=0.9, gt=0, le=2)
    max_new_tokens: int = Field(default=2048, ge=2, le=4096)

    @field_validator("input", "voice")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must not be blank")
        return value

    @field_validator("language")
    @classmethod
    def known_language(cls, value: str) -> str:
        if value.lower() not in LANGUAGES:
            raise ValueError("Unsupported language")
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


def create_app(*, device="cuda:0", model=None, max_seq_len=4096, api_key=None, loader=None):
    @asynccontextmanager
    async def lifespan(app):
        # Load and generate on one dedicated thread. CUDA graphs and their caches
        # must never be used by concurrent requests.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts") as executor:
            app.state.executor = executor
            load_model = loader
            if load_model is None:
                import fast_tts
                load_model = fast_tts.load
            app.state.tts = await asyncio.get_running_loop().run_in_executor(
                executor, partial(load_model, model_id=model, device=device, max_seq_len=max_seq_len))
            log.info("Loaded speech model: %s", app.state.tts.plan)
            yield
            app.state.tts = None

    app = FastAPI(title="Qwen3-TTS", lifespan=lifespan)

    async def authenticate(request: Request):
        if api_key and not secrets.compare_digest(
            request.headers.get("authorization", "").encode(), f"Bearer {api_key}".encode()
        ):
            raise HTTPException(401, "Invalid API key")

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(status_code=exc.status_code, content={"error": {
            "message": str(exc.detail), "type": "invalid_request_error" if exc.status_code < 500 else "server_error",
            "param": None, "code": None}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        return JSONResponse(status_code=400, content={"error": {
            "message": error["msg"], "type": "invalid_request_error",
            "param": ".".join(str(part) for part in error["loc"][1:]), "code": None}})

    @app.get("/health")
    async def health():
        return {"status": "ok", "device": device}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models():
        return {"object": "list", "data": [{"id": "qwen3-tts", "object": "model",
                                              "created": 0, "owned_by": "local"}]}

    def synthesize(body: SpeechRequest):
        import torch
        tts = app.state.tts
        instruct = body.instructions or VOICES.get(body.voice, body.voice)
        with torch.inference_mode():
            if body.seed is not None:
                torch.manual_seed(body.seed)
            if device.startswith("cuda"):
                torch.cuda.set_device(device)
                # Validate the actual prefill length before writing static KV caches.
                prepared = tts._prepare_generation_custom(
                    text=body.input, language=body.language, speaker=None,
                    instruct=instruct, non_streaming_mode=True)
                prompt_length = prepared[3].shape[1]
                del prepared
                if prompt_length + body.max_new_tokens >= max_seq_len:
                    raise HTTPException(400, "Prompt plus max_new_tokens exceeds --max-seq-len. "
                                        "Lower max_new_tokens or start the server with a larger sequence budget.")
            started = time.perf_counter()
            wavs, sr = tts.generate_voice_design(
                text=body.input, instruct=instruct, language=body.language,
                temperature=body.temperature, max_new_tokens=body.max_new_tokens)
            wav = np.asarray(wavs[0])
            if wav.size == 0 or not np.isfinite(wav).all():
                raise RuntimeError("Model produced empty or invalid audio")
            log.info("Generated %.2fs of audio in %.2fs", len(wav) / sr, time.perf_counter() - started)
            return encode_audio(wav, sr, body.response_format, body.speed)

    @app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
    async def speech(body: SpeechRequest):
        if body.model not in MODEL_NAMES:
            raise HTTPException(400, "Unknown model. Use qwen3-tts.")
        if (body.speed != 1.0 or body.response_format in {"opus", "aac"}) and not shutil.which("ffmpeg"):
            raise HTTPException(400, "This format or speed requires ffmpeg on PATH.")
        try:
            audio = await asyncio.get_running_loop().run_in_executor(app.state.executor, synthesize, body)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("Speech generation failed")
            raise HTTPException(500, "Speech generation failed; see server logs.") from exc
        return Response(audio, media_type=MEDIA_TYPES[body.response_format])

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()
    if args.max_seq_len < 128:
        parser.error("--max-seq-len must be at least 128")
    logging.basicConfig(level=logging.INFO)
    import uvicorn
    uvicorn.run(create_app(device=args.device, model=args.model, max_seq_len=args.max_seq_len,
                           api_key=os.environ.get("TTS_API_KEY")), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
