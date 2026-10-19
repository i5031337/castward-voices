"""Local OpenAI-compatible Qwen3-TTS speech server."""
import argparse
import asyncio
import io
import logging
import os
import re
import secrets
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from npc_tts import CLONE_MODEL_ID, LOCAL_CLONE_MODEL, SpeechEngine
from narrator_tts import LOCAL_MODEL as LOCAL_KOKORO_MODEL, MODEL_ID as KOKORO_MODEL_ID
from sillynpc_bridge import decode_sillynpc_voice
from starlette.background import BackgroundTask

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
MODEL_NAMES = {"qwen3-tts", MODEL_ID, "tts-1", "tts-1-hd", "gpt-4o-mini-tts", "qwen3-tts-0.6b", CLONE_MODEL_ID}
KOKORO_NAMES = {"kokoro", KOKORO_MODEL_ID}
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
    input: str = Field(min_length=1, max_length=16384)
    voice: str = Field(default="alloy", min_length=1, max_length=4096)
    instructions: str | None = Field(default=None, max_length=4096)
    response_format: Literal["wav", "mp3", "flac", "pcm", "opus", "aac"] = "mp3"
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    stream_format: Literal["audio"] = "audio"
    stream: bool = False  # Local extension: Kokoro PCM chunks during synthesis.
    context_before: str = Field(default="", max_length=240)
    context_after: str = Field(default="", max_length=240)
    # Qwen-specific extensions; pass through extra_body with the OpenAI SDK.
    language: str = "English"
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    temperature: float = Field(default=0.9, gt=0, le=2)
    max_new_tokens: int = Field(default=2048, ge=2, le=4096)
    npc_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")

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


def create_app(*, device="cuda:0", model=None, max_seq_len=4096, api_key=None, loader=None,
               clone_model=str(LOCAL_CLONE_MODEL), voices_dir="voices", selector=None,
               kokoro_enabled=True, kokoro_loader=None, kokoro_model_dir=LOCAL_KOKORO_MODEL,
               kokoro_device="cpu", clone_device=None, keep_models_loaded=False):
    def prepare_worker(engine, npc_id):
        try:
            engine.prepare(npc_id)
        except Exception:
            # The first audio was already delivered; readiness/failure is exposed
            # separately and later requests can retry without changing the voice.
            log.exception("Background reference preparation failed for NPC %s", npc_id)

    def queue_preparation(npc_id):
        future = app.state.preparations.get(npc_id)
        if future is None or future.done():
            future = asyncio.get_running_loop().run_in_executor(
                app.state.executor, prepare_worker, app.state.engine, npc_id)
            app.state.preparations[npc_id] = future
            future.add_done_callback(lambda done: app.state.preparations.pop(npc_id, None)
                                     if app.state.preparations.get(npc_id) is done else None)
        return future

    async def prepare_after_response(npc_id):
        # Starlette calls this only after it sends the complete audio response.
        await asyncio.shield(queue_preparation(npc_id))

    @asynccontextmanager
    async def lifespan(app):
        # Load and generate on one dedicated thread. CUDA graphs and their caches
        # must never be used by concurrent requests.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts") as executor, \
                ThreadPoolExecutor(max_workers=1, thread_name_prefix="narrator") as narrator_executor:
            app.state.executor = executor
            # Narration always owns its worker; model transitions alone acquire
            # MODEL_LOAD_LOCK to exclude Kokoro inference during graph capture.
            app.state.narrator_executor = narrator_executor
            app.state.shared_cuda_worker = False
            app.state.narrator = None
            app.state.preparations = {}
            load_model = loader
            if load_model is None:
                import fast_tts
                def load_model(**kwargs):
                    import torch
                    target = torch.device(kwargs["device"])
                    if target.type == "cuda" and target.index is None:
                        target = torch.device("cuda", torch.cuda.current_device())
                    narrator = app.state.narrator
                    reserve = (narrator.workspace_bytes if narrator is not None
                               and narrator.device == target else 0)
                    return fast_tts.load(**kwargs, memory_reserve_bytes=reserve)
            if kokoro_enabled:
                if kokoro_loader is None:
                    from narrator_tts import NarratorEngine
                    load_narrator = NarratorEngine
                else:
                    load_narrator = kokoro_loader
                app.state.narrator = await asyncio.get_running_loop().run_in_executor(
                    app.state.narrator_executor, partial(load_narrator, model_dir=kokoro_model_dir,
                                                        device=kokoro_device))
                log.info("Loaded Kokoro narrator on %s", kokoro_device)
            # Resident Kokoro weights must be present when Qwen plans VRAM use.
            app.state.engine = await asyncio.get_running_loop().run_in_executor(
                executor, partial(SpeechEngine, loader=load_model, design_model=model, device=device,
                                  clone_device=clone_device,
                                  clone_model=clone_model, max_seq_len=max_seq_len, voices_dir=voices_dir,
                                  selector=selector, keep_models_loaded=keep_models_loaded))
            log.info("Loaded speech model: %s", app.state.engine.plan)
            log.info("Narration runs on an independent worker; model loading pauses narration safely")
            # Resume durable records left preparing by a previous process.
            pending = await asyncio.get_running_loop().run_in_executor(
                executor, lambda: list(app.state.engine.pending_ids()))
            for npc_id in pending if clone_model else []:
                queue_preparation(npc_id)
            try:
                yield
            finally:
                await asyncio.gather(*list(app.state.preparations.values()), return_exceptions=True)
            app.state.engine = None
            app.state.narrator = None

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
        return {"status": "ok", "device": device,
                "kokoro_device": kokoro_device if kokoro_enabled else None,
                "keep_models_loaded": keep_models_loaded,
                "shared_cuda_worker": app.state.shared_cuda_worker}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models():
        names = ["qwen3-tts"] + (["qwen3-tts-0.6b"] if clone_model else []) + (["kokoro"] if kokoro_enabled else [])
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0,
                                              "owned_by": "local"} for name in names]}

    @app.get("/v1/voices", dependencies=[Depends(authenticate)])
    async def preset_voices():
        narrator = app.state.narrator
        return {"object": "list", "data": [{"id": voice, "model": "kokoro", "language": "English",
                                             "accent": "American" if voice[0] == "a" else "British"}
                                            for voice in narrator.voices] if narrator else []}

    @app.get("/v1/voices/{npc_id}", dependencies=[Depends(authenticate)])
    async def voice_status(npc_id: str):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", npc_id):
            raise HTTPException(400, "Invalid NPC ID")
        _, metadata = app.state.engine.record(npc_id)
        if metadata is None:
            raise HTTPException(404, "Unknown NPC")
        return {"npc_id": npc_id, "status": app.state.engine.status(npc_id),
                "reference_seconds": metadata.get("reference_seconds"),
                "error": metadata.get("error")}

    def synthesize(body: SpeechRequest):
        import torch
        instruct = body.instructions or (None if body.npc_id else VOICES.get(body.voice, body.voice))
        with torch.inference_mode():
            started = time.perf_counter()
            wavs, sr = app.state.engine.generate(body, instruct)
            wav = np.asarray(wavs[0])
            if wav.size == 0 or not np.isfinite(wav).all():
                raise RuntimeError("Model produced empty or invalid audio")
            log.info("Generated %.2fs of audio in %.2fs", len(wav) / sr, time.perf_counter() - started)
            audio = encode_audio(wav, sr, body.response_format, body.speed)
            status = app.state.engine.status(body.npc_id) if body.npc_id else None
            return audio, status

    def narrate(body):
        started = time.perf_counter()
        status = None
        if body.context_before or body.context_after:
            wav, sr, status = app.state.narrator.generate_context(body)
        else:
            wav, sr = app.state.narrator.generate(body)
        log.info("Kokoro generated %.2fs audio in %.2fs", len(wav) / sr, time.perf_counter() - started)
        # Kokoro applies speed while synthesizing; do not apply it a second time.
        return encode_audio(wav, sr, body.response_format, 1.0), status

    def next_pcm(iterator):
        wav = next(iterator, None)
        return None if wav is None else encode_audio(wav, 24000, "pcm", 1.0)

    async def narrate_response(body):
        if app.state.narrator is None:
            raise HTTPException(400, "Kokoro is disabled on this server")
        app.state.narrator.validate(body)
        loop, executor = asyncio.get_running_loop(), app.state.narrator_executor
        if not body.stream:
            audio, status = await loop.run_in_executor(executor, narrate, body)
            return Response(audio, media_type=MEDIA_TYPES[body.response_format],
                            headers={"X-Narration-Context": status} if status else None)
        if body.response_format != "pcm":
            raise ValueError("Kokoro streaming requires response_format=pcm (24 kHz mono, signed 16-bit little-endian)")
        iterator = app.state.narrator.chunks(body)
        try:
            first = await loop.run_in_executor(executor, next_pcm, iterator)
            if first is None:
                raise ValueError("Input contains no speakable text")
        except BaseException:
            # Closing on the same worker waits for any cancelled, still-running
            # inference before touching the generator.
            await loop.run_in_executor(executor, iterator.close)
            raise

        async def audio_chunks():
            try:
                yield first
                while True:
                    chunk = await loop.run_in_executor(executor, next_pcm, iterator)
                    if chunk is None:
                        break
                    yield chunk
            except Exception:
                log.exception("Kokoro stream failed after audio delivery started")
                raise

        class NarrationStream(StreamingResponse):
            async def __call__(self, scope, receive, send):
                try:
                    await super().__call__(scope, receive, send)
                finally:
                    # ASGI send can fail while the generator is suspended at a
                    # yield. Explicitly close both iterators on disconnection.
                    await self.body_iterator.aclose()
                    await loop.run_in_executor(executor, iterator.close)

        return NarrationStream(audio_chunks(), media_type=MEDIA_TYPES["pcm"],
                               headers={"X-Audio-Sample-Rate": "24000", "X-Audio-Channels": "1"})

    @app.post("/v1/audio/speech", dependencies=[Depends(authenticate)])
    async def speech(body: SpeechRequest):
        try:
            body = decode_sillynpc_voice(body)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        is_kokoro = body.model in KOKORO_NAMES
        if body.model not in MODEL_NAMES and not is_kokoro:
            raise HTTPException(400, "Unknown model. Use qwen3-tts or kokoro.")
        if (body.context_before or body.context_after) and not is_kokoro:
            raise HTTPException(400, "Narration context is supported for Kokoro only")
        if not is_kokoro and len(body.input) > 4096:
            raise HTTPException(400, "Qwen input accepts up to 4096 characters")
        if body.stream and not is_kokoro:
            raise HTTPException(400, "Streaming during generation is supported for Kokoro PCM only")
        if body.model in {"qwen3-tts-0.6b", CLONE_MODEL_ID} and not body.npc_id:
            raise HTTPException(400, "The 0.6B Base model needs npc_id to select a saved voice.")
        if body.npc_id and not clone_model and not is_kokoro:
            raise HTTPException(400, "NPC cloning is disabled on this server.")
        if ((body.speed != 1.0 and not is_kokoro) or body.response_format in {"opus", "aac"}) and not shutil.which("ffmpeg"):
            raise HTTPException(400, "This format or speed requires ffmpeg on PATH.")
        try:
            if is_kokoro:
                return await narrate_response(body)
            audio, status = await asyncio.get_running_loop().run_in_executor(app.state.executor, synthesize, body)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("Speech generation failed")
            raise HTTPException(500, "Speech generation failed; see server logs.") from exc
        return Response(audio, media_type=MEDIA_TYPES[body.response_format],
                        headers={"X-Voice-Status": status} if status else None,
                        background=BackgroundTask(prepare_after_response, body.npc_id)
                        if status == "preparing" else None)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=None)
    parser.add_argument("--clone-model", default=str(LOCAL_CLONE_MODEL), help="Local 0.6B Base model directory")
    parser.add_argument("--clone-device", default=None, help="Device for 0.6B Base; defaults to --device")
    parser.add_argument("--voices-dir", default="voices", help="Persistent NPC voice directory")
    parser.add_argument("--design-only", action="store_true", help="Disable NPC cloning and use only VoiceDesign")
    parser.add_argument("--keep-models-loaded", action="store_true",
                        help="Preload and retain both Qwen models and CUDA graphs; uses more VRAM, avoids swaps")
    parser.add_argument("--no-kokoro", action="store_true", help="Disable the Kokoro narrator backend")
    parser.add_argument("--kokoro-model", default=str(LOCAL_KOKORO_MODEL), help="Local Kokoro model and English voices directory")
    parser.add_argument("--kokoro-device", default="cpu",
                        help="Kokoro device (cpu or cuda:0); narration runs alongside Qwen automatically")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    args = parser.parse_args()
    if args.max_seq_len < 128:
        parser.error("--max-seq-len must be at least 128")
    if not args.design_only and not (Path(args.clone_model) / "model.safetensors").is_file():
        parser.error("Clone model missing. Download 0.6B Base with hf, set --clone-model, or use --design-only.")
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import uvicorn
    uvicorn.run(create_app(device=args.device, clone_device=args.clone_device,
                           model=args.model, max_seq_len=args.max_seq_len,
                           api_key=os.environ.get("TTS_API_KEY"), voices_dir=args.voices_dir,
                           kokoro_enabled=not args.no_kokoro, kokoro_model_dir=args.kokoro_model,
                           kokoro_device=args.kokoro_device,
                           keep_models_loaded=args.keep_models_loaded,
                           clone_model=None if args.design_only else args.clone_model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
