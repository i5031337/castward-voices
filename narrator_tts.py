"""Kokoro narration with local preset voices and bounded English text chunks."""
import re
import logging
import math
import threading
from contextlib import nullcontext
from pathlib import Path

import numpy as np

MODEL_ID = "hexgrad/Kokoro-82M"
LOCAL_MODEL = Path(__file__).resolve().parent / "models" / "kokoro"
MODEL_LOAD_LOCK = threading.RLock()
SAMPLE_RATE = 24000
log = logging.getLogger(__name__)


def contextual_cut(wav, tokens, before, text, after):
    """Return only target words when native token coverage and boundaries are valid."""
    normalized = lambda value: "".join(c.casefold() for c in value if c.isalnum())
    words = [token for token in tokens or [] if normalized(token.text)]
    if normalized("".join(word.text for word in words)) != normalized(before + text + after):
        raise ValueError("Native token coverage differs from the requested text")
    left, right = len(normalized(before)), len(normalized(before + text))
    if left == right:
        raise ValueError("Target has no speakable words")
    selected, preceding, following = [], [], []
    offset, previous_end = 0, 0.0
    for word in words:
        start, end = word.start_ts, word.end_ts
        if (start is None or end is None or not math.isfinite(start) or not math.isfinite(end)
                or start < previous_end - 0.001 or end <= start or end > len(wav) / SAMPLE_RATE + 0.03):
            raise ValueError("Native token timings are missing or invalid")
        previous_end = end
        next_offset = offset + len(normalized(word.text))
        if offset < left < next_offset or offset < right < next_offset:
            raise ValueError("Target boundary crosses a token")
        if next_offset <= left:
            preceding.append(word)
        elif offset >= right:
            following.append(word)
        else:
            selected.append(word)
        offset = next_offset
    if not selected:
        raise ValueError("No target tokens")
    start = max(0, selected[0].start_ts - 0.04)
    end = min(len(wav) / SAMPLE_RATE, selected[-1].end_ts + 0.08)
    if preceding:
        start = max(start, (preceding[-1].end_ts + selected[0].start_ts) / 2)
    if following:
        end = min(end, (selected[-1].end_ts + following[0].start_ts) / 2)
    clip = wav[round(start * SAMPLE_RATE):round(end * SAMPLE_RATE)].copy()
    if not len(clip) or not np.isfinite(clip).all() or np.max(np.abs(clip)) < 1e-5:
        raise ValueError("Context cut is empty, invalid, or silent")
    # Tiny edge fades prevent a discontinuity at cuts without changing word timing.
    fade = min(120, len(clip) // 2)
    if fade:
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
    return clip
VOICE_ALIASES = {
    # American English (Female)
    "alloy": "af_alloy",
    "aoede": "af_aoede",
    "bella": "af_bella",
    "heart": "af_heart",
    "jessica": "af_jessica",
    "kore": "af_kore",
    "nicole": "af_nicole",
    "nova": "af_nova",
    "river": "af_river",
    "sarah": "af_sarah",
    "sky": "af_sky",
    # American English (Male)
    "adam": "am_adam",
    "echo": "am_echo",
    "eric": "am_eric",
    "fenrir": "am_fenrir",
    "liam": "am_liam",
    "michael": "am_michael",
    "onyx": "am_onyx",
    "puck": "am_puck",
    "santa": "am_santa",
    # British English (Female)
    "alice": "bf_alice",
    "emma": "bf_emma",
    "isabella": "bf_isabella",
    "lily": "bf_lily",
    # British English (Male)
    "daniel": "bm_daniel",
    "fable": "bm_fable",
    "george": "bm_george",
    "lewis": "bm_lewis",
}


def text_chunks(text, limit=240):
    """Prefer sentence boundaries; split unusually long sentences between words."""
    current = ""
    for sentence in re.split(r'(?<=[.!?])\s+|\n+', text.strip()):
        for word in sentence.split():
            if len(word) > limit:
                raise ValueError("Narration contains a word longer than 240 characters")
            if len(current) + len(word) + 1 > limit:
                yield current
                current = ""
            current = f"{current} {word}".strip()
        # Sentence-sized chunks let playback begin without waiting for a paragraph.
        if current:
            yield current
            current = ""


class NarratorEngine:
    def __init__(self, *, model_dir=LOCAL_MODEL, device="cpu"):
        from kokoro import KModel, KPipeline
        import spacy

        self.model_dir = Path(model_dir)
        if not spacy.util.is_package("en_core_web_sm"):
            raise RuntimeError("Kokoro needs en_core_web_sm; install the English model from requirements.txt")
        for name in ("config.json", "kokoro-v1_0.pth"):
            if not (self.model_dir / name).is_file():
                raise RuntimeError(f"Kokoro file missing: {self.model_dir / name}. See README setup.")
        self.voices = sorted(path.stem for path in (self.model_dir / "voices").glob("*.pt")
                             if re.fullmatch(r"[ab][fm]_[a-z]+", path.stem))
        if not self.voices:
            raise RuntimeError("No local English Kokoro voice packs; see README setup")
        self.model = KModel(repo_id=MODEL_ID, config=str(self.model_dir / "config.json"),
                            model=str(self.model_dir / "kokoro-v1_0.pth")).to(device).eval()
        self.pipelines = {accent: KPipeline(lang_code=accent, repo_id=MODEL_ID, model=self.model)
                          for accent in {voice[0] for voice in self.voices}}
        # Pass local tensors to the pipeline so requests cannot trigger Hub downloads.
        import torch
        self.packs = {voice: torch.load(self.model_dir / "voices" / f"{voice}.pt",
                                        map_location="cpu", weights_only=True)
                      for voice in self.voices}
        self.aliases = {**VOICE_ALIASES, **{v.split("_", 1)[1]: v for v in self.voices if "_" in v}}
        self.device = self.model.device
        self.generator = torch.Generator(device=self.device).manual_seed(0)
        self.stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            self.stream.synchronize()
        # Reserve temporary Kokoro tensors in addition to its resident weights.
        # Qwen's own decoder workspace is budgeted separately for overlap.
        self.workspace_bytes = 2**30 if self.stream is not None else 0

    def validate(self, body):
        if body.npc_id:
            raise ValueError("Kokoro uses preset voices; omit npc_id")
        if body.instructions and body.instructions.strip():
            raise ValueError("Kokoro uses preset voices; omit instructions and select voice")
        if body.language.lower() not in {"english", "auto"}:
            raise ValueError("This Kokoro backend supports English narration only")
        aliases = getattr(self, "aliases", None)
        if aliases is None:
            aliases = {**VOICE_ALIASES, **{v.split("_", 1)[1]: v for v in self.voices if "_" in v}}
        voice = aliases.get(body.voice, body.voice)
        if voice not in self.voices:
            raise ValueError("Unknown Kokoro voice. Use george, alloy, or GET /v1/voices for presets")
        # Check pathological words before starting any streaming response.
        if any(len(word) > 240 for word in body.input.split()):
            raise ValueError("Narration contains a word longer than 240 characters")
        before, after = getattr(body, "context_before", ""), getattr(body, "context_after", "")
        if before or after:
            if body.stream:
                raise ValueError("Contextual narration requires a non-streaming request")
            if len(before + body.input + after) > 240:
                raise ValueError("Narration context plus input accepts up to 240 characters")
        return voice

    def _infer(self, results):
        import torch
        from narrator_rng import NarratorRandomMode
        results = iter(results)
        while True:
            # Default-dtype changes and CUDA graph capture exclude narrator work.
            with MODEL_LOAD_LOCK, torch.inference_mode():
                stream = getattr(self, "stream", None)
                generator = getattr(self, "generator", None)
                try:
                    with (torch.cuda.stream(stream) if stream is not None else nullcontext()), \
                            (NarratorRandomMode(generator) if generator is not None else nullcontext()):
                        result = next(results, None)
                        wav = None if result is None else result.audio.detach().cpu().numpy().astype(np.float32)
                finally:
                    if stream is not None:
                        stream.synchronize()
            if result is None:
                break
            if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
                raise RuntimeError("Kokoro produced empty or invalid audio")
            yield result, wav

    def chunks(self, body):
        voice = self.validate(body)
        pipeline = self.pipelines[voice[0]]
        for text in text_chunks(body.input):
            for _, wav in self._infer(pipeline(text, voice=self.packs[voice], speed=body.speed)):
                yield wav

    def generate_context(self, body):
        voice = self.validate(body)
        full_text = body.context_before + body.input + body.context_after
        results = list(self._infer(self.pipelines[voice[0]](
            full_text, voice=self.packs[voice], speed=body.speed, split_pattern=None)))
        try:
            if len(results) != 1:
                raise ValueError("Context did not fit in one native inference chunk")
            result, wav = results[0]
            return contextual_cut(wav, getattr(result, "tokens", None), body.context_before, body.input, body.context_after), SAMPLE_RATE, "applied"
        except ValueError as error:
            log.warning("Contextual narration fell back to standalone: %s", error)
            wav, rate = self.generate(body)
            return wav, rate, "fallback"

    def generate(self, body):
        chunks = list(self.chunks(body))
        if not chunks:
            raise ValueError("Input contains no speakable text")
        return np.concatenate(chunks), SAMPLE_RATE
