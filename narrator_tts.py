"""Kokoro narration with local preset voices and bounded English text chunks."""
import re
import threading
from pathlib import Path

import numpy as np

MODEL_ID = "hexgrad/Kokoro-82M"
LOCAL_MODEL = Path(__file__).resolve().parent / "models" / "kokoro"
MODEL_LOAD_LOCK = threading.RLock()
SAMPLE_RATE = 24000
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
        return voice

    def chunks(self, body):
        import torch
        voice = self.validate(body)
        pipeline = self.pipelines[voice[0]]
        for text in text_chunks(body.input):
            # Do not hold a thread-local inference context across a generator yield.
            results = iter(pipeline(text, voice=self.packs[voice], speed=body.speed))
            while True:
                # Transformers temporarily changes the process-wide default
                # dtype while loading Qwen. Kokoro creates FP32 intermediates
                # using that default, so model loading cannot overlap this step.
                with MODEL_LOAD_LOCK, torch.inference_mode():
                    result = next(results, None)
                if result is None:
                    break
                wav = result.audio.detach().cpu().numpy().astype(np.float32)
                if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
                    raise RuntimeError("Kokoro produced empty or invalid audio")
                yield wav

    def generate(self, body):
        chunks = list(self.chunks(body))
        if not chunks:
            raise ValueError("Input contains no speakable text")
        return np.concatenate(chunks), SAMPLE_RATE
