"""Optional context-aware cuts using CPU alignment, never Python synthesis."""
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from audio_utils import resample


def contextual_cut(wav, tokens, before, text, after, sample_rate=24000):
    normalized = lambda value: "".join(c.casefold() for c in value if c.isalnum())
    words = [token for token in tokens or [] if normalized(token.text)]
    if normalized("".join(word.text for word in words)) != normalized(before + text + after):
        raise ValueError("Aligned token coverage differs from the requested text")
    left, right = len(normalized(before)), len(normalized(before + text))
    selected, preceding, following = [], [], []
    offset, previous_end = 0, 0.0
    for word in words:
        start, end = word.start_ts, word.end_ts
        if (start is None or end is None or not math.isfinite(start) or not math.isfinite(end)
                or start < previous_end - 0.001 or end <= start or end > len(wav) / sample_rate + 0.03):
            raise ValueError("Alignment timings are missing or invalid")
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
    start, end = max(0, selected[0].start_ts - 0.04), min(len(wav) / sample_rate, selected[-1].end_ts + 0.08)
    if preceding:
        start = max(start, (preceding[-1].end_ts + selected[0].start_ts) / 2)
    if following:
        end = min(end, (selected[-1].end_ts + following[0].start_ts) / 2)
    clip = wav[round(start * sample_rate):round(end * sample_rate)].copy()
    if not len(clip) or not np.isfinite(clip).all() or np.max(np.abs(clip)) < 1e-5:
        raise ValueError("Context cut is empty, invalid, or silent")
    fade = min(round(sample_rate * 0.005), len(clip) // 2)
    if fade:
        clip[:fade] *= np.linspace(0, 1, fade)
        clip[-fade:] *= np.linspace(1, 0, fade)
    return clip


class ContextAudio:
    def __init__(self, model_path):
        self.path = Path(model_path) if model_path else None
        self.model = None

    def cut(self, audio, rate, body):
        if not self.path or not self.path.is_file():
            raise ValueError("Optional alignment model is not installed")
        if self.model is None:
            import stable_whisper
            self.model = stable_whisper.load_model(str(self.path), device="cpu")
        full_text = body.context_before + body.input + body.context_after
        language = body.language if body.language.lower() != "auto" else None
        result = self.model.align(resample(audio, rate, 16000), full_text, language=language,
                                  verbose=None, regroup=False, vad=False)
        if result is None:
            raise ValueError("Alignment failed")
        tokens = [SimpleNamespace(text=w.word, start_ts=w.start, end_ts=w.end) for w in result.all_words()]
        return contextual_cut(audio, tokens, body.context_before, body.input, body.context_after, rate)
