"""Select a bounded reference with sentence boundaries and an exact transcript."""
import re
from pathlib import Path

import numpy as np

ALIGNMENT_MODEL = Path(__file__).resolve().parent / "models" / "whisper" / "tiny.pt"
LANGUAGE_CODES = {"english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
                  "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt",
                  "spanish": "es", "italian": "it", "auto": None}
MIN_REFERENCE_SECONDS = 3.0
MAX_REFERENCE_SECONDS = 10.0


def normalized(text):
    return "".join(c.casefold() for c in text if c.isalnum())


def sentence_spans(text):
    # Keep offsets into the original string rather than rebuilding a transcript
    # from recognized words. Include CJK punctuation and sentence-ending quotes.
    boundaries = list(re.finditer(r"[.!?。！？]+[\"'”’»)]*(?=\s|$|[^\x00-\x7f])", text))
    spans, start = [], 0
    for match in boundaries:
        if normalized(text[start:match.end()]):
            spans.append((start, match.end()))
        start = match.end()
    if normalized(text[start:]):
        spans.append((start, len(text)))
    return spans


def select_aligned_segment(wav, sr, text, words, min_seconds=MIN_REFERENCE_SECONDS, max_seconds=MAX_REFERENCE_SECONDS):
    """Return a complete sentence/group only when aligned text matches exactly."""
    if not words or normalized("".join(word.word for word in words)) != normalized(text):
        raise ValueError("Alignment did not cover the exact dialogue text")
    offsets, offset, previous = [], 0, 0.0
    for word in words:
        start, end = float(word.start), float(word.end)
        if start < previous - 0.03 or end <= start or end > len(wav) / sr + 0.05:
            raise ValueError("Alignment has invalid or missing word timestamps")
        size = len(normalized(word.word))
        offsets.append((offset, offset + size, start, end))
        offset += size
        previous = end
    sentences = sentence_spans(text)
    candidates = []
    for i, (char_start, _) in enumerate(sentences):
        for _, char_end in sentences[i:]:
            nstart, nend = len(normalized(text[:char_start])), len(normalized(text[:char_end]))
            aligned = [word for word in offsets if word[1] > nstart and word[0] < nend]
            if not aligned or aligned[0][0] != nstart or aligned[-1][1] != nend:
                continue
            start = max(0.0, aligned[0][2] - 0.06)
            end = min(len(wav) / sr, aligned[-1][3] + 0.06)
            # Avoid cutting into neighboring words when adding a small margin.
            before = [word for word in offsets if word[1] <= nstart]
            after = [word for word in offsets if word[0] >= nend]
            if before:
                start = max(start, (before[-1][3] + aligned[0][2]) / 2)
            if after:
                end = min(end, (aligned[-1][3] + after[0][2]) / 2)
            duration = end - start
            if min_seconds <= duration <= max_seconds:
                candidates.append((abs(duration - 7.0), start, end, char_start, char_end))
    if not candidates:
        raise ValueError("No complete sentence fits the reference duration budget")
    _, start, end, a, b = min(candidates)
    return wav[round(start * sr):round(end * sr)].copy(), text[a:b].strip()


class ReferenceSelector:
    def __init__(self, model_path=None, min_seconds=MIN_REFERENCE_SECONDS, max_seconds=MAX_REFERENCE_SECONDS):
        self.model = None
        self.model_path = Path(model_path) if model_path else None
        self.min_seconds = min_seconds
        self.max_seconds = max_seconds

    def select(self, wav, sr, text, language, *, may_be_truncated=False):
        duration = len(wav) / sr
        if may_be_truncated:
            raise ValueError("Dialogue may have reached its generation token budget")
        if self.min_seconds <= duration <= self.max_seconds:
            return wav, text, "dialogue"
        if duration < self.min_seconds:
            raise ValueError("Dialogue is too short for a reference")
        model_path = self.model_path or ALIGNMENT_MODEL
        if not model_path.is_file():
            raise ValueError("Local alignment model is missing")
        if self.model is None:
            import stable_whisper
            self.model = stable_whisper.load_model(str(model_path), device="cpu")
        from scipy.signal import resample_poly
        from math import gcd
        divisor = gcd(sr, 16000)
        audio = resample_poly(wav, 16000 // divisor, sr // divisor).astype(np.float32)
        alignment_language = LANGUAGE_CODES[language.lower()]
        if alignment_language is None:
            import whisper
            mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio),
                                             n_mels=self.model.dims.n_mels)
            _, probabilities = self.model.detect_language(mel)
            alignment_language = max(probabilities, key=probabilities.get)
        result = self.model.align(audio, text, language=alignment_language,
                                  verbose=None, regroup=False, vad=False)
        if result is None:
            raise ValueError("Dialogue alignment failed")
        segment, transcript = select_aligned_segment(wav, sr, text, result.all_words(), self.min_seconds, self.max_seconds)
        return segment, transcript, "aligned_dialogue"
