"""Audio validation and the public output-rate compatibility boundary."""
from math import gcd
import numpy as np
from scipy.signal import resample_poly


def mono_audio(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not len(audio) or sample_rate <= 0 or not np.isfinite(audio).all():
        raise RuntimeError("Backend produced empty or invalid audio")
    return audio


def resample(audio, source_rate, target_rate):
    audio = mono_audio(audio, source_rate)
    if source_rate == target_rate:
        return audio
    divisor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


class OutputNormalizer:
    """Match active speech level, with a peak ceiling and bounded amplification."""

    def __init__(self, sample_rate, target_dbfs=-20.0):
        self.sample_rate = sample_rate
        self.target = 10 ** (target_dbfs / 20)
        self.gain = None

    def apply(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if not len(audio):
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak < 0.001:
            return audio
        frame = max(1, round(self.sample_rate * 0.02))
        usable = len(audio) // frame * frame
        if usable:
            frames = audio[:usable].reshape(-1, frame)
            levels = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        else:
            levels = np.array([np.sqrt(np.mean(audio.astype(np.float64) ** 2))])
        threshold = max(10 ** (-45 / 20), float(np.max(levels)) * 10 ** (-25 / 20))
        active = levels[levels >= threshold]
        if not len(active):
            return audio
        desired = float(np.clip(self.target / np.median(active), 0.25, 4.0))
        # Streaming chunks keep a running gain to avoid jumps between sentences.
        self.gain = desired if self.gain is None else 0.75 * self.gain + 0.25 * desired
        gain = min(self.gain, 10 ** (-1 / 20) / peak)
        return (audio * gain).astype(np.float32)
