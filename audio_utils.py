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
