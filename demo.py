"""Qwen3-TTS VoiceDesign demo: generate distinct voices from text descriptions.

Examples:
    python demo.py                                    # render the built-in voice gallery
    python demo.py -v "A gruff old pirate, gravelly and slow" -t "Arr, the tide waits for no one."
"""
import argparse
import logging
import re
import time
from pathlib import Path

import soundfile as sf
import torch

import fast_tts

GALLERY = [
    ("narrator", "A calm, deep middle-aged male narrator with a warm baritone and measured, documentary-style pacing.",
     "In the quiet hours before dawn, the city holds its breath, waiting for the first train to break the silence."),
    ("excited_teen", "An energetic teenage girl, high-pitched and bubbly, talking fast with lots of enthusiasm.",
     "Oh my gosh, you will not believe what just happened at school today, it was literally the best thing ever!"),
    ("old_wizard", "A frail, raspy elderly man with a slight tremor, speaking slowly and mysteriously, like an ancient wizard.",
     "Many have sought the crystal of Elandor, young one. Few have returned to tell of what they found."),
    ("news_anchor", "A crisp, confident female news anchor with a neutral American accent and a professional, authoritative tone.",
     "Good evening. Tonight's top story: city officials have approved a new plan to expand public transit."),
    ("nervous_clerk", "A nervous young man with a soft, slightly shaky voice, hesitant and apologetic.",
     "Um, sorry, I think there might have been a mistake with your order. I can, uh, fix that right away."),
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "voice"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--voice", help="Natural-language voice description")
    p.add_argument("-t", "--text", help="Text to speak (required with --voice)")
    p.add_argument("-l", "--language", default="English", help="English, Chinese, Japanese, Korean, German, French, "
                   "Russian, Portuguese, Spanish, Italian, or Auto")
    p.add_argument("-o", "--out", default="out", help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Fix sampling seed for reproducible voices")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--max-seq-len", type=int, default=1024,
                   help="Prompt + audio frame budget (12 frames/s); lower is faster")
    args = p.parse_args()

    if args.voice and not args.text:
        p.error("--text is required with --voice")
    jobs = [(slug(args.voice), args.voice, args.text)] if args.voice else GALLERY

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.WARNING)

    t0 = time.perf_counter()
    tts = fast_tts.load(max_seq_len=args.max_seq_len)
    print(f"Loaded in {time.perf_counter() - t0:.1f}s, VRAM {torch.cuda.memory_allocated() / 2**30:.2f} GiB\n"
          f"Plan: {tts.plan}")

    for name, voice, text in jobs:
        if args.seed is not None:
            torch.manual_seed(args.seed)
        t0 = time.perf_counter()
        wavs, sr = tts.generate_voice_design(text=text, instruct=voice, language=args.language,
                                             temperature=args.temperature)
        dt = time.perf_counter() - t0
        path = out / f"{name}.wav"
        sf.write(path, wavs[0], sr)
        dur = len(wavs[0]) / sr
        print(f"{path}  {dur:.1f}s audio in {dt:.1f}s (RTF {dt / dur:.2f}), "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
