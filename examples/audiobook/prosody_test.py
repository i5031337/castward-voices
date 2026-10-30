#!/usr/bin/env python3
"""Compare isolated, comma-ended, and context-generated narrator speech tags."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from audiobook import HERE, digest, http, save_json, speech_text, validate_manifest


def extract_tag(waveform, rate, prefix, tag, words):
    """Extract exact aligned suffix words, with margins bounded by preceding speech."""
    normalized = lambda text: "".join(c.casefold() for c in text if c.isalnum())
    if normalized("".join(word.word for word in words)) != normalized(prefix + " " + tag):
        raise ValueError("Alignment did not cover the exact context and tag")
    boundary = len(normalized(prefix))
    offset, preceding_end, selected = 0, 0.0, []
    previous_start = 0.0
    for word in words:
        start, end = float(word.start), float(word.end)
        if not (math.isfinite(start) and math.isfinite(end) and
                0 <= start < end <= len(waveform) / rate + 0.03 and start >= previous_start):
            raise ValueError("Alignment has invalid timestamps")
        previous_start = start
        following = offset + len(normalized(word.word))
        if offset < boundary < following:
            raise ValueError("Tag boundary falls inside an aligned word")
        if offset >= boundary:
            selected.append(word)
        else:
            preceding_end = end
        offset = following
    if not selected or preceding_end > selected[0].start + 0.01:
        raise ValueError("Missing tag or overlapping word alignment")
    start = max(0, selected[0].start - 0.04, (preceding_end + selected[0].start) / 2)
    end = min(len(waveform) / rate, selected[-1].end + 0.08)
    clip = waveform[round(start * rate):round(end * rate)].copy()
    if not len(clip):
        raise ValueError("Aligned tag clip is empty")
    return clip, {"start_seconds": start, "end_seconds": end,
                  "words": [{"word": w.word, "start": float(w.start), "end": float(w.end)} for w in words]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=HERE / "out/reference-audio")
    parser.add_argument("--output", type=Path, default=HERE / "out/prosody-tests")
    parser.add_argument("--tts-url", default="http://localhost:8001/v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.source / "manifest.json").read_text())
    validate_manifest(manifest)
    cast = {c["id"]: c for c in json.loads((args.source / "voices.json").read_text())}
    segments = manifest["segments"]
    source = manifest["source"]
    # Existing full-story measurements keep narrator/characters matched across variants.
    loudness_path = args.source / "loudness.json"
    if not loudness_path.exists():
        loudness_path = args.source.parent / "loudness-check/loudness.json"
    levels = json.loads(loudness_path.read_text())["voices"]
    gains = {voice: 10 ** (stats["gain_db"] / 20) for voice, stats in levels.items()}

    def read(path):
        waveform, rate = sf.read(path, dtype="float32")
        if rate != 24000 or waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError(f"Invalid audio: {path}")
        return waveform

    def original(segment):
        text = speech_text(source[segment["start"]:segment["end"]], segment["kind"] == "dialogue")
        character = cast.get(segment["speaker"])
        payload = {"model": "qwen3-tts-0.6b" if character else "kokoro", "input": text, "response_format": "wav"}
        if character:
            payload.update(npc_id=character["npc_id"], seed=42, max_new_tokens=1024)
        else:
            payload["voice"] = manifest["narrator_voice"]
        key = digest(json.dumps({"endpoint": args.tts_url, "request": payload}, sort_keys=True))
        return read(args.source / "segments" / (key + ".wav"))

    def synthesize(name, text):
        path = args.output / (name + ".wav")
        if not path.exists():
            data = http(args.tts_url, "/audio/speech", {"model": "kokoro", "voice": manifest["narrator_voice"],
                        "input": text, "response_format": "wav"}, os.getenv("TTS_API_KEY"))
            path.write_bytes(data)
        return read(path)

    import torch
    torch.set_num_threads(4)
    # Kokoro exposes predicted token durations locally. These avoid Whisper's
    # collapsed word boundaries on very short proper names such as "Peterson".
    sys.path.insert(0, str(HERE.parents[1]))
    from narrator_tts import NarratorEngine
    from narrator_rng import NarratorRandomMode
    engine = NarratorEngine(device="cpu")
    voice = manifest["narrator_voice"]
    cases, clips = [], []
    for tag_id, name in (("seg0003", "he-said"), ("seg0088", "peterson-said"), ("seg0080", "wub-said")):
        print(f"Testing {name}...", flush=True)
        index = next(i for i, segment in enumerate(segments) if segment["id"] == tag_id)
        before, tag_segment, after = segments[index - 1:index + 2]
        if before["kind"] != "dialogue" or after["kind"] != "dialogue":
            raise ValueError("Expected a tag between two character quotes")
        tag = speech_text(source[tag_segment["start"]:tag_segment["end"]])
        prefix = source[before["start"]:before["end"]].replace("_", "").strip()
        full_text = prefix + " " + tag
        baseline = original(tag_segment)
        gentle_text = tag.rstrip(".") + ","
        gentle = synthesize(name + "-comma-source", gentle_text)
        with torch.inference_mode(), NarratorRandomMode(engine.generator):
            results = list(engine.pipelines[voice[0]](full_text, voice=engine.packs[voice], split_pattern=None))
        if len(results) != 1:
            raise ValueError("Context must fit in one Kokoro inference chunk")
        context = results[0].audio.detach().cpu().numpy().astype(np.float32)
        words = [SimpleNamespace(word=token.text, start=token.start_ts, end=token.end_ts)
                 for token in results[0].tokens if any(c.isalnum() for c in token.text)]
        sf.write(args.output / (name + "-context-native-source.wav"), context, 24000, subtype="PCM_16")
        contextual, cut = extract_tag(context, 24000, prefix, tag, words)
        cut["method"] = "Kokoro predicted token durations; local CPU context synthesis"
        sf.write(args.output / (name + "-context-tag.wav"), contextual, 24000, subtype="PCM_16")
        save_json(args.output / (name + "-alignment.json"), cut)
        character = before["speaker"]
        if after["speaker"] != character:
            raise ValueError("Expected the same character before and after the tag")
        opening, closing = original(before) * gains[character], original(after) * gains[character]
        for variant, tag_audio, pause in (("baseline", baseline, 0.12), ("gentle", gentle, 0.02),
                                         ("context", contextual, 0.02)):
            gap = np.zeros(round(pause * 24000), dtype="float32")
            waveform = np.concatenate((opening, gap, tag_audio * gains["narrator"], gap, closing))
            clips.append((name + "-" + variant, waveform))
        cases.append({"name": name, "tag": tag, "context": full_text, "gentle_input": gentle_text,
                      "before": source[before["start"]:before["end"]],
                      "after": source[after["start"]:after["end"]], "alignment_cut": cut})
    # Use the same additional headroom for every clip, so variant levels stay comparable.
    peak = max(float(np.max(np.abs(waveform))) for _, waveform in clips)
    common_gain = min(1.0, 10 ** (-1.1 / 20) / max(peak, 1e-6))
    combined, timeline, offset = [], [], 0.0
    for name, waveform in clips:
        waveform *= common_gain
        sf.write(args.output / (name + ".wav"), waveform, 24000, subtype="PCM_16")
        timeline.append({"name": name, "start_seconds": offset, "end_seconds": offset + len(waveform) / 24000})
        combined.extend((waveform, np.zeros(24000, dtype="float32")))
        offset += len(waveform) / 24000 + 1
    sf.write(args.output / "comparison.wav", np.concatenate(combined[:-1]), 24000, subtype="PCM_16")
    save_json(args.output / "tests.json", {"cases": cases, "order": [name for name, _ in clips],
               "timeline": timeline, "additional_gain": common_gain,
               "baseline_pause_seconds": 0.12, "test_pause_seconds": 0.02,
               "notes": "Each case plays baseline, comma-ended tag with shorter pauses, then context-generated aligned tag. Character clips and gain are identical across variants. Cuts require a listening check."})
    print(f"Saved {args.output / 'comparison.wav'} ({offset - 1:.1f} seconds)")


if __name__ == "__main__":
    main()
