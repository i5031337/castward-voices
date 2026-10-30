#!/usr/bin/env python3
"""Annotate a story with a local LLM, then render it through the TTS HTTP API."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
OPENING = "They had almost finished with the loading."
SEPARATOR = re.compile(r"\n\s*\*(?:\s+\*){2,}\s*\n")
QUOTES = re.compile(r'"[^"\n]*(?:\n[^"\n]*)*"|“[^”]*”')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def http(base, route, payload=None, key=None, timeout=600, include_headers=False):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = Request(base.rstrip("/") + route,
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
            return (data, {key.lower(): value for key, value in response.headers.items()}) if include_headers else data
    except HTTPError as error:
        message = error.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {route}: {message}") from error


def load_story(path, start, end):
    text = Path(path).read_text(encoding="utf-8-sig")
    if "*** START OF" in text:
        text = text.split("*** START OF", 1)[1].split("\n", 1)[1]
        text = text.split("*** END OF", 1)[0]
    if start:
        if start not in text:
            raise ValueError(f"Opening text not found: {start!r}; set --start-text for your story")
        text = text[text.index(start):]
    if end and end in text:
        text = text[:text.index(end)]
    # Gutenberg wraps prose lines. Preserve paragraphs; remove only wrap whitespace.
    scenes = []
    for scene in SEPARATOR.split(text):
        paragraphs = [re.sub(r"\s+", " ", p).strip()
                      for p in re.split(r"\n\s*\n", scene.strip()) if p.strip()]
        if paragraphs:
            scenes.append("\n\n".join(paragraphs))
    return scenes


def segment_scenes(scenes):
    source = "\n\n".join(scenes)
    segments, scene_records = [], []
    offset = 0
    for scene_id, scene in enumerate(scenes, 1):
        # Include the inter-scene whitespace in the source partition.
        scene_source = ("\n\n" if scene_id > 1 else "") + scene
        scene_start = offset
        cursor = 0
        quote_ids = []
        for match in QUOTES.finditer(scene_source):
            if match.start() > cursor:
                segments.append({"kind": "narration", "start": offset + cursor,
                                 "end": offset + match.start(), "speaker": "narrator"})
            quote_id = f"q{len(quote_ids) + 1:03d}_s{scene_id:02d}"
            quote_ids.append(quote_id)
            segments.append({"kind": "dialogue", "start": offset + match.start(),
                             "end": offset + match.end(), "quote_id": quote_id})
            cursor = match.end()
        if cursor < len(scene_source):
            segments.append({"kind": "narration", "start": offset + cursor,
                             "end": offset + len(scene_source), "speaker": "narrator"})
        offset += len(scene_source)
        scene_records.append({"id": scene_id, "start": scene_start, "end": offset,
                              "quote_ids": quote_ids})
        for segment in segments:
            if "scene" not in segment:
                segment["scene"] = scene_id
    for index, segment in enumerate(segments, 1):
        segment["id"] = f"seg{index:04d}"
        if segment["kind"] == "narration" and re.search(r'["“”]', source[segment["start"]:segment["end"]]):
            raise ValueError("Unmatched quotation marks; correct the source before annotation")
    return source, segments, scene_records


class Annotator:
    def __init__(self, args):
        self.args = args
        self.key = os.getenv("LLM_API_KEY")
        self.model = args.llm_model
        if not self.model:
            models = json.loads(http(args.llm_url, "/models", key=self.key))
            self.model = models["data"][0]["id"]
        self.calls = 0

    def ask(self, name, instruction, content, validate, *, reasoning_effort=None):
        messages = [{"role": "system", "content": instruction},
                    {"role": "user", "content": content}]
        for attempt in range(1, 4):
            request = {"model": self.model, "messages": messages,
                       "temperature": 0, "max_tokens": self.args.llm_max_tokens,
                       "response_format": {"type": "json_object"}}
            effort = reasoning_effort if reasoning_effort is not None else getattr(self.args, "reasoning_effort", None)
            if effort is not None:
                request["reasoning_effort"] = effort
            started = time.monotonic()
            response = json.loads(http(self.args.llm_url, "/chat/completions", request, self.key))
            self.calls += 1
            save_json(self.args.output / "llm" / f"{name}-{attempt}.json",
                      {"request": request, "response": response,
                       "seconds": round(time.monotonic() - started, 3)})
            answer = response["choices"][0]["message"]["content"]
            if response["choices"][0].get("finish_reason") == "length":
                raise ValueError(f"{name}: output truncated; increase --llm-max-tokens. See llm/ logs.")
            try:
                value = json.loads(answer)
                validate(value)
                return value
            except (ValueError, KeyError, TypeError) as error:
                print(f"{name}: attempt {attempt} rejected: {error}", flush=True)
                messages += [{"role": "assistant", "content": answer},
                             {"role": "user", "content": f"Fix the JSON: {error}. Return the complete corrected object."}]
        raise ValueError(f"{name}: invalid model output after three attempts; inspect llm/ logs")


def check_cast(value):
    cast = value["characters"]
    if not isinstance(cast, list) or not cast:
        raise ValueError("characters must be a nonempty list")
    ids = []
    for character in cast:
        identifier = character["id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,47}", identifier):
            raise ValueError("Character IDs must be lowercase snake_case, at most 48 characters")
        if identifier in ("narrator", "unknown") or identifier in ids:
            raise ValueError("Duplicate or reserved character ID")
        ids.append(identifier)
        for field in ("name", "instructions", "evidence"):
            if not isinstance(character[field], str) or not character[field].strip():
                raise ValueError(f"{field} must be nonempty text")
        if len(character["instructions"]) > 1000:
            raise ValueError("Voice instructions must be at most 1000 characters")


def check_assignments(value, expected, cast):
    assignments = value["assignments"]
    if not isinstance(assignments, list):
        raise ValueError("assignments must be a list")
    ids = [a["quote_id"] for a in assignments]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"Return each of these quote IDs exactly once: {expected}")
    canonical_ids = {speaker.casefold(): speaker for speaker in cast | {"unknown"}}
    for assignment in assignments:
        speaker = assignment["speaker"]
        if isinstance(speaker, str) and speaker.casefold() in canonical_ids:
            assignment["speaker"] = canonical_ids[speaker.casefold()]
        if assignment["speaker"] not in cast | {"unknown"}:
            raise ValueError(f"Invalid speaker {assignment['speaker']!r}. Use exactly one of {sorted(cast | {'unknown'})}")
        if assignment["confidence"] not in ("high", "medium", "low"):
            raise ValueError("confidence must be high, medium, or low")
        if not isinstance(assignment["reason"], str) or not assignment["reason"].strip():
            raise ValueError("reason must be nonempty text")


def attribution_context(source, selected, paragraphs=False):
    start, end = selected[0]["start"], selected[-1]["end"]
    text = source[start:end]
    insertions = []
    if paragraphs:
        for index, match in enumerate(re.finditer(r"\S.*?(?=\n\n|\Z)", text, re.S), 1):
            insertions.append((match.start(), f"[p{index:03d}] "))
    for segment in selected:
        if segment["kind"] == "dialogue":
            insertions.append((segment["start"] - start,
                               ("" if paragraphs else "\n") + f"[{segment['quote_id']}] "))
    pieces, cursor = [], 0
    for position, label in sorted(insertions, key=lambda entry: entry[0]):
        pieces.extend((text[cursor:position], label))
        cursor = position
    pieces.append(text[cursor:])
    return "".join(pieces)


def quote_batches(quote_ids, size):
    if size < 0:
        raise ValueError("Quote batch size must be nonnegative")
    size = size or len(quote_ids) or 1
    return [quote_ids[start:start + size] for start in range(0, len(quote_ids), size)]


def target_selection(source, quote_spans, targets):
    target_texts = [{"quote_id": quote_id,
                     "text": source[quote_spans[quote_id]["start"]:quote_spans[quote_id]["end"]]}
                    for quote_id in targets]
    return ('\nTARGET QUOTES: ' + json.dumps(target_texts, ensure_ascii=False)
            + '\nEach target is spoken dialogue, not narration. Identify the speaker of its exact text. '
            'Return exactly one assignment for each TARGET quote_id, and no other IDs. '
            'All other quotations in the scene are context only. '
            'Use unknown only when the speaker cannot be identified.')


def retry_unknowns(annotator, source, segments, scene_prompts, cast_ids, effort):
    retries = []
    spans = {s["quote_id"]: s for s in segments if s["kind"] == "dialogue"}
    for quote_id, segment in spans.items():
        if segment["speaker"] != "unknown":
            continue
        system_prompt, instructions = scene_prompts[segment["scene"]]
        before = {key: segment[key] for key in ("quote_id", "speaker", "confidence", "reason")}
        print(f"Reasoning retry: {quote_id}...", flush=True)
        result = annotator.ask(f"fallback-{quote_id}", system_prompt,
                               instructions + target_selection(source, spans, [quote_id]),
                               lambda value: check_assignments(value, [quote_id], cast_ids),
                               reasoning_effort=effort)
        after = result["assignments"][0]
        segment["attribution_first_pass"] = before
        segment["attribution_reasoning_effort"] = effort
        segment.update(after)
        segment["needs_review"] = segment["speaker"] == "unknown" or segment["confidence"] != "high"
        retries.append({"quote_id": quote_id, "before": before, "after": after})
    return retries


def prepare(args):
    scenes = load_story(args.story, args.start_text, args.end_text)
    if args.scene:
        if args.scene > len(scenes):
            raise ValueError(f"Story has only {len(scenes)} scenes")
        scenes = [scenes[args.scene - 1]]
    source, segments, scene_records = segment_scenes(scenes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "source.txt").write_text(source)
    annotator = Annotator(args)
    print(f"Model: {annotator.model}", flush=True)
    if args.cast_manifest:
        original = json.loads(args.cast_manifest.read_text())
        validate_manifest(original)
        if original["source_sha256"] != digest(source):
            raise ValueError("The reused cast manifest must have exactly the same normalized source")
        cast = original["characters"]
        print(f"Reusing {len(cast)} cast identities from {args.cast_manifest}", flush=True)
    else:
        print(f"Casting {len(scenes)} scene(s)...", flush=True)
        cast = annotator.ask("cast", "You cast voices for an unabridged audiobook. Return JSON only.",
            'List every speaking character in this story, including unnamed speaking roles. '
            'Merge aliases of the same character. Do not include the narrator. '
            'Return {"characters":[{"id":"snake_case", "name":"display name", '
            '"instructions":"a concise stable TTS voice description", '
            '"evidence":"textual basis; explicitly mark invented casting choices"}]}. '
            'Make voices distinguishable. Describe vocal sound and delivery, not appearance. '
            'Every instructions description must explicitly state the intended voice gender '
            '(male, female, or gender-neutral), and age range when supported by the text. '
            'Use the character\'s stated gender and contextual evidence, including pronouns and titles; '
            'do not infer gender from pitch or personality. For an adult man, explicitly request '
            'an adult male voice; for an adult woman, explicitly request an adult female voice. '
            'If gender or age is unspecified, choose a voice for casting and clearly mark that '
            'choice as invented in evidence. For a nonhuman character whose gender is unspecified, '
            'explicitly describe the intended voice gender without claiming a biological gender. '
            'Do not invent facts about the story.\n\nSTORY:\n' + source, check_cast)["characters"]
    cast_ids = {c["id"] for c in cast}
    previous = []
    scene_prompts = {}
    for scene in scene_records:
        if not scene["quote_ids"]:
            continue
        # Labels are inserted into context only; they never alter the spoken source.
        selected = [s for s in segments if s["scene"] == scene["id"]]
        paragraphs = args.attribution_prompt == "paragraphs"
        context = attribution_context(source, selected, paragraphs)
        attribution_cast = ([{"id": c["id"], "name": c["name"], "aliases": c.get("aliases", [])}
                             for c in cast] if paragraphs else cast)
        continuity = (" Preserve speaker continuity across multiple quotations in the same paragraph. "
                      "Intervening speech tags and actions often identify the same continuing speaker. "
                      "Do not assume quotation IDs alternate speakers. Paragraphs are labeled [pNNN]; "
                      "quotes are labeled [qNNN_sNN]. Paragraph continuity is evidence, not an absolute rule: "
                      "an explicit speaker change takes priority. "
                      'Example: [p001] [q001] "Ready?" Mara checked her watch. [q002] "We should leave." '
                      "Both q001 and q002 are Mara. Use tags, pronoun antecedents, actions, and turns "
                      "before personality or what a character might plausibly say."
                      if paragraphs else "")
        print(f"Attributing scene {scene['id']}: {len(scene['quote_ids'])} quotes...", flush=True)
        system_prompt = (
            "Identify the speaker of each labeled quotation. Return JSON only. "
            "Follow explicit speech tags, pronouns, and conversational context. "
            "A quotation containing several sentences has one speaker. "
            "Use unknown if the evidence is insufficient. Confidence is a self-assessment, not a probability."
            + continuity)
        instructions = (
            'Return {"assignments":[{"quote_id":"exact supplied ID", "speaker":"cast ID or unknown", '
            '"confidence":"high|medium|low", "reason":"brief attribution evidence"}]}. '
            + ('Return every quote exactly once. ' if not args.quote_batch_size else '')
            + 'Reasons should identify a speech tag or conversational turn, '
            'not a match to voice instructions. Keep each reason under 20 words. Never rewrite the story. '
            'Use these exact speaker IDs: ' + json.dumps(sorted(cast_ids | {"unknown"})) + '\nCAST:\n'
            + json.dumps(attribution_cast) + '\nPREVIOUS SCENE END:\n' + json.dumps(previous[-6:])
            + '\nSCENE:\n' + context)
        scene_prompts[scene["id"]] = (system_prompt, instructions)
        assignments = []
        batches = quote_batches(scene["quote_ids"], args.quote_batch_size)
        quote_spans = {segment["quote_id"]: segment for segment in selected if segment["kind"] == "dialogue"}
        for batch_index, targets in enumerate(batches, 1):
            name = f"scene-{scene['id']:02d}"
            if args.quote_batch_size:
                name += f"-batch-{batch_index:02d}"
                print(f"  Batch {batch_index}/{len(batches)}: {len(targets)} quotes", flush=True)
                selection = target_selection(source, quote_spans, targets)
            else:
                selection = ''
            result = annotator.ask(name, system_prompt, instructions + selection,
                                   lambda value: check_assignments(value, targets, cast_ids))
            assignments.extend(result["assignments"])
        by_id = {a["quote_id"]: a for a in assignments}
        for segment in selected:
            if segment["kind"] == "dialogue":
                segment.update(by_id[segment["quote_id"]])
                segment["needs_review"] = segment["speaker"] == "unknown" or segment["confidence"] != "high"
        # Preserve source order regardless of the model's response ordering.
        previous = [by_id[quote_id] for quote_id in scene["quote_ids"]]
    # Finish the entire first pass before retries so fallback answers cannot change
    # another scene's first-pass context. Retain exactly the original scene prompts.
    fallback_effort = None if args.no_reasoning_fallback else args.fallback_reasoning_effort
    retries = (retry_unknowns(annotator, source, segments, scene_prompts, cast_ids, fallback_effort)
               if fallback_effort else [])
    save_json(args.output / "reasoning_fallback.json",
              {"enabled": bool(fallback_effort), "reasoning_effort": fallback_effort, "retries": retries})
    manifest = {"version": 1, "story_file": str(args.story.resolve()), "source": source,
                "source_sha256": digest(source), "llm_model": annotator.model,
                "selected_scene": args.scene, "attribution_prompt": args.attribution_prompt,
                "quote_batch_size": args.quote_batch_size,
                "reasoning_effort": args.reasoning_effort,
                "fallback_reasoning_effort": fallback_effort,
                "fallback_quotes": [retry["quote_id"] for retry in retries],
                "cast_manifest": str(args.cast_manifest) if args.cast_manifest else None,
                "narrator_voice": "bm_george",
                "characters": cast, "segments": segments}
    validate_manifest(manifest)
    save_json(args.output / "manifest.json", manifest)
    review = []
    for segment in segments:
        if segment["kind"] == "dialogue":
            review.append(f"{segment['quote_id']} -> {segment['speaker']} ({segment['confidence']})"
                          + (" REVIEW" if segment["needs_review"] else "")
                          + "\n" + source[segment["start"]:segment["end"]]
                          + "\nReason: " + segment["reason"] + "\n")
    (args.output / "review.txt").write_text("\n".join(review))
    flagged = sum(s.get("needs_review", False) for s in segments)
    print(f"Saved {args.output / 'manifest.json'}; {len(review)} quotes, {flagged} flagged; "
          f"{annotator.calls} LLM calls. Inspect review.txt even for high-confidence assignments.")


def validate_manifest(manifest):
    source = manifest["source"]
    if manifest.get("version") != 1 or digest(source) != manifest["source_sha256"]:
        raise ValueError("Unsupported manifest or source hash mismatch")
    check_cast(manifest)
    cast = {c["id"] for c in manifest["characters"]}
    cursor, ids = 0, set()
    for segment in manifest["segments"]:
        if segment["id"] in ids:
            raise ValueError("Duplicate segment ID")
        ids.add(segment["id"])
        if segment["start"] != cursor or not cursor < segment["end"] <= len(source):
            raise ValueError("Segments must partition the original source without omissions or overlaps")
        cursor = segment["end"]
        if segment["kind"] not in ("narration", "dialogue"):
            raise ValueError("Invalid segment kind")
        allowed = {"narrator"} if segment["kind"] == "narration" else cast | {"unknown"}
        if segment["speaker"] not in allowed:
            raise ValueError(f"Invalid speaker in {segment['id']}")
        if segment["kind"] == "dialogue" and not isinstance(segment.get("needs_review"), bool):
            raise ValueError("Dialogue must have a boolean needs_review field")
    if cursor != len(source) or not source:
        raise ValueError("Incomplete or empty source partition")


def speech_text(text, dialogue=False):
    if dialogue:
        text = text[1:-1]  # Strip only the enclosing quotation marks.
    return re.sub(r"\s+", " ", text.replace("_", "")).strip()


def chunks(text, limit=300):
    while len(text) > limit:
        candidates = list(re.finditer(r"[.!?;][\"”']?\s+", text[:limit + 1]))
        split = candidates[-1].end() if candidates else text.rfind(" ", 0, limit + 1)
        if split <= 0:
            raise ValueError("A word exceeds the synthesis chunk limit")
        yield text[:split].strip()
        text = text[split:].lstrip()
    if text:
        yield text


def narration_context(source, segments, index):
    """Use real paragraph context for short narration adjacent to dialogue."""
    segment = segments[index]
    if segment["kind"] != "narration":
        return {}
    original = source[segment["start"]:segment["end"]]
    start = segment["start"] + len(original) - len(original.lstrip())
    end = segment["end"] - len(original) + len(original.rstrip())
    target = speech_text(source[start:end])
    if not target or len(target) > 100 or len(target.split()) > 12 or "\n\n" in source[start:end]:
        return {}
    left = source.rfind("\n\n", 0, start) + 2
    if left == 1:  # rfind returns -1 for the first paragraph.
        left = 0
    right = source.find("\n\n", end)
    if right < 0:
        right = len(source)
    before = speech_text(source[left:start])
    after = speech_text(source[end:right])
    previous = segments[index - 1] if index else None
    following = segments[index + 1] if index + 1 < len(segments) else None
    has_before = previous and previous["kind"] == "dialogue" and previous["start"] >= left
    has_after = following and following["kind"] == "dialogue" and following["end"] <= right
    if not (has_before or has_after):
        return {}
    def bounded(prefix, suffix):
        prefix = prefix + " " if prefix else ""
        suffix = " " + suffix if suffix else ""
        return ({"context_before": prefix, "context_after": suffix}
                if (prefix or suffix) and len(prefix + target + suffix) <= 240 else {})
    full = bounded(before, after)
    if full:
        return full
    # A long paragraph may still have a complete bounded adjacent quote.
    if has_before:
        result = bounded(speech_text(source[previous["start"]:start]), "")
        if result:
            return result
    if has_after:
        return bounded("", speech_text(source[end:following["end"]]))
    return {}


def measure_loudness(path):
    """Measure gated integrated LUFS and oversampled true peak with FFmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=I=-20:TP=-1:LRA=11:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise RuntimeError(f"FFmpeg loudness measurement failed: {result.stderr[-2000:]}")
    try:
        stats, _ = json.JSONDecoder().raw_decode(result.stderr[result.stderr.rindex("{"):])
        values = [float(stats[field]) for field in ("input_i", "input_tp")]
        return tuple(value if math.isfinite(value) else None for value in values)
    except (ValueError, KeyError) as error:
        raise RuntimeError("FFmpeg did not return valid loudness measurements") from error


def voice_loudness_gains(jobs, output_dir, target_lufs=-20.0, true_peak_db=-1.0):
    """One fixed gain per voice, with shared attenuation to preserve peak headroom."""
    import numpy as np
    import soundfile as sf

    groups = {}
    for job in jobs:
        voice = job["segment"]["speaker"] if job["payload"]["model"] != "kokoro" else "narrator"
        job["loudness_voice"] = voice
        groups.setdefault(voice, []).append(job)
    measurements = {}
    attenuation = 0.0
    with tempfile.TemporaryDirectory(prefix="loudness-", dir=output_dir) as folder:
        for voice, voice_jobs in groups.items():
            path = Path(folder) / (digest(voice) + ".wav")
            peak = 0.0
            with sf.SoundFile(path, "w", samplerate=24000, channels=1, subtype="FLOAT") as aggregate:
                for job in voice_jobs:
                    waveform, rate = sf.read(job["path"], dtype="float32", always_2d=True)
                    if rate != 24000 or waveform.shape[1] != 1 or not len(waveform) or not np.isfinite(waveform).all():
                        raise ValueError(f"Invalid cached audio: {job['path']}")
                    aggregate.write(waveform)
                    peak = max(peak, float(np.max(np.abs(waveform))))
            integrated, true_peak = measure_loudness(path)
            # Silent/too-short material has no valid LUFS measurement: do not boost it.
            gain_db = target_lufs - integrated if integrated is not None else 0.0
            sample_peak_db = 20 * math.log10(peak) if peak else None
            peaks = [value for value in (true_peak, sample_peak_db) if value is not None]
            if peaks:
                # A small margin covers rounded FFmpeg measurements and PCM quantization.
                attenuation = min(attenuation, true_peak_db - 0.1 - max(peaks) - gain_db)
            measurements[voice] = {"input_lufs": integrated, "input_true_peak_db": true_peak,
                                   "input_sample_peak_db": sample_peak_db, "gain_db": gain_db,
                                   "chunks": len(voice_jobs)}
    gains = {}
    for voice, measurement in measurements.items():
        measurement["gain_db"] += attenuation
        measurement["output_lufs_estimate"] = (measurement["input_lufs"] + measurement["gain_db"]
                                                if measurement["input_lufs"] is not None else None)
        gains[voice] = 10 ** (measurement["gain_db"] / 20)
    save_json(Path(output_dir) / "loudness.json",
              {"requested_lufs": target_lufs, "effective_lufs": target_lufs + attenuation,
               "true_peak_ceiling_db": true_peak_db, "shared_attenuation_db": attenuation,
               "voices": measurements})
    return gains


def render(args):
    import numpy as np
    import soundfile as sf

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    flagged = [s for s in manifest["segments"] if s.get("needs_review") or s["speaker"] == "unknown"]
    if flagged and args.manual_review:
        raise ValueError(f"{len(flagged)} passages need review. Edit manifest.json and clear reviewed flags, "
                         "or omit --manual-review (unknown speakers use Kokoro).")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required for perceived-loudness normalization")
    args.output.mkdir(parents=True, exist_ok=True)
    key = os.getenv("TTS_API_KEY")
    source = manifest["source"]
    used = {s["speaker"] for s in manifest["segments"] if s["kind"] == "dialogue"}
    cast = {c["id"]: c for c in manifest["characters"] if c["id"] in used}
    for character in cast.values():
        # Voice ID changes when casting instructions or the voice seed changes.
        identity = digest(manifest["source_sha256"] + character["id"] + character["instructions"] + str(args.seed))
        character["npc_id"] = "book_" + identity[:32]

    def audio(payload):
        contextual = bool(payload.get("context_before") or payload.get("context_after"))
        response = http(args.tts_url, "/audio/speech", payload, key, include_headers=contextual)
        if contextual:
            data, headers = response
            status = headers.get("x-narration-context")
            if status not in ("applied", "fallback"):
                raise RuntimeError("TTS server did not report a validated context cut; restart it with the updated code")
        else:
            data, status = response, None
        waveform, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        if rate != 24000 or waveform.shape[1] != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Expected finite, nonempty 24 kHz mono audio")
        return data, status

    # Design all voices before any cloning requests: minimize model transitions.
    for character in cast.values():
        npc_id = character["npc_id"]
        try:
            status = json.loads(http(args.tts_url, "/voices/" + npc_id, key=key))["status"]
        except RuntimeError as error:
            if not str(error).startswith("HTTP 404 "):
                raise
            status = "missing"
        if status in ("missing", "failed"):
            print(f"Preparing voice: {character['name']}...", flush=True)
            audio({"model": "qwen3-tts", "npc_id": npc_id, "instructions": character["instructions"],
                   "input": "Let us take a moment to consider this carefully. There is more to this story than we first imagined.",
                   "seed": args.seed, "response_format": "wav", "max_new_tokens": args.tts_max_tokens})
    for character in cast.values():
        deadline = time.monotonic() + args.voice_timeout
        while True:
            record = json.loads(http(args.tts_url, "/voices/" + character["npc_id"], key=key))
            if record["status"] == "ready":
                break
            if record["status"] == "failed" or time.monotonic() > deadline:
                raise RuntimeError(f"Voice preparation did not complete: {record}")
            time.sleep(1)
    save_json(args.output / "voices.json", list(cast.values()))
    jobs = []
    for segment_index, segment in enumerate(manifest["segments"]):
        text = speech_text(source[segment["start"]:segment["end"]], segment["kind"] == "dialogue")
        for index, chunk in enumerate(chunks(text)):
            character = cast.get(segment["speaker"])
            payload = {"model": "qwen3-tts-0.6b" if character else "kokoro", "input": chunk,
                       "response_format": "wav"}
            if character:
                payload.update(npc_id=character["npc_id"], seed=args.seed, max_new_tokens=args.tts_max_tokens)
            else:
                payload["voice"] = manifest["narrator_voice"]
                if not args.no_narration_context:
                    payload.update(narration_context(source, manifest["segments"], segment_index))
            cache_key = digest(json.dumps({"endpoint": args.tts_url, "request": payload}, sort_keys=True))
            path = args.output / "segments" / (cache_key + ".wav")
            jobs.append({"path": path, "payload": payload, "segment": segment,
                         "last": index, "text": chunk})

    def worker(dialogue):
        for job in jobs:
            if (job["payload"]["model"] != "kokoro") != dialogue:
                continue
            path = job["path"]
            contextual = bool(job["payload"].get("context_before") or job["payload"].get("context_after"))
            metadata = path.with_suffix(".json")
            cached_status = json.loads(metadata.read_text()).get("context_status") if metadata.exists() else None
            if not path.exists() or (contextual and cached_status != "applied"):
                print(f"Rendering {job['segment']['id']}: {job['text'][:65]}", flush=True)
                data, status = audio(job["payload"])
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(data)
                temporary.replace(path)
                if contextual:
                    save_json(metadata, {"context_status": status})
                job["context_status"] = status
            else:
                job["context_status"] = cached_status

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, dialogue) for dialogue in (False, True)]
        for future in futures:
            future.result()
    save_json(args.output / "narration_context.json",
              [{"segment_id": job["segment"]["id"], "input": job["text"],
                "status": job.get("context_status") or "standalone",
                "context_before": job["payload"].get("context_before", ""),
                "context_after": job["payload"].get("context_after", "")}
               for job in jobs if job["payload"]["model"] == "kokoro"])
    print("Matching perceived loudness across voices...", flush=True)
    gains = voice_loudness_gains(jobs, args.output, args.target_lufs, args.true_peak_db)
    # Stream assembly to disk instead of holding a whole book's waveform in memory.
    timeline = []
    frames = 0
    target = args.output / "audiobook.wav"
    temporary = target.with_suffix(".tmp")
    with sf.SoundFile(temporary, "w", samplerate=24000, channels=1, subtype="PCM_16", format="WAV") as output:
        for index, job in enumerate(jobs):
            waveform, rate = sf.read(job["path"], dtype="float32", always_2d=True)
            if rate != 24000 or waveform.shape[1] != 1 or not len(waveform) or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid cached audio: {job['path']}")
            waveform *= gains[job["loudness_voice"]]
            if float(np.max(np.abs(waveform))) > 10 ** (args.true_peak_db / 20):
                raise RuntimeError(f"Normalized audio exceeds the peak ceiling: {job['path']}")
            output.write(waveform)
            timeline.append({"segment_id": job["segment"]["id"], "speaker": job["segment"]["speaker"],
                             "text": job["text"], "start_seconds": frames / 24000,
                             "end_seconds": (frames + len(waveform)) / 24000})
            frames += len(waveform)
            if index + 1 < len(jobs):
                following = jobs[index + 1]["segment"]
                pause = (args.scene_pause if following["scene"] != job["segment"]["scene"]
                         else args.pause if following["id"] != job["segment"]["id"] else 0)
                silence = np.zeros(round(pause * 24000), dtype="float32")
                output.write(silence)
                frames += len(silence)
    temporary.replace(target)
    save_json(args.output / "timeline.json", timeline)
    print(f"Saved {target}: {frames / 24000:.1f} seconds, {len(jobs)} cached chunks")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="LLM casting and speaker attribution; no TTS")
    prepare_parser.add_argument("story_file", type=Path, nargs="?",
                                help="Plaintext story to annotate; default: bundled Beyond Lies the Wub example")
    prepare_parser.add_argument("--story", type=Path, help="Alternative spelling for the story_file argument")
    prepare_parser.add_argument("--start-text", help="Optional opening text to trim preceding material")
    prepare_parser.add_argument("--end-text", help="Optional marker at which to stop reading")
    prepare_parser.add_argument("--scene", type=int, help="One scene (1-based); private conversation is scene 3")
    prepare_parser.add_argument("--llm-url", default="http://localhost:8080/v1")
    prepare_parser.add_argument("--llm-model", help="Default: first model from /v1/models")
    prepare_parser.add_argument("--cast-manifest", type=Path, help="Reuse cast identities for a controlled prompt comparison")
    prepare_parser.add_argument("--attribution-prompt", choices=("legacy", "paragraphs"), default="paragraphs")
    prepare_parser.add_argument("--llm-max-tokens", type=int, default=8192)
    prepare_parser.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high"),
                               default="none", help="First-pass reasoning effort (default: none)")
    prepare_parser.add_argument("--no-reasoning-fallback", action="store_true",
                               help="Disable individual reasoning retries of unknown speakers")
    prepare_parser.add_argument("--fallback-reasoning-effort", choices=("low", "medium", "high"),
                               default="high", help="Reasoning effort for unknown-speaker retries (default: high)")
    prepare_parser.add_argument("--quote-batch-size", type=int, default=5,
                               help="Target quotes per attribution call; 0 means one call per scene")
    prepare_parser.add_argument("--output", type=Path, default=HERE / "out")
    render_parser = commands.add_parser("render", help="Render an inspected/editable manifest")
    render_parser.add_argument("--manifest", type=Path, default=HERE / "out/manifest.json")
    render_parser.add_argument("--tts-url", default="http://localhost:8001/v1")
    render_parser.add_argument("--output", type=Path, default=HERE / "out")
    render_parser.add_argument("--seed", type=int, default=42)
    render_parser.add_argument("--tts-max-tokens", type=int, default=1024,
                               help="Qwen generation budget; must fit server cache together with prompt")
    render_parser.add_argument("--voice-timeout", type=float, default=600)
    render_parser.add_argument("--pause", type=float, default=0.12, help="Seconds between source segments")
    render_parser.add_argument("--scene-pause", type=float, default=0.7)
    render_parser.add_argument("--target-lufs", type=float, default=-20.0,
                               help="Per-voice integrated loudness target (default -20 LUFS)")
    render_parser.add_argument("--true-peak-db", type=float, default=-1.0,
                               help="Peak ceiling; lowers all voices equally if necessary (default -1 dBTP)")
    review_options = render_parser.add_mutually_exclusive_group()
    review_options.add_argument("--manual-review", action="store_true",
                                help="Stop before TTS if any attribution needs review; default renders unknowns with Kokoro")
    # Retain the old opt-out spelling for existing launch commands.
    review_options.add_argument("--allow-review", dest="manual_review", action="store_false",
                                help=argparse.SUPPRESS)
    render_parser.set_defaults(manual_review=False)
    render_parser.add_argument("--no-narration-context", action="store_true",
                               help="Disable contextual synthesis of short narration fragments")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.story_file is not None and args.story is not None:
            parser.error("Specify the text file either as a positional argument or with --story, not both")
        supplied_story = args.story_file or args.story
        args.story = supplied_story or HERE / "beyond_lies_the_wub.txt"
        # Keep existing example commands working, without applying its prose
        # markers to a user-supplied story.
        if supplied_story is None:
            if args.start_text is None:
                args.start_text = OPENING
            if args.end_text is None:
                args.end_text = "Transcriber's Note:"
    if args.command == "prepare" and args.scene is not None and args.scene < 1:
        parser.error("--scene must be positive")
    if args.command == "prepare" and args.quote_batch_size < 0:
        parser.error("--quote-batch-size must be nonnegative")
    if args.command == "render" and (min(args.pause, args.scene_pause) < 0 or args.voice_timeout <= 0
                                     or not 0 <= args.seed < 2**63 or not 2 <= args.tts_max_tokens <= 4096
                                     or not -40 <= args.target_lufs <= -10 or not -9 <= args.true_peak_db <= -0.1):
        parser.error("Invalid pause, timeout, seed, token budget, or loudness target")
    try:
        (prepare if args.command == "prepare" else render)(args)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
