# Audiobook proof of concept

This HTTP client turns the downloaded `beyond_lies_the_wub.txt` into a cast,
speaker annotations, and an audiobook. It uses llama-server for casting and
attribution, Kokoro for narration, and saved Qwen NPC voices for dialogue.
The TTS engine itself needs no changes.

Run these commands from the repository root. Annotation uses only the Python
standard library; rendering also uses the existing NumPy/soundfile installation
and FFmpeg for perceived-loudness measurement.

## Use another story

Pass a plaintext file after `prepare`, then render the generated manifest:

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare path/to/story.txt \
  --output examples/audiobook/out/my-story

.venv/bin/python examples/audiobook/audiobook.py render \
  --manifest examples/audiobook/out/my-story/manifest.json \
  --output examples/audiobook/out/my-story
```

`--story path/to/story.txt` also remains supported. A supplied story is read
without Wub-specific opening/ending markers. Optional `--start-text` and
`--end-text` trim front or back matter. Omitting the file selects the bundled
Wub example with its existing cleanup settings.

## Try the two-character conversation

With llama-server listening on port 8080:

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare \
  --scene 3 --output examples/audiobook/out/scene3
```

The program discovers the model through `/v1/models`. Override it with
`--llm-model MODEL`; `--llm-url` defaults to `http://localhost:8080/v1`.
Set `LLM_API_KEY` if authentication is enabled.

Inspect these generated files:

- `manifest.json`: cast, stable voice instructions, and ordered source spans.
- `review.txt`: every quote, predicted speaker, confidence, and attribution reason.
- `source.txt`: normalized prose, before speech-specific punctuation cleanup.
- `llm/*.json`: exact requests, responses, usage, and elapsed times, including retries.

The casting prompt requires each voice description to explicitly include its
intended gender (male, female, or gender-neutral), plus age when supported by
the text. It uses character information from the story and marks unsupported
casting choices as invented. Reusing a cast with `--cast-manifest` keeps its
existing descriptions; it does not apply the updated casting prompt.

Check all attributions, including those labeled high confidence. The labels are
the model's self-assessment. Change a dialogue segment's `speaker` to a character
ID and set `needs_review` to `false` after resolving a flagged assignment. Voice
instructions and `narrator_voice` are also editable. Keep the source and offsets
intact: the renderer validates complete coverage with no gaps or overlaps.

Start the native TTS server in another terminal with the Qwen and Kokoro profiles enabled:

```bash
./run-local.sh --kokoro-cpu --port 8001
```

Then render the inspected manifest:

```bash
.venv/bin/python examples/audiobook/audiobook.py render \
  --manifest examples/audiobook/out/scene3/manifest.json \
  --output examples/audiobook/out/scene3
```

`--tts-url` defaults to `http://localhost:8001/v1`. Set `TTS_API_KEY` if enabled.
The server needs the Qwen design and clone profiles plus Kokoro. The example's
`--tts-max-tokens` setting passes `max_new_tokens` to the Qwen profile, which
audio.cpp interprets as a text-chunk token limit. The default is 1024; each
speech chunk is also limited to 300 characters by this client.

Output is `audiobook.wav` (24 kHz mono), `timeline.json` with text and timestamps,
`voices.json` with server NPC IDs, and cached WAV files under `segments/`.
Re-running rendering reuses cached chunks. Changing a voice description or seed
creates a new NPC voice and invalidates its dialogue cache. If you replace the
TTS models at the same URL, use a fresh output directory to avoid stale audio.

The renderer creates character voices using a neutral reference sentence and
waits for preparation before rendering the story. The reference sentence is
never included in the audiobook. This avoids casting each character during
story playback, although the server's background reference preparation can
still trigger model swaps. During rendering, one worker submits narration and
another submits dialogue; the server serializes Qwen inference. Final assembly
always follows the manifest's original order.

## Process the complete story

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare \
  --output examples/audiobook/out/full

.venv/bin/python examples/audiobook/audiobook.py render \
  --manifest examples/audiobook/out/full/manifest.json \
  --output examples/audiobook/out/full
```

Annotation makes one casting call over the selected text. By default it uses
the paragraph prompt and requests at most five
target quote assignments per call, while retaining the entire scene as context.
Each target includes its exact quoted text alongside its ID to avoid confusing
quote IDs with paragraph labels or narration. Only the requested quote IDs may
appear in the response. Scene boundaries are
preserved, so the 162-quote story uses 35 attribution calls at this batch size.
Each batch gets the same last six assignments from the preceding scene;
earlier batches within the current scene are not supplied as speaker hints.
`--quote-batch-size 0` keeps the original one-call-per-scene behavior.
The first pass sends `reasoning_effort: "none"` by default. After all scenes
are attributed, each `unknown` answer is retried individually with
`reasoning_effort: "high"`, using its original scene prompt and exact target
text. First-pass results remain the context for subsequent scenes; fallback
results are not added as hints. Each unknown gets one reasoning pass, subject
to the same JSON correction policy. Known low/medium-confidence answers are
left for review. Resolved fallback answers also remain flagged when confidence
is below high; unresolved answers always remain flagged.

`--no-reasoning-fallback` disables that second pass for controlled comparisons.
`--fallback-reasoning-effort low|medium|high` changes its effort, and
`--reasoning-effort none|low|medium|high` changes the first pass. Raw requests
and responses for both passes are saved under `llm/`; `reasoning_fallback.json`
records original and retry assignments. Retried manifest segments retain
`attribution_first_pass` and `attribution_reasoning_effort`; the manifest also
records first-pass settings, fallback effort, and retried quote IDs.
The short attribution `reason` in the JSON response is still requested.
It requests JSON with temperature zero and validates character
IDs, exact quote coverage, and speaker membership. Invalid output gets up to
two correction attempts. Truncated output stops immediately; increase
`--llm-max-tokens` (default 8192, including reasoning tokens). Re-running preparation replaces the manifest,
including any manual edits; use separate output directories for comparisons.

The default `--attribution-prompt paragraphs` keeps quote labels inline,
adds paragraph IDs, and explains that multiple quotations in one paragraph
can share a speaker. Attribution receives character IDs, names, and optional
`aliases`, without casting instructions or personality descriptions.
`--attribution-prompt legacy` reproduces the original attribution prompt and
its separate-line quote labels.

Rendering proceeds by default, including flagged passages. `unknown` dialogue
uses the Kokoro narrator voice in its original position. Known speakers use
their assigned character voices even when flagged. `--manual-review` stops
before any TTS request if an attribution needs review; edit the manifest's
assignments and clear reviewed flags before relaunching. The old `--allow-review`
flag remains accepted for existing commands. `--pause` and `--scene-pause` control gaps
between source segments and scenes. Speech chunks are at most 300 characters.
Assembly matches perceived loudness per voice using
[FFmpeg's loudness measurements](https://ffmpeg.org/ffmpeg-filters.html#loudnorm)
(gated integrated LUFS and true peak). It applies one constant gain to all chunks of
each voice, including narration, preserving delivery changes within a voice.
The default target is `--target-lufs -20` with `--true-peak-db -1` headroom.
If any voice would exceed that peak ceiling, it lowers the common target for
all voices rather than clipping or compressing individual lines. Silent or
too-short audio without a valid LUFS measurement is not boosted. Measurements,
applied gains, and the effective target are saved in `loudness.json`.
Cached synthesis WAVs remain unchanged, so changing these settings only
requires reassembly. Assembly does not trim silence or verify transcripts.
Listen for omissions,
pronunciation problems, abrupt joins, and voice drift.

Short narration fragments adjacent to dialogue now use contextual synthesis
automatically. The client supplies the real surrounding paragraph through
`context_before` and `context_after`; the server synthesizes it as one Kokoro
inference and returns only the requested `input` span using native token
timings. No surrounding dialogue is included in the returned audio.

Restart the TTS server with the updated code before using this feature.
The client selects fragments of at most 100 characters/12 words, never crosses
a paragraph, and caps the full synthesis context at 240 characters. For long
paragraphs it tries a complete adjacent quote rather than truncating words.
Unsupported contexts remain standalone narration. The backend validates exact
token coverage, word boundaries, finite/non-overlapping timings, and a nonempty
cut. It falls back to synthesizing the original input if these checks fail.
Small bounded margins and 5 ms edge fades reduce cut artifacts.

The response header `X-Narration-Context` reports `applied` or `fallback`.
`narration_context.json` records the mode and exact context for every narration
chunk. Context is part of the audio cache key; validated cuts are reused,
while fallback results are retried on the next render. Per-voice loudness
matching runs after extraction. Use `--no-narration-context` for a comparison
with standalone narration. The original source and voice assignments remain
unchanged. Native timings provide strong checks but do not guarantee every
cut sounds natural; a listening review is still appropriate.

## Scope and limitations

The bundled text is Philip K. Dick's *Beyond Lies the Wub*, downloaded from
[Project Gutenberg](https://www.gutenberg.org/ebooks/28554), which lists it as
public domain in the USA. The downloaded source retains its license.

The bundled example starts at its first prose sentence and stops before the
transcriber's note when no file is supplied. Other files have no default prose
boundaries. Gutenberg start/end boilerplate is removed automatically, and lines
of asterisks separate scenes. Preserve blank lines between paragraphs. Quotation parsing
supports paired straight double quotes and curly double quotes; unmatched marks
are rejected. Single-quoted dialogue, nested dialogue, and multi-paragraph speech
with repeated opening marks need a more capable parser. Underscores used for
Gutenberg italics and enclosing dialogue quotation marks are removed only when
constructing speech requests.

The small model still sees a whole story for casting and each whole scene for
attribution. This is deliberately a short-story experiment, not a book-length
context-management system. Characters are attributed to their textual speaker;
special performance choices such as impersonation or a change of identity need
manual direction.

## Tests

```bash
.venv/bin/python -m unittest discover -s examples/audiobook -v
```

Tests cover story cleanup, source conservation, malformed attributions,
chunking, review blocking, ordered audio assembly, and cache invalidation using
mock TTS responses. Real FFmpeg measurements also verify loudness matching for
different timbres, preservation of quiet/loud delivery, silent audio handling,
and peak headroom. They require FFmpeg but no running servers.

## Initial local trial (2026-10-13)

The local Gemma E4B Q4 model completed the full story with eight speaking
characters and 162 quotation spans in six calls, without validation retries
after the cast-ID prompt and output budget were adjusted. Total request time
was about 226 seconds, with 14,594 prompt tokens and 22,458 completion tokens
(including reasoning). It flagged only one assignment for review.

Manual checking found 20/23 correct assignments in the opening scene and
29/29 in the captain/Wub conversation. All three opening-scene errors were
labeled high confidence. These are checks of two scenes, not a score for the
whole story. Raw responses, the complete manifest, and the manual audit are
saved locally under `out/full-v2/`; the separate two-character trial is under
`out/scene3/`. Generated output directories are ignored by Git.

No real TTS audio was rendered during this trial because the TTS server was
not listening on port 8001. Audio assembly, voice preparation ordering, and
cache behavior were verified with mock responses; generated speech still
needs a listening test.

## Compare attribution prompts

Keep source, cast, and model fixed when comparing prompts. Reuse a prior
manifest's cast instead of generating a new one:

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare \
  --cast-manifest examples/audiobook/out/full-v2/manifest.json \
  --attribution-prompt paragraphs --quote-batch-size 0 --reasoning-effort high \
  --no-reasoning-fallback \
  --output examples/audiobook/out/paragraphs

.venv/bin/python examples/audiobook/compare_prompts.py \
  --baseline examples/audiobook/out/full-v2/manifest.json \
  --candidate examples/audiobook/out/paragraphs/manifest.json
```

The comparison writes `out/prompt-comparison.json`, including per-scene scores,
every changed assignment, fixes, regressions, remaining errors, and request
times/token counts. Request metrics exclude the initial casting call.
`--cast-manifest` requires exactly the same normalized source; the comparison
also checks model, cast, quote coverage, and source spans.

To isolate batch size using the previous paragraph-prompt run as the baseline:

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare \
  --cast-manifest examples/audiobook/out/paragraphs/manifest.json \
  --attribution-prompt paragraphs --quote-batch-size 5 --reasoning-effort high \
  --no-reasoning-fallback \
  --output examples/audiobook/out/paragraphs-batch5

.venv/bin/python examples/audiobook/compare_prompts.py \
  --baseline examples/audiobook/out/paragraphs/manifest.json \
  --candidate examples/audiobook/out/paragraphs-batch5/manifest.json \
  --output examples/audiobook/out/batch5-comparison.json
```

Logs contain one file per batch/attempt. The comparison records batch size,
request times, completion tokens, and reported cached prompt tokens as well as
accuracy. Full scene context means repeated input tokens even though output is
limited to five assignments; smaller batches are not guaranteed to be faster.

To test individual quotes without internal reasoning:

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare \
  --cast-manifest examples/audiobook/out/paragraphs/manifest.json \
  --attribution-prompt paragraphs --quote-batch-size 1 --reasoning-effort none \
  --no-reasoning-fallback \
  --output examples/audiobook/out/paragraphs-single-none

.venv/bin/python examples/audiobook/compare_prompts.py \
  --baseline examples/audiobook/out/paragraphs-batch5/manifest.json \
  --candidate examples/audiobook/out/paragraphs-single-none/manifest.json \
  --output examples/audiobook/out/single-none-comparison.json
```

Speaker IDs are canonicalized only for case-equivalent matches to known IDs
(for example, `Peterson` becomes `peterson`); different names are rejected.
Raw responses retain the model's original spelling. Comparisons record both
batch size and requested reasoning effort.

`reference_labels.json` contains fixed reference speaker labels for this story,
manually assigned by the coding agent from the text before reviewing revised
model output. They are not independent human annotations. All 162 quotes are
covered, but two untagged lines in scene 2 (`q022_s02`, `q023_s02`) have ambiguous
speakers and are excluded from accuracy scoring. The other 160 use explicit
tags or contextual attribution. The ending is scored by textual speaker
(Captain Franco), leaving performance identity as a separate casting decision.
These labels are editable if a reviewer finds an attribution error.

## Short narrator-tag listening tests

`prosody_test.py` compares “he said,” “Peterson said,” and “the Wub said” between
the same cached character lines. Each example plays three variants in order:

1. Original isolated tag, with the original 120 ms inserted pauses.
2. Comma-ended tag, with 20 ms inserted pauses.
3. Tag synthesized together with its preceding quoted dialogue, then extracted
   using Kokoro's native predicted token timings, with 20 ms inserted pauses.

The context audio is synthesized locally on CPU using the same Kokoro weights
and narrator preset; comma-ended tags use the server. Whisper tiny was tried
first but returned a zero-duration “Peterson” word, so the saved comparisons
use Kokoro timings throughout. Context synthesis must remain one inference
chunk to retain the sentence's prosody. Cuts have small bounded margins and
still require a listening check for clipped consonants or awkward transitions.

```bash
.venv/bin/python examples/audiobook/prosody_test.py
```

The default source is `out/reference-audio/`, with its manifest, saved voice IDs,
and synthesis cache. Levels come from its `loudness.json`, or the existing
`out/loudness-check/loudness.json` for the initial recording. Character audio
and volume gains are identical across each comparison. Outputs under
`out/prosody-tests/` include individual WAVs, `comparison.wav`, contextual
source/tag audio, and exact cut timings and playback order in `tests.json`.
These experiments do not change the full audiobook or production chunking.

### Local comparison result (2026-10-13)

With the same Gemma E4B model, cast, source, temperature zero, and 8192-token
output budget, the combined paragraph/continuity/minimal-cast prompt produced:

| Scene | Original | Revised |
| --- | ---: | ---: |
| Opening | 20/23 | 22/23 |
| Crew meets the Wub | 36/39 | 38/39 |
| Captain/Wub conversation | 29/29 | 26/29 |
| Confrontation | 48/52 | 52/52 |
| Ending | 17/17 | 16/17 |
| Total scored | 150/160 (93.75%) | 154/160 (96.25%) |

The revision fixed eight errors and introduced four. It fixed two of the three
known opening mistakes, but still assigned the captain's “You're getting paid
for all this” to the Optus. Three new mistakes reversed a run of alternating
captain/Wub turns; the fourth assigned a continuing captain line to `unknown`.
Five of the six remaining errors were labeled high confidence.

There were 14 changed assignments overall, including changes to both ambiguous
lines excluded from scoring. Each version used five attribution calls with no
correction retries. Original/revised attribution request time was 209.5/201.6
seconds; prompt tokens were 10,647/9,958 and completion tokens were 20,832/20,261.
Timing is one local run per version, not a stable speed benchmark.

This is a combined prompt change, so it does not identify which individual
change helped. Better prompting improved the overall score on this story but
did not remove confident attribution mistakes or avoid regressions. The revised
prompt was opt-in at the time of this trial. Saved manifests/logs are under
`out/full-v2/` and `out/paragraphs/`, with the full comparison in
`out/prompt-comparison.json`.

### Five-quote batch trial (2026-10-14)

Using the same paragraph prompt, source, cast, model, and generation settings,
five-quote batches with explicit target text scored 156/160 (97.5%), compared
with 154/160 (96.25%) for whole-scene output. The scene scores changed from
22/23, 38/39, 26/29, 52/52, 16/17 to 23/23, 37/39, 29/29, 50/52, 17/17.
Five errors were fixed and three new errors appeared. High-confidence errors
fell from five to two; the remaining four errors include one `unknown` and one
medium-confidence wrong speaker.

The completed batch run used 35 calls with no correction retries and took
378.6 seconds, versus 201.6 seconds for five whole-scene calls. Completion
tokens increased from 20,261 to 37,076. Repeated scene context produced 84,002
prompt tokens, of which 60,202 were reported cached. These are single local
runs, so timings are descriptive rather than a stable speed benchmark.

An initial ID-only batching attempt sometimes confused quoted dialogue with
narration and was stopped. The completed run adds exact target text beside
each ID and explicitly states that targets are spoken dialogue. Therefore
the improvement cannot be attributed solely to batch size; target grounding
also changed. Its partial logs are preserved in `out/paragraphs-batch5-id-only/`.
The complete manifest and logs are in `out/paragraphs-batch5/`, with scored
changes and remaining errors in `out/batch5-comparison.json`. Two ambiguous
quotes remain excluded from scoring under the same fixed reference policy.

### Individual quotes, reasoning off (2026-10-14)

With the same model, cast, scene context, and exact target-text format,
`--quote-batch-size 1 --reasoning-effort none` scored 155/160 (96.875%), versus
156/160 for the five-quote reasoning-enabled run. Scene scores were 22/23,
38/39, 28/29, 51/52, and 16/17. Three errors were fixed and four introduced;
four of the five remaining errors were high confidence.

All 162 requests explicitly sent `reasoning_effort: "none"`; none of their
responses contained reasoning content. They completed without correction
retries in 141.9 seconds (2m 22s), compared with 378.6 seconds (6m 19s) for
five-quote batches. Completion tokens fell from 37,076 to 12,716. Of 367,503
prompt tokens, 346,197 were reported cached. Thus this configuration was about
2.7 times faster with one fewer correct assignment on the fixed 160-quote
reference set. Both batch size and reasoning mode changed, so this compares
combined configurations rather than isolating either factor.

The initial attempt stopped on a case-only ID mismatch (`Peterson` instead
of `peterson`). It is preserved under `out/paragraphs-single-none-case-strict/`.
The completed run accepts case-equivalent known IDs without changing speaker
identity; raw spellings remain in its logs. Complete results are under
`out/paragraphs-single-none/` and `out/single-none-comparison.json`. As with
the earlier trials, this is one local run on one story, with coding-agent
reference labels and two ambiguous quotes excluded from scoring.

### Compare a larger model

After loading Gemma 12B in llama-server, repeat the individual-quote setup
without changing cast, source, prompt, or reasoning mode:

```bash
.venv/bin/python examples/audiobook/audiobook.py prepare \
  --cast-manifest examples/audiobook/out/paragraphs/manifest.json \
  --attribution-prompt paragraphs --quote-batch-size 1 --reasoning-effort none \
  --no-reasoning-fallback \
  --output examples/audiobook/out/gemma12b-single-none

.venv/bin/python examples/audiobook/compare_prompts.py \
  --baseline examples/audiobook/out/paragraphs-single-none/manifest.json \
  --candidate examples/audiobook/out/gemma12b-single-none/manifest.json \
  --allow-model-change --output examples/audiobook/out/gemma12b-comparison.json
```

`--allow-model-change` explicitly permits a model comparison. Source hash,
cast, quote coverage, and span checks still apply. Both model IDs are recorded
in the report; without that flag the tool continues to require the same model
for a controlled prompt comparison.

The local Gemma 4 12B Q6_K trial scored 155/160 (96.875%), matching the E4B
individual-quote reasoning-off run. Scene scores were 23/23, 38/39, 29/29,
48/52, and 17/17. It fixed four E4B errors and introduced four, leaving five
errors, all labeled high confidence; the model flagged no passages for review.
Remaining errors were `q009_s02` and `q006_s04`, `q007_s04`, `q027_s04`,
`q038_s04`. The exact ballast-calculations request is preserved as
`out/gemma12b-single-none/ballast-prompt.txt` and
`out/gemma12b-single-none/llm/scene-02-batch-09-1.json`.

All 162 calls used reasoning off, returned no reasoning content, and completed
without retries. Total request time was 333.1 seconds (5m 33s), compared with
141.9 seconds for E4B. Completion tokens were 8,358; input tokens were 368,689,
including 345,241 reported cached. The model IDs/quantizations differ and these
are single local runs. With this particular prompt and story, moving to 12B
without reasoning did not improve accuracy. Full results are in
`out/gemma12b-comparison.json` under the same fixed reference/scoring policy.

### Gemma 26B-A4B, individual quotes with reasoning off (2026-10-14)

Repeating the same source, cast, paragraph prompt, individual-quote protocol,
and `reasoning_effort: "none"` with
`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` scored 158/160 (98.75%). This is the
26B-A4B model. Scene scores were 23/23, 38/39, 29/29, 51/52, and 17/17.
Compared with 12B's 155/160, it fixed three errors with no regressions:
Peterson's two Odysseus lines (`q006_s04`, `q007_s04`) and Franco's
"I'll do it. You can watch." (`q038_s04`).

The two remaining scored errors were low-confidence `unknown` assignments:
the ballast-calculations line (`q009_s02`, reference French) and
"You didn't see him--like a statue..." (`q027_s04`, reference Jones).
There were no high-confidence errors on the scored set. Four quotes were
flagged for review, including both ambiguous quotes already excluded from
scoring. Thus the two scored errors were surfaced for review rather than
confidently assigned to the wrong voice.

All 162 calls returned one assignment, explicitly used reasoning off, and
returned no reasoning content or correction retries. Total request time was
475.4 seconds (7m 55s), versus 333.1 seconds (5m 33s) for 12B. Completion
tokens were 8,339; prompt tokens were 369,724, including 346,259 reported
cached. These remain single local runs with different quantizations on one
story, using coding-agent reference labels and the same two ambiguous quotes
excluded. The full manifest, review, and request logs are under
`out/gemma26b-single-none/`; scored changes and metrics are in
`out/gemma26b-comparison.json`.

### Reasoning fallback for Gemma 26B unknowns (2026-10-14)

Replayed the four original `unknown` requests individually, changing only
`reasoning_effort` from `none` to `high`. The same loaded model, original
messages, scene context, temperature, and token limit were retained; previous
answers and reference labels were not added to the prompts. All four responses
contained reasoning content and passed assignment validation without retries.

| Quote | Reasoning result | Confidence | Fixed reference |
| --- | --- | --- | --- |
| `q009_s02` (ballast calculations) | French | low | French |
| `q022_s02` ("What did it say?") | unknown | low | Excluded as ambiguous |
| `q023_s02` ("It suggested we talk about other things.") | Peterson | medium | Excluded as ambiguous |
| `q027_s04` ("You didn't see him--like a statue...") | Jones | medium | Jones |

The two scored errors were resolved, bringing the combined first-pass plus
reasoning-fallback result to 160/160 on the fixed scored set. One of the two
excluded ambiguous quotes remains unknown. All four retry results still have
low/medium confidence and warrant review. The four calls took 168.2 seconds
(2m 48s); combined request time with the first pass was 643.7 seconds
(10m 44s). This is one story with coding-agent labels, not evidence of perfect
general accuracy. Original results are preserved; retry requests, responses,
reasoning, and a summary are under `out/gemma26b-unknowns-reasoning/`.

### Gemma 26B-A4B, five-quote batches with reasoning off (2026-10-14)

Changing only quote batch size from one to five, with the same loaded model,
source, cast, paragraph prompt, scene context, and reasoning off, matched the
individual first pass at 158/160 (98.75%). All scene scores were unchanged:
23/23, 38/39, 29/29, 51/52, and 17/17. There were no scored fixes or regressions
and no high-confidence errors. The same ballast and Jones lines remained
`unknown`, now with medium confidence. Both excluded ambiguous quotes also
remained unknown; five passages in total were flagged for review.

The run used 35 calls with no correction retries. All requests explicitly
sent `reasoning_effort: "none"`, returned no reasoning content, and covered
their target quotes. Request time fell from 475.4 seconds (7m 55s) to 279.2
seconds (4m 39s): about 41% less time, or 1.70 times faster. Completion tokens
fell from 8,339 to 6,329; prompt tokens fell from 369,724 to 83,024, including
68,036 reported cached. This comparison is between reasoning-off first
passes; reasoning fallback was not rerun on the batch output. These are single
local timings on the same story and fixed reference policy. Results are in
`out/gemma26b-batch5-none/` and `out/gemma26b-batch5-comparison.json`.
