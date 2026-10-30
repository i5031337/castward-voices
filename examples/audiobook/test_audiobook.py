"""Run with: .venv/bin/python -m unittest discover -s examples/audiobook -v"""
import argparse
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audiobook as book
from compare_prompts import compare, request_metrics


class AudiobookTests(unittest.TestCase):
    def test_story_file_argument_does_not_apply_example_trimming(self):
        for flags in (["another-story.txt"], ["--story", "another-story.txt"]):
            with patch("sys.argv", ["audiobook.py", "prepare"] + flags), patch.object(book, "prepare") as prepare:
                book.main()
                args = prepare.call_args.args[0]
                self.assertEqual(args.story, Path("another-story.txt"))
                self.assertIsNone(args.start_text)
                self.assertIsNone(args.end_text)
        with patch("sys.argv", ["audiobook.py", "prepare"]), patch.object(book, "prepare") as prepare:
            book.main()
            args = prepare.call_args.args[0]
            self.assertEqual(args.story, book.HERE / "beyond_lies_the_wub.txt")
            self.assertEqual(args.start_text, book.OPENING)

    def test_prepare_finishes_first_pass_before_fallback_and_saves_final_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            story = root / "story.txt"
            story.write_text('"Hello." Alice waved.\n\n"Goodbye." Alice left.')
            calls = []
            def fake_ask(annotator, name, instruction, content, validate, **kwargs):
                calls.append((name, content, kwargs))
                if name == "cast":
                    result = {"characters": self.manifest()["characters"]}
                else:
                    targets = json.loads(content.split("TARGET QUOTES: ", 1)[1].split("\n", 1)[0])
                    result = {"assignments": [{"quote_id": target["quote_id"],
                             "speaker": "alice" if kwargs else "unknown", "confidence": "medium",
                             "reason": "Alice waved"} for target in targets]}
                validate(result)
                return result
            with patch("sys.argv", ["audiobook.py", "prepare", str(story),
                                    "--llm-model", "test", "--quote-batch-size", "1",
                                    "--output", str(root / "out")]), \
                    patch.object(book.Annotator, "ask", fake_ask):
                book.main()
            self.assertEqual([c[0] for c in calls], ["cast", "scene-01-batch-01", "scene-01-batch-02",
                                                   "fallback-q001_s01", "fallback-q002_s01"])
            manifest = json.loads((root / "out/manifest.json").read_text())
            book.validate_manifest(manifest)
            dialogue = [s for s in manifest["segments"] if s["kind"] == "dialogue"]
            self.assertTrue(all(s["speaker"] == "alice" and s["needs_review"] for s in dialogue))
            self.assertTrue(all(s["attribution_first_pass"]["speaker"] == "unknown" for s in dialogue))
            self.assertEqual(manifest["fallback_quotes"], [s["quote_id"] for s in dialogue])
            report = json.loads((root / "out/reasoning_fallback.json").read_text())
            self.assertEqual(len(report["retries"]), 2)
            for initial, fallback in zip(calls[1:3], calls[3:]):
                self.assertEqual(initial[1], fallback[1])
                self.assertEqual(fallback[2], {"reasoning_effort": "high"})

    def test_request_metrics_include_fallback_but_exclude_cast(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            log = {"seconds": 2, "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}}
            for name in ("scene-01-1.json", "fallback-q001_s01-1.json", "cast-1.json"):
                book.save_json(root / "llm" / name, log)
            metrics = request_metrics(root)
            self.assertEqual((metrics["calls"], metrics["fallback_calls"], metrics["seconds"]), (2, 1, 4))
            self.assertEqual(metrics["completion_tokens"], 10)

    def test_default_workflow_and_fallback_opt_out(self):
        with patch("sys.argv", ["audiobook.py", "prepare"]), patch.object(book, "prepare") as prepare:
            book.main()
            args = prepare.call_args.args[0]
            self.assertEqual((args.attribution_prompt, args.quote_batch_size, args.reasoning_effort),
                             ("paragraphs", 5, "none"))
            self.assertFalse(args.no_reasoning_fallback)
            self.assertEqual(args.fallback_reasoning_effort, "high")
        with patch("sys.argv", ["audiobook.py", "prepare", "--no-reasoning-fallback"]), \
                patch.object(book, "prepare") as prepare:
            book.main()
            self.assertTrue(prepare.call_args.args[0].no_reasoning_fallback)

    def test_unknown_fallback_uses_single_targets_and_preserves_review_and_history(self):
        manifest = self.manifest()
        source = manifest["source"]
        dialogue = [s for s in manifest["segments"] if s["kind"] == "dialogue"]
        for s in dialogue:
            s.update(speaker="unknown", confidence="low", reason="Insufficient evidence", scene=1)
        known = dict(dialogue[0], quote_id="known", speaker="alice")
        prompts = {1: ("original system", "original scene context")}
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(llm_model="test", llm_url="http://test/v1", llm_max_tokens=8192,
                                      reasoning_effort="none", output=Path(folder))
            responses = []
            for segment, speaker in zip(dialogue, ("alice", "unknown")):
                content = {"assignments": [{"quote_id": segment["quote_id"], "speaker": speaker,
                                             "confidence": "medium", "reason": "Turn evidence"}]}
                responses.append(json.dumps({"choices": [{"finish_reason": "stop",
                                  "message": {"content": json.dumps(content)}}]}).encode())
            with patch.object(book, "http", side_effect=responses) as http:
                retries = book.retry_unknowns(book.Annotator(args), source,
                                              manifest["segments"] + [known], prompts, {"alice"}, "high")
            self.assertEqual(len(retries), 2)
            self.assertEqual(http.call_count, 2)  # Still-unknown result is not retried in a loop.
            for call, segment in zip(http.call_args_list, dialogue):
                request = call.args[2]
                self.assertEqual(request["reasoning_effort"], "high")
                self.assertEqual(request["messages"][0]["content"], "original system")
                text = request["messages"][1]["content"]
                self.assertTrue(text.startswith("original scene context\nTARGET QUOTES: "))
                targets = json.loads(text.split("TARGET QUOTES: ", 1)[1].split("\n", 1)[0])
                self.assertEqual(targets, [{"quote_id": segment["quote_id"],
                                           "text": source[segment["start"]:segment["end"]]}])
                self.assertEqual(segment["attribution_first_pass"]["speaker"], "unknown")
                self.assertTrue(segment["needs_review"])
            self.assertEqual(dialogue[0]["speaker"], "alice")
            self.assertEqual(dialogue[1]["speaker"], "unknown")
            self.assertNotIn("attribution_first_pass", known)
            self.assertEqual(args.reasoning_effort, "none")

    def test_model_comparison_requires_explicit_opt_in(self):
        baseline = self.manifest()
        baseline["llm_model"] = "small-model"
        candidate = json.loads(json.dumps(baseline))
        candidate["llm_model"] = "large-model"
        reference = {"source_sha256": baseline["source_sha256"], "method": "test", "policy": "test",
                     "quotes": {s["quote_id"]: {"speaker": "alice", "scored": True}
                                for s in baseline["segments"] if s["kind"] == "dialogue"}}
        with self.assertRaises(ValueError):
            compare(baseline, candidate, reference)
        report = compare(baseline, candidate, reference, allow_model_change=True)
        self.assertTrue(report["model_changed"])
        self.assertEqual(report["baseline_model"], "small-model")
        self.assertEqual(report["candidate_model"], "large-model")
        self.assertEqual(report["candidate_correct"], 2)

    def test_reasoning_effort_is_explicit_and_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(llm_model="test-4b", llm_url="http://test/v1",
                                      llm_max_tokens=8192, output=Path(folder))
            response = b'{"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]}'
            with patch.object(book, "http", return_value=response) as http:
                book.Annotator(args).ask("default", "Instruction", "Context", lambda value: None)
                self.assertNotIn("reasoning_effort", http.call_args.args[2])
                args.reasoning_effort = "none"
                book.Annotator(args).ask("none", "Instruction", "Context", lambda value: None)
                self.assertEqual(http.call_args.args[2]["reasoning_effort"], "none")

    def test_known_speaker_id_case_is_normalized_without_guessing(self):
        value = {"assignments": [{"quote_id": "q1", "speaker": "Peterson", "confidence": "high", "reason": "Speech tag"}]}
        book.check_assignments(value, ["q1"], {"peterson"})
        self.assertEqual(value["assignments"][0]["speaker"], "peterson")
        value["assignments"][0]["speaker"] = "Peter"
        with self.assertRaises(ValueError):
            book.check_assignments(value, ["q1"], {"peterson"})

    def test_quote_batches_preserve_order_and_exact_coverage(self):
        quotes = [f"q{i}" for i in range(12)]
        batches = book.quote_batches(quotes, 5)
        self.assertEqual([len(batch) for batch in batches], [5, 5, 2])
        self.assertEqual([quote for batch in batches for quote in batch], quotes)
        self.assertEqual(book.quote_batches(quotes, 0), [quotes])
        self.assertEqual(book.quote_batches([], 5), [])
        with self.assertRaises(ValueError):
            book.quote_batches(quotes, -1)

    def test_context_fallbacks_are_reported_and_retried_without_recloning(self):
        import numpy as np
        import soundfile as sf
        stream = io.BytesIO()
        sf.write(stream, np.ones(2400) * 0.1, 24000, format="WAV")
        calls = []
        def fake_http(base, route, payload=None, key=None, **kwargs):
            if route.startswith("/voices/"):
                return b'{"status":"ready"}'
            calls.append(payload)
            return ((stream.getvalue(), {"x-narration-context": "fallback"})
                    if kwargs.get("include_headers") else stream.getvalue())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            book.save_json(root / "manifest.json", self.manifest())
            args = argparse.Namespace(manifest=root / "manifest.json", output=root, manual_review=False,
                                      seed=42, tts_url="http://test/v1", voice_timeout=10,
                                      pause=0.12, scene_pause=0.7, tts_max_tokens=1024,
                                      target_lufs=-20, true_peak_db=-1, no_narration_context=False)
            with patch.object(book, "http", side_effect=fake_http):
                book.render(args)
                book.render(args)
            report = json.loads((root / "narration_context.json").read_text())
            self.assertEqual([row["status"] for row in report], ["fallback", "fallback"])
            self.assertEqual(sum(call["model"] == "kokoro" for call in calls), 4)
            self.assertEqual(sum(call["model"] != "kokoro" for call in calls), 2)

    def test_narration_context_respects_paragraphs_and_bounds(self):
        source, segments, _ = book.segment_scenes(['"Hello," he said. "Goodbye."',
                                                  'He waited.\n\n"Why?"'])
        tag = next(i for i, s in enumerate(segments) if source[s["start"]:s["end"]].strip() == "he said.")
        self.assertEqual(book.narration_context(source, segments, tag),
                         {"context_before": '"Hello," ', "context_after": ' "Goodbye."'})
        self.assertTrue(all(not book.narration_context(source, segments, i)
                            for i, s in enumerate(segments) if "He waited" in source[s["start"]:s["end"]]))
        long, spans, _ = book.segment_scenes(['"' + 'hello ' * 60 + '", he said.'])
        self.assertFalse(book.narration_context(long, spans, 1))

    def test_contextual_tag_cut_keeps_suffix_and_rejects_bad_alignment(self):
        import numpy as np
        from types import SimpleNamespace
        from prosody_test import extract_tag
        words = [SimpleNamespace(word="Hello", start=0.0, end=0.3),
                 SimpleNamespace(word="he", start=0.4, end=0.6),
                 SimpleNamespace(word="said", start=0.6, end=0.8)]
        audio = np.arange(100, dtype="float32")
        clip, cut = extract_tag(audio, 100, "Hello", "he said.", words)
        self.assertTrue(np.array_equal(clip, audio[36:88]))
        self.assertGreater(cut["start_seconds"], words[0].end)
        words[1].end = words[1].start
        with self.assertRaises(ValueError):
            extract_tag(audio, 100, "Hello", "he said.", words)

    def audio_jobs(self, root, signals):
        import soundfile as sf
        jobs = []
        for index, (voice, waveform) in enumerate(signals):
            path = root / f"{index}.wav"
            sf.write(path, waveform, 24000, subtype="FLOAT")
            jobs.append({"path": path, "payload": {"model": "qwen3-tts-0.6b"},
                         "segment": {"speaker": voice}})
        return jobs

    def test_perceived_loudness_matches_different_voice_timbres(self):
        import numpy as np
        import soundfile as sf
        time = np.arange(24000 * 4) / 24000
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = self.audio_jobs(root, [("deep", 0.3 * np.sin(2 * np.pi * 150 * time)),
                                          ("bright", 0.04 * np.sin(2 * np.pi * 2000 * time))])
            gains = book.voice_loudness_gains(jobs, root)
            measured = []
            for job in jobs:
                waveform, rate = sf.read(job["path"])
                path = root / (job["loudness_voice"] + ".wav")
                sf.write(path, waveform * gains[job["loudness_voice"]], rate, subtype="FLOAT")
                measured.append(book.measure_loudness(path)[0])
            self.assertLess(abs(measured[0] - measured[1]), 0.15)
            self.assertAlmostEqual(measured[0], -20, delta=0.15)
            self.assertLess(gains["deep"], 1)
            self.assertGreater(gains["bright"], 1)

    def test_peak_headroom_uses_shared_attenuation_and_preserves_dynamics(self):
        import numpy as np
        import soundfile as sf
        time = np.arange(24000 * 4) / 24000
        signal = 0.005 * np.sin(2 * np.pi * 400 * time)
        signal[24000] = 0.98
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = self.audio_jobs(root, [("spiky", signal), ("steady", 0.1 * np.sin(2 * np.pi * 800 * time)),
                                          ("steady", 0.05 * np.sin(2 * np.pi * 800 * time)),
                                          ("silent", np.zeros(24000))])
            gains = book.voice_loudness_gains(jobs, root, target_lufs=-12, true_peak_db=-1)
            report = json.loads((root / "loudness.json").read_text())
            self.assertLess(report["shared_attenuation_db"], 0)
            self.assertEqual(report["voices"]["silent"]["input_lufs"], None)
            self.assertAlmostEqual(report["voices"]["spiky"]["output_lufs_estimate"],
                                   report["voices"]["steady"]["output_lufs_estimate"])
            steady_peaks = []
            for job in jobs:
                waveform, rate = sf.read(job["path"])
                normalized = waveform * gains[job["loudness_voice"]]
                if job["loudness_voice"] == "steady":
                    steady_peaks.append(float(np.max(np.abs(normalized))))
                self.assertLessEqual(float(np.max(np.abs(normalized))), 10 ** (-1 / 20))
                if job["loudness_voice"] == "silent":
                    self.assertTrue(np.all(normalized == 0))
                else:
                    path = root / (job["loudness_voice"] + ".wav")
                    sf.write(path, normalized, rate, subtype="FLOAT")
                    self.assertLessEqual(book.measure_loudness(path)[1], -1)
            # The quieter line of one voice stays quieter by the same amount.
            self.assertAlmostEqual(steady_peaks[0] / steady_peaks[1], 2, places=5)

    def manifest(self):
        source, segments, _ = book.segment_scenes(['Alice said, "Hello."', '"Goodbye," she said.'])
        for segment in segments:
            if segment["kind"] == "dialogue":
                segment.update(speaker="alice", needs_review=False)
        return {"version": 1, "source": source, "source_sha256": book.digest(source),
                "narrator_voice": "bm_george", "segments": segments,
                "characters": [{"id": "alice", "name": "Alice", "instructions": "Warm clear voice",
                                "evidence": "Invented casting choice"}]}

    def test_story_cleanup_and_scene_selection(self):
        scenes = book.load_story(book.HERE / "beyond_lies_the_wub.txt", book.OPENING, "Transcriber's Note:")
        self.assertEqual(len(scenes), 5)
        self.assertTrue(scenes[2].startswith("The wub eased itself"))
        self.assertNotIn("PROJECT GUTENBERG", "".join(scenes))
        _, segments, _ = book.segment_scenes([scenes[2]])
        self.assertEqual(sum(s["kind"] == "dialogue" for s in segments), 29)

    def test_source_conservation(self):
        manifest = self.manifest()
        book.validate_manifest(manifest)
        self.assertEqual("".join(manifest["source"][s["start"]:s["end"]]
                                 for s in manifest["segments"]), manifest["source"])
        manifest["segments"][1]["start"] += 1
        with self.assertRaises(ValueError):
            book.validate_manifest(manifest)

    def test_inline_context_preserves_paragraphs_and_text(self):
        manifest = self.manifest()
        context = book.attribution_context(manifest["source"], manifest["segments"], paragraphs=True)
        import re
        unlabelled = re.sub(r"\[(?:p\d+|q\d+_s\d+)\] ", "", context)
        self.assertEqual(unlabelled, manifest["source"])
        self.assertIn('[p001] Alice said, [q001_s01] "Hello."', context)
        self.assertIn('\n\n[p002] [q001_s02] "Goodbye,"', context)

    def test_comparison_counts_fixes_regressions_and_exclusions(self):
        baseline = self.manifest()
        baseline["llm_model"] = "test-4b"
        candidate = json.loads(json.dumps(baseline))
        quotes = [s for s in baseline["segments"] if s["kind"] == "dialogue"]
        reference = {"source_sha256": baseline["source_sha256"], "method": "test", "policy": "test",
                     "quotes": {s["quote_id"]: {"speaker": "alice", "scored": True} for s in quotes}}
        quotes[0].update(speaker="unknown", confidence="high")
        candidate["segments"][3].update(speaker="unknown", confidence="high")
        result = compare(baseline, candidate, reference)
        self.assertEqual((result["fixes"], result["regressions"]), (1, 1))
        self.assertEqual(result["candidate_high_confidence_errors"], 1)
        reference["quotes"][quotes[1]["quote_id"]]["scored"] = False
        result = compare(baseline, candidate, reference)
        self.assertEqual((result["scored_quotes"], result["regressions"]), (1, 0))

    def test_unmatched_quote_is_rejected(self):
        with self.assertRaises(ValueError):
            book.segment_scenes(['He said "Hello.'])

    def test_assignments_require_exact_ids_and_known_speakers(self):
        assignment = {"quote_id": "q1", "speaker": "alice", "confidence": "high", "reason": "Speech tag"}
        book.check_assignments({"assignments": [assignment]}, ["q1"], {"alice"})
        for values in ([], [assignment, assignment], [dict(assignment, speaker="bob")]):
            with self.assertRaises(ValueError):
                book.check_assignments({"assignments": values}, ["q1"], {"alice"})

    def test_chunking_preserves_spoken_words(self):
        text = "A short sentence. " * 60
        result = list(book.chunks(text.strip()))
        self.assertTrue(all(len(chunk) <= 300 for chunk in result))
        self.assertEqual(" ".join(result), text.strip())
        self.assertEqual(book.speech_text('"It was _very_ strange."', True), "It was very strange.")

    def test_render_order_cache_and_cast_invalidation(self):
        import numpy as np
        import soundfile as sf

        stream = io.BytesIO()
        sf.write(stream, np.ones(2400, dtype="float32") * 0.1, 24000, format="WAV", subtype="PCM_16")
        requests = []

        def fake_http(base, route, payload=None, key=None, **kwargs):
            requests.append((route, payload))
            if route.startswith("/voices/"):
                return b'{"status":"ready"}'
            return (stream.getvalue(), {"x-narration-context": "applied"}) if kwargs.get("include_headers") else stream.getvalue()

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.manifest()
            book.save_json(root / "manifest.json", manifest)
            args = argparse.Namespace(manifest=root / "manifest.json", output=root, manual_review=False,
                                      seed=42, tts_url="http://test/v1", voice_timeout=10,
                                      pause=0.12, scene_pause=0.7, tts_max_tokens=1024,
                                      target_lufs=-20, true_peak_db=-1, no_narration_context=False)
            with patch.object(book, "http", side_effect=fake_http):
                book.render(args)
                first_count = sum(route == "/audio/speech" for route, _ in requests)
                self.assertEqual(first_count, 4)
                book.render(args)
                self.assertEqual(sum(route == "/audio/speech" for route, _ in requests), first_count)
                manifest["characters"][0]["instructions"] = "A new contrasting voice"
                book.save_json(args.manifest, manifest)
                book.render(args)
                self.assertEqual(sum(route == "/audio/speech" for route, _ in requests), first_count + 2)
            timeline = json.loads((root / "timeline.json").read_text())
            self.assertEqual([t["speaker"] for t in timeline], ["narrator", "alice", "alice", "narrator"])
            waveform, rate = sf.read(root / "audiobook.wav")
            self.assertEqual(rate, 24000)
            self.assertEqual(len(waveform), round((0.4 + 0.24 + 0.7) * 24000))

    def test_review_blocks_before_any_http(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            manifest = self.manifest()
            manifest["segments"][1]["needs_review"] = True
            book.save_json(path, manifest)
            with patch.object(book, "http") as http:
                with self.assertRaises(ValueError):
                    book.render(argparse.Namespace(manifest=path, manual_review=True))
                http.assert_not_called()

    def test_render_review_cli_is_opt_in(self):
        for flags, expected in (([], False), (["--manual-review"], True), (["--allow-review"], False)):
            with patch("sys.argv", ["audiobook.py", "render"] + flags), patch.object(book, "render") as render:
                book.main()
                self.assertEqual(render.call_args.args[0].manual_review, expected)

    def test_unknown_dialogue_renders_with_kokoro_by_default(self):
        import numpy as np
        import soundfile as sf
        stream = io.BytesIO()
        sf.write(stream, np.ones(2400) * 0.1, 24000, format="WAV")
        requests = []
        def fake_http(base, route, payload=None, key=None, **kwargs):
            self.assertEqual(route, "/audio/speech")  # No character voice preparation for unknowns.
            requests.append(payload)
            return stream.getvalue()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.manifest()
            for segment in manifest["segments"]:
                if segment["kind"] == "dialogue":
                    segment.update(speaker="unknown", needs_review=True)
            book.save_json(root / "manifest.json", manifest)
            args = argparse.Namespace(manifest=root / "manifest.json", output=root, manual_review=False,
                                      seed=42, tts_url="http://test/v1", voice_timeout=10,
                                      pause=0.12, scene_pause=0.7, tts_max_tokens=1024,
                                      target_lufs=-20, true_peak_db=-1, no_narration_context=True)
            with patch.object(book, "http", side_effect=fake_http):
                book.render(args)
            self.assertTrue(all(r["model"] == "kokoro" and r["voice"] == manifest["narrator_voice"]
                                for r in requests))
            self.assertEqual([r["input"] for r in requests], ["Alice said,", "Hello.", "Goodbye,", "she said."])
            timeline = json.loads((root / "timeline.json").read_text())
            self.assertEqual([t["speaker"] for t in timeline], ["narrator", "unknown", "unknown", "narrator"])

    def test_voice_creation_precedes_cloning_and_is_not_in_book(self):
        import numpy as np
        import soundfile as sf

        stream = io.BytesIO()
        sf.write(stream, np.ones(2400) * 0.1, 24000, format="WAV")
        requests, voices = [], set()

        def fake_http(base, route, payload=None, key=None, **kwargs):
            if route.startswith("/voices/"):
                if route.rsplit("/", 1)[1] not in voices:
                    raise RuntimeError("HTTP 404 from /voices/: missing")
                return b'{"status":"ready"}'
            requests.append(payload)
            if payload["model"] == "qwen3-tts":
                voices.add(payload["npc_id"])
            if payload["model"] == "qwen3-tts-0.6b":
                self.assertIn(payload["npc_id"], voices)
            return (stream.getvalue(), {"x-narration-context": "applied"}) if kwargs.get("include_headers") else stream.getvalue()

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            book.save_json(root / "manifest.json", self.manifest())
            args = argparse.Namespace(manifest=root / "manifest.json", output=root, manual_review=False,
                                      seed=42, tts_url="http://test/v1", voice_timeout=10,
                                      pause=0.12, scene_pause=0.7, tts_max_tokens=1024,
                                      target_lufs=-20, true_peak_db=-1, no_narration_context=False)
            with patch.object(book, "http", side_effect=fake_http):
                book.render(args)
            self.assertEqual(requests[0]["model"], "qwen3-tts")
            timeline = json.loads((root / "timeline.json").read_text())
            self.assertEqual(len(timeline), 4)
            self.assertNotIn(requests[0]["input"], [item["text"] for item in timeline])

    def test_llm_validation_retry_and_response_logging(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(llm_model="test-4b", llm_url="http://test/v1",
                                      llm_max_tokens=8192, output=Path(folder))
            values = ["not JSON", '{"assignments":[]}']
            responses = [json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": value}}]}).encode()
                         for value in values]
            with patch.object(book, "http", side_effect=responses) as http:
                annotator = book.Annotator(args)
                result = annotator.ask("test", "Instruction", "Context",
                                       lambda value: book.check_assignments(value, [], set()))
                self.assertEqual(result, {"assignments": []})
                self.assertEqual(http.call_count, 2)
                self.assertEqual(len(list((args.output / "llm").glob("*.json"))), 2)

    def test_truncated_llm_output_stops_without_redundant_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(llm_model="test-4b", llm_url="http://test/v1",
                                      llm_max_tokens=8192, output=Path(folder))
            response = b'{"choices":[{"finish_reason":"length","message":{"content":"{}"}}]}'
            with patch.object(book, "http", return_value=response) as http:
                with self.assertRaisesRegex(ValueError, "truncated"):
                    book.Annotator(args).ask("test", "Instruction", "Context", lambda value: None)
                self.assertEqual(http.call_count, 1)


if __name__ == "__main__":
    unittest.main()
