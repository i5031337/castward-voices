"""Score two attribution manifests against fixed reference labels."""
import argparse
import json
from pathlib import Path

from audiobook import HERE, save_json, validate_manifest


def compare(baseline, candidate, reference, allow_model_change=False):
    for manifest in (baseline, candidate):
        validate_manifest(manifest)
        if manifest["source_sha256"] != reference["source_sha256"]:
            raise ValueError("Both manifests must match the reference source hash")
    if baseline["llm_model"] != candidate["llm_model"] and not allow_model_change:
        raise ValueError("A controlled prompt comparison requires the same model")
    if baseline["characters"] != candidate["characters"]:
        raise ValueError("A controlled prompt comparison requires the same cast")
    quotes = [{s["quote_id"]: s for s in m["segments"] if s["kind"] == "dialogue"}
              for m in (baseline, candidate)]
    if set(quotes[0]) != set(quotes[1]) or set(quotes[0]) != set(reference["quotes"]):
        raise ValueError("Both manifests and the reference must cover exactly the same quote IDs")
    report = {"reference_method": reference["method"], "reference_policy": reference["policy"],
              "model": baseline["llm_model"] if baseline["llm_model"] == candidate["llm_model"] else None,
              "baseline_model": baseline["llm_model"], "candidate_model": candidate["llm_model"],
              "model_changed": baseline["llm_model"] != candidate["llm_model"],
              "source_sha256": reference["source_sha256"],
              "total_quotes": len(quotes[0]), "scored_quotes": 0, "baseline_correct": 0,
              "candidate_correct": 0, "fixes": 0, "regressions": 0,
              "baseline_high_confidence_errors": 0, "candidate_high_confidence_errors": 0,
              "scenes": {}, "changes": [], "candidate_errors": []}
    for quote_id, original in quotes[0].items():
        revised = quotes[1][quote_id]
        if (original["start"], original["end"], original["scene"]) != (revised["start"], revised["end"], revised["scene"]):
            raise ValueError(f"Quote span changed: {quote_id}")
        label = reference["quotes"][quote_id]
        expected = label["speaker"]
        scored = label["scored"]
        old_correct, new_correct = original["speaker"] == expected, revised["speaker"] == expected
        detail = {"quote_id": quote_id, "scene": original["scene"],
                  "text": baseline["source"][original["start"]:original["end"]],
                  "reference_speaker": expected, "scored": scored,
                  "baseline_speaker": original["speaker"], "candidate_speaker": revised["speaker"],
                  "baseline_reason": original.get("reason"), "candidate_reason": revised.get("reason"),
                  "baseline_confidence": original.get("confidence"), "candidate_confidence": revised.get("confidence")}
        if original["speaker"] != revised["speaker"]:
            report["changes"].append(detail)
        if not scored:
            continue
        scene = report["scenes"].setdefault(str(original["scene"]),
                     {"scored": 0, "baseline_correct": 0, "candidate_correct": 0})
        scene["scored"] += 1
        report["scored_quotes"] += 1
        for name, correct, assignment in (("baseline", old_correct, original), ("candidate", new_correct, revised)):
            report[name + "_correct"] += int(correct)
            scene[name + "_correct"] += int(correct)
            report[name + "_high_confidence_errors"] += int(not correct and assignment.get("confidence") == "high")
        report["fixes"] += int(not old_correct and new_correct)
        report["regressions"] += int(old_correct and not new_correct)
        if not new_correct:
            report["candidate_errors"].append(detail)
    return report


def request_metrics(folder):
    paths = list((folder / "llm").glob("scene-*.json")) + list((folder / "llm").glob("fallback-*.json"))
    logs = [json.loads(path.read_text()) for path in paths]
    return {"calls": len(logs), "fallback_calls": sum(path.name.startswith("fallback-") for path in paths),
            "seconds": round(sum(log["seconds"] for log in logs), 3),
            "prompt_tokens": sum(log["response"]["usage"]["prompt_tokens"] for log in logs),
            "cached_prompt_tokens": sum(log["response"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                        for log in logs),
            "completion_tokens": sum(log["response"]["usage"]["completion_tokens"] for log in logs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=HERE / "reference_labels.json")
    parser.add_argument("--output", type=Path, default=HERE / "out/prompt-comparison.json")
    parser.add_argument("--allow-model-change", action="store_true",
                        help="Explicitly permit comparing models while preserving source and cast checks")
    args = parser.parse_args()
    try:
        report = compare(json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text()),
                         json.loads(args.reference.read_text()), allow_model_change=args.allow_model_change)
        report["baseline_manifest"] = str(args.baseline)
        report["candidate_manifest"] = str(args.candidate)
        report["baseline_batch_size"] = json.loads(args.baseline.read_text()).get("quote_batch_size", 0)
        report["candidate_batch_size"] = json.loads(args.candidate.read_text()).get("quote_batch_size", 0)
        report["baseline_reasoning_effort"] = json.loads(args.baseline.read_text()).get("reasoning_effort")
        report["candidate_reasoning_effort"] = json.loads(args.candidate.read_text()).get("reasoning_effort")
        report["baseline_fallback_reasoning_effort"] = json.loads(args.baseline.read_text()).get("fallback_reasoning_effort")
        report["candidate_fallback_reasoning_effort"] = json.loads(args.candidate.read_text()).get("fallback_reasoning_effort")
        report["baseline_requests"] = request_metrics(args.baseline.parent)
        report["candidate_requests"] = request_metrics(args.candidate.parent)
        save_json(args.output, report)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    for scene, scores in report["scenes"].items():
        print(f"Scene {scene}: {scores['baseline_correct']}/{scores['scored']} -> "
              f"{scores['candidate_correct']}/{scores['scored']}")
    print(f"Total: {report['baseline_correct']}/{report['scored_quotes']} -> "
          f"{report['candidate_correct']}/{report['scored_quotes']}; "
          f"{report['fixes']} fixes, {report['regressions']} regressions")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
