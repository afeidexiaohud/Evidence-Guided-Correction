from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from egip.candidates import build_candidate_pool
from egip.config import load_config
from egip.localization import token_iou
from egip.spans import Span


def span_from_dict(item: dict[str, object]) -> Span:
    return Span(int(item["start"]), int(item["end"]), str(item["text"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.local.toml")
    parser.add_argument("--setting", choices=("gold", "retrieved"), default="gold")
    parser.add_argument("--source", choices=("raw", "filtered", "diff_only"), default="filtered")
    parser.add_argument("--split", choices=("train", "dev", "heldout"), required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--allow-unscored", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    trace_dir = Path(config["project"]["root"]) / "artifacts" / "trajectories" / f"{args.setting}_{args.source}"
    scored_path = trace_dir / f"{args.split}.scored.jsonl"
    source_path = scored_path if scored_path.exists() and not args.allow_unscored else trace_dir / f"{args.split}.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_dir = Path(config["project"]["root"]) / "artifacts" / "locator_data" / f"{args.setting}_{args.source}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.split}.jsonl"
    counts: Counter[str] = Counter()
    with source_path.open(encoding="utf-8") as source_handle, output_path.open("w", encoding="utf-8") as output_handle:
        for line in source_handle:
            trace = json.loads(line)
            counts["input_traces"] += 1
            if not args.allow_unscored and not trace.get("accepted_fact", False):
                counts["fact_rejected"] += 1
                continue
            claim = trace["claim_minus"]
            gold_spans = [span_from_dict(item) for item in trace["target_error_spans"]]
            candidates = build_candidate_pool(claim)
            candidate_items = []
            positive = 0
            for candidate in candidates:
                label = int(any(token_iou(claim, candidate.span, gold) >= args.iou_threshold for gold in gold_spans))
                positive += label
                candidate_items.append({**candidate.to_dict(), "label": label})
            if not candidate_items:
                counts["empty_candidate_pool"] += 1
                continue
            if not positive:
                counts["candidate_uncovered"] += 1
                continue
            record = {
                "id": trace["id"],
                "group_id": trace["group_id"],
                "split_group": trace["split_group"],
                "split": args.split,
                "supervision_variant": trace.get("supervision_variant", f"{args.setting}_{args.source}"),
                "diff_only_origin": trace.get("diff_only_origin"),
                "claim": claim,
                "evidence": trace["evidence"],
                "gold_error_spans": trace["target_error_spans"],
                "candidates": candidate_items,
                "metadata": trace["metadata"],
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["accepted"] += 1
            counts["positive_candidates"] += positive
            counts["negative_candidates"] += len(candidate_items) - positive
    summary = {
        "input": str(source_path),
        "output": str(output_path),
        "split": args.split,
        "iou_threshold": args.iou_threshold,
        "counts": dict(counts),
        "coverage": counts["accepted"] / max(1, counts["input_traces"] - counts["fact_rejected"]),
    }
    (output_dir / f"{args.split}.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
