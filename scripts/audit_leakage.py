from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from egip.config import load_config
from egip.trace_data import normalize_text


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.local.toml")
    parser.add_argument("--setting", choices=("gold", "retrieved"), default="gold")
    parser.add_argument("--source", choices=("raw", "filtered", "diff_only"), default="filtered")
    parser.add_argument("--output", default="artifacts/leakage_audit.json")
    args = parser.parse_args()
    config = load_config(args.config)
    trace_dir = Path(config["project"]["root"]) / "artifacts" / "trajectories" / f"{args.setting}_{args.source}"
    splits = {split: read_jsonl(trace_dir / f"{split}.jsonl") for split in ("train", "dev", "heldout")}

    benchmark_texts: set[str] = set()
    benchmark_claim_ids: set[str] = set()
    for key in ("fever_dev", "fever_test", "scifact_test"):
        for record in read_jsonl(Path(config["data"][key])):
            if key.startswith("fever") and record.get("claim_id") is not None:
                benchmark_claim_ids.add(str(record["claim_id"]))
            for field in ("input_claim", "gt_claim", "mutated", "original"):
                if record.get(field):
                    benchmark_texts.add(normalize_text(record[field]))

    groups: defaultdict[str, set[str]] = defaultdict(set)
    source_texts: defaultdict[str, set[str]] = defaultdict(set)
    pair_ids: defaultdict[str, set[str]] = defaultdict(set)
    benchmark_overlap = []
    synthetic_claim_ids: set[str] = set()
    for split, records in splits.items():
        for record in records:
            groups[str(record["group_id"])].add(split)
            source_texts[normalize_text(record["claim_plus"])].add(split)
            pair_ids[str(record["id"])].add(split)
            for field in ("claim_plus", "claim_minus"):
                if normalize_text(record[field]) in benchmark_texts:
                    benchmark_overlap.append({"split": split, "id": record["id"], "field": field})
            for field in ("claim_id", "original_id"):
                value = record.get("metadata", {}).get(field)
                if value is not None:
                    synthetic_claim_ids.add(str(value))

    result = {
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "group_cross_split": sum(len(values) > 1 for values in groups.values()),
        "source_cross_split": sum(len(values) > 1 for values in source_texts.values()),
        "pair_cross_split": sum(len(values) > 1 for values in pair_ids.values()),
        "benchmark_exact_overlap": benchmark_overlap,
        "benchmark_claim_id_overlap": sorted(benchmark_claim_ids & synthetic_claim_ids),
        "passed": not benchmark_overlap
        and not (benchmark_claim_ids & synthetic_claim_ids)
        and not any(len(values) > 1 for values in groups.values())
        and not any(len(values) > 1 for values in source_texts.values())
        and not any(len(values) > 1 for values in pair_ids.values()),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
