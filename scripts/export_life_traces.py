from __future__ import annotations

import argparse
import json
from pathlib import Path

from egip.config import load_config
from egip.trace_data import export_traces, load_text_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.local.toml")
    parser.add_argument("--setting", choices=("gold", "retrieved"), default="gold")
    parser.add_argument("--source", choices=("raw", "filtered"), default="filtered")
    parser.add_argument("--max-edit-ratio", type=float, default=0.35)
    parser.add_argument("--max-target-spans", type=int, default=3)
    parser.add_argument("--min-anchor-accuracy", type=float, default=0.95)
    parser.add_argument("--min-trace-coverage", type=float, default=0.8)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(config["data"][f"life_{args.setting}_{args.source}"])
    benchmark_paths = [
        Path(config["data"]["fever_test"]),
        Path(config["data"]["scifact_test"]),
    ]
    test_index = load_text_index(benchmark_paths)
    output_dir = Path(config["project"]["root"]) / "artifacts" / "trajectories" / f"{args.setting}_{args.source}"
    summary = export_traces(
        input_path,
        output_dir,
        setting=args.setting,
        test_text_index=test_index,
        max_edit_ratio=args.max_edit_ratio,
        max_target_spans=args.max_target_spans,
        min_anchor_accuracy=args.min_anchor_accuracy,
        min_trace_coverage=args.min_trace_coverage,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
