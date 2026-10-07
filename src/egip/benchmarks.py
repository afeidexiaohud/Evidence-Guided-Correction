from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator

from .alignment import diff_spans
from .spans import Span


def _load_evidence(path: Path) -> list[list[str]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected evidence list, got {type(data)!r}")
    return data


def load_benchmark(input_path: Path, evidence_path: Path, dataset: str) -> list[dict[str, object]]:
    with input_path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    evidence = _load_evidence(evidence_path)
    if len(records) != len(evidence):
        raise ValueError(f"Input/evidence length mismatch: {len(records)} != {len(evidence)}")
    output: list[dict[str, object]] = []
    for index, (record, contexts) in enumerate(zip(records, evidence)):
        claim = str(record["input_claim"])
        target = str(record["gt_claim"])
        if dataset == "fever":
            is_error = record.get("verdict") == "REFUTES"
        elif dataset == "scifact":
            is_error = claim.strip() != target.strip()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        gold_error_spans = diff_spans(claim, target).source_spans if is_error else []
        output.append(
            {
                **record,
                "id": record.get("id", index),
                "dataset": dataset,
                "evidence": list(contexts),
                "is_error": is_error,
                "gold_error_spans": [span.to_dict() for span in gold_error_spans],
            }
        )
    return output


def load_fecdata(path: Path, evidence_setting: str = "gold") -> list[dict[str, object]]:
    evidence_key = "gold_evidence" if evidence_setting == "gold" else "retrieved_evidence"
    output: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            claim = str(record["mutated"])
            target = str(record["original"])
            is_error = record.get("verdict") == "REFUTES"
            evidence = []
            for item in record.get(evidence_key, []):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    evidence.append(f"{item[0]}\t{item[1]}")
                elif item:
                    evidence.append(str(item))
            gold_error_spans = diff_spans(claim, target).source_spans if is_error else []
            output.append(
                {
                    "id": index,
                    "claim_id": record.get("claim_id"),
                    "input_claim": claim,
                    "gt_claim": target,
                    "verdict": record.get("verdict"),
                    "mutation": record.get("mutation"),
                    "dataset": "fever",
                    "evidence": evidence,
                    "is_error": is_error,
                    "gold_error_spans": [span.to_dict() for span in gold_error_spans],
                }
            )
    return output


def dict_spans(items: list[dict[str, object]]) -> list[Span]:
    return [Span(int(item["start"]), int(item["end"]), str(item["text"])) for item in items]


def iter_errors(records: list[dict[str, object]]) -> Iterator[dict[str, object]]:
    return (record for record in records if record["is_error"] and record["gold_error_spans"])
