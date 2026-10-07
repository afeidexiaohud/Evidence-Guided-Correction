from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .alignment import diff_spans, indices_to_spans, masked_source_token_indices, tokenize


SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "").strip()).casefold()


def evidence_text(record: dict[str, object], setting: str) -> list[str]:
    key = "gold_evidence" if setting == "gold" else "retrieved_evidence"
    items = record.get(key) or []
    evidence: list[str] = []
    for item in items:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            title, content = item[0], item[1]
            evidence.append(f"{title}\t{content}".strip())
        elif item:
            evidence.append(str(item).strip())
    return evidence


def load_text_index(paths: Iterable[Path]) -> set[str]:
    index: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                for key in ("input_claim", "gt_claim"):
                    if record.get(key):
                        index.add(normalize_text(record[key]))
    return index


def quality_record(
    record: dict[str, object],
    *,
    setting: str,
    test_text_index: set[str],
    max_edit_ratio: float,
    max_target_spans: int,
    min_anchor_accuracy: float,
    min_trace_coverage: float,
) -> dict[str, object]:
    source = str(record.get("original") or "").strip()
    target = str(record.get("mutated") or "").strip()
    masked_source = str(record.get("masked_src") or "").strip()
    diff = diff_spans(source, target)
    mask_indices, anchor_accuracy = masked_source_token_indices(source, masked_source)
    source_edit = diff.source_changed_tokens
    trace_overlap = len(source_edit & mask_indices)
    trace_coverage = trace_overlap / len(source_edit) if source_edit else 0.0
    trace_precision = trace_overlap / len(mask_indices) if mask_indices else 0.0
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    changed = len(source_edit) + len(diff.target_changed_tokens)
    edit_ratio = changed / max(1, len(source_tokens) + len(target_tokens))
    leakage = normalize_text(source) in test_text_index or normalize_text(target) in test_text_index
    reasons: list[str] = []
    if not source or not target or source == target:
        reasons.append("no_change")
    if not masked_source or not mask_indices:
        reasons.append("no_trace")
    if anchor_accuracy < min_anchor_accuracy:
        reasons.append("trace_alignment")
    if not diff.target_spans:
        reasons.append("no_target_span")
    if len(diff.target_spans) > max_target_spans:
        reasons.append("too_many_target_spans")
    if edit_ratio > max_edit_ratio:
        reasons.append("non_local_edit")
    if trace_coverage < min_trace_coverage:
        reasons.append("trace_diff_mismatch")
    if leakage:
        reasons.append("benchmark_text_overlap")
    evidence = evidence_text(record, setting)
    if not evidence:
        reasons.append("missing_evidence")

    source_group_id = str(record.get("original_id") or record.get("claim_id") or source)
    split_group = hashlib.sha1(normalize_text(source).encode("utf-8")).hexdigest()
    stable_id = hashlib.sha1(
        f"{setting}\0{normalize_text(source)}\0{normalize_text(target)}".encode("utf-8")
    ).hexdigest()
    return {
        "id": stable_id,
        "group_id": source_group_id,
        "split_group": split_group,
        "setting": setting,
        "claim_plus": source,
        "claim_minus": target,
        "masked_source": masked_source,
        "evidence": evidence,
        "source_mask_spans": [span.to_dict() for span in indices_to_spans(source, mask_indices)],
        "source_diff_spans": [span.to_dict() for span in diff.source_spans],
        "target_error_spans": [span.to_dict() for span in diff.target_spans],
        "operations": diff.operations,
        "metrics": {
            "anchor_accuracy": anchor_accuracy,
            "trace_coverage": trace_coverage,
            "trace_precision": trace_precision,
            "edit_ratio": edit_ratio,
            "num_source_masks": len(mask_indices),
            "num_source_edits": len(source_edit),
            "num_target_edits": len(diff.target_changed_tokens),
        },
        "metadata": {
            key: record.get(key)
            for key in ("claim_id", "original_id", "sentence_id", "mutation", "verdict")
        },
        "accepted_structural": not reasons,
        "rejection_reasons": reasons,
    }


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def split_for_group(group_id: str) -> str:
    bucket = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "heldout"


def export_traces(
    input_path: Path,
    output_dir: Path,
    *,
    setting: str,
    test_text_index: set[str],
    max_edit_ratio: float = 0.35,
    max_target_spans: int = 3,
    min_anchor_accuracy: float = 0.95,
    min_trace_coverage: float = 0.8,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        split: (output_dir / f"{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "dev", "heldout")
    }
    rejected = (output_dir / "rejected.jsonl").open("w", encoding="utf-8")
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    try:
        for raw in iter_jsonl(input_path):
            counts["total"] += 1
            item = quality_record(
                raw,
                setting=setting,
                test_text_index=test_text_index,
                max_edit_ratio=max_edit_ratio,
                max_target_spans=max_target_spans,
                min_anchor_accuracy=min_anchor_accuracy,
                min_trace_coverage=min_trace_coverage,
            )
            if item["id"] in seen:
                counts["duplicate"] += 1
                continue
            seen.add(item["id"])
            if not item["accepted_structural"]:
                counts["rejected"] += 1
                for reason in item["rejection_reasons"]:
                    reasons[reason] += 1
                rejected.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue
            split = split_for_group(str(item["split_group"]))
            item["split"] = split
            handles[split].write(json.dumps(item, ensure_ascii=False) + "\n")
            counts["accepted"] += 1
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()
        rejected.close()
    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "setting": setting,
        "thresholds": {
            "max_edit_ratio": max_edit_ratio,
            "max_target_spans": max_target_spans,
            "min_anchor_accuracy": min_anchor_accuracy,
            "min_trace_coverage": min_trace_coverage,
        },
        "counts": dict(counts),
        "rejection_reasons": dict(reasons),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
