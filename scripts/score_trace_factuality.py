from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from egip.config import load_config
from egip.factuality import FORMAT_VERSION, life_format_evidence


class TraceDataset(Dataset):
    def __init__(self, records: list[dict[str, object]], tokenizer, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records) * 2

    def __getitem__(self, index: int):
        record_index = index // 2
        is_minus = index % 2 == 1
        record = self.records[record_index]
        claim = record["claim_minus"] if is_minus else record["claim_plus"]
        evidence = life_format_evidence(record["evidence"])
        encoded = self.tokenizer(
            claim,
            evidence,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return record_index, int(is_minus), encoded


def collate(batch, pad_token_id: int):
    record_ids = torch.tensor([item[0] for item in batch], dtype=torch.long)
    is_minus = torch.tensor([item[1] for item in batch], dtype=torch.long)
    max_length = max(len(item[2]["input_ids"]) for item in batch)
    input_ids = []
    attention_masks = []
    for _, _, encoded in batch:
        padding = max_length - len(encoded["input_ids"])
        input_ids.append(encoded["input_ids"] + [pad_token_id] * padding)
        attention_masks.append(encoded["attention_mask"] + [0] * padding)
    return {
        "record_ids": record_ids,
        "is_minus": is_minus,
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.local.toml")
    parser.add_argument("--setting", choices=("gold", "retrieved"), default="gold")
    parser.add_argument("--source", choices=("raw", "filtered", "diff_only"), default="filtered")
    parser.add_argument("--split", choices=("train", "dev", "heldout"), required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    trace_dir = Path(config["project"]["root"]) / "artifacts" / "trajectories" / f"{args.setting}_{args.source}"
    input_path = trace_dir / f"{args.split}.jsonl"
    output_path = trace_dir / f"{args.split}.scored.jsonl"
    summary_path = trace_dir / f"{args.split}.factuality_summary.json"
    lock_handle = (trace_dir / f"{args.split}.score.lock").open("w", encoding="utf-8")
    fcntl.flock(lock_handle, fcntl.LOCK_EX)
    source_count = sum(1 for line in input_path.open("rb") if line.strip())
    if args.limit is None and output_path.exists() and summary_path.exists():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        scored_count = sum(1 for line in output_path.open("rb") if line.strip())
        if (
            prior.get("records") == source_count == scored_count
            and prior.get("format_version") == FORMAT_VERSION
        ):
            print(json.dumps({**prior, "status": "already_complete"}, ensure_ascii=False, indent=2))
            return
    records = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    if args.limit is not None:
        records = records[: args.limit]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    reusable = [
        bool(
            record.get("fact_scores")
            and "accepted_fact" in record
            and float(record["fact_scores"].get("threshold", -1)) == args.threshold
        )
        for record in records
    ]
    pending_indices = [index for index, is_reusable in enumerate(reusable) if not is_reusable]
    pending_records = [records[index] for index in pending_indices]
    scores: list[list[list[float] | None]] = [[None, None] for _ in records]
    if pending_records:
        model_path = config["models"][f"life_verifier_{args.setting}"]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        dataset = TraceDataset(pending_records, tokenizer, args.max_length)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        )
        with torch.no_grad():
            for batch in loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
                for pending_id, is_minus, probability in zip(
                    batch["record_ids"].tolist(), batch["is_minus"].tolist(), probabilities
                ):
                    scores[pending_indices[pending_id]][is_minus] = probability

    accepted = 0
    temporary_path = output_path.with_name(f"{output_path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for is_reusable, record, (plus, minus) in zip(reusable, records, scores):
            if is_reusable:
                accepted += int(record["accepted_fact"])
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue
            if plus is None or minus is None:
                raise RuntimeError(f"Missing factuality score for record {record['id']}")
            plus_support = plus[0]
            minus_refute = minus[1]
            fact_valid = (
                plus_support >= args.threshold
                and plus_support == max(plus)
                and minus_refute >= args.threshold
                and minus_refute == max(minus)
            )
            record["fact_scores"] = {
                "plus": {"support": plus[0], "refute": plus[1], "nei": plus[2]},
                "minus": {"support": minus[0], "refute": minus[1], "nei": minus[2]},
                "threshold": args.threshold,
            }
            record["accepted_fact"] = fact_valid
            accepted += int(fact_valid)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "records": len(records),
        "accepted": accepted,
        "acceptance_rate": accepted / len(records) if records else 0.0,
        "reused_scores": sum(reusable),
        "newly_scored": len(pending_records),
        "device": str(device),
        "threshold": args.threshold,
        "format_version": FORMAT_VERSION,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
