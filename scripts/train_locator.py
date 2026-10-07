from __future__ import annotations

import argparse
import json
import math
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from egip.config import load_config
from egip.locator import (
    EvidenceAwareLocator,
    LocatorDataset,
    build_tokenizer,
    collate_locator,
    flatten_records,
    load_locator_records,
    ranking_metrics,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            scores = torch.sigmoid(logits).cpu().tolist()
            for record_id, candidate_id, label, score in zip(
                batch["record_ids"], batch["candidate_ids"], batch["labels"].tolist(), scores
            ):
                rows.append(
                    {
                        "record_id": record_id,
                        "candidate_id": candidate_id,
                        "label": int(label),
                        "score": score,
                    }
                )
    return ranking_metrics(rows), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.local.toml")
    parser.add_argument("--setting", default="gold_filtered")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-negatives", type=int, default=8)
    parser.add_argument("--claim-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    data_dir = Path(config["project"]["root"]) / "artifacts" / "locator_data" / args.setting
    train_records = load_locator_records(data_dir / "train.jsonl")
    dev_records = load_locator_records(data_dir / "dev.jsonl")
    heldout_records = load_locator_records(data_dir / "heldout.jsonl")
    train_examples = flatten_records(
        train_records, max_negatives_per_record=args.max_negatives, seed=args.seed
    )
    dev_examples = flatten_records(dev_records)
    heldout_examples = flatten_records(heldout_records)
    backbone = config["models"]["locator_backbone"]
    tokenizer = build_tokenizer(backbone)
    train_dataset = LocatorDataset(
        train_examples, tokenizer, max_length=args.max_length, claim_only=args.claim_only
    )
    dev_dataset = LocatorDataset(
        dev_examples, tokenizer, max_length=args.max_length, claim_only=args.claim_only
    )
    heldout_dataset = LocatorDataset(
        heldout_examples, tokenizer, max_length=args.max_length, claim_only=args.claim_only
    )
    collate = partial(collate_locator, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=2
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate, num_workers=2
    )
    heldout_loader = DataLoader(
        heldout_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=2,
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = EvidenceAwareLocator(backbone, tokenizer_size=len(tokenizer)).to(device)
    positives = sum(example.label for example in train_examples)
    negatives = len(train_examples) - positives
    positive_weight = torch.tensor([negatives / max(1, positives)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps = math.ceil(len(train_loader) / args.gradient_accumulation) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(update_steps * args.warmup_ratio),
        num_training_steps=update_steps,
    )
    output_dir = Path(args.output_dir or (Path(config["project"]["root"]) / "runs" / f"locator_{'claim_only' if args.claim_only else 'evidence'}_seed{args.seed}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_mrr = -1.0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                loss = criterion(logits, batch["labels"].to(device)) / args.gradient_accumulation
            loss.backward()
            total_loss += loss.item() * args.gradient_accumulation
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        metrics, rows = evaluate(model, dev_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            **metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, ensure_ascii=False), flush=True)
        if metrics["mrr"] > best_mrr:
            best_mrr = metrics["mrr"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "backbone": backbone,
                    "tokenizer_size": len(tokenizer),
                    "args": vars(args),
                    "metrics": metrics,
                },
                output_dir / "best.pt",
            )
            tokenizer.save_pretrained(output_dir / "tokenizer")
            (output_dir / "dev_predictions.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model"])
    model.to(device)
    heldout_metrics, heldout_rows = evaluate(model, heldout_loader, device)
    (output_dir / "heldout_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in heldout_rows), encoding="utf-8"
    )
    summary = {
        "device": str(device),
        "seed": args.seed,
        "claim_only": args.claim_only,
        "train_records": len(train_records),
        "train_examples": len(train_examples),
        "dev_records": len(dev_records),
        "dev_examples": len(dev_examples),
        "heldout_records": len(heldout_records),
        "heldout_examples": len(heldout_examples),
        "heldout_metrics": heldout_metrics,
        "positive_weight": positive_weight.item(),
        "best_mrr": best_mrr,
        "history": history,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
