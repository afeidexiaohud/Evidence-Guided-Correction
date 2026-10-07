from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from egip.ablations import evidence_inputs
from egip.config import load_config
from egip.locator import (
    EvidenceAwareLocator,
    LocatorDataset,
    LocatorExample,
    build_tokenizer,
    collate_locator,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.local.toml")
    parser.add_argument("--dataset", choices=("fever_dev", "fever_test", "scifact_test"), default="fever_test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--backbone", default="cross-encoder/nli-roberta-base")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="bfloat16 reproduces frozen runs; float32 is a separate numerical variant.",
    )
    parser.add_argument(
        "--evidence-mode",
        choices=("normal", "claim_only", "shuffled"),
        default="normal",
        help="Controlled evidence input used at inference time.",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument(
        "--claim-only",
        action="store_true",
        help="Deprecated alias for --evidence-mode claim_only.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config) if args.input is None else None
    input_path = args.input or (Path(config["project"]["root"]) / "artifacts" / "benchmark_candidates" / f"{args.dataset}.jsonl")
    records = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    if args.claim_only and args.evidence_mode != "normal":
        parser.error("--claim-only cannot be combined with --evidence-mode")
    evidence_mode = "claim_only" if args.claim_only else args.evidence_mode
    record_evidence = evidence_inputs(records, mode=evidence_mode, seed=args.shuffle_seed)
    examples = []
    for record, evidence in zip(records, record_evidence):
        for candidate_id, candidate in enumerate(record["candidates"]):
            examples.append(
                LocatorExample(
                    record_id=str(record["id"]),
                    candidate_id=candidate_id,
                    claim=str(record["input_claim"]),
                    evidence=evidence,
                    start=int(candidate["start"]),
                    end=int(candidate["end"]),
                    candidate=str(candidate["text"]),
                    label=0,
                )
            )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone = args.backbone
    checkpoint_args = checkpoint.get("args", {})
    tokenizer = build_tokenizer(str(args.tokenizer) if args.tokenizer else backbone)
    model = EvidenceAwareLocator(backbone, tokenizer_size=len(tokenizer), pretrained=False)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = LocatorDataset(
        examples,
        tokenizer,
        max_length=args.max_length,
        claim_only=evidence_mode == "claim_only",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_locator, pad_token_id=tokenizer.pad_token_id),
        num_workers=2,
    )
    scores: dict[tuple[str, int], float] = {}
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda" and args.precision == "bfloat16",
            ):
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
            probabilities = torch.sigmoid(logits).cpu().tolist()
            for record_id, candidate_id, score in zip(
                batch["record_ids"], batch["candidate_ids"], probabilities
            ):
                scores[(str(record_id), int(candidate_id))] = score
    output_path = args.output or input_path.with_name(f"{args.dataset}.scored.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            for candidate_id, candidate in enumerate(record["candidates"]):
                candidate["error_probability"] = scores[(str(record["id"]), candidate_id)]
            record["locator_evidence_mode"] = evidence_mode
            record["locator_shuffle_seed"] = args.shuffle_seed if evidence_mode == "shuffled" else None
            record["locator_training_seed"] = checkpoint_args.get("seed")
            record["locator_training_setting"] = checkpoint_args.get("setting")
            record["locator_claim_only_training"] = checkpoint_args.get("claim_only")
            record["locator_scoring_precision"] = args.precision if device.type == "cuda" else "float32"
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "records": len(records),
                "examples": len(examples),
                "evidence_mode": evidence_mode,
                "shuffle_seed": args.shuffle_seed if evidence_mode == "shuffled" else None,
                "training_seed": checkpoint_args.get("seed"),
                "training_setting": checkpoint_args.get("setting"),
                "claim_only_training": checkpoint_args.get("claim_only"),
                "precision": args.precision if device.type == "cuda" else "float32",
                "output": str(output_path),
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
