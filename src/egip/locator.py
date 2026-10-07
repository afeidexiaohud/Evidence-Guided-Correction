from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedTokenizerFast


CAND_START = "<cand>"
CAND_END = "</cand>"


@dataclass(frozen=True)
class LocatorExample:
    record_id: str
    candidate_id: int
    claim: str
    evidence: str
    start: int
    end: int
    candidate: str
    label: int

    @property
    def marked_claim(self) -> str:
        return (
            self.claim[: self.start]
            + CAND_START
            + self.claim[self.start : self.end]
            + CAND_END
            + self.claim[self.end :]
            + f" Candidate: {self.candidate}"
        )


def load_locator_records(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def flatten_records(
    records: list[dict[str, object]],
    *,
    max_negatives_per_record: int | None = None,
    seed: int = 42,
) -> list[LocatorExample]:
    examples: list[LocatorExample] = []
    for record in records:
        candidates = list(enumerate(record["candidates"]))
        positives = [(index, candidate) for index, candidate in candidates if candidate["label"]]
        negatives = [(index, candidate) for index, candidate in candidates if not candidate["label"]]
        if max_negatives_per_record is not None and len(negatives) > max_negatives_per_record:
            local_seed = int(hashlib.sha1(f"{seed}:{record['id']}".encode()).hexdigest()[:8], 16)
            negatives = random.Random(local_seed).sample(negatives, max_negatives_per_record)
        for index, candidate in positives + negatives:
            examples.append(
                LocatorExample(
                    record_id=str(record["id"]),
                    candidate_id=index,
                    claim=str(record["claim"]),
                    evidence=" ### ".join(record["evidence"]),
                    start=int(candidate["start"]),
                    end=int(candidate["end"]),
                    candidate=str(candidate["text"]),
                    label=int(candidate["label"]),
                )
            )
    return examples


class LocatorDataset(Dataset):
    def __init__(
        self,
        examples: list[LocatorExample],
        tokenizer,
        *,
        max_length: int = 512,
        claim_only: bool = False,
    ) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.claim_only = claim_only

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example.marked_claim,
            None if self.claim_only else example.evidence,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        return {
            "record_id": example.record_id,
            "candidate_id": example.candidate_id,
            "label": float(example.label),
            **encoded,
        }


def collate_locator(batch: list[dict[str, object]], pad_token_id: int) -> dict[str, object]:
    max_length = max(len(item["input_ids"]) for item in batch)
    input_ids = []
    attention_mask = []
    for item in batch:
        padding = max_length - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
    return {
        "record_ids": [item["record_id"] for item in batch],
        "candidate_ids": [item["candidate_id"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class EvidenceAwareLocator(nn.Module):
    def __init__(self, backbone_path: str, tokenizer_size: int | None = None, *, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = (AutoModel.from_pretrained(backbone_path, add_pooling_layer=False) if pretrained else
                         AutoModel.from_config(AutoConfig.from_pretrained(backbone_path), add_pooling_layer=False))
        if tokenizer_size is not None:
            self.backbone.resize_token_embeddings(tokenizer_size)
        hidden_size = self.backbone.config.hidden_size
        dropout_prob = getattr(self.backbone.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled)).squeeze(-1)


def build_tokenizer(backbone_path: str):
    tokenizer_dir = Path(backbone_path)
    if tokenizer_dir.is_dir() and (tokenizer_dir / "tokenizer.json").exists():
        # Frozen training used a tokenizers backend despite use_fast=False.
        # Load its saved BPE/postprocessor verbatim across Transformers versions.
        settings = json.loads((tokenizer_dir / "tokenizer_config.json").read_text())
        keys = ("bos_token", "cls_token", "eos_token", "pad_token", "sep_token", "mask_token", "unk_token", "model_max_length")
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_dir / "tokenizer.json"),
                                           model_input_names=["input_ids", "attention_mask"],
                                           **{k: settings[k] for k in keys if k in settings})
    else:
        tokenizer = AutoTokenizer.from_pretrained(backbone_path, use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [CAND_START, CAND_END]})
    return tokenizer


def ranking_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["record_id"])].append(row)
    hit_at_1 = []
    recall_at_3 = []
    reciprocal_ranks = []
    for candidates in grouped.values():
        ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
        labels = [int(item["label"]) for item in ranked]
        positives = sum(labels)
        if not positives:
            continue
        hit_at_1.append(float(labels[0]))
        recall_at_3.append(sum(labels[:3]) / positives)
        first_positive = labels.index(1)
        reciprocal_ranks.append(1.0 / (first_positive + 1))
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "records": float(len(reciprocal_ranks)),
        "hit_at_1": mean(hit_at_1),
        "recall_at_3": mean(recall_at_3),
        "mrr": mean(reciprocal_ranks),
    }
