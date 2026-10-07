from __future__ import annotations

import random


def evidence_inputs(
    records: list[dict[str, object]],
    *,
    mode: str,
    seed: int,
) -> list[str]:
    """Return one evidence string per record for a controlled input ablation.

    ``shuffled`` uses a seeded cyclic permutation. This guarantees that no
    record receives its own evidence (unlike a plain random permutation, which
    can contain fixed points) while retaining the benchmark evidence-length
    distribution.
    """

    evidence = [" ### ".join(str(item) for item in record["evidence"]) for record in records]
    if mode in {"normal", "claim_only"}:
        return evidence
    if mode != "shuffled":
        raise ValueError(f"Unknown evidence mode: {mode}")
    if len(records) < 2:
        raise ValueError("Shuffled evidence requires at least two records")
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    shuffled = [""] * len(records)
    for position, record_index in enumerate(order):
        donor_index = order[(position + 1) % len(order)]
        shuffled[record_index] = evidence[donor_index]
    return shuffled
