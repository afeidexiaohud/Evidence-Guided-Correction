"""CPU walkthrough: synthetic priorities, TF-IDF vectors, real planning functions.

This is a fictional example, not neural inference or a correction benchmark.
No trained model, external service, private record, or generation output is used.
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from egip.candidates import apply_candidate_mask, build_candidate_pool, mmr_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()

    claim = "Atlas SDK supports legacy Python and cannot run on Linux."
    evidence = "Atlas SDK requires modern Python and supports Linux."
    # Illustrative scores; the real pipeline obtains priorities from a trained
    # evidence-conditioned model. No learning or evidence scoring occurs here.
    demo_priorities = {
        "atlas sdk": 0.05,
        "supports": 0.10,
        "legacy python": 0.98,
        "cannot run on linux": 0.96,
    }
    pool = build_candidate_pool(claim)
    by_phrase = {candidate.phrase: candidate for candidate in pool}
    candidates = [by_phrase[phrase] for phrase in demo_priorities]
    vectors = TfidfVectorizer().fit_transform(
        [claim] + [candidate.phrase for candidate in candidates]
    ).toarray()
    order, _ = mmr_rank(
        candidates,
        vectors[0],
        vectors[1:],
        top_k=args.budget,
        diversity=0.3,
        error_probabilities=np.array(list(demo_priorities.values())),
        error_weight=0.9,
    )

    print("ILLUSTRATIVE WALKTHROUGH: manual priorities + TF-IDF; no model inference")
    print(f"Claim:    {claim}")
    print(f"Evidence: {evidence}")
    print(f"Constructed {len(pool)} candidates; showing a subset of {len(candidates)}.")
    print(f"Correction attempt budget: K = {args.budget}")
    for attempt, index in enumerate(order, start=1):
        candidate = candidates[index]
        masked = apply_candidate_mask(claim, candidate, mode="legacy_global")
        print(f"\nAttempt {attempt}: {candidate.text!r}")
        print(f"  Illustrative priority: {demo_priorities[candidate.phrase]:.2f}")
        print(f"  Masked original: {masked}")
    print("\nNext in the full pipeline: generate independent revisions, then select")
    print("from those revisions and the unchanged input. Generation is not run here.")


if __name__ == "__main__":
    main()
