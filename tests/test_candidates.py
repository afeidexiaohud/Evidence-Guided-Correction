from egip.candidates import (
    Candidate,
    apply_candidate_mask,
    build_candidate_pool,
    mmr_rank,
    m2c_mask_candidate,
    mask_candidate,
    phrase_occurrences,
)

import numpy as np
import pytest


def test_phrase_offsets_are_span_specific() -> None:
    text = "Paris and Paris are cities."
    spans = phrase_occurrences(text, "paris")
    assert [(span.start, span.end) for span in spans] == [(0, 5), (10, 15)]
    assert mask_candidate(text, spans[1]) == "Paris and [MASK] are cities."


def test_candidate_pool_contains_numeric_error() -> None:
    text = "The war ended in 301 BC."
    candidates = build_candidate_pool(text)
    assert any(candidate.text == "301" for candidate in candidates)


def test_m2c_mapping_keeps_terminal_punctuation_and_replaces_all() -> None:
    text = "Paris city follows Paris."
    candidate = next(candidate for candidate in build_candidate_pool(text) if candidate.phrase == "paris")
    assert candidate.text == "Paris"
    assert m2c_mask_candidate(text, candidate) == "[MASK] city follows [MASK]."
    terminal = next(candidate for candidate in build_candidate_pool(text) if candidate.phrase == "follows paris")
    assert terminal.text == "follows Paris."
    assert m2c_mask_candidate(text, terminal) == "Paris city [MASK]"


def test_explicit_span_mask_mode_does_not_replace_other_occurrences() -> None:
    text = "Paris and Paris are cities."
    candidate = phrase_occurrences(text, "paris")[1]
    assert apply_candidate_mask(text, candidate, mode="legacy_global") == (
        "[MASK] and [MASK] are cities."
    )
    assert apply_candidate_mask(text, candidate, mode="span") == "Paris and [MASK] are cities."
