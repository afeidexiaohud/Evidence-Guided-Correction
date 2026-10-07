from __future__ import annotations

from .alignment import tokenize
from .spans import Span, char_iou


def token_indices_for_span(text: str, span: Span) -> set[int]:
    indices: set[int] = set()
    for index, token in enumerate(tokenize(text)):
        if max(token.start, span.start) < min(token.end, span.end):
            indices.add(index)
    return indices


def token_iou(text: str, left: Span, right: Span) -> float:
    left_ids = token_indices_for_span(text, left)
    right_ids = token_indices_for_span(text, right)
    union = left_ids | right_ids
    return len(left_ids & right_ids) / len(union) if union else char_iou(left, right)


def candidate_hits(
    text: str,
    candidates: list[Span],
    gold_spans: list[Span],
    threshold: float = 0.5,
) -> bool:
    return any(token_iou(text, candidate, gold) >= threshold for candidate in candidates for gold in gold_spans)
