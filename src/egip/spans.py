from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid span: {self.start}:{self.end}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def merge_spans(spans: list[Span], text: str, max_gap: int = 1) -> list[Span]:
    if not spans:
        return []
    ordered = sorted(spans)
    merged: list[Span] = []
    start, end = ordered[0].start, ordered[0].end
    for span in ordered[1:]:
        if span.start <= end + max_gap:
            end = max(end, span.end)
        else:
            merged.append(Span(start, end, text[start:end]))
            start, end = span.start, span.end
    merged.append(Span(start, end, text[start:end]))
    return merged


def char_iou(left: Span, right: Span) -> float:
    intersection = max(0, min(left.end, right.end) - max(left.start, right.start))
    union = max(left.end, right.end) - min(left.start, right.start)
    return intersection / union if union else 0.0
