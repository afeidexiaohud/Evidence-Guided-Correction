from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .spans import Span, merge_spans


TOKEN_RE = re.compile(r"\w+(?:[-'’]\w+)*|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int

    @property
    def norm(self) -> str:
        return self.text.casefold()


@dataclass(frozen=True)
class DiffResult:
    source_spans: list[Span]
    target_spans: list[Span]
    operations: list[dict[str, object]]
    source_changed_tokens: set[int]
    target_changed_tokens: set[int]


def tokenize(text: str) -> list[Token]:
    return [Token(match.group(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]


def diff_spans(source: str, target: str) -> DiffResult:
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    matcher = SequenceMatcher(
        a=[token.norm for token in source_tokens],
        b=[token.norm for token in target_tokens],
        autojunk=False,
    )
    source_spans: list[Span] = []
    target_spans: list[Span] = []
    operations: list[dict[str, object]] = []
    source_changed: set[int] = set()
    target_changed: set[int] = set()
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        source_changed.update(range(i1, i2))
        target_changed.update(range(j1, j2))
        source_span = None
        target_span = None
        if i1 < i2:
            source_span = Span(
                source_tokens[i1].start,
                source_tokens[i2 - 1].end,
                source[source_tokens[i1].start : source_tokens[i2 - 1].end],
            )
            source_spans.append(source_span)
        if j1 < j2:
            target_span = Span(
                target_tokens[j1].start,
                target_tokens[j2 - 1].end,
                target[target_tokens[j1].start : target_tokens[j2 - 1].end],
            )
            target_spans.append(target_span)
        operations.append(
            {
                "tag": tag,
                "source_token_range": [i1, i2],
                "target_token_range": [j1, j2],
                "source_span": source_span.to_dict() if source_span else None,
                "target_span": target_span.to_dict() if target_span else None,
            }
        )
    return DiffResult(
        source_spans=merge_spans(source_spans, source),
        target_spans=merge_spans(target_spans, target),
        operations=operations,
        source_changed_tokens=source_changed,
        target_changed_tokens=target_changed,
    )


def _alignment_cost(source_token: Token, masked_token: Token) -> float:
    if masked_token.text == "*":
        return 0.05
    if source_token.norm == masked_token.norm:
        return 0.0
    return 1.0


def masked_source_token_indices(source: str, masked_source: str) -> tuple[set[int], float]:
    """Align LIFE's masked source to the original and recover masked token ids.

    LIFE masks word tokens with ``*`` and normally preserves token count. Dynamic
    programming handles punctuation/tokenization disagreements without assuming
    that token positions are identical.
    """

    source_tokens = tokenize(source)
    masked_tokens = tokenize(masked_source)
    n, m = len(source_tokens), len(masked_tokens)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = float(i)
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = float(j)
        back[0][j] = "insert"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = [
                (dp[i - 1][j - 1] + _alignment_cost(source_tokens[i - 1], masked_tokens[j - 1]), "align"),
                (dp[i - 1][j] + 1.0, "delete"),
                (dp[i][j - 1] + 1.0, "insert"),
            ]
            dp[i][j], back[i][j] = min(options, key=lambda item: item[0])

    masked_indices: set[int] = set()
    aligned = 0
    i, j = n, m
    while i or j:
        action = back[i][j]
        if action == "align":
            if masked_tokens[j - 1].text == "*":
                masked_indices.add(i - 1)
            elif source_tokens[i - 1].norm == masked_tokens[j - 1].norm:
                aligned += 1
            i -= 1
            j -= 1
        elif action == "delete":
            i -= 1
        elif action == "insert":
            j -= 1
        else:
            break
    non_masked = sum(token.text != "*" for token in masked_tokens)
    anchor_accuracy = aligned / non_masked if non_masked else 1.0
    return masked_indices, anchor_accuracy


def indices_to_spans(text: str, indices: set[int]) -> list[Span]:
    tokens = tokenize(text)
    spans = [Span(tokens[index].start, tokens[index].end, tokens[index].text) for index in sorted(indices)]
    return merge_spans(spans, text)
