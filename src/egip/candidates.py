from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .alignment import tokenize
from .spans import Span


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    text: str
    phrase: str

    @property
    def span(self) -> Span:
        return Span(self.start, self.end, self.text)

    @property
    def key(self) -> tuple[int, int, str]:
        return self.start, self.end, self.phrase.casefold()

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "phrase": self.phrase,
        }


def _lexical_tokens(text: str) -> list[tuple[int, str, int, int]]:
    return [
        (index, token.norm, token.start, token.end)
        for index, token in enumerate(tokenize(text))
        if any(char.isalnum() or char == "_" for char in token.text)
    ]


def phrase_occurrences(text: str, phrase: str) -> list[Candidate]:
    claim_tokens = _lexical_tokens(text)
    phrase_tokens = [token.norm for token in tokenize(phrase) if any(char.isalnum() or char == "_" for char in token.text)]
    if not phrase_tokens:
        return []
    occurrences: list[Candidate] = []
    width = len(phrase_tokens)
    norms = [token[1] for token in claim_tokens]
    for start_index in range(0, len(claim_tokens) - width + 1):
        if norms[start_index : start_index + width] != phrase_tokens:
            continue
        char_start = claim_tokens[start_index][2]
        char_end = claim_tokens[start_index + width - 1][3]
        occurrences.append(Candidate(char_start, char_end, text[char_start:char_end], phrase))
    return occurrences


def m2c_phrase_candidate(text: str, phrase: str) -> Candidate | None:
    """Reproduce M2C's first/last whitespace-token phrase mapping."""

    sentence_tokens = list(re.finditer(r"\S+", text))
    normalized = [match.group().lower().strip(",.") for match in sentence_tokens]
    phrase_tokens = phrase.lower().split()
    if not phrase_tokens:
        return None
    for start, token in enumerate(normalized):
        if token != phrase_tokens[0]:
            continue
        for end in range(start, len(sentence_tokens)):
            if normalized[end] == phrase_tokens[-1]:
                char_start = sentence_tokens[start].start()
                char_end = sentence_tokens[end].end()
                return Candidate(char_start, char_end, text[char_start:char_end], phrase)
    return None


def keybert_candidate_phrases(claim: str, max_ngram_ratio: float = 0.5) -> list[str]:
    whitespace_tokens = [token for token in claim.split(" ") if token]
    max_ngram = max(1, int(max_ngram_ratio * len(whitespace_tokens)))
    try:
        vectorizer = CountVectorizer(
            ngram_range=(1, max_ngram),
            stop_words=None,
            min_df=1,
        ).fit([claim])
    except ValueError:
        return []
    return list(vectorizer.get_feature_names_out())


def build_candidate_pool(
    claim: str,
    max_ngram_ratio: float = 0.5,
    *,
    all_occurrences: bool = False,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[int, int, str]] = set()
    for phrase in keybert_candidate_phrases(claim, max_ngram_ratio=max_ngram_ratio):
        if all_occurrences:
            occurrences = phrase_occurrences(claim, phrase)
        else:
            mapped = m2c_phrase_candidate(claim, phrase)
            occurrences = [mapped] if mapped is not None else []
        for candidate in occurrences:
            if candidate.key not in seen:
                candidates.append(candidate)
                seen.add(candidate.key)
    return candidates


def mask_candidate(claim: str, candidate: Candidate, mask_token: str = "[MASK]") -> str:
    return claim[: candidate.start] + mask_token + claim[candidate.end :]


def m2c_mask_candidate(claim: str, candidate: Candidate, mask_token: str = "[MASK]") -> str:
    return claim.replace(candidate.text, mask_token)


def apply_candidate_mask(
    claim: str,
    candidate: Candidate,
    *,
    mode: str = "legacy_global",
    mask_token: str = "[MASK]",
) -> str:
    """Mask a candidate with an explicit legacy or span-specific contract."""

    if mode == "legacy_global":
        return m2c_mask_candidate(claim, candidate, mask_token)
    if mode == "span":
        return mask_candidate(claim, candidate, mask_token)
    raise ValueError(f"Unknown mask mode: {mode}")


def mmr_rank(
    candidates: list[Candidate],
    doc_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    *,
    top_k: int = 10,
    diversity: float = 0.3,
    error_probabilities: np.ndarray | None = None,
    error_weight: float = 0.0,
) -> tuple[list[int], np.ndarray]:
    if len(candidates) != len(candidate_embeddings):
        raise ValueError("Candidate and embedding counts differ")
    if not candidates:
        return [], np.empty(0)
    relevance = cosine_similarity(candidate_embeddings, doc_embedding.reshape(1, -1)).reshape(-1)
    similarities = cosine_similarity(candidate_embeddings)
    error_probabilities = (
        np.zeros(len(candidates), dtype=float)
        if error_probabilities is None
        else np.asarray(error_probabilities, dtype=float)
    )
    selected = [int(np.argmax((1 - error_weight) * relevance + error_weight * error_probabilities))]
    remaining = [index for index in range(len(candidates)) if index != selected[0]]
    while remaining and len(selected) < top_k:
        redundancy = np.max(similarities[remaining][:, selected], axis=1)
        score = (
            error_weight * error_probabilities[remaining]
            + (1 - error_weight) * ((1 - diversity) * relevance[remaining] - diversity * redundancy)
        )
        chosen = remaining[int(np.argmax(score))]
        selected.append(chosen)
        remaining.remove(chosen)
    return selected, relevance
