"""Data structures for MaxSimE explanations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TokenPair:
    """A single MaxSim token match: one query token -> one doc token."""

    query_idx: int
    doc_idx: int
    query_token: str
    doc_token: str
    similarity: float

    def __repr__(self) -> str:
        return f"{self.query_token!r} -> {self.doc_token!r} ({self.similarity:.3f})"


@dataclass
class MaxSimExplanation:
    """Full MaxSimE explanation for a query-document pair.

    Attributes:
        query_tokens: List of decoded query token strings
        doc_tokens: List of decoded document token strings
        alignment: MaxSim pairs (filtered, excluding special tokens if requested)
        total_score: Sum of all MaxSim scores (ColBERT-style late interaction score)
        similarity_matrix: Full (Q x D) cosine similarity matrix
    """

    query_tokens: list[str]
    doc_tokens: list[str]
    alignment: list[TokenPair]
    total_score: float
    similarity_matrix: "np.ndarray"

    @property
    def top_pairs(self) -> list[TokenPair]:
        """Alignment sorted by similarity descending."""
        return sorted(self.alignment, key=lambda p: p.similarity, reverse=True)

    def top_k(self, k: int = 10) -> list[TokenPair]:
        """Return top-k highest similarity pairs."""
        return self.top_pairs[:k]

    @property
    def matched_doc_indices(self) -> set[int]:
        """Set of doc token indices that were matched by any query token."""
        return {pair.doc_idx for pair in self.alignment}

    def __repr__(self) -> str:
        return (
            f"MaxSimExplanation(query_len={len(self.query_tokens)}, "
            f"doc_len={len(self.doc_tokens)}, "
            f"alignments={len(self.alignment)}, "
            f"total_score={self.total_score:.3f})"
        )
