"""Core MaxSimE computation - model-agnostic implementation.

This module provides the MaxSimExplainer class which computes MaxSimE explanations
from pre-computed token embeddings. It is intentionally model-agnostic: the caller
is responsible for encoding queries and documents using their preferred model
(ColBERT, SentenceTransformers, PyLate, etc.).
"""

from __future__ import annotations

import numpy as np

from .data_structures import MaxSimExplanation, TokenPair


class MaxSimExplainer:
    """Compute MaxSimE explanations from pre-computed token embeddings.

    This class implements the core MaxSimE algorithm from:
    "MaxSimE: Explaining Transformer-based Semantic Similarity via
     Contextualized Best Matching Token Pairs" (SIGIR 2023)

    The algorithm:
    1. Normalize embeddings to unit vectors
    2. Compute full cosine similarity matrix (Q x D)
    3. For each query token, find the max-similarity doc token (MaxSim)
    4. Return the alignment with scores and full matrix

    Example:
        >>> explainer = MaxSimExplainer()
        >>> # Get embeddings from your model (e.g., SentenceTransformer)
        >>> q_emb = model.encode("What is CET1?", output_value="token_embeddings")
        >>> d_emb = model.encode("The CET1 ratio is 13.9%", output_value="token_embeddings")
        >>> explanation = explainer.explain(q_emb, d_emb, q_tokens, d_tokens)
        >>> for pair in explanation.top_k(5):
        ...     print(pair)
    """

    # Tokens to exclude from explanation display (not from computation)
    DEFAULT_SPECIAL_TOKENS = frozenset({
        "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[Q]", "[D]",
        "[unused0]", "[unused1]", "<s>", "</s>", "<pad>", "<unk>",
    })

    def __init__(self, special_tokens: set[str] | None = None):
        """Initialize the explainer.

        Args:
            special_tokens: Set of token strings to filter out of alignments.
                           Defaults to common BERT/ColBERT special tokens.
        """
        self.special_tokens = (
            special_tokens if special_tokens is not None
            else self.DEFAULT_SPECIAL_TOKENS
        )

    def explain(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray,
        query_tokens: list[str],
        doc_tokens: list[str],
        filter_special_tokens: bool = True,
    ) -> MaxSimExplanation:
        """Compute the MaxSimE explanation for a query-document pair.

        Args:
            query_embeddings: Token embeddings for the query, shape (Q, dim)
            doc_embeddings: Token embeddings for the document, shape (D, dim)
            query_tokens: Decoded token strings for the query
            doc_tokens: Decoded token strings for the document
            filter_special_tokens: If True, exclude special tokens from the
                alignment list (they are still included in the similarity matrix
                and total score computation)

        Returns:
            MaxSimExplanation with full similarity matrix and alignment

        Raises:
            AssertionError: If embedding dimensions don't match or token counts
                           don't match embedding counts
        """
        # Validate inputs
        assert query_embeddings.shape[0] == len(query_tokens), (
            f"Query embeddings ({query_embeddings.shape[0]}) != "
            f"tokens ({len(query_tokens)})"
        )
        assert doc_embeddings.shape[0] == len(doc_tokens), (
            f"Doc embeddings ({doc_embeddings.shape[0]}) != "
            f"tokens ({len(doc_tokens)})"
        )
        assert query_embeddings.shape[1] == doc_embeddings.shape[1], (
            f"Dimension mismatch: query {query_embeddings.shape[1]} != "
            f"doc {doc_embeddings.shape[1]}"
        )

        # Normalize to unit vectors (L2 normalization)
        q_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        d_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)

        # Avoid division by zero for zero vectors
        q_norm = query_embeddings / np.maximum(q_norms, 1e-10)
        d_norm = doc_embeddings / np.maximum(d_norms, 1e-10)

        # Compute full similarity matrix: (Q, dim) @ (dim, D) -> (Q, D)
        sim_matrix = q_norm @ d_norm.T

        # MaxSim alignment: for each query token, find best matching doc token
        max_indices = np.argmax(sim_matrix, axis=1)
        max_scores = np.max(sim_matrix, axis=1)

        # Build alignment pairs
        alignment = []
        for qi in range(len(query_tokens)):
            di = int(max_indices[qi])
            pair = TokenPair(
                query_idx=qi,
                doc_idx=di,
                query_token=query_tokens[qi],
                doc_token=doc_tokens[di],
                similarity=float(max_scores[qi]),
            )
            if filter_special_tokens and pair.query_token in self.special_tokens:
                continue
            alignment.append(pair)

        # Total score is sum of MaxSim for ALL query tokens (including special)
        # This matches ColBERT's late interaction scoring
        total_score = float(np.sum(max_scores))

        return MaxSimExplanation(
            query_tokens=query_tokens,
            doc_tokens=doc_tokens,
            alignment=alignment,
            total_score=total_score,
            similarity_matrix=sim_matrix,
        )

    def explain_batch(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings_list: list[np.ndarray],
        query_tokens: list[str],
        doc_tokens_list: list[list[str]],
        filter_special_tokens: bool = True,
    ) -> list[MaxSimExplanation]:
        """Explain a single query against multiple documents.

        This is a convenience method for computing explanations for
        multiple documents in a retrieval result set.

        Args:
            query_embeddings: Token embeddings for the query, shape (Q, dim)
            doc_embeddings_list: List of document token embeddings
            query_tokens: Decoded token strings for the query
            doc_tokens_list: List of decoded token strings for each document
            filter_special_tokens: If True, exclude special tokens from alignments

        Returns:
            List of MaxSimExplanation, one per document
        """
        return [
            self.explain(
                query_embeddings, doc_emb,
                query_tokens, doc_tok,
                filter_special_tokens
            )
            for doc_emb, doc_tok in zip(doc_embeddings_list, doc_tokens_list)
        ]
