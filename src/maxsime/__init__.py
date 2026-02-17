"""MaxSimE: Explaining Transformer-based Semantic Similarity.

This package provides model-agnostic tools for explaining transformer-based
semantic similarity via contextualized best matching token pairs.

Based on the paper:
    "MaxSimE: Explaining Transformer-based Semantic Similarity via
     Contextualized Best Matching Token Pairs"
    Brito & Iser, SIGIR 2023

Example:
    >>> from maxsime import MaxSimExplainer, plot_similarity_heatmap
    >>> import numpy as np
    >>>
    >>> explainer = MaxSimExplainer()
    >>>
    >>> # Get token embeddings from your model (ColBERT, SentenceTransformer, etc.)
    >>> query_emb = ...  # shape (Q, dim)
    >>> doc_emb = ...    # shape (D, dim)
    >>>
    >>> explanation = explainer.explain(
    ...     query_emb, doc_emb,
    ...     query_tokens=["what", "is", "CET1"],
    ...     doc_tokens=["the", "CET1", "ratio", "is", "13.9%"],
    ... )
    >>>
    >>> # View top matches
    >>> for pair in explanation.top_k(5):
    ...     print(pair)
    >>>
    >>> # Visualize
    >>> fig = plot_similarity_heatmap(explanation)
    >>> fig.savefig("explanation.png")
"""

from .core import MaxSimExplainer
from .data_structures import MaxSimExplanation, TokenPair
from .visualization import (
    plot_similarity_heatmap,
    plot_token_alignment,
    save_explanation_figure,
)
from .utils import (
    filter_token_indices,
    filter_tokens,
    truncate_tokens,
    clean_token,
    DEFAULT_SPECIAL_TOKENS,
)

__version__ = "2.0.0"
__all__ = [
    # Core
    "MaxSimExplainer",
    # Data structures
    "MaxSimExplanation",
    "TokenPair",
    # Visualization
    "plot_similarity_heatmap",
    "plot_token_alignment",
    "save_explanation_figure",
    # Utils
    "filter_token_indices",
    "filter_tokens",
    "truncate_tokens",
    "clean_token",
    "DEFAULT_SPECIAL_TOKENS",
]
