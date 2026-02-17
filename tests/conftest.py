"""Shared test fixtures for MaxSimE tests."""

import numpy as np
import pytest


@pytest.fixture
def toy_embeddings():
    """Small deterministic embeddings for testing.

    Returns:
        Tuple of (query_emb, doc_emb, query_tokens, doc_tokens)
        - 3 query tokens, 5 doc tokens, dim=4
    """
    np.random.seed(42)
    query_emb = np.random.randn(3, 4).astype(np.float32)
    doc_emb = np.random.randn(5, 4).astype(np.float32)
    query_tokens = ["what", "is", "CET1"]
    doc_tokens = ["the", "CET1", "ratio", "is", "13.9%"]
    return query_emb, doc_emb, query_tokens, doc_tokens


@pytest.fixture
def embeddings_with_special_tokens():
    """Embeddings including special tokens for filtering tests.

    Returns:
        Tuple of (query_emb, doc_emb, query_tokens, doc_tokens)
        - Query has [CLS], capital, [SEP]
        - Doc has [CLS], equity, [SEP]
    """
    np.random.seed(123)
    query_emb = np.random.randn(3, 4).astype(np.float32)
    doc_emb = np.random.randn(3, 4).astype(np.float32)
    query_tokens = ["[CLS]", "capital", "[SEP]"]
    doc_tokens = ["[CLS]", "equity", "[SEP]"]
    return query_emb, doc_emb, query_tokens, doc_tokens


@pytest.fixture
def large_embeddings():
    """Larger embeddings for stress testing.

    Returns:
        Tuple of (query_emb, doc_emb, query_tokens, doc_tokens)
        - 20 query tokens, 100 doc tokens, dim=128
    """
    np.random.seed(456)
    query_emb = np.random.randn(20, 128).astype(np.float32)
    doc_emb = np.random.randn(100, 128).astype(np.float32)
    query_tokens = [f"q{i}" for i in range(20)]
    doc_tokens = [f"d{i}" for i in range(100)]
    return query_emb, doc_emb, query_tokens, doc_tokens


@pytest.fixture
def identical_embeddings():
    """Identical embeddings to test perfect similarity.

    Returns:
        Tuple of (query_emb, doc_emb, query_tokens, doc_tokens)
        - Query and doc have identical embeddings
    """
    np.random.seed(789)
    emb = np.random.randn(3, 4).astype(np.float32)
    query_tokens = ["a", "b", "c"]
    doc_tokens = ["a", "b", "c"]
    return emb.copy(), emb.copy(), query_tokens, doc_tokens
