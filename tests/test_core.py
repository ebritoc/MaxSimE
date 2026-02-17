"""Unit tests for MaxSimExplainer."""

import numpy as np
import pytest

from maxsime import MaxSimExplainer, MaxSimExplanation, TokenPair


class TestMaxSimExplainer:
    """Tests for the MaxSimExplainer class."""

    def test_explain_returns_explanation(self, toy_embeddings):
        """explain() returns a MaxSimExplanation object."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        assert isinstance(result, MaxSimExplanation)
        assert len(result.alignment) == len(q_tok)  # One match per query token
        assert result.similarity_matrix.shape == (3, 5)
        assert result.total_score > 0

    def test_alignment_pairs_are_valid(self, toy_embeddings):
        """Alignment pairs have valid indices and scores."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        for pair in result.alignment:
            assert isinstance(pair, TokenPair)
            assert 0 <= pair.query_idx < len(q_tok)
            assert 0 <= pair.doc_idx < len(d_tok)
            assert -1.0 <= pair.similarity <= 1.0
            assert pair.query_token in q_tok
            assert pair.doc_token in d_tok

    def test_special_token_filtering(self, embeddings_with_special_tokens):
        """Special tokens are filtered from alignment when requested."""
        q_emb, d_emb, q_tok, d_tok = embeddings_with_special_tokens
        explainer = MaxSimExplainer()

        result = explainer.explain(
            q_emb, d_emb, q_tok, d_tok, filter_special_tokens=True
        )

        # Only "capital" should survive filtering
        assert len(result.alignment) == 1
        assert result.alignment[0].query_token == "capital"

    def test_no_filtering_when_disabled(self, embeddings_with_special_tokens):
        """All tokens included when filtering is disabled."""
        q_emb, d_emb, q_tok, d_tok = embeddings_with_special_tokens
        explainer = MaxSimExplainer()

        result = explainer.explain(
            q_emb, d_emb, q_tok, d_tok, filter_special_tokens=False
        )

        # All 3 tokens should be present
        assert len(result.alignment) == 3

    def test_custom_special_tokens(self, toy_embeddings):
        """Custom special tokens can be specified."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer(special_tokens={"what", "is"})

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        # Only "CET1" should survive custom filtering
        assert len(result.alignment) == 1
        assert result.alignment[0].query_token == "CET1"

    def test_top_k(self, toy_embeddings):
        """top_k returns correct number of pairs sorted by similarity."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)
        top2 = result.top_k(2)

        assert len(top2) == 2
        assert top2[0].similarity >= top2[1].similarity

    def test_top_pairs_sorted_descending(self, toy_embeddings):
        """top_pairs returns all pairs sorted by similarity descending."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)
        top_pairs = result.top_pairs

        for i in range(len(top_pairs) - 1):
            assert top_pairs[i].similarity >= top_pairs[i + 1].similarity

    def test_dimension_mismatch_raises(self):
        """AssertionError raised when embedding dimensions don't match."""
        explainer = MaxSimExplainer()

        with pytest.raises(AssertionError, match="Dimension mismatch"):
            explainer.explain(
                np.random.randn(3, 4),
                np.random.randn(5, 8),  # Different dimension
                ["a", "b", "c"],
                ["d", "e", "f", "g", "h"],
            )

    def test_token_count_mismatch_raises(self):
        """AssertionError raised when token count doesn't match embeddings."""
        explainer = MaxSimExplainer()

        with pytest.raises(AssertionError, match="Query embeddings"):
            explainer.explain(
                np.random.randn(3, 4),
                np.random.randn(5, 4),
                ["a", "b"],  # Only 2 tokens for 3 embeddings
                ["d", "e", "f", "g", "h"],
            )

    def test_identical_embeddings_high_similarity(self, identical_embeddings):
        """Identical embeddings produce similarity of 1.0."""
        q_emb, d_emb, q_tok, d_tok = identical_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        # Diagonal should be 1.0 (self-similarity)
        for i in range(len(q_tok)):
            assert np.isclose(result.similarity_matrix[i, i], 1.0, atol=1e-5)

    def test_similarity_matrix_shape(self, large_embeddings):
        """Similarity matrix has correct shape (Q x D)."""
        q_emb, d_emb, q_tok, d_tok = large_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        assert result.similarity_matrix.shape == (20, 100)

    def test_total_score_is_sum_of_maxsim(self, toy_embeddings):
        """Total score equals sum of all MaxSim values."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        # Compute expected total from matrix
        expected = np.max(result.similarity_matrix, axis=1).sum()
        assert np.isclose(result.total_score, expected, atol=1e-5)

    def test_matched_doc_indices(self, toy_embeddings):
        """matched_doc_indices returns correct set of indices."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        result = explainer.explain(q_emb, d_emb, q_tok, d_tok)
        indices = result.matched_doc_indices

        assert isinstance(indices, set)
        assert all(0 <= idx < len(d_tok) for idx in indices)
        assert len(indices) <= len(q_tok)  # At most one match per query token


class TestExplainBatch:
    """Tests for the explain_batch method."""

    def test_explain_batch_returns_list(self, toy_embeddings):
        """explain_batch returns list of explanations."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        # Create 3 documents
        doc_embs = [d_emb, d_emb * 0.5, d_emb * 2]
        doc_toks = [d_tok, d_tok, d_tok]

        results = explainer.explain_batch(q_emb, doc_embs, q_tok, doc_toks)

        assert len(results) == 3
        assert all(isinstance(r, MaxSimExplanation) for r in results)

    def test_explain_batch_scores_vary(self, toy_embeddings):
        """Different documents produce different scores."""
        q_emb, d_emb, q_tok, d_tok = toy_embeddings
        explainer = MaxSimExplainer()

        np.random.seed(111)
        doc_embs = [np.random.randn(5, 4).astype(np.float32) for _ in range(3)]
        doc_toks = [d_tok for _ in range(3)]

        results = explainer.explain_batch(q_emb, doc_embs, q_tok, doc_toks)
        scores = [r.total_score for r in results]

        # Scores should be different (very unlikely to be identical)
        assert len(set(scores)) == 3
