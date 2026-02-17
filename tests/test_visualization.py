"""Tests for MaxSimE visualization functions."""

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from maxsime import (
    MaxSimExplainer,
    plot_similarity_heatmap,
    plot_token_alignment,
    save_explanation_figure,
)


@pytest.fixture
def sample_explanation(toy_embeddings):
    """Create a sample explanation for testing."""
    q_emb, d_emb, q_tok, d_tok = toy_embeddings
    explainer = MaxSimExplainer()
    return explainer.explain(q_emb, d_emb, q_tok, d_tok)


class TestPlotSimilarityHeatmap:
    """Tests for plot_similarity_heatmap function."""

    def test_returns_figure(self, sample_explanation):
        """Function returns a matplotlib Figure."""
        fig = plot_similarity_heatmap(sample_explanation)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self, sample_explanation):
        """Custom title is applied."""
        fig = plot_similarity_heatmap(sample_explanation, title="Test Title")

        ax = fig.get_axes()[0]
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_default_title_includes_score(self, sample_explanation):
        """Default title includes the total score."""
        fig = plot_similarity_heatmap(sample_explanation)

        ax = fig.get_axes()[0]
        title = ax.get_title()
        assert "MaxSimE" in title
        assert "score=" in title
        plt.close(fig)

    def test_custom_figsize(self, sample_explanation):
        """Custom figure size is applied."""
        fig = plot_similarity_heatmap(sample_explanation, figsize=(8, 4))

        width, height = fig.get_size_inches()
        assert np.isclose(width, 8, atol=0.5)
        assert np.isclose(height, 4, atol=0.5)
        plt.close(fig)

    def test_colorbar_shown_by_default(self, sample_explanation):
        """Colorbar is shown by default."""
        fig = plot_similarity_heatmap(sample_explanation)

        # Figure should have 2 axes (main plot + colorbar)
        assert len(fig.get_axes()) >= 2
        plt.close(fig)

    def test_colorbar_can_be_hidden(self, sample_explanation):
        """Colorbar can be hidden."""
        fig = plot_similarity_heatmap(sample_explanation, show_colorbar=False)

        # Figure should have only 1 axis (main plot)
        assert len(fig.get_axes()) == 1
        plt.close(fig)

    def test_renders_without_error_large(self, large_embeddings):
        """Renders without error for large matrices."""
        q_emb, d_emb, q_tok, d_tok = large_embeddings
        explainer = MaxSimExplainer()
        explanation = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        fig = plot_similarity_heatmap(explanation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTokenAlignment:
    """Tests for plot_token_alignment function."""

    def test_returns_figure(self, sample_explanation):
        """Function returns a matplotlib Figure."""
        fig = plot_token_alignment(sample_explanation)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_top_k_limits_bars(self, sample_explanation):
        """top_k parameter limits number of bars."""
        fig = plot_token_alignment(sample_explanation, top_k=2)

        ax = fig.get_axes()[0]
        # Number of patches (bars) should be limited
        bars = [p for p in ax.patches if hasattr(p, "get_width")]
        assert len(bars) <= 2
        plt.close(fig)

    def test_custom_title(self, sample_explanation):
        """Custom title is applied."""
        fig = plot_token_alignment(sample_explanation, title="My Alignment")

        ax = fig.get_axes()[0]
        assert ax.get_title() == "My Alignment"
        plt.close(fig)

    def test_handles_empty_alignment(self, embeddings_with_special_tokens):
        """Handles explanation with no alignments gracefully."""
        q_emb, d_emb, q_tok, d_tok = embeddings_with_special_tokens
        # Use custom special tokens that filter everything
        explainer = MaxSimExplainer(special_tokens={"[CLS]", "[SEP]", "capital"})
        explanation = explainer.explain(q_emb, d_emb, q_tok, d_tok)

        fig = plot_token_alignment(explanation)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestSaveExplanationFigure:
    """Tests for save_explanation_figure function."""

    def test_saves_heatmap(self, sample_explanation, tmp_path):
        """Saves heatmap to file."""
        path = tmp_path / "heatmap.png"

        save_explanation_figure(sample_explanation, str(path), plot_type="heatmap")

        assert path.exists()
        assert path.stat().st_size > 0

    def test_saves_alignment(self, sample_explanation, tmp_path):
        """Saves alignment chart to file."""
        path = tmp_path / "alignment.png"

        save_explanation_figure(sample_explanation, str(path), plot_type="alignment")

        assert path.exists()
        assert path.stat().st_size > 0

    def test_invalid_plot_type_raises(self, sample_explanation, tmp_path):
        """Invalid plot_type raises ValueError."""
        path = tmp_path / "output.png"

        with pytest.raises(ValueError, match="Unknown plot_type"):
            save_explanation_figure(sample_explanation, str(path), plot_type="invalid")

    def test_custom_dpi(self, sample_explanation, tmp_path):
        """Custom DPI affects file size."""
        path_low = tmp_path / "low_dpi.png"
        path_high = tmp_path / "high_dpi.png"

        save_explanation_figure(sample_explanation, str(path_low), dpi=50)
        save_explanation_figure(sample_explanation, str(path_high), dpi=300)

        # Higher DPI should produce larger file
        assert path_high.stat().st_size > path_low.stat().st_size
