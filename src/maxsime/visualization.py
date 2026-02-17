"""Visualization functions for MaxSimE explanations.

This module provides matplotlib-based visualizations for MaxSimE explanations,
including similarity heatmaps and token alignment bar charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from .data_structures import MaxSimExplanation

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Use non-interactive backend by default to avoid display issues
matplotlib.use("Agg")


def plot_similarity_heatmap(
    explanation: MaxSimExplanation,
    title: str | None = None,
    highlight_alignment: bool = True,
    figsize: tuple[float, float] = (14, 6),
    cmap: str = "YlOrRd",
    filter_special_tokens: bool = True,
    special_tokens: set[str] | None = None,
    max_doc_tokens: int = 80,
    show_colorbar: bool = True,
    annotate: bool = False,
) -> Figure:
    """Render the (Q x D) similarity matrix as a heatmap.

    Args:
        explanation: MaxSimExplanation to visualize
        title: Plot title (defaults to "MaxSimE: score={total_score:.2f}")
        highlight_alignment: If True, mark MaxSim pairs with a rectangle border
        figsize: Figure size in inches
        cmap: Matplotlib colormap name
        filter_special_tokens: If True, hide special tokens from axes
        special_tokens: Set of special tokens to filter (defaults to common ones)
        max_doc_tokens: Truncate document tokens beyond this count
        show_colorbar: If True, show colorbar on the right
        annotate: If True, show similarity values in cells

    Returns:
        matplotlib Figure object

    Visual spec:
        - Y-axis: query tokens (top to bottom, reading order)
        - X-axis: doc tokens (left to right)
        - Color: cosine similarity (darker = higher)
        - MaxSim pairs: bold border rectangle around the cell
    """
    if special_tokens is None:
        special_tokens = {
            "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[Q]", "[D]",
            "[unused0]", "[unused1]", "<s>", "</s>", "<pad>", "<unk>",
        }

    # Build index mappings for filtered display
    if filter_special_tokens:
        q_indices = [i for i, t in enumerate(explanation.query_tokens)
                     if t not in special_tokens]
        d_indices = [i for i, t in enumerate(explanation.doc_tokens)
                     if t not in special_tokens]
    else:
        q_indices = list(range(len(explanation.query_tokens)))
        d_indices = list(range(len(explanation.doc_tokens)))

    # Truncate document tokens if needed
    truncated = False
    if len(d_indices) > max_doc_tokens:
        d_indices = d_indices[:max_doc_tokens]
        truncated = True

    # Get filtered tokens and matrix slice
    q_tokens = [explanation.query_tokens[i] for i in q_indices]
    d_tokens = [explanation.doc_tokens[i] for i in d_indices]
    if truncated:
        d_tokens.append("...")

    # Slice similarity matrix
    matrix = explanation.similarity_matrix[np.ix_(q_indices, d_indices)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    # Set tick labels
    ax.set_xticks(np.arange(len(d_tokens)))
    ax.set_yticks(np.arange(len(q_tokens)))
    ax.set_xticklabels(d_tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(q_tokens, fontsize=8)

    # Annotate cells with values if requested
    if annotate and matrix.size < 500:  # Limit annotation for large matrices
        for i in range(len(q_tokens)):
            for j in range(len(d_tokens)):
                text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                              ha="center", va="center", fontsize=6,
                              color="white" if matrix[i, j] > 0.5 else "black")

    # Highlight MaxSim alignment pairs
    if highlight_alignment:
        # Map filtered indices back to original indices
        q_idx_to_filtered = {orig: filt for filt, orig in enumerate(q_indices)}
        d_idx_to_filtered = {orig: filt for filt, orig in enumerate(d_indices)}

        for pair in explanation.alignment:
            if pair.query_idx in q_idx_to_filtered and pair.doc_idx in d_idx_to_filtered:
                qi = q_idx_to_filtered[pair.query_idx]
                di = d_idx_to_filtered[pair.doc_idx]
                rect = patches.Rectangle(
                    (di - 0.5, qi - 0.5), 1, 1,
                    linewidth=2, edgecolor="blue", facecolor="none"
                )
                ax.add_patch(rect)

    # Title
    if title is None:
        title = f"MaxSimE: score={explanation.total_score:.2f}"
    ax.set_title(title)

    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Cosine Similarity")

    plt.tight_layout()
    return fig


def plot_token_alignment(
    explanation: MaxSimExplanation,
    top_k: int = 10,
    title: str = "Top Token Alignments",
    figsize: tuple[float, float] = (10, 5),
    color_thresholds: tuple[float, float] = (0.5, 0.8),
) -> Figure:
    """Horizontal bar chart showing top-k token pairs by similarity.

    Args:
        explanation: MaxSimExplanation to visualize
        top_k: Number of top pairs to show
        title: Plot title
        figsize: Figure size in inches
        color_thresholds: (low, high) thresholds for coloring bars
                         - similarity > high: green
                         - low < similarity <= high: orange
                         - similarity <= low: red

    Returns:
        matplotlib Figure object
    """
    pairs = explanation.top_k(top_k)

    if not pairs:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No alignments to display",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return fig

    # Prepare data
    labels = [f"{p.query_token} -> {p.doc_token}" for p in pairs]
    scores = [p.similarity for p in pairs]

    # Color based on score
    low_thresh, high_thresh = color_thresholds
    colors = []
    for s in scores:
        if s > high_thresh:
            colors.append("#2ecc71")  # green
        elif s > low_thresh:
            colors.append("#f39c12")  # orange
        else:
            colors.append("#e74c3c")  # red

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Horizontal bar chart (reversed so highest is on top)
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, scores, color=colors, edgecolor="black", linewidth=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1])  # Reverse for top-to-bottom ordering
    ax.invert_yaxis()  # Highest score on top
    ax.set_xlabel("Cosine Similarity")
    ax.set_xlim(0, 1)
    ax.set_title(title)

    # Add score annotations
    for i, (score, label) in enumerate(zip(scores, labels)):
        ax.text(score + 0.02, i, f"{score:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    return fig


def save_explanation_figure(
    explanation: MaxSimExplanation,
    path: str,
    plot_type: str = "heatmap",
    dpi: int = 150,
    **kwargs,
) -> None:
    """Convenience function to render and save an explanation figure.

    Args:
        explanation: MaxSimExplanation to visualize
        path: Output file path (e.g., "explanation.png")
        plot_type: "heatmap" or "alignment"
        dpi: Resolution in dots per inch
        **kwargs: Additional arguments passed to the plot function
    """
    if plot_type == "heatmap":
        fig = plot_similarity_heatmap(explanation, **kwargs)
    elif plot_type == "alignment":
        fig = plot_token_alignment(explanation, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'heatmap' or 'alignment'")

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
