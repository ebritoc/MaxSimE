"""Utility functions for MaxSimE.

Helper functions for token filtering, normalization, and other common operations.
"""

from __future__ import annotations

DEFAULT_SPECIAL_TOKENS = frozenset({
    "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[Q]", "[D]",
    "[unused0]", "[unused1]", "<s>", "</s>", "<pad>", "<unk>",
})


def filter_token_indices(
    tokens: list[str],
    special_tokens: set[str] | None = None,
) -> list[int]:
    """Return indices of tokens that are NOT special tokens.

    Args:
        tokens: List of token strings
        special_tokens: Set of tokens to exclude (defaults to common BERT tokens)

    Returns:
        List of indices for non-special tokens
    """
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS
    return [i for i, t in enumerate(tokens) if t not in special_tokens]


def filter_tokens(
    tokens: list[str],
    special_tokens: set[str] | None = None,
) -> list[str]:
    """Return tokens that are NOT special tokens.

    Args:
        tokens: List of token strings
        special_tokens: Set of tokens to exclude (defaults to common BERT tokens)

    Returns:
        List of non-special tokens
    """
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS
    return [t for t in tokens if t not in special_tokens]


def truncate_tokens(
    tokens: list[str],
    max_length: int = 80,
) -> tuple[list[str], bool]:
    """Truncate token list to maximum length.

    Args:
        tokens: List of token strings
        max_length: Maximum number of tokens to keep

    Returns:
        Tuple of (truncated_list, was_truncated)
    """
    if len(tokens) <= max_length:
        return tokens, False
    return tokens[:max_length], True


def clean_token(token: str) -> str:
    """Clean a token string for display.

    Removes common subword prefixes like ## (BERT), Ġ (GPT-2/RoBERTa),
    and _ (SentencePiece).

    Args:
        token: Raw token string

    Returns:
        Cleaned token string for display
    """
    # BERT WordPiece continuation marker
    if token.startswith("##"):
        return token[2:]
    # GPT-2 / RoBERTa space marker
    if token.startswith("Ġ"):
        return token[1:]
    # SentencePiece space marker
    if token.startswith("▁"):
        return token[1:]
    return token
