"""Runnable demo that prints a small causal attention matrix."""

from __future__ import annotations

import numpy as np

from attnviz.attention import MultiHeadAttention, apply_causal_mask


def main() -> None:
    """Run a tiny seeded multi-head attention forward pass and print head 0."""
    rng = np.random.default_rng(0)
    batch, seq_len, d_model, num_heads = 1, 6, 16, 2
    x = rng.normal(size=(batch, seq_len, d_model))
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=0)
    mask = apply_causal_mask(seq_len)
    _, attn = mha.forward(x, mask=mask)

    print("AttnViz demo: multi-head attention with causal masking")
    print(f"batch={batch}  seq_len={seq_len}  d_model={d_model}  num_heads={num_heads}  seed=0")
    print()
    print("Causal mask (True = position may attend):")
    print(mask)
    print()
    print("Attention weights, head 0 (rows = query, cols = key; rows sum to 1):")
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    print(attn[0, 0])
    print("Row sums:", attn[0, 0].sum(axis=-1))


if __name__ == "__main__":
    main()
