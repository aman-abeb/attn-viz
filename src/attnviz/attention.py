"""Pure-numpy attention primitives: softmax, causal masking, and multi-head attention."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def softmax(x: Array, axis: int = -1) -> Array:
    """Numerically stable softmax along ``axis``.

    Args:
        x: Input array of any shape.
        axis: Axis along which to normalize.

    Returns:
        Array of the same shape as ``x`` whose slices along ``axis`` sum to 1.
    """
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def apply_causal_mask(seq_len: int) -> Array:
    """Build a boolean causal mask of shape ``(seq_len, seq_len)``.

    Entry ``[i, j]`` is ``True`` when query position ``i`` may attend to key
    position ``j`` (that is, ``j <= i``), so no position can see the future.

    Args:
        seq_len: Number of sequence positions; must be positive.

    Returns:
        Lower-triangular boolean array of shape ``(seq_len, seq_len)``.
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None = None,
) -> tuple[Array, Array]:
    """Compute scaled dot-product attention ``softmax(QK^T / sqrt(d_k)) V``.

    Args:
        query: Queries of shape ``(..., seq_q, d_k)``.
        key: Keys of shape ``(..., seq_kv, d_k)``.
        value: Values of shape ``(..., seq_kv, d_v)``.
        mask: Optional boolean array broadcastable to ``(..., seq_q, seq_kv)``;
            ``True`` keeps a position and ``False`` masks it out with ``-inf``
            before the softmax.

    Returns:
        A tuple ``(weights, output)`` where ``weights`` has shape
        ``(..., seq_q, seq_kv)`` with rows summing to 1, and ``output`` has
        shape ``(..., seq_q, d_v)``.
    """
    d_k = query.shape[-1]
    scores = (query @ np.swapaxes(key, -1, -2)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax(scores, axis=-1)
    return weights, weights @ value


class MultiHeadAttention:
    """Multi-head attention block implemented with batched numpy matmuls.

    Args:
        d_model: Model dimension; must be divisible by ``num_heads``.
        num_heads: Number of attention heads.
        seed: Seed for the RNG that initializes the projection weights, making
            construction and forward passes deterministic.
    """

    def __init__(self, d_model: int, num_heads: int, seed: int | None = None) -> None:
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.w_q = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_k = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_v = rng.normal(0.0, scale, size=(d_model, d_model))
        self.w_o = rng.normal(0.0, scale, size=(d_model, d_model))

    def forward(self, x: Array, mask: Array | None = None) -> tuple[Array, Array]:
        """Run multi-head attention over ``x``.

        Args:
            x: Input of shape ``(batch, seq_len, d_model)``.
            mask: Optional boolean mask broadcastable to ``(seq_len, seq_len)``,
                e.g. from :func:`apply_causal_mask`.

        Returns:
            A tuple ``(output, attn_weights)`` where ``output`` has shape
            ``(batch, seq_len, d_model)`` and ``attn_weights`` has shape
            ``(batch, num_heads, seq_len, seq_len)``.
        """
        batch, seq_len, _ = x.shape

        def split_heads(t: Array) -> Array:
            return t.reshape(batch, seq_len, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

        q = split_heads(x @ self.w_q)
        k = split_heads(x @ self.w_k)
        v = split_heads(x @ self.w_v)
        weights, context = scaled_dot_product_attention(q, k, v, mask=mask)
        merged = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        return merged @ self.w_o, weights

    def __call__(self, x: Array, mask: Array | None = None) -> tuple[Array, Array]:
        """Alias for :meth:`forward`."""
        return self.forward(x, mask=mask)
