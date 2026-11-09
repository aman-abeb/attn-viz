"""Attention-only sequence copier with a pure-numpy forward pass and npz I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from attnviz.attention import MultiHeadAttention, apply_causal_mask

Array = np.ndarray


class AttentionCopier:
    """A small attention-only model that copies its input sequence.

    Architecture: learned token and position embeddings, ``num_layers``
    residual multi-head attention blocks with causal masking, and a linear
    head projecting each position to vocabulary logits.

    Args:
        vocab_size: Number of distinct tokens, including BOS and EOS.
        d_model: Model dimension; must be divisible by ``num_heads``.
        num_heads: Number of attention heads per block.
        num_layers: Number of residual attention blocks.
        max_seq_len: Capacity of the position embedding table; sequences longer
            than this are rejected by :meth:`forward`.
        seed: Seed for the RNG that initializes all weights, making construction
            and forward passes deterministic.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
        seed: int = 0,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.seed = seed

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.tok_emb = rng.normal(0.0, scale, size=(vocab_size, d_model))
        self.pos_emb = rng.normal(0.0, scale, size=(max_seq_len, d_model))
        self.blocks = [
            MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            for _ in range(num_layers)
        ]
        self.w_head = rng.normal(0.0, scale, size=(d_model, vocab_size))
        self.b_head = np.zeros(vocab_size, dtype=np.float64)

    def forward(self, x: Array) -> tuple[Array, Array]:
        """Run a causal forward pass over a batch of token ids.

        Args:
            x: Integer token ids of shape ``(batch, seq_len)`` with
                ``seq_len <= max_seq_len``.

        Returns:
            A tuple ``(logits, attn)`` where ``logits`` has shape
            ``(batch, seq_len, vocab_size)`` and ``attn`` stacks the per-layer
            per-head attention weights into shape
            ``(num_layers, batch, num_heads, seq_len, seq_len)``. Attention
            rows are causal (upper triangle exactly zero) and sum to 1.
        """
        if x.ndim != 2:
            raise ValueError("x must have shape (batch, seq_len)")
        _, seq_len = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        mask = apply_causal_mask(seq_len)
        h = self.tok_emb[x] + self.pos_emb[:seq_len][np.newaxis, :, :]
        attn_layers: list[Array] = []
        for block in self.blocks:
            attn_out, weights = block.forward(h, mask=mask)
            h = h + attn_out
            attn_layers.append(weights)
        logits = h @ self.w_head + self.b_head
        return logits, np.stack(attn_layers, axis=0)

    def __call__(self, x: Array) -> tuple[Array, Array]:
        """Alias for :meth:`forward`."""
        return self.forward(x)

    def save(self, path: str | Path) -> None:
        """Save the config and all weights to a portable ``.npz`` file.

        Args:
            path: Destination file path; written exactly as given (no suffix is
                appended because the file is opened as a binary stream).
        """
        params: dict[str, Array] = {
            "vocab_size": np.array(self.vocab_size),
            "d_model": np.array(self.d_model),
            "num_heads": np.array(self.num_heads),
            "num_layers": np.array(self.num_layers),
            "max_seq_len": np.array(self.max_seq_len),
            "tok_emb": self.tok_emb,
            "pos_emb": self.pos_emb,
            "w_head": self.w_head,
            "b_head": self.b_head,
        }
        for i, block in enumerate(self.blocks):
            params[f"block{i}_w_q"] = block.w_q
            params[f"block{i}_w_k"] = block.w_k
            params[f"block{i}_w_v"] = block.w_v
            params[f"block{i}_w_o"] = block.w_o
        with open(path, "wb") as fh:
            np.savez(fh, **params)

    @classmethod
    def load(cls, path: str | Path) -> AttentionCopier:
        """Load a model written by :meth:`save`, restoring weights exactly.

        Args:
            path: Path to an ``.npz`` file produced by :meth:`save`.

        Returns:
            A new :class:`AttentionCopier` whose forward pass reproduces the
            saved model bit-for-bit.
        """
        with np.load(path) as data:
            model = cls(
                vocab_size=int(data["vocab_size"]),
                d_model=int(data["d_model"]),
                num_heads=int(data["num_heads"]),
                num_layers=int(data["num_layers"]),
                max_seq_len=int(data["max_seq_len"]),
            )
            model.tok_emb = np.array(data["tok_emb"], dtype=np.float64)
            model.pos_emb = np.array(data["pos_emb"], dtype=np.float64)
            model.w_head = np.array(data["w_head"], dtype=np.float64)
            model.b_head = np.array(data["b_head"], dtype=np.float64)
            for i, block in enumerate(model.blocks):
                block.w_q = np.array(data[f"block{i}_w_q"], dtype=np.float64)
                block.w_k = np.array(data[f"block{i}_w_k"], dtype=np.float64)
                block.w_v = np.array(data[f"block{i}_w_v"], dtype=np.float64)
                block.w_o = np.array(data[f"block{i}_w_o"], dtype=np.float64)
        return model
