"""Seeded synthetic sequence-copying dataset generator.

The copying task: given the input ``[BOS, t1, ..., tn]`` the model must emit
``[t1, ..., tn, EOS]``. Special tokens occupy ids 0 (BOS) and 1 (EOS); content
tokens are drawn from ``[2, vocab_size)`` so special tokens only ever appear at
the sequence boundaries.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray

BOS_TOKEN = 0
EOS_TOKEN = 1
FIRST_CONTENT_TOKEN = 2


def make_copy_dataset(
    n_samples: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> tuple[Array, Array]:
    """Generate a deterministic sequence-copying dataset.

    Args:
        n_samples: Number of sequences; must be positive.
        seq_len: Number of content tokens per sequence; must be positive.
        vocab_size: Vocabulary size; must exceed 2 so that BOS, EOS, and at
            least one content token exist.
        seed: Seed passed to ``numpy.random.default_rng``; the same seed always
            yields the same dataset.

    Returns:
        A tuple ``(inputs, targets)`` of integer arrays with shape
        ``(n_samples, seq_len + 1)``. Row ``i`` of ``inputs`` is
        ``[BOS, t1, ..., tn]`` and row ``i`` of ``targets`` is
        ``[t1, ..., tn, EOS]``.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if vocab_size <= FIRST_CONTENT_TOKEN:
        raise ValueError("vocab_size must be greater than 2 (ids 0 and 1 are BOS/EOS)")
    rng = np.random.default_rng(seed)
    content = rng.integers(
        low=FIRST_CONTENT_TOKEN,
        high=vocab_size,
        size=(n_samples, seq_len),
    )
    inputs = np.empty((n_samples, seq_len + 1), dtype=np.int64)
    inputs[:, 0] = BOS_TOKEN
    inputs[:, 1:] = content
    targets = np.empty((n_samples, seq_len + 1), dtype=np.int64)
    targets[:, :-1] = content
    targets[:, -1] = EOS_TOKEN
    return inputs, targets


def make_copy_splits(
    n_train: int,
    n_heldout: int,
    seq_len: int,
    vocab_size: int,
    seed: int,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """Generate train and held-out copy datasets from a single base seed.

    The held-out split is drawn from an independent RNG stream derived from
    ``seed``, so for any fixed seed the two splits contain different sequences
    (the test suite verifies row-level disjointness).

    Args:
        n_train: Number of training sequences.
        n_heldout: Number of held-out sequences.
        seq_len: Number of content tokens per sequence.
        vocab_size: Vocabulary size.
        seed: Base seed; the held-out split uses ``seed + 1``.

    Returns:
        ``((train_inputs, train_targets), (heldout_inputs, heldout_targets))``.
    """
    train = make_copy_dataset(n_train, seq_len, vocab_size, seed=seed)
    heldout = make_copy_dataset(n_heldout, seq_len, vocab_size, seed=seed + 1)
    return train, heldout
