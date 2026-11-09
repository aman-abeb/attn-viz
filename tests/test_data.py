"""Tests for the seeded sequence-copying dataset generator."""

from __future__ import annotations

import numpy as np
import pytest

from attnviz.data import (
    BOS_TOKEN,
    EOS_TOKEN,
    FIRST_CONTENT_TOKEN,
    make_copy_dataset,
    make_copy_splits,
)


def test_dataset_shapes_and_dtypes() -> None:
    inputs, targets = make_copy_dataset(n_samples=8, seq_len=5, vocab_size=12, seed=0)
    assert inputs.shape == (8, 6)
    assert targets.shape == (8, 6)
    assert np.issubdtype(inputs.dtype, np.integer)
    assert np.issubdtype(targets.dtype, np.integer)


def test_bos_and_eos_placement() -> None:
    inputs, targets = make_copy_dataset(16, 4, 10, seed=1)
    assert (inputs[:, 0] == BOS_TOKEN).all()
    assert (targets[:, -1] == EOS_TOKEN).all()
    assert (inputs[:, 1:] >= FIRST_CONTENT_TOKEN).all()
    assert (targets[:, :-1] >= FIRST_CONTENT_TOKEN).all()


def test_targets_are_shifted_inputs() -> None:
    inputs, targets = make_copy_dataset(16, 6, 10, seed=2)
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])


def test_token_range_respects_vocab_size() -> None:
    inputs, targets = make_copy_dataset(32, 7, 9, seed=3)
    assert inputs.min() >= BOS_TOKEN
    assert inputs.max() < 9
    assert targets.min() >= BOS_TOKEN
    assert targets.max() < 9


def test_same_seed_is_deterministic() -> None:
    a = make_copy_dataset(10, 5, 12, seed=42)
    b = make_copy_dataset(10, 5, 12, seed=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_different_seed_differs() -> None:
    a = make_copy_dataset(10, 5, 12, seed=42)
    b = make_copy_dataset(10, 5, 12, seed=43)
    assert not np.array_equal(a[0], b[0])


def test_invalid_arguments_raise() -> None:
    with pytest.raises(ValueError, match="n_samples"):
        make_copy_dataset(0, 5, 12, seed=0)
    with pytest.raises(ValueError, match="seq_len"):
        make_copy_dataset(4, 0, 12, seed=0)
    with pytest.raises(ValueError, match="vocab_size"):
        make_copy_dataset(4, 5, 2, seed=0)


def test_train_heldout_splits_are_disjoint() -> None:
    (train_in, _), (held_in, _) = make_copy_splits(
        n_train=64, n_heldout=32, seq_len=6, vocab_size=16, seed=7
    )
    train_rows = {tuple(row) for row in train_in.tolist()}
    held_rows = {tuple(row) for row in held_in.tolist()}
    assert len(train_rows) == 64
    assert len(held_rows) == 32
    assert train_rows.isdisjoint(held_rows)


def test_splits_match_individual_generation() -> None:
    (train_in, train_tg), (held_in, held_tg) = make_copy_splits(5, 3, 4, 10, seed=11)
    ref_train = make_copy_dataset(5, 4, 10, seed=11)
    ref_held = make_copy_dataset(3, 4, 10, seed=12)
    np.testing.assert_array_equal(train_in, ref_train[0])
    np.testing.assert_array_equal(train_tg, ref_train[1])
    np.testing.assert_array_equal(held_in, ref_held[0])
    np.testing.assert_array_equal(held_tg, ref_held[1])
