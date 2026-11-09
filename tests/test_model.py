"""Tests for the attention-only copier forward pass and npz persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from attnviz.data import make_copy_dataset
from attnviz.model import AttentionCopier


def _model(seed: int = 0) -> AttentionCopier:
    return AttentionCopier(
        vocab_size=12, d_model=16, num_heads=4, num_layers=2, max_seq_len=16, seed=seed
    )


def test_forward_logit_and_attention_shapes() -> None:
    model = _model()
    inputs, _ = make_copy_dataset(4, 6, 12, seed=0)
    logits, attn = model.forward(inputs)
    assert logits.shape == (4, 7, 12)
    assert attn.shape == (2, 4, 4, 7, 7)
    assert np.isfinite(logits).all()
    assert np.isfinite(attn).all()


def test_attention_weights_are_causal_probabilities() -> None:
    model = _model()
    inputs, _ = make_copy_dataset(3, 5, 12, seed=1)
    _, attn = model.forward(inputs)
    seq_len = inputs.shape[1]
    np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-12)
    assert (attn >= 0.0).all()
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    assert np.allclose(attn[..., future], 0.0, atol=1e-15)


def test_seed_determinism() -> None:
    inputs, _ = make_copy_dataset(2, 5, 12, seed=2)
    logits_a, attn_a = _model(seed=123).forward(inputs)
    logits_b, attn_b = _model(seed=123).forward(inputs)
    np.testing.assert_array_equal(logits_a, logits_b)
    np.testing.assert_array_equal(attn_a, attn_b)
    logits_c, _ = _model(seed=124).forward(inputs)
    assert not np.allclose(logits_a, logits_c)


def test_call_alias_matches_forward() -> None:
    model = _model(seed=5)
    inputs, _ = make_copy_dataset(2, 4, 12, seed=6)
    logits_a, attn_a = model.forward(inputs)
    logits_b, attn_b = model(inputs)
    np.testing.assert_array_equal(logits_a, logits_b)
    np.testing.assert_array_equal(attn_a, attn_b)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    model = _model(seed=9)
    inputs, _ = make_copy_dataset(3, 6, 12, seed=3)
    logits_a, attn_a = model.forward(inputs)
    path = tmp_path / "model.npz"
    model.save(path)
    assert path.exists()
    assert path.stat().st_size > 0
    loaded = AttentionCopier.load(path)
    assert loaded.vocab_size == model.vocab_size
    assert loaded.d_model == model.d_model
    assert loaded.num_heads == model.num_heads
    assert loaded.num_layers == model.num_layers
    assert loaded.max_seq_len == model.max_seq_len
    logits_b, attn_b = loaded.forward(inputs)
    np.testing.assert_array_equal(logits_a, logits_b)
    np.testing.assert_array_equal(attn_a, attn_b)


def test_forward_accepts_max_seq_len_boundary() -> None:
    model = _model()
    inputs, _ = make_copy_dataset(2, 15, 12, seed=4)
    logits, attn = model.forward(inputs)
    assert logits.shape == (2, 16, 12)
    assert attn.shape == (2, 2, 4, 16, 16)


def test_forward_rejects_bad_input() -> None:
    model = _model()
    with pytest.raises(ValueError, match="batch"):
        model.forward(np.zeros((2, 3, 4), dtype=np.int64))
    inputs, _ = make_copy_dataset(2, 20, 12, seed=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        model.forward(inputs)


def test_invalid_config_raises() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        AttentionCopier(vocab_size=0)
    with pytest.raises(ValueError, match="num_layers"):
        AttentionCopier(vocab_size=8, num_layers=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        AttentionCopier(vocab_size=8, max_seq_len=0)
    with pytest.raises(ValueError, match="divisible"):
        AttentionCopier(vocab_size=8, d_model=10, num_heads=3)
