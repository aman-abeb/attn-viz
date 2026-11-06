"""Tests for the pure-numpy attention core."""

from __future__ import annotations

import numpy as np
import pytest

from attnviz import demo
from attnviz.attention import (
    MultiHeadAttention,
    apply_causal_mask,
    scaled_dot_product_attention,
    softmax,
)


def test_scaled_dot_product_attention_output_shapes() -> None:
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2, 5, 8))
    k = rng.normal(size=(2, 5, 8))
    v = rng.normal(size=(2, 5, 4))
    weights, output = scaled_dot_product_attention(q, k, v)
    assert weights.shape == (2, 5, 5)
    assert output.shape == (2, 5, 4)


def test_attention_weights_are_valid_probabilities() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(3, 7, 6)) * 10.0
    k = rng.normal(size=(3, 7, 6)) * 10.0
    v = rng.normal(size=(3, 7, 6))
    weights, _ = scaled_dot_product_attention(q, k, v)
    assert np.isfinite(weights).all()
    assert (weights >= 0.0).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-12)


def test_softmax_is_stable_and_normalized() -> None:
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 999.0]])
    p = softmax(x)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=-1), 1.0, atol=1e-12)


def test_apply_causal_mask_is_lower_triangular() -> None:
    mask = apply_causal_mask(4)
    expected = np.array(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    np.testing.assert_array_equal(mask, expected)


def test_apply_causal_mask_rejects_non_positive_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_causal_mask(0)


def test_causal_mask_zeroes_future_attention() -> None:
    rng = np.random.default_rng(2)
    seq_len = 6
    q = rng.normal(size=(1, seq_len, 8))
    k = rng.normal(size=(1, seq_len, 8))
    v = rng.normal(size=(1, seq_len, 8))
    weights, _ = scaled_dot_product_attention(q, k, v, mask=apply_causal_mask(seq_len))
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    assert np.allclose(weights[0][future], 0.0, atol=1e-15)


def test_causal_mask_prevents_future_leakage() -> None:
    rng = np.random.default_rng(3)
    seq_len, d = 5, 8
    q = rng.normal(size=(1, seq_len, d))
    k = rng.normal(size=(1, seq_len, d))
    v = rng.normal(size=(1, seq_len, d))
    mask = apply_causal_mask(seq_len)
    _, out_a = scaled_dot_product_attention(q, k, v, mask=mask)
    k_future = k.copy()
    k_future[0, -1] += 100.0
    v_future = v.copy()
    v_future[0, -1] += 100.0
    _, out_b = scaled_dot_product_attention(q, k_future, v_future, mask=mask)
    np.testing.assert_allclose(out_a[0, :-1], out_b[0, :-1], atol=1e-12)


def test_multi_head_attention_output_shapes() -> None:
    mha = MultiHeadAttention(d_model=16, num_heads=4, seed=0)
    x = np.random.default_rng(4).normal(size=(2, 7, 16))
    output, attn = mha.forward(x)
    assert output.shape == (2, 7, 16)
    assert attn.shape == (2, 4, 7, 7)
    np.testing.assert_allclose(attn.sum(axis=-1), 1.0, atol=1e-12)


def test_multi_head_attention_causal_mask_blocks_future() -> None:
    seq_len = 5
    mha = MultiHeadAttention(d_model=8, num_heads=2, seed=0)
    x = np.random.default_rng(6).normal(size=(1, seq_len, 8))
    _, attn = mha.forward(x, mask=apply_causal_mask(seq_len))
    future = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    assert np.allclose(attn[0, :, future], 0.0, atol=1e-15)


def test_multi_head_attention_seed_determinism() -> None:
    x = np.random.default_rng(5).normal(size=(1, 4, 8))
    mha_a = MultiHeadAttention(d_model=8, num_heads=2, seed=42)
    mha_b = MultiHeadAttention(d_model=8, num_heads=2, seed=42)
    out_a, attn_a = mha_a.forward(x)
    out_b, attn_b = mha_b.forward(x)
    np.testing.assert_array_equal(mha_a.w_q, mha_b.w_q)
    np.testing.assert_array_equal(out_a, out_b)
    np.testing.assert_array_equal(attn_a, attn_b)
    mha_c = MultiHeadAttention(d_model=8, num_heads=2, seed=43)
    out_c, _ = mha_c.forward(x)
    assert not np.allclose(out_a, out_c)


def test_multi_head_attention_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(d_model=10, num_heads=3)


def test_demo_prints_attention_matrix(capsys: pytest.CaptureFixture[str]) -> None:
    demo.main()
    captured = capsys.readouterr()
    assert "attention" in captured.out.lower()
    assert "row sums" in captured.out.lower()
