# AttnViz

<!-- builder:badges:start -->
[![CI](https://github.com/aman-abeb/attn-viz/actions/workflows/ci.yml/badge.svg)](https://github.com/aman-abeb/attn-viz/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white) ![License: MIT](https://img.shields.io/badge/license-MIT-green)
<!-- builder:badges:end -->


Pure-numpy multi-head attention trained on a sequence-copying task, rendering heatmaps that reveal which tokens each head focuses on.

## Features

- Scaled dot-product attention, causal masking, and a multi-head attention block in pure numpy (`src/attnviz/attention.py`)
- Deterministic, seeded weight initialization and forward passes
- Runnable demo: `python -m attnviz.demo` prints a small causal attention matrix
- Offline pytest suite covering shapes, softmax normalization, causal masking, and seed determinism

## Quickstart

```bash
pip install -e ".[test]"
python -m attnviz.demo
```

## Example

`python -m attnviz.demo` runs a seeded forward pass (batch=1, seq_len=6, d_model=16, 2 heads) and prints:

```text
AttnViz demo: multi-head attention with causal masking
batch=1  seq_len=6  d_model=16  num_heads=2  seed=0

Causal mask (True = position may attend):
[[ True False False False False False]
 [ True  True False False False False]
 [ True  True  True False False False]
 [ True  True  True  True False False]
 [ True  True  True  True  True False]
 [ True  True  True  True  True  True]]

Attention weights, head 0 (rows = query, cols = key; rows sum to 1):
[[1.    0.    0.    0.    0.    0.   ]
 [0.38  0.62  0.    0.    0.    0.   ]
 ...                                  (abridged; deterministic for seed=0)
]
Row sums: [1. 1. 1. 1. 1. 1.]
```

The upper triangle of every attention row is exactly zero: the causal mask forbids attending to future tokens.

## How it works

- `scaled_dot_product_attention(Q, K, V, mask)` computes `softmax(Q K^T / sqrt(d_k)) V` with a numerically stable softmax; a boolean mask replaces masked scores with `-inf` before the softmax so they receive zero weight.
- `apply_causal_mask(seq_len)` returns a lower-triangular boolean mask, so position `i` can only attend to positions `j <= i`.
- `MultiHeadAttention` projects the input into per-head query/key/value subspaces with learned linear maps, runs scaled dot-product attention on all heads at once via batched matmuls of shape `(batch, heads, seq, d_head)`, concatenates the head outputs, and applies a final output projection. The per-head weight matrices are returned alongside the output so they can be rendered as heatmaps later.

## Running tests

```bash
python -m pytest
```

Tests run fully offline with fixed seeds; no network access or API keys are needed.

<!-- builder:roadmap:start -->
## Status

**In active development:** 0 of 3 milestones done.

## Roadmap

- [ ] **Numpy attention core with tests**: Implement scaled dot-product attention, causal masking, and a multi-head attention block in pure numpy under src/attnviz/. Add pytest tests for output shapes, softmax rows summing to 1, causal mask preventing future leakage, seed determinism, and a runnable demo: python -m attnviz.demo prints a small attention matrix.
- [ ] **Trainable attention-only copier**: Add manual backprop for the attention-only model, a synthetic sequence-copying dataset generator, and a training CLI: python -m attnviz.train --epochs 200 --out model.npz. Tests include numerical gradient checks, loss decreasing over epochs, and held-out exact-match accuracy above threshold; everything runs offline with fixed seeds.
- [ ] **Heatmap visualizer, CLI, and README showcase**: Add a matplotlib Agg-backend renderer that saves per-head attention heatmaps for sample inputs, plus python -m attnviz.viz --model model.npz --out examples/. Generate committed example PNGs, write a README that embeds the heatmaps and explains the learned copy pattern, and add tests asserting PNGs are created, non-empty, and backed by valid probability matrices.
<!-- builder:roadmap:end -->
