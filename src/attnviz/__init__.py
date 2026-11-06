"""AttnViz: pure-numpy multi-head attention with attention heatmap visualization."""

from attnviz.attention import (
    MultiHeadAttention,
    apply_causal_mask,
    scaled_dot_product_attention,
    softmax,
)

__all__ = [
    "MultiHeadAttention",
    "apply_causal_mask",
    "scaled_dot_product_attention",
    "softmax",
]
__version__ = "0.1.0"
