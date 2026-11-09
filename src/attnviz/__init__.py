"""AttnViz: pure-numpy multi-head attention with attention heatmap visualization."""

from attnviz.attention import (
    MultiHeadAttention,
    apply_causal_mask,
    scaled_dot_product_attention,
    softmax,
)
from attnviz.data import (
    BOS_TOKEN,
    EOS_TOKEN,
    FIRST_CONTENT_TOKEN,
    make_copy_dataset,
    make_copy_splits,
)
from attnviz.model import AttentionCopier

__all__ = [
    "AttentionCopier",
    "BOS_TOKEN",
    "EOS_TOKEN",
    "FIRST_CONTENT_TOKEN",
    "MultiHeadAttention",
    "apply_causal_mask",
    "make_copy_dataset",
    "make_copy_splits",
    "scaled_dot_product_attention",
    "softmax",
]
__version__ = "0.1.0"
