"""V0-PT: pure-PyTorch naive scaled dot-product attention.

This is the inverse of FlashAttention (Dao et al., arXiv 2205.14135): it
explicitly materializes the full (B, H, N, N) attention matrix in HBM with
no fusion and no IO awareness. It exists to establish a numerical reference
for V0..V4 — *not* as a competitive baseline.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def attention_naive_pt(Q: Tensor, K: Tensor, V: Tensor, causal: bool = False) -> Tensor:
    """Naive scaled dot-product attention. Reference for correctness.

    Shapes: Q, K, V are ``(batch, num_heads, seq_len, head_dim)``. Returns a
    tensor of the same shape. Uses the explicit pipeline
    ``Q @ K^T / sqrt(d) -> mask -> softmax -> @ V`` rather than
    ``torch.nn.functional.scaled_dot_product_attention`` so the math is
    transparent and matches the V0 CUDA kernel's algorithm exactly.

    Args:
        Q, K, V: query / key / value tensors, identical shape and dtype.
        causal: if True, apply an upper-triangular ``-inf`` mask so position
            ``i`` can only attend to positions ``<= i``.

    Returns:
        Output tensor with shape ``(batch, num_heads, seq_len, head_dim)``.
    """
    if Q.shape != K.shape or Q.shape != V.shape:
        raise ValueError(f"Q/K/V shape mismatch: {Q.shape}, {K.shape}, {V.shape}")
    if Q.dim() != 4:
        raise ValueError(f"expected 4-D (B, H, N, D); got shape {tuple(Q.shape)}")

    head_dim = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    if causal:
        seq_len = Q.shape[-2]
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device).triu_(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, V)


# Harness-callable alias. The harness's ``KernelFn`` signature is
# ``(Q, K, V, causal) -> Tensor``, which ``attention_naive_pt`` already
# satisfies; the alias is here so callers don't need to remember the rename.
attention_naive_pt_fn = attention_naive_pt


__all__ = ["attention_naive_pt", "attention_naive_pt_fn"]
