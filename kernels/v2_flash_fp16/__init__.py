"""V2 FP16 fused (FlashAttention-style) attention — Python entry point.

Online softmax + single-kernel fusion of QK^T, softmax, and PV. No
materialization of the full (B, H, N, N) attention matrix in HBM. See
``attention_v2.cu`` for the algorithmic details.
"""

from __future__ import annotations

try:
    from ._C import attention_flash_cu as _attention_flash_cu

    HAS_CUDA_EXT = True
except ImportError:
    _attention_flash_cu = None
    HAS_CUDA_EXT = False


def attention_flash_cu(Q, K, V, causal: bool = False):
    """Compiled V2 fused FlashAttention kernel."""
    if _attention_flash_cu is None:
        raise ImportError(
            "v2_flash_fp16._C is not built. Run "
            "`pip install -e kernels/v2_flash_fp16` from the repo root."
        )
    return _attention_flash_cu(Q, K, V, causal)


__all__ = ["HAS_CUDA_EXT", "attention_flash_cu"]
