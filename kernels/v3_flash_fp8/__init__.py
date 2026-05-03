"""V3 FP8 (E4M3) fused attention — Python entry point (stub)."""

from __future__ import annotations

try:
    from ._C import attention_flash_fp8_cu as _attention_flash_fp8_cu

    HAS_CUDA_EXT = True
except ImportError:
    _attention_flash_fp8_cu = None
    HAS_CUDA_EXT = False


def attention_flash_fp8_cu(Q, K, V, causal: bool = False):
    """Compiled V3 CUDA kernel — stub; raises until implemented."""
    if _attention_flash_fp8_cu is None:
        raise ImportError(
            "v3_flash_fp8._C is not built. Run "
            "`pip install -e kernels/v3_flash_fp8` from the repo root."
        )
    return _attention_flash_fp8_cu(Q, K, V, causal)


__all__ = ["HAS_CUDA_EXT", "attention_flash_fp8_cu"]
