"""V4 NVFP4 (microscaled FP4) fused attention — Python entry point (stub)."""

from __future__ import annotations

try:
    from ._C import attention_flash_nvfp4_cu as _attention_flash_nvfp4_cu

    HAS_CUDA_EXT = True
except ImportError:
    _attention_flash_nvfp4_cu = None
    HAS_CUDA_EXT = False


def attention_flash_nvfp4_cu(Q, K, V, causal: bool = False):
    """Compiled V4 CUDA kernel — stub; raises until implemented."""
    if _attention_flash_nvfp4_cu is None:
        raise ImportError(
            "v4_flash_nvfp4._C is not built. Run "
            "`pip install -e kernels/v4_flash_nvfp4` from the repo root."
        )
    return _attention_flash_nvfp4_cu(Q, K, V, causal)


__all__ = ["HAS_CUDA_EXT", "attention_flash_nvfp4_cu"]
