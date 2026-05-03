"""V3 FP8 (E4M3) fused FlashAttention -- Python entry point."""

from __future__ import annotations

try:
    from ._C import attention_flash_fp8_cu as _attention_flash_fp8_cu

    HAS_CUDA_EXT = True
except ImportError:
    _attention_flash_fp8_cu = None
    HAS_CUDA_EXT = False


def attention_flash_fp8_cu(Q, K, V, causal: bool = False):
    """Compiled V3 CUDA kernel: FP8 (E4M3) fused FlashAttention with per-tile scaling.

    Inputs are FP16; quantization to FP8 is performed inside the kernel with
    one FP32 scale per tile (Q, K, V, P). Output is FP16 (matches V0/V1/V2
    contract). Supports head_dim in {64, 128}, requires sm_120.
    """
    if _attention_flash_fp8_cu is None:
        raise ImportError(
            "v3_flash_fp8._C is not built. Run "
            "`pip install -e kernels/v3_flash_fp8` from the repo root."
        )
    return _attention_flash_fp8_cu(Q, K, V, causal)


__all__ = ["HAS_CUDA_EXT", "attention_flash_fp8_cu"]
