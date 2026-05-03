"""V1 FP16 tiled (Tensor Core) attention — Python entry point (stub).

Future sessions will implement the WMMA / CUTLASS kernel. Until then,
``HAS_CUDA_EXT`` reports the actual build state of the compiled extension
(False if you haven't run ``pip install -e kernels/v1_tiled_fp16``, True if
you have — but the function still raises on call until the kernel body
in ``attention_v1.cu`` is filled in).
"""

from __future__ import annotations

try:
    from ._C import attention_tiled_cu as _attention_tiled_cu

    HAS_CUDA_EXT = True
except ImportError:
    _attention_tiled_cu = None
    HAS_CUDA_EXT = False


def attention_tiled_cu(Q, K, V, causal: bool = False):
    """Compiled V1 CUDA kernel — stub; raises until implemented."""
    if _attention_tiled_cu is None:
        raise ImportError(
            "v1_tiled_fp16._C is not built. Run "
            "`pip install -e kernels/v1_tiled_fp16` from the repo root."
        )
    return _attention_tiled_cu(Q, K, V, causal)


__all__ = ["HAS_CUDA_EXT", "attention_tiled_cu"]
