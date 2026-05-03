"""V0 naive FP32 attention — Python entry points.

This package exposes both:

- :func:`attention_naive_pt` — pure-PyTorch reference (importable on any host).
- :func:`attention_naive_cu` — the compiled CUDA kernel (importable only after
  ``pip install -e kernels/v0_naive_fp32`` from the repo root, which builds
  the ``v0_naive_fp32._C`` extension targeting sm_120).

Both share the signature ``(Q, K, V, causal) -> Tensor`` and operate on
tensors of shape ``(batch, num_heads, seq_len, head_dim)``.

Install / use::

    # PowerShell, from repo root, with the `gpu` venv active:
    pip install -e kernels/v0_naive_fp32

    python -c "from v0_naive_fp32 import attention_naive_cu"
"""

from __future__ import annotations

from .reference_pytorch import attention_naive_pt, attention_naive_pt_fn

try:
    from ._C import attention_naive_cu as _attention_naive_cu

    HAS_CUDA_EXT = True
except ImportError:
    _attention_naive_cu = None
    HAS_CUDA_EXT = False


def attention_naive_cu(Q, K, V, causal: bool = False):
    """Compiled V0 CUDA kernel; raises ImportError until the extension is built."""
    if _attention_naive_cu is None:
        raise ImportError(
            "v0_naive_fp32._C is not built. Run "
            "`pip install -e kernels/v0_naive_fp32` from the repo root."
        )
    return _attention_naive_cu(Q, K, V, causal)


__all__ = [
    "HAS_CUDA_EXT",
    "attention_naive_cu",
    "attention_naive_pt",
    "attention_naive_pt_fn",
]
