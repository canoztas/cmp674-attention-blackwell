"""Correctness tests for V0 naive FP32 attention.

Validates that the V0-CU CUDA kernel agrees with V0-PT (the explicit
PyTorch reference) and, more loosely, with PyTorch's built-in
``F.scaled_dot_product_attention``.

Determinism note: a single fixed seed (``seed=0``) is used per test. Three
seeds was overkill given the small grid; if a numerical regression
appears, expand here.
"""

from __future__ import annotations

from itertools import product

import pytest
import torch
import torch.nn.functional as F

try:
    from v0_naive_fp32.reference_pytorch import attention_naive_pt
except ImportError:
    # Fall back to a path-based import so the V0-PT correctness checks can
    # run before `pip install -e kernels/v0_naive_fp32` has been executed.
    import importlib.util
    import pathlib
    import sys

    _ref_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "kernels"
        / "v0_naive_fp32"
        / "reference_pytorch.py"
    )
    _spec = importlib.util.spec_from_file_location("_v0_pt_reference", _ref_path)
    assert _spec is not None and _spec.loader is not None
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["_v0_pt_reference"] = _mod
    _spec.loader.exec_module(_mod)
    attention_naive_pt = _mod.attention_naive_pt

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


try:
    from v0_naive_fp32 import HAS_CUDA_EXT as _HAS_V0_CU_EXT
    from v0_naive_fp32 import attention_naive_cu

    HAS_V0_CU = _HAS_V0_CU_EXT
    _V0_CU_IMPORT_ERR: str | None = (
        None if _HAS_V0_CU_EXT else "compiled extension v0_naive_fp32._C not built"
    )
except ImportError as e:
    HAS_V0_CU = False
    _V0_CU_IMPORT_ERR = str(e)

    def attention_naive_cu(*_a, **_kw):  # type: ignore[no-redef]
        raise ImportError(_V0_CU_IMPORT_ERR)


_SEQ_LENS = (64, 128, 256)
_HEAD_DIMS = (32, 64)
_BATCHES = (1, 2)
_NUM_HEADS = (1, 4)
_CAUSAL = (False, True)
_GRID = list(product(_SEQ_LENS, _HEAD_DIMS, _BATCHES, _NUM_HEADS, _CAUSAL))
_SEED = 0


def _make_qkv(
    batch: int, num_heads: int, seq_len: int, head_dim: int, seed: int = _SEED
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, num_heads, seq_len, head_dim)
    Q = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)
    K = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)
    V = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32)
    return Q, K, V


@pytest.mark.skipif(not HAS_V0_CU, reason=f"v0_naive_fp32 extension not built: {_V0_CU_IMPORT_ERR}")
@pytest.mark.parametrize("seq_len,head_dim,batch,num_heads,causal", _GRID)
def test_v0_cu_matches_v0_pt(
    seq_len: int, head_dim: int, batch: int, num_heads: int, causal: bool
) -> None:
    """V0-CU must match the V0-PT reference to FP32 tolerance."""
    Q, K, V = _make_qkv(batch, num_heads, seq_len, head_dim)
    out_cu = attention_naive_cu(Q, K, V, causal)
    out_pt = attention_naive_pt(Q, K, V, causal)
    torch.testing.assert_close(out_cu, out_pt, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq_len,head_dim,batch,num_heads,causal", _GRID)
def test_v0_pt_matches_sdpa(
    seq_len: int, head_dim: int, batch: int, num_heads: int, causal: bool
) -> None:
    """V0-PT vs F.scaled_dot_product_attention.

    Slightly looser tolerance (atol=rtol=1e-4) because PyTorch's SDPA may
    dispatch to a different backend with a different reduction order. A
    failure here is interesting but not necessarily a bug in V0.
    """
    Q, K, V = _make_qkv(batch, num_heads, seq_len, head_dim)
    out_pt = attention_naive_pt(Q, K, V, causal)
    out_sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    torch.testing.assert_close(out_pt, out_sdpa, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not HAS_V0_CU, reason=f"v0_naive_fp32 extension not built: {_V0_CU_IMPORT_ERR}")
@pytest.mark.parametrize("seq_len,head_dim,batch,num_heads,causal", _GRID)
def test_v0_cu_matches_sdpa(
    seq_len: int, head_dim: int, batch: int, num_heads: int, causal: bool
) -> None:
    """V0-CU vs F.scaled_dot_product_attention. Same loose tolerance as V0-PT."""
    Q, K, V = _make_qkv(batch, num_heads, seq_len, head_dim)
    out_cu = attention_naive_cu(Q, K, V, causal)
    out_sdpa = F.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    torch.testing.assert_close(out_cu, out_sdpa, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not HAS_V0_CU, reason=f"v0_naive_fp32 extension not built: {_V0_CU_IMPORT_ERR}")
@pytest.mark.parametrize("causal", [False, True])
def test_v0_cu_minimal_shape(causal: bool) -> None:
    """Edge case: seq_len=1, head_dim=1, batch=1, num_heads=1.

    Stresses indexing arithmetic and softmax-with-one-element behavior
    (exp(x - x) / 1 == 1, so the output equals V exactly).
    """
    Q, K, V = _make_qkv(1, 1, 1, 1)
    out_cu = attention_naive_cu(Q, K, V, causal)
    out_pt = attention_naive_pt(Q, K, V, causal)
    torch.testing.assert_close(out_cu, out_pt, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out_cu, V, atol=1e-6, rtol=1e-6)
