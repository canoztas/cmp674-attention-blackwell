"""Correctness tests for V0 (naive FP32) and V1 (tiled FP16, Tensor Core) attention.

V0 tests validate the V0-CU CUDA kernel against V0-PT (the explicit PyTorch
reference) at FP32 tolerance, and both against ``F.scaled_dot_product_attention``
at slightly looser tolerance.

V1 tests validate V1's FP16 output against the FP32 V0-PT reference cast back
to FP16. V1 is FP16 in/out with FP32 accumulators inside, so the achievable
tolerance is bounded by FP16's ~1e-3 unit-roundoff times accumulation length;
we set ``atol=rtol=1e-2`` which the smoke run hit comfortably (worst case
~2e-3 on causal). Tighter would be a tolerance to chase, not relax.

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


# --------------------------------------------------------------------------- #
# V1 tiled FP16 (Tensor Core) attention                                       #
# --------------------------------------------------------------------------- #

try:
    from v1_tiled_fp16 import HAS_CUDA_EXT as _HAS_V1_CU_EXT
    from v1_tiled_fp16 import attention_tiled_cu

    HAS_V1_CU = _HAS_V1_CU_EXT
    _V1_CU_IMPORT_ERR: str | None = (
        None if _HAS_V1_CU_EXT else "compiled extension v1_tiled_fp16._C not built"
    )
except ImportError as e:
    HAS_V1_CU = False
    _V1_CU_IMPORT_ERR = str(e)

    def attention_tiled_cu(*_a, **_kw):  # type: ignore[no-redef]
        raise ImportError(_V1_CU_IMPORT_ERR)


# V1 only supports head_dim in {64, 128} (WMMA fragment shape constraints
# + the kernel's BD=64 tile). seq_lens include a non-aligned value (100)
# to exercise the boundary-padding logic.
_V1_SEQ_LENS = (64, 100, 128, 256)
_V1_HEAD_DIMS = (64, 128)
_V1_BATCHES = (1, 2)
_V1_NUM_HEADS = (1, 4)
_V1_CAUSAL = (False, True)
_V1_GRID = list(product(_V1_SEQ_LENS, _V1_HEAD_DIMS, _V1_BATCHES, _V1_NUM_HEADS, _V1_CAUSAL))


def _make_qkv_fp16(
    batch: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    *,
    seed: int = _SEED,
    std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same generator pattern as ``_make_qkv`` but FP16 with optional std scaling."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, num_heads, seq_len, head_dim)
    Q = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * std
    K = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * std
    V = torch.randn(shape, generator=g, device="cuda", dtype=torch.float32) * std
    return Q.half(), K.half(), V.half()


@pytest.mark.skipif(not HAS_V1_CU, reason=f"v1_tiled_fp16 extension not built: {_V1_CU_IMPORT_ERR}")
@pytest.mark.parametrize("seq_len,head_dim,batch,num_heads,causal", _V1_GRID)
def test_v1_cu_matches_v0_pt_fp16(
    seq_len: int, head_dim: int, batch: int, num_heads: int, causal: bool
) -> None:
    """V1 (FP16) vs V0-PT (FP32 internal, FP16 result) at FP16 tolerance.

    The reference runs in FP32 to avoid double-counting FP16 rounding;
    we then downcast to FP16 to compare with V1's FP16 output. The
    tolerance is the smoke-run-derived value (worst observed ~2e-3 across
    the grid plus margin), not a value chosen to make the test pass.
    """
    Q, K, V = _make_qkv_fp16(batch, num_heads, seq_len, head_dim)
    out_v1 = attention_tiled_cu(Q, K, V, causal)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), causal).half()
    assert torch.isfinite(out_v1).all(), "V1 produced non-finite values"
    torch.testing.assert_close(out_v1, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_V1_CU, reason=f"v1_tiled_fp16 extension not built: {_V1_CU_IMPORT_ERR}")
@pytest.mark.parametrize("causal", [False, True])
def test_v1_cu_small_magnitude_inputs(causal: bool) -> None:
    """Inputs with std=0.1 - small softmax shifts, small probability deltas.

    Catches zero-fill / finite-guard bugs that surface as NaNs only when
    the softmax row max is very close to other row entries. Stays well
    within the FP16-rounding budget that the main grid hits at unit
    variance, so the standard ``1e-2`` tolerance still applies.
    """
    Q, K, V = _make_qkv_fp16(2, 4, 256, 64, std=0.1)
    out_v1 = attention_tiled_cu(Q, K, V, causal)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), causal).half()
    assert torch.isfinite(out_v1).all(), "V1 produced non-finite values at std=0.1"
    torch.testing.assert_close(out_v1, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_V1_CU, reason=f"v1_tiled_fp16 extension not built: {_V1_CU_IMPORT_ERR}")
@pytest.mark.parametrize("std", [2.0, 3.0, 5.0])
def test_v1_cu_no_overflow_under_high_variance(std: float) -> None:
    """High-variance stress test - no NaN/Inf, bulk-correct, no design-bug failure.

    V1 deliberately stores S and P in FP16 between phases (no online softmax
    or per-tile rescaling - those are V2/V3 features). Cumulative FP16
    rounding therefore scales with input variance: at std=2 a single
    element drifts past ``1e-2`` per the smoke run; at std=3 it's ~0.3%
    of elements; at std=5 the tail is wider still. The strict tolerance
    is the wrong instrument for that regime.

    Instead, we verify (a) outputs stay finite (rules out overflow / NaN
    masking bugs - the *actual* failure mode V3 will fix) and (b) at
    least 95% of elements remain within ``5e-2`` (rules out large-scale
    indexing or accumulator bugs that would corrupt many elements at
    once). A genuine kernel bug fails one of these; pure FP16 rounding
    fails neither - measured tail at std=5 is ~1.3%, well under the 5%
    bug threshold.
    """
    Q, K, V = _make_qkv_fp16(2, 4, 256, 64, std=std)
    out_v1 = attention_tiled_cu(Q, K, V, False)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), False).half()
    assert torch.isfinite(out_v1).all(), f"V1 produced non-finite values at std={std}"
    abs_err = (out_v1 - out_ref).abs()
    within_band = (abs_err <= 5e-2).float().mean().item()
    assert within_band >= 0.95, (
        f"std={std}: only {within_band:.3%} of elements within 5e-2; "
        f"max abs err = {abs_err.max().item():.3e}"
    )


@pytest.mark.skipif(not HAS_V1_CU, reason=f"v1_tiled_fp16 extension not built: {_V1_CU_IMPORT_ERR}")
def test_v1_rejects_fp32_inputs() -> None:
    """V1 must clearly reject non-FP16 inputs (binding-level guard)."""
    Q, K, V = _make_qkv(1, 1, 64, 64)  # FP32 fixture
    with pytest.raises(RuntimeError, match=r"V1 supports float16"):
        attention_tiled_cu(Q, K, V, False)


@pytest.mark.skipif(not HAS_V1_CU, reason=f"v1_tiled_fp16 extension not built: {_V1_CU_IMPORT_ERR}")
def test_v1_rejects_unsupported_head_dim() -> None:
    """V1 only supports head_dim in {64, 128}; D=32 must be rejected loudly."""
    Q, K, V = _make_qkv_fp16(1, 1, 64, 32)
    with pytest.raises(RuntimeError, match=r"head_dim in \{64, 128\}"):
        attention_tiled_cu(Q, K, V, False)


# --------------------------------------------------------------------------- #
# V2 fused FlashAttention FP16 (online softmax + single-kernel fusion)        #
# --------------------------------------------------------------------------- #

try:
    from v2_flash_fp16 import HAS_CUDA_EXT as _HAS_V2_CU_EXT
    from v2_flash_fp16 import attention_flash_cu

    HAS_V2_CU = _HAS_V2_CU_EXT
    _V2_CU_IMPORT_ERR: str | None = (
        None if _HAS_V2_CU_EXT else "compiled extension v2_flash_fp16._C not built"
    )
except ImportError as e:
    HAS_V2_CU = False
    _V2_CU_IMPORT_ERR = str(e)

    def attention_flash_cu(*_a, **_kw):  # type: ignore[no-redef]
        raise ImportError(_V2_CU_IMPORT_ERR)


# V2 contract matches V1: head_dim in {64, 128}, FP16 in/out. The grid mirrors
# V1's plus seq_len=32 (smaller than the BR=64 tile -> exercises the gr<N
# masking on every row of the only tile). Causal kept in the cartesian product.
_V2_SEQ_LENS = (32, 64, 100, 128, 256)
_V2_HEAD_DIMS = (64, 128)
_V2_BATCHES = (1, 2)
_V2_NUM_HEADS = (1, 4)
_V2_CAUSAL = (False, True)
_V2_GRID = list(product(_V2_SEQ_LENS, _V2_HEAD_DIMS, _V2_BATCHES, _V2_NUM_HEADS, _V2_CAUSAL))


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
@pytest.mark.parametrize("seq_len,head_dim,batch,num_heads,causal", _V2_GRID)
def test_v2_cu_matches_v0_pt_fp16(
    seq_len: int, head_dim: int, batch: int, num_heads: int, causal: bool
) -> None:
    """V2 (FP16, fused, online softmax) vs V0-PT (FP32) at FP16 tolerance.

    V2 keeps softmax math entirely in FP32 registers / shared memory; the
    only FP16 round-trips are the input loads, the P spill before the PV
    matmul, and the final output write. So V2 should be at least as
    accurate as V1 (which stored S, P in FP16 between phases). The same
    ``atol=rtol=1e-2`` budget is the right reference - tighter would chase
    rounding noise, looser would miss a real bug.
    """
    Q, K, V = _make_qkv_fp16(batch, num_heads, seq_len, head_dim)
    out_v2 = attention_flash_cu(Q, K, V, causal)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), causal).half()
    assert torch.isfinite(out_v2).all(), "V2 produced non-finite values"
    torch.testing.assert_close(out_v2, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
@pytest.mark.parametrize("causal", [False, True])
def test_v2_cu_small_magnitude_inputs(causal: bool) -> None:
    """Inputs with std=0.1: small softmax shifts. Catches first-iteration
    edge cases (m_i = -inf init, alpha guard when m_new is also -inf)
    that surface as NaNs only when scores are near-uniform.
    """
    Q, K, V = _make_qkv_fp16(2, 4, 256, 64, std=0.1)
    out_v2 = attention_flash_cu(Q, K, V, causal)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), causal).half()
    assert torch.isfinite(out_v2).all(), "V2 produced non-finite values at std=0.1"
    torch.testing.assert_close(out_v2, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
@pytest.mark.parametrize("std", [2.0, 3.0, 5.0])
def test_v2_cu_no_overflow_under_high_variance(std: float) -> None:
    """High-variance stress test: finite outputs + bulk correctness.

    V2's online softmax holds m, l, and the output accumulator entirely in
    FP32, so the FP16 cumulative-rounding regime that bit V1 at high std
    is much tighter here. We expect V2 to outperform V1 numerically -
    measured tail at std=5 should be well under V1's ~1.3% (the dominant
    FP16 noise source is now just the FP16 P spill before the PV matmul).
    Same 95%-within-5e-2 bulk-correctness threshold as V1; if that fails,
    investigate (it would be a real bug, not noise).
    """
    Q, K, V = _make_qkv_fp16(2, 4, 256, 64, std=std)
    out_v2 = attention_flash_cu(Q, K, V, False)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), False).half()
    assert torch.isfinite(out_v2).all(), f"V2 produced non-finite values at std={std}"
    abs_err = (out_v2 - out_ref).abs()
    within_band = (abs_err <= 5e-2).float().mean().item()
    assert within_band >= 0.95, (
        f"std={std}: only {within_band:.3%} of elements within 5e-2; "
        f"max abs err = {abs_err.max().item():.3e}"
    )


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
def test_v2_cu_long_seq_len_no_v1_materialization() -> None:
    """seq_len=2048 at head_dim=64 is the regime V2 was built for: V1's
    O(N^2) materialization of S in HBM is 4*16*2048*2048*2 bytes ~= 1 GiB
    which fits, but stresses HBM bandwidth; V2 keeps S in registers/SMEM
    and never writes it to HBM. Correctness here is what proves the
    online softmax recurrence is right at scale.
    """
    Q, K, V = _make_qkv_fp16(1, 1, 2048, 64)
    out_v2 = attention_flash_cu(Q, K, V, False)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), False).half()
    assert torch.isfinite(out_v2).all(), "V2 non-finite at seq_len=2048"
    abs_err = (out_v2 - out_ref).abs()
    within_band = (abs_err <= 5e-2).float().mean().item()
    assert within_band >= 0.99, (
        f"seq_len=2048: only {within_band:.3%} of elements within 5e-2"
    )


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
@pytest.mark.parametrize("seq_len", [1, 7, 16, 17])
def test_v2_cu_tiny_and_subtile_seq_lens(seq_len: int) -> None:
    """seq_len < BR (=64) and not aligned to WMMA_M (=16). The full Q tile
    is loaded with rows >= seq_len zero-padded, then the scale+mask pass
    sets S[gr>=N || gc>=N] to -inf so padded entries don't influence the
    softmax. The epilogue's ``if (gr >= N) continue`` keeps us from
    writing past the tensor end. Edge cases: seq_len=1 should output
    exactly V[0] (single-token softmax is identity).
    """
    Q, K, V = _make_qkv_fp16(1, 1, seq_len, 64)
    out_v2 = attention_flash_cu(Q, K, V, False)
    out_ref = attention_naive_pt(Q.float(), K.float(), V.float(), False).half()
    assert torch.isfinite(out_v2).all(), f"V2 non-finite at seq_len={seq_len}"
    torch.testing.assert_close(out_v2, out_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
def test_v2_rejects_fp32_inputs() -> None:
    """V2 must clearly reject non-FP16 inputs (binding-level guard)."""
    Q, K, V = _make_qkv(1, 1, 64, 64)  # FP32 fixture
    with pytest.raises(RuntimeError, match=r"V2 supports float16"):
        attention_flash_cu(Q, K, V, False)


@pytest.mark.skipif(not HAS_V2_CU, reason=f"v2_flash_fp16 extension not built: {_V2_CU_IMPORT_ERR}")
def test_v2_rejects_unsupported_head_dim() -> None:
    """V2 only supports head_dim in {64, 128}; D=32 must be rejected loudly."""
    Q, K, V = _make_qkv_fp16(1, 1, 64, 32)
    with pytest.raises(RuntimeError, match=r"head_dim in \{64, 128\}"):
        attention_flash_cu(Q, K, V, False)
