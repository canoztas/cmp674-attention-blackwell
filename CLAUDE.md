# Project: attention-blackwell

Mixed-precision attention kernel benchmarks on NVIDIA GeForce RTX 5080 (Blackwell, sm_120).
Goal: a 4–6 week graduate research paper (IEEE format, 8–12 pages) mapping the Pareto
frontier of latency vs. numerical accuracy vs. memory across five progressively optimized
attention variants, plus an open-source reproducible repository. Quality bar is graduate
publication, not demo code.

## Hardware target

- **GPU:** NVIDIA GeForce RTX 5080
- **Architecture:** Blackwell, compute capability 12.0 (sm_120)
- **VRAM:** 16 GB GDDR7, ~960 GB/s peak memory bandwidth
- **Tensor Cores:** 5th generation, native FP8 (E4M3 / E5M2) and FP4 (NVFP4)
- **OS:** Windows 11; default shell PowerShell (pwsh)
- **CUDA Toolkit:** 12.8 (via the PyTorch wheel)
- **PyTorch:** 2.9.1+cu128

## Environment activation

The user has a custom PowerShell command, `gpu`, that activates the shared CUDA venv,
configures CUDA paths, and prints a hardware/driver summary. Run it before any Python
command:

```powershell
gpu
```

The `gpu` venv is **shared across multiple GPU/CUDA projects** — do not create a
project-local venv. Pinned dependency versions live in [env/requirements.txt](env/requirements.txt)
and are installed inside the `gpu` venv with:

```powershell
pip install -r env/requirements.txt
```

`pyproject.toml` holds project metadata and tooling config only (ruff / black / pytest);
runtime deps are not declared there.

## Variant roadmap

| ID  | Precision   | Design                                                          | Status          |
| --- | ----------- | --------------------------------------------------------------- | --------------- |
| V0  | FP32        | Naive — materializes the full N×N attention matrix in HBM        | **In progress** |
| V1  | FP16        | Tiled, Tensor Core (WMMA / CUTLASS), still materializes outputs  | Pending         |
| V2  | FP16        | Fused + online softmax (FlashAttention-style)                    | Pending         |
| V3  | FP8 (E4M3)  | V2 + per-tile scaling                                            | Pending         |
| V4  | FP4 (NVFP4) | V3 + microscaling (Blackwell-only, exploratory)                  | Pending         |

**One variant per session.** V0 must be complete and tested before V1. V0 is
intentionally slow and obvious — its job is correctness baseline + worst case.

## Build / test / benchmark commands

All commands run from the repo root with the `gpu` venv activated.

```powershell
# Toolchain validation (run after `gpu`, first thing each session)
python env/check_env.py

# Install / refresh pinned deps
pip install -r env/requirements.txt

# Build a kernel extension (editable, idempotent, AOT)
pip install -e kernels/v0_naive_fp32

# Run all tests
pytest

# Run V0 correctness tests only
pytest tests/test_correctness.py

# Run V0 benchmark sweep
python bench/run_v0.py

# Generate the V0 figure
python analysis/plot_v0_initial.py
```

## Repository layout

- [env/check_env.py](env/check_env.py) — toolchain validation; run first thing each session.
- [env/requirements.txt](env/requirements.txt) — pinned deps; install inside the `gpu` venv.
- `kernels/v{0..4}_<name>/` — one directory per variant. Each has its own `setup.py`
  using `torch.utils.cpp_extension.CUDAExtension` for AOT build. V0 is implemented;
  V1–V4 are skeletons (compilable stubs that raise at runtime) so future sessions
  skip the boilerplate. Variant directories: `v0_naive_fp32`, `v1_tiled_fp16`,
  `v2_flash_fp16`, `v3_flash_fp8`, `v4_flash_nvfp4`.
- [bench/harness.py](bench/harness.py) — core benchmarking machinery (CUDA event timing, p50/p95/p99,
  forward-compatible Parquet schema all variants share).
- [bench/run_v0.py](bench/run_v0.py) — V0 paper-experiment driver (V0-PT, V0-CU, SDPA over
  the V0 grid). Snapshot, kept for reproducibility of the V0 figure.
- [bench/run_all.py](bench/run_all.py) — long-lived master comparison driver. Resolves
  the variant registry at startup and gracefully skips variants whose extension
  isn't built yet. Use this from V1 onwards.
- `bench/configs/` — YAML sweep configs.
- `bench/results/` — Parquet output (gitignored except `.gitkeep`).
- [tests/test_correctness.py](tests/test_correctness.py) — every kernel ships with a passing correctness test
  against the PyTorch reference. **A kernel without a passing test does not get committed.**
- `analysis/` — `.py` scripts for paper figures; `notebooks/` is for ad-hoc exploration only.
- `paper/` — LaTeX, set up later.

## Coding conventions

- **Python:** type hints on all public functions (`from __future__ import annotations`),
  short docstrings, no print-debugging (use `logging` or `rich`), pinned deps,
  ruff + black formatting (config in [pyproject.toml](pyproject.toml)).
- **CUDA:** kernel filenames `<variant>_<purpose>.cu`. Each `setup.py` always targets
  sm_120 explicitly via `'-gencode=arch=compute_120,code=sm_120'` rather than relying on
  `TORCH_CUDA_ARCH_LIST`. Builds must fail loudly if the wrong arch slips in.
- **Reproducibility:** every benchmark output records driver version, torch version,
  CUDA version, GPU model, timestamp, and git commit hash.
- **Determinism where possible:** fix random seeds for input tensors. Document where
  determinism cannot be guaranteed (reduction order, atomic adds).

## Known risks

- **Consumer Blackwell tooling immaturity (2026):** CUTLASS / Transformer Engine /
  FlashInfer support for sm_120 is still landing. Toolchain validation is the highest-
  priority de-risking step every session — do not skip it.
- **V4 (NVFP4) may degrade to a case-study** if NVFP4 tooling is too immature to run
  end-to-end attention. Acceptable for the paper; degrade rather than block.
- **Windows + CUDA extension builds** can be brittle; always surface exact `nvcc` errors
  rather than silently retrying.

## Deferred work (do NOT do until the relevant variant session)

- V1, V2, V3, V4 implementations.
- V0 optimization. V0 is intentionally slow and obvious.
- CI / GitHub Actions.
- CUTLASS, Transformer Engine, FlashInfer installs.
- Paper text. `paper/` stays empty until V3+.

## Reference papers (cite, don't re-read each session)

- Dao et al., **FlashAttention** — arXiv 2205.14135 (basis for V2).
- Dao, **FlashAttention-2** — arXiv 2307.08691 (work partitioning).
- Shah et al., **FlashAttention-3** — arXiv 2407.08608 (Hopper-specific; partial transferability to sm_120).
- Micikevicius et al., **FP8 Formats for Deep Learning** — arXiv 2209.05433 (basis for V3).
- Zhang et al., **SageAttention3** — arXiv 2505.11594 (FP4 attention on RTX 5090; closest to V4).
- NVIDIA, **RTX Blackwell Architecture Whitepaper**.

## Lessons from V0 session

Hindsight from the V0 session, captured because none of these were obvious upfront
and every one cost real time. Read this list before starting V1 — it answers most of
the "why is the build doing X?" questions you would otherwise hit again.

- **Toolchain layering on Windows is a stack of separate installs.** PyTorch's
  `cu128` wheel ships the CUDA *runtime* but not `nvcc`, headers, or a host C++
  compiler. A working extension build needs three independent installs: (1) CUDA
  Toolkit 12.8 (`winget install --id Nvidia.CUDA --version 12.8`), (2) Visual
  Studio Build Tools 2022 with the "Desktop development with C++" workload
  (`winget install --id Microsoft.VisualStudio.2022.BuildTools --override
  "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`),
  (3) `pip install ninja`. The first two need admin (UAC); ninja goes in the venv.
  The CUDA installer auto-adds `bin/` and `libnvvp/` to system PATH and sets
  `CUDA_PATH`, but does NOT set `CUDA_HOME` (PyTorch checks `CUDA_HOME` first) and
  does NOT add Nsight Compute to PATH (separate install dir under `Program Files\
  NVIDIA Corporation\Nsight Compute <ver>`). Mirror those in the `gpu` shell
  function.
- **MSVC 14.40+ collides with torch 2.9's `compiled_autograd.h`** with
  `error C2872: 'std': ambiguous symbol` at line 1134's `if-constexpr` cascade.
  PyTorch PR #144707 already added a guard, but it triggers only when `USE_CUDA`
  is defined — and extension builds don't define that by default. Fix: add
  `-DUSE_CUDA` to BOTH cxx and nvcc flags. The runtime stub the guard installs is
  inert because we never call `torch.compile` from custom kernels.
- **`pip install -e` of a torch CUDA extension on Windows requires two flags
  together**: `--no-build-isolation` (so the build env sees the venv's torch
  instead of `ModuleNotFoundError: No module named 'torch'`) and
  `DISTUTILS_USE_SDK=1` env var (so distutils doesn't try to re-activate VC env
  on top of an already-sourced vcvarsall, which fails with "VC env activated but
  DISTUTILS_USE_SDK is not set"). Either alone is insufficient.
- **Use `/O2` (MSVC syntax), not `-O3` (GCC syntax), as the cxx host flag on
  Windows.** cl.exe is lenient — it warns D9002 and ignores unknown flags — so
  the bug is silent: optimization quietly drops to MSVC's default. Same applies
  to anything you put in `extra_cflags` / `extra_compile_args["cxx"]`. Gate by
  `sys.platform == "win32"`.
- **`AT_CUDA_CHECK` vs `C10_CUDA_CHECK`.** `AT_CUDA_CHECK` lives in
  `<ATen/cuda/Exceptions.h>`, which `<torch/extension.h>` does NOT pull in
  transitively on Windows. Use `C10_CUDA_CHECK` from `<c10/cuda/CUDAException.h>`
  for `cudaGetLastError()` checks at kernel launch sites — same semantics,
  reliably available.
- **Shared-memory block reductions: `__syncthreads()` between phases is not
  optional even when it "looks safe."** If you read the reduction result into a
  per-thread register (e.g. `float row_max = reduce[0];`) and then later overwrite
  the same buffer (`reduce[tid] = local_sum;`) for a second reduction, you must
  `__syncthreads()` between the read and the write. Otherwise fast threads (e.g.
  thread 0) finish the second-phase compute and overwrite `reduce[0]` before slow
  threads (in another warp) have read it. Symptom in V0: nondeterministic ~6%
  mismatch rate with all errors concentrated on exactly one row per launch, only
  visible when total_rows is large enough for warp-level divergence to matter.
  Pattern: read → `__syncthreads()` → write.
- **`--use_fast_math` did not measurably hurt FP32 accuracy in V0.** Output diverged
  from the PyTorch reference by < 1e-6 across the full 48-config test grid (atol=
  1e-5 passed). Keep the flag for V1+ unless a precision study says otherwise.
- **Make `env/check_env.py` the canary, not the V0 build.** The dummy JIT kernel
  in `check_env.py` discovered, in order, every layer of toolchain breakage:
  missing `ninja`, missing `cl.exe`, missing `nvcc` / `CUDA_HOME`, the C2872
  MSVC + torch ABI clash. If the first build attempt had been V0-CU directly,
  every error would have surfaced tangled with kernel-specific noise. Keep
  `check_env.py` in sync with whatever build flags V1+ uses.
- **The `gpu` PowerShell function is conda-based** (env name "gpu" at
  `C:\Users\jeustache\anaconda3\envs\gpu`), not a venv. The user's profile lives
  at `C:\Users\jeustache\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1`.
  The function activates the conda env but does not currently source vcvarsall,
  set `CUDA_HOME`, or add Nsight Compute to PATH — that snippet was provided
  separately and the user will apply it before the next session.

## Current status

**Session 1 — V0 complete + V1–V4 scaffolding in place.** Toolchain end-to-end
green: VS Build Tools 2022 (MSVC 14.44) + CUDA Toolkit 12.8 + Nsight Compute
2025.1.1 installed and on PATH; `env/check_env.py` reports 10/10 PASS; V0-CU kernel
(`kernels/v0_naive_fp32/`) builds for sm_120 via `pip install -e`; 147/147 pytest
tests pass (V0-CU vs V0-PT atol=1e-5, both vs `F.scaled_dot_product_attention`
atol=1e-4); first sweep produced 2400 rows in `bench/results/v0_initial.parquet`
(24 (variant, config) results × 100 iterations) and the precursor figure at
`analysis/figures/v0_initial.png`.

V1–V4 directories exist as compilable stubs (`setup.py` + `binding.cpp` +
`attention_v{N}.cu` + `__init__.py` each). Their `setup.py` already targets
sm_120 + applies `-DUSE_CUDA` on Windows + uses `/O2` for cxx; the `.cu` body
is a `TORCH_CHECK(false, "not yet implemented")` stub. Future sessions can
`pip install -e kernels/v{N}_<name>` immediately to smoke-test the build pipeline,
then fill in the kernel body. `bench/run_all.py` resolves the variant registry at
startup and gracefully skips un-built variants — runs cleanly today with V0 + SDPA.

**Next session — V1.** FP16 tiled with WMMA / CUTLASS, still materializing outputs.
First action of the V1 session: re-run `python env/check_env.py` to confirm the
toolchain is still healthy, then `pip install -e kernels/v1_tiled_fp16` to verify
the stub builds, then decide between hand-rolled WMMA and CUTLASS templates for the
two matmuls and start there. CUTLASS install is also deferred to that session.

Update this section at the end of every session.
