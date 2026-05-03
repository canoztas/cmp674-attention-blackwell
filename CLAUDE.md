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
  using `torch.utils.cpp_extension.CUDAExtension` for AOT build.
- `bench/harness.py` — core benchmarking machinery (CUDA event timing, p50/p95/p99,
  Parquet output with a future-proof schema all variants share).
- `bench/configs/` — YAML sweep configs.
- `bench/results/` — Parquet output (gitignored except `.gitkeep`).
- `tests/test_correctness.py` — every kernel ships with a passing correctness test
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

## Current status

**Session 1 — in progress.** Repo skeleton in place. Toolchain validation
(`env/check_env.py`) written and awaiting first run. After it passes cleanly, the next
session steps are: benchmark harness, V0-PT reference, V0-CU custom kernel, correctness
tests, first benchmark sweep + figure.

Update this section at the end of every session.
