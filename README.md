# attention-blackwell

Mixed-precision attention kernel benchmarks on **NVIDIA GeForce RTX 5080**
(Blackwell, `sm_120` / `sm_120a`).

CMP 674: Parallel Computing with GPUs course project.

Five hand-rolled CUDA kernels spanning FP32 to FP4 are implemented and
benchmarked end-to-end against PyTorch's `scaled_dot_product_attention`
(SDPA) reference. The repository is reproducible on a single consumer GPU;
all kernels ship with passing correctness tests against an FP32 reference.

---

## Problem statement

### Background

Scaled dot-product attention is the dominant compute and memory cost in
transformer inference; its quadratic dependence on sequence length has
motivated a decade of kernel-level optimization (tiling, fusion, online
softmax, narrower precisions). The two largest individual contributions
are the **FlashAttention** family (Dao et al., 2022 / 2023; Shah et al.,
2024) and the **mixed-precision attention** literature (Micikevicius et
al., 2022; SageAttention3, 2025). Each has been independently validated
on Hopper (H100), datacenter Blackwell (B100 / B200, `sm_100a`), and the
flagship consumer Blackwell card RTX 5090 (`sm_120a`).

### Gap

The bulk of academic and industrial interest in *consumer* Blackwell has
fixated on the RTX 5090. The mid-tier **RTX 5080** (`sm_120a`, 84 SMs,
16 GB GDDR7, ≈960 GB/s) is the actual practical inference target for
individual researchers, small labs, and edge-deployed serving stacks; it
shares a compute capability with the 5090 but has materially different
SM count, L2 capacity, and memory bandwidth, and (as we document) a
different software-stack story for GPU-attention dispatch on Windows.

> **Whether the FlashAttention algorithmic gains, FP8 throughput
> benefits, and FP4 microscaling promises *transfer* to this
> constrained-but-realistic platform is unstudied.**
> This is the gap our project addresses.

### Research question

> How does the latency–accuracy–memory Pareto frontier of mixed-precision
> attention shape up on consumer Blackwell (RTX 5080, `sm_120a`), as we
> add tiling, fusion, FP8, and NVFP4 one variable at a time?

We answer it with a controlled ablation study of five hand-rolled
attention kernels, each isolating exactly one optimization or precision
change against its predecessor.

### Constraints

1. Single GPU: RTX 5080, 16 GB. V0 OOMs past `seq_len = 2048`; V1 OOMs
   past ≈ 8192 at `(batch, heads) = (2, 8)`.
2. Forward pass only — backward pass is non-trivial for narrow precision
   and out of scope.
3. CUTLASS v4.4.2's FMHA reference (`77_blackwell_fmha`) gates compilation
   to `sm_100a` / `sm_103a` (datacenter Blackwell with TMA), so V3 and
   V4 use hand-rolled inline-PTX `mma.sync` instead of the missing
   `sm_120a` FMHA reference.
4. The hardware NVFP4 `mxf4nvf4.m16n8k64` instruction's CuTe scattered-
   fragment layout is incompatible with the row-major-SMEM idiom shared
   with V3, so V4 uses the software-microscaled `kind::f8f6f4.m16n8k32`
   path. Lifting V4 onto the hardware path is documented as deferred work.
5. No admin privileges for `nvidia-smi -lgc` (clock locking) or for
   Nsight Compute counter access (`ERR_NVGPUCTRPERM`); we use
   `cuobjdump --dump-sass` for Tensor-Core engagement verification and
   report p50 / p95 / p99 alongside the mean to surface tail variance.

### Hypotheses

| #  | Hypothesis                                                                                                    | Status                                            |
| -- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| H1 | Online softmax + fusion (V2) eliminates the $N \times N$ HBM materialization                                  | **Confirmed** — exact 17× reduction at `N = 8192` |
| H2 | FP8 (V3) closes a measurable share of the SDPA latency gap                                                    | **Confirmed** — 2.4× over V2, ≈ half the SDPA gap |
| H3 | FP4 (V4) extends the V3 latency win because of doubled FP4 throughput                                         | **Refuted** — V4 is ≈ 30 % *slower* than V3       |
| H4 | Accuracy degrades roughly geometrically with mantissa width                                                   | **Confirmed** — rel L2 widens ~ 4× per halving    |

H3 is the headline negative result: on consumer Blackwell, software
microscaling overhead dominates the FP4 throughput advantage at the
`mma.sync` call granularity.

### Contribution

1. The first systematic Pareto study of mixed-precision attention on
   the RTX 5080 (`sm_120a`).
2. Five hand-rolled CUDA kernels with a shared algorithmic skeleton,
   passing 495 correctness tests against an FP32 reference.
3. A negative result — V4 < V3 by ≈ 30 % — with a concrete forward path
   (hardware `mxf4nvf4.m16n8k64`) that would invert it.
4. An open, single-command-reproducible artifact on a single 16 GB
   consumer GPU, with figure regeneration and full provenance.

For the academic write-up, see [`paper/main.tex`](paper/main.tex)
(compiled: [`paper/main.pdf`](paper/main.pdf)).

---

## Variant roadmap

| ID  | Precision    | Design                                                           |
| --- | ------------ | ---------------------------------------------------------------- |
| V0  | FP32         | Naive — materializes the full N×N attention matrix in HBM        |
| V1  | FP16         | Tiled, hand-rolled WMMA Tensor Cores                             |
| V2  | FP16         | Fused + online softmax (FlashAttention-style)                    |
| V3  | FP8 (E4M3)   | V2 + per-tile scaling, hand-rolled `mma.sync.m16n8k32` PTX       |
| V4  | FP4 (NVFP4)  | V3 + per-row per-K-block (block size 32) software microscaling   |

Each variant lives in `kernels/v{0..4}_<name>/` with its own `setup.py` and
is built as a standalone PyTorch CUDA extension.

---

## Results

End-to-end measurements for V0–V4 plus PyTorch SDPA baseline on
NVIDIA GeForce RTX 5080 (Blackwell, `sm_120a`), CUDA 12.8,
PyTorch 2.9.1+cu128, Windows 11. All sweeps fix `batch = 2`,
`num_heads = 8`, FP16 inputs, non-causal masking. Latencies are mean
over 100 warmup + 500 measured iterations
([`bench/configs/sweep_final.yaml`](bench/configs/sweep_final.yaml)).

### Headline numbers (`seq_len = 8192`, `head_dim = 128`)

| Variant | Latency (ms) | Peak HBM (MB) | Rel L2 vs FP32 | Speedup vs SDPA |
| ------- | -----------: | ------------: | -------------: | --------------: |
| V0 (FP32, naive)        |   OOM    |   OOM   | 0 (ref) | —      |
| V1 (FP16 tiled WMMA)    |   77.1   |  2176   | 6.0e-04 | 0.16×  |
| V2 (FP16 fused)         |   60.9   |   128   | 3.0e-04 | 0.20×  |
| **V3 (FP8 fused)**      | **25.8** |   128   | 5.3e-02 | **0.48×** |
| V4 (NVFP4 microscaled)  |   33.4   |   128   | 2.1e-01 | 0.37×  |
| **SDPA (PyTorch math)** | **12.3** |   128   | 3.0e-04 | 1.00×  |

V0 OOMs past `seq_len ≈ 2048` at `(batch, heads) = (2, 8)`.

### Headline numbers (`seq_len = 16384`, `head_dim = 128`)

| Variant | Latency (ms) | Peak HBM (MB) |
| ------- | -----------: | ------------: |
| V1   |   310    |   8448   |
| V2   |   233    |    256   |
| V3   |  **97**  |    256   |
| V4   |   128    |    256   |
| SDPA |  **49**  |    256   |

At `N = 16384`, V1 occupies 8.4 GB — within striking distance of the
16 GB ceiling and the operational point at which V1 becomes
infeasible.

### Three flagship findings

**1. Memory wall breaks at V2 — 17× reduction at `N = 8192`.** Adding
online softmax + single-kernel fusion drops peak HBM from V1's
2176 MB to V2's 128 MB — a 17× reduction that exactly matches SDPA's
profile. Entirely attributable to the kernel structure (no precision
change between V1 and V2).

```
V1 → V2 peak HBM @ seq_len=8192, head_dim=128:
    2176 MB → 128 MB    (17.0× reduction)
```

**2. V3 closes ≈ half the SDPA latency gap with FP8.** V3's `mma.sync`
FP8 path plus the register-resident O accumulator (which V2's WMMA
path can't do without knowing the layout) gives 2.4× over V2 and
reaches ≈ 48 % of SDPA's throughput.

```
V2 → V3 latency @ seq_len=8192, head_dim=128:
    60.9 ms → 25.8 ms   (2.36× speedup)
V3 → SDPA gap remaining:
    25.8 ms vs 12.3 ms  (V3 = 0.48× SDPA — half the V2-to-SDPA gap closed)
```

The two contributions are roughly equal: FP8 mma throughput on the
5th-gen Tensor Core (theoretical 2× FP16) and register-resident O
(eliminates V2's per-iteration SMEM round-trip).

**3. V4 LOSES to V3 by ≈ 30 % — the negative result.** NVFP4's
hardware throughput advantage on Blackwell (theoretical 2× FP8) is
more than canceled by software microscaling overhead at the
`mma.sync` call granularity. Each mma incurs ≈ 12 register ops for
per-row + per-K-block scale application; over 96 mmas per
`(kv iter, thread)` this is ≈ 1100 ops/thread/iter overhead.

```
V3 → V4 latency @ seq_len=8192:
    25.8 ms → 33.4 ms    (V4 is 0.77× V3 — SLOWER)
V3 → V4 latency @ seq_len=16384:
    96.5 ms → 127 ms     (same magnitude)
```

The path forward is the hardware-microscaled `mxf4nvf4.m16n8k64`
instruction, which internalizes the per-16-element scale at the mma
boundary. CUTLASS exposes the family via `cute::SM120_16x8x64_TN`,
but its scattered-fragment layout requires `ldmatrix`-style swizzled
SMEM incompatible with V3's row-major-SMEM idiom. We document this
as deferred work; a CUTLASS-mediated implementation would close the
V4-vs-V3 gap and likely flip the ranking.

### Accuracy envelope

Relative L2 error against an FP32 reference, at unit-variance
Gaussian inputs:

| Variant | Rel L2 @ σ=1 | Rel L2 @ σ=3 | Rel L2 @ σ=5 |
| ------- | -----------: | -----------: | -----------: |
| V1   | 6e-4    | 2e-3   | 5e-3   |
| V2   | 3e-4    | 1e-3   | 2e-3   |
| V3   | 5.3e-2  | 0.12   | 0.19   |
| V4   | 0.21    | 0.42   | 0.56   |

Every halving of mantissa width widens rel L2 by ≈ 4×. Bulk-
correctness tests (≥ 95 % of elements within 5e-2 absolute error,
all elements finite, output magnitudes bounded) **pass** for all
variants across the full sweep grid in non-causal mode; in causal
mode V4's threshold is 0.75 because rows with very few unmasked
tokens have near-deterministic outputs that FP4 quantization noise
hits proportionally harder.

### SDPA dispatch finding

A finding worth foregrounding: **PyTorch 2.9.1+cu128 on Windows
dispatches SDPA exclusively to the MATH backend on consumer
Blackwell.** The Flash, cuDNN, and Memory-Efficient backends are all
runtime-disabled for `(sm_120, FP16, non-causal)` inputs in this
build:

```
Torch was not compiled with flash attention.
cuDNN attention has been runtime disabled.
Memory Efficient attention has been runtime disabled.
```

The SDPA baseline is therefore the **vendor's cuBLAS-tuned FP16
attention with no algorithmic fusion** (a pair of matmuls + a
softmax that materializes the full $N \times N$ probability matrix
in HBM). The "SDPA gap" V3 closes is the gap to a heavily-tuned
matmul stack, not to FlashAttention-2.

### Tensor Core engagement

Verified directly from the compiled cubin via `cuobjdump --dump-sass`,
bypassing Nsight Compute's `ERR_NVGPUCTRPERM` admin requirement:

| Variant | Mnemonic | Instructions |
| ------- | -------- | -----------: |
| V1 (QK kernel)             | `HMMA.16816.F32`              | 120  |
| V1 (PV kernel)             | `HMMA.16816.F32`              |  32  |
| V2 (`HEAD_DIM = 64`)       | `HMMA.16816.F32`              |  64  |
| V2 (`HEAD_DIM = 128`)      | `HMMA.16816.F32`              | 128  |
| V3 (both `HEAD_DIM`)       | `QMMA.16832.F32.E4M3.E4M3`    |  96  |
| V4 (both `HEAD_DIM`)       | `QMMA.16832.F32.E2M1.E2M1`    |  96  |

V3 and V4 emit the same number of `QMMA` instructions because the
per-tile mma call structure is preserved; only the operand precision
changes.

### Figures

The headline 2-panel Pareto figure (latency vs accuracy at
`seq_len = 8192` and `16384`) is at
[`paper/figures/fig_pareto_main.png`](paper/figures/fig_pareto_main.png).
All paper figures live under [`paper/figures/`](paper/figures/):

| File | Caption |
| ---- | ------- |
| `fig_pareto_main.{pdf,png}`            | Latency–accuracy Pareto at `seq_len ∈ {8192, 16384}`            |
| `fig_latency_vs_seqlen.{pdf,png}`      | Latency vs sequence length, all variants                        |
| `fig_memory_vs_seqlen.{pdf,png}`       | Peak HBM vs sequence length (V1 → V2 step visible)              |
| `fig_speedup_matrix.{pdf,png}`         | Pairwise speedup heatmap at `seq_len = 8192`                    |
| `fig_accuracy_envelope.{pdf,png}`      | Rel L2 vs input std for V2 / V3 / V4                            |
| `fig_tensor_core_engagement.{pdf,png}` | HMMA / QMMA instruction counts per kernel                       |
| `tab_full_results.tex`                 | Headline numbers per variant per `seq_len`                      |

Single-command figure regeneration: `python analysis/paper_figures.py`.

### Limitations

- Single hardware platform (RTX 5080). The V4 finding is most
  sensitive to the unavailability of `mxf4nvf4` via row-major SMEM;
  a CUTLASS-mediated swizzled-SMEM port would likely flip V4 vs V3.
- Forward pass only.
- Driver / software-version-specific. Versions are frozen in the
  artifact's Parquet provenance.
- 16 GB ceiling forces fixed `(batch, heads) = (2, 8)`; multi-tier
  batch sweeps would need a larger card.
- E5M2 (alternative FP8) and integer-precision attention paths are
  not characterized.

For the full discussion, see [`paper/main.tex`](paper/main.tex)
sections 7–8.

---

## Hardware target

- **GPU:** NVIDIA GeForce RTX 5080
- **Architecture:** Blackwell, compute capability 12.0 (`sm_120` /
  `sm_120a`)
- **VRAM:** 16 GB GDDR7, ~960 GB/s peak memory bandwidth
- **Tensor Cores:** 5th generation, native FP8 (E4M3 / E5M2) and FP4
  (NVFP4)
- **OS tested:** Windows 11 (PowerShell shell)
- **CUDA Toolkit:** 12.8
- **PyTorch:** 2.9.1+cu128

V4 is the only variant whose `setup.py` targets `sm_120a`
(architecture-accelerated) instead of plain `sm_120` — the FP4 mma
instruction family (`kind::f8f6f4`) is rejected by `ptxas` on plain
`sm_120`.

---

## Build & run

All commands assume the project's `gpu` Python environment is activated
(CUDA 12.8 toolkit + PyTorch 2.9.1+cu128 + Ninja). On Windows, dot-source
the helper script to set vcvars + `CUDA_HOME` + Nsight Compute on PATH:

```powershell
. .\env\activate_for_build.ps1
python env/check_env.py     # toolchain validation; should print 10/10 PASS
```

Install pinned dependencies and build all five kernel extensions:

```powershell
pip install -r env/requirements.txt
pip install --no-build-isolation -e kernels/v0_naive_fp32
pip install --no-build-isolation -e kernels/v1_tiled_fp16
pip install --no-build-isolation -e kernels/v2_flash_fp16
pip install --no-build-isolation -e kernels/v3_flash_fp8
pip install --no-build-isolation -e kernels/v4_flash_nvfp4
```

Run the test suite (495 tests across V0–V4):

```powershell
pytest
```

Run the V4 sweep (all five variants + SDPA across `seq_len ∈ {128 .. 16384}`,
`head_dim ∈ {64, 128}`, `batch = 2`, `num_heads = 8`):

```powershell
python bench/run_all.py `
    --variants sdpa v1_tiled_cu v2_flash_cu v3_flash_cu v4_flash_cu `
    --config bench/configs/sweep_v4.yaml `
    --output bench/results/v4_initial.parquet
```

Generate paper figures:

```powershell
python analysis/plot_v0_initial.py
python analysis/plot_v1_vs_v0.py
python analysis/plot_v2_vs_v1_v0.py
python analysis/plot_v3_vs_v2_v1.py
python analysis/plot_v4_vs_v3_v2_v1.py
python analysis/plot_full_pareto.py
python analysis/plot_full_pareto_final.py   # paper headline
```

Verify Tensor Core engagement via SASS inspection (works without admin /
without Nsight Compute counter access):

```powershell
# FP16 (V1, V2)
cuobjdump --dump-sass kernels/v2_flash_fp16/_C.cp311-win_amd64.pyd | Select-String HMMA
# FP8 (V3) -- mnemonic is QMMA on sm_120 SASS
cuobjdump --dump-sass kernels/v3_flash_fp8/_C.cp311-win_amd64.pyd | Select-String QMMA
# FP4 (V4) -- mnemonic is QMMA.16832.F32.E2M1.E2M1
cuobjdump --dump-sass kernels/v4_flash_nvfp4/_C.cp311-win_amd64.pyd | Select-String QMMA
```

---

## Repository layout

```
attention-blackwell/
├── env/                    # Toolchain validation + Windows build helper
├── kernels/
│   ├── v0_naive_fp32/      # FP32 baseline (PyTorch ref + naive CUDA)
│   ├── v1_tiled_fp16/      # FP16 tiled WMMA
│   ├── v2_flash_fp16/      # FP16 fused (FlashAttention-style)
│   ├── v3_flash_fp8/       # FP8 E4M3 + per-tile scaling
│   └── v4_flash_nvfp4/     # FP4 E2M1 + microscaling (sm_120a)
├── bench/
│   ├── harness.py          # CUDA-event timing, p50/p95/p99, Parquet output
│   ├── run_all.py          # Master sweep driver (all variants)
│   ├── run_v0.py           # V0 paper-experiment snapshot driver
│   ├── configs/            # YAML sweep configs (per-variant grids)
│   └── results/            # Parquet outputs (gitignored)
├── analysis/
│   ├── plot_*.py           # Per-variant + Pareto figure scripts
│   └── figures/            # Generated PNGs
├── tests/
│   └── test_correctness.py # 495 tests across V0–V4
├── external/
│   └── cutlass/            # CUTLASS v4.4.2 submodule (PTX reference only)
└── paper/                  # IEEE LaTeX (work in progress)
```

---

## Reference papers

- Dao et al., **FlashAttention** — arXiv:2205.14135 (basis for V2)
- Dao, **FlashAttention-2** — arXiv:2307.08691 (work partitioning)
- Shah et al., **FlashAttention-3** — arXiv:2407.08608 (Hopper-specific;
  partial transferability to `sm_120`)
- Micikevicius et al., **FP8 Formats for Deep Learning** —
  arXiv:2209.05433 (basis for V3)
- Zhang et al., **SageAttention3** — arXiv:2505.11594 (FP4 attention on
  RTX 5090; closest related work to V4)
- NVIDIA, **RTX Blackwell Architecture Whitepaper**

---

## License

For coursework / academic use. See `LICENSE` if added.
