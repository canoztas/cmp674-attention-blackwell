# attention-blackwell

Mixed-precision attention kernel benchmarks on **NVIDIA GeForce RTX 5080**
(Blackwell, `sm_120` / `sm_120a`).

CMP 674: Parallel Computing with GPUs course project.

Five hand-rolled CUDA kernels spanning FP32 to FP4 are implemented and
benchmarked end-to-end against PyTorch's `scaled_dot_product_attention`
(SDPA) reference. The repository is reproducible on a single consumer GPU;
all kernels ship with passing correctness tests against an FP32 reference.

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

## Headline results

At `seq_len = 8192`, `head_dim = 128`, `batch = 2`, `num_heads = 8`:

| Variant | Latency (ms) | Peak Mem (MB) | Rel L2 vs FP32 |
| ------- | -----------: | ------------: | -------------: |
| V1      |         77.1 |          2176 |        6.0e-04 |
| V2      |         60.9 |           128 |        3.0e-04 |
| V3      |     **25.8** |           128 |        5.3e-02 |
| V4      |         33.4 |           128 |        2.1e-01 |
| SDPA    |     **12.3** |           128 |        3.0e-04 |

Key findings:

- **V1 → V2** is the largest memory win in the sweep: a 17× reduction at
  `seq_len = 8192`. FlashAttention-style online softmax + kernel fusion
  drops the peak from 2.2 GB to 128 MB.
- **V2 → V3** is the largest latency win: 2.4× at `seq_len = 8192`,
  rising to 2.4× at `seq_len = 16384`. FP8 throughput on Blackwell 5th-gen
  Tensor Cores plus a register-resident O accumulator.
- **V3 → V4** is *negative*: V4 is 1.3× slower than V3. The hardware FP4
  throughput advantage (theoretical 2× over FP8) is more than canceled by
  software per-row per-K-block microscaling overhead. With the hardware
  block-scaled `mxf4nvf4.m16n8k64` instruction the calculus would flip,
  but its scattered per-thread fragment layout is non-trivial to hand-roll
  on row-major SMEM and is left as future work.
- The accuracy/precision Pareto: every halving of mantissa width widens
  relative L2 error by ~4×: V2 = 3e-4, V3 = 5e-2, V4 = 0.21.

The headline 2-panel Pareto figure (latency vs accuracy at
`seq_len = 8192` and `16384`) is at
[`analysis/figures/full_pareto_final.png`](analysis/figures/full_pareto_final.png).

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
