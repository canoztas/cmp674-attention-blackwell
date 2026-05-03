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

| ID  | Precision   | Design                                                                | Status      |
| --- | ----------- | --------------------------------------------------------------------- | ----------- |
| V0  | FP32        | Naive — materializes the full N×N attention matrix in HBM             | **Done**    |
| V1  | FP16        | Tiled, Tensor Core (hand-rolled WMMA), still materializes outputs     | **Done**    |
| V2  | FP16        | Fused + online softmax (FlashAttention-style)                         | **Done**    |
| V3  | FP8 (E4M3)  | V2 + per-tile scaling                                                 | Pending     |
| V4  | FP4 (NVFP4) | V3 + microscaling (Blackwell-only, exploratory)                       | Pending     |

**One variant per session.** V0 was the correctness baseline + worst case.
V1 isolates the contribution of *tiling + Tensor Cores* before V2 adds
fusion. V1 went WMMA over CUTLASS for didactic clarity and zero install
risk; V2 may revisit that decision (CUTLASS's `CollectiveEpilogue` is more
compelling for fused kernels).

## Build / test / benchmark commands

All commands run from the repo root with the `gpu` venv activated. Interactive
shells use the user's `gpu` PowerShell function; **non-interactive shells**
(automation, IDE-spawned terminals) instead dot-source the project's helper:

```powershell
. .\env\activate_for_build.ps1   # vcvars + CUDA + ncu + conda gpu env
```

Both achieve the same result; `gpu` and `activate_for_build.ps1` are
interchangeable for everything below.

```powershell
# Toolchain validation (first thing each session)
python env/check_env.py

# Install / refresh pinned deps
pip install -r env/requirements.txt

# Build kernel extensions (editable, idempotent, AOT)
pip install --no-build-isolation -e kernels/v0_naive_fp32
pip install --no-build-isolation -e kernels/v1_tiled_fp16
pip install --no-build-isolation -e kernels/v2_flash_fp16

# Run all tests (310 expected)
pytest

# Run a single variant's correctness tests
pytest tests/test_correctness.py -k v2

# Run V0 paper snapshot (FP32 sweep, V0-PT + V0-CU + SDPA)
python bench/run_v0.py

# Run V1 sweep (FP16, V1-CU + SDPA)
python bench/run_all.py --variants sdpa v1_tiled_cu `
    --config bench/configs/sweep_v1.yaml `
    --output bench/results/v1_initial.parquet

# Run V2 sweep (FP16, V1-CU + V2-CU + SDPA, extended to seq_len=8192)
python bench/run_all.py --variants sdpa v1_tiled_cu v2_flash_cu `
    --config bench/configs/sweep_v2.yaml `
    --output bench/results/v2_initial.parquet

# Generate figures
python analysis/plot_v0_initial.py
python analysis/plot_v1_vs_v0.py
python analysis/plot_v2_vs_v1_v0.py

# Verify Tensor Core engagement (no admin needed; reads cubin SASS)
cuobjdump --dump-sass kernels/v2_flash_fp16/_C.cp311-win_amd64.pyd | Select-String HMMA
```

## Repository layout

- [env/check_env.py](env/check_env.py) — toolchain validation; run first thing each session.
- [env/requirements.txt](env/requirements.txt) — pinned deps; install inside the `gpu` venv.
- [env/activate_for_build.ps1](env/activate_for_build.ps1) — non-interactive PowerShell
  helper: vcvars64.bat + `CUDA_HOME` + Nsight Compute on PATH + conda `gpu` activation.
  Dot-source it in shells where the user's `gpu` profile function isn't available.
- `kernels/v{0..4}_<name>/` — one directory per variant. Each has its own `setup.py`
  using `torch.utils.cpp_extension.CUDAExtension` for AOT build. V0, V1, and V2 are
  implemented; V3–V4 are skeletons (compilable stubs that raise at runtime) so
  future sessions skip the boilerplate. Variant directories: `v0_naive_fp32`,
  `v1_tiled_fp16`, `v2_flash_fp16`, `v3_flash_fp8`, `v4_flash_nvfp4`.
- [bench/harness.py](bench/harness.py) — core benchmarking machinery (CUDA event timing, p50/p95/p99,
  forward-compatible Parquet schema all variants share).
- [bench/run_v0.py](bench/run_v0.py) — V0 paper-experiment driver (V0-PT, V0-CU, SDPA over
  the V0 grid). Snapshot, kept for reproducibility of the V0 figure.
- [bench/run_all.py](bench/run_all.py) — long-lived master comparison driver. Resolves
  the variant registry at startup and gracefully skips variants whose extension
  isn't built yet. Use this from V1 onwards.
- `bench/configs/` — YAML sweep configs. `sweep_default.yaml` is FP32 (V0);
  `sweep_v1.yaml` is the FP16 twin used by V1 (same shape grid, different dtype);
  `sweep_v2.yaml` extends to seq_len=8192 with smaller (B, H) for V2's long-seq story.
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

- V3, V4 implementations.
- V0 / V1 / V2 optimization. V0 is intentionally slow and obvious; V1 picks
  defensible tile defaults (BR=BC=BD=64); V2 keeps O accumulator in shared
  memory across iterations rather than register-resident — this is the main
  perf left on the table (see V2 lessons). Tile-shape tuning is V3+ work.
- Tensor Core engagement profiling via Nsight Compute. Blocked on
  consumer-driver `ERR_NVGPUCTRPERM`; cubin SASS check (`cuobjdump | grep HMMA`)
  is sufficient through V2 and doesn't need admin.
- Re-running V0 / V1 at the V2 sweep grid (seq_len up to 8192) for full Pareto
  curves. Current snapshots are at separate (B, H) for memory headroom.
- CI / GitHub Actions.
- CUTLASS, Transformer Engine, FlashInfer installs. Reconsider CUTLASS at
  V3 (FP8 epilogue scaling makes the install cost worthwhile; see V2 lessons).
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

## Lessons from V1 session

Captured because each cost real time and none are obvious from the code or git
history. Read before starting V2.

- **Non-interactive PowerShell can't use the `gpu` profile function.** The user's
  interactive `gpu` activates conda + sets paths, but profile functions don't load
  in non-interactive shells (the ones automation spawns). Solution: a project-level
  helper at [env/activate_for_build.ps1](env/activate_for_build.ps1) that sources
  vcvars64.bat, sets `CUDA_HOME` + `CUDA_PATH`, prepends Nsight Compute, and
  activates the conda `gpu` env. Dot-source it at the start of every PowerShell
  call: `. .\env\activate_for_build.ps1`. The interactive `gpu` function still works
  for terminal use; this file only matters for tooling.
- **PowerShell 5.1 reads `.ps1` files in the OS ANSI codepage by default**,
  not UTF-8. The `Write` tool emits UTF-8 without BOM. Non-ASCII characters
  (em-dashes, smart quotes, etc.) get reinterpreted mid-file and break the
  parser with confusing "string is missing the terminator" errors at lines
  *after* the bad character. Stick to ASCII in PowerShell scripts, or write
  with a UTF-8 BOM if non-ASCII is necessary.
- **PowerShell 5.1 wraps native-command stderr as `NativeCommandError`** with
  exit code 1 — even when the command exited 0. Python's `logging` module
  writes INFO-level messages to stderr, so `python script.py` looks like a
  failure under PowerShell's strict-mode handling. The script *did* run; the
  noise just masks success. If you need a clean exit code, run via
  `python -c "..."` with prints to stdout, or accept that `2>&1` and `Out-String`
  don't fix it (the wrapping happens in PowerShell's host, not the redirector).
  Don't waste time chasing "errors" that match this pattern.
- **PowerShell 5.1 can mis-parse `|` inside double-quoted argument strings**
  passed to native commands. ncu's `--kernel-name "regex:foo|bar"` was
  treating `bar` as a separate command. Workaround: avoid `|` in regex args
  to native tools, or run two invocations and concatenate.
- **Nsight Compute on consumer GPUs requires admin** for performance-counter
  access (`ERR_NVGPUCTRPERM`). On RTX 5080 + non-elevated shell this blocks
  ncu's per-kernel metrics. **Better proxy for "are Tensor Cores engaged?":
  `cuobjdump --dump-sass <pyd> | grep HMMA`.** This reads the actual emitted
  machine code — `HMMA.16816.F32` is the Blackwell Tensor Core instruction
  for FP16-in / FP32-acc. V1 emits 120 HMMA instructions in `qk_tile_kernel`
  and 32 in `pv_tile_kernel`, with zero in `softmax_fp16_kernel` (FP32 ALU,
  as expected). This is *more* rigorous than ncu's sampled metric — it's the
  cubin itself.
- **WMMA `matrix_b col_major` is the ergonomic way to do `Q @ K^T` without an
  explicit transpose pass.** K is row-major (BC, D). Declaring B as
  `matrix_b col_major` with `ldm = D` and pointer `K_smem + n*WMMA_N*D + k0`
  makes B equal to K^T because col-major stride pattern `ptr[k + n*ldm]`
  matches row-major K's element layout. No data movement, no extra kernel.
  This pattern transfers directly to V2.
- **CUTLASS-vs-WMMA: WMMA was the right call for V1** despite the kickoff's
  lean toward CUTLASS. Build-and-iterate cycle in seconds (CUTLASS would be
  multi-minute), zero install risk on a brand-new sm_120 toolchain, didactic
  clarity for the methodology section. CUTLASS's `CollectiveEpilogue` advantage
  doesn't materialize until V2's online softmax fusion. If V2 wants CUTLASS,
  switch then; V1's WMMA path becomes the "hand-rolled Tensor Core baseline"
  data point for the paper.
- **Per-precision sweep configs are inevitable.** V0 takes FP32, V1 takes
  FP16, the harness has one `dtype` per sweep YAML. Solution:
  [bench/configs/sweep_v1.yaml](bench/configs/sweep_v1.yaml) is the FP16
  twin of `sweep_default.yaml` with the same shape grid; the V1-vs-V0 plot
  reads both Parquets and joins on `(seq_len, head_dim)`. Each variant
  benchmarked at its native precision is the honest comparison for the paper.
- **FP16 cumulative rounding is the real signal at high input variance**, not
  a kernel bug. V1 stores S and P in FP16 between phases, which compounds
  ~`eps_fp16` per stored value across the PV reduction. Measured: at unit
  variance, max abs err ~2e-3 (well under the 1e-2 tolerance). At std=2 a
  single element drifts past 1e-2; at std=3 ~0.3% of elements; at std=5
  ~1.3%. None are bugs — they are V1's design envelope. The strict 1e-2
  tolerance is right for the main grid; high-variance stress tests should
  verify *finiteness + bulk-correctness* (no NaN, ≥95% within 5e-2), which
  rules out indexing/accumulator bugs without false-failing on FP16 noise
  V1 was never designed to suppress. V3's per-tile FP8 scaling is what
  fixes this regime.
- **Static `__shared__` arrays count toward the per-block 48KB cap on
  Blackwell.** Pulling all smem into a single dynamic allocation
  (`extern __shared__ unsigned char smem_raw[]; ...reinterpret_cast`)
  centralizes the budget and avoids surprise cap hits. V1's QK kernel at
  D=128 totals exactly 48KB (32KB FP16 tiles + 16KB FP32 scratch) — at the
  default cap with no headroom; if V2 grows the tile, opt-in via
  `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`.
- **Performance, measured.** V1 lands 4.5–10.9× over V0 across the grid
  (8.3–8.6× at seq_len=2048); SDPA is 3–9× faster than V1, with the gap
  widening at large seq_len because SDPA uses FlashAttention's online
  softmax + fused kernel — exactly what V2 introduces. V1 catching SDPA at
  this stage would have been a correctness-bug signal, not a victory.

## Lessons from V2 session

Captured before V3. Read these first; each one is a real-time-cost item that
the code alone won't tell you.

- **The CUTLASS-vs-WMMA decision held in WMMA's favor again, but for
  different reasons.** V1's lesson said "CUTLASS at V2 if CollectiveEpilogue
  pays off." It does in principle, but the install-and-verify cost on a
  brand-new sm_120 toolchain is multi-hour yak shaving in a session that's
  already algorithmically dense. V1's WMMA scaffolding (col_major K^T trick,
  FP16 binding, dynamic SMEM idiom, sm_120 build) lifted verbatim. **V3 is
  the natural CUTLASS migration point**: FP8 epilogue scaling (per-tile
  scale factors threaded through the matmul output stage) is what
  CollectiveEpilogue is actually for, and the install cost amortizes over
  V3 + V4. Don't spend it earlier.
- **Online softmax stability hinges on two corner cases that don't appear
  in the paper's algorithm box.** (1) First iteration: m_i = -inf and l_i = 0;
  m_i = 0 silently corrupts results when scores are negative (every
  realistic input). (2) When an entire tile of scores is masked out for a
  row (causal masking past the diagonal, or padded rows), m_new stays
  -inf and the naive `alpha = exp(m_old - m_new)` becomes NaN. Guard:
  `alpha = isfinite(m_new) ? __expf(m_old - m_new) : 1.0f;` and similarly
  in the P = exp() pass. Combined with the deferred normalization (divide
  by l only at the epilogue), these guards keep the recurrence robust;
  V2 passed the small-magnitude-input test on the first build, which the
  same kernel would have failed without the guards.
- **Storing the O accumulator in shared memory between iterations is the
  algorithmic correctness shortcut and the main perf compromise.** The
  per-iteration update `O = alpha * O + P @ V` requires multiplying every
  element of O by a per-row scalar alpha[r]. WMMA accumulator fragments
  have implementation-defined per-thread layouts, so doing the alpha
  rescale on register-resident fragments requires either knowing the layout
  (FlashAttention's manual MMA PTX path) or using a shfl-based broadcast
  scheme that depends on the layout. The shared-memory round trip
  (store frag → SMEM → rescale → load SMEM → frag) is slower but correct
  regardless of layout. Measured cost: V2 is only 1.0–1.5× V1 at long
  seq_len despite the algorithmic improvement; SDPA is 4–7× V2 because
  it does the register-resident update. **For V3, the register-resident
  path is mandatory** — FP8 + per-tile scaling adds another rescale per
  iteration that compounds the smem-traffic problem. CUTLASS handles this
  via CollectiveEpilogue.
- **The structural V2 win is memory, not latency.** At seq_len=8192,
  head_dim=128, V1 peaks at 2176 MB (the N×N S tensor in HBM), V2 peaks
  at 128 MB (matches SDPA exactly: just inputs + outputs). That's a 17×
  reduction, and it's what makes seq_len=8192 even runnable on a 16 GB
  card with non-trivial batch sizes. Lead with this in the paper; the
  modest latency speedup is a secondary story.
- **Dynamic SMEM opt-in is one line, but you need it.** V2's per-block
  SMEM is 96 KB at HEAD_DIM=128 (Q + O + K/P alias + V + S). The default
  cap is 48 KB; the launch silently fails without
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
  bytes)`. Cache the call with `static bool attr_set` so you don't re-pay
  the API call every launch. RTX 5080 sm_120 supports up to ~99 KB
  dynamic SMEM per block — fits 96 KB with a couple KB headroom. Don't
  push past this for V3 without verifying the limit empirically.
- **Templating by HEAD_DIM is worth a few extra lines.** V1 carries D as
  a runtime int and dispatches to identical code paths. V2 templates by
  HEAD_DIM ∈ {64, 128} so FRAG_BD = HEAD_DIM / WMMA_N is a compile-time
  constant. Two effects: (1) the WMMA accumulator fragment array is
  size-fixed (`fragment ... o_frag[FRAG_BD]`) which is required by C++
  for register-resident arrays, and (2) the SMEM offset arithmetic is
  fully constant-folded. Two instantiations cover the supported grid;
  if D=256 or D=192 ever lands, add another template arg.
- **Single fused kernel = fewer blocks, lower hardware utilization at
  small N.** V2's grid is `(ceil(N/BR), B*H)` with one block doing all
  K, V iterations sequentially. V1's QK grid is
  `(ceil(N/BR), ceil(N/BC), B*H)` — one block per output tile. At
  seq_len=512, head_dim=64, V1 wins because more blocks = better SM
  occupancy; at seq_len=2048+ V2 catches up because per-block work
  dominates. The crossover is around seq_len=1024 in our measurements.
  This is expected and matches the FlashAttention papers — don't chase
  it down as a bug.
- **The cubin HMMA count for templated kernels is per-instantiation.**
  cuobjdump emits the SASS for each instantiation separately. V2 emits
  64 HMMAs for `flash_attention_fp16_kernel<64>` and 128 for
  `<128>` — the count scales with the inner-D loop trip count
  (D/WMMA_K iterations per warp). Total 192. Verify both instantiations
  show HMMA, not just the count.
- **Don't fight the WMMA fragment layout.** In V1 the QK^T matmul exposed
  the col_major trick; in V2 the trickiest pattern is `load_matrix_sync`
  for the accumulator fragment (loading O from SMEM into the o_frag) so
  the next mma_sync accumulates onto the loaded values. This is supported
  for accumulator fragments at the same layout used by store_matrix_sync
  (mem_row_major in our case) — same layout in / out, no surprises.
- **PowerShell wrap-and-pipe gotcha (carried over from V1):** when you
  use `bash` to invoke `powershell -NoProfile -Command "...; ... | Select-...":`
  the bash layer interprets the `|` as a pipe before powershell sees it.
  Use the `PowerShell` tool directly when you need to pipe inside a
  PowerShell command — or, when invoking via `Bash`, wrap the entire
  thing including pipes in the `-Command` quoted argument. This bit twice
  in the V2 session before I switched to the `PowerShell` tool for
  inspection commands.

## Current status

**Session 3 — V0 + V1 + V2 implemented; V3 + V4 still stubs.** 310 pytest
tests pass: 147 V0 (unchanged) + 70 V1 (unchanged) + 92 V2 + 1 harness. V2-CU
(`kernels/v2_flash_fp16/`) builds for sm_120 via `pip install -e`; 192 HMMA
Tensor Core instructions across the two HEAD_DIM template instantiations
(64 for `<64>`, 128 for `<128>`).

V2 algorithmic state: online softmax with running (m, l) per row, single
fused kernel (Q@K^T → softmax → P@V) with no full N×N materialization. O
accumulator in SMEM (FP32) across iterations; running stats in static
`__shared__` arrays. Per-iteration update via store-frag → SMEM rescale by
alpha → load-frag-back, accumulate via mma_sync, store-frag again.
Templated by HEAD_DIM ∈ {64, 128}. Dynamic SMEM opt-in via
`cudaFuncSetAttribute` (96 KB at HEAD_DIM=128).

Sweep at `bench/results/v2_initial.parquet` (3600 rows: 36 (variant, config)
pairs × 100 iterations of V1 + V2 + SDPA on FP16 across seq_len ∈ {128…8192},
head_dim ∈ {64, 128}, b=2, h=8). Figure at `analysis/figures/v2_vs_v1_v0.png`.

Headline numbers at seq_len=8192, head_dim=128:
- Latency: V1 = 75.6 ms, V2 = 58.8 ms, SDPA = 12.5 ms (V2 is 1.29× V1; SDPA is 4.7× V2)
- Peak memory: V1 = 2176 MB, V2 = 128 MB, SDPA = 128 MB (V2 is 17× lower than V1)

The latency gap between V2 and SDPA is the cost of the SMEM-resident O
accumulator vs. the register-resident path SDPA / FlashAttention use.
That's the V3+ design lever (CUTLASS CollectiveEpilogue + FP8 = the move).

V3 + V4 directories remain compilable stubs. `bench/run_all.py` resolves
the registry at startup and runs cleanly across V0 + V1 + V2 + SDPA today,
gracefully skipping V3 + V4 stubs.

**Next session — V3 (FP8 / E4M3).** Per-tile scaling on top of V2's fused
structure. First-actions checklist:

1. `. .\env\activate_for_build.ps1; python env/check_env.py` (10/10 PASS).
2. `pip install -e kernels/v3_flash_fp8` to confirm the stub still builds.
3. Re-litigate WMMA vs CUTLASS one more time. **CUTLASS is the right call
   for V3** because (a) FP8 epilogue scaling is exactly what
   `CollectiveEpilogue` exists for, (b) V2's SMEM-resident O accumulator
   is the perf wall and CUTLASS's register-resident epilogue is the way
   through it, (c) CUTLASS Blackwell sm_120 FP8 GEMM examples have been
   landing through 2025–2026 and should be runnable now. Install as a git
   submodule under `external/cutlass/`, verify the FP8 sm_120 example
   from `examples/` runs end-to-end before touching V3 code.
4. If CUTLASS sm_120 FP8 still isn't ready, fall back to hand-rolled
   FP8 with `__nv_fp8_e4m3` and per-tile scale factors carried alongside
   the M, L running stats. This is a real risk; budget time accordingly.
5. Numerical envelope: FP8 E4M3 has ~4-bit mantissa, so per-tile scaling
   is mandatory (a single global scale would saturate). Per-tile = one
   scale per BR×BC block of S, propagated through the softmax and PV.
   Test grid should include the same high-variance regime that V1 / V2
   stress, with looser tolerance (probably atol=5e-2 not 1e-2 — verify
   from the FP8 paper and the SageAttention3 paper, both cited above).

Update this section at the end of every session.
