# attention-blackwell

Mixed-precision attention kernel benchmarks on NVIDIA RTX 5080 (Blackwell, sm_120).

This repository accompanies an in-progress graduate research paper that maps the Pareto
frontier of latency vs. numerical accuracy vs. memory across a family of attention kernels:

| ID  | Precision   | Design                                            |
| --- | ----------- | ------------------------------------------------- |
| V0  | FP32        | Naive (full N×N attention matrix in HBM)          |
| V1  | FP16        | Tiled, Tensor Core (WMMA / CUTLASS)               |
| V2  | FP16        | Fused + online softmax (FlashAttention-style)     |
| V3  | FP8 (E4M3)  | V2 + per-tile scaling                             |
| V4  | FP4 (NVFP4) | V3 + microscaling (Blackwell-only, exploratory)   |

## Quick start

```powershell
gpu                                # activate the shared CUDA venv
pip install -r env/requirements.txt
python env/check_env.py            # validates the toolchain on sm_120
```

For full project context, conventions, and the per-session workflow, see [CLAUDE.md](CLAUDE.md).
