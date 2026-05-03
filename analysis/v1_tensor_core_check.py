"""One-call V1 driver for Nsight Compute Tensor Core verification.

Run with::

    ncu --set full --target-processes all --kernel-name regex:qk_tile_kernel \
        python analysis/v1_tensor_core_check.py

Or, for a quick sanity check on Tensor Core engagement only::

    ncu --metrics smsp__inst_executed_pipe_tensor.sum \
        python analysis/v1_tensor_core_check.py

Per CLAUDE.md V1 risk #3: a kernel can compile and produce correct
results without using Tensor Cores. If ``smsp__inst_executed_pipe_tensor``
is near zero we are running on regular FP16 ALUs - silent perf bug.
"""

from __future__ import annotations

import torch

from v1_tiled_fp16 import attention_tiled_cu


def main() -> None:
    torch.manual_seed(0)
    B, H, N, D = 4, 16, 1024, 64
    Q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    # Warmup once to amortize JIT-ish setup. The metric we care about
    # samples the kernel body itself, not warmup overhead.
    _ = attention_tiled_cu(Q, K, V, False)
    torch.cuda.synchronize()

    out = attention_tiled_cu(Q, K, V, False)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    main()
