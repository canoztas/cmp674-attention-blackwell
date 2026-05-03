"""Latency vs accuracy Pareto across V1, V2, V3, V4 + SDPA at one shape.

The paper's headline figure: each variant is a single point in (latency,
accuracy) space at a fixed (seq_len, head_dim, batch, heads). Accuracy
is measured as relative L2 error vs an FP32 V0-PT reference. Latency
comes from the V4 sweep parquet (which contains all five variants).

V3's role on this Pareto: pushes the latency frontier left (faster) at
the cost of larger relative error than V2 -- the FP8 envelope.
V4's role: pushes further left at the cost of larger error than V3 --
the FP4 envelope.

V0 is omitted: at seq_len=8192 with the b=2, h=8 grid V0 would materialize
a (2*8*8192*8192) = 1 GiB FP32 attention matrix per call, doable but not
in the current sweep snapshot. V0 latency is added back in the paper from
the v0_initial.parquet at the V0 grid (smaller b, h).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_full_pareto")


def _make_qkv_fp16(B: int, H: int, N: int, D: int, seed: int = 0) -> tuple:
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, N, D, generator=g, device="cuda", dtype=torch.float32)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    return Q.half(), K.half(), V.half()


def _ref(Q, K, V, causal=False) -> torch.Tensor:
    """FP32 V0-PT-equivalent reference."""
    return F.scaled_dot_product_attention(
        Q.float(), K.float(), V.float(), is_causal=causal
    ).half()


def _rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",
        type=Path, default=Path("bench/results/v4_initial.parquet"))
    parser.add_argument("--output",
        type=Path, default=Path("analysis/figures/full_pareto.png"))
    parser.add_argument("--seq-len",  type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch",    type=int, default=2)
    parser.add_argument("--heads",    type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.input.exists():
        logger.error("missing parquet: %s", args.input); return 1
    df = pq.read_table(args.input).to_pandas()
    sub = df[
        (df["seq_len"] == args.seq_len) & (df["head_dim"] == args.head_dim)
        & (df["batch"] == args.batch) & (df["num_heads"] == args.heads)
    ]
    if sub.empty:
        logger.error("no parquet rows at the requested shape; check sweep coverage"); return 1

    # Latency per variant.
    lat = {}
    for v in ["v1_tiled_cu", "v2_flash_cu", "v3_flash_cu", "v4_flash_cu", "sdpa"]:
        rows = sub[sub["variant_name"] == v]
        if rows.empty:
            continue
        lat[v] = rows["latency_us"].median()  # microseconds

    # Accuracy: rerun each kernel once and compute rel-L2 vs FP32 reference.
    Q, K, V = _make_qkv_fp16(args.batch, args.heads, args.seq_len, args.head_dim)
    ref = _ref(Q, K, V)

    accuracy = {}

    if "sdpa" in lat:
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        accuracy["sdpa"] = _rel_l2(out, ref)

    if "v1_tiled_cu" in lat:
        try:
            import v1_tiled_fp16 as m
            if m.HAS_CUDA_EXT:
                out = m.attention_tiled_cu(Q, K, V, False)
                accuracy["v1_tiled_cu"] = _rel_l2(out, ref)
        except Exception as e:
            logger.warning("V1 accuracy run skipped: %s", e)

    if "v2_flash_cu" in lat:
        try:
            import v2_flash_fp16 as m
            if m.HAS_CUDA_EXT:
                out = m.attention_flash_cu(Q, K, V, False)
                accuracy["v2_flash_cu"] = _rel_l2(out, ref)
        except Exception as e:
            logger.warning("V2 accuracy run skipped: %s", e)

    if "v3_flash_cu" in lat:
        try:
            import v3_flash_fp8 as m
            if m.HAS_CUDA_EXT:
                out = m.attention_flash_fp8_cu(Q, K, V, False)
                accuracy["v3_flash_cu"] = _rel_l2(out, ref)
        except Exception as e:
            logger.warning("V3 accuracy run skipped: %s", e)

    if "v4_flash_cu" in lat:
        try:
            import v4_flash_nvfp4 as m
            if m.HAS_CUDA_EXT:
                out = m.attention_flash_nvfp4_cu(Q, K, V, False)
                accuracy["v4_flash_cu"] = _rel_l2(out, ref)
        except Exception as e:
            logger.warning("V4 accuracy run skipped: %s", e)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    label_map = {
        "v1_tiled_cu": "V1 (FP16, tiled)",
        "v2_flash_cu": "V2 (FP16, fused, online softmax)",
        "v3_flash_cu": "V3 (FP8 E4M3, per-tile scaling)",
        "v4_flash_cu": "V4 (NVFP4 E2M1, per-row per-K-block microscaling)",
        "sdpa": "SDPA (FP16, cuDNN)",
    }
    color_map = {
        "v1_tiled_cu": "tab:blue",
        "v2_flash_cu": "tab:green",
        "v3_flash_cu": "tab:purple",
        "v4_flash_cu": "tab:red",
        "sdpa": "tab:gray",
    }
    marker_map = {
        "v1_tiled_cu": "s",
        "v2_flash_cu": "D",
        "v3_flash_cu": "*",
        "v4_flash_cu": "P",
        "sdpa": "^",
    }

    for v in lat:
        if v not in accuracy:
            continue
        x = lat[v] / 1000.0  # ms
        y = accuracy[v]
        ax.scatter(x, y,
                   marker=marker_map[v],
                   s=180 if v in ("v3_flash_cu", "v4_flash_cu") else 130,
                   color=color_map[v],
                   edgecolors="black",
                   linewidths=0.8,
                   label=label_map[v],
                   zorder=3)
        ax.annotate(label_map[v].split(" (")[0],
                    (x, y), xytext=(8, -3), textcoords="offset points",
                    fontsize=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median latency (ms, log)")
    ax.set_ylabel("relative L2 error vs FP32 reference (log)")
    title = (
        f"Latency-accuracy Pareto on RTX 5080 (sm_120)\n"
        f"seq_len={args.seq_len}, head_dim={args.head_dim}, "
        f"batch={args.batch}, heads={args.heads}"
    )
    ax.set_title(title)
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.output)
    for v in lat:
        if v in accuracy:
            logger.info("%s: latency=%.3f ms, rel_l2=%.4f",
                        v, lat[v] / 1000, accuracy[v])
    return 0


if __name__ == "__main__":
    sys.exit(main())
