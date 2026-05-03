"""Paper headline figure: two-panel latency vs accuracy Pareto.

Two panels at seq_len=8192 and seq_len=16384. All five variants (V1, V2,
V3, V4) + SDPA. Latency on x-axis (log), relative L2 error on y-axis (log).

This is the figure that tells the entire story of the paper: as we move
from V1 (FP16 baseline) to V4 (FP4 microscaled), each variant trades off
accuracy for latency in a documented way. SDPA is the cuDNN/FlashAttention
reference.

Inputs:
  - bench/results/v4_initial.parquet (V1, V2, V3, V4, SDPA at b=2, h=8 across
    seq_len 128..16384, head_dim {64, 128}).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_full_pareto_final")


def _make_qkv_fp16(B: int, H: int, N: int, D: int, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    Q = torch.randn(B, H, N, D, generator=g, device="cuda", dtype=torch.float32)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    return Q.half(), K.half(), V.half()


def _ref(Q, K, V) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q.float(), K.float(), V.float()).half()


def _rel_l2(out, ref) -> float:
    return ((out.float() - ref.float()).norm() / ref.float().norm()).item()


def _accuracy_for_panel(B: int, H: int, N: int, D: int) -> dict[str, float]:
    Q, K, V = _make_qkv_fp16(B, H, N, D)
    ref = _ref(Q, K, V)
    out_acc: dict[str, float] = {"sdpa": _rel_l2(F.scaled_dot_product_attention(Q, K, V), ref)}
    for variant_pkg, variant_attr, key in [
        ("v1_tiled_fp16",  "attention_tiled_cu",       "v1_tiled_cu"),
        ("v2_flash_fp16",  "attention_flash_cu",       "v2_flash_cu"),
        ("v3_flash_fp8",   "attention_flash_fp8_cu",   "v3_flash_cu"),
        ("v4_flash_nvfp4", "attention_flash_nvfp4_cu", "v4_flash_cu"),
    ]:
        try:
            mod = __import__(variant_pkg)
            if getattr(mod, "HAS_CUDA_EXT", False):
                out = getattr(mod, variant_attr)(Q, K, V, False)
                out_acc[key] = _rel_l2(out, ref)
        except Exception as e:  # noqa: BLE001
            logger.warning("accuracy skip %s: %s", key, e)
    return out_acc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",
        type=Path, default=Path("bench/results/v4_initial.parquet"))
    parser.add_argument("--output",
        type=Path, default=Path("analysis/figures/full_pareto_final.png"))
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch",    type=int, default=2)
    parser.add_argument("--heads",    type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.input.exists():
        logger.error("missing parquet: %s", args.input); return 1
    df = pq.read_table(args.input).to_pandas()

    label_map = {
        "v1_tiled_cu": "V1 (FP16 tiled)",
        "v2_flash_cu": "V2 (FP16 fused)",
        "v3_flash_cu": "V3 (FP8 per-tile)",
        "v4_flash_cu": "V4 (NVFP4 microscale)",
        "sdpa": "SDPA (FP16 cuDNN)",
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
    variant_order = ["v1_tiled_cu", "v2_flash_cu", "v3_flash_cu", "v4_flash_cu", "sdpa"]

    seq_lens = [8192, 16384]
    fig, axes = plt.subplots(1, len(seq_lens), figsize=(7.0 * len(seq_lens), 5.5),
                             sharey=True)

    for ax, seq_len in zip(axes, seq_lens):
        sub = df[
            (df["seq_len"] == seq_len) & (df["head_dim"] == args.head_dim)
            & (df["batch"] == args.batch) & (df["num_heads"] == args.heads)
        ]
        if sub.empty:
            logger.warning("no rows at seq_len=%d", seq_len)
            continue

        lat = {}
        for v in variant_order:
            rows = sub[sub["variant_name"] == v]
            if not rows.empty:
                lat[v] = rows["latency_us"].median()

        acc = _accuracy_for_panel(args.batch, args.heads, seq_len, args.head_dim)

        for v in variant_order:
            if v not in lat or v not in acc:
                continue
            x = lat[v] / 1000.0  # ms
            y = acc[v]
            ax.scatter(x, y,
                       marker=marker_map[v],
                       s=200 if v == "v4_flash_cu" else 150,
                       color=color_map[v],
                       edgecolors="black",
                       linewidths=0.8,
                       label=label_map[v],
                       zorder=3)
            ax.annotate(v.split("_")[0].upper(),
                        (x, y), xytext=(8, -3), textcoords="offset points",
                        fontsize=10)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("median latency (ms, log)")
        ax.set_title(f"seq_len = {seq_len}")
        ax.grid(which="both", linestyle=":", alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("relative L2 error vs FP32 reference (log)")
            ax.legend(loc="lower left", fontsize=9)

        for v in variant_order:
            if v in lat and v in acc:
                logger.info("seq_len=%d %s: latency=%.3f ms, rel_l2=%.4f",
                            seq_len, v, lat[v] / 1000, acc[v])

    gpu_name = df["gpu_name"].iloc[0]
    fig.suptitle(
        f"Latency vs accuracy Pareto on {gpu_name} -- "
        f"head_dim={args.head_dim}, batch={args.batch}, heads={args.heads}",
        y=1.02, fontsize=11,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
