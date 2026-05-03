"""Plot V2 (FP16, fused, online softmax) vs V1, V0, SDPA - latency + memory.

Two-panel figure:
  Left/middle: log-log latency vs seq_len, one subplot per head_dim. Four
  series per subplot: V0-CU (FP32), V1-CU (FP16, tiled), V2-CU (FP16, fused),
  SDPA (FP16). V0 only appears at seq_lens where its sweep covered them
  (sweep_default.yaml: 128, 512, 1024, 2048).

  Right: peak memory vs seq_len at the largest head_dim. The V2 vs V1
  comparison here is the structural win: V1's O(N^2) S in HBM grows with
  seq_len^2 while V2 / SDPA stay flat at O(N) inputs+outputs.

V0 comes from bench/results/v0_initial.parquet (FP32 at b=4, h=16).
V1, V2, SDPA come from bench/results/v2_initial.parquet (FP16 at b=2, h=8).
The (B, H) discrepancy is acknowledged in the figure caption: V0's snapshot
preserved as-is for reproducibility (CLAUDE.md), V2's sweep at smaller
(B, H) to leave headroom at seq_len=8192.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_v2_vs_v1_v0")


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        logger.error("%s parquet not found: %s", label, path)
        sys.exit(1)
    return pq.read_table(path).to_pandas()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v0-input",
        type=Path,
        default=Path("bench/results/v0_initial.parquet"),
        help="V0 (FP32) sweep parquet.",
    )
    parser.add_argument(
        "--v2-input",
        type=Path,
        default=Path("bench/results/v2_initial.parquet"),
        help="V2 sweep parquet (also contains V1 and SDPA).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figures/v2_vs_v1_v0.png"),
        help="PNG output path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    v0 = _load(args.v0_input, "V0")
    v2_run = _load(args.v2_input, "V2 sweep")

    v0_cu = v0[v0["variant_name"] == "v0_naive_cu"]
    v1_cu = v2_run[v2_run["variant_name"] == "v1_tiled_cu"]
    v2_cu = v2_run[v2_run["variant_name"] == "v2_flash_cu"]
    sdpa  = v2_run[v2_run["variant_name"] == "sdpa"]

    if v1_cu.empty or v2_cu.empty or sdpa.empty:
        logger.error(
            "missing variants in v2 parquet: v1=%d v2=%d sdpa=%d",
            len(v1_cu), len(v2_cu), len(sdpa),
        )
        return 1

    gpu_name = v2_run["gpu_name"].iloc[0]
    head_dims = sorted(set(v1_cu["head_dim"].unique()) | set(v2_cu["head_dim"].unique()))
    largest_hd = head_dims[-1]

    fig = plt.figure(figsize=(5.5 * len(head_dims) + 5.5, 4.2))
    # Layout: latency subplots side-by-side then memory subplot on the right.
    n_lat = len(head_dims)
    axes_lat = [fig.add_subplot(1, n_lat + 1, i + 1) for i in range(n_lat)]
    ax_mem  = fig.add_subplot(1, n_lat + 1, n_lat + 1)

    series_specs = [
        ("V0-CU (FP32)",  v0_cu, "o", "tab:red"),
        ("V1-CU (FP16)",  v1_cu, "s", "tab:blue"),
        ("V2-CU (FP16)",  v2_cu, "D", "tab:green"),
        ("SDPA (FP16)",   sdpa,  "^", "tab:gray"),
    ]

    headline_speedup_lines: list[str] = []
    for ax, head_dim in zip(axes_lat, head_dims):
        v1_at = v1_cu[v1_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()
        v2_at = v2_cu[v2_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()

        for label, src, marker, color in series_specs:
            sub = src[src["head_dim"] == head_dim]
            if sub.empty:
                continue
            grouped = sub.groupby("seq_len")["latency_us"].median().sort_index()
            ax.plot(
                grouped.index.to_numpy(),
                grouped.to_numpy(),
                marker=marker,
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                label=label,
                color=color,
            )

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("median latency (us)")
        ax.set_title(f"latency, head_dim = {head_dim}")
        ax.grid(which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=9)

        # Headline: V2 vs V1 speedup at the largest seq_len both ran.
        common = sorted(set(v1_at.index) & set(v2_at.index))
        if common:
            largest = common[-1]
            speedup = v1_at.loc[largest] / v2_at.loc[largest]
            headline_speedup_lines.append(
                f"head_dim={head_dim}, seq_len={largest}: V2 is {speedup:.2f}x V1"
            )

    # Memory panel - peak memory bytes vs seq_len at largest head_dim.
    mem_specs = [
        ("V1-CU (FP16)", v1_cu, "s", "tab:blue"),
        ("V2-CU (FP16)", v2_cu, "D", "tab:green"),
        ("SDPA (FP16)",  sdpa,  "^", "tab:gray"),
    ]
    for label, src, marker, color in mem_specs:
        sub = src[src["head_dim"] == largest_hd]
        if sub.empty:
            continue
        # peak_memory_bytes is a single value per (variant, config); take first.
        grouped = (
            sub.groupby("seq_len")["peak_memory_bytes"].first().sort_index() / (1024 * 1024)
        )
        ax_mem.plot(
            grouped.index.to_numpy(),
            grouped.to_numpy(),
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=6,
            label=label,
            color=color,
        )
    ax_mem.set_xscale("log", base=2)
    ax_mem.set_yscale("log")
    ax_mem.set_xlabel("sequence length")
    ax_mem.set_ylabel("peak GPU memory (MB)")
    ax_mem.set_title(f"peak memory, head_dim = {largest_hd}")
    ax_mem.grid(which="both", linestyle=":", alpha=0.5)
    ax_mem.legend(loc="upper left", fontsize=9)

    # Memory headline.
    v1_mem = v1_cu[v1_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    v2_mem = v2_cu[v2_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    common_mem = sorted(set(v1_mem.index) & set(v2_mem.index))
    mem_headline = ""
    if common_mem:
        largest_n = common_mem[-1]
        ratio = v1_mem.loc[largest_n] / v2_mem.loc[largest_n]
        mem_headline = (
            f" | mem at seq_len={largest_n}, hd={largest_hd}: V1={v1_mem.loc[largest_n]/(1024*1024):.0f} MB, "
            f"V2={v2_mem.loc[largest_n]/(1024*1024):.0f} MB ({ratio:.0f}x lower)"
        )

    headline = (
        f"V2 (FP16, fused, online softmax) vs V1 vs V0 vs SDPA on {gpu_name}\n"
        + " | ".join(headline_speedup_lines) + mem_headline
    )
    fig.suptitle(headline, y=1.05, fontsize=10)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.output)
    for line in headline_speedup_lines:
        logger.info(line)
    if mem_headline:
        logger.info(mem_headline.lstrip(" |"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
