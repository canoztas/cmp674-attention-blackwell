"""Plot V3 (FP8 E4M3, per-tile scaling) vs V2, V1, SDPA -- latency + memory.

Three-panel figure mirroring plot_v2_vs_v1_v0.py: log-log latency vs seq_len
per head_dim, plus peak memory at the largest head_dim. V3 extends the
sweep grid to seq_len=16384.

Headline numbers reported:
  - V3 vs V2 speedup at the largest seq_len in the sweep (V3's perf claim:
    register-resident O accumulator + FP8 Tensor Cores beat V2's SMEM-
    resident O + FP16 Tensor Cores).
  - V3 vs SDPA gap (the "are we competitive with cuDNN/FlashAttention-2"
    question: SDPA on RTX 5080 in mid-2026 likely dispatches FP16
    FlashAttention; V3's FP8 throughput should be a meaningful fraction).
  - Peak memory ratios across all four kernels at seq_len=16384 (V1's
    O(N^2) HBM cost vs V2/V3/SDPA's O(N)).

Inputs:
  - bench/results/v3_initial.parquet (V1, V2, V3, SDPA at b=2, h=8 across
    seq_len 128..16384, head_dim {64, 128}).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_v3_vs_v2_v1")


def _load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        logger.error("%s parquet not found: %s", label, path)
        sys.exit(1)
    return pq.read_table(path).to_pandas()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("bench/results/v3_initial.parquet"),
        help="V3 sweep parquet (also contains V1, V2, SDPA).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figures/v3_vs_v2_v1.png"),
        help="PNG output path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = _load(args.input, "V3 sweep")

    v1_cu = run[run["variant_name"] == "v1_tiled_cu"]
    v2_cu = run[run["variant_name"] == "v2_flash_cu"]
    v3_cu = run[run["variant_name"] == "v3_flash_cu"]
    sdpa  = run[run["variant_name"] == "sdpa"]

    if v3_cu.empty:
        logger.error("V3 results missing from %s; rerun the sweep with v3_flash_cu.", args.input)
        return 1

    gpu_name = run["gpu_name"].iloc[0]
    head_dims = sorted(v3_cu["head_dim"].unique())
    largest_hd = head_dims[-1]

    fig = plt.figure(figsize=(5.5 * len(head_dims) + 5.5, 4.2))
    n_lat = len(head_dims)
    axes_lat = [fig.add_subplot(1, n_lat + 1, i + 1) for i in range(n_lat)]
    ax_mem  = fig.add_subplot(1, n_lat + 1, n_lat + 1)

    series_specs = [
        ("V1-CU (FP16)",  v1_cu, "s", "tab:blue"),
        ("V2-CU (FP16)",  v2_cu, "D", "tab:green"),
        ("V3-CU (FP8)",   v3_cu, "*", "tab:purple"),
        ("SDPA (FP16)",   sdpa,  "^", "tab:gray"),
    ]

    headline_speedup_lines: list[str] = []
    for ax, head_dim in zip(axes_lat, head_dims):
        v2_at = v2_cu[v2_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()
        v3_at = v3_cu[v3_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()
        sdpa_at = sdpa[sdpa["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()

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
                markersize=8 if label.startswith("V3") else 6,
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

        common = sorted(set(v2_at.index) & set(v3_at.index) & set(sdpa_at.index))
        if common:
            largest = common[-1]
            v3_vs_v2   = v2_at.loc[largest] / v3_at.loc[largest]
            v3_vs_sdpa = sdpa_at.loc[largest] / v3_at.loc[largest]
            headline_speedup_lines.append(
                f"hd={head_dim}, seq_len={largest}: "
                f"V3 is {v3_vs_v2:.2f}x V2, {v3_vs_sdpa:.2f}x SDPA"
            )

    # Memory panel.
    mem_specs = [
        ("V1-CU (FP16)", v1_cu, "s", "tab:blue"),
        ("V2-CU (FP16)", v2_cu, "D", "tab:green"),
        ("V3-CU (FP8)",  v3_cu, "*", "tab:purple"),
        ("SDPA (FP16)",  sdpa,  "^", "tab:gray"),
    ]
    for label, src, marker, color in mem_specs:
        sub = src[src["head_dim"] == largest_hd]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("seq_len")["peak_memory_bytes"].first().sort_index() / (1024 * 1024)
        )
        ax_mem.plot(
            grouped.index.to_numpy(),
            grouped.to_numpy(),
            marker=marker,
            linestyle="-",
            linewidth=1.5,
            markersize=8 if label.startswith("V3") else 6,
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

    # Memory headline at largest seq_len, largest hd.
    v1_mem = v1_cu[v1_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    v2_mem = v2_cu[v2_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    v3_mem = v3_cu[v3_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    common_mem = sorted(set(v3_mem.index))
    mem_headline = ""
    if common_mem:
        largest_n = common_mem[-1]
        v3m = v3_mem.loc[largest_n]
        v2m = v2_mem.loc[largest_n] if largest_n in v2_mem.index else None
        v1m = v1_mem.loc[largest_n] if largest_n in v1_mem.index else None
        parts = [f"V3={v3m/(1024*1024):.0f} MB"]
        if v2m is not None:
            parts.insert(0, f"V2={v2m/(1024*1024):.0f}")
        if v1m is not None:
            parts.insert(0, f"V1={v1m/(1024*1024):.0f}")
        mem_headline = f" | mem at seq_len={largest_n}, hd={largest_hd}: " + ", ".join(parts)

    headline = (
        f"V3 (FP8 E4M3, per-tile scaling) vs V2 vs V1 vs SDPA on {gpu_name}\n"
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
