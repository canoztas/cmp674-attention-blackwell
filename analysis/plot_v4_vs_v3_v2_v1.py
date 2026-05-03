"""Plot V4 (NVFP4 / FP4 microscaled) vs V3, V2, V1, SDPA -- latency + memory.

Three-panel figure mirroring plot_v3_vs_v2_v1.py: log-log latency vs seq_len
per head_dim, plus peak memory at the largest head_dim. V4 is the FP4
endpoint of the precision sweep.

Headline numbers reported:
  - V4 vs V3 speedup at the largest seq_len in the sweep (V4's perf claim:
    Blackwell 5th-gen Tensor Cores at FP4 throughput should approach 2x FP8;
    practical ratio depends on microscaling overhead).
  - V4 vs SDPA gap (consumer-Blackwell FP16 FlashAttention reference).
  - Peak memory ratios at seq_len=16384 (V4 is the same as V3/V2/SDPA at
    the input/output level since the binding contract is FP16 in/out).

Inputs:
  - bench/results/v4_initial.parquet (V1, V2, V3, V4, SDPA at b=2, h=8
    across seq_len 128..16384, head_dim {64, 128}).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_v4_vs_v3_v2_v1")


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
        default=Path("bench/results/v4_initial.parquet"),
        help="V4 sweep parquet (also contains V1, V2, V3, SDPA).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figures/v4_vs_v3_v2_v1.png"),
        help="PNG output path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = _load(args.input, "V4 sweep")

    v1_cu = run[run["variant_name"] == "v1_tiled_cu"]
    v2_cu = run[run["variant_name"] == "v2_flash_cu"]
    v3_cu = run[run["variant_name"] == "v3_flash_cu"]
    v4_cu = run[run["variant_name"] == "v4_flash_cu"]
    sdpa  = run[run["variant_name"] == "sdpa"]

    if v4_cu.empty:
        logger.error("V4 results missing from %s; rerun the sweep with v4_flash_cu.", args.input)
        return 1

    gpu_name = run["gpu_name"].iloc[0]
    head_dims = sorted(v4_cu["head_dim"].unique())
    largest_hd = head_dims[-1]

    fig = plt.figure(figsize=(5.5 * len(head_dims) + 5.5, 4.2))
    n_lat = len(head_dims)
    axes_lat = [fig.add_subplot(1, n_lat + 1, i + 1) for i in range(n_lat)]
    ax_mem  = fig.add_subplot(1, n_lat + 1, n_lat + 1)

    series_specs = [
        ("V1-CU (FP16)",  v1_cu, "s", "tab:blue"),
        ("V2-CU (FP16)",  v2_cu, "D", "tab:green"),
        ("V3-CU (FP8)",   v3_cu, "*", "tab:purple"),
        ("V4-CU (FP4)",   v4_cu, "P", "tab:red"),
        ("SDPA (FP16)",   sdpa,  "^", "tab:gray"),
    ]

    headline_speedup_lines: list[str] = []
    for ax, head_dim in zip(axes_lat, head_dims):
        v3_at = v3_cu[v3_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()
        v4_at = v4_cu[v4_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()
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
                markersize=8 if label.startswith("V4") else 6,
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

        common = sorted(set(v3_at.index) & set(v4_at.index) & set(sdpa_at.index))
        if common:
            largest = common[-1]
            v4_vs_v3   = v3_at.loc[largest] / v4_at.loc[largest]
            v4_vs_sdpa = sdpa_at.loc[largest] / v4_at.loc[largest]
            headline_speedup_lines.append(
                f"hd={head_dim}, seq_len={largest}: "
                f"V4 is {v4_vs_v3:.2f}x V3, {v4_vs_sdpa:.2f}x SDPA"
            )

    # Memory panel.
    mem_specs = [
        ("V1-CU (FP16)", v1_cu, "s", "tab:blue"),
        ("V2-CU (FP16)", v2_cu, "D", "tab:green"),
        ("V3-CU (FP8)",  v3_cu, "*", "tab:purple"),
        ("V4-CU (FP4)",  v4_cu, "P", "tab:red"),
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
            markersize=8 if label.startswith("V4") else 6,
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
    v3_mem = v3_cu[v3_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    v4_mem = v4_cu[v4_cu["head_dim"] == largest_hd].groupby("seq_len")["peak_memory_bytes"].first()
    common_mem = sorted(set(v4_mem.index))
    mem_headline = ""
    if common_mem:
        largest_n = common_mem[-1]
        v4m = v4_mem.loc[largest_n]
        v3m = v3_mem.loc[largest_n] if largest_n in v3_mem.index else None
        parts = [f"V4={v4m/(1024*1024):.0f} MB"]
        if v3m is not None:
            parts.insert(0, f"V3={v3m/(1024*1024):.0f}")
        mem_headline = f" | mem at seq_len={largest_n}, hd={largest_hd}: " + ", ".join(parts)

    headline = (
        f"V4 (NVFP4 microscaled) vs V3 vs V2 vs V1 vs SDPA on {gpu_name}\n"
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
