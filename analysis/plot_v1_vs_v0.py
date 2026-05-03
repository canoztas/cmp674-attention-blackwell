"""Plot V1 (FP16, tiled, Tensor Cores) vs V0 (FP32, naive) vs SDPA.

V0-CU comes from ``bench/results/v0_initial.parquet`` (FP32 sweep), V1-CU
and SDPA come from ``bench/results/v1_initial.parquet`` (FP16 sweep). Each
variant runs at its native precision - V0 in FP32, V1 in FP16, SDPA in
FP16 (matching V1's input dtype). Layout follows ``plot_v0_initial.py``:
log-log latency vs seq_len, one subplot per head_dim. The V1-vs-V0
speedup at the largest seq_len is computed at plot-time and embedded in
the figure suptitle so the headline number stays in sync with the data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_v1_vs_v0")


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
        "--v1-input",
        type=Path,
        default=Path("bench/results/v1_initial.parquet"),
        help="V1 (FP16) sweep parquet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figures/v1_vs_v0.png"),
        help="PNG output path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    v0 = _load(args.v0_input, "V0")
    v1 = _load(args.v1_input, "V1")

    v0_cu = v0[v0["variant_name"] == "v0_naive_cu"]
    v1_cu = v1[v1["variant_name"] == "v1_tiled_cu"]
    sdpa  = v1[v1["variant_name"] == "sdpa"]

    if v0_cu.empty or v1_cu.empty or sdpa.empty:
        logger.error(
            "missing variants: v0_naive_cu rows=%d, v1_tiled_cu rows=%d, sdpa rows=%d",
            len(v0_cu), len(v1_cu), len(sdpa),
        )
        return 1

    gpu_name = v1["gpu_name"].iloc[0]
    head_dims = sorted(set(v0_cu["head_dim"].unique()) | set(v1_cu["head_dim"].unique()))

    fig, axes = plt.subplots(
        1, len(head_dims), figsize=(5.5 * len(head_dims), 4.2), sharey=False, squeeze=False
    )
    axes = axes[0]

    series_specs = [
        ("V0-CU (FP32)",  v0_cu, "o", "tab:red"),
        ("V1-CU (FP16)",  v1_cu, "s", "tab:blue"),
        ("SDPA (FP16)",   sdpa,  "^", "tab:gray"),
    ]

    headline_speedup_lines: list[str] = []
    for ax, head_dim in zip(axes, head_dims):
        v0_at = v0_cu[v0_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()
        v1_at = v1_cu[v1_cu["head_dim"] == head_dim].groupby("seq_len")["latency_us"].median()

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
        ax.set_ylabel("median latency (µs)")
        ax.set_title(f"head_dim = {head_dim}")
        ax.grid(which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left")

        if not v0_at.empty and not v1_at.empty:
            largest = max(set(v0_at.index) & set(v1_at.index), default=None)
            if largest is not None:
                speedup = v0_at.loc[largest] / v1_at.loc[largest]
                headline_speedup_lines.append(
                    f"head_dim={head_dim}, seq_len={largest}: V1 is {speedup:.2f}x V0"
                )

    headline = (
        f"V1 (FP16, WMMA) vs V0 (FP32, naive) vs SDPA on {gpu_name}\n"
        + " | ".join(headline_speedup_lines)
    )
    fig.suptitle(headline, y=1.05)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.output)
    for line in headline_speedup_lines:
        logger.info(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
