"""Plot V0 latency vs sequence length, grouped by head_dim.

This script is the *precursor* to a paper figure: layout, font sizing, and
styling are intentionally minimal and will be replaced in a later session
with a publication-styled (IEEE-column-width, vector output) version.

Reads ``bench/results/v0_initial.parquet`` (default), groups rows by
``(variant_name, head_dim)``, plots median latency vs ``seq_len`` on
log-log axes — one subplot per ``head_dim``, three lines per subplot
(V0-PT, V0-CU, SDPA).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

logger = logging.getLogger("analysis.plot_v0_initial")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("bench/results/v0_initial.parquet"),
        help="Parquet input path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/figures/v0_initial.png"),
        help="PNG output path.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not args.input.exists():
        logger.error("input parquet not found: %s", args.input)
        return 1

    table = pq.read_table(args.input)
    df = table.to_pandas()

    if df.empty:
        logger.error("input parquet is empty: %s", args.input)
        return 1

    gpu_name = df["gpu_name"].iloc[0]
    head_dims = sorted(df["head_dim"].unique())
    variants = sorted(df["variant_name"].unique())

    fig, axes = plt.subplots(
        1, len(head_dims), figsize=(5.5 * len(head_dims), 4.2), sharey=False, squeeze=False
    )
    axes = axes[0]

    markers = {"v0_naive_pt": "o", "v0_naive_cu": "s", "sdpa": "^"}

    for ax, head_dim in zip(axes, head_dims):
        sub_hd = df[df["head_dim"] == head_dim]
        for variant in variants:
            sub = sub_hd[sub_hd["variant_name"] == variant]
            if sub.empty:
                continue
            grouped = sub.groupby("seq_len")["latency_us"].median().sort_index()
            ax.plot(
                grouped.index.to_numpy(),
                grouped.to_numpy(),
                marker=markers.get(variant, "x"),
                linestyle="-",
                linewidth=1.5,
                markersize=6,
                label=variant,
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("median latency (µs)")
        ax.set_title(f"head_dim = {head_dim}")
        ax.grid(which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left")

    fig.suptitle(f"V0 attention — median latency vs seq_len ({gpu_name})", y=1.02)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
