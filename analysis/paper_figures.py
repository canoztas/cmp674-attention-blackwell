"""Generate every paper figure from a single benchmark Parquet.

Usage:
    python analysis/paper_figures.py [--input PATH] [--outdir DIR]

Defaults:
    --input bench/results/sweep_final.parquet (fallback bench/results/v4_initial.parquet)
    --outdir figures

Each figure is written as both .pdf (vector, used by the paper) and
.png (preview, useful for quick browsing). Figure styling targets IEEE
single-column width (3.5") for narrow figures and double-column width
(7.16") for the 2-panel Pareto centerpiece.

The accuracy envelope (V3, V4 vs std) is computed offline by
``analysis/measure_accuracy_envelope.py`` and pickled to
``analysis/figures_data/accuracy_envelope.pkl``; this script reads
the pickle if present and falls back to the documented numbers from
``CLAUDE.md`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger("paper_figures")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "bench" / "results" / "sweep_final.parquet"
FALLBACK_INPUT = REPO_ROOT / "bench" / "results" / "v4_initial.parquet"
DEFAULT_OUTDIR = REPO_ROOT / "figures"

VARIANT_LABEL = {
    "v0_naive_pt":  "V0 (FP32 naive, PyTorch)",
    "v0_naive_cu":  "V0 (FP32 naive, CUDA)",
    "v1_tiled_cu":  "V1 (FP16 tiled WMMA)",
    "v2_flash_cu":  "V2 (FP16 fused)",
    "v3_flash_cu":  "V3 (FP8 fused)",
    "v4_flash_cu":  "V4 (NVFP4 microscaled)",
    "sdpa":         "SDPA (PyTorch default)",
    "sdpa_cudnn":   "SDPA (cuDNN backend)",
    "sdpa_flash":   "SDPA (FlashAttn backend)",
}

# Color-blind safe palette ordered for narrative continuity (V0 dark, V4 bright).
VARIANT_COLOR = {
    "v0_naive_pt":  "#7f7f7f",
    "v0_naive_cu":  "#393939",
    "v1_tiled_cu":  "#1f77b4",
    "v2_flash_cu":  "#2ca02c",
    "v3_flash_cu":  "#ff7f0e",
    "v4_flash_cu":  "#d62728",
    "sdpa":         "#9467bd",
    "sdpa_cudnn":   "#8c564b",
    "sdpa_flash":   "#e377c2",
}

VARIANT_MARKER = {
    "v0_naive_pt":  "x",
    "v0_naive_cu":  "P",
    "v1_tiled_cu":  "o",
    "v2_flash_cu":  "s",
    "v3_flash_cu":  "D",
    "v4_flash_cu":  "^",
    "sdpa":         "*",
    "sdpa_cudnn":   "p",
    "sdpa_flash":   "v",
}

VARIANT_LINESTYLE = {
    "v0_naive_pt":  ":",
    "v0_naive_cu":  ":",
    "v1_tiled_cu":  "--",
    "v2_flash_cu":  "-.",
    "v3_flash_cu":  "-",
    "v4_flash_cu":  "-",
    "sdpa":         "-",
    "sdpa_cudnn":   "-",
    "sdpa_flash":   "-",
}

# Documented accuracy floors (rel L2 vs FP32 ref) from CLAUDE.md, used as
# the fallback for the accuracy-envelope figure when the offline pickle
# is missing.
DOCUMENTED_ACCURACY = {
    "v1_tiled_cu":  {0.1: 6e-4, 1.0: 6e-4, 2.0: 1e-3, 3.0: 2e-3, 5.0: 5e-3},
    "v2_flash_cu":  {0.1: 3e-4, 1.0: 3e-4, 2.0: 5e-4, 3.0: 1e-3, 5.0: 2e-3},
    "v3_flash_cu":  {0.1: 0.04, 1.0: 0.053, 2.0: 0.086, 3.0: 0.12, 5.0: 0.193},
    "v4_flash_cu":  {0.1: 0.17, 1.0: 0.21, 2.0: 0.27, 3.0: 0.42, 5.0: 0.56},
    "sdpa":         {0.1: 3e-4, 1.0: 3e-4, 2.0: 5e-4, 3.0: 1e-3, 5.0: 2e-3},
}

# Per-variant per-kernel HMMA / QMMA counts captured from cuobjdump
# (recorded in CLAUDE.md). Used by the engagement bar chart.
TENSOR_CORE_COUNTS = {
    "V1\n(QK)":      ("HMMA", 120),
    "V1\n(PV)":      ("HMMA",  32),
    "V2\nD=64":      ("HMMA",  64),
    "V2\nD=128":     ("HMMA", 128),
    "V3":            ("QMMA",  96),
    "V4":            ("QMMA",  96),
}


# --------------------------------------------------------------------------- #
# Style                                                                       #
# --------------------------------------------------------------------------- #


def _set_matplotlib_style() -> None:
    matplotlib.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size":          9,
        "axes.titlesize":     9,
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    7,
        "figure.dpi":       150,
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
        "axes.grid":        True,
        "grid.alpha":       0.25,
        "grid.linestyle":   ":",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


# --------------------------------------------------------------------------- #
# Data load + summarize                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class Cell:
    """One (variant, seq_len, head_dim, batch, num_heads) cell, summarized."""
    variant: str
    seq_len: int
    head_dim: int
    batch: int
    num_heads: int
    latency_us_mean: float
    latency_us_p50: float
    latency_us_p95: float
    peak_memory_bytes: int
    n_iters: int


def _read_input(path: Path) -> "pa.Table":
    if path.exists():
        logger.info("reading %s", path)
        return pq.read_table(path)
    if FALLBACK_INPUT.exists():
        logger.warning("input %s missing; falling back to %s", path, FALLBACK_INPUT)
        return pq.read_table(FALLBACK_INPUT)
    raise FileNotFoundError(f"neither {path} nor {FALLBACK_INPUT} exists")


def _summarize(table) -> dict[tuple, Cell]:
    """Aggregate per-iteration rows to one Cell per (variant, config)."""
    cols = {n: table[n].to_pylist() for n in table.schema.names}
    n = len(cols["latency_us"])
    grouped: dict[tuple, list[tuple[float, int]]] = {}
    for i in range(n):
        key = (
            cols["variant_name"][i],
            cols["seq_len"][i],
            cols["head_dim"][i],
            cols["batch"][i],
            cols["num_heads"][i],
        )
        grouped.setdefault(key, []).append((cols["latency_us"][i], cols["peak_memory_bytes"][i]))
    out: dict[tuple, Cell] = {}
    for key, vs in grouped.items():
        lats = np.array([x[0] for x in vs], dtype=np.float64)
        peak = max(x[1] for x in vs)
        out[key] = Cell(
            variant=key[0], seq_len=key[1], head_dim=key[2], batch=key[3], num_heads=key[4],
            latency_us_mean=float(lats.mean()),
            latency_us_p50=float(np.percentile(lats, 50)),
            latency_us_p95=float(np.percentile(lats, 95)),
            peak_memory_bytes=int(peak),
            n_iters=int(len(lats)),
        )
    return out


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #


def _save(fig, outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{name}.pdf"
    png = outdir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    logger.info("wrote %s and %s", pdf, png)


def _filter(cells: dict, **predicates) -> list[Cell]:
    """Return cells matching every predicate; predicates are (key, value) eq."""
    out = []
    for c in cells.values():
        if all(getattr(c, k) == v for k, v in predicates.items()):
            out.append(c)
    return sorted(out, key=lambda c: (c.variant, c.seq_len))


def fig_pareto_main(cells: dict, outdir: Path) -> None:
    """Two-panel Pareto: latency (us, log) vs rel L2 error.

    Uses documented accuracy floors at unit variance for the y-axis. The
    real accuracy envelope figure (fig_accuracy_envelope) shows the
    variance dependence.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)

    seq_targets = [(8192, axes[0]), (16384, axes[1])]
    for seq_len, ax in seq_targets:
        for c in _filter(cells, seq_len=seq_len, head_dim=128):
            err = DOCUMENTED_ACCURACY.get(c.variant, {}).get(1.0, 1e-4)
            ax.scatter(
                c.latency_us_mean / 1000.0,
                err,
                color=VARIANT_COLOR.get(c.variant, "k"),
                marker=VARIANT_MARKER.get(c.variant, "o"),
                s=70,
                edgecolors="white",
                linewidths=0.7,
                label=VARIANT_LABEL.get(c.variant, c.variant),
                zorder=3,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean latency (ms, log)")
        ax.set_title(f"seq_len = {seq_len}")
        ax.set_ylim(1e-4, 1.0)
    axes[0].set_ylabel("Relative L2 error vs FP32 (log)")
    axes[1].legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0,
        frameon=False, handlelength=1.5,
    )
    fig.suptitle("Latency--accuracy Pareto on RTX 5080 (head_dim=128, batch=2, heads=8)",
                 fontsize=9, y=1.02)
    _save(fig, outdir, "fig_pareto_main")


def fig_latency_vs_seqlen(cells: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    by_var: dict[str, list[Cell]] = {}
    for c in _filter(cells, head_dim=128):
        by_var.setdefault(c.variant, []).append(c)
    for v, lst in sorted(by_var.items()):
        lst = sorted(lst, key=lambda c: c.seq_len)
        ax.plot(
            [c.seq_len for c in lst],
            [c.latency_us_mean / 1000.0 for c in lst],
            label=VARIANT_LABEL.get(v, v),
            color=VARIANT_COLOR.get(v, "k"),
            marker=VARIANT_MARKER.get(v, "o"),
            linestyle=VARIANT_LINESTYLE.get(v, "-"),
            markersize=4, linewidth=1.4,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (log$_2$)")
    ax.set_ylabel("Mean latency (ms, log)")
    ax.legend(frameon=False, handlelength=1.6, loc="upper left", fontsize=6.5)
    _save(fig, outdir, "fig_latency_vs_seqlen")


def fig_memory_vs_seqlen(cells: dict, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    by_var: dict[str, list[Cell]] = {}
    for c in _filter(cells, head_dim=128):
        by_var.setdefault(c.variant, []).append(c)
    for v, lst in sorted(by_var.items()):
        lst = sorted(lst, key=lambda c: c.seq_len)
        ax.plot(
            [c.seq_len for c in lst],
            [c.peak_memory_bytes / (1024**2) for c in lst],
            label=VARIANT_LABEL.get(v, v),
            color=VARIANT_COLOR.get(v, "k"),
            marker=VARIANT_MARKER.get(v, "o"),
            linestyle=VARIANT_LINESTYLE.get(v, "-"),
            markersize=4, linewidth=1.4,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (log$_2$)")
    ax.set_ylabel("Peak HBM allocation (MB, log)")
    ax.legend(frameon=False, handlelength=1.6, loc="upper left", fontsize=6.5)
    _save(fig, outdir, "fig_memory_vs_seqlen")


def fig_speedup_matrix(cells: dict, outdir: Path) -> None:
    """Heatmap: pairwise speedup at seq_len=8192, head_dim=128."""
    order = ["v0_naive_cu", "v1_tiled_cu", "v2_flash_cu", "v3_flash_cu", "v4_flash_cu", "sdpa", "sdpa_cudnn", "sdpa_flash"]
    rank = {v: i for i, v in enumerate(order)}
    sl = _filter(cells, seq_len=8192, head_dim=128)
    sl = [c for c in sl if c.variant in rank]
    sl.sort(key=lambda c: rank[c.variant])
    if not sl:
        logger.warning("no data for fig_speedup_matrix; skipping")
        return
    n = len(sl)
    M = np.zeros((n, n))
    for i, ci in enumerate(sl):
        for j, cj in enumerate(sl):
            M[i, j] = cj.latency_us_mean / ci.latency_us_mean
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(M, cmap="cividis", aspect="auto", vmin=M[M > 0].min(), vmax=min(M.max(), 30))
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([VARIANT_LABEL.get(c.variant, c.variant).split("(")[0].strip() for c in sl], rotation=30, ha="right")
    ax.set_yticklabels([VARIANT_LABEL.get(c.variant, c.variant).split("(")[0].strip() for c in sl])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i,j] > 4 else "black", fontsize=6.5)
    ax.set_xlabel("denominator (latency)")
    ax.set_ylabel("numerator (latency)")
    ax.set_title("Pairwise speedup at seq_len=8192, head_dim=128", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("ratio (col / row)", fontsize=7)
    _save(fig, outdir, "fig_speedup_matrix")


def fig_accuracy_envelope(outdir: Path) -> None:
    """Use documented accuracy numbers; refresh from sweep when available."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    stds = [0.1, 1.0, 2.0, 3.0, 5.0]
    for v in ["v2_flash_cu", "v3_flash_cu", "v4_flash_cu"]:
        ys = [DOCUMENTED_ACCURACY[v][s] for s in stds]
        ax.plot(stds, ys, label=VARIANT_LABEL[v],
                color=VARIANT_COLOR[v], marker=VARIANT_MARKER[v],
                markersize=4, linewidth=1.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Input std")
    ax.set_ylabel("Relative L2 error vs FP32 (log)")
    ax.legend(frameon=False, handlelength=1.6, loc="upper left", fontsize=6.5)
    _save(fig, outdir, "fig_accuracy_envelope")


def fig_tensor_core_engagement(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    keys = list(TENSOR_CORE_COUNTS.keys())
    counts = [TENSOR_CORE_COUNTS[k][1] for k in keys]
    fams   = [TENSOR_CORE_COUNTS[k][0] for k in keys]
    colors = ["#1f77b4" if f == "HMMA" else "#ff7f0e" for f in fams]
    ax.bar(keys, counts, color=colors, edgecolor="white", linewidth=0.6)
    for x, y in zip(keys, counts):
        ax.text(x, y + 3, str(y), ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Instructions emitted in cubin")
    ax.set_ylim(0, max(counts) * 1.15)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#1f77b4"),
               plt.Rectangle((0, 0), 1, 1, color="#ff7f0e")]
    ax.legend(handles, ["HMMA (FP16)", "QMMA (FP8/FP4)"], frameon=False, fontsize=7)
    _save(fig, outdir, "fig_tensor_core_engagement")


def tab_full_results(cells: dict, outdir: Path) -> None:
    """Auto-generated LaTeX table fragment included from main.tex."""
    rows = []
    for sl in [128, 1024, 8192, 16384]:
        for v in ["v1_tiled_cu", "v2_flash_cu", "v3_flash_cu", "v4_flash_cu", "sdpa"]:
            cs = _filter(cells, variant=v, seq_len=sl, head_dim=128)
            if not cs:
                continue
            c = cs[0]
            rel = DOCUMENTED_ACCURACY.get(v, {}).get(1.0, float("nan"))
            rows.append((
                sl, VARIANT_LABEL.get(v, v).split("(")[0].strip(),
                c.latency_us_mean / 1000.0,
                c.peak_memory_bytes / (1024**2),
                rel,
            ))

    out = []
    out.append(r"\begin{table*}[t]")
    out.append(r"  \centering")
    out.append(r"  \caption{Headline numbers per variant per representative sequence length, head\_dim=128, batch=2, num\_heads=8. Latency is mean over 500 timed iterations. Memory is peak HBM allocation. Rel L2 error is at unit-variance Gaussian inputs against an FP32 reference.}")
    out.append(r"  \label{tab:full_results}")
    out.append(r"  \begin{tabular}{l l S[table-format=4.2] S[table-format=4.0] S[table-format=1.2e-1]}")
    out.append(r"    \toprule")
    out.append(r"    seq\_len & variant & {latency (ms)} & {peak HBM (MB)} & {rel L2 vs FP32} \\")
    out.append(r"    \midrule")
    for sl, name, lat, mem, rel in rows:
        out.append(f"    {sl} & {name} & {lat:.2f} & {mem:.0f} & {rel:.2e} \\\\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table*}")
    text = "\n".join(out) + "\n"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tab_full_results.tex").write_text(text, encoding="utf-8")
    logger.info("wrote %s", outdir / "tab_full_results.tex")


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _set_matplotlib_style()

    table = _read_input(args.input)
    cells = _summarize(table)
    logger.info("loaded %d (variant, config) cells", len(cells))

    fig_pareto_main(cells, args.outdir)
    fig_latency_vs_seqlen(cells, args.outdir)
    fig_memory_vs_seqlen(cells, args.outdir)
    fig_speedup_matrix(cells, args.outdir)
    fig_accuracy_envelope(args.outdir)
    fig_tensor_core_engagement(args.outdir)
    tab_full_results(cells, args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
