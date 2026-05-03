"""Run the V0 sweep across V0-PT, V0-CU, and PyTorch SDPA.

All three variants share one Parquet file so downstream analysis can
compare them with a single :func:`pyarrow.parquet.read_table` call.

PowerShell, from repo root::

    gpu
    pip install -e kernels/v0_naive_fp32
    python bench/run_v0.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.harness import (  # noqa: E402
    PARQUET_SCHEMA,
    BenchmarkResult,
    benchmark_one,
    load_sweep_config,
    results_to_rows,
    write_parquet,
)

logger = logging.getLogger("bench.run_v0")


def _sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def _load_v0_kernels() -> dict[str, callable]:
    """Resolve ``(variant_name -> kernel_fn)`` for the three V0 variants."""
    from v0_naive_fp32 import HAS_CUDA_EXT, attention_naive_cu, attention_naive_pt

    variants: dict[str, callable] = {
        "v0_naive_pt": attention_naive_pt,
        "sdpa": _sdpa,
    }
    if HAS_CUDA_EXT:
        variants["v0_naive_cu"] = attention_naive_cu
    else:
        logger.warning(
            "v0_naive_fp32._C extension not built; skipping v0_naive_cu. "
            "Run `pip install -e kernels/v0_naive_fp32` to enable."
        )
    return variants


def _make_progress(total: int):
    """Return ``(advance_fn, context_manager)`` for a progress UI.

    Uses ``rich.progress`` if available; otherwise returns a no-op pair so
    the call site stays linear. The returned context manager is entered
    for the duration of the loop.
    """
    import contextlib

    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )
    except ImportError:
        return (lambda: None, contextlib.nullcontext())

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=False,
    )
    task_id = progress.add_task("V0 sweep", total=total)
    return (lambda: progress.advance(task_id), progress)


def _setup_logging() -> None:
    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[RichHandler(show_time=True, show_path=False, markup=False)],
        )
    except ImportError:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the V0 sweep (V0-PT, V0-CU, SDPA).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("bench/configs/sweep_default.yaml"),
        help="YAML sweep config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/v0_initial.parquet"),
        help="Output Parquet path.",
    )
    args = parser.parse_args()

    _setup_logging()

    if not torch.cuda.is_available():
        logger.error("CUDA not available; aborting.")
        return 1

    configs, meta = load_sweep_config(args.config)
    variants = _load_v0_kernels()

    logger.info(
        "V0 sweep '%s': %d configs × %d variants = %d (variant, config) pairs",
        meta["name"],
        len(configs),
        len(variants),
        len(configs) * len(variants),
    )

    device = torch.device("cuda")
    results: list[BenchmarkResult] = []
    pairs = [(name, fn, cfg) for name, fn in variants.items() for cfg in configs]

    advance, progress_cm = _make_progress(len(pairs))
    with progress_cm:
        for name, fn, cfg in pairs:
            try:
                result = benchmark_one(
                    variant_name=name,
                    kernel_fn=fn,
                    cfg=cfg,
                    num_warmup=meta["num_warmup"],
                    num_iter=meta["num_iter"],
                    device=device,
                )
                results.append(result)
            except torch.cuda.OutOfMemoryError as e:
                logger.warning("OOM on %s %s: %s — skipping", name, cfg, e)
                torch.cuda.empty_cache()
            advance()

    rows = results_to_rows(results)
    write_parquet(rows, args.output)
    logger.info(
        "wrote %d rows (%d results, schema=%d cols) to %s",
        len(rows),
        len(results),
        len(PARQUET_SCHEMA),
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
