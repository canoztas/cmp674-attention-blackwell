"""Master comparison driver: run a sweep across all built variants.

This is the long-lived counterpart to ``bench/run_v0.py``. ``run_v0.py`` is a
snapshot of the V0-paper experiment (V0-PT, V0-CU, SDPA over the V0 grid).
``run_all.py`` is the rolling driver that grows with the project: it
introspects the variant registry, skips variants whose compiled extension
isn't built yet, and writes a single Parquet.

PowerShell, from the repo root::

    gpu                          # activate the shared CUDA venv
    python bench/run_all.py
    python bench/run_all.py --variants v0_naive_pt v0_naive_cu sdpa
    python bench/run_all.py --config bench/configs/sweep_default.yaml `
                            --output bench/results/all.parquet

Variants are resolved at startup by trying to import each registered package
and looking up its ``HAS_CUDA_EXT`` flag (for compiled variants) or just the
callable (for the PyTorch references and SDPA). Missing variants log a
warning and are dropped from the sweep — this keeps the script useful in
early sessions where V1+ are not yet built.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
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

logger = logging.getLogger("bench.run_all")


@dataclass(frozen=True)
class VariantSpec:
    """Registry entry: how to import a variant and what to call it."""

    name: str            # tag stored in `variant_name` column
    package: str | None  # importable package; None for built-ins like SDPA
    attr: str | None     # callable attribute on the package
    requires_ext: bool   # if True, also check `HAS_CUDA_EXT`


# Order matters for output ergonomics (groups by variant in the Parquet).
VARIANT_REGISTRY: list[VariantSpec] = [
    VariantSpec("sdpa",         None,              None,                       False),
    VariantSpec("v0_naive_pt",  "v0_naive_fp32",   "attention_naive_pt",       False),
    VariantSpec("v0_naive_cu",  "v0_naive_fp32",   "attention_naive_cu",       True),
    VariantSpec("v1_tiled_cu",  "v1_tiled_fp16",   "attention_tiled_cu",       True),
    VariantSpec("v2_flash_cu",  "v2_flash_fp16",   "attention_flash_cu",       True),
    VariantSpec("v3_flash_cu",  "v3_flash_fp8",    "attention_flash_fp8_cu",   True),
    VariantSpec("v4_flash_cu",  "v4_flash_nvfp4",  "attention_flash_nvfp4_cu", True),
]


def _sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool) -> torch.Tensor:
    return F.scaled_dot_product_attention(Q, K, V, is_causal=causal)


def _resolve_variant(spec: VariantSpec) -> Callable | None:
    """Return the kernel callable, or ``None`` if it isn't usable.

    Logs a warning explaining the skip reason on every failure path.
    """
    if spec.package is None:
        return _sdpa
    try:
        module = importlib.import_module(spec.package)
    except ImportError as e:
        logger.warning("variant %s: cannot import %s (%s) — skipping", spec.name, spec.package, e)
        return None
    if spec.requires_ext and not getattr(module, "HAS_CUDA_EXT", False):
        logger.warning(
            "variant %s: %s.HAS_CUDA_EXT is False — extension not built. "
            "Run `pip install -e kernels/%s` to enable.",
            spec.name, spec.package, spec.package,
        )
        return None
    fn = getattr(module, spec.attr or "", None)
    if not callable(fn):
        logger.warning("variant %s: %s.%s is not callable — skipping", spec.name, spec.package, spec.attr)
        return None
    return fn


def _resolve_variants(requested: list[str] | None) -> dict[str, Callable]:
    """Build the (variant_name -> kernel_fn) dict for this run."""
    if requested is not None:
        unknown = set(requested) - {s.name for s in VARIANT_REGISTRY}
        if unknown:
            logger.warning("ignoring unknown variants: %s", ", ".join(sorted(unknown)))
        registry = [s for s in VARIANT_REGISTRY if s.name in requested]
        # Honor the user's requested order if any.
        registry.sort(key=lambda s: requested.index(s.name))
    else:
        registry = list(VARIANT_REGISTRY)

    resolved: dict[str, Callable] = {}
    for spec in registry:
        fn = _resolve_variant(spec)
        if fn is not None:
            resolved[spec.name] = fn
    return resolved


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


def _make_progress(total: int):
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
    task_id = progress.add_task("attention sweep", total=total)
    return (lambda: progress.advance(task_id), progress)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("bench/configs/sweep_default.yaml"),
        help="YAML sweep config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bench/results/all.parquet"),
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help=(
            "Explicit whitelist (default: all registered variants that are "
            "importable + built). Names: "
            + ", ".join(s.name for s in VARIANT_REGISTRY)
        ),
    )
    args = parser.parse_args()

    _setup_logging()

    if not torch.cuda.is_available():
        logger.error("CUDA not available; aborting.")
        return 1

    configs, meta = load_sweep_config(args.config)
    variants = _resolve_variants(args.variants)
    if not variants:
        logger.error("no variants resolved; nothing to run")
        return 1

    logger.info(
        "sweep '%s': %d configs × %d variants = %d (variant, config) pairs",
        meta["name"],
        len(configs),
        len(variants),
        len(configs) * len(variants),
    )
    logger.info("variants: %s", ", ".join(variants.keys()))

    device = torch.device("cuda")
    pairs = [(name, fn, cfg) for name, fn in variants.items() for cfg in configs]
    results: list[BenchmarkResult] = []

    advance, progress_cm = _make_progress(len(pairs))
    with progress_cm:
        for name, fn, cfg in pairs:
            try:
                results.append(
                    benchmark_one(
                        variant_name=name,
                        kernel_fn=fn,
                        cfg=cfg,
                        num_warmup=meta["num_warmup"],
                        num_iter=meta["num_iter"],
                        device=device,
                    )
                )
            except torch.cuda.OutOfMemoryError as e:
                logger.warning("OOM on %s %s: %s — skipping", name, cfg, e)
                torch.cuda.empty_cache()
            except NotImplementedError as e:
                # Stub kernels (V1-V4 before their session) raise this.
                logger.warning("not implemented: %s %s: %s — skipping", name, cfg, e)
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
