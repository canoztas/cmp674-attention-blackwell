"""Benchmark harness for attention kernel variants V0..V4.

Provides a stable :func:`run_sweep` entry point and a :class:`BenchmarkResult`
dataclass that any variant can consume without modification. Output is a
single Parquet file with a forward-compatible schema (variant-specific
tile / block columns are nullable so V0 rows coexist with V2+ rows).

Timing notes
------------
- Per-iteration latencies are measured with paired ``torch.cuda.Event``s
  (``enable_timing=True``). ``time.perf_counter`` is *not* sufficient for
  GPU work — it misses kernel-launch queueing and host/device sync cost.
- The first ``num_warmup`` runs are discarded to amortize JIT compilation,
  caching, and clock ramp-up.
- ``torch.cuda.synchronize()`` is called once before timing starts and once
  after every iteration's stop event so the recorded interval bounds only
  that iteration's GPU work.

Determinism caveats
-------------------
- Input tensors are seeded per-call (``input_seed`` from the YAML config or
  CLI override). Identical seeds produce bit-identical inputs across runs.
- CUDA reductions (softmax row-sums, etc.) are *not* bitwise reproducible
  across launches because thread-block scheduling and atomic-add ordering
  are non-deterministic. We accept this as a documented limitation.
- ``torch.nn.functional.scaled_dot_product_attention`` may dispatch to
  different backends (math / mem-efficient / flash) across versions or
  even within a process; treat its numerics as a moving reference.

Clock-locking caveat
--------------------
For steady-state benchmarks the GPU clocks should be locked, e.g.::

    nvidia-smi -lgc 2400        # admin shell; persists until reset
    nvidia-smi -rgc             # restore default behavior

This harness does *not* lock clocks automatically: it requires admin
privileges and the change persists across processes (and Steam, Chrome,
etc. would inherit the locked clocks). Document the clock state in the
paper's methodology section.
"""

from __future__ import annotations

import logging
import platform
import socket
import statistics
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public types                                                                #
# --------------------------------------------------------------------------- #


KernelFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]


@dataclass(frozen=True)
class BenchmarkConfig:
    """A single point in the sweep cartesian product."""

    batch: int
    num_heads: int
    seq_len: int
    head_dim: int
    dtype: str
    causal: bool
    input_seed: int


@dataclass
class BenchmarkResult:
    """Aggregated statistics for one (variant, config) pair."""

    variant_name: str
    precision_label: str
    config: BenchmarkConfig
    latency_us_mean: float
    latency_us_std: float
    latency_us_p50: float
    latency_us_p95: float
    latency_us_p99: float
    achieved_tflops_mean: float
    peak_memory_bytes: int
    per_run_latency_us: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Provenance                                                                  #
# --------------------------------------------------------------------------- #


def _git_commit_short() -> str:
    """Return ``git rev-parse --short HEAD`` or ``"unknown"`` if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            cwd=Path(__file__).resolve().parent.parent,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _driver_version() -> str:
    """Best-effort NVIDIA driver version string.

    ``torch.cuda.driver_version`` is not available on all torch builds; fall
    back to ``nvidia-smi --query-gpu=driver_version``. Returns ``"unknown"``
    when neither succeeds.
    """
    fn = getattr(torch.cuda, "driver_version", None)
    if callable(fn):
        try:
            return str(fn())
        except Exception:
            pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()
            if first:
                return first[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _gpu_metadata() -> dict[str, str]:
    if not torch.cuda.is_available():
        return {
            "gpu_name": "cpu",
            "gpu_compute_capability": "n/a",
            "driver_version": "n/a",
        }
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    return {
        "gpu_name": name,
        "gpu_compute_capability": f"sm_{cc[0]}{cc[1]}",
        "driver_version": _driver_version(),
    }


def _provenance() -> dict[str, str]:
    meta = _gpu_metadata()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit_short(),
        "torch_version": torch.__version__,
        "cuda_version": str(torch.version.cuda),
        "host_name": socket.gethostname() or platform.node() or "unknown",
        **meta,
    }


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #


# Forward-compatible schema. Variant-specific tile / block columns are
# declared nullable so V0 rows (which leave them null) and V2+ rows
# (which populate them) coexist in the same Parquet file.
PARQUET_SCHEMA = pa.schema(
    [
        # Provenance.
        ("timestamp", pa.string()),
        ("git_commit", pa.string()),
        ("gpu_name", pa.string()),
        ("gpu_compute_capability", pa.string()),
        ("driver_version", pa.string()),
        ("torch_version", pa.string()),
        ("cuda_version", pa.string()),
        ("host_name", pa.string()),
        # Identity.
        ("variant_name", pa.string()),
        ("precision_label", pa.string()),
        # Configuration.
        ("batch", pa.int64()),
        ("num_heads", pa.int64()),
        ("seq_len", pa.int64()),
        ("head_dim", pa.int64()),
        ("dtype", pa.string()),
        ("causal", pa.bool_()),
        ("input_seed", pa.int64()),
        # Variant-specific (nullable).
        ("tile_m", pa.int64()),
        ("tile_n", pa.int64()),
        ("tile_k", pa.int64()),
        ("block_size", pa.int64()),
        # Run-level.
        ("run_idx", pa.int64()),
        ("latency_us", pa.float64()),
        ("achieved_tflops", pa.float64()),
        ("peak_memory_bytes", pa.int64()),
    ]
)


# --------------------------------------------------------------------------- #
# Timing                                                                      #
# --------------------------------------------------------------------------- #


def time_kernel(
    kernel_fn: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    num_warmup: int,
    num_iter: int,
) -> list[float]:
    """Time ``kernel_fn(*args)`` with CUDA events; return per-iter latencies (us).

    The warmup runs are executed but not recorded. The function synchronizes
    once before the timed loop and once at the end of every iteration. Caller
    is responsible for ensuring all ``args`` live on the correct device.
    """
    if num_warmup < 0 or num_iter <= 0:
        raise ValueError(f"num_warmup>=0 and num_iter>0 required, got {num_warmup}, {num_iter}")

    for _ in range(num_warmup):
        kernel_fn(*args)
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(num_iter):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel_fn(*args)
        stop.record()
        torch.cuda.synchronize()
        latencies_us.append(start.elapsed_time(stop) * 1e3)
    return latencies_us


# --------------------------------------------------------------------------- #
# Configuration helpers                                                       #
# --------------------------------------------------------------------------- #


def _resolve_dtype(dtype_str: str) -> torch.dtype:
    """Map e.g. ``"torch.float32"`` or ``"float32"`` to a ``torch.dtype``."""
    key = dtype_str.removeprefix("torch.")
    dt = getattr(torch, key, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"Unknown dtype string: {dtype_str!r}")
    return dt


_PRECISION_LABEL = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float64: "fp64",
}


def _precision_label_for_dtype(dt: torch.dtype) -> str:
    return _PRECISION_LABEL.get(dt, str(dt).removeprefix("torch."))


def _as_list(x: Any) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _expand_configs(cfg: dict[str, Any]) -> tuple[list[BenchmarkConfig], dict[str, Any]]:
    """Expand the cartesian product of the YAML 'configs' block into a list."""
    sweep = cfg["sweep"]
    grid = sweep["configs"]
    seed = int(sweep.get("input_seed", 0))

    seq_lens = _as_list(grid["seq_len"])
    head_dims = _as_list(grid["head_dim"])
    batches = _as_list(grid["batch_size"])
    num_heads_list = _as_list(grid["num_heads"])
    dtypes = _as_list(grid["dtype"])
    causals = _as_list(grid["causal"])

    configs = [
        BenchmarkConfig(
            batch=int(b),
            num_heads=int(h),
            seq_len=int(s),
            head_dim=int(d),
            dtype=str(dt),
            causal=bool(c),
            input_seed=seed,
        )
        for s, d, b, h, dt, c in product(
            seq_lens, head_dims, batches, num_heads_list, dtypes, causals
        )
    ]
    meta = {
        "name": sweep.get("name", "unnamed"),
        "num_warmup": int(sweep.get("num_warmup", 25)),
        "num_iter": int(sweep.get("num_iter", 100)),
    }
    return configs, meta


def load_sweep_config(path: str | Path) -> tuple[list[BenchmarkConfig], dict[str, Any]]:
    """Parse a YAML sweep config into ``(configs, sweep_meta)``."""
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return _expand_configs(cfg)


# --------------------------------------------------------------------------- #
# Inputs / FLOPs                                                              #
# --------------------------------------------------------------------------- #


def make_inputs(
    cfg: BenchmarkConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate ``(Q, K, V)`` tensors with a fixed seed for reproducibility."""
    g = torch.Generator(device=device)
    g.manual_seed(cfg.input_seed)
    dt = _resolve_dtype(cfg.dtype)
    shape = (cfg.batch, cfg.num_heads, cfg.seq_len, cfg.head_dim)
    q = torch.randn(shape, generator=g, device=device, dtype=dt)
    k = torch.randn(shape, generator=g, device=device, dtype=dt)
    v = torch.randn(shape, generator=g, device=device, dtype=dt)
    return q, k, v


def attention_flops(cfg: BenchmarkConfig) -> int:
    """FLOP count for a single forward pass of scaled dot-product attention.

    The two matmuls (Q @ K^T and P @ V) each contribute
    ``2 * batch * num_heads * seq_len * seq_len * head_dim`` FLOPs (a multiply
    + an add per accumulation step), giving a total of
    ``4 * batch * num_heads * seq_len^2 * head_dim``. The softmax is O(N^2)
    in elementwise ops and is negligible against the matmul cost; we follow
    FlashAttention (arXiv 2205.14135) and others in ignoring it.
    """
    return 4 * cfg.batch * cfg.num_heads * cfg.seq_len * cfg.seq_len * cfg.head_dim


# --------------------------------------------------------------------------- #
# Sweep                                                                       #
# --------------------------------------------------------------------------- #


def _percentile(xs: Iterable[float], q: float) -> float:
    return float(np.percentile(np.asarray(list(xs), dtype=np.float64), q))


def _summarize(latencies_us: list[float], cfg: BenchmarkConfig) -> dict[str, float]:
    flops = attention_flops(cfg)
    mean_us = statistics.fmean(latencies_us)
    tflops = (flops / (mean_us * 1e-6)) / 1e12 if mean_us > 0 else float("nan")
    return {
        "latency_us_mean": float(mean_us),
        "latency_us_std": float(statistics.pstdev(latencies_us)) if len(latencies_us) > 1 else 0.0,
        "latency_us_p50": _percentile(latencies_us, 50.0),
        "latency_us_p95": _percentile(latencies_us, 95.0),
        "latency_us_p99": _percentile(latencies_us, 99.0),
        "achieved_tflops_mean": float(tflops),
    }


def benchmark_one(
    variant_name: str,
    kernel_fn: KernelFn,
    cfg: BenchmarkConfig,
    num_warmup: int,
    num_iter: int,
    device: torch.device | None = None,
) -> BenchmarkResult:
    """Time ``kernel_fn`` on one config and return aggregated stats."""
    device = device or torch.device("cuda")
    q, k, v = make_inputs(cfg, device)

    torch.cuda.reset_peak_memory_stats(device)
    latencies_us = time_kernel(kernel_fn, (q, k, v, cfg.causal), num_warmup, num_iter)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))

    summary = _summarize(latencies_us, cfg)
    precision_label = _precision_label_for_dtype(_resolve_dtype(cfg.dtype))
    return BenchmarkResult(
        variant_name=variant_name,
        precision_label=precision_label,
        config=cfg,
        peak_memory_bytes=peak_bytes,
        per_run_latency_us=latencies_us,
        **summary,
    )


def _result_to_rows(result: BenchmarkResult, prov: dict[str, str]) -> list[dict[str, Any]]:
    """Expand one ``BenchmarkResult`` to one row per timed iteration."""
    cfg = result.config
    rows: list[dict[str, Any]] = []
    for i, lat in enumerate(result.per_run_latency_us):
        rows.append(
            {
                **prov,
                "variant_name": result.variant_name,
                "precision_label": result.precision_label,
                "batch": cfg.batch,
                "num_heads": cfg.num_heads,
                "seq_len": cfg.seq_len,
                "head_dim": cfg.head_dim,
                "dtype": cfg.dtype,
                "causal": cfg.causal,
                "input_seed": cfg.input_seed,
                "tile_m": None,
                "tile_n": None,
                "tile_k": None,
                "block_size": None,
                "run_idx": i,
                "latency_us": float(lat),
                "achieved_tflops": result.achieved_tflops_mean,
                "peak_memory_bytes": result.peak_memory_bytes,
            }
        )
    return rows


def results_to_rows(results: list[BenchmarkResult]) -> list[dict[str, Any]]:
    """Expand a list of ``BenchmarkResult`` into Parquet-ready row dicts.

    Each result yields ``num_iter`` rows, each carrying the same provenance
    block (captured once at call time).
    """
    prov = _provenance()
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.extend(_result_to_rows(r, prov))
    return rows


def write_parquet(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    """Write rows to a Parquet file using the shared schema."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, list[Any]] = {f.name: [] for f in PARQUET_SCHEMA}
    for r in rows:
        for name in columns:
            columns[name].append(r.get(name))
    table = pa.Table.from_pydict(columns, schema=PARQUET_SCHEMA)
    pq.write_table(table, output_path)
    return output_path


def _progress_iter(items: list, description: str):
    """Yield items, with a rich progress bar if available."""
    try:
        from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn
    except ImportError:
        for it in items:
            yield it
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        transient=False,
    ) as p:
        task = p.add_task(description, total=len(items))
        for it in items:
            yield it
            p.advance(task)


def run_sweep(
    variant_name: str,
    kernel_fn: KernelFn,
    config_path: str | Path,
    output_path: str | Path,
    *,
    extra_results: list[BenchmarkResult] | None = None,
    write: bool = True,
) -> Path:
    """Run a full sweep for a single kernel variant and write a Parquet file.

    Args:
        variant_name: Tag stored in the ``variant_name`` column (e.g. "v0_naive_cu").
        kernel_fn: Callable with signature ``(Q, K, V, causal) -> Tensor``.
        config_path: Path to the YAML sweep config.
        output_path: Path to the Parquet file to write.
        extra_results: Optional pre-computed results (e.g. from another variant)
            to append to the same file. Useful for ``run_v0.py`` which writes
            three variants into one file with a single ``write_parquet`` call.
        write: If False, return the rows-equivalent path without writing —
            primarily used by tests.

    Returns:
        The Parquet path.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("run_sweep requires a CUDA device")

    configs, meta = load_sweep_config(config_path)
    device = torch.device("cuda")
    prov = _provenance()

    logger.info(
        "starting sweep '%s' variant=%s configs=%d warmup=%d iter=%d",
        meta["name"],
        variant_name,
        len(configs),
        meta["num_warmup"],
        meta["num_iter"],
    )

    results: list[BenchmarkResult] = list(extra_results or [])
    for cfg in _progress_iter(configs, f"{variant_name}"):
        try:
            result = benchmark_one(
                variant_name,
                kernel_fn,
                cfg,
                num_warmup=meta["num_warmup"],
                num_iter=meta["num_iter"],
                device=device,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.warning("OOM on %s %s: %s — skipping", variant_name, cfg, e)
            torch.cuda.empty_cache()
            continue
        results.append(result)

    rows: list[dict[str, Any]] = []
    for r in results:
        rows.extend(_result_to_rows(r, prov))

    output_path = Path(output_path)
    if write:
        write_parquet(rows, output_path)
        logger.info("wrote %d rows to %s", len(rows), output_path)
    return output_path


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "KernelFn",
    "PARQUET_SCHEMA",
    "attention_flops",
    "benchmark_one",
    "load_sweep_config",
    "make_inputs",
    "results_to_rows",
    "run_sweep",
    "time_kernel",
    "write_parquet",
]
