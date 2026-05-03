"""Smoke test for ``bench.harness.run_sweep``.

Runs the harness against a trivial in-place identity matmul (one ``Q @ K^T``
etc. would also work, but identity keeps the test independent of the V0
kernel build status). Verifies that:

1. The Parquet file is written.
2. It has the schema declared by ``bench.harness.PARQUET_SCHEMA``.
3. Row count matches ``num_iter * num_configs`` for the single variant.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest
import torch
import yaml

from bench.harness import PARQUET_SCHEMA, run_sweep

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _identity_kernel(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool) -> torch.Tensor:
    return V.clone()


def test_run_sweep_writes_parquet(tmp_path) -> None:
    cfg = {
        "sweep": {
            "name": "harness_smoke",
            "num_warmup": 1,
            "num_iter": 2,
            "input_seed": 0,
            "configs": {
                "seq_len": [64],
                "head_dim": [32],
                "batch_size": [1],
                "num_heads": [1],
                "dtype": ["torch.float32"],
                "causal": [False],
            },
        }
    }
    cfg_path = tmp_path / "smoke.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)

    out_path = tmp_path / "smoke.parquet"
    written = run_sweep(
        variant_name="smoke_identity",
        kernel_fn=_identity_kernel,
        config_path=cfg_path,
        output_path=out_path,
    )
    assert written.exists()

    table = pq.read_table(written)
    assert table.schema.equals(PARQUET_SCHEMA), (
        f"schema drift:\n got: {table.schema}\n want: {PARQUET_SCHEMA}"
    )

    expected_rows = 1 * cfg["sweep"]["num_iter"]  # 1 config × num_iter
    assert table.num_rows == expected_rows, (
        f"expected {expected_rows} rows, got {table.num_rows}"
    )

    df = table.to_pydict()
    assert set(df["variant_name"]) == {"smoke_identity"}
    assert set(df["run_idx"]) == {0, 1}
    assert all(lat >= 0.0 for lat in df["latency_us"])
