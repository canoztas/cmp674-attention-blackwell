"""Toolchain validation for the attention-blackwell project.

Run this after activating the `gpu` venv:

    gpu
    python env/check_env.py

The script verifies that the host environment can compile and run a custom CUDA
kernel for sm_120 (RTX 5080, Blackwell) today. It is the first thing to run each
session — Blackwell tooling is still maturing, so silent breakage is expected
unless we explicitly catch it.

Exit code 0 if all critical checks pass, 1 otherwise. Non-critical checks
(VRAM info, ncu / nvcc availability) emit warnings but never fail the script.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

try:
    from rich.console import Console
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# Required hardware/software targets.
REQUIRED_CC: tuple[int, int] = (12, 0)  # sm_120 — RTX 5080 Blackwell
REQUIRED_PYTHON: tuple[int, int] = (3, 10)


Status = Literal["PASS", "FAIL", "WARN", "INFO"]


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    critical: bool = True


# --------------------------------------------------------------------------- #
# Individual checks                                                           #
# --------------------------------------------------------------------------- #


def check_python_version() -> CheckResult:
    v = sys.version_info
    have = (v.major, v.minor)
    if have >= REQUIRED_PYTHON:
        return CheckResult("Python version", "PASS", f"{v.major}.{v.minor}.{v.micro}")
    return CheckResult(
        "Python version",
        "FAIL",
        f"need >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}, got {v.major}.{v.minor}.{v.micro}",
    )


def check_torch() -> tuple[CheckResult, object | None]:
    try:
        import torch
    except ImportError as e:
        return CheckResult("PyTorch import", "FAIL", f"ImportError: {e}"), None
    detail = f"torch=={torch.__version__}, built against CUDA {torch.version.cuda}"
    return CheckResult("PyTorch import", "PASS", detail), torch


def check_cuda_available(torch) -> CheckResult:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return CheckResult("CUDA available", "PASS", f"{n} device(s) visible")
    return CheckResult(
        "CUDA available",
        "FAIL",
        "torch.cuda.is_available() == False — driver, env, or wrong torch wheel",
    )


def check_device(torch) -> tuple[CheckResult, CheckResult]:
    name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    name_check = CheckResult("Device 0 name", "INFO", name, critical=False)

    have = f"sm_{cc[0]}{cc[1]}"
    want = f"sm_{REQUIRED_CC[0]}{REQUIRED_CC[1]}"
    if cc == REQUIRED_CC:
        cc_check = CheckResult("Compute capability", "PASS", f"{have} matches required {want}")
    else:
        cc_check = CheckResult(
            "Compute capability",
            "FAIL",
            f"{have} does not match required {want} — kernels will not run",
        )
    return name_check, cc_check


def check_vram(torch) -> CheckResult:
    free, total = torch.cuda.mem_get_info(0)
    gib = 1024**3
    return CheckResult(
        "VRAM",
        "INFO",
        f"{free / gib:.2f} GiB free / {total / gib:.2f} GiB total",
        critical=False,
    )


def check_sdpa(torch) -> CheckResult:
    """Smoke test PyTorch's built-in SDPA on a tiny FP16 input."""
    try:
        import torch.nn.functional as F  # noqa: N812

        q = torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        out = F.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
    except Exception as e:
        return CheckResult("SDPA smoke test", "FAIL", f"{type(e).__name__}: {e}")

    if out.shape != q.shape:
        return CheckResult(
            "SDPA smoke test",
            "FAIL",
            f"unexpected output shape: got {tuple(out.shape)}, expected {tuple(q.shape)}",
        )
    if not torch.isfinite(out).all().item():
        return CheckResult("SDPA smoke test", "FAIL", "output contains non-finite values")
    return CheckResult(
        "SDPA smoke test",
        "PASS",
        "F.scaled_dot_product_attention OK on (1, 1, 64, 64) FP16",
    )


def check_cuda_jit(torch) -> CheckResult:
    """JIT-compile a trivial CUDA extension targeting sm_120 and run it.

    This is the most important check in the script: if a tiny add-kernel can't
    be built and executed for sm_120 today, no V0+ kernel will work either.
    """
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as e:
        return CheckResult("CUDA JIT compile (sm_120)", "FAIL", f"cpp_extension import: {e}")

    cuda_src = r"""
    #include <torch/extension.h>

    __global__ void env_check_add_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) c[i] = a[i] + b[i];
    }

    torch::Tensor env_check_add(torch::Tensor a, torch::Tensor b) {
        TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
        TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32,
                    "fp32 only");
        TORCH_CHECK(a.numel() == b.numel(), "size mismatch");
        auto c = torch::empty_like(a);
        const int n = static_cast<int>(a.numel());
        const int threads = 256;
        const int blocks = (n + threads - 1) / threads;
        env_check_add_kernel<<<blocks, threads>>>(
            a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
        return c;
    }
    """
    cpp_src = "torch::Tensor env_check_add(torch::Tensor a, torch::Tensor b);"

    # MSVC 14.40+ triggers a `C2872: 'std' ambiguous` error in
    # torch/csrc/dynamo/compiled_autograd.h's if-constexpr cascade when an
    # extension is built without `-DUSE_CUDA`. PyTorch added a Windows guard
    # (PR #144707) that replaces the cascade with a runtime stub when
    # `_WIN32 && (USE_CUDA || USE_ROCM)` is defined. Extension builds don't
    # set USE_CUDA by default, so we add it explicitly. We don't use
    # torch.compile from custom kernels, so the runtime stub is harmless.
    if sys.platform == "win32":
        cxx_flags = ["/O2", "-DUSE_CUDA"]
        cuda_extra = ["-DUSE_CUDA"]
    else:
        cxx_flags = ["-O3"]
        cuda_extra = []

    try:
        ext = load_inline(
            name="env_check_ext",
            cpp_sources=[cpp_src],
            cuda_sources=[cuda_src],
            functions=["env_check_add"],
            extra_cflags=cxx_flags,
            extra_cuda_cflags=[
                "-O3",
                "-gencode=arch=compute_120,code=sm_120",
                "--use_fast_math",
                *cuda_extra,
            ],
            verbose=False,
        )
    except Exception as e:
        return CheckResult(
            "CUDA JIT compile (sm_120)",
            "FAIL",
            f"build failed — {type(e).__name__}: {str(e)[:1500]}",
        )

    try:
        a = torch.randn(1024, device="cuda", dtype=torch.float32)
        b = torch.randn(1024, device="cuda", dtype=torch.float32)
        c = ext.env_check_add(a, b)
        torch.cuda.synchronize()
    except Exception as e:
        return CheckResult(
            "CUDA JIT compile (sm_120)",
            "FAIL",
            f"runtime error — {type(e).__name__}: {e}",
        )

    if not torch.allclose(c, a + b, atol=0.0, rtol=0.0):
        max_err = (c - (a + b)).abs().max().item()
        return CheckResult(
            "CUDA JIT compile (sm_120)",
            "FAIL",
            f"kernel ran but result diverged from CPU reference (max abs err {max_err})",
        )

    return CheckResult(
        "CUDA JIT compile (sm_120)",
        "PASS",
        "trivial add kernel built for sm_120 and produced correct results",
    )


def check_nvcc() -> CheckResult:
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return CheckResult(
            "nvcc on PATH",
            "WARN",
            "not found on PATH — required for AOT extension builds in V0+",
            critical=False,
        )
    try:
        out = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        version_line = next(
            (ln for ln in out.stdout.splitlines() if "release" in ln.lower()),
            (out.stdout.strip().splitlines() or ["(no output)"])[-1],
        )
        return CheckResult(
            "nvcc on PATH", "PASS", f"{nvcc} — {version_line.strip()}", critical=False
        )
    except Exception as e:
        return CheckResult(
            "nvcc on PATH",
            "WARN",
            f"found at {nvcc} but invocation failed: {e}",
            critical=False,
        )


def check_ncu() -> CheckResult:
    ncu = shutil.which("ncu")
    if not ncu:
        return CheckResult(
            "Nsight Compute (ncu)",
            "WARN",
            "not on PATH — required for kernel profiling later but not for V0",
            critical=False,
        )
    try:
        out = subprocess.run(
            [ncu, "--version"], capture_output=True, text=True, timeout=10, check=False
        )
        first = (out.stdout.strip().splitlines() or ["(no output)"])[0]
        return CheckResult("Nsight Compute (ncu)", "PASS", first, critical=False)
    except Exception as e:
        return CheckResult(
            "Nsight Compute (ncu)",
            "WARN",
            f"invocation failed: {e}",
            critical=False,
        )


# --------------------------------------------------------------------------- #
# Output                                                                      #
# --------------------------------------------------------------------------- #

STATUS_STYLE = {"PASS": "green", "FAIL": "red", "WARN": "yellow", "INFO": "cyan"}


def render_rich(results: list[CheckResult]) -> None:
    console = Console()
    table = Table(title="attention-blackwell — toolchain validation", show_lines=False)
    table.add_column("Check", style="bold")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    for r in results:
        style = STATUS_STYLE.get(r.status, "white")
        table.add_row(r.name, f"[{style}]{r.status}[/{style}]", r.detail)
    console.print(table)


def render_plain(results: list[CheckResult]) -> None:
    print("attention-blackwell — toolchain validation")
    print("-" * 72)
    for r in results:
        print(f"[{r.status:4s}] {r.name}: {r.detail}")
    print("-" * 72)


def emit(message: str, style: str) -> None:
    if HAS_RICH:
        Console().print(f"[bold {style}]{message}[/bold {style}]")
    else:
        print(message)


# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #


def main() -> int:
    results: list[CheckResult] = []

    results.append(check_python_version())

    torch_result, torch = check_torch()
    results.append(torch_result)

    if torch is not None:
        cuda_result = check_cuda_available(torch)
        results.append(cuda_result)
        if cuda_result.status == "PASS":
            name_r, cc_r = check_device(torch)
            results.append(name_r)
            results.append(cc_r)
            results.append(check_vram(torch))
            results.append(check_sdpa(torch))
            # Only attempt JIT compile if we are on the right device — a wrong-cc
            # build will hide useful error context behind a wall of nvcc output.
            if cc_r.status == "PASS":
                results.append(check_cuda_jit(torch))
            else:
                results.append(
                    CheckResult(
                        "CUDA JIT compile (sm_120)",
                        "FAIL",
                        "skipped — compute capability check did not pass",
                    )
                )

    results.append(check_nvcc())
    results.append(check_ncu())

    if HAS_RICH:
        render_rich(results)
    else:
        render_plain(results)

    failed_critical = [r for r in results if r.critical and r.status == "FAIL"]
    n_warn = sum(1 for r in results if r.status == "WARN")

    if failed_critical:
        emit(
            f"\n{len(failed_critical)} critical check(s) FAILED. See details above.",
            "red",
        )
        return 1

    suffix = f" ({n_warn} warning(s))" if n_warn else ""
    emit(f"\nAll {len(results)} checks completed; no critical failures{suffix}.", "green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
