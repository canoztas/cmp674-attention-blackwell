"""AOT build for the V0 naive FP32 attention extension.

Install (editable, idempotent) from the repo root with::

    pip install -e kernels/v0_naive_fp32

The extension targets sm_120 (RTX 5080, Blackwell) explicitly via nvcc's
``-gencode=arch=compute_120,code=sm_120`` rather than relying on
``TORCH_CUDA_ARCH_LIST`` — silent arch fallback would defeat the project's
benchmarking goal. Builds for any other arch must fail loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

# Platform-specific host compiler flags. On Windows + recent MSVC (14.40+),
# torch's compiled_autograd.h hits a `C2872: 'std' ambiguous` error in its
# if-constexpr cascade unless `USE_CUDA` is defined — PyTorch's PR #144707
# guards that branch with `_WIN32 && (USE_CUDA || USE_ROCM)`. We don't use
# torch.compile from custom kernels, so the runtime stub the guard installs
# is inert for us.
if sys.platform == "win32":
    _CXX_FLAGS = ["/O2", "-DUSE_CUDA"]
    _NVCC_EXTRA = ["-DUSE_CUDA"]
else:
    _CXX_FLAGS = ["-O3"]
    _NVCC_EXTRA = []

ext_modules = [
    CUDAExtension(
        name="v0_naive_fp32._C",
        sources=[
            str(HERE / "binding.cpp"),
            str(HERE / "attention_v0.cu"),
        ],
        extra_compile_args={
            "cxx": _CXX_FLAGS,
            "nvcc": [
                "-O3",
                "-gencode=arch=compute_120,code=sm_120",
                "--use_fast_math",
                *_NVCC_EXTRA,
            ],
        },
    ),
]

setup(
    name="v0_naive_fp32",
    version="0.0.1",
    description="V0 naive FP32 attention kernel for RTX 5080 (sm_120)",
    packages=["v0_naive_fp32"],
    package_dir={"v0_naive_fp32": str(HERE)},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
