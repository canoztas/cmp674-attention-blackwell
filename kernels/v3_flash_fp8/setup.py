"""AOT build for the V3 FP8 (E4M3) fused attention extension.

Stub: implemented in the V3 session. Boilerplate only — sm_120 explicit, the
Windows-only ``-DUSE_CUDA`` macro for torch 2.9 + MSVC 14.40+ compatibility.

V3 design target: V2 + FP8 (E4M3) inputs with per-tile dynamic scaling, FP32
accumulation. Reference: Micikevicius 2022 (arXiv 2209.05433) for FP8 formats.
The 5th-gen Tensor Cores on Blackwell (sm_120) natively support E4M3 / E5M2.
"""

from __future__ import annotations

import sys
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).resolve().parent

if sys.platform == "win32":
    _CXX_FLAGS = ["/O2", "-DUSE_CUDA"]
    _NVCC_EXTRA = ["-DUSE_CUDA"]
else:
    _CXX_FLAGS = ["-O3"]
    _NVCC_EXTRA = []

ext_modules = [
    CUDAExtension(
        name="v3_flash_fp8._C",
        sources=[
            str(HERE / "binding.cpp"),
            str(HERE / "attention_v3.cu"),
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
    name="v3_flash_fp8",
    version="0.0.1",
    description="V3 FP8 (E4M3) fused attention for RTX 5080 (sm_120) [stub]",
    packages=["v3_flash_fp8"],
    package_dir={"v3_flash_fp8": str(HERE)},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
