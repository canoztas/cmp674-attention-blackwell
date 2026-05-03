"""AOT build for the V2 FP16 fused (FlashAttention-style) attention extension.

sm_120 targeted explicitly; the Windows-only ``-DUSE_CUDA`` macro dodges the
torch 2.9 + MSVC 14.40+ ``C2872 'std' ambiguous`` clash in
``compiled_autograd.h`` (PyTorch PR #144707).

V2 implements fused QK -> online-softmax -> PV with no full (B, H, N, N)
materialization in HBM. Reference: Dao 2022 (arXiv 2205.14135) and Dao 2023
(arXiv 2307.08691) for FlashAttention and FlashAttention-2.
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
        name="v2_flash_fp16._C",
        sources=[
            str(HERE / "binding.cpp"),
            str(HERE / "attention_v2.cu"),
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
    name="v2_flash_fp16",
    version="0.0.1",
    description="V2 FP16 fused (FlashAttention-style) attention for RTX 5080 (sm_120) [stub]",
    packages=["v2_flash_fp16"],
    package_dir={"v2_flash_fp16": str(HERE)},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
