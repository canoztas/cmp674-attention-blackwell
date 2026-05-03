"""AOT build for the V4 NVFP4 (microscaled FP4) fused attention extension.

Stub: implemented in the V4 session, which is exploratory and may degrade to
a case-study if NVFP4 tooling on consumer Blackwell is too immature for an
end-to-end attention kernel. Reference: Zhang et al., SageAttention3
(arXiv 2505.11594) — closest related work, FP4 attention on RTX 5090.

Boilerplate only — sm_120 explicit, the Windows-only ``-DUSE_CUDA`` macro
for torch 2.9 + MSVC 14.40+ compatibility.
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
        name="v4_flash_nvfp4._C",
        sources=[
            str(HERE / "binding.cpp"),
            str(HERE / "attention_v4.cu"),
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
    name="v4_flash_nvfp4",
    version="0.0.1",
    description="V4 NVFP4 (microscaled FP4) fused attention for RTX 5080 (sm_120) [stub]",
    packages=["v4_flash_nvfp4"],
    package_dir={"v4_flash_nvfp4": str(HERE)},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
