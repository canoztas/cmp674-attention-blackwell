"""AOT build for the V4 NVFP4 (microscaled FP4) fused attention extension.

V4 uses the FP4 mma.sync instruction family (kind::f8f6f4 m16n8k32 with
e2m1 inputs), which requires the architecture-accelerated target
``sm_120a`` rather than plain ``sm_120``. ptxas rejects the kind::f8f6f4
mma instruction when targeting sm_120 and accepts it when targeting
sm_120a. The 'a' suffix denotes "Architecture-specific accelerated"
features (per NVIDIA's nvcc docs) and is required on consumer Blackwell
for the FP4/FP6 narrow-precision MMA family. Reference: Zhang et al.,
SageAttention3 (arXiv 2505.11594) -- closest related work, FP4 attention
on RTX 5090.

Boilerplate aside from the sm_120a flag matches V0..V3.
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
                "-gencode=arch=compute_120a,code=sm_120a",
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
