"""AOT build for the V1 FP16 tiled (Tensor Core) attention extension.

Stub: the kernel itself is implemented in the V1 session. The setup.py is in
place so future sessions can ``pip install -e kernels/v1_tiled_fp16`` without
re-deriving the boilerplate. sm_120 is targeted explicitly; the Windows-only
``-DUSE_CUDA`` macro dodges torch 2.9 + MSVC 14.40+ ``C2872 'std' ambiguous``
in ``compiled_autograd.h`` (PyTorch PR #144707).

When V1 is implemented, replace the TORCH_CHECK(false, ...) body in
``attention_v1.cu`` and update this docstring.
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
        name="v1_tiled_fp16._C",
        sources=[
            str(HERE / "binding.cpp"),
            str(HERE / "attention_v1.cu"),
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
    name="v1_tiled_fp16",
    version="0.0.1",
    description="V1 FP16 tiled (Tensor Core) attention kernel for RTX 5080 (sm_120) [stub]",
    packages=["v1_tiled_fp16"],
    package_dir={"v1_tiled_fp16": str(HERE)},
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
