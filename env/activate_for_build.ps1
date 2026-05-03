# Build-environment activation for non-interactive PowerShell sessions.
#
# The user's interactive `gpu` PowerShell function (in their profile) handles
# the same job for terminal use, but non-interactive shells (e.g. those spawned
# by automation) don't auto-load profile functions, and they don't have conda /
# CUDA / Nsight Compute on PATH. Dot-source this script first in those shells:
#
#     . env\activate_for_build.ps1
#     python env\check_env.py
#     pip install -e kernels\v0_naive_fp32
#
# What it does, in order:
#   1. Sources Visual Studio 2022 Build Tools' vcvars64.bat into the current
#      shell, which puts cl.exe + the MSVC headers on PATH and INCLUDE.
#      Required for any PyTorch C++ / CUDA extension build on Windows.
#   2. Sets CUDA_HOME and prepends the CUDA Toolkit 12.8 bin directory to PATH
#      so nvcc.exe is discoverable. PyTorch's cpp_extension checks CUDA_HOME
#      first, before CUDA_PATH.
#   3. Prepends Nsight Compute 2025.1.1 to PATH so `ncu` resolves for kernel
#      profiling. This install lives outside the system-default PATH.
#   4. Activates the conda env named `gpu` (Anaconda3 install at the user's
#      profile root). After this, `python`, `pip`, etc. resolve to the gpu env.
#
# All four steps fail loudly if the underlying tool isn't where we expect it.

$ErrorActionPreference = "Stop"

# --- Paths (single source of truth) ---
$VcVars     = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$CudaHome   = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$NsightDir  = "C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.1"
$CondaExe   = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
$CondaEnv   = "gpu"

# --- 1. vcvars64.bat → current PowerShell environment ---
if (-not (Test-Path $VcVars)) { throw "vcvars64.bat not found at $VcVars" }
& cmd /c "`"$VcVars`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
# distutils refuses to re-source vcvars on top of an already-active shell
# unless this is set; without it `pip install -e` fails on Windows.
$env:DISTUTILS_USE_SDK = "1"

# --- 2. CUDA Toolkit 12.8 ---
if (-not (Test-Path $CudaHome)) { throw "CUDA Toolkit not found at $CudaHome" }
$env:CUDA_HOME = $CudaHome
$env:CUDA_PATH = $CudaHome
$env:PATH      = "$CudaHome\bin;$CudaHome\libnvvp;$env:PATH"

# --- 3. Nsight Compute ---
if (Test-Path $NsightDir) {
    $env:PATH = "$NsightDir;$env:PATH"
} else {
    Write-Warning "Nsight Compute not found at $NsightDir - ncu profiling will be unavailable."
}

# --- 4. Conda env activation ---
if (-not (Test-Path $CondaExe)) { throw "conda.exe not found at $CondaExe" }
(& $CondaExe "shell.powershell" "hook") | Out-String | Invoke-Expression
conda activate $CondaEnv
if ($env:CONDA_DEFAULT_ENV -ne $CondaEnv) {
    throw "Failed to activate conda env '$CondaEnv' (CONDA_DEFAULT_ENV='$env:CONDA_DEFAULT_ENV')"
}
