// pybind11 binding for V4 NVFP4 (microscaled FP4) fused FlashAttention kernel.

#include <torch/extension.h>

torch::Tensor attention_flash_nvfp4_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V4 NVFP4 (microscaled FP4) fused FlashAttention with per-row "
              "per-K-block scaling (sm_120)";
    m.def(
        "attention_flash_nvfp4_cu",
        &attention_flash_nvfp4_cu,
        "NVFP4 (microscaled FP4) fused FlashAttention forward "
        "(Q, K, V, causal) -> O. Inputs are FP16; quantization to FP4 "
        "(E2M1) is performed inside the kernel with per-row per-K-block "
        "(block size 32) FP32 scales for Q, K, V, P. Uses kind::f8f6f4 "
        "m16n8k32 mma.sync.",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
