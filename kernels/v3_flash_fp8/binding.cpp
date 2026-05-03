// pybind11 binding for V3 FP8 (E4M3) fused FlashAttention kernel.

#include <torch/extension.h>

torch::Tensor attention_flash_fp8_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V3 FP8 (E4M3) fused FlashAttention with per-tile scaling (sm_120)";
    m.def(
        "attention_flash_fp8_cu",
        &attention_flash_fp8_cu,
        "FP8 (E4M3) fused FlashAttention forward (Q, K, V, causal) -> O. "
        "Inputs are FP16; quantization to FP8 is performed inside the kernel "
        "with one FP32 scale per tile (Q, K, V, P).",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
