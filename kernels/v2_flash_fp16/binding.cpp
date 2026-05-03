// pybind11 binding for V2 FP16 fused (FlashAttention-style) attention.

#include <torch/extension.h>

torch::Tensor attention_flash_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V2 FP16 fused (FlashAttention-style) attention (sm_120)";
    m.def(
        "attention_flash_cu",
        &attention_flash_cu,
        "FP16 fused FlashAttention forward (Q, K, V, causal) -> O",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
