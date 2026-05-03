// pybind11 binding for V3 FP8 (E4M3) fused attention (stub).

#include <torch/extension.h>

torch::Tensor attention_flash_fp8_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V3 FP8 (E4M3) fused attention (sm_120) — stub";
    m.def(
        "attention_flash_fp8_cu",
        &attention_flash_fp8_cu,
        "FP8 (E4M3) fused attention forward (Q, K, V, causal) -> O — stub",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
