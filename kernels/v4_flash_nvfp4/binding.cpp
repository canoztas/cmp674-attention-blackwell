// pybind11 binding for V4 NVFP4 (microscaled FP4) fused attention (stub).

#include <torch/extension.h>

torch::Tensor attention_flash_nvfp4_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V4 NVFP4 (microscaled FP4) fused attention (sm_120) — stub";
    m.def(
        "attention_flash_nvfp4_cu",
        &attention_flash_nvfp4_cu,
        "NVFP4 (microscaled FP4) fused attention forward (Q, K, V, causal) -> O — stub",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
