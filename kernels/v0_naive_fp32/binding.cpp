// pybind11 binding for V0 naive FP32 attention.

#include <torch/extension.h>

torch::Tensor attention_naive_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V0 naive FP32 scaled dot-product attention (sm_120)";
    m.def(
        "attention_naive_cu",
        &attention_naive_cu,
        "Naive FP32 attention forward (Q, K, V, causal) -> O",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
