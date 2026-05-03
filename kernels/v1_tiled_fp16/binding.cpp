// pybind11 binding for V1 FP16 tiled attention (stub).

#include <torch/extension.h>

torch::Tensor attention_tiled_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "V1 FP16 tiled (Tensor Core) attention (sm_120) — stub";
    m.def(
        "attention_tiled_cu",
        &attention_tiled_cu,
        "FP16 tiled attention forward (Q, K, V, causal) -> O — stub",
        pybind11::arg("Q"),
        pybind11::arg("K"),
        pybind11::arg("V"),
        pybind11::arg("causal") = false);
}
