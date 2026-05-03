// V2 FP16 fused (FlashAttention-style) attention — stub.
//
// Implemented in the V2 session. Design target: online softmax + tile-wise
// fused QK^T / softmax / PV with no full (B, H, N, N) materialization in HBM.
// Reference: Dao 2022 (arXiv 2205.14135), Dao 2023 (arXiv 2307.08691).

#include <torch/extension.h>

torch::Tensor attention_flash_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal) {
    TORCH_CHECK(false, "v2_flash_fp16: kernel not yet implemented (V2 session work)");
    return torch::Tensor();
}
