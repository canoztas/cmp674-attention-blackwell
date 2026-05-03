// V3 FP8 (E4M3) fused attention — stub.
//
// Implemented in the V3 session. Design target: V2 + FP8 (E4M3) inputs with
// per-tile scaling, FP32 accumulation. Reference: Micikevicius 2022
// (arXiv 2209.05433) for FP8 numerics. 5th-gen Tensor Cores on Blackwell
// natively support E4M3 / E5M2.

#include <torch/extension.h>

torch::Tensor attention_flash_fp8_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal) {
    TORCH_CHECK(false, "v3_flash_fp8: kernel not yet implemented (V3 session work)");
    return torch::Tensor();
}
