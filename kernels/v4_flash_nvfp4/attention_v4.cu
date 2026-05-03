// V4 NVFP4 (microscaled FP4) fused attention — stub.
//
// Implemented (or proven infeasible) in the V4 session. Exploratory:
// consumer-Blackwell NVFP4 tooling is still maturing and the variant may
// degrade to a case-study. Reference: SageAttention3 (arXiv 2505.11594) —
// FP4 attention on RTX 5090, closest prior art.

#include <torch/extension.h>

torch::Tensor attention_flash_nvfp4_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal) {
    TORCH_CHECK(false, "v4_flash_nvfp4: kernel not yet implemented (V4 session work)");
    return torch::Tensor();
}
