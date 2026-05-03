// V1 FP16 tiled (Tensor Core) attention — stub.
//
// Will be implemented in the V1 session: tiled GEMM via WMMA or CUTLASS,
// FP16 inputs, FP32 accumulation, materialized output (no online softmax —
// that's V2). Cite Dao 2023 (arXiv 2307.08691) for FP16 tile design.
//
// This stub compiles cleanly and raises at runtime, so the build pipeline
// (setup.py + binding.cpp) can be smoke-tested before kernel work begins.

#include <torch/extension.h>

torch::Tensor attention_tiled_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal) {
    TORCH_CHECK(false, "v1_tiled_fp16: kernel not yet implemented (V1 session work)");
    return torch::Tensor();
}
