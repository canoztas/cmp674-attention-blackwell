// V0 naive FP32 attention.
//
// Goal: a deliberately slow, transparently correct baseline. Three phases,
// three kernels, no Tensor Cores, full (B, H, N, N) materialization in HBM.
// The materialization is the whole point — it is the worst case the rest
// of the variant family is measured against, and it will be replaced by
// the online-softmax fusion in V2 (FlashAttention-2, Dao 2023, arXiv
// 2307.08691).
//
// Phase 1 (qk_scaled_kernel): S = (Q @ K^T) / sqrt(d), with optional causal
//   mask. One thread per element of S.
// Phase 2 (softmax_kernel): row-wise softmax over the last dim of S, in
//   place. One block per (batch, head, row); shared-memory reductions for
//   the row max and row sum.
// Phase 3 (pv_kernel): O = P @ V. One thread per element of O.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace {

constexpr int kSoftmaxThreads = 256;

__global__ void qk_scaled_kernel(
    const float* __restrict__ Q,   // (B, H, N, D)
    const float* __restrict__ K,   // (B, H, N, D)
    float* __restrict__ S,         // (B, H, N, N)
    int B, int H, int N, int D,
    float scale,
    bool causal) {
    const long long total = static_cast<long long>(B) * H * N * N;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const long long n_col = idx % N;
    const long long n_row = (idx / N) % N;
    const long long h = (idx / (static_cast<long long>(N) * N)) % H;
    const long long b = idx / (static_cast<long long>(H) * N * N);

    if (causal && n_col > n_row) {
        S[idx] = -CUDART_INF_F;
        return;
    }

    const long long q_off = (((b * H) + h) * N + n_row) * D;
    const long long k_off = (((b * H) + h) * N + n_col) * D;

    float acc = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < D; ++d) {
        acc += Q[q_off + d] * K[k_off + d];
    }
    S[idx] = acc * scale;
}

__global__ void softmax_kernel(
    float* __restrict__ S,         // (B, H, N, N), updated in place
    int B, int H, int N) {
    const int row = blockIdx.x;
    const int total_rows = B * H * N;
    if (row >= total_rows) return;

    float* row_ptr = S + static_cast<long long>(row) * N;
    const int tid = threadIdx.x;

    __shared__ float reduce[kSoftmaxThreads];

    float local_max = -CUDART_INF_F;
    for (int j = tid; j < N; j += blockDim.x) {
        const float v = row_ptr[j];
        if (v > local_max) local_max = v;
    }
    reduce[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other = reduce[tid + stride];
            if (other > reduce[tid]) reduce[tid] = other;
        }
        __syncthreads();
    }
    const float row_max = reduce[0];
    // Without this barrier, a fast thread reaches `reduce[tid] = local_sum`
    // below (overwriting reduce[0] when tid==0) before slow threads have
    // read reduce[0] as their row_max — non-deterministic ~6% mismatch rate.
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        // If row_max is -inf (entire row masked, e.g. impossible with our
        // mask but defensive), fall back to 0 to avoid NaN.
        const float e = isfinite(row_max) ? __expf(row_ptr[j] - row_max) : 0.0f;
        row_ptr[j] = e;
        local_sum += e;
    }
    reduce[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float row_sum = reduce[0];
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        row_ptr[j] *= inv_sum;
    }
}

__global__ void pv_kernel(
    const float* __restrict__ P,   // (B, H, N, N)
    const float* __restrict__ V,   // (B, H, N, D)
    float* __restrict__ O,         // (B, H, N, D)
    int B, int H, int N, int D) {
    const long long total = static_cast<long long>(B) * H * N * D;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const long long d = idx % D;
    const long long n = (idx / D) % N;
    const long long h = (idx / (static_cast<long long>(D) * N)) % H;
    const long long b = idx / (static_cast<long long>(H) * N * D);

    const long long p_off = (((b * H) + h) * N + n) * N;
    const long long v_off = ((b * H) + h) * N * D;

    float acc = 0.0f;
    for (int j = 0; j < N; ++j) {
        acc += P[p_off + j] * V[v_off + j * D + d];
    }
    O[idx] = acc;
}

}  // namespace


torch::Tensor attention_naive_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 && K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32,
                "V0 supports float32 only");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q, K, V must be 4-D (batch, num_heads, seq_len, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V must have identical shapes");

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int N = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));

    TORCH_CHECK(B > 0 && H > 0 && N > 0 && D > 0, "all dims must be positive");

    auto opts = Q.options();
    auto S = torch::empty({B, H, N, N}, opts);
    auto O = torch::empty({B, H, N, D}, opts);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    {
        const long long total = static_cast<long long>(B) * H * N * N;
        const int threads = 256;
        const long long blocks_ll = (total + threads - 1) / threads;
        TORCH_CHECK(blocks_ll <= static_cast<long long>(INT32_MAX),
                    "QK^T launch grid exceeds INT_MAX blocks; reduce config size");
        const int blocks = static_cast<int>(blocks_ll);
        qk_scaled_kernel<<<blocks, threads, 0, stream>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), S.data_ptr<float>(),
            B, H, N, D, scale, causal);
    }

    {
        const int blocks = B * H * N;
        softmax_kernel<<<blocks, kSoftmaxThreads, 0, stream>>>(
            S.data_ptr<float>(), B, H, N);
    }

    {
        const long long total = static_cast<long long>(B) * H * N * D;
        const int threads = 256;
        const long long blocks_ll = (total + threads - 1) / threads;
        TORCH_CHECK(blocks_ll <= static_cast<long long>(INT32_MAX),
                    "PV launch grid exceeds INT_MAX blocks; reduce config size");
        const int blocks = static_cast<int>(blocks_ll);
        pv_kernel<<<blocks, threads, 0, stream>>>(
            S.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(),
            B, H, N, D);
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}
