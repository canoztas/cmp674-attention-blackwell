// V1 tiled FP16 attention with Tensor Cores (WMMA), FP32 accumulation,
// (B, H, N, N) attention matrix materialized in HBM.
//
// Goal of V1 (per CLAUDE.md): isolate the contribution of *tiling +
// Tensor Cores* relative to V0's naive baseline, before V2 introduces
// online softmax + single-kernel fusion. We deliberately keep V0's
// three-phase structure (QK^T -> softmax -> PV) so the only delta is
// "the matmul kernels are now WMMA-tiled" - exactly the ablation needed
// for the paper's methodology section. Cite Dao et al., FlashAttention-2
// (arXiv 2307.08691) for the FP16 tile design lineage.
//
// Decision log: WMMA over CUTLASS for V1 because (1) didactic clarity
// matters for the methodology section, (2) compile times are seconds vs
// minutes, (3) zero install risk on a brand-new sm_120 toolchain, and
// (4) V1 exercises *no* CUTLASS-specific features (CollectiveEpilogue
// pays off in V2+). If V2 wants CUTLASS, it can switch then; V1's WMMA
// path will become the "hand-rolled Tensor Core baseline" data point
// for the paper.
//
// Tile shapes:
//   BR = BC = BD = 64; WMMA fragments are 16x16x16 (FP16). Each block
//   handles a 64x64 output tile decomposed as a 4x4 grid of WMMA fragments.
//   4 warps/block (128 threads); each warp owns one row of 4 fragments.
//   This is a defensible default - smaller than the 128-tile sizes used
//   by FlashAttention-2 on H100 (RTX 5080 has less L2 than H100/RTX 5090),
//   and a multiple of 16 in every dim so the WMMA shape constraints are
//   satisfied without padding heroics. Tuning is deferred (CLAUDE.md).
//
// Padding strategy:
//   seq_len need not be a multiple of BR/BC. We initialize S with zeros
//   (torch::zeros) and zero-pad Q/K/V tile loads beyond row N. Softmax
//   reads only j < N, so out-of-N S regions are inert; PV iterates over
//   the full ceil(N/BC)*BC range, but P[r, n] for n >= N is zero and so
//   contributes nothing. Result: no special-case code, correct math.
//
// Numerical notes:
//   FP16 tensor inputs / outputs, FP32 accumulators in WMMA, FP32 in the
//   softmax (max + exp + sum), then cast back to FP16 when writing
//   probabilities. This matches the standard FA convention. With
//   normalized inputs at moderate seq_len, FP16 has enough range; the
//   correctness tests include an adversarial-magnitude grid to guard
//   against overflow in pathological inputs.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

namespace {

using namespace nvcuda::wmma;

// Tile shape. See file header for rationale.
constexpr int BR = 64;
constexpr int BC = 64;
constexpr int BD = 64;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int FRAG_M  = BR / WMMA_M;   // 4 fragment rows per output tile
constexpr int FRAG_N  = BC / WMMA_N;   // 4 fragment cols per output tile (QK)
constexpr int FRAG_BD = BD / WMMA_N;   // 4 fragment cols per output tile (PV)

constexpr int WARPS_PER_BLOCK   = FRAG_M;            // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

// Softmax kernel uses the same block size as V0; 256 is plenty for the
// row-wise reduction.
constexpr int kSoftmaxThreads = 256;

// ----------------------------- Phase 1: QK^T -------------------------- //
// Each block computes a BR x BC tile of S(b, h, :, :):
//   S[r:r+BR, c:c+BC] = (Q[r:r+BR, :] @ K[c:c+BC, :]^T) * scale
// with optional causal mask and -inf for j > i.
//
// Grid: (ceil(N/BR), ceil(N/BC), B*H).
// Dynamic shared memory layout (single allocation, hand-cast):
//   [ Q tile : BR*D halfs ][ K tile : BC*D halfs ][ scratch : BR*BC floats ]
//
// The Q/K tiles are loaded once per block, used across the inner D loop.
// Accumulators stay in registers across the D loop. After the matmul we
// store the FP32 accumulators into the shared scratch, sync, and then
// cooperatively scale + causal-mask + downcast + write to global S.
__global__ void qk_tile_kernel(
    const half* __restrict__ Q,   // (B, H, N, D)
    const half* __restrict__ K,   // (B, H, N, D)
    half* __restrict__ S,         // (B, H, N, N), zero-initialized
    int N, int D, int H,
    float scale,
    bool causal) {

    extern __shared__ unsigned char smem_raw[];
    half*  Q_smem  = reinterpret_cast<half*>(smem_raw);
    half*  K_smem  = Q_smem + BR * D;
    float* scratch = reinterpret_cast<float*>(K_smem + BC * D);

    const int row_block = blockIdx.x;
    const int col_block = blockIdx.y;
    const int bh        = blockIdx.z;
    const int b = bh / H;
    const int h = bh % H;
    const int q_row_start = row_block * BR;
    const int k_row_start = col_block * BC;

    const int warp_id = threadIdx.x >> 5;

    const long long qk_off = ((static_cast<long long>(b) * H) + h) * N * D;
    const long long s_off  = ((static_cast<long long>(b) * H) + h) * N * N;
    const half* Q_ptr = Q + qk_off;
    const half* K_ptr = K + qk_off;
    half*       S_ptr = S + s_off;

    // Cooperative tile loads, zero-padding beyond N. Q rows [q_row_start,
    // q_row_start+BR) and K rows [k_row_start, k_row_start+BC) are taken
    // verbatim; rows beyond N become 0 (zero contribution to the matmul).
    for (int i = threadIdx.x; i < BR * D; i += THREADS_PER_BLOCK) {
        const int r  = i / D;
        const int c  = i % D;
        const int gr = q_row_start + r;
        Q_smem[i] = (gr < N) ? Q_ptr[gr * D + c] : __float2half(0.0f);
    }
    for (int i = threadIdx.x; i < BC * D; i += THREADS_PER_BLOCK) {
        const int r  = i / D;
        const int c  = i % D;
        const int gr = k_row_start + r;
        K_smem[i] = (gr < N) ? K_ptr[gr * D + c] : __float2half(0.0f);
    }
    __syncthreads();

    // Each warp owns 16 rows of the BR x BC output tile (rows
    // [warp_id*16, warp_id*16+16)) and computes FRAG_N=4 horizontal
    // 16x16 fragments. Accumulators live in registers across the D loop.
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FRAG_N];
    #pragma unroll
    for (int n = 0; n < FRAG_N; ++n) fill_fragment(acc[n], 0.0f);

    // Inner contraction over the head_dim D, in chunks of WMMA_K=16.
    // For C = A @ B with C = Q @ K^T:
    //   A = Q[warp_id*16:warp_id*16+16, k0:k0+16]   row-major, ldm=D
    //   B = K^T[k0:k0+16, n*16:n*16+16]             col-major over K row-major
    //       ptr = K_smem + n*16*D + k0, ldm=D
    // The col-major B layout treats the row-major K as its transpose,
    // which is exactly what we need without an explicit transpose pass.
    for (int k0 = 0; k0 < D; k0 += WMMA_K) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a;
        load_matrix_sync(a, Q_smem + warp_id * WMMA_M * D + k0, D);

        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b;
            load_matrix_sync(b, K_smem + n * WMMA_N * D + k0, D);
            mma_sync(acc[n], a, b, acc[n]);
        }
    }

    // Spill the FP32 accumulators to shared so a cooperative pass can
    // apply the scale + causal mask + FP16 downcast + bounds check + global
    // write. WMMA fragment storage layout is implementation-defined, so
    // store_matrix_sync to shared is the portable way to access values
    // by (row, col).
    #pragma unroll
    for (int n = 0; n < FRAG_N; ++n) {
        store_matrix_sync(
            scratch + warp_id * WMMA_M * BC + n * WMMA_N,
            acc[n], BC, mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BR * BC; i += THREADS_PER_BLOCK) {
        const int lr = i / BC;
        const int lc = i % BC;
        const int gr = q_row_start + lr;
        const int gc = k_row_start + lc;
        if (gr >= N || gc >= N) continue;
        float v = scratch[i] * scale;
        if (causal && gc > gr) v = -CUDART_INF_F;
        S_ptr[gr * N + gc] = __float2half(v);
    }
}

// ----------------------------- Phase 2: softmax ------------------------- //
// Same algorithm as V0's softmax_kernel: row-wise max + exp(x - max) + sum.
// Reads/writes FP16, computes in FP32. One block per (B*H*N) row.
// The mid-loop __syncthreads() is the V0 lesson #6 readers-before-writers
// barrier - omitting it would let a fast warp overwrite reduce[0] before
// slow warps have read it as their row_max.
__global__ void softmax_fp16_kernel(half* __restrict__ S, int N) {
    const int row = blockIdx.x;
    half* row_ptr = S + static_cast<long long>(row) * N;
    const int tid = threadIdx.x;

    __shared__ float reduce[kSoftmaxThreads];

    float local_max = -CUDART_INF_F;
    for (int j = tid; j < N; j += blockDim.x) {
        const float v = __half2float(row_ptr[j]);
        if (v > local_max) local_max = v;
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            const float other = reduce[tid + s];
            if (other > reduce[tid]) reduce[tid] = other;
        }
        __syncthreads();
    }
    const float row_max = reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        const float e = isfinite(row_max)
            ? __expf(__half2float(row_ptr[j]) - row_max)
            : 0.0f;
        row_ptr[j] = __float2half(e);
        local_sum += e;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        __syncthreads();
    }
    const float row_sum = reduce[0];
    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    for (int j = tid; j < N; j += blockDim.x) {
        const float v = __half2float(row_ptr[j]) * inv;
        row_ptr[j] = __float2half(v);
    }
}

// ----------------------------- Phase 3: P @ V --------------------------- //
// Each block computes a BR x BD tile of O(b, h, :, :):
//   O[r:r+BR, d:d+BD] = sum over n of P[r:r+BR, n:n+BC] @ V[n:n+BC, d:d+BD]
//
// Grid: (ceil(N/BR), ceil(D/BD), B*H).
// Dynamic shared memory layout:
//   [ P tile : BR*BC halfs ][ V tile : BC*BD halfs ][ scratch : BR*BD floats ]
__global__ void pv_tile_kernel(
    const half* __restrict__ P,   // (B, H, N, N)
    const half* __restrict__ V,   // (B, H, N, D)
    half* __restrict__ O,         // (B, H, N, D)
    int N, int D, int H) {

    extern __shared__ unsigned char smem_raw[];
    half*  P_smem  = reinterpret_cast<half*>(smem_raw);
    half*  V_smem  = P_smem + BR * BC;
    float* scratch = reinterpret_cast<float*>(V_smem + BC * BD);

    const int row_block = blockIdx.x;
    const int col_block = blockIdx.y;
    const int bh        = blockIdx.z;
    const int b = bh / H;
    const int h = bh % H;
    const int o_row_start = row_block * BR;
    const int o_col_start = col_block * BD;

    const int warp_id = threadIdx.x >> 5;

    const long long p_off = ((static_cast<long long>(b) * H) + h) * N * N;
    const long long v_off = ((static_cast<long long>(b) * H) + h) * N * D;
    const half* P_ptr = P + p_off;
    const half* V_ptr = V + v_off;
    half*       O_ptr = O + v_off;  // O has same (B,H,N,D) shape as V

    // Accumulators stay in registers across the n_tile loop.
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FRAG_BD];
    #pragma unroll
    for (int d = 0; d < FRAG_BD; ++d) fill_fragment(acc[d], 0.0f);

    const int n_tiles = (N + BC - 1) / BC;
    for (int t = 0; t < n_tiles; ++t) {
        const int n_start = t * BC;

        // Load P_tile [BR, BC]; out-of-N elements are zero (padded P) so
        // they contribute zero to the matmul.
        for (int i = threadIdx.x; i < BR * BC; i += THREADS_PER_BLOCK) {
            const int r  = i / BC;
            const int c  = i % BC;
            const int gr = o_row_start + r;
            const int gc = n_start + c;
            P_smem[i] = (gr < N && gc < N) ? P_ptr[gr * N + gc]
                                           : __float2half(0.0f);
        }
        // Load V_tile [BC, BD]; out-of-N rows zero-padded.
        for (int i = threadIdx.x; i < BC * BD; i += THREADS_PER_BLOCK) {
            const int r  = i / BD;
            const int c  = i % BD;
            const int gr = n_start + r;
            const int gc = o_col_start + c;
            V_smem[i] = (gr < N && gc < D) ? V_ptr[gr * D + gc]
                                           : __float2half(0.0f);
        }
        __syncthreads();

        // C = A @ B with C = P_tile @ V_tile:
        //   A = P_tile[warp_id*16:warp_id*16+16, k0:k0+16]   row-major, ldm=BC
        //   B = V_tile[k0:k0+16, d*16:d*16+16]              row-major, ldm=BD
        for (int k0 = 0; k0 < BC; k0 += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a;
            load_matrix_sync(a, P_smem + warp_id * WMMA_M * BC + k0, BC);

            #pragma unroll
            for (int d = 0; d < FRAG_BD; ++d) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b;
                load_matrix_sync(b, V_smem + k0 * BD + d * WMMA_N, BD);
                mma_sync(acc[d], a, b, acc[d]);
            }
        }
        // Sync before reusing smem for the next n_tile load.
        __syncthreads();
    }

    #pragma unroll
    for (int d = 0; d < FRAG_BD; ++d) {
        store_matrix_sync(
            scratch + warp_id * WMMA_M * BD + d * WMMA_N,
            acc[d], BD, mem_row_major);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < BR * BD; i += THREADS_PER_BLOCK) {
        const int lr = i / BD;
        const int lc = i % BD;
        const int gr = o_row_start + lr;
        const int gc = o_col_start + lc;
        if (gr >= N || gc >= D) continue;
        O_ptr[gr * D + gc] = __float2half(scratch[i]);
    }
}

}  // namespace


torch::Tensor attention_tiled_cu(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    bool causal) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(),
                "Q, K, V must be CUDA tensors");
    TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous(),
                "Q, K, V must be contiguous");
    TORCH_CHECK(Q.dtype() == torch::kHalf && K.dtype() == torch::kHalf &&
                V.dtype() == torch::kHalf,
                "V1 supports float16 only (got "
                + std::string(c10::toString(Q.scalar_type())) + ")");
    TORCH_CHECK(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4,
                "Q, K, V must be 4-D (batch, num_heads, seq_len, head_dim)");
    TORCH_CHECK(Q.sizes() == K.sizes() && Q.sizes() == V.sizes(),
                "Q, K, V must have identical shapes");

    const int B = static_cast<int>(Q.size(0));
    const int H = static_cast<int>(Q.size(1));
    const int N = static_cast<int>(Q.size(2));
    const int D = static_cast<int>(Q.size(3));

    TORCH_CHECK(B > 0 && H > 0 && N > 0 && D > 0, "all dims must be positive");
    TORCH_CHECK(D == 64 || D == 128,
                "V1 only supports head_dim in {64, 128}; got " + std::to_string(D));

    auto opts = Q.options();
    auto S = torch::zeros({B, H, N, N}, opts);  // zero-init: see header comment
    auto O = torch::empty({B, H, N, D}, opts);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Phase 1: QK^T tiled GEMM with WMMA.
    {
        dim3 grid((N + BR - 1) / BR, (N + BC - 1) / BC, B * H);
        const size_t smem_bytes =
            (BR + BC) * D * sizeof(half) + BR * BC * sizeof(float);
        qk_tile_kernel<<<grid, THREADS_PER_BLOCK, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(Q.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(K.data_ptr<at::Half>()),
            reinterpret_cast<half*>(S.data_ptr<at::Half>()),
            N, D, H, scale, causal);
    }

    // Phase 2: row-wise softmax in FP32, FP16 storage.
    {
        const long long total_rows = static_cast<long long>(B) * H * N;
        TORCH_CHECK(total_rows <= static_cast<long long>(INT32_MAX),
                    "softmax launch grid exceeds INT_MAX rows");
        softmax_fp16_kernel<<<static_cast<int>(total_rows), kSoftmaxThreads, 0, stream>>>(
            reinterpret_cast<half*>(S.data_ptr<at::Half>()), N);
    }

    // Phase 3: P @ V tiled GEMM with WMMA.
    {
        dim3 grid((N + BR - 1) / BR, (D + BD - 1) / BD, B * H);
        const size_t smem_bytes =
            (BR * BC + BC * BD) * sizeof(half) + BR * BD * sizeof(float);
        pv_tile_kernel<<<grid, THREADS_PER_BLOCK, smem_bytes, stream>>>(
            reinterpret_cast<const half*>(S.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(V.data_ptr<at::Half>()),
            reinterpret_cast<half*>(O.data_ptr<at::Half>()),
            N, D, H);
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}
