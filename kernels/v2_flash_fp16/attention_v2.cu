// V2 FP16 fused (FlashAttention-style) attention: online softmax + tile-wise
// fused QK -> softmax -> PV with NO full (B, H, N, N) materialization in HBM.
//
// Algorithmic core of the project (CLAUDE.md). Where V1 kept the V0 three-
// phase structure (QK^T -> store S -> softmax in S -> store P -> PV -> store O)
// and isolated the speedup of "tiling + Tensor Cores", V2 introduces the two
// FlashAttention contributions:
//
//   1) Online softmax: maintain (m_i, l_i) running max + sum across K, V tile
//      iterations, never materializing the full softmax denominator.
//   2) Single-kernel fusion: each Q tile is loaded once and the K, V iteration
//      streams through it; the output O accumulator lives in shared memory
//      across iterations and is written to HBM once at the end.
//
// IO complexity drops from V1's O(N^2) HBM traffic for the S/P intermediates
// to O(N * D) -- no S, no P in HBM. This is what V2 buys; the matmul work
// itself is the same shape, same WMMA fragments.
//
// Reference: Dao et al. FlashAttention (arXiv 2205.14135), Algorithm 1.
// FlashAttention-2 (2307.08691) introduces partitioning improvements that
// are deferred (V3+).
//
// Decision log (re-litigated from V1):
//   - WMMA over CUTLASS again. Reasons that held in V1 still hold: V1's WMMA
//     scaffolding (col_major K^T trick, FP16 binding, sm_120 build) is
//     reusable verbatim; CUTLASS's CollectiveEpilogue advantage *exists* for
//     V2's fused softmax in principle, but the install + sm_120 verification
//     is multi-hour yak shaving in a session that's already algorithmically
//     dense. V3 (FP8) is the natural CUTLASS migration point because FP8
//     epilogue scaling is the actual reason to switch.
//   - Online softmax math runs in shared memory, not registers. Doing it in
//     registers would require knowing the WMMA accumulator fragment layout
//     (which is implementation-defined) to extract per-row indices for the
//     rescale step. The smem round-trip is small relative to the matmuls
//     and keeps the kernel verifiable.
//
// Tile shape: BR = BC = 64; WMMA fragments 16x16x16 (FP16). 4 warps per
// block (128 threads); each warp owns 16 rows of the BR=64 Q-tile. The
// output O accumulator lives in SMEM as FP32 [BR, HEAD_DIM]; running max
// and running sum live in SMEM as FP32 [BR]. This is the same default
// V1 settled on; tuning is V3+ work per CLAUDE.md.
//
// Shared-memory budget per block (dynamic, single allocation):
//   Q_smem  : BR * HEAD_DIM half     (persistent across the K, V loop)
//   O_smem  : BR * HEAD_DIM float    (persistent; final output accumulator)
//   KP_smem : max(BC*HEAD_DIM, BR*BC) half   (K during matmul, P after)
//   V_smem  : BC * HEAD_DIM half
//   S_smem  : BR * BC float          (scratch for QK^T spill + row math)
// Plus static __shared__ float m_smem[BR], l_smem[BR] (~512 B).
//
// At HEAD_DIM=64: 8 + 16 + 8 + 8 + 16 = 56 KB. At HEAD_DIM=128: 16 + 32 + 16
// + 16 + 16 = 96 KB. Both exceed the 48 KB default per-block SMEM cap on
// Blackwell, so the host launcher opts in via cudaFuncSetAttribute(...,
// cudaFuncAttributeMaxDynamicSharedMemorySize, ...). RTX 5080 sm_120
// supports up to ~99 KB dynamic SMEM per block. V1 lesson #6 flagged this.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <math_constants.h>

namespace {

using namespace nvcuda::wmma;

constexpr int BR = 64;
constexpr int BC = 64;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int FRAG_M = BR / WMMA_M;            // 4
constexpr int FRAG_N = BC / WMMA_N;            // 4
constexpr int WARPS_PER_BLOCK   = FRAG_M;      // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

// Warp-level reductions over a 32-lane mask. __shfl_xor_sync with butterfly
// pattern (16, 8, 4, 2, 1) reaches all-lanes consensus in 5 hops.
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, v, off);
        if (other > v) v = other;
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, off);
    }
    return v;
}

// Templated by HEAD_DIM so FRAG_BD and the SMEM-layout offsets are
// compile-time constants. Two instantiations (64, 128) cover the supported
// head_dim grid; matches V1's contract.
template <int HEAD_DIM>
__global__ void flash_attention_fp16_kernel(
    const half* __restrict__ Q,   // (B, H, N, D)
    const half* __restrict__ K,   // (B, H, N, D)
    const half* __restrict__ V,   // (B, H, N, D)
    half* __restrict__ O,         // (B, H, N, D)
    int N, int H,
    float scale,
    bool causal) {

    constexpr int FRAG_BD       = HEAD_DIM / WMMA_N;
    constexpr int Q_HALF_COUNT  = BR * HEAD_DIM;
    constexpr int O_FLOAT_COUNT = BR * HEAD_DIM;
    // KP_smem is aliased: holds K (BC*D halfs) during QK^T, then is
    // overwritten with P (BR*BC halfs) for the PV matmul. Allocate the max.
    constexpr int KP_HALF_COUNT = (BC * HEAD_DIM > BR * BC)
                                  ? (BC * HEAD_DIM) : (BR * BC);
    constexpr int V_HALF_COUNT  = BC * HEAD_DIM;
    // (S_FLOAT_COUNT = BR * BC, not used as a constant explicitly below.)

    extern __shared__ unsigned char smem_raw[];
    half*  Q_smem  = reinterpret_cast<half*>(smem_raw);
    float* O_smem  = reinterpret_cast<float*>(Q_smem + Q_HALF_COUNT);
    half*  KP_smem = reinterpret_cast<half*>(O_smem + O_FLOAT_COUNT);
    half*  V_smem  = KP_smem + KP_HALF_COUNT;
    float* S_smem  = reinterpret_cast<float*>(V_smem + V_HALF_COUNT);

    // Per-row running stats. Static SMEM (~512 B); negligible vs the dynamic
    // pool. Each warp only ever touches its own 16 rows, so no cross-warp
    // contention; the broader __syncthreads() barriers we already need for
    // the matmul SMEM are sufficient to publish lane-0 writes to other lanes.
    __shared__ float m_smem[BR];
    __shared__ float l_smem[BR];

    const int q_block = blockIdx.x;        // which Q tile
    const int bh      = blockIdx.y;        // which (batch, head) pair
    const int b = bh / H;
    const int h = bh % H;
    const int q_row_start = q_block * BR;

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const int my_row_base = warp_id * WMMA_M;  // first of warp's 16 rows

    const long long base_off =
        ((static_cast<long long>(b) * H) + h) * static_cast<long long>(N) * HEAD_DIM;
    const half* Q_ptr = Q + base_off;
    const half* K_ptr = K + base_off;
    const half* V_ptr = V + base_off;
    half*       O_ptr = O + base_off;

    // ---------- Load Q tile (persistent across the K, V loop) -------------
    // Rows beyond N are zero-padded; combined with the explicit -inf mask
    // applied to S below, padding contributes nothing to softmax or PV.
    for (int i = tid; i < BR * HEAD_DIM; i += THREADS_PER_BLOCK) {
        const int r  = i / HEAD_DIM;
        const int c  = i % HEAD_DIM;
        const int gr = q_row_start + r;
        Q_smem[i] = (gr < N) ? Q_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
    }
    // Zero the O accumulator (FP32) and init m, l. Online softmax requires
    // m_i = -inf and l_i = 0 at iteration t=0; m_i = 0 silently corrupts
    // outputs whenever the first tile contains negative scores, which is
    // every realistic input.
    for (int i = tid; i < BR * HEAD_DIM; i += THREADS_PER_BLOCK) {
        O_smem[i] = 0.0f;
    }
    if (tid < BR) {
        m_smem[tid] = -CUDART_INF_F;
        l_smem[tid] = 0.0f;
    }
    __syncthreads();

    const int n_tiles = (N + BC - 1) / BC;
    for (int t = 0; t < n_tiles; ++t) {
        const int k_row_start = t * BC;

        // ---------- Stage K, V tiles into SMEM ----------
        for (int i = tid; i < BC * HEAD_DIM; i += THREADS_PER_BLOCK) {
            const int r  = i / HEAD_DIM;
            const int c  = i % HEAD_DIM;
            const int gr = k_row_start + r;
            KP_smem[i] = (gr < N) ? K_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
        }
        for (int i = tid; i < BC * HEAD_DIM; i += THREADS_PER_BLOCK) {
            const int r  = i / HEAD_DIM;
            const int c  = i % HEAD_DIM;
            const int gr = k_row_start + r;
            V_smem[i] = (gr < N) ? V_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
        }
        __syncthreads();

        // ---------- Phase 1: QK^T into accumulator fragments ----------
        // Same col_major-on-row-major-K trick as V1: B = K^T without an
        // explicit transpose pass.
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[FRAG_N];
        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) fill_fragment(s_frag[n], 0.0f);

        for (int k0 = 0; k0 < HEAD_DIM; k0 += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a;
            load_matrix_sync(a, Q_smem + my_row_base * HEAD_DIM + k0, HEAD_DIM);

            #pragma unroll
            for (int n = 0; n < FRAG_N; ++n) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b;
                load_matrix_sync(b, KP_smem + n * WMMA_N * HEAD_DIM + k0, HEAD_DIM);
                mma_sync(s_frag[n], a, b, s_frag[n]);
            }
        }

        // Spill S (FP32 acc) to SMEM. WMMA fragment storage layout is
        // implementation-defined; store_matrix_sync gives us a portable view.
        #pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            store_matrix_sync(
                S_smem + my_row_base * BC + n * WMMA_N,
                s_frag[n], BC, mem_row_major);
        }
        __syncthreads();

        // Apply scale, causal mask, and N-padding mask in one cooperative pass.
        // Mask BEFORE max so masked positions can't influence the running max.
        for (int i = tid; i < BR * BC; i += THREADS_PER_BLOCK) {
            const int lr = i / BC;
            const int lc = i % BC;
            const int gr = q_row_start + lr;
            const int gc = k_row_start + lc;
            float v = S_smem[i] * scale;
            if (gr >= N || gc >= N || (causal && gc > gr)) {
                v = -CUDART_INF_F;
            }
            S_smem[i] = v;
        }
        __syncthreads();

        // ---------- Phase 2: per-warp online softmax row math ----------
        // Each warp owns 16 rows of S_smem (a 16xBC slab). Within a warp,
        // all 32 lanes cooperate on one row at a time, sweeping the BC=64
        // columns in stride-32 chunks and reducing via __shfl_xor_sync.
        // Sequential row loop is fine: 16 iterations of small warp reductions
        // is dominated by the matmuls in any sane regime.
        #pragma unroll
        for (int r = 0; r < WMMA_M; ++r) {
            const int row = my_row_base + r;

            // -- Row max --
            float local_max = -CUDART_INF_F;
            for (int c = lane_id; c < BC; c += 32) {
                float v = S_smem[row * BC + c];
                if (v > local_max) local_max = v;
            }
            local_max = warp_reduce_max(local_max);

            const float m_old = m_smem[row];
            const float m_new = (m_old > local_max) ? m_old : local_max;

            // alpha = exp(m_old - m_new). Two corners worth being explicit
            // about: (1) first iteration, m_old = -inf and m_new finite ->
            // alpha = exp(-inf) = 0, correctly discarding the (zero) prior O.
            // (2) entire tile masked out for this row -> m_new = -inf and
            // alpha = exp(-inf - -inf) = NaN under naive exp; we guard by
            // checking isfinite(m_new) and use alpha = 1 (don't touch O,
            // don't touch l). Combined with p=0 below, this leaves the row
            // untouched, which is the math we want.
            const float alpha = isfinite(m_new) ? __expf(m_old - m_new) : 1.0f;

            // -- P = exp(S - m_new); accumulate per-lane row sum --
            // Write P back into S_smem in place (FP32). It will be downcast
            // to FP16 in KP_smem after the row loop.
            float local_sum = 0.0f;
            for (int c = lane_id; c < BC; c += 32) {
                float v = S_smem[row * BC + c];
                float p = isfinite(m_new) ? __expf(v - m_new) : 0.0f;
                S_smem[row * BC + c] = p;
                local_sum += p;
            }
            const float row_sum_p = warp_reduce_sum(local_sum);

            // -- l_new = alpha * l_old + row_sum_p --
            const float l_old = l_smem[row];
            const float l_new = alpha * l_old + row_sum_p;

            // -- Rescale O[row, :] in place by alpha (the trick that lets us
            //    keep the accumulator un-normalized across iterations) --
            for (int d = lane_id; d < HEAD_DIM; d += 32) {
                O_smem[row * HEAD_DIM + d] *= alpha;
            }

            // Publish new (m, l) for this row. All 32 lanes have the same
            // values after the warp reductions; lane 0 wins.
            if (lane_id == 0) {
                m_smem[row] = m_new;
                l_smem[row] = l_new;
            }
        }
        __syncthreads();

        // ---------- Phase 3: P @ V into O accumulator ----------
        // First, downcast P (still in S_smem as FP32) to FP16 in KP_smem.
        // K from this iteration is no longer needed; KP_smem can be reused
        // (same dynamic SMEM region, capacity already sized for max(K, P)).
        for (int i = tid; i < BR * BC; i += THREADS_PER_BLOCK) {
            KP_smem[i] = __float2half(S_smem[i]);
        }
        __syncthreads();

        // The crucial fused-update step: load O_smem (already alpha-rescaled
        // above) into accumulator fragments, perform P @ V mma_sync, store
        // back. This realizes O_new = alpha*O_old + P @ V in registers.
        fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[FRAG_BD];
        #pragma unroll
        for (int d = 0; d < FRAG_BD; ++d) {
            load_matrix_sync(
                o_frag[d],
                O_smem + my_row_base * HEAD_DIM + d * WMMA_N,
                HEAD_DIM, mem_row_major);
        }

        for (int k0 = 0; k0 < BC; k0 += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a;
            load_matrix_sync(a, KP_smem + my_row_base * BC + k0, BC);

            #pragma unroll
            for (int d = 0; d < FRAG_BD; ++d) {
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b;
                load_matrix_sync(b, V_smem + k0 * HEAD_DIM + d * WMMA_N, HEAD_DIM);
                mma_sync(o_frag[d], a, b, o_frag[d]);
            }
        }

        #pragma unroll
        for (int d = 0; d < FRAG_BD; ++d) {
            store_matrix_sync(
                O_smem + my_row_base * HEAD_DIM + d * WMMA_N,
                o_frag[d], HEAD_DIM, mem_row_major);
        }
        __syncthreads();  // before next iter's K, V loads stomp KP, V smem
    }

    // ---------- Epilogue: deferred normalization, write to HBM ----------
    // Deferred normalization (divide by l only at the end) is both faster
    // (no per-iter division) and more numerically stable (l can be very
    // small mid-loop; the final l is a sum of positives that won't underflow).
    #pragma unroll
    for (int r = 0; r < WMMA_M; ++r) {
        const int row = my_row_base + r;
        const int gr  = q_row_start + row;
        if (gr >= N) continue;  // padded Q rows: skip the global write entirely
        const float l_final = l_smem[row];
        const float inv_l   = (l_final > 0.0f) ? (1.0f / l_final) : 0.0f;
        for (int d = lane_id; d < HEAD_DIM; d += 32) {
            const float o = O_smem[row * HEAD_DIM + d] * inv_l;
            O_ptr[gr * HEAD_DIM + d] = __float2half(o);
        }
    }
}

// Compute the dynamic SMEM size for one HEAD_DIM instantiation. Mirrors the
// __shared__ layout in the kernel exactly; if you change one, change the other.
template <int HEAD_DIM>
constexpr size_t flash_smem_bytes() {
    constexpr int Q_HALF  = BR * HEAD_DIM;
    constexpr int O_FLOAT = BR * HEAD_DIM;
    constexpr int KP_HALF = (BC * HEAD_DIM > BR * BC)
                            ? (BC * HEAD_DIM) : (BR * BC);
    constexpr int V_HALF  = BC * HEAD_DIM;
    constexpr int S_FLOAT = BR * BC;
    return Q_HALF  * sizeof(half)
         + O_FLOAT * sizeof(float)
         + KP_HALF * sizeof(half)
         + V_HALF  * sizeof(half)
         + S_FLOAT * sizeof(float);
}

template <int HEAD_DIM>
void launch_flash(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int N,
    float scale, bool causal,
    cudaStream_t stream) {

    constexpr size_t smem_bytes = flash_smem_bytes<HEAD_DIM>();
    // Opt in to the higher dynamic-SMEM cap. 96 KB at HEAD_DIM=128 is well
    // below the per-block ceiling on sm_120 (~99 KB) but above the 48 KB
    // default; without this call the launch silently fails.
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            (const void*)flash_attention_fp16_kernel<HEAD_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        attr_set = true;
    }

    dim3 grid((N + BR - 1) / BR, B * H);
    flash_attention_fp16_kernel<HEAD_DIM>
        <<<grid, THREADS_PER_BLOCK, smem_bytes, stream>>>(
            Q, K, V, O, N, H, scale, causal);
}

}  // namespace


torch::Tensor attention_flash_cu(
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
                "V2 supports float16 only (got "
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
                "V2 only supports head_dim in {64, 128}; got " + std::to_string(D));

    auto opts = Q.options();
    auto O = torch::empty({B, H, N, D}, opts);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const half* qp = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* kp = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* vp = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half*       op = reinterpret_cast<half*>(O.data_ptr<at::Half>());

    if (D == 64) {
        launch_flash<64>(qp, kp, vp, op, B, H, N, scale, causal, stream);
    } else {  // D == 128 (validated above)
        launch_flash<128>(qp, kp, vp, op, B, H, N, scale, causal, stream);
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}
