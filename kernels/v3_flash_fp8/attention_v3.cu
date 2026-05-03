// V3 FP8 (E4M3) fused FlashAttention-style attention kernel for sm_120.
//
// V3 inherits V2's algorithmic skeleton (online softmax with running (m, l),
// fused single-kernel Q@K^T -> softmax -> P@V, deferred normalization) and
// adds two contributions:
//
//   1) FP8 (E4M3) inputs with per-tile scaling. Q, K, V are loaded from HBM
//      as FP16 (binding contract matches V0/V1/V2), quantized in-kernel to
//      FP8 with one FP32 scale per tile (computed via absmax / 448, the FP8
//      E4M3 finite-max). P (post-softmax) is similarly quantized for the
//      second matmul. Scales are propagated through the matmuls by a single
//      multiply on the FP32 accumulator.
//
//   2) Register-resident O accumulator (V2's main perf compromise was an
//      SMEM-resident O accumulator; the per-iteration alpha rescale was done
//      in shared memory because WMMA fragment layout is implementation-
//      defined). With hand-rolled mma.sync the C/D fragment layout is fully
//      specified by the PTX ISA, so we can keep O in registers and apply the
//      alpha rescale in-place. This is the lever the V2 lessons flagged for
//      V3 (CLAUDE.md, "Lessons from V2 session", item on storing O in SMEM).
//
// Tooling: hand-rolled inline-PTX mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// (TN layout). This is the SM_89-introduced FP8 sync-mma instruction; PTX is
// forward-compatible to sm_120 (consumer Blackwell). Pattern follows CUTLASS
// 4.4.2's cute::SM89_16x8x32_F32E4M3E4M3F32_TN. CUTLASS itself was
// re-litigated (V2 lessons promised a V3 migration) and judged not worth the
// abstraction tax: CUTLASS v4.4.2 has no FMHA reference for sm_120a (the
// `77_blackwell_fmha` example is sm_100a/sm_103a only -- datacenter Blackwell,
// requires TMA which sm_120 lacks). Composing an FMHA from CollectiveBuilder
// primitives at the warp-specialized level is multi-week scope. Inline PTX
// keeps the kernel self-contained and the SASS directly inspectable for the
// paper's Tensor Core engagement claim.
//
// Tile shape inherits V2: BR = BC = 64, HEAD_DIM in {64, 128}. mma fragments
// are 16x8x32 (M=16, N=8, K=32) in FP8 -- the K-dim is 32 (vs 16 for V1/V2's
// FP16 mma) so each k iter consumes more of HEAD_DIM per cycle. With BR=64 and
// 4 warps per block, each warp owns 16 rows; for BC=64 each warp issues 8
// N-tiles of m16n8 per K-tile; HEAD_DIM/32 K-tiles per QK^T inner loop.
//
// SMEM budget at HEAD_DIM=128 (per block, dynamic single allocation):
//   Q_fp8     : BR * HEAD_DIM     =   8 KB   (persistent across kv loop)
//   K_fp8     : BC * HEAD_DIM     =   8 KB   (per kv tile)
//   V_fp8     : BC * HEAD_DIM     =   8 KB   (col-major layout, see below)
//   P_fp8     : BR * BC           =   4 KB   (post-softmax quantization)
//   S_fp32    : BR * BC * 4       =  16 KB   (softmax scratch + matmul spill)
//   total dynamic                    44 KB
// Plus static __shared__: alpha[BR], m[BR], l[BR], scale_{q,k,v,p}, scratch =~1 KB.
//
// 44 KB fits the default 48 KB per-block cap, but we still call
// cudaFuncSetAttribute to opt into the higher dynamic-SMEM cap explicitly --
// this matches V2's pattern and is forward-compatible if a future tile-shape
// experiment grows the budget.
//
// V layout: V is loaded row-major from HBM (matches V0/V1/V2 contract) but
// stored col-major in SMEM. Reason: the m16n8k32 mma is fixed at TN (A
// row-major, B col-major), and for P @ V we want B = V (no transpose). V's
// natural HBM layout has rows = N positions, cols = D positions; the col-
// major SMEM store puts D positions along columns and N positions along rows
// in the SMEM addressing, which matches the mma B fragment's expected access
// pattern. The transpose is folded into the cooperative load (each thread
// reads V_global[r, c] and writes V_smem_t[c * BC + r]).
//
// Per-tile scaling math:
//   Q_fp8 = round(Q_fp16 / scale_q),   scale_q = max(|Q_tile|) / 448
//   K_fp8 = round(K_fp16 / scale_k),   scale_k = max(|K_tile|) / 448
//   V_fp8 = round(V_fp16 / scale_v),   scale_v = max(|V_tile|) / 448
//   P_fp8 = round(P_fp32 / scale_p),   scale_p = max(P_tile) / 448
// (P is in [0, 1] after softmax so we use plain max, not absmax.) Dequant
// at the matmul output:
//   S_fp32 = (Q_fp8 @ K_fp8^T)_acc * scale_q * scale_k * sm_scale
//   O_inc  = (P_fp8 @ V_fp8)_acc   * scale_p * scale_v
// scales are computed once per tile (one FP32 scalar per BR*HEAD_DIM Q tile,
// per BC*HEAD_DIM K/V tile, per BR*BC P tile) and broadcast via SMEM.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

namespace {

constexpr int BR              = 64;
constexpr int BC              = 64;
constexpr int MMA_M           = 16;
constexpr int MMA_N           = 8;
constexpr int MMA_K           = 32;
constexpr int WARPS_PER_BLOCK = BR / MMA_M;            // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 128

// FP8 E4M3 finite max (single-bit-of-mantissa-saturation away from inf).
// 0b0111_1110 -> 1.75 * 2^8 = 448. The Micikevicius 2022 paper recommends
// per-tile scale = max(|x|) / 448 to map the tile's largest element to
// FP8's representable range. We clamp the scale at FP8_SCALE_EPS to avoid
// divide-by-zero on all-zero tiles (e.g. early KV iterations of a causally
// masked sequence) and clamp the quantized values to [-448, 448] to defend
// against rare overshoots when the tile's max is rounded.
constexpr float FP8_E4M3_MAX  = 448.0f;
constexpr float FP8_SCALE_EPS = 1.0e-30f;

// Warp-level reductions. Same butterfly pattern as V1/V2.
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

// Block-wide max across all THREADS_PER_BLOCK threads. Each warp produces
// its lane-broadcast max; warp 0 reduces across the WARPS_PER_BLOCK partial
// values. Returns the result on every thread of warp 0; non-warp-0 threads
// see undefined output but the caller only reads after a __syncthreads() that
// publishes via SMEM.
__device__ __forceinline__ float block_reduce_max_via_smem(
    float local_v, float* scratch /* size WARPS_PER_BLOCK */,
    int warp_id, int lane_id) {
    float warp_v = warp_reduce_max(local_v);
    if (lane_id == 0) scratch[warp_id] = warp_v;
    __syncthreads();
    float block_v = -CUDART_INF_F;
    if (warp_id == 0) {
        block_v = (lane_id < WARPS_PER_BLOCK) ? scratch[lane_id] : -CUDART_INF_F;
        block_v = warp_reduce_max(block_v);
    }
    return block_v;
}

// FP8 E4M3 quantize: clamp to [-448, 448] then convert. CUDA's
// __nv_cvt_float_to_fp8(.., __NV_SATFINITE, __NV_E4M3) does the saturation
// internally but we still clamp for symmetry and to keep the math explicit.
__device__ __forceinline__ __nv_fp8_storage_t f32_to_e4m3(float v) {
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// Inline-PTX wrapper for the FP8 m16n8k32 mma.sync. TN layout: A row-major
// (16x32 FP8 per MMA), B col-major (32x8 FP8). Accumulator is FP32 16x8.
// Per-thread fragments:
//   A: 4 b32  (= 16 FP8 = 16 bytes / thread; 32 threads -> 512 bytes = 16x32)
//   B: 2 b32  (=  8 FP8 =  8 bytes / thread; 32 threads -> 256 bytes = 32x8)
//   D: 4 f32  (=  4 floats = 16 bytes / thread; 32 threads -> 512 bytes = 16x8)
// Thread layout for A and D (per PTX ISA 8.x section 9.7.13.5): with
//   g = laneID >> 2, t = laneID & 3,
// A[0]: row=g,   cols=4t..4t+3       (4 FP8 packed in one b32, MSB->LSB col)
// A[1]: row=g+8, cols=4t..4t+3
// A[2]: row=g,   cols=4t+16..4t+19
// A[3]: row=g+8, cols=4t+16..4t+19
// B[0]: col=g,   rows=4t..4t+3
// B[1]: col=g,   rows=4t+16..4t+19
// D[0]: row=g,   col=2t
// D[1]: row=g,   col=2t+1
// D[2]: row=g+8, col=2t
// D[3]: row=g+8, col=2t+1
__device__ __forceinline__ void mma_m16n8k32_e4m3(
    float       (&d)[4],
    uint32_t const (&a)[4],
    uint32_t const (&b)[2],
    float const (&c)[4]) {
    asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]));
}

// Templated by HEAD_DIM so per-thread fragment counts and SMEM offsets are
// compile-time constants. Two instantiations cover the supported grid;
// matches V1/V2's contract.
template <int HEAD_DIM>
__global__ void flash_attention_fp8_kernel(
    const half* __restrict__ Q,   // (B, H, N, D)
    const half* __restrict__ K,   // (B, H, N, D)
    const half* __restrict__ V,   // (B, H, N, D)
    half* __restrict__ O,         // (B, H, N, D)
    int N, int H,
    float sm_scale,               // 1/sqrt(D) (passed in for clarity)
    bool causal) {

    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128,
                  "V3 only supports HEAD_DIM in {64, 128}");
    static_assert(BR == 64 && BC == 64,
                  "V3 tile geometry assumes BR = BC = 64");

    constexpr int FRAG_N_QK = BC / MMA_N;          // 8 N-tiles per warp for QK^T
    constexpr int FRAG_K_QK = HEAD_DIM / MMA_K;    // 2 (D=64) or 4 (D=128)
    constexpr int FRAG_N_PV = HEAD_DIM / MMA_N;    // 8 (D=64) or 16 (D=128)
    constexpr int FRAG_K_PV = BC / MMA_K;          // 2

    // Per-thread element counts for cooperative loads. All these divide evenly
    // because BR=BC=64 and HEAD_DIM in {64, 128} with THREADS_PER_BLOCK=128.
    constexpr int Q_LOAD_PER_THREAD  = (BR * HEAD_DIM) / THREADS_PER_BLOCK;
    constexpr int KV_LOAD_PER_THREAD = (BC * HEAD_DIM) / THREADS_PER_BLOCK;

    extern __shared__ unsigned char smem_raw[];
    __nv_fp8_storage_t* Q_smem = reinterpret_cast<__nv_fp8_storage_t*>(smem_raw);
    __nv_fp8_storage_t* K_smem = Q_smem + BR * HEAD_DIM;
    __nv_fp8_storage_t* V_smem = K_smem + BC * HEAD_DIM;        // col-major layout
    __nv_fp8_storage_t* P_smem = V_smem + BC * HEAD_DIM;
    float*              S_smem = reinterpret_cast<float*>(P_smem + BR * BC);

    // Per-row running stats + alpha broadcast buffer + per-tile scales +
    // 4-warp scratch for block reductions.
    __shared__ float m_smem[BR];
    __shared__ float l_smem[BR];
    __shared__ float alpha_smem[BR];
    __shared__ float scale_q;
    __shared__ float scale_k;
    __shared__ float scale_v;
    __shared__ float scale_p;
    __shared__ float warp_scratch[WARPS_PER_BLOCK];

    const int q_block     = blockIdx.x;
    const int bh          = blockIdx.y;
    const int b           = bh / H;
    const int h           = bh % H;
    const int q_row_start = q_block * BR;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int g       = lane_id >> 2;   // 0..7 (group within warp)
    const int t       = lane_id & 3;    // 0..3 (thread within group)
    const int my_row_base = warp_id * MMA_M;  // first of this warp's 16 rows

    const long long base_off =
        ((static_cast<long long>(b) * H) + h) * static_cast<long long>(N) * HEAD_DIM;
    const half* Q_ptr = Q + base_off;
    const half* K_ptr = K + base_off;
    const half* V_ptr = V + base_off;
    half*       O_ptr = O + base_off;

    // ---------- Load + quantize Q tile (persistent across kv loop) ----------
    // Each thread holds its slice in registers, computes a local absmax, the
    // block max-reduces, scale_q is published, then each thread quantizes its
    // own slice and writes FP8 to SMEM.
    half q_reg[Q_LOAD_PER_THREAD];
    float q_local_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < Q_LOAD_PER_THREAD; ++i) {
        const int idx = tid + i * THREADS_PER_BLOCK;
        const int r   = idx / HEAD_DIM;
        const int c   = idx % HEAD_DIM;
        const int gr  = q_row_start + r;
        q_reg[i] = (gr < N) ? Q_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
        const float v = __half2float(q_reg[i]);
        q_local_amax = fmaxf(q_local_amax, fabsf(v));
    }
    {
        float blk_max = block_reduce_max_via_smem(q_local_amax, warp_scratch, warp_id, lane_id);
        if (warp_id == 0 && lane_id == 0) {
            scale_q = fmaxf(blk_max / FP8_E4M3_MAX, FP8_SCALE_EPS);
        }
    }
    __syncthreads();

    const float inv_scale_q = 1.0f / scale_q;
    #pragma unroll
    for (int i = 0; i < Q_LOAD_PER_THREAD; ++i) {
        const int idx = tid + i * THREADS_PER_BLOCK;
        const float v_q = __half2float(q_reg[i]) * inv_scale_q;
        Q_smem[idx] = f32_to_e4m3(v_q);
    }
    // No __syncthreads() yet -- we still need to init O / m / l.

    // ---------- Init O (register-resident), m (= -inf), l (= 0) ----------
    // O is held in per-warp register fragments. One fragment per N-tile of
    // the output; each fragment is 4 FP32 per thread (m16n8 accumulator
    // layout). Initialized to 0 so the first iteration's `alpha * O_old + new`
    // recurrence starts cleanly with alpha=0 (m_old = -inf gives alpha = 0).
    float o_frag[FRAG_N_PV][4];
    #pragma unroll
    for (int n = 0; n < FRAG_N_PV; ++n) {
        #pragma unroll
        for (int e = 0; e < 4; ++e) o_frag[n][e] = 0.0f;
    }
    if (tid < BR) {
        m_smem[tid] = -CUDART_INF_F;
        l_smem[tid] = 0.0f;
    }
    __syncthreads();

    const int n_tiles = (N + BC - 1) / BC;
    for (int kv_iter = 0; kv_iter < n_tiles; ++kv_iter) {
        const int k_row_start = kv_iter * BC;

        // ----- Load + quantize K tile (one absmax block reduction) -----
        half k_reg[KV_LOAD_PER_THREAD];
        float k_local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const int r   = idx / HEAD_DIM;
            const int c   = idx % HEAD_DIM;
            const int gr  = k_row_start + r;
            k_reg[i] = (gr < N) ? K_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
            const float v = __half2float(k_reg[i]);
            k_local_amax = fmaxf(k_local_amax, fabsf(v));
        }
        {
            float blk_max = block_reduce_max_via_smem(k_local_amax, warp_scratch, warp_id, lane_id);
            if (warp_id == 0 && lane_id == 0) {
                scale_k = fmaxf(blk_max / FP8_E4M3_MAX, FP8_SCALE_EPS);
            }
        }
        __syncthreads();

        const float inv_scale_k = 1.0f / scale_k;
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const float v_k = __half2float(k_reg[i]) * inv_scale_k;
            K_smem[idx] = f32_to_e4m3(v_k);   // row-major (BC rows of D)
        }

        // ----- Load + quantize V tile, transposing to col-major ------
        half v_reg[KV_LOAD_PER_THREAD];
        float v_local_amax = 0.0f;
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const int r   = idx / HEAD_DIM;
            const int c   = idx % HEAD_DIM;
            const int gr  = k_row_start + r;
            v_reg[i] = (gr < N) ? V_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
            const float v = __half2float(v_reg[i]);
            v_local_amax = fmaxf(v_local_amax, fabsf(v));
        }
        {
            float blk_max = block_reduce_max_via_smem(v_local_amax, warp_scratch, warp_id, lane_id);
            if (warp_id == 0 && lane_id == 0) {
                scale_v = fmaxf(blk_max / FP8_E4M3_MAX, FP8_SCALE_EPS);
            }
        }
        __syncthreads();

        const float inv_scale_v = 1.0f / scale_v;
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const int r   = idx / HEAD_DIM;
            const int c   = idx % HEAD_DIM;
            const float v_v = __half2float(v_reg[i]) * inv_scale_v;
            // Col-major store: V_smem[c * BC + r] holds V[r, c]. This puts
            // the D axis along columns and the BC (= K-dim of P@V mma) along
            // rows in the SMEM addressing, matching the col-major B fragment
            // layout the m16n8k32 mma expects.
            V_smem[c * BC + r] = f32_to_e4m3(v_v);
        }
        __syncthreads();

        // ============== GEMM 1: S = (Q_fp8 @ K_fp8^T) * dequant ==============
        // QK^T in mma terms: A = Q (16x32 row-major per fragment), B = K^T
        // achieved by reading K row-major as col-major B with stride D --
        // same trick as V1's WMMA col_major K^T but written out as PTX.
        // Per-warp output: 16 rows x BC cols of S, in 8 N-tiles of m16n8.
        float s_frag[FRAG_N_QK][4];
        #pragma unroll
        for (int n = 0; n < FRAG_N_QK; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) s_frag[n][e] = 0.0f;
        }

        #pragma unroll
        for (int kk = 0; kk < FRAG_K_QK; ++kk) {
            const int k0 = kk * MMA_K;

            // Load A (Q tile) fragment: 4 b32 / thread.
            // Q row layout in SMEM: row-major, Q_smem[row * HEAD_DIM + col].
            uint32_t a_frag[4];
            const int qr0 = my_row_base + g;
            const int qr1 = my_row_base + g + 8;
            const int qc0 = k0 + 4 * t;
            const int qc1 = k0 + 4 * t + 16;
            a_frag[0] = *reinterpret_cast<const uint32_t*>(&Q_smem[qr0 * HEAD_DIM + qc0]);
            a_frag[1] = *reinterpret_cast<const uint32_t*>(&Q_smem[qr1 * HEAD_DIM + qc0]);
            a_frag[2] = *reinterpret_cast<const uint32_t*>(&Q_smem[qr0 * HEAD_DIM + qc1]);
            a_frag[3] = *reinterpret_cast<const uint32_t*>(&Q_smem[qr1 * HEAD_DIM + qc1]);

            #pragma unroll
            for (int n = 0; n < FRAG_N_QK; ++n) {
                const int n_base = n * MMA_N;
                // B (K^T) col-major view of K_smem (which is row-major).
                // K_smem[k_row * HEAD_DIM + k_col]; we want K^T[d, k_row] read
                // col-major with stride HEAD_DIM, ptr = K_smem + 0. For
                // fragment B (col=g_col, rows=k0+4t..k0+4t+3) the col index
                // (= original k_row) is n_base + g; the row index (= original
                // k_col) is k0 + 4t (resp. k0 + 4t + 16). So the address is
                //   K_smem[(n_base + g) * HEAD_DIM + (k0 + 4t)].
                const int kr  = n_base + g;
                const int kc0 = k0 + 4 * t;
                const int kc1 = k0 + 4 * t + 16;
                uint32_t b_frag[2];
                b_frag[0] = *reinterpret_cast<const uint32_t*>(&K_smem[kr * HEAD_DIM + kc0]);
                b_frag[1] = *reinterpret_cast<const uint32_t*>(&K_smem[kr * HEAD_DIM + kc1]);

                mma_m16n8k32_e4m3(s_frag[n], a_frag, b_frag, s_frag[n]);
            }
        }

        // Dequantize and apply softmax scale: S *= scale_q * scale_k * sm_scale.
        const float qk_dequant = scale_q * scale_k * sm_scale;
        #pragma unroll
        for (int n = 0; n < FRAG_N_QK; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) s_frag[n][e] *= qk_dequant;
        }

        // Spill s_frag -> S_smem (FP32 row-major BR x BC).
        // Accumulator layout: c[0]=row=g, col=2t; c[1]=row=g, col=2t+1;
        //                     c[2]=row=g+8, col=2t; c[3]=row=g+8, col=2t+1.
        #pragma unroll
        for (int n = 0; n < FRAG_N_QK; ++n) {
            const int n_base = n * MMA_N;
            const int row0 = my_row_base + g;
            const int row1 = my_row_base + g + 8;
            const int col0 = n_base + 2 * t;
            const int col1 = n_base + 2 * t + 1;
            S_smem[row0 * BC + col0] = s_frag[n][0];
            S_smem[row0 * BC + col1] = s_frag[n][1];
            S_smem[row1 * BC + col0] = s_frag[n][2];
            S_smem[row1 * BC + col1] = s_frag[n][3];
        }
        __syncthreads();

        // ----- Apply causal mask + N-padding mask, BEFORE the row max -----
        // Masking before max means -inf masked positions don't pollute the
        // running max statistic (the same pattern V1 / V2 use).
        for (int i = tid; i < BR * BC; i += THREADS_PER_BLOCK) {
            const int lr = i / BC;
            const int lc = i % BC;
            const int gr_global = q_row_start + lr;
            const int gc_global = k_row_start + lc;
            float v = S_smem[i];
            if (gr_global >= N || gc_global >= N || (causal && gc_global > gr_global)) {
                v = -CUDART_INF_F;
            }
            S_smem[i] = v;
        }
        __syncthreads();

        // ----- Online softmax row math (same shape as V2) -----
        // Each warp owns 16 rows of S_smem (a 16xBC slab); within a warp, all
        // 32 lanes cooperate on one row at a time, sweeping BC=64 columns in
        // stride-32 chunks. After this loop:
        //   S_smem holds P (= exp(S - m_new), still FP32)
        //   alpha_smem[row] holds the rescale factor for O
        //   m_smem[row], l_smem[row] are updated in place.
        #pragma unroll
        for (int r = 0; r < MMA_M; ++r) {
            const int row = my_row_base + r;

            float local_max = -CUDART_INF_F;
            for (int c = lane_id; c < BC; c += 32) {
                const float v = S_smem[row * BC + c];
                if (v > local_max) local_max = v;
            }
            local_max = warp_reduce_max(local_max);

            const float m_old = m_smem[row];
            const float m_new = (m_old > local_max) ? m_old : local_max;
            // Same isfinite guard as V2: when the entire row is masked (m_new
            // = -inf) alpha = exp(-inf - -inf) is NaN; we use 1.0 and rely on
            // p = 0 below to leave the row untouched. m_old = -inf with m_new
            // finite gives alpha = exp(-inf) = 0, correctly discarding the
            // (zero-initialized) prior O. Both corners covered.
            const float alpha = isfinite(m_new) ? __expf(m_old - m_new) : 1.0f;

            float local_sum = 0.0f;
            for (int c = lane_id; c < BC; c += 32) {
                const float v = S_smem[row * BC + c];
                const float p = isfinite(m_new) ? __expf(v - m_new) : 0.0f;
                S_smem[row * BC + c] = p;
                local_sum += p;
            }
            const float row_sum_p = warp_reduce_sum(local_sum);

            const float l_old = l_smem[row];
            const float l_new = alpha * l_old + row_sum_p;

            if (lane_id == 0) {
                m_smem[row]    = m_new;
                l_smem[row]    = l_new;
                alpha_smem[row] = alpha;
            }
        }
        __syncthreads();

        // ----- Apply alpha to register-resident O (V3's perf gain) ------
        // Each thread holds o_frag[n][0..3] for n in [0, FRAG_N_PV); these
        // are accumulator entries at (row=my_row_base + g, col=...) and
        // (row=my_row_base + g + 8, col=...). Multiply each entry by the
        // alpha for its row.
        const float alpha_g0 = alpha_smem[my_row_base + g];
        const float alpha_g1 = alpha_smem[my_row_base + g + 8];
        #pragma unroll
        for (int n = 0; n < FRAG_N_PV; ++n) {
            o_frag[n][0] *= alpha_g0;
            o_frag[n][1] *= alpha_g0;
            o_frag[n][2] *= alpha_g1;
            o_frag[n][3] *= alpha_g1;
        }

        // ----- Quantize P to FP8 (per-tile scale_p) -----
        // P is in [0, 1] after softmax (with the masked-row corner handled by
        // p = 0). max(P) here suffices; we still use absmax for symmetry and
        // to defend against any future change that lets P go signed.
        float p_local_amax = 0.0f;
        for (int i = tid; i < BR * BC; i += THREADS_PER_BLOCK) {
            const float p = S_smem[i];
            p_local_amax = fmaxf(p_local_amax, fabsf(p));
        }
        {
            float blk_max = block_reduce_max_via_smem(p_local_amax, warp_scratch, warp_id, lane_id);
            if (warp_id == 0 && lane_id == 0) {
                scale_p = fmaxf(blk_max / FP8_E4M3_MAX, FP8_SCALE_EPS);
            }
        }
        __syncthreads();

        const float inv_scale_p = 1.0f / scale_p;
        for (int i = tid; i < BR * BC; i += THREADS_PER_BLOCK) {
            const float p_q = S_smem[i] * inv_scale_p;
            P_smem[i] = f32_to_e4m3(p_q);
        }
        __syncthreads();

        // ============== GEMM 2: O += (P_fp8 @ V_fp8) * dequant ==============
        // P (BR x BC) row-major in P_smem; V col-major in V_smem (transposed
        // at load time). A in mma terms = P, B = V. Per-warp output: 16 rows
        // x HEAD_DIM cols of O, in FRAG_N_PV (= 8 or 16) N-tiles of m16n8.
        // K dim = BC = 64, so FRAG_K_PV = 2 inner k iterations.
        const float pv_dequant = scale_p * scale_v;
        #pragma unroll
        for (int kk = 0; kk < FRAG_K_PV; ++kk) {
            const int k0 = kk * MMA_K;

            // Load A (P) fragment from P_smem (row-major BR x BC).
            uint32_t a_frag[4];
            const int pr0 = my_row_base + g;
            const int pr1 = my_row_base + g + 8;
            const int pc0 = k0 + 4 * t;
            const int pc1 = k0 + 4 * t + 16;
            a_frag[0] = *reinterpret_cast<const uint32_t*>(&P_smem[pr0 * BC + pc0]);
            a_frag[1] = *reinterpret_cast<const uint32_t*>(&P_smem[pr1 * BC + pc0]);
            a_frag[2] = *reinterpret_cast<const uint32_t*>(&P_smem[pr0 * BC + pc1]);
            a_frag[3] = *reinterpret_cast<const uint32_t*>(&P_smem[pr1 * BC + pc1]);

            #pragma unroll
            for (int n = 0; n < FRAG_N_PV; ++n) {
                const int n_base = n * MMA_N;
                // B (V) col-major in V_smem, so V_smem[d_col * BC + k_row].
                // For fragment B at col=n_base+g (= d-axis pos), rows=k0+4t..,
                // address is V_smem[(n_base + g) * BC + (k0 + 4t)].
                const int dcol = n_base + g;
                const int kr0  = k0 + 4 * t;
                const int kr1  = k0 + 4 * t + 16;
                uint32_t b_frag[2];
                b_frag[0] = *reinterpret_cast<const uint32_t*>(&V_smem[dcol * BC + kr0]);
                b_frag[1] = *reinterpret_cast<const uint32_t*>(&V_smem[dcol * BC + kr1]);

                // Accumulate into a *temporary* fragment that we then dequant
                // and add to o_frag. Doing it in-place would mix scaled and
                // unscaled values in the accumulator across kk iterations.
                // Two iterations is small enough that the temp is cheap.
                float pv_inc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_m16n8k32_e4m3(pv_inc, a_frag, b_frag, pv_inc);
                // Dequant + accumulate. (We *could* fold the dequant into the
                // first kk iteration only and keep the accumulator scaled,
                // but the readability cost outweighs the saved muls.)
                o_frag[n][0] += pv_inc[0] * pv_dequant;
                o_frag[n][1] += pv_inc[1] * pv_dequant;
                o_frag[n][2] += pv_inc[2] * pv_dequant;
                o_frag[n][3] += pv_inc[3] * pv_dequant;
            }
        }
        // No __syncthreads() needed before next iter's K load: the next
        // iteration's first action is `for (i = tid; ... ) k_reg[i] = K_ptr[..]`
        // which writes registers, then a block-reduction with __syncthreads().
        // K_smem is overwritten only after the next __syncthreads() following
        // the K quantization loop, by which point the previous P@V mma is
        // complete on every thread (we left the GEMM 2 inner loop only after
        // every warp finished its k iterations). For safety we add an
        // explicit __syncthreads() here -- the cost is negligible relative
        // to the matmul and avoids depending on subtle scheduling.
        __syncthreads();
    }

    // =================== Epilogue: divide by l, write to HBM ===================
    // Each thread writes its 4-per-fragment x FRAG_N_PV output values to HBM.
    // Mapping: o_frag[n][0,1] -> (row = my_row_base + g, col = n*8 + 2t..2t+1)
    //          o_frag[n][2,3] -> (row = my_row_base + g + 8, col = ...)
    const int gr0 = q_row_start + my_row_base + g;
    const int gr1 = q_row_start + my_row_base + g + 8;
    const float l0 = l_smem[my_row_base + g];
    const float l1 = l_smem[my_row_base + g + 8];
    const float inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    #pragma unroll
    for (int n = 0; n < FRAG_N_PV; ++n) {
        const int col0 = n * MMA_N + 2 * t;
        const int col1 = n * MMA_N + 2 * t + 1;
        if (gr0 < N) {
            O_ptr[gr0 * HEAD_DIM + col0] = __float2half(o_frag[n][0] * inv_l0);
            O_ptr[gr0 * HEAD_DIM + col1] = __float2half(o_frag[n][1] * inv_l0);
        }
        if (gr1 < N) {
            O_ptr[gr1 * HEAD_DIM + col0] = __float2half(o_frag[n][2] * inv_l1);
            O_ptr[gr1 * HEAD_DIM + col1] = __float2half(o_frag[n][3] * inv_l1);
        }
    }
}

template <int HEAD_DIM>
constexpr size_t flash_smem_bytes() {
    constexpr size_t Q_BYTES = BR * HEAD_DIM * sizeof(__nv_fp8_storage_t);
    constexpr size_t K_BYTES = BC * HEAD_DIM * sizeof(__nv_fp8_storage_t);
    constexpr size_t V_BYTES = BC * HEAD_DIM * sizeof(__nv_fp8_storage_t);
    constexpr size_t P_BYTES = BR * BC       * sizeof(__nv_fp8_storage_t);
    constexpr size_t S_BYTES = BR * BC       * sizeof(float);
    return Q_BYTES + K_BYTES + V_BYTES + P_BYTES + S_BYTES;
}

template <int HEAD_DIM>
void launch_flash_fp8(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int N,
    float sm_scale, bool causal,
    cudaStream_t stream) {

    constexpr size_t smem_bytes = flash_smem_bytes<HEAD_DIM>();

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            (const void*)flash_attention_fp8_kernel<HEAD_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        attr_set = true;
    }

    dim3 grid((N + BR - 1) / BR, B * H);
    flash_attention_fp8_kernel<HEAD_DIM>
        <<<grid, THREADS_PER_BLOCK, smem_bytes, stream>>>(
            Q, K, V, O, N, H, sm_scale, causal);
}

}  // namespace


torch::Tensor attention_flash_fp8_cu(
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
                "V3 supports float16 inputs only (quantized to FP8 internally); got "
                + std::string(c10::toString(Q.scalar_type())));
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
                "V3 only supports head_dim in {64, 128}; got " + std::to_string(D));

    auto opts = Q.options();
    auto O = torch::empty({B, H, N, D}, opts);

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const half* qp = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* kp = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* vp = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half*       op = reinterpret_cast<half*>(O.data_ptr<at::Half>());

    if (D == 64) {
        launch_flash_fp8<64>(qp, kp, vp, op, B, H, N, sm_scale, causal, stream);
    } else {  // D == 128 (validated above)
        launch_flash_fp8<128>(qp, kp, vp, op, B, H, N, sm_scale, causal, stream);
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}
