// V4 NVFP4 (microscaled FP4) fused FlashAttention-style attention kernel
// for sm_120 (consumer Blackwell, RTX 5080).
//
// V4 inherits V3's algorithmic skeleton (online softmax with running (m, l),
// fused single-kernel Q@K^T -> softmax -> P@V, deferred normalization,
// register-resident O accumulator, V transposed col-major in SMEM) and adds:
//
//   1) FP4 (E2M1) inputs for Q, K, V, P with PER-ROW PER-K-BLOCK
//      microscaling (block size 32). Each row of Q/K/P gets one FP32 scale
//      per 32-element K-block; V gets one FP32 scale per d-position per
//      32-element N-block. The microscale block size is aligned with the
//      mma K-dim (32) so each mma call covers exactly one K-block, allowing
//      a single per-(row, col) scale multiply on the FP32 accumulator at the
//      end of each mma.
//
//      This is a coarser microscaling than the NVFP4 standard (block size
//      16, hardware-managed by the mxf4nvf4 mma family), but the standard's
//      per-thread fragment layout is undocumented in the kind::f8f6f4 family
//      and uses CuTe-style scattered-V mappings that require ldmatrix-style
//      swizzled SMEM loads. Block size 32 with software microscaling lets
//      us reuse V3's well-understood per-thread fragment access pattern
//      (4 b32 / thread for A, 2 b32 for B, FP4 in middle bits of each byte
//      container) and stay within the hand-rolled-PTX comfort zone V3
//      established. It is a defensible perf/granularity tradeoff that the
//      paper documents as a finding.
//
//   2) FP4 dynamic range. E2M1 has 4 mantissa bits effective (0, 0.5, 1,
//      1.5, 2, 3, 4, 6 finite positive values). With block-size-32
//      microscaling a single tile's outliers no longer pollute the whole
//      tile's quantization grid (V3's per-tile failure mode); each row
//      gets its own dynamic range allocation per K-block. Theoretical
//      throughput is 2x FP8 on Blackwell 5th-gen Tensor Cores.
//
// MMA: hand-rolled inline-PTX
//   mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
// (TN layout). Per CUTLASS 4.4.2's mma_traits_sm120.hpp comment (lines
// 211-225), the FP4 operands must be shifted LEFT by 2 bits because
// ld.matrix b4x16_p64 places FP4 data in the LOW 4 bits while the mma
// instruction expects them in the MIDDLE 4 bits of each 8-bit container:
// 0b0000ABCD -> 0b00ABCD00. CUDA's __nv_cvt_float_to_fp4 returns the value
// in the low 4 bits, so we shift after conversion.
//
// Tile shape inherits V3: BR = BC = 64, HEAD_DIM in {64, 128}. mma fragments
// remain 16x8x32 (M=16, N=8, K=32) -- same as V3's FP8 mma. The K-block
// (microscale block) size equals the mma K-dim so one mma call = one
// microscale block.
//
// SMEM budget at HEAD_DIM=128 (per block, dynamic single allocation):
//   Q_fp4     : BR * HEAD_DIM * 1 byte    =   8 KB   (1 byte / value, FP4 in middle bits)
//   K_fp4     : BC * HEAD_DIM * 1 byte    =   8 KB
//   V_fp4     : BC * HEAD_DIM * 1 byte    =   8 KB   (col-major layout)
//   P_fp4     : BR * BC * 1 byte          =   4 KB
//   S_fp32    : BR * BC * 4 bytes         =  16 KB   (softmax + matmul spill)
//   scale_q   : BR * (D/32) * 4 bytes     =   1 KB
//   scale_k   : BC * (D/32) * 4 bytes     =   1 KB
//   scale_v   : D  * (BC/32) * 4 bytes    =   1 KB
//   scale_p   : BR * (BC/32) * 4 bytes    =   0.5 KB
//   total dynamic                              ~48 KB
// Plus static __shared__: alpha[BR], m[BR], l[BR], warp_scratch =~1 KB.
//
// Fits the default 48 KB cap but we still call cudaFuncSetAttribute to opt
// into the higher dynamic-SMEM cap explicitly (matches V2/V3 pattern).
//
// V layout: V is loaded row-major from HBM (matches V0..V3 contract) but
// stored col-major in SMEM. Reason: the m16n8k32 mma is fixed at TN (A
// row-major, B col-major) and for P @ V we want B = V (no transpose). The
// transpose is folded into the cooperative load.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <math_constants.h>

namespace {

constexpr int BR              = 64;
constexpr int BC              = 64;
constexpr int MMA_M           = 16;
constexpr int MMA_N           = 8;
constexpr int MMA_K           = 32;
constexpr int KBLOCK          = 32;     // microscale block size = mma K dim
constexpr int WARPS_PER_BLOCK = BR / MMA_M;            // 4
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32; // 128

// FP4 E2M1 finite max. Representable positive values:
//   0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
// Max = 6.0. Per-row, per-K-block scale = max(|x_block|) / 6.
// Clamp at FP4_SCALE_EPS to avoid divide-by-zero on all-zero blocks.
constexpr float FP4_E2M1_MAX  = 6.0f;
constexpr float FP4_SCALE_EPS = 1.0e-30f;

// Warp-level reductions. Same butterfly pattern as V1/V2/V3.
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

// FP4 (E2M1) quantize: convert FP32 -> FP4, place in MIDDLE 4 bits of byte.
// CUDA's __nv_cvt_float_to_fp4 returns the FP4 value in the LOW 4 bits;
// kind::f8f6f4 mma expects MIDDLE 4 bits per byte. Shift left by 2.
__device__ __forceinline__ uint8_t f32_to_e2m1_middle(float v) {
    __nv_fp4_storage_t low = __nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
    return static_cast<uint8_t>(low) << 2;
}

// Inline-PTX wrapper for the FP4 m16n8k32 mma.sync (kind::f8f6f4 family).
// TN layout: A row-major (16x32 FP4-in-middle-bits per fragment), B col-major
// (32x8 FP4-in-middle-bits). Accumulator is FP32 16x8.
//
// Per-thread fragment access (PTX ISA section 9.7.13.5; identical layout to
// V3's FP8 m16n8k32 mma, since kind::f8f6f4 reuses the FP8-shape register
// format with FP4 occupying the middle 4 bits of each 8-bit slot):
//   With g = laneID >> 2, t = laneID & 3:
//     A[0]: row=g,   cols=4t..4t+3       (4 byte-containers in one b32)
//     A[1]: row=g+8, cols=4t..4t+3
//     A[2]: row=g,   cols=4t+16..4t+19
//     A[3]: row=g+8, cols=4t+16..4t+19
//     B[0]: col=g,   rows=4t..4t+3       (col-major K-direction)
//     B[1]: col=g,   rows=4t+16..4t+19
//     D[0]: row=g,   col=2t
//     D[1]: row=g,   col=2t+1
//     D[2]: row=g+8, col=2t
//     D[3]: row=g+8, col=2t+1
__device__ __forceinline__ void mma_m16n8k32_e2m1(
    float       (&d)[4],
    uint32_t const (&a)[4],
    uint32_t const (&b)[2],
    float const (&c)[4]) {
    asm("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),
           "r"(b[0]),  "r"(b[1]),
           "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3]));
}

// Templated by HEAD_DIM so per-thread fragment counts and SMEM offsets are
// compile-time constants. Two instantiations (64, 128) cover the supported
// grid; matches V1/V2/V3 contract.
template <int HEAD_DIM>
__global__ void flash_attention_fp4_kernel(
    const half* __restrict__ Q,   // (B, H, N, D)
    const half* __restrict__ K,   // (B, H, N, D)
    const half* __restrict__ V,   // (B, H, N, D)
    half* __restrict__ O,         // (B, H, N, D)
    int N, int H,
    float sm_scale,               // 1/sqrt(D) (passed in for clarity)
    bool causal) {

    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128,
                  "V4 only supports HEAD_DIM in {64, 128}");
    static_assert(BR == 64 && BC == 64,
                  "V4 tile geometry assumes BR = BC = 64");
    static_assert(KBLOCK == MMA_K,
                  "V4 microscale block size must equal mma K-dim for clean per-block scaling");

    constexpr int FRAG_N_QK     = BC / MMA_N;          // 8 N-tiles per warp for QK^T
    constexpr int FRAG_K_QK     = HEAD_DIM / MMA_K;    // 2 (D=64) or 4 (D=128) K-blocks
    constexpr int FRAG_N_PV     = HEAD_DIM / MMA_N;    // 8 (D=64) or 16 (D=128)
    constexpr int FRAG_K_PV     = BC / MMA_K;          // 2 K-blocks for PV
    constexpr int NB_QK         = FRAG_K_QK;           // # of K-blocks along D-dim
    constexpr int NB_PV         = FRAG_K_PV;           // # of K-blocks along BC-dim

    // Per-thread element counts for cooperative loads. All these divide
    // evenly because BR=BC=64 and HEAD_DIM in {64, 128} with THREADS=128.
    constexpr int Q_LOAD_PER_THREAD  = (BR * HEAD_DIM) / THREADS_PER_BLOCK;
    constexpr int KV_LOAD_PER_THREAD = (BC * HEAD_DIM) / THREADS_PER_BLOCK;

    extern __shared__ unsigned char smem_raw[];
    uint8_t* Q_smem = reinterpret_cast<uint8_t*>(smem_raw);
    uint8_t* K_smem = Q_smem + BR * HEAD_DIM;
    uint8_t* V_smem = K_smem + BC * HEAD_DIM;          // col-major layout
    uint8_t* P_smem = V_smem + BC * HEAD_DIM;
    float*   S_smem = reinterpret_cast<float*>(P_smem + BR * BC);
    float*   scale_q_smem = S_smem + BR * BC;          // [BR][NB_QK]
    float*   scale_k_smem = scale_q_smem + BR * NB_QK; // [BC][NB_QK]
    float*   scale_v_smem = scale_k_smem + BC * NB_QK; // [HEAD_DIM][NB_PV]
    float*   scale_p_smem = scale_v_smem + HEAD_DIM * NB_PV; // [BR][NB_PV]

    // Per-row running stats + alpha broadcast buffer + warp scratch.
    __shared__ float m_smem[BR];
    __shared__ float l_smem[BR];
    __shared__ float alpha_smem[BR];
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

    // ============== Load + microscale-quantize Q tile (persistent) ==============
    // V3-style load pattern: thread tid loads idx = tid + i*THREADS for
    // i in [0, Q_LOAD_PER_THREAD). Decoded for D=128:
    //   Each thread covers ONE column (col = tid) for ALL BR rows.
    //   Each warp covers 32 cols = exactly ONE K-block (kb = warp_id).
    //   Per-(row, kb) absmax = warp shfl across 32 lanes for one row, repeated BR times.
    // For D=64:
    //   Threads tid 0..63 cover even rows; 64..127 cover odd rows.
    //   Warps {0,2} cover kb=0, warps {1,3} cover kb=1. Each warp owns 32 rows
    //   (its row-set) for its single K-block. 32 reductions per warp.

    // The kb that this warp's columns fall into:
    constexpr int Q_KB_PER_WARP = (HEAD_DIM == 128) ? 1 : 1;  // each warp covers exactly 1 kb
    const int q_kb_for_warp     = (HEAD_DIM == 128) ? warp_id : (warp_id & 1);

    half  q_reg[Q_LOAD_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < Q_LOAD_PER_THREAD; ++i) {
        const int idx = tid + i * THREADS_PER_BLOCK;
        const int r   = idx / HEAD_DIM;
        const int c   = idx % HEAD_DIM;
        const int gr  = q_row_start + r;
        q_reg[i] = (gr < N) ? Q_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
    }

    // For each value held by this thread, identify its row. Then warp-reduce
    // absmax across lanes (which span the K-block in cols) for that row.
    // Lane 0 of each warp writes scale_q_smem[r * NB_QK + kb].
    #pragma unroll
    for (int i = 0; i < Q_LOAD_PER_THREAD; ++i) {
        const int idx = tid + i * THREADS_PER_BLOCK;
        const int r   = idx / HEAD_DIM;
        const float a = fabsf(__half2float(q_reg[i]));
        const float row_amax = warp_reduce_max(a);
        if (lane_id == 0) {
            scale_q_smem[r * NB_QK + q_kb_for_warp] =
                fmaxf(row_amax / FP4_E2M1_MAX, FP4_SCALE_EPS);
        }
    }
    __syncthreads();

    // Quantize Q to FP4 (with 4 bits in middle of each byte container).
    // Each value's scale is at scale_q_smem[r * NB_QK + kb], where kb is
    // determined by the column (col / KBLOCK).
    #pragma unroll
    for (int i = 0; i < Q_LOAD_PER_THREAD; ++i) {
        const int idx = tid + i * THREADS_PER_BLOCK;
        const int r   = idx / HEAD_DIM;
        const int c   = idx % HEAD_DIM;
        const int kb  = c / KBLOCK;
        const float scale = scale_q_smem[r * NB_QK + kb];
        const float vq = __half2float(q_reg[i]) / scale;
        Q_smem[idx] = f32_to_e2m1_middle(vq);
    }

    // ---------- Init O (register-resident), m (= -inf), l (= 0) ----------
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

        // ===== Load + microscale-quantize K tile =====
        // K shape (BC, D). Same load pattern as Q. Per-(K_row, kb_d) absmax
        // via warp shfl. K_smem is row-major (BC rows of D); the col-major
        // K^T view used by the QK mma reads through this layout.
        const int k_kb_for_warp = (HEAD_DIM == 128) ? warp_id : (warp_id & 1);

        half k_reg[KV_LOAD_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const int r   = idx / HEAD_DIM;
            const int c   = idx % HEAD_DIM;
            const int gr  = k_row_start + r;
            k_reg[i] = (gr < N) ? K_ptr[gr * HEAD_DIM + c] : __float2half(0.0f);
        }
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const int r   = idx / HEAD_DIM;
            const float a = fabsf(__half2float(k_reg[i]));
            const float row_amax = warp_reduce_max(a);
            if (lane_id == 0) {
                scale_k_smem[r * NB_QK + k_kb_for_warp] =
                    fmaxf(row_amax / FP4_E2M1_MAX, FP4_SCALE_EPS);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            const int idx = tid + i * THREADS_PER_BLOCK;
            const int r   = idx / HEAD_DIM;
            const int c   = idx % HEAD_DIM;
            const int kb  = c / KBLOCK;
            const float scale = scale_k_smem[r * NB_QK + kb];
            const float v_k = __half2float(k_reg[i]) / scale;
            K_smem[idx] = f32_to_e2m1_middle(v_k);   // row-major (BC rows of D)
        }

        // ===== Load + microscale-quantize V tile (col-major in SMEM) =====
        // For V we use a DIFFERENT load pattern so per-(d, kb_n) absmax is
        // single-thread (no cross-thread reduction). Each thread handles
        // ONE d-position for a fixed contiguous range of n's:
        //
        //   D=128: thread tid -> d = tid (in [0, 128)), n = 0..63 (all rows).
        //          NB_PV = BC/KBLOCK = 2 K-blocks per d.
        //          Each thread computes 2 scales (kb_n=0, kb_n=1) over its
        //          first 32 / second 32 n-values.
        //   D=64:  thread tid -> d = tid % 64, n_start = (tid/64) * 32.
        //          Each thread holds 32 n-values for a single (d, kb_n).
        //
        // Loaded value v_reg[i] for thread tid corresponds to:
        //   D=128: (n = i, d = tid)  for i in [0, 64)
        //   D=64:  (n = (tid/64)*32 + i, d = tid%64) for i in [0, 32)

        half v_reg[KV_LOAD_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            int n_pos, d_pos;
            if constexpr (HEAD_DIM == 128) {
                n_pos = i;
                d_pos = tid;
            } else {  // HEAD_DIM == 64
                n_pos = (tid >> 6) * 32 + i;
                d_pos = tid & 63;
            }
            const int gn = k_row_start + n_pos;
            v_reg[i] = (gn < N) ? V_ptr[gn * HEAD_DIM + d_pos] : __float2half(0.0f);
        }

        // Per-(d, kb_n) absmax. Each thread owns its own scales -- no shfl.
        if constexpr (HEAD_DIM == 128) {
            // Thread tid owns d=tid, all 64 n-values. Two scales per thread.
            float amax_kb0 = 0.0f, amax_kb1 = 0.0f;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                amax_kb0 = fmaxf(amax_kb0, fabsf(__half2float(v_reg[i])));
            }
            #pragma unroll
            for (int i = 32; i < 64; ++i) {
                amax_kb1 = fmaxf(amax_kb1, fabsf(__half2float(v_reg[i])));
            }
            scale_v_smem[tid * NB_PV + 0] = fmaxf(amax_kb0 / FP4_E2M1_MAX, FP4_SCALE_EPS);
            scale_v_smem[tid * NB_PV + 1] = fmaxf(amax_kb1 / FP4_E2M1_MAX, FP4_SCALE_EPS);
        } else {  // HEAD_DIM == 64
            // Thread tid owns (d=tid%64, kb_n=tid/64). One scale per thread.
            float amax = 0.0f;
            #pragma unroll
            for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
                amax = fmaxf(amax, fabsf(__half2float(v_reg[i])));
            }
            const int d_pos = tid & 63;
            const int kb_n  = tid >> 6;
            scale_v_smem[d_pos * NB_PV + kb_n] = fmaxf(amax / FP4_E2M1_MAX, FP4_SCALE_EPS);
        }
        __syncthreads();

        // Quantize V and write col-major: V_smem[d * BC + n] = quantize(V[n, d]).
        #pragma unroll
        for (int i = 0; i < KV_LOAD_PER_THREAD; ++i) {
            int n_pos, d_pos;
            if constexpr (HEAD_DIM == 128) {
                n_pos = i;
                d_pos = tid;
            } else {
                n_pos = (tid >> 6) * 32 + i;
                d_pos = tid & 63;
            }
            const int kb_n = n_pos / KBLOCK;
            const float scale = scale_v_smem[d_pos * NB_PV + kb_n];
            const float v_v = __half2float(v_reg[i]) / scale;
            V_smem[d_pos * BC + n_pos] = f32_to_e2m1_middle(v_v);
        }
        __syncthreads();

        // ============== GEMM 1: S = Q_fp4 @ K_fp4^T (with microscaling) ==============
        // QK^T: A = Q (16x32 row-major per fragment of FP4-in-middle-bits),
        // B = K^T achieved by reading K row-major as col-major B with stride D.
        // Per-warp output: 16 rows x BC cols of S, in 8 N-tiles of m16n8.
        //
        // For each K-block (kk in [0, FRAG_K_QK)):
        //   Load A frag for this K-block, scale_q at (rows g, g+8, kb=kk).
        //   For each n-tile:
        //     Load B frag, scale_k at (cols n*8+2t, n*8+2t+1, kb=kk).
        //     mma_inc <- mma(A, B, 0).
        //     Apply per-element scale: c[i] += scale_a[row(i)] * scale_b[col(i)] * sm_scale * mma_inc[i]
        //     Accumulate into s_frag[n] (FP32).
        float s_frag[FRAG_N_QK][4];
        #pragma unroll
        for (int n = 0; n < FRAG_N_QK; ++n) {
            #pragma unroll
            for (int e = 0; e < 4; ++e) s_frag[n][e] = 0.0f;
        }

        #pragma unroll
        for (int kk = 0; kk < FRAG_K_QK; ++kk) {
            const int k0 = kk * MMA_K;

            // Per-row scales for A (Q) at this K-block. Loaded once per kk
            // and reused across all FRAG_N_QK n-tiles.
            const float sa_g  = scale_q_smem[(my_row_base + g    ) * NB_QK + kk];
            const float sa_g8 = scale_q_smem[(my_row_base + g + 8) * NB_QK + kk];

            // Load A (Q tile) fragment: 4 b32 / thread.
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
                // Fragment B at col=g (the n-axis position = K-row in original)
                // and rows=k0+4t..k0+4t+3 (the K-axis = K-col in original).
                // Address: K_smem[(n_base + g) * HEAD_DIM + (k0 + 4t)].
                const int kr  = n_base + g;
                const int kc0 = k0 + 4 * t;
                const int kc1 = k0 + 4 * t + 16;
                uint32_t b_frag[2];
                b_frag[0] = *reinterpret_cast<const uint32_t*>(&K_smem[kr * HEAD_DIM + kc0]);
                b_frag[1] = *reinterpret_cast<const uint32_t*>(&K_smem[kr * HEAD_DIM + kc1]);

                // Per-col scales for B (K) at this K-block. K's "row in B's
                // col-major view" = K_row in original = the col of the
                // resulting S matrix.
                const float sb_2t  = scale_k_smem[(n_base + 2 * t    ) * NB_QK + kk];
                const float sb_2t1 = scale_k_smem[(n_base + 2 * t + 1) * NB_QK + kk];

                float qk_inc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_m16n8k32_e2m1(qk_inc, a_frag, b_frag, qk_inc);

                // Per-element accumulator dequant + softmax-scale fold-in.
                // Accumulator layout per V3:
                //   c[0]: row=g, col=2t      -> sa_g  * sb_2t
                //   c[1]: row=g, col=2t+1    -> sa_g  * sb_2t1
                //   c[2]: row=g+8, col=2t    -> sa_g8 * sb_2t
                //   c[3]: row=g+8, col=2t+1  -> sa_g8 * sb_2t1
                s_frag[n][0] += qk_inc[0] * (sa_g  * sb_2t  * sm_scale);
                s_frag[n][1] += qk_inc[1] * (sa_g  * sb_2t1 * sm_scale);
                s_frag[n][2] += qk_inc[2] * (sa_g8 * sb_2t  * sm_scale);
                s_frag[n][3] += qk_inc[3] * (sa_g8 * sb_2t1 * sm_scale);
            }
        }

        // Spill s_frag -> S_smem (FP32 row-major BR x BC).
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

        // ----- Apply causal + N-padding mask, BEFORE the row max -----
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

        // ----- Online softmax row math (same shape as V2/V3) -----
        // Each warp owns 16 rows of S_smem; within a warp, the 32 lanes
        // sweep BC=64 columns in stride-32 chunks. After this loop:
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

        // ----- Apply alpha to register-resident O (V3's perf gain, inherited) ------
        const float alpha_g0 = alpha_smem[my_row_base + g];
        const float alpha_g1 = alpha_smem[my_row_base + g + 8];
        #pragma unroll
        for (int n = 0; n < FRAG_N_PV; ++n) {
            o_frag[n][0] *= alpha_g0;
            o_frag[n][1] *= alpha_g0;
            o_frag[n][2] *= alpha_g1;
            o_frag[n][3] *= alpha_g1;
        }

        // ===== Microscale-quantize P to FP4 (per-row, per-K-block, block size 32) =====
        // P (BR x BC) is in [0, 1] post-softmax, sitting in S_smem (FP32).
        // We need NB_PV = BC/32 = 2 scales per row.
        //
        // Each warp owns BR/WARPS = 16 rows. Within a warp, do NB_PV
        // (= 2) per-row absmax reductions across 32 lanes (sweeping BC=64
        // cols in stride-32 chunks, then reducing).
        #pragma unroll
        for (int r = 0; r < MMA_M; ++r) {
            const int row = my_row_base + r;
            #pragma unroll
            for (int kb = 0; kb < NB_PV; ++kb) {
                const int c_lo = kb * KBLOCK;
                // 32 lanes cover 32 cols of the K-block in one sweep.
                const int c = c_lo + lane_id;
                const float a = fabsf(S_smem[row * BC + c]);
                const float blk_amax = warp_reduce_max(a);
                if (lane_id == 0) {
                    scale_p_smem[row * NB_PV + kb] =
                        fmaxf(blk_amax / FP4_E2M1_MAX, FP4_SCALE_EPS);
                }
            }
        }
        __syncthreads();

        // Quantize P to FP4 (cooperative across all threads).
        for (int i = tid; i < BR * BC; i += THREADS_PER_BLOCK) {
            const int r  = i / BC;
            const int c  = i % BC;
            const int kb = c / KBLOCK;
            const float scale = scale_p_smem[r * NB_PV + kb];
            const float p_q = S_smem[i] / scale;
            P_smem[i] = f32_to_e2m1_middle(p_q);
        }
        __syncthreads();

        // ============== GEMM 2: O += P_fp4 @ V_fp4  (with microscaling) ==============
        // P (BR x BC) row-major; V col-major in V_smem (transposed at load).
        // A = P, B = V. Per-warp output: 16 rows x HEAD_DIM cols of O,
        // in FRAG_N_PV (= 8 or 16) N-tiles of m16n8.
        // K dim = BC = 64 -> FRAG_K_PV = 2 K-blocks.
        #pragma unroll
        for (int kk = 0; kk < FRAG_K_PV; ++kk) {
            const int k0 = kk * MMA_K;

            const float sa_g  = scale_p_smem[(my_row_base + g    ) * NB_PV + kk];
            const float sa_g8 = scale_p_smem[(my_row_base + g + 8) * NB_PV + kk];

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
                // Fragment B at col=n_base+g (= d-axis pos), rows=k0+4t..,
                // address: V_smem[(n_base + g) * BC + (k0 + 4t)].
                const int dcol = n_base + g;
                const int kr0  = k0 + 4 * t;
                const int kr1  = k0 + 4 * t + 16;
                uint32_t b_frag[2];
                b_frag[0] = *reinterpret_cast<const uint32_t*>(&V_smem[dcol * BC + kr0]);
                b_frag[1] = *reinterpret_cast<const uint32_t*>(&V_smem[dcol * BC + kr1]);

                // V's microscale lookup. dcol is the d-position; the kk loop
                // index IS the kb_n (since FRAG_K_PV = NB_PV = BC/32).
                const float sb_2t  = scale_v_smem[(n_base + 2 * t    ) * NB_PV + kk];
                const float sb_2t1 = scale_v_smem[(n_base + 2 * t + 1) * NB_PV + kk];

                float pv_inc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                mma_m16n8k32_e2m1(pv_inc, a_frag, b_frag, pv_inc);

                // Per-element scale + accumulate directly into o_frag.
                o_frag[n][0] += pv_inc[0] * (sa_g  * sb_2t );
                o_frag[n][1] += pv_inc[1] * (sa_g  * sb_2t1);
                o_frag[n][2] += pv_inc[2] * (sa_g8 * sb_2t );
                o_frag[n][3] += pv_inc[3] * (sa_g8 * sb_2t1);
            }
        }
        // Sync before next iter overwrites SMEM.
        __syncthreads();
    }

    // =================== Epilogue: divide by l, write to HBM ===================
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
    constexpr size_t Q_BYTES = BR * HEAD_DIM * sizeof(uint8_t);
    constexpr size_t K_BYTES = BC * HEAD_DIM * sizeof(uint8_t);
    constexpr size_t V_BYTES = BC * HEAD_DIM * sizeof(uint8_t);
    constexpr size_t P_BYTES = BR * BC       * sizeof(uint8_t);
    constexpr size_t S_BYTES = BR * BC       * sizeof(float);
    constexpr size_t SCALE_Q_BYTES = BR * (HEAD_DIM / KBLOCK) * sizeof(float);
    constexpr size_t SCALE_K_BYTES = BC * (HEAD_DIM / KBLOCK) * sizeof(float);
    constexpr size_t SCALE_V_BYTES = HEAD_DIM * (BC / KBLOCK) * sizeof(float);
    constexpr size_t SCALE_P_BYTES = BR * (BC / KBLOCK) * sizeof(float);
    return Q_BYTES + K_BYTES + V_BYTES + P_BYTES + S_BYTES
         + SCALE_Q_BYTES + SCALE_K_BYTES + SCALE_V_BYTES + SCALE_P_BYTES;
}

template <int HEAD_DIM>
void launch_flash_fp4(
    const half* Q, const half* K, const half* V, half* O,
    int B, int H, int N,
    float sm_scale, bool causal,
    cudaStream_t stream) {

    constexpr size_t smem_bytes = flash_smem_bytes<HEAD_DIM>();

    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(
            (const void*)flash_attention_fp4_kernel<HEAD_DIM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        attr_set = true;
    }

    dim3 grid((N + BR - 1) / BR, B * H);
    flash_attention_fp4_kernel<HEAD_DIM>
        <<<grid, THREADS_PER_BLOCK, smem_bytes, stream>>>(
            Q, K, V, O, N, H, sm_scale, causal);
}

}  // namespace


torch::Tensor attention_flash_nvfp4_cu(
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
                "V4 supports float16 inputs only (quantized to FP4 internally); got "
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
                "V4 only supports head_dim in {64, 128}; got " + std::to_string(D));

    auto opts = Q.options();
    auto O = torch::empty({B, H, N, D}, opts);

    const float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const half* qp = reinterpret_cast<const half*>(Q.data_ptr<at::Half>());
    const half* kp = reinterpret_cast<const half*>(K.data_ptr<at::Half>());
    const half* vp = reinterpret_cast<const half*>(V.data_ptr<at::Half>());
    half*       op = reinterpret_cast<half*>(O.data_ptr<at::Half>());

    if (D == 64) {
        launch_flash_fp4<64>(qp, kp, vp, op, B, H, N, sm_scale, causal, stream);
    } else {  // D == 128 (validated above)
        launch_flash_fp4<128>(qp, kp, vp, op, B, H, N, sm_scale, causal, stream);
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}
