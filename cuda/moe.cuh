/* =========================================================================
 * cuda/moe.cuh - DS4 routed MoE: IQ2_XXS (gate/up) + Q2_K (down) dequant.
 * =========================================================================
 *
 * Reference: metal/moe.metal and ds4.c CPU reference (ds4_vec_dot_*).
 *
 * GGUF block layouts (must match metal/moe.metal and ds4.c struct
 * definitions exactly):
 *
 *   struct block_q2_K {
 *       uint8_t scales[16];   // 16 nibble pairs of dl/ml scale codes
 *       uint8_t qs[64];       // 256 2-bit quants, 4 elements per byte
 *       half    d;            // FP16 master scale
 *       half    dmin;         // FP16 master min
 *   };                        // 84 bytes per 256 elements
 *
 *   struct block_iq2_xxs {
 *       half     d;           // FP16 master scale
 *       uint16_t qs[32];      // 32 16-bit codes per 256 elements
 *   };                        // 66 bytes per 256 elements
 *
 * Both formats are reduced via the same matmul shape used elsewhere in the
 * backend: one block per (output_row, token), threads stripe across QK_K
 * blocks of the input dimension.
 *
 * The IQ2_XXS dequant uses two lookup tables (`iq2xxs_grid` and
 * `ksigns_iq2xs`) that must be present on the device.  We bake them in as
 * __constant__ arrays so per-thread lookups stay cached.
 */
#ifndef DS4_CUDA_MOE_CUH
#define DS4_CUDA_MOE_CUH

#include "common.cuh"

#define DS4_CUDA_QK_K 256

struct block_q2_K {
    uint8_t scales[DS4_CUDA_QK_K / 16];
    uint8_t qs[DS4_CUDA_QK_K / 4];
    __half  d;
    __half  dmin;
};

struct block_iq2_xxs {
    __half   d;
    uint16_t qs[DS4_CUDA_QK_K / 8];
};

/* =========================================================================
 * Lookup tables in __constant__ memory.
 *
 * iq2xxs_grid: 256 entries, each 8 packed bytes giving the magnitudes of
 *   eight 2-bit codes after the canonical sign/grid encoding.
 * ksigns_iq2xs: 128 sign masks, indexed by the 7-bit sign code in aux32_s.
 * Both arrays are kept in __constant__ so per-thread lookups stay cached.
 * ========================================================================= */

__device__ __constant__ uint8_t ds4_cuda_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

__device__ __constant__ uint8_t ds4_cuda_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

__device__ __constant__ uint64_t ds4_cuda_iq2xxs_grid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

/* Dequantise one Q2_K block and accumulate its dot product with x[0..255].
 * Reference: ggml's dequantize_row_q2_K (also mirrored in ds4.c CPU path).
 * Layout per block:
 *   for n in {0, 128}:                    (two 128-element halves)
 *       qs += 32 bytes per half
 *       for j in 0..3:                    (4 shifts: 0, 2, 4, 6)
 *           for two 16-element halves of this 32-byte group:
 *               sc = scales[is++]; dl = d*(sc & 0xF); ml = dmin*(sc >> 4)
 *               y[..16] = dl * ((qs[..16] >> shift) & 0x3) - ml */
__device__ __forceinline__ float ds4_cuda_q2k_block_dot(
        const block_q2_K *blk, const float *x) {
    const float d  = __half2float(blk->d);
    const float mn = __half2float(blk->dmin);
    const uint8_t *q = blk->qs;
    const uint8_t *sc = blk->scales;

    float acc = 0.0f;
    int is = 0;
    const float *xp = x;
    #pragma unroll
    for (int n = 0; n < 2; n++) {        /* two 128-element halves */
        int shift = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {    /* 4 shifts */
            uint8_t sc0 = sc[is++];
            float dl = d * (float)(sc0 & 0xF);
            float ml = mn * (float)(sc0 >> 4);
            #pragma unroll
            for (int l = 0; l < 16; l++)
                acc += xp[l] * (dl * (float)((q[l] >> shift) & 0x3) - ml);
            xp += 16;
            uint8_t sc1 = sc[is++];
            dl = d * (float)(sc1 & 0xF);
            ml = mn * (float)(sc1 >> 4);
            #pragma unroll
            for (int l = 0; l < 16; l++)
                acc += xp[l] * (dl * (float)((q[l + 16] >> shift) & 0x3) - ml);
            xp += 16;
            shift += 2;
        }
        q += 32;
    }
    return acc;
}

/* Dequantise one IQ2_XXS block and accumulate its dot product with x[0..255].
 * Each ib32 of 32 elements uses 4 grid-index bytes + 4 sign codes packed into
 * aux32_g/s, plus a per-ib32 4-bit scale modifier on top of the block's d. */
__device__ __forceinline__ float ds4_cuda_iq2xxs_block_dot(
        const block_iq2_xxs *blk, const float *x) {
    const float d = __half2float(blk->d);
    const uint16_t *q = blk->qs;
    float acc = 0.0f;
    int yi = 0;
    #pragma unroll
    for (int ib32 = 0; ib32 < 8; ib32++) {
        const uint32_t aux32_g = (uint32_t)q[4 * ib32 + 0] | ((uint32_t)q[4 * ib32 + 1] << 16);
        const uint32_t aux32_s = (uint32_t)q[4 * ib32 + 2] | ((uint32_t)q[4 * ib32 + 3] << 16);
        const float db = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            const uint8_t grid_idx = (uint8_t)(aux32_g >> (8 * l));
            const uint8_t *grid = (const uint8_t *)(&ds4_cuda_iq2xxs_grid[grid_idx]);
            const uint8_t signs = ds4_cuda_ksigns_iq2xs[(aux32_s >> (7 * l)) & 127];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const float sgn = (signs & ds4_cuda_kmask_iq2xs[j]) ? -1.0f : 1.0f;
                acc += x[yi++] * (db * (float)grid[j] * sgn);
            }
        }
    }
    return acc;
}

/* Q2_K matmul for one (token, output_row) per block: y[t, o] = sum_b Q2K_dot.
 * The expert id selects which weight row block we read. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_matmul_q2_k_expert_f32(
        float *y, const block_q2_K *w_all, const float *x,
        const int32_t *expert_ids,        /* [n_tok, K] -> which expert per slot */
        uint64_t expert_stride_blocks,    /* blocks per expert in the weight tensor */
        uint64_t in_dim, uint64_t out_dim, uint32_t K, uint32_t k_idx, uint32_t n_tok) {
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const int32_t e = expert_ids[tok * K + k_idx];

    const uint64_t blocks_per_row = in_dim / DS4_CUDA_QK_K;
    const block_q2_K *w_row = w_all
        + (uint64_t)e * expert_stride_blocks
        + row * blocks_per_row;
    const float *x_row = x + tok * in_dim;

    float partial = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        partial += ds4_cuda_q2k_block_dot(&w_row[b], x_row + (uint64_t)b * DS4_CUDA_QK_K);
    }
    __shared__ float warp_sums[32];
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        partial = (threadIdx.x < (BLOCK + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (lane == 0) {
            /* Caller does the weighted accumulate; we just write the dot. */
            y[(uint64_t)tok * out_dim + row] = partial;
        }
    }
}

/* IQ2_XXS matmul, same shape as Q2_K above. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_matmul_iq2xxs_expert_f32(
        float *y, const block_iq2_xxs *w_all, const float *x,
        const int32_t *expert_ids,
        uint64_t expert_stride_blocks,
        uint64_t in_dim, uint64_t out_dim, uint32_t K, uint32_t k_idx, uint32_t n_tok) {
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const int32_t e = expert_ids[tok * K + k_idx];

    const uint64_t blocks_per_row = in_dim / DS4_CUDA_QK_K;
    const block_iq2_xxs *w_row = w_all
        + (uint64_t)e * expert_stride_blocks
        + row * blocks_per_row;
    const float *x_row = x + tok * in_dim;

    float partial = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        partial += ds4_cuda_iq2xxs_block_dot(&w_row[b], x_row + (uint64_t)b * DS4_CUDA_QK_K);
    }
    __shared__ float warp_sums[32];
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        partial = (threadIdx.x < (BLOCK + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (lane == 0) {
            y[(uint64_t)tok * out_dim + row] = partial;
        }
    }
}

/* Fuse gate+up into mid: mid[t, i] = silu(clamp(gate[t,i])) * up[t,i] * w[t, k_idx].
 * One block per row, threads stripe across mid_dim. */
__global__ void ds4_cuda_kernel_moe_swiglu_weight_f32(
        float *mid, const float *gate, const float *up, const float *weights,
        uint32_t mid_dim, uint32_t n_tok, uint32_t K, uint32_t k_idx, float clamp) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= mid_dim || t >= n_tok) return;
    const float w = weights[t * K + k_idx];
    float g = gate[(uint64_t)t * mid_dim + i];
    float u = up  [(uint64_t)t * mid_dim + i];
    if (clamp > 1.0e-6f) {
        if (g >  clamp) g =  clamp;
        if (u >  clamp) u =  clamp;
        if (u < -clamp) u = -clamp;
    }
    const float silu = g / (1.0f + expf(-g));
    mid[(uint64_t)t * mid_dim + i] = silu * u * w;
}

/* Accumulate down projection result into out: out[t, o] += contrib[t, o]. */
__global__ void ds4_cuda_kernel_acc_f32(
        float *out, const float *contrib, uint32_t out_dim, uint32_t n_tok) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= out_dim || t >= n_tok) return;
    out[(uint64_t)t * out_dim + i] += contrib[(uint64_t)t * out_dim + i];
}

#endif /* DS4_CUDA_MOE_CUH */
