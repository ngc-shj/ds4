/* =========================================================================
 * cuda/q8_0.cuh - Q8_0 GGUF dequant + matmul for the DS4 CUDA backend.
 * =========================================================================
 *
 * Q8_0 block layout (must match metal/dense.metal and ds4_metal.m string):
 *
 *     struct block_q8_0 {
 *         half   d;          // FP16 per-block scale, 2 bytes
 *         int8_t qs[32];     // 32 quantised values, 32 bytes
 *     };                     // 34 bytes per block of 32 elements
 *
 * The matrix W is stored row-major: for each output row there are
 * (in_dim / 32) consecutive blocks.  This kernel is a straightforward
 * dequant-and-MAC: each block handles one output row, threads stripe across
 * the input dimension.  It is "correct but slow"; a tensor-core optimised
 * variant lives on the PORT_CUDA.md tier 1 list.
 */
#ifndef DS4_CUDA_Q8_0_CUH
#define DS4_CUDA_Q8_0_CUH

#include "common.cuh"

#define DS4_CUDA_QK8_0 32

struct block_q8_0 {
    __half d;
    int8_t qs[DS4_CUDA_QK8_0];
};

/* y[t, o] = sum over blocks b of (W_blk(o, b).d * sum_i (qs[i] * x[t, b*32+i])).
 * One block per (out_row, token).  Reduction with warp-shuffle + smem. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_matmul_q8_0_f32(
        float *y, const block_q8_0 *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;

    const uint64_t blocks_per_row = in_dim / DS4_CUDA_QK8_0;
    const block_q8_0 *w_row = w + row * blocks_per_row;
    const float      *x_row = x + tok * in_dim;

    float partial = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        const block_q8_0 *blk = &w_row[b];
        const float d = __half2float(blk->d);
        const float *xp = x_row + (uint64_t)b * DS4_CUDA_QK8_0;
        float sum = 0.0f;
        /* Unrolled 32-MAC.  The compiler will vectorise the loads. */
        #pragma unroll
        for (int i = 0; i < DS4_CUDA_QK8_0; i++) {
            sum += (float)blk->qs[i] * xp[i];
        }
        partial += sum * d;
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
            y[tok * out_dim + row] = partial;
        }
    }
}

/* Fused shared-FFN gate-up + SwiGLU.  Two Q8_0 weight rows (gate, up)
 * dotted with the same x, then mid = silu(gate) * up.  This is the
 * "shared" expert in DS4 - applied once per token per layer, not per
 * routed expert. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_shared_gate_up_swiglu_q8_0_f32(
        float *gate_out, float *up_out, float *mid_out,
        const block_q8_0 *w_gate, const block_q8_0 *w_up,
        const float *x,
        uint64_t in_dim, uint64_t out_dim, float clamp /*<=0 disables*/) {
    const uint64_t row = blockIdx.x;
    if (row >= out_dim) return;

    const uint64_t blocks_per_row = in_dim / DS4_CUDA_QK8_0;
    const block_q8_0 *gr = w_gate + row * blocks_per_row;
    const block_q8_0 *ur = w_up   + row * blocks_per_row;

    float pg = 0.0f, pu = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks_per_row; b += BLOCK) {
        const float dg = __half2float(gr[b].d);
        const float du = __half2float(ur[b].d);
        const float *xp = x + (uint64_t)b * DS4_CUDA_QK8_0;
        float sg = 0.0f, su = 0.0f;
        #pragma unroll
        for (int i = 0; i < DS4_CUDA_QK8_0; i++) {
            const float xi = xp[i];
            sg += (float)gr[b].qs[i] * xi;
            su += (float)ur[b].qs[i] * xi;
        }
        pg += sg * dg;
        pu += su * du;
    }

    __shared__ float warp_g[32], warp_u[32];
    for (int off = 16; off > 0; off >>= 1) {
        pg += __shfl_xor_sync(0xffffffff, pg, off);
        pu += __shfl_xor_sync(0xffffffff, pu, off);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_g[warp] = pg; warp_u[warp] = pu; }
    __syncthreads();
    if (warp == 0) {
        pg = (threadIdx.x < (BLOCK + 31) / 32) ? warp_g[lane] : 0.0f;
        pu = (threadIdx.x < (BLOCK + 31) / 32) ? warp_u[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            pg += __shfl_xor_sync(0xffffffff, pg, off);
            pu += __shfl_xor_sync(0xffffffff, pu, off);
        }
        if (lane == 0) {
            if (clamp > 0.0f) {
                if (pg >  clamp) pg =  clamp;
                if (pg < -clamp) pg = -clamp;
            }
            const float silu = pg / (1.0f + expf(-pg));
            if (gate_out) gate_out[row] = pg;
            if (up_out)   up_out[row]   = pu;
            mid_out[row] = silu * pu;
        }
    }
}

/* Attention-output LoRA stage 1: heads [n_tok, n_groups, group_dim]
 * times per-group W_a (Q8_0, shape [n_groups, rank, group_dim]) producing
 * low [n_tok, n_groups, rank].  One block per (rank, group, token); threads
 * stripe across the Q8_0 blocks of the group's input row. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_output_low_q8_f32(
        float *low, const block_q8_0 *w_a, const float *heads,
        uint32_t n_tok, uint32_t n_groups, uint32_t rank, uint32_t group_dim) {
    const uint32_t r = blockIdx.x;
    const uint32_t g = blockIdx.y;
    const uint32_t t = blockIdx.z;
    if (r >= rank || g >= n_groups || t >= n_tok) return;

    const uint32_t blocks = group_dim / DS4_CUDA_QK8_0;
    const block_q8_0 *wr = w_a + ((uint64_t)g * rank + r) * blocks;
    const float *hp = heads + ((uint64_t)t * n_groups + g) * group_dim;

    float partial = 0.0f;
    for (uint32_t b = threadIdx.x; b < blocks; b += BLOCK) {
        const float d = __half2float(wr[b].d);
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < DS4_CUDA_QK8_0; i++) {
            sum += (float)wr[b].qs[i] * hp[(uint64_t)b * DS4_CUDA_QK8_0 + i];
        }
        partial += sum * d;
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
        if (lane == 0) low[((uint64_t)t * n_groups + g) * rank + r] = partial;
    }
}

#endif /* DS4_CUDA_Q8_0_CUH */
