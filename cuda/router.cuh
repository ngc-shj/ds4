/* =========================================================================
 * cuda/router.cuh - MoE router top-K selection.
 * =========================================================================
 *
 * Reference: ds4_metal.m::ds4_metal_router_select_batch_tensor and
 * metal/dsv4_misc.metal kernel_dsv4_router_finalize_one /
 * kernel_dsv4_router_weights_one.
 *
 * DS4 V4 Flash routes through 256 experts with top-6 selection per token.
 * Two modes:
 *   - normal: softmax(logits), pick top-K by (probs + bias), renormalise weights
 *   - hash:   token id -> precomputed 6-tuple from a hash table; uniform weights
 *
 * The engine restricts production to n_expert_groups=1 / n_group_used=0.
 */
#ifndef DS4_CUDA_ROUTER_CUH
#define DS4_CUDA_ROUTER_CUH

#include "common.cuh"
#include <float.h>

#define DS4_CUDA_N_EXPERTS 256
#define DS4_CUDA_TOP_K     6

/* One block per token, BLOCK threads. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_router_select_f32(
        int32_t *selected, float *weights, float *probs,
        const float *logits, const float *bias,
        const int32_t *hash, const int32_t *tokens,
        uint32_t n_tok, uint32_t hash_rows, int has_bias, int hash_mode) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;

    if (hash_mode) {
        if (threadIdx.x < DS4_CUDA_TOP_K) {
            const uint32_t row = (uint32_t)tokens[t] % hash_rows;
            selected[(uint64_t)t * DS4_CUDA_TOP_K + threadIdx.x] =
                hash[(uint64_t)row * DS4_CUDA_TOP_K + threadIdx.x];
            weights[(uint64_t)t * DS4_CUDA_TOP_K + threadIdx.x] = 1.0f / (float)DS4_CUDA_TOP_K;
        }
        for (uint32_t e = threadIdx.x; e < DS4_CUDA_N_EXPERTS; e += BLOCK) {
            probs[(uint64_t)t * DS4_CUDA_N_EXPERTS + e] = 0.0f;
        }
        return;
    }

    const float *lp = logits + (uint64_t)t * DS4_CUDA_N_EXPERTS;
    float       *pp = probs  + (uint64_t)t * DS4_CUDA_N_EXPERTS;

    /* Softmax over 256 logits. */
    float my_max = -FLT_MAX;
    for (uint32_t e = threadIdx.x; e < DS4_CUDA_N_EXPERTS; e += BLOCK) {
        if (lp[e] > my_max) my_max = lp[e];
    }
    __shared__ float wm[32];
    for (int off = 16; off > 0; off >>= 1)
        my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) wm[warp] = my_max;
    __syncthreads();
    if (warp == 0) {
        my_max = (threadIdx.x < (BLOCK + 31) / 32) ? wm[lane] : -FLT_MAX;
        for (int off = 16; off > 0; off >>= 1)
            my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
        if (lane == 0) wm[0] = my_max;
    }
    __syncthreads();
    const float lmax = wm[0];

    float my_sum = 0.0f;
    for (uint32_t e = threadIdx.x; e < DS4_CUDA_N_EXPERTS; e += BLOCK) {
        const float v = expf(lp[e] - lmax);
        pp[e] = v;
        my_sum += v;
    }
    __shared__ float ws[32];
    for (int off = 16; off > 0; off >>= 1) my_sum += __shfl_xor_sync(0xffffffff, my_sum, off);
    if (lane == 0) ws[warp] = my_sum;
    __syncthreads();
    if (warp == 0) {
        my_sum = (threadIdx.x < (BLOCK + 31) / 32) ? ws[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) my_sum += __shfl_xor_sync(0xffffffff, my_sum, off);
        if (lane == 0) ws[0] = my_sum;
    }
    __syncthreads();
    const float total = ws[0];
    const float inv_total = 1.0f / total;
    for (uint32_t e = threadIdx.x; e < DS4_CUDA_N_EXPERTS; e += BLOCK) {
        pp[e] *= inv_total;
    }
    __syncthreads();

    /* Top-K selection.  256 experts is small enough to do serially on thread 0;
     * the gain from a parallel top-K kernel here is dwarfed by the MoE matmul
     * that follows.  See PORT_CUDA.md tier 6 for the parallel variant. */
    if (threadIdx.x == 0) {
        /* Score = prob + bias (bias gates the choice but probs stay unbiased
         * for the weights).  We can't malloc 256 floats; use a hand-rolled
         * "remove best by setting to -inf" approach with a copy buffer.  Local
         * register array won't fit, so we use shared memory we already have. */
        /* Reuse wm/ws for scratch (they were size 32 each).  Need 256: stage
         * out to pp temporarily, knowing pp will be overwritten on the next
         * call anyway. */
        float score[DS4_CUDA_N_EXPERTS];
        for (int e = 0; e < DS4_CUDA_N_EXPERTS; e++) {
            score[e] = pp[e] + (has_bias ? bias[e] : 0.0f);
        }
        float w_sum = 0.0f;
        for (int k = 0; k < DS4_CUDA_TOP_K; k++) {
            int   best = 0;
            float bv   = -FLT_MAX;
            for (int e = 0; e < DS4_CUDA_N_EXPERTS; e++) {
                if (score[e] > bv) { bv = score[e]; best = e; }
            }
            selected[(uint64_t)t * DS4_CUDA_TOP_K + k] = best;
            const float wk = pp[best];
            weights[(uint64_t)t * DS4_CUDA_TOP_K + k] = wk;
            w_sum += wk;
            score[best] = -FLT_MAX;
        }
        const float inv = 1.0f / w_sum;
        for (int k = 0; k < DS4_CUDA_TOP_K; k++) {
            weights[(uint64_t)t * DS4_CUDA_TOP_K + k] *= inv;
        }
    }
}

#endif /* DS4_CUDA_ROUTER_CUH */
