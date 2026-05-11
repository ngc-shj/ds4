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

/* Softplus with the same saturation Metal uses (>20 collapses to the input
 * to avoid overflow in exp()). */
__device__ __forceinline__ float ds4_cuda_softplus(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

/* DS4 router probability transform.  Matches metal kernel_dsv4_softplus_sqrt:
 *   probs[e] = sqrt(softplus(logits[e]))
 * which is NOT a softmax distribution.  Top-K is then taken by
 * (probs + bias) (bias is a learned per-expert scalar that influences
 * selection only) and the surviving weights are renormalised by their own
 * sum and multiplied by 1.5 (DS4's expert-weight scale). */
template <int BLOCK>
__global__ void ds4_cuda_kernel_router_select_f32(
        int32_t *selected, float *weights, float *probs,
        const float *logits, const float *bias,
        const int32_t *hash, const int32_t *tokens,
        uint32_t n_tok, uint32_t hash_rows, int has_bias, int hash_mode) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;

    const float *lp = logits + (uint64_t)t * DS4_CUDA_N_EXPERTS;
    float       *pp = probs  + (uint64_t)t * DS4_CUDA_N_EXPERTS;

    /* Phase 1 (both modes): probs = sqrt(softplus(logits)).  Hash mode also
     * needs the unbiased probs to compute the final route weights. */
    for (uint32_t e = threadIdx.x; e < DS4_CUDA_N_EXPERTS; e += BLOCK) {
        pp[e] = sqrtf(ds4_cuda_softplus(lp[e]));
    }
    __syncthreads();

    if (hash_mode) {
        /* Selected experts come from the hash table indexed by token id. */
        if (threadIdx.x < DS4_CUDA_TOP_K) {
            const uint32_t row = (uint32_t)tokens[t] % hash_rows;
            selected[(uint64_t)t * DS4_CUDA_TOP_K + threadIdx.x] =
                hash[(uint64_t)row * DS4_CUDA_TOP_K + threadIdx.x];
        }
        __syncthreads();
        /* Weights[k] = probs[sel[k]] / max(sum, eps) * 1.5 -- same normalisation
         * as top-K mode (see layer_hash_router_weights_from_probs in ds4.c). */
        if (threadIdx.x == 0) {
            int32_t sel[DS4_CUDA_TOP_K];
            float w_sum = 0.0f;
            for (int k = 0; k < DS4_CUDA_TOP_K; k++) {
                sel[k] = selected[(uint64_t)t * DS4_CUDA_TOP_K + k];
                w_sum += pp[sel[k]];
            }
            const float denom = fmaxf(w_sum, 6.103515625e-5f);
            const float inv = 1.5f / denom;
            for (int k = 0; k < DS4_CUDA_TOP_K; k++) {
                weights[(uint64_t)t * DS4_CUDA_TOP_K + k] = pp[sel[k]] * inv;
            }
        }
        return;
    }

    /* Top-K mode: select top-K by (probs + bias).  Serial on thread 0; 256
     * experts make this cheap relative to the MoE matmul that follows. */
    if (threadIdx.x == 0) {
        float scratch[DS4_CUDA_N_EXPERTS];
        for (int e = 0; e < DS4_CUDA_N_EXPERTS; e++) {
            scratch[e] = pp[e] + (has_bias ? bias[e] : 0.0f);
        }
        int32_t sel[DS4_CUDA_TOP_K];
        for (int k = 0; k < DS4_CUDA_TOP_K; k++) {
            int   best = 0;
            float bv   = -FLT_MAX;
            for (int e = 0; e < DS4_CUDA_N_EXPERTS; e++) {
                if (scratch[e] > bv) { bv = scratch[e]; best = e; }
            }
            sel[k] = best;
            selected[(uint64_t)t * DS4_CUDA_TOP_K + k] = best;
            scratch[best] = -FLT_MAX;
        }
        float w_sum = 0.0f;
        for (int k = 0; k < DS4_CUDA_TOP_K; k++) w_sum += pp[sel[k]];
        const float denom = fmaxf(w_sum, 6.103515625e-5f);
        const float inv = 1.5f / denom;
        for (int k = 0; k < DS4_CUDA_TOP_K; k++) {
            weights[(uint64_t)t * DS4_CUDA_TOP_K + k] = pp[sel[k]] * inv;
        }
    }
}

#endif /* DS4_CUDA_ROUTER_CUH */
