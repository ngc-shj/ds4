/* =========================================================================
 * cuda/indexer.cuh - DS4 ratio-4 indexer (score, top-K, mask).
 * =========================================================================
 *
 * Reference: metal/dsv4_misc.metal::{kernel_dsv4_indexer_score_one_direct,
 *   kernel_dsv4_topk_mask, kernel_dsv4_topk_mask_scatter} and
 * ds4_metal.m::ds4_metal_indexer_*.
 *
 * For each query token, the indexer scores every compressed row using
 *     score[c] = sum_h max(<q[h], index_comp[c]>, 0) * weights[h] * scale
 * with optional ReLU on the inner product.  The engine then picks the top-K
 * compressed rows per token and builds a boolean mask.
 *
 * Implementations here are deliberately straightforward: one block per
 * (compressed-row, token) for scoring, and per-token serial selection for
 * top-K.  Both can be replaced with CUTLASS / CUB based variants later.
 */
#ifndef DS4_CUDA_INDEXER_CUH
#define DS4_CUDA_INDEXER_CUH

#include "common.cuh"
#include <float.h>

/* Decode-batch indexer score: scores[t, c] = ReLU-sum over heads.
 *   q:           [n_tok, n_head, head_dim]
 *   weights:     [n_tok, n_head]   per-head route weight from the indexer
 *   index_comp:  [n_comp, head_dim]   compressed K projection rows
 * Block grid (n_comp, n_tok); BLOCK threads cooperatively reduce heads. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_indexer_scores_f32(
        float *scores, const float *q, const float *weights, const float *index_comp,
        uint32_t n_comp, uint32_t n_tok, uint32_t n_head, uint32_t head_dim, float scale) {
    const uint32_t c = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (c >= n_comp || t >= n_tok) return;

    const float *kc = index_comp + (uint64_t)c * head_dim;
    const float *qt = q + (uint64_t)t * n_head * head_dim;
    const float *wt = weights + (uint64_t)t * n_head;

    float partial = 0.0f;
    for (uint32_t h = threadIdx.x; h < n_head; h += BLOCK) {
        const float *qh = qt + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) dot += qh[i] * kc[i];
        partial += fmaxf(dot, 0.0f) * wt[h] * scale;
    }
    __shared__ float warp_sums[32];
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        partial = (threadIdx.x < (BLOCK + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (lane == 0) {
            scores[(uint64_t)t * n_comp + c] = partial;
        }
    }
}

/* Per-token selection of top-K compressed rows by score.  Serial selection
 * (n_comp <= 4096 typically), one block per token, single thread does the
 * work.  Output: selected[t, k] = compressed-row index. */
__global__ void ds4_cuda_kernel_indexer_topk_f32(
        int32_t *selected, const float *scores,
        uint32_t n_comp, uint32_t n_tok, uint32_t top_k) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;
    if (threadIdx.x != 0) return;

    /* Local scratch held in shared memory to handle larger n_comp without
     * blowing register file. */
    extern __shared__ float smem_scores[];
    for (uint32_t c = 0; c < n_comp; c++)
        smem_scores[c] = scores[(uint64_t)t * n_comp + c];
    for (uint32_t k = 0; k < top_k; k++) {
        int   best = 0;
        float bv   = -FLT_MAX;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (smem_scores[c] > bv) { bv = smem_scores[c]; best = (int)c; }
        }
        selected[(uint64_t)t * top_k + k] = best;
        smem_scores[best] = -FLT_MAX;
    }
}

/* Build a boolean mask (per token) of which compressed rows are in the top-K
 * set.  mask: [n_tok, n_comp] of 0/1 floats. */
__global__ void ds4_cuda_kernel_topk_mask_f32(
        float *mask, const int32_t *topk,
        uint32_t n_comp, uint32_t n_tok, uint32_t top_k) {
    const uint32_t t = blockIdx.x;
    const uint32_t c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= n_tok || c >= n_comp) return;
    /* Initialise to 0. */
    mask[(uint64_t)t * n_comp + c] = 0.0f;
}

__global__ void ds4_cuda_kernel_topk_mask_scatter_f32(
        float *mask, const int32_t *topk,
        uint32_t n_comp, uint32_t n_tok, uint32_t top_k) {
    const uint32_t t = blockIdx.x;
    const uint32_t k = threadIdx.x;
    if (t >= n_tok || k >= top_k) return;
    const int32_t c = topk[(uint64_t)t * top_k + k];
    if (c >= 0 && (uint32_t)c < n_comp) {
        mask[(uint64_t)t * n_comp + c] = 1.0f;
    }
}

#endif /* DS4_CUDA_INDEXER_CUH */
