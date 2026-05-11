/* =========================================================================
 * cuda/attention.cuh - naive FlashAttention-style kernels for DS4.
 * =========================================================================
 *
 * Reference: metal/flash_attn.metal and ds4_metal.m's encode helpers.
 *
 * DS4 V4 Flash uses MLA (multi-latent attention): K and V share a single
 * compressed `raw_kv` buffer of shape [n_kv_rows, head_dim] while Q is per
 * head [n_tok, n_head, head_dim].  Attention scores are scaled by
 * 1/sqrt(head_dim), softmaxed with a per-head attention sink scalar, and
 * gathered into per-head output rows of shape [n_tok, n_head, head_dim].
 *
 * This file implements the "naive but correct" reference variant: each block
 * handles one (token, head) pair, threads tile head_dim across the row, and
 * the softmax runs as a numerically stable two-pass (max, then exp/sum) over
 * the sliding window.  It is the algorithm Metal's flash kernel optimises;
 * having it as the CUDA baseline lets the rest of the engine run end to end
 * while we port the tiled / pipelined version separately.  See PORT_CUDA.md
 * tier 4 for the perf-tuned follow-up.
 */
#ifndef DS4_CUDA_ATTENTION_CUH
#define DS4_CUDA_ATTENTION_CUH

#include "common.cuh"
#include <float.h>

/* Block-wide sum across blockDim.x threads using warp shuffles + shared mem.
 * Returns the reduced value on every thread (rebroadcast through smem). */
template <int BLOCK>
__device__ __forceinline__ float ds4_cuda_attn_block_sum(float val, float *smem_warp) {
    for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) smem_warp[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (threadIdx.x < (BLOCK + 31) / 32) ? smem_warp[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
        if (lane == 0) smem_warp[0] = val;
    }
    __syncthreads();
    return smem_warp[0];
}

/* Prefill raw-only causal attention with sliding window.
 *   q:        [n_tok, n_head, head_dim]
 *   raw_kv:   [n_tok, head_dim]           (K and V share this row)
 *   sinks:    [n_head]                     attention-sink logits
 *   heads:    [n_tok, n_head, head_dim]   output
 *   window:   sliding-window length (token attends to [t-window+1 .. t]) */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_prefill_raw_f32(
        float *heads, const float *q, const float *raw_kv, const float *sinks,
        uint32_t n_tok, uint32_t window, uint32_t n_head, uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qv = q + ((uint64_t)t * n_head + h) * head_dim;
    float       *out = heads + ((uint64_t)t * n_head + h) * head_dim;

    const float qval = (threadIdx.x < head_dim) ? qv[threadIdx.x] : 0.0f;

    const int32_t jstart = (int32_t)t + 1 - (int32_t)window;
    const int32_t jlo = jstart > 0 ? jstart : 0;
    const int32_t jhi = (int32_t)t;  /* inclusive (causal includes self) */

    extern __shared__ float smem[];
    float *warp_buf = smem;            /* 32 entries */
    float *score_bcast = smem + 32;    /* 1 entry, reused */

    const float sink = sinks ? sinks[h] : -FLT_MAX;

    /* Pass 1: max logit over the window plus the attention sink. */
    float max_logit = sink;
    for (int32_t j = jlo; j <= jhi; j++) {
        const float kj = (threadIdx.x < head_dim)
                           ? raw_kv[(uint64_t)j * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_logit) max_logit = s;
    }

    /* Pass 2: weighted sum + softmax denominator. */
    float denom = expf(sink - max_logit);     /* sink contribution */
    float acc   = 0.0f;
    for (int32_t j = jlo; j <= jhi; j++) {
        const float kj = (threadIdx.x < head_dim)
                           ? raw_kv[(uint64_t)j * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_logit);
        denom += w;
        acc   += w * kj;
    }

    if (threadIdx.x < head_dim) {
        out[threadIdx.x] = acc / denom;
    }
}

/* Decode (n_tokens new tokens) reading from a ring-buffered raw cache.  The
 * cache has capacity raw_cap; row j corresponds to logical position
 * (raw_start + j) mod raw_cap.  Causal mask uses pos0 + t as the current
 * absolute position. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_decode_raw_batch_f32(
        float *heads, const float *q, const float *raw_kv, const float *sinks,
        uint32_t n_tok, uint32_t pos0,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t window, uint32_t n_head, uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;

    const float scale = rsqrtf((float)head_dim);
    const float *qv = q + ((uint64_t)t * n_head + h) * head_dim;
    float       *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    const float qval = (threadIdx.x < head_dim) ? qv[threadIdx.x] : 0.0f;

    const uint32_t cur_pos = pos0 + t;
    /* Window: take min(window, n_raw) latest rows that are <= cur_pos. */
    uint32_t take = window;
    if (take > n_raw) take = n_raw;
    if (take > cur_pos + 1) take = cur_pos + 1;

    extern __shared__ float smem[];
    float *warp_buf = smem;
    float *score_bcast = smem + 32;
    const float sink = sinks ? sinks[h] : -FLT_MAX;

    float max_logit = sink;
    for (uint32_t j = 0; j < take; j++) {
        const uint32_t row = (raw_start + j) % raw_cap;
        const float *kvr = raw_kv + (uint64_t)row * head_dim;
        const float kj = (threadIdx.x < head_dim) ? kvr[threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_logit) max_logit = s;
    }

    float denom = expf(sink - max_logit);
    float acc   = 0.0f;
    for (uint32_t j = 0; j < take; j++) {
        const uint32_t row = (raw_start + j) % raw_cap;
        const float *kvr = raw_kv + (uint64_t)row * head_dim;
        const float kj = (threadIdx.x < head_dim) ? kvr[threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_logit);
        denom += w;
        acc   += w * kj;
    }

    if (threadIdx.x < head_dim) {
        out[threadIdx.x] = acc / denom;
    }
}

/* =========================================================================
 * Mixed (raw + compressed) attention variants.
 *
 * Each query token attends to two sources:
 *   - raw KV rows in a causal sliding window
 *   - compressed KV rows produced by the ratio-4 compressor, filtered by
 *     either a static causal cutoff, an explicit mask, or an indexer top-K.
 *
 * All variants share the same 2-pass softmax (find max, then weight-sum) and
 * write attention sink contributions just like the raw-only kernels above.
 * ========================================================================= */

/* Prefill static-mixed: token t attends to raw rows [max(0, t-window+1)..t]
 * plus all compressed rows c whose last-covered position is strictly before
 * the window start.  q, raw_kv shape: [n_tok, head_dim] (q is heads x dim);
 * comp_kv shape: [n_comp, head_dim]. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_prefill_static_mixed_f32(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *sinks,
        uint32_t n_tok, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;
    const float scale = rsqrtf((float)head_dim);

    const float *qv = q + ((uint64_t)t * n_head + h) * head_dim;
    float       *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    const float qval = (threadIdx.x < head_dim) ? qv[threadIdx.x] : 0.0f;

    /* Window for raw KV. */
    const int32_t jlo = (int32_t)t + 1 - (int32_t)window;
    const int32_t jraw_lo = jlo > 0 ? jlo : 0;
    const int32_t jraw_hi = (int32_t)t;
    /* Compressed visibility: token q sees comp rows [0, (q+1)/ratio).  This is
     * an additional view of older positions that the model attends to in
     * parallel with the raw window — both are visible even when their
     * position ranges overlap.  Matches metal's
     * ds4_metal_fill_static_mixed_prefill_mask:
     *   const uint32_t n_visible = (q + 1u) / ratio;
     * (was previously gated on `jraw_lo / ratio`, which silently zeroed comp
     * for any token whose raw window still reached position 0 — i.e. always
     * for typical prefills shorter than DS4_N_SWA=128.) */
    const uint32_t comp_visible = ((uint32_t)t + 1u) / ratio;
    const uint32_t comp_hi = comp_visible < n_comp ? comp_visible : n_comp;

    extern __shared__ float smem[];
    float *warp_buf = smem;
    float *score_bcast = smem + 32;
    const float sink = sinks ? sinks[h] : -FLT_MAX;

    /* Pass 1: max. */
    float max_l = sink;
    for (int32_t j = jraw_lo; j <= jraw_hi; j++) {
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)j * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }
    for (uint32_t c = 0; c < comp_hi; c++) {
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }

    /* Pass 2: weighted sum. */
    float denom = expf(sink - max_l);
    float acc   = 0.0f;
    for (int32_t j = jraw_lo; j <= jraw_hi; j++) {
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)j * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    for (uint32_t c = 0; c < comp_hi; c++) {
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    if (threadIdx.x < head_dim) out[threadIdx.x] = acc / denom;
}

/* Prefill masked-mixed: same as static but compressed visibility is dictated
 * by an explicit per-token mask: comp_mask[t, c] = 1 if visible. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_prefill_masked_mixed_f32(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, const float *sinks,
        uint32_t n_tok, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    (void)ratio;
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;
    const float scale = rsqrtf((float)head_dim);
    const float *qv = q + ((uint64_t)t * n_head + h) * head_dim;
    float       *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    const float qval = (threadIdx.x < head_dim) ? qv[threadIdx.x] : 0.0f;

    const int32_t jlo = (int32_t)t + 1 - (int32_t)window;
    const int32_t jraw_lo = jlo > 0 ? jlo : 0;
    const int32_t jraw_hi = (int32_t)t;

    extern __shared__ float smem[];
    float *warp_buf = smem;
    float *score_bcast = smem + 32;
    const float sink = sinks ? sinks[h] : -FLT_MAX;

    float max_l = sink;
    for (int32_t j = jraw_lo; j <= jraw_hi; j++) {
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)j * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        if (comp_mask[(uint64_t)t * n_comp + c] == 0.0f) continue;
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }

    float denom = expf(sink - max_l);
    float acc   = 0.0f;
    for (int32_t j = jraw_lo; j <= jraw_hi; j++) {
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)j * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        if (comp_mask[(uint64_t)t * n_comp + c] == 0.0f) continue;
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    if (threadIdx.x < head_dim) out[threadIdx.x] = acc / denom;
}

/* Decode mixed-batch: ring-buffered raw_kv + all (or masked) comp_kv.
 * One block per (token, head), threads stripe head_dim. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_decode_mixed_batch_f32(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const float *comp_mask, uint32_t use_comp_mask, const float *sinks,
        uint32_t n_tok, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    (void)ratio;
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;
    const float scale = rsqrtf((float)head_dim);
    const float *qv = q + ((uint64_t)t * n_head + h) * head_dim;
    float       *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    const float qval = (threadIdx.x < head_dim) ? qv[threadIdx.x] : 0.0f;

    const uint32_t cur_pos = pos0 + t;
    uint32_t take = window;
    if (take > n_raw) take = n_raw;
    if (take > cur_pos + 1) take = cur_pos + 1;

    extern __shared__ float smem[];
    float *warp_buf = smem;
    float *score_bcast = smem + 32;
    const float sink = sinks ? sinks[h] : -FLT_MAX;

    float max_l = sink;
    for (uint32_t j = 0; j < take; j++) {
        const uint32_t row = (raw_start + j) % raw_cap;
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)row * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        if (use_comp_mask && comp_mask[(uint64_t)t * n_comp + c] == 0.0f) continue;
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }

    float denom = expf(sink - max_l);
    float acc   = 0.0f;
    for (uint32_t j = 0; j < take; j++) {
        const uint32_t row = (raw_start + j) % raw_cap;
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)row * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        if (use_comp_mask && comp_mask[(uint64_t)t * n_comp + c] == 0.0f) continue;
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    if (threadIdx.x < head_dim) out[threadIdx.x] = acc / denom;
}

/* Indexed mixed-batch: like decode_mixed but compressed rows come from a
 * top-K indices array `topk[t, 0..top_k-1]`. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_attn_indexed_mixed_batch_f32(
        float *heads, const float *q, const float *raw_kv, const float *comp_kv,
        const int32_t *topk, const float *sinks,
        uint32_t n_tok, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    if (t >= n_tok || h >= n_head) return;
    const float scale = rsqrtf((float)head_dim);
    const float *qv = q + ((uint64_t)t * n_head + h) * head_dim;
    float       *out = heads + ((uint64_t)t * n_head + h) * head_dim;
    const float qval = (threadIdx.x < head_dim) ? qv[threadIdx.x] : 0.0f;

    const uint32_t cur_pos = pos0 + t;
    uint32_t take = window;
    if (take > n_raw) take = n_raw;
    if (take > cur_pos + 1) take = cur_pos + 1;
    /* Per-token causal visibility: token at qpos sees comp rows [0, (qpos+1)/ratio).
     * Matches metal's kernel_dsv4_indexed_mixed_attention_heads8 visible cutoff
     * and prevents reading compressed rows that cover positions >= qpos+1. */
    uint32_t visible = (cur_pos + 1u) / ratio;
    if (visible > n_comp) visible = n_comp;

    extern __shared__ float smem[];
    float *warp_buf = smem;
    float *score_bcast = smem + 32;
    const float sink = sinks ? sinks[h] : -FLT_MAX;

    float max_l = sink;
    for (uint32_t j = 0; j < take; j++) {
        const uint32_t row = (raw_start + j) % raw_cap;
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)row * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }
    for (uint32_t k = 0; k < top_k; k++) {
        const int32_t c = topk[(uint64_t)t * top_k + k];
        if (c < 0) continue;
        if ((uint32_t)c >= visible) break;  // matches metal: stop on first invisible idx
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        if (s > max_l) max_l = s;
    }

    float denom = expf(sink - max_l);
    float acc   = 0.0f;
    for (uint32_t j = 0; j < take; j++) {
        const uint32_t row = (raw_start + j) % raw_cap;
        const float kj = (threadIdx.x < head_dim)
            ? raw_kv[(uint64_t)row * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    for (uint32_t k = 0; k < top_k; k++) {
        const int32_t c = topk[(uint64_t)t * top_k + k];
        if (c < 0) continue;
        if ((uint32_t)c >= visible) break;
        const float kj = (threadIdx.x < head_dim)
            ? comp_kv[(uint64_t)c * head_dim + threadIdx.x] : 0.0f;
        float dot = qval * kj;
        dot = ds4_cuda_attn_block_sum<BLOCK>(dot, warp_buf);
        if (threadIdx.x == 0) score_bcast[0] = dot * scale;
        __syncthreads();
        const float s = score_bcast[0];
        const float w = expf(s - max_l);
        denom += w;
        acc   += w * kj;
    }
    if (threadIdx.x < head_dim) out[threadIdx.x] = acc / denom;
}

#endif /* DS4_CUDA_ATTENTION_CUH */
