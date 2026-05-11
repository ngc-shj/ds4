/* =========================================================================
 * cuda/compress.cuh - DS4 ratio-4 compressed KV state.
 * =========================================================================
 *
 * Reference: metal/dsv4_kv.metal::kernel_dsv4_compressor_store_one and
 * metal/dsv4_misc.metal::kernel_dsv4_softmax_pool, plus the orchestration in
 * ds4_metal.m::ds4_metal_compressor_prefill_tensor.
 *
 * Layout summary
 * --------------
 *   width = 2 * head_dim         (the KV tensor stores both halves of width)
 *   ratio = 4                    (production)
 *   coff  = 2 for ratio==4       (state has 8 rows = 2 * ratio)
 *
 * Each compressed cache row is produced by softmax-pooling 8 rows: the four
 * "prior ratio's first-half-of-width" rows plus the four "current ratio's
 * second-half-of-width" rows.  Pool weights come from per-element scores
 * (broadcast across head_dim) plus an APE (absolute position embedding) bias
 * indexed by row-within-ratio.
 *
 *   pool_kv [c, 0..3, w] = kv[(c-1)*4 + r, w]              for c > 0 else 0
 *   pool_kv [c, 4..7, w] = kv[ c   *4 + r, head_dim + w]
 *   pool_sc [c, 0..3, w] = sc[(c-1)*4 + r, w] + ape[r, w]              if c > 0
 *   pool_sc [c, 4..7, w] = sc[ c   *4 + r, head_dim + w] + ape[r, head_dim + w]
 *   comp    [c, w]       = softmax-pool over 8 rows (max-stable expf)
 */
#ifndef DS4_CUDA_COMPRESS_CUH
#define DS4_CUDA_COMPRESS_CUH

#include "common.cuh"
#include <float.h>

__device__ __forceinline__ float ds4_cuda_ape_read(
        const void *ape, uint32_t ape_type, uint32_t idx) {
    if (ape_type == 1u) {
        return __half2float(((const __half *)ape)[idx]);
    }
    return ((const float *)ape)[idx];
}

/* Online-softmax pool helper.  Updates (max, sum, acc) for one (score, value)
 * sample in a numerically stable way. */
__device__ __forceinline__ void ds4_cuda_pool_step(
        float &max_s, float &sum, float &acc, float scv, float vv) {
    if (scv > max_s) {
        const float scale_old = expf(max_s - scv);
        sum *= scale_old;
        acc *= scale_old;
        max_s = scv;
    }
    const float w = expf(scv - max_s);
    sum += w;
    acc += w * vv;
}

/* Generic prefill compressor.  ratio==4 uses coff=2 (width=2*head_dim, with
 * prev-half priming); other ratios use coff=1 (width=head_dim, no prev).
 * One block per c, threads stripe head_dim. */
__global__ void ds4_cuda_kernel_compressor_prefill_f32(
        float *comp_cache,
        const float *kv, const float *sc,
        const void *ape, uint32_t ape_type,
        uint32_t head_dim, uint32_t ratio, uint32_t coff, uint32_t n_comp) {
    const uint32_t c = blockIdx.x;
    const uint32_t w = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= n_comp || w >= head_dim) return;

    const uint32_t width = coff * head_dim;
    float max_s = -FLT_MAX, sum = 0.0f, acc = 0.0f;

    /* Phase A (ratio=4 only): previous ratio's first-half-of-width. */
    if (coff == 2u && c > 0) {
        #pragma unroll
        for (uint32_t r = 0; r < 4u; r++) {
            const uint32_t src_tok = (c - 1) * ratio + r;
            const float vv = kv[(uint64_t)src_tok * width + w];
            const float ape_v = ds4_cuda_ape_read(ape, ape_type, r * width + w);
            const float scv = sc[(uint64_t)src_tok * width + w] + ape_v;
            ds4_cuda_pool_step(max_s, sum, acc, scv, vv);
        }
    }
    /* Phase B: current ratio's data.  Offset is head_dim when coff==2
     * (second-half-of-width) and 0 when coff==1 (whole row). */
    const uint32_t in_off = (coff == 2u) ? head_dim : 0u;
    for (uint32_t r = 0; r < ratio; r++) {
        const uint32_t src_tok = c * ratio + r;
        const float vv = kv[(uint64_t)src_tok * width + in_off + w];
        const float ape_v = ds4_cuda_ape_read(ape, ape_type, r * width + in_off + w);
        const float scv = sc[(uint64_t)src_tok * width + in_off + w] + ape_v;
        ds4_cuda_pool_step(max_s, sum, acc, scv, vv);
    }
    if (sum > 0.0f) {
        comp_cache[(uint64_t)c * head_dim + w] = acc / sum;
    } else {
        comp_cache[(uint64_t)c * head_dim + w] = 0.0f;
    }
}

/* Initialise the rolling state at end of prefill.  The engine passes the
 * cutoff token position; this kernel writes state[0..3] from the last
 * completed ratio's first-half-of-width and zeroes state[4..7] (or fills
 * remainder rows).  Called separately after prefill_ratio4. */
__global__ void ds4_cuda_kernel_compressor_state_init_ratio4_f32(
        float *state_kv, float *state_score,
        const float *kv_cutoff_minus_ratio, const float *sc_cutoff_minus_ratio,
        const void *ape, uint32_t ape_type,
        uint32_t head_dim, int have_prev) {
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t r = blockIdx.y;
    if (w >= head_dim || r >= 4) return;
    const uint32_t width = 2u * head_dim;

    if (have_prev) {
        const uint64_t src = (uint64_t)r * width + w;
        state_kv[(uint64_t)r * width + w] = kv_cutoff_minus_ratio[src];
        const float ape_v = ds4_cuda_ape_read(ape, ape_type,
                                              (uint32_t)r * width + w);
        state_score[(uint64_t)r * width + w] =
            sc_cutoff_minus_ratio[src] + ape_v;
        /* Also copy second-half-of-width to row r (in state width has same
         * 2*head_dim layout). */
        state_kv[(uint64_t)r * width + head_dim + w] =
            kv_cutoff_minus_ratio[(uint64_t)r * width + head_dim + w];
        const float ape_v2 = ds4_cuda_ape_read(ape, ape_type,
                                               (uint32_t)r * width + head_dim + w);
        state_score[(uint64_t)r * width + head_dim + w] =
            sc_cutoff_minus_ratio[(uint64_t)r * width + head_dim + w] + ape_v2;
    } else {
        state_kv[(uint64_t)r * width + w] = 0.0f;
        state_kv[(uint64_t)r * width + head_dim + w] = 0.0f;
        state_score[(uint64_t)r * width + w] = -FLT_MAX;
        state_score[(uint64_t)r * width + head_dim + w] = -FLT_MAX;
    }
    /* Rows 4..7 stay at their post-prefill initial (-inf score, 0 kv). */
    state_kv[(uint64_t)(4 + r) * width + w] = 0.0f;
    state_kv[(uint64_t)(4 + r) * width + head_dim + w] = 0.0f;
    state_score[(uint64_t)(4 + r) * width + w] = -FLT_MAX;
    state_score[(uint64_t)(4 + r) * width + head_dim + w] = -FLT_MAX;
}

/* Single-token rolling update.  Mirrors metal kernel_dsv4_compressor_store_one.
 * For ratio == 4: writes incoming row into state[ratio + pos_mod, :] so the
 * "current half" fills up; the caller emits a compressed row + shifts state
 * when pos_mod hits ratio-1. */
__global__ void ds4_cuda_kernel_compressor_store_one_f32(
        const float *kv, const float *sc, const void *ape, uint32_t ape_type,
        float *state_kv, float *state_score,
        uint32_t width, uint32_t ratio, uint32_t pos) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    const uint32_t dst = dst_row * width + i;
    const uint32_t ape_i = pos_mod * width + i;
    const float ape_v = ds4_cuda_ape_read(ape, ape_type, ape_i);
    state_kv[dst] = kv[i];
    state_score[dst] = sc[i] + ape_v;
}

/* Batched store: n_tokens contiguous tokens starting at pos0.  Used for
 * decode batches > 1 and the post-prefill state catch-up. */
__global__ void ds4_cuda_kernel_compressor_store_batch_f32(
        const float *kv, const float *sc, const void *ape, uint32_t ape_type,
        float *state_kv, float *state_score,
        uint32_t width, uint32_t ratio, uint32_t pos0, uint32_t n_tokens) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= width || t >= n_tokens) return;
    const uint32_t pos = pos0 + t;
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    const uint32_t dst = dst_row * width + i;
    const uint32_t ape_i = pos_mod * width + i;
    const float ape_v = ds4_cuda_ape_read(ape, ape_type, ape_i);
    state_kv[dst] = kv[(uint64_t)t * width + i];
    state_score[dst] = sc[(uint64_t)t * width + i] + ape_v;
}

/* Single-token emit + shift (called when pos_mod transitions ratio-1 -> 0).
 *   1. Pool state_kv[0..7] / state_score[0..7] into comp_cache[comp_row, :].
 *   2. state_kv[0..3] = state_kv[4..7]; state_kv[4..7] = init values.
 * One block per head_dim element. */
__global__ void ds4_cuda_kernel_compressor_emit_shift_f32(
        float *comp_cache, uint32_t comp_row,
        float *state_kv, float *state_score,
        uint32_t head_dim) {
    const uint32_t w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= head_dim) return;
    const uint32_t width = 2u * head_dim;

    /* Pool: read first head_dim entries of width plane.  Width has K (0..hd)
     * and V (hd..2*hd) but the softmax pool used in prefill above only used
     * one half; mirror that here. */
    float kv8[8], sc8[8];
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        kv8[r] = state_kv[(uint64_t)r * width + w];
        sc8[r] = state_score[(uint64_t)r * width + w];
    }
    float max_s = sc8[0];
    #pragma unroll
    for (int r = 1; r < 8; r++) if (sc8[r] > max_s) max_s = sc8[r];
    float sum = 0.0f, acc = 0.0f;
    #pragma unroll
    for (int r = 0; r < 8; r++) {
        const float wgt = expf(sc8[r] - max_s);
        sum += wgt;
        acc += wgt * kv8[r];
    }
    comp_cache[(uint64_t)comp_row * head_dim + w] = acc / sum;

    /* Shift: rows 4..7 -> rows 0..3, rows 4..7 reset.  Done across both
     * halves of width to keep state consistent. */
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        state_kv[(uint64_t)r * width + w] = state_kv[(uint64_t)(4 + r) * width + w];
        state_kv[(uint64_t)r * width + head_dim + w] =
            state_kv[(uint64_t)(4 + r) * width + head_dim + w];
        state_score[(uint64_t)r * width + w] =
            state_score[(uint64_t)(4 + r) * width + w];
        state_score[(uint64_t)r * width + head_dim + w] =
            state_score[(uint64_t)(4 + r) * width + head_dim + w];
        state_kv[(uint64_t)(4 + r) * width + w] = 0.0f;
        state_kv[(uint64_t)(4 + r) * width + head_dim + w] = 0.0f;
        state_score[(uint64_t)(4 + r) * width + w] = -FLT_MAX;
        state_score[(uint64_t)(4 + r) * width + head_dim + w] = -FLT_MAX;
    }
}

#endif /* DS4_CUDA_COMPRESS_CUH */
