/* =========================================================================
 * cuda/rope.cuh - DeepSeek V4 partial RoPE with YaRN scaling.
 * =========================================================================
 *
 * Direct port of metal/dsv4_rope.metal::kernel_dsv4_rope_tail_f32.  DS4 uses a
 * partial rotation: the first (head_dim - n_rot) elements ("nope" prefix) are
 * copied through unchanged, only the last n_rot elements are rotated.  Mode 0
 * (interleaved pair) is the only DS4 production mode; mode 2 (NeoX layout) is
 * preserved for parity but not exercised by the engine.
 *
 * Grid is (n_head, n_tok, 1) so positions are derived as pos0 + token_index.
 * In place on x.
 */
#ifndef DS4_CUDA_ROPE_CUH
#define DS4_CUDA_ROPE_CUH

#include "common.cuh"
#include <math_constants.h>

__device__ __forceinline__ float ds4_cuda_rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

__device__ __forceinline__ void ds4_cuda_rope_yarn(
        float theta_extrap, float freq_scale, float corr0, float corr1,
        int i0, float ext_factor, float mscale,
        float *cos_theta, float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = ds4_cuda_rope_yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

__device__ __forceinline__ float ds4_cuda_rope_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * CUDART_PI_F))
         / (2.0f * logf(base));
}

/* In-place RoPE-tail.  x has logical shape [n_tok, n_head, head_dim] in
 * row-major.  Only x[..., n_nope:] is touched.  Each block handles one
 * (head, token) row; threads iterate the head_dim columns. */
__global__ void ds4_cuda_kernel_rope_tail_f32(
        float    *x,
        uint32_t  n_tok,
        uint32_t  n_head,
        uint32_t  head_dim,
        uint32_t  n_rot,
        uint32_t  pos0,
        uint32_t  n_ctx_orig,
        int32_t   inverse,
        float     freq_base,
        float     freq_scale,
        float     ext_factor,
        float     attn_factor,
        float     beta_fast,
        float     beta_slow) {
    const uint32_t head = blockIdx.x;
    const uint32_t tok  = blockIdx.y;
    if (head >= n_head || tok >= n_tok) return;

    const int n_nope = (int)head_dim - (int)n_rot;
    if (n_nope < 0) return;

    float *row = x + ((uint64_t)tok * n_head + head) * head_dim;
    const float theta_base = (float)(pos0 + tok);
    const float inv_ndims = -1.0f / (float)n_rot;

    /* YaRN correction dims, computed once per block. */
    const float corr_a_raw = ds4_cuda_rope_corr_factor((int)n_rot, (int)n_ctx_orig, beta_fast, freq_base);
    const float corr_b_raw = ds4_cuda_rope_corr_factor((int)n_rot, (int)n_ctx_orig, beta_slow, freq_base);
    const float corr0 = fmaxf(0.0f, floorf(corr_a_raw));
    const float corr1 = fminf((float)n_rot - 1.0f, ceilf(corr_b_raw));

    /* Interleaved pair layout (DS4 production mode).  Each thread that lands
     * on an even rotated index pairs (r, r+1) and rotates them; nope prefix
     * threads are no-ops (the data is already in place because we read+write
     * row in place). */
    for (uint32_t i0 = threadIdx.x; i0 < head_dim; i0 += blockDim.x) {
        if ((int)i0 < n_nope) continue;
        const int r = (int)i0 - n_nope;
        if ((r & 1) != 0) continue;

        const float theta = theta_base * powf(freq_base, inv_ndims * (float)r);
        float c, s;
        ds4_cuda_rope_yarn(theta, freq_scale, corr0, corr1, r,
                           ext_factor, attn_factor, &c, &s);
        if (inverse) s = -s;

        const int j0 = n_nope + r;
        const int j1 = j0 + 1;
        const float x0 = row[j0];
        const float x1 = row[j1];
        row[j0] = x0 * c - x1 * s;
        row[j1] = x0 * s + x1 * c;
    }
}

#endif /* DS4_CUDA_ROPE_CUH */
