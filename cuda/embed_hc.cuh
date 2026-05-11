/* =========================================================================
 * cuda/embed_hc.cuh - embeddings and HC (hyper-connection) mixers.
 * =========================================================================
 *
 * Reference: ds4_metal.m around line 4191 (embeddings) and 13421+
 * (HC mixers).  metal/dsv4_misc.metal has the HC kernels.
 *
 * DS4's HC layout keeps 4 parallel residual streams ("hyper-connections").
 * Before each sublayer the four streams are reduced into one 4096-wide row
 * using a learned mix; after the sublayer the result is expanded back into
 * the four streams.  The mix tensor has a fixed shape:
 *   mix[2*n_hc + n_hc*n_hc]
 * where the leading 2*n_hc entries are pre/post element scales and the
 * trailing n_hc*n_hc form a per-hc combine matrix.
 *
 * For correctness here we model only the n_hc == 4 production case, since
 * the engine assumes it everywhere (see ds4_metal_hc_split_weighted_sum).
 */
#ifndef DS4_CUDA_EMBED_HC_CUH
#define DS4_CUDA_EMBED_HC_CUH

#include "common.cuh"

/* Embed one token id and replicate the row to n_hc streams.
 * out_hc layout: [n_hc, n_embd] row-major. */
__global__ void ds4_cuda_kernel_embed_token_hc_f16(
        float *out_hc, const __half *w,
        uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t i  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hc = blockIdx.y;
    if (i >= n_embd || hc >= n_hc) return;
    const float v = __half2float(w[(uint64_t)token * n_embd + i]);
    out_hc[(uint64_t)hc * n_embd + i] = v;
}

/* Batched embed: lookup `tokens[t]` and replicate to n_hc streams per token.
 * out_hc layout: [n_tok, n_hc, n_embd] row-major. */
__global__ void ds4_cuda_kernel_embed_tokens_hc_f16(
        float *out_hc, const __half *w, const int32_t *tokens,
        uint32_t n_tok, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t i  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hc = blockIdx.y;
    const uint32_t t  = blockIdx.z;
    if (i >= n_embd || hc >= n_hc || t >= n_tok) return;
    const int32_t tok = tokens[t];
    const float v = __half2float(w[(uint64_t)tok * n_embd + i]);
    out_hc[((uint64_t)t * n_hc + hc) * n_embd + i] = v;
}

/* Repeat row -> [n_hc, n_embd]: out_hc[hc, i] = row[i]. */
__global__ void ds4_cuda_kernel_repeat_hc_f32(
        float *out_hc, const float *row, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t i  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hc = blockIdx.y;
    if (i >= n_embd || hc >= n_hc) return;
    out_hc[(uint64_t)hc * n_embd + i] = row[i];
}

/* HC weighted sum: out[i] = sum_{hc} weights[hc] * residual_hc[hc, i].
 * weights stride (in floats) lets the engine point us at either a dedicated
 * weight vector (stride = n_hc) or the mix tensor (stride = 2*n_hc + n_hc*n_hc).
 * residual_hc layout: [n_rows, n_hc, n_embd].  out layout: [n_rows, n_embd]. */
__global__ void ds4_cuda_kernel_hc_weighted_sum_f32(
        float *out, const float *residual_hc, const float *weights,
        uint32_t weight_stride, uint32_t n_embd, uint32_t n_hc) {
    const uint32_t i   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (i >= n_embd) return;
    const float *w = weights + (uint64_t)row * weight_stride;
    const float *r = residual_hc + (uint64_t)row * n_hc * n_embd;
    float acc = 0.0f;
    for (uint32_t hc = 0; hc < n_hc; hc++) {
        acc += w[hc] * r[(uint64_t)hc * n_embd + i];
    }
    out[(uint64_t)row * n_embd + i] = acc;
}

/* HC expand: combine the sublayer output back into the n_hc residual streams.
 * Math from metal/dsv4_misc.metal::kernel_hc_expand_f32:
 *   out_hc[hc, i] = residual_hc[hc, i] + post[hc] * (block_out[i] + comb[hc] * 0)
 * Simplified DS4 form: out_hc[hc, i] = residual_hc[hc, i] + post[hc] * block_out[i].
 * The optional `comb` vector lets multi-stream blocks blend before adding;
 * we keep the API but pass only the simple variant below. */
__global__ void ds4_cuda_kernel_hc_expand_split_f32(
        float *out_hc, const float *block_out, const float *residual_hc,
        const float *split, uint32_t n_embd, uint32_t n_hc) {
    /* split layout matches metal/dsv4_misc.metal: the first n_hc entries are
     * the per-hc "post" scales used here.  Following entries (combine matrix)
     * are unused by hc_expand_split, which is the simple post-add path. */
    const uint32_t i  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hc = blockIdx.y;
    if (i >= n_embd || hc >= n_hc) return;
    const float post = split[hc];
    const float r = residual_hc[(uint64_t)hc * n_embd + i];
    out_hc[(uint64_t)hc * n_embd + i] = r + post * block_out[i];
}

/* Same idea but with two summed block outputs (used after the FFN where
 * shared + routed expert outputs both contribute). */
__global__ void ds4_cuda_kernel_hc_expand_add_split_f32(
        float *out_hc, const float *block_out, const float *block_add,
        const float *residual_hc, const float *split,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t i  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t hc = blockIdx.y;
    if (i >= n_embd || hc >= n_hc) return;
    const float post = split[hc];
    const float r = residual_hc[(uint64_t)hc * n_embd + i];
    const float b = block_out[i] + block_add[i];
    out_hc[(uint64_t)hc * n_embd + i] = r + post * b;
}

/* HC split sinkhorn: per-token mixer logits -> per-token split tensor with
 *   - pre weights[n_hc]       (sigmoid + eps)
 *   - post gates[n_hc]        (2*sigmoid)
 *   - combination[n_hc*n_hc]  (softmax-per-row + sinkhorn double-stochastic normalisation)
 * Specialised to n_hc==4 which is the DS4 production case (see metal/dsv4_hc.metal).
 * One thread per row. */
__global__ void ds4_cuda_kernel_hc_split_sinkhorn_hc4_f32(
        float *dst, const float *mixes, const float *scale, const float *base,
        uint32_t n_rows, uint32_t sinkhorn_iters, float eps) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    constexpr int HC = 4;
    const int mix_hc = 2 * HC + HC * HC;        /* = 24 */
    const float *mix = mixes + (uint64_t)row * mix_hc;
    float       *out = dst   + (uint64_t)row * mix_hc;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    /* pre weights: sigmoid(mix * pre_scale + base) + eps */
    #pragma unroll
    for (int i = 0; i < HC; i++) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    /* post gates: 2 * sigmoid(mix * post_scale + base) */
    #pragma unroll
    for (int i = 0; i < HC; i++) {
        const float z = mix[HC + i] * post_scale + base[HC + i];
        out[HC + i] = 2.0f / (1.0f + expf(-z));
    }
    /* combination matrix: 4x4 softmax-per-row then Sinkhorn iterations to make
     * it doubly stochastic.  We work in registers (16 floats). */
    float r[HC][HC];
    for (int d = 0; d < HC; d++) {
        float row_max = -INFINITY;
        for (int s = 0; s < HC; s++) {
            const int off = 2 * HC + s + d * HC;
            const float v = mix[off] * comb_scale + base[off];
            r[d][s] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (int s = 0; s < HC; s++) {
            r[d][s] = expf(r[d][s] - row_max);
            row_sum += r[d][s];
        }
        const float inv = 1.0f / row_sum;
        for (int s = 0; s < HC; s++) r[d][s] = r[d][s] * inv + eps;
    }
    /* Column normalize once (paired with row normalize below in iter 0). */
    {
        float col_sum[HC];
        for (int s = 0; s < HC; s++) {
            col_sum[s] = r[0][s] + r[1][s] + r[2][s] + r[3][s] + eps;
        }
        for (int d = 0; d < HC; d++)
            for (int s = 0; s < HC; s++)
                r[d][s] /= col_sum[s];
    }
    /* Additional sinkhorn iterations alternate row / column normalize. */
    for (uint32_t it = 1; it < sinkhorn_iters; it++) {
        for (int d = 0; d < HC; d++) {
            const float rs = r[d][0] + r[d][1] + r[d][2] + r[d][3] + eps;
            for (int s = 0; s < HC; s++) r[d][s] /= rs;
        }
        float col_sum[HC];
        for (int s = 0; s < HC; s++) {
            col_sum[s] = r[0][s] + r[1][s] + r[2][s] + r[3][s] + eps;
        }
        for (int d = 0; d < HC; d++)
            for (int s = 0; s < HC; s++)
                r[d][s] /= col_sum[s];
    }
    for (int d = 0; d < HC; d++)
        for (int s = 0; s < HC; s++)
            out[2 * HC + s + d * HC] = r[d][s];
}

/* Fused HC split-sinkhorn + weighted sum.  After computing the per-token
 * split (pre/post/combination) the kernel immediately reduces the residual
 * streams using the pre weights (the first n_hc entries of the split). */
__global__ void ds4_cuda_kernel_hc_split_weighted_sum_hc4_f32(
        float *out, float *split,
        const float *mix, const float *residual_hc,
        const float *scale, const float *base,
        uint32_t n_embd, uint32_t n_rows, uint32_t sinkhorn_iters, float eps) {
    const uint32_t i   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= n_rows) return;

    constexpr int HC = 4;
    const int mix_hc = 2 * HC + HC * HC;

    /* Thread (0, row) computes the split once into smem. */
    __shared__ float sp[24];
    if (threadIdx.x == 0) {
        const float *mixr  = mix  + (uint64_t)row * mix_hc;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];
        for (int k = 0; k < HC; k++)
            sp[k] = 1.0f / (1.0f + expf(-(mixr[k] * pre_scale + base[k]))) + eps;
        for (int k = 0; k < HC; k++)
            sp[HC + k] = 2.0f / (1.0f + expf(-(mixr[HC + k] * post_scale + base[HC + k])));
        float r[HC][HC];
        for (int d = 0; d < HC; d++) {
            float row_max = -INFINITY;
            for (int s = 0; s < HC; s++) {
                const int off = 2 * HC + s + d * HC;
                const float v = mixr[off] * comb_scale + base[off];
                r[d][s] = v;
                if (v > row_max) row_max = v;
            }
            float row_sum = 0.0f;
            for (int s = 0; s < HC; s++) { r[d][s] = expf(r[d][s] - row_max); row_sum += r[d][s]; }
            const float inv = 1.0f / row_sum;
            for (int s = 0; s < HC; s++) r[d][s] = r[d][s] * inv + eps;
        }
        {
            float col_sum[HC];
            for (int s = 0; s < HC; s++) col_sum[s] = r[0][s] + r[1][s] + r[2][s] + r[3][s] + eps;
            for (int d = 0; d < HC; d++) for (int s = 0; s < HC; s++) r[d][s] /= col_sum[s];
        }
        for (uint32_t it = 1; it < sinkhorn_iters; it++) {
            for (int d = 0; d < HC; d++) {
                const float rs = r[d][0] + r[d][1] + r[d][2] + r[d][3] + eps;
                for (int s = 0; s < HC; s++) r[d][s] /= rs;
            }
            float col_sum[HC];
            for (int s = 0; s < HC; s++) col_sum[s] = r[0][s] + r[1][s] + r[2][s] + r[3][s] + eps;
            for (int d = 0; d < HC; d++) for (int s = 0; s < HC; s++) r[d][s] /= col_sum[s];
        }
        for (int d = 0; d < HC; d++)
            for (int s = 0; s < HC; s++)
                sp[2 * HC + s + d * HC] = r[d][s];
        /* Persist to global split tensor too. */
        float *sp_out = split + (uint64_t)row * mix_hc;
        for (int k = 0; k < mix_hc; k++) sp_out[k] = sp[k];
    }
    __syncthreads();

    if (i >= n_embd) return;
    /* Reduce residual_hc[row, :, i] with pre weights sp[0..HC-1]. */
    const float *res = residual_hc + (uint64_t)row * HC * n_embd;
    float acc = 0.0f;
    #pragma unroll
    for (int hc = 0; hc < HC; hc++) acc += sp[hc] * res[(uint64_t)hc * n_embd + i];
    out[(uint64_t)row * n_embd + i] = acc;
}

/* Output HC weights: out = sigmoid(pre * scale + base) + eps.  Per-token row
 * of n_hc weights. */
__global__ void ds4_cuda_kernel_output_hc_weights_f32(
        float *out, const float *pre, const float *scale, const float *base,
        uint32_t n_hc, uint32_t n_tok, float eps) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (i >= n_hc || t >= n_tok) return;
    float v = pre[(uint64_t)t * n_hc + i] * scale[0];
    v += base[i];
    v = 1.0f / (1.0f + expf(-v));
    out[(uint64_t)t * n_hc + i] = v + eps;
}

/* HC expand with combine matrix (n_hc==4 production case).
 * dst[t, dst_hc, d] = block_out[t, d] * post[t, dst_hc] + sum_src_hc
 *                     comb[t, dst_hc, src_hc] * residual_hc[t, src_hc, d] */
__global__ void ds4_cuda_kernel_hc_expand_with_comb_hc4_f32(
        float *dst, const float *block_out, const float *residual,
        const float *post, const float *comb,
        const float *block_add, int has_add,
        uint32_t n_embd, uint32_t n_tok) {
    constexpr int HC = 4;
    const uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t = blockIdx.y;
    if (d >= n_embd || t >= n_tok) return;

    float bv = block_out[(uint64_t)t * n_embd + d];
    if (has_add) bv += block_add[(uint64_t)t * n_embd + d];
    const float *res = residual + (uint64_t)t * HC * n_embd;
    const float r0 = res[(uint64_t)0 * n_embd + d];
    const float r1 = res[(uint64_t)1 * n_embd + d];
    const float r2 = res[(uint64_t)2 * n_embd + d];
    const float r3 = res[(uint64_t)3 * n_embd + d];
    const float *pp = post + (uint64_t)t * HC;
    const float *cm = comb + (uint64_t)t * HC * HC;
    #pragma unroll
    for (int dst_hc = 0; dst_hc < HC; dst_hc++) {
        float acc = bv * pp[dst_hc];
        acc += cm[dst_hc * HC + 0] * r0;
        acc += cm[dst_hc * HC + 1] * r1;
        acc += cm[dst_hc * HC + 2] * r2;
        acc += cm[dst_hc * HC + 3] * r3;
        dst[((uint64_t)t * HC + dst_hc) * n_embd + d] = acc;
    }
}

/* Fused HC split-sinkhorn + weighted_sum + RMSNorm-with-weight.  This is the
 * pre-sublayer entry path: produces both the reduced-sublayer row (out) and
 * its RMS-normed version (norm_out) in one dispatch. */
__global__ void ds4_cuda_kernel_hc_split_weighted_sum_norm_hc4_f32(
        float *out, float *norm_out, float *split,
        const float *mix, const float *residual_hc,
        const float *scale, const float *base, const float *norm_w,
        uint32_t n_embd, uint32_t n_rows, uint32_t sinkhorn_iters,
        float eps, float norm_eps) {
    /* Reuse the split-weighted-sum kernel logic to produce `out` and `split`,
     * then RMSNorm.  Single block per row (n_embd <= 4096 fits comfortably). */
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    constexpr int HC = 4;
    const int mix_hc = 2 * HC + HC * HC;

    /* Stage 1: split (sp) computed once on thread 0 of each block. */
    __shared__ float sp[24];
    if (threadIdx.x == 0) {
        const float *mixr = mix + (uint64_t)row * mix_hc;
        const float pre_s = scale[0], post_s = scale[1], comb_s = scale[2];
        for (int k = 0; k < HC; k++)
            sp[k] = 1.0f / (1.0f + expf(-(mixr[k] * pre_s + base[k]))) + eps;
        for (int k = 0; k < HC; k++)
            sp[HC + k] = 2.0f / (1.0f + expf(-(mixr[HC + k] * post_s + base[HC + k])));
        float r[HC][HC];
        for (int d = 0; d < HC; d++) {
            float row_max = -INFINITY;
            for (int s = 0; s < HC; s++) {
                const int off = 2 * HC + s + d * HC;
                const float v = mixr[off] * comb_s + base[off];
                r[d][s] = v;
                if (v > row_max) row_max = v;
            }
            float row_sum = 0.0f;
            for (int s = 0; s < HC; s++) { r[d][s] = expf(r[d][s] - row_max); row_sum += r[d][s]; }
            const float inv = 1.0f / row_sum;
            for (int s = 0; s < HC; s++) r[d][s] = r[d][s] * inv + eps;
        }
        {
            float col_sum[HC];
            for (int s = 0; s < HC; s++) col_sum[s] = r[0][s] + r[1][s] + r[2][s] + r[3][s] + eps;
            for (int d = 0; d < HC; d++) for (int s = 0; s < HC; s++) r[d][s] /= col_sum[s];
        }
        for (uint32_t it = 1; it < sinkhorn_iters; it++) {
            for (int d = 0; d < HC; d++) {
                const float rs = r[d][0] + r[d][1] + r[d][2] + r[d][3] + eps;
                for (int s = 0; s < HC; s++) r[d][s] /= rs;
            }
            float col_sum[HC];
            for (int s = 0; s < HC; s++) col_sum[s] = r[0][s] + r[1][s] + r[2][s] + r[3][s] + eps;
            for (int d = 0; d < HC; d++) for (int s = 0; s < HC; s++) r[d][s] /= col_sum[s];
        }
        for (int d = 0; d < HC; d++)
            for (int s = 0; s < HC; s++)
                sp[2 * HC + s + d * HC] = r[d][s];
        float *sp_out = split + (uint64_t)row * mix_hc;
        for (int k = 0; k < mix_hc; k++) sp_out[k] = sp[k];
    }
    __syncthreads();

    /* Stage 2: weighted sum over residual streams + RMSNorm with weight.
     * Compute partial sum-of-squares first. */
    const float *res = residual_hc + (uint64_t)row * HC * n_embd;
    extern __shared__ float ssum[];
    float sumsq = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float acc = 0.0f;
        #pragma unroll
        for (int hc = 0; hc < HC; hc++) acc += sp[hc] * res[(uint64_t)hc * n_embd + i];
        out[(uint64_t)row * n_embd + i] = acc;
        sumsq += acc * acc;
    }
    for (int off = 16; off > 0; off >>= 1) sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) ssum[warp] = sumsq;
    __syncthreads();
    if (warp == 0) {
        sumsq = (threadIdx.x < (blockDim.x + 31) / 32) ? ssum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sumsq += __shfl_xor_sync(0xffffffff, sumsq, off);
        if (lane == 0) ssum[0] = sumsq;
    }
    __syncthreads();
    const float scale_n = rsqrtf(ssum[0] / (float)n_embd + norm_eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        norm_out[(uint64_t)row * n_embd + i] = out[(uint64_t)row * n_embd + i] * scale_n * norm_w[i];
    }
}

#endif /* DS4_CUDA_EMBED_HC_CUH */
