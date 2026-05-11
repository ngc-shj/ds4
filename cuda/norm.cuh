/* =========================================================================
 * cuda/norm.cuh - RMSNorm family kernels for the DS4 CUDA backend.
 * =========================================================================
 *
 * The Metal reference lives in metal/norm.metal.  All these kernels share the
 * same shape: parallel sum of squares across a row, warp/block reduce, divide
 * by row length, sqrt+eps, scale, optionally multiply by a learned weight.
 * The 3D variant (head RMS norm) is the same math run independently for each
 * (head, token) pair.
 */
#ifndef DS4_CUDA_NORM_CUH
#define DS4_CUDA_NORM_CUH

#include "common.cuh"

/* Block-level reduction of `val` across BLOCK threads in one block.  Uses warp
 * shuffles and a single shared-memory slot per warp.  BLOCK must be <= 1024
 * and a multiple of 32. */
template <int BLOCK>
__device__ __forceinline__ float ds4_cuda_block_sum(float val) {
    __shared__ float warp_sums[32];
    for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();
    if (warp == 0) {
        val = (threadIdx.x < (BLOCK + 31) / 32) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
        if (lane == 0) warp_sums[0] = val;
    }
    __syncthreads();
    return warp_sums[0];
}

/* y[row, i] = x[row, i] * w[i] * rsqrt(mean(x[row]^2) + eps).
 * One block per row.  x and y may alias.  w is a per-column F32 vector of
 * length n (host-or-device-reachable). */
template <int BLOCK>
__global__ void ds4_cuda_kernel_rms_norm_w_f32(
        float *y, const float *x, const float *w, uint32_t n, float eps) {
    const uint32_t row = blockIdx.x;
    const float *xp = x + (uint64_t)row * n;
    float *yp = y + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        const float v = xp[i];
        sum += v * v;
    }
    sum = ds4_cuda_block_sum<BLOCK>(sum);
    const float scale = rsqrtf(sum / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        yp[i] = xp[i] * scale * w[i];
    }
}

/* 3D head-wise RMSNorm: x has shape [n_tok, n_head, head_dim] in row-major
 * order.  Each (token, head) row is normalised independently with no learned
 * weight.  Launch with grid=(n_head, n_tok, 1). */
template <int BLOCK>
__global__ void ds4_cuda_kernel_head_rms_norm_f32(
        float *x, uint32_t head_dim, uint32_t n_head, float eps) {
    const uint32_t head = blockIdx.x;
    const uint32_t tok  = blockIdx.y;
    float *row = x + ((uint64_t)tok * n_head + head) * head_dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = row[i];
        sum += v * v;
    }
    sum = ds4_cuda_block_sum<BLOCK>(sum);
    const float scale = rsqrtf(sum / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += BLOCK) {
        row[i] = row[i] * scale;
    }
}

/* Fused Q and KV RMSNorm with learned weight, one dispatch.  grid.x = rows,
 * grid.y = 0 (q-task) or 1 (kv-task).  Mirrors metal/norm.metal:
 *   kernel_dsv4_qkv_rms_norm_f32_4
 * but spelled in scalar form.  Both halves use the same BLOCK-thread
 * reduction. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_qkv_rms_norm_w_f32(
        float *q_out, const float *q_src, const float *q_w, uint32_t q_n,
        float *kv_out, const float *kv_src, const float *kv_w, uint32_t kv_n,
        float eps) {
    const uint32_t row = blockIdx.x;
    const bool kv_task = blockIdx.y != 0;
    const uint32_t n = kv_task ? kv_n : q_n;

    const float *xp = kv_task ? kv_src + (uint64_t)row * kv_n : q_src + (uint64_t)row * q_n;
    const float *wp = kv_task ? kv_w : q_w;
    float       *yp = kv_task ? kv_out + (uint64_t)row * kv_n : q_out + (uint64_t)row * q_n;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        const float v = xp[i];
        sum += v * v;
    }
    sum = ds4_cuda_block_sum<BLOCK>(sum);
    const float scale = rsqrtf(sum / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        yp[i] = xp[i] * scale * wp[i];
    }
}

#endif /* DS4_CUDA_NORM_CUH */
