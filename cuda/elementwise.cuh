/* =========================================================================
 * cuda/elementwise.cuh - simple elementwise / norm / activation kernels.
 * =========================================================================
 *
 * These mirror metal/{bin,glu,norm,unary}.metal closely enough that the math
 * is one-to-one.  More complex kernels (flash_attn, moe, dsv4_hc/kv) live in
 * their own files because their state and shared-memory layout is involved.
 */
#ifndef DS4_CUDA_ELEMENTWISE_CUH
#define DS4_CUDA_ELEMENTWISE_CUH

#include "common.cuh"

/* y[i] = a[i] + b[i] over n contiguous floats. */
__global__ void ds4_cuda_kernel_add_f32(float *y, const float *a, const float *b, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

/* y[i] = silu(gate[i]) * up[i] * weight, with optional gate clamp.
 * Mirrors kernel_swiglu_f32 plus the (clamp, weight) options used by DS4. */
__global__ void ds4_cuda_kernel_swiglu_f32(
        float *y, const float *gate, const float *up,
        uint32_t n, float clamp, float weight) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = gate[i];
    if (clamp > 0.0f) {
        if (g >  clamp) g =  clamp;
        if (g < -clamp) g = -clamp;
    }
    const float silu = g / (1.0f + expf(-g));
    y[i] = silu * up[i] * weight;
}

/* Plain RMSNorm over one row of n floats: y = x * rsqrt(mean(x^2) + eps).
 * One CUDA block per row, blockDim.x threads cooperating with shared memory.
 * Closely follows kernel_rms_norm_f32_4 but uses fp32 throughout for clarity.
 * The DS4 graph also calls a 'rows' variant; that just launches with grid.y
 * equal to the row count. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_rms_norm_plain_f32(
        float *y, const float *x, uint32_t n, float eps) {
    const uint32_t row = blockIdx.x;
    const float *xp = x + row * n;
    float *yp = y + row * n;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        const float v = xp[i];
        sum += v * v;
    }

    __shared__ float ssum[32];
    /* warp reduce */
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) ssum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (threadIdx.x < (BLOCK + 31) / 32) ? ssum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) ssum[0] = sum;
    }
    __syncthreads();
    const float scale = rsqrtf(ssum[0] / (float)n + eps);

    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        yp[i] = xp[i] * scale;
    }
}

/* Same as above but fused with a per-element learned weight (model_map src).
 * The Metal kernel reads weight in float4 chunks; here we go scalar f32 for
 * simplicity until the IO path is profiled. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_rms_norm_weight_f32(
        float *y, const float *x, const float *w, uint32_t n, float eps) {
    const uint32_t row = blockIdx.x;
    const float *xp = x + row * n;
    float *yp = y + row * n;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        const float v = xp[i];
        sum += v * v;
    }

    __shared__ float ssum[32];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) ssum[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (threadIdx.x < (BLOCK + 31) / 32) ? ssum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
        if (lane == 0) ssum[0] = sum;
    }
    __syncthreads();
    const float scale = rsqrtf(ssum[0] / (float)n + eps);

    for (uint32_t i = threadIdx.x; i < n; i += BLOCK) {
        yp[i] = xp[i] * scale * w[i];
    }
}

#endif /* DS4_CUDA_ELEMENTWISE_CUH */
