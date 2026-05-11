/* =========================================================================
 * cuda/kv.cuh - raw KV cache writes and FP8 KV round-trip.
 * =========================================================================
 *
 * Reference: metal/dsv4_kv.metal and ds4_metal.m's store_raw_kv tensor
 * helpers.  ds4_metal_store_raw_kv* stores an F16-rounded copy of one (or
 * many) KV row(s) into the raw attention cache slot.  The "round trip"
 * matters: it bakes the F32 -> F16 quantisation loss into the cached value so
 * downstream attention reads the same value the future decode path would see.
 *
 * The FP8 variant additionally quantises and dequantises the no-rope tail
 * using DS4's e4m3-style FP8 format before storing.  We model that with the
 * built-in __nv_fp8_e4m3 type, which is bit-identical on Hopper/Blackwell.
 */
#ifndef DS4_CUDA_KV_CUH
#define DS4_CUDA_KV_CUH

#include "common.cuh"
#include <cuda_fp8.h>

/* dst[row, :] = (float)(__half)src[:].  raw_cap is the row capacity of the
 * cache, head_dim is the row width. */
__global__ void ds4_cuda_kernel_store_raw_kv_f32(
        float *raw_cache, const float *kv, uint32_t head_dim, uint32_t row) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= head_dim) return;
    /* F32 -> F16 -> F32 round trip to match the Metal kernel's behaviour. */
    const __half h = __float2half(kv[i]);
    raw_cache[(uint64_t)row * head_dim + i] = __half2float(h);
}

/* Batched form: pos0..pos0+n_tokens-1 written. */
__global__ void ds4_cuda_kernel_store_raw_kv_batch_f32(
        float *raw_cache, const float *kv,
        uint32_t head_dim, uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens) {
    const uint32_t t = blockIdx.y;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tokens || i >= head_dim) return;
    const uint32_t row = pos0 + t;
    if (row >= raw_cap) return;
    const __half h = __float2half(kv[(uint64_t)t * head_dim + i]);
    raw_cache[(uint64_t)row * head_dim + i] = __half2float(h);
}

/* In-place FP8 e4m3 quantise across the nope prefix of one KV row, with
 * per-64-element block scaling (Metal kernel_dsv4_fp8_kv_quantize_f32).
 * The rotated tail is LEFT UNCHANGED -- Metal's kernel copies src->dst for
 * the tail, which is a no-op when called in place. */
__global__ void ds4_cuda_kernel_fp8_kv_quantize_f32(
        float *kv, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t tid = threadIdx.x;
    if (tid >= 64) return;

    __shared__ float scratch[64];

    for (uint32_t off = 0; off < n_nope; off += 64) {
        const uint32_t i = off + tid;
        float v = 0.0f;
        if (i < n_nope) {
            v = kv[i];
            scratch[tid] = fabsf(v);
        } else {
            scratch[tid] = 0.0f;
        }
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (i < n_nope) {
            const float scaled = fmaxf(-448.0f, fminf(448.0f, v / fp8_scale));
            const __nv_fp8_e4m3 fp8(scaled);
            kv[i] = __half2float((__half)fp8) * fp8_scale;
        }
        __syncthreads();
    }
    /* Rotated tail: leave kv[n_nope..head_dim-1] untouched (Metal's src->dst
     * copy is a no-op when called in place, which is the engine's usage). */
}

/* Fused FP8 quantise (nope prefix, with per-64-element scaling) + raw cache
 * F16 round-trip store.  Mirrors metal/dsv4_kv.metal::kernel_dsv4_kv_fp8_store_f32:
 *  - nope prefix: rescale + FP8 round-trip, write back to kv[] (in place) AND
 *    write F16-rounded copy into raw_cache[row, :].
 *  - rotated tail: F16 round-trip from kv[] into raw_cache[row, :] only
 *    (kv[] is not rewritten for the tail).
 * Launch with blockDim.x == 64. */
__global__ void ds4_cuda_kernel_kv_fp8_store_raw_f32(
        float *kv, float *raw_cache,
        uint32_t head_dim, uint32_t n_rot, uint32_t row) {
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t tid = threadIdx.x;
    if (tid >= 64) return;
    float *raw = raw_cache + (uint64_t)row * head_dim;

    __shared__ float scratch[64];

    for (uint32_t off = 0; off < n_nope; off += 64) {
        const uint32_t i = off + tid;
        float v = 0.0f;
        if (i < n_nope) {
            v = kv[i];
            scratch[tid] = fabsf(v);
        } else {
            scratch[tid] = 0.0f;
        }
        __syncthreads();
        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            __syncthreads();
        }
        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float fp8_scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (i < n_nope) {
            const float scaled = fmaxf(-448.0f, fminf(448.0f, v / fp8_scale));
            const __nv_fp8_e4m3 fp8(scaled);
            const float q = __half2float((__half)fp8) * fp8_scale;
            kv[i] = q;
            raw[i] = __half2float(__float2half(q));
        }
        __syncthreads();
    }
    /* Rotated tail: F16 round-trip from kv[] into raw cache (kv unchanged). */
    for (uint32_t i = n_nope + tid; i < head_dim; i += 64) {
        raw[i] = __half2float(__float2half(kv[i]));
    }
}

#endif /* DS4_CUDA_KV_CUH */
