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

/* In-place FP8 e4m3 round trip across the nope prefix of one KV row, then
 * F16 round trip across the rotated tail, mirroring metal/dsv4_kv.metal's
 * fused store kernel.  After this kernel the caller's kv buffer holds the
 * quantised values that will be written into the raw cache. */
__global__ void ds4_cuda_kernel_fp8_kv_quantize_f32(
        float *kv, uint32_t head_dim, uint32_t n_rot) {
    /* One block, blockDim.x threads, single token.  Each thread walks across
     * head_dim with stride blockDim.x. */
    const uint32_t nope = head_dim - n_rot;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = kv[i];
        if (i < nope) {
            /* FP8 e4m3: 1 sign + 4 exp + 3 mantissa, saturate on overflow,
             * matches metal/dsv4_kv.metal's e4m3 round trip. */
            const __nv_fp8_e4m3 fp8(v);
            kv[i] = __half2float((__half)fp8);
        } else {
            const __half h = __float2half(v);
            kv[i] = __half2float(h);
        }
    }
}

/* Fused FP8 quantise + raw store.  Equivalent to running the FP8 quantise on
 * a working KV vector then writing F16-rounded copy into raw_cache[row, :]. */
__global__ void ds4_cuda_kernel_kv_fp8_store_raw_f32(
        const float *kv, float *raw_cache,
        uint32_t head_dim, uint32_t n_rot, uint32_t row) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= head_dim) return;
    const float v = kv[i];
    float out;
    if (i < head_dim - n_rot) {
        const __nv_fp8_e4m3 fp8(v);
        out = __half2float((__half)fp8);
    } else {
        const __half h = __float2half(v);
        out = __half2float(h);
    }
    raw_cache[(uint64_t)row * head_dim + i] = out;
}

#endif /* DS4_CUDA_KV_CUH */
