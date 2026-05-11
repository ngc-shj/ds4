/* =========================================================================
 * cuda/gemm.cuh - cuBLAS-backed dense matmul for the DS4 CUDA backend.
 * =========================================================================
 *
 * Maps ds4_metal_matmul_{f16,f32}_tensor to cublasGemmEx.  Q8_0 / IQ2_XXS /
 * Q2_K / Q4_K dequant + matmul fused kernels stay as stubs for now: each one
 * needs the GGUF block layout that lives in metal/dense.metal and
 * metal/moe.metal.  See PORT_CUDA.md for the tier 1 plan.
 *
 * The math we want is row-major:
 *   y[t, o] = sum_i W[o, i] * x[t, i]
 *
 * Treating row-major (out, in) as column-major (in, out) and similarly for
 * x and y, we get a column-major call:
 *   y'(out x n_tok) = W'(in x out)^T @ x'(in x n_tok)
 * which is exactly cublasGemmEx with op(A)=T, op(B)=N, m=out, n=n_tok, k=in,
 * lda=in, ldb=in, ldc=out.
 */
#ifndef DS4_CUDA_GEMM_CUH
#define DS4_CUDA_GEMM_CUH

#include "common.cuh"
#include <cublas_v2.h>

extern cublasHandle_t ds4_cuda_cublas;

/* Lazily initialise a single cuBLAS handle bound to ds4_cuda_stream.  Called
 * by the matmul wrappers; returns 1 on success. */
static inline int ds4_cuda_cublas_ensure(void) {
    if (ds4_cuda_cublas) return 1;
    cublasStatus_t s = cublasCreate(&ds4_cuda_cublas);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ds4-cuda: cublasCreate failed: %d\n", (int)s);
        return 0;
    }
    s = cublasSetStream(ds4_cuda_cublas, ds4_cuda_stream);
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "ds4-cuda: cublasSetStream failed: %d\n", (int)s);
        return 0;
    }
    /* Default math mode.  Per-call we may want TF32 for F32 GEMM and
     * CUBLAS_DEFAULT_MATH for F16 GEMM; mixing them at handle scope rejects
     * F16 GemmEx with CUBLAS_STATUS_NOT_SUPPORTED on Blackwell. */
    cublasSetMathMode(ds4_cuda_cublas, CUBLAS_DEFAULT_MATH);
    return 1;
}

static inline int ds4_cuda_gemm_check(cublasStatus_t s, const char *label,
                                      uint64_t m, uint64_t n, uint64_t k) {
    if (s == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "ds4-cuda: cublasGemmEx %s failed (%d) m=%llu n=%llu k=%llu\n",
            label, (int)s,
            (unsigned long long)m, (unsigned long long)n, (unsigned long long)k);
    return 0;
}

/* Custom F16 matmul.  cuBLAS GemmEx returns NOT_SUPPORTED for unaligned host
 * GGUF pointers + small-m shapes on this driver, so we keep correctness with
 * a simple block-per-output-row kernel.  Tensor-core / cuBLAS-LT replacement
 * is on PORT_CUDA.md tier 1. */
template <int BLOCK>
__global__ void ds4_cuda_kernel_matmul_f16(
        float *y, const __half *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    const uint64_t row = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    const __half *wr = w + row * in_dim;
    const float  *xt = x + tok * in_dim;

    float partial = 0.0f;
    for (uint64_t i = threadIdx.x; i < in_dim; i += BLOCK) {
        partial += __half2float(wr[i]) * xt[i];
    }
    __shared__ float warp_sum[32];
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sum[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        partial = (threadIdx.x < (BLOCK + 31) / 32) ? warp_sum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffff, partial, off);
        if (lane == 0) y[tok * out_dim + row] = partial;
    }
}

static inline int ds4_cuda_matmul_f16(
        float *y, const __half *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    constexpr int BLK = 128;
    dim3 grid((unsigned)out_dim, (unsigned)n_tok);
    ds4_cuda_kernel_matmul_f16<BLK><<<grid, BLK, 0, ds4_cuda_stream>>>(
        y, w, x, in_dim, out_dim, n_tok);
    return 1;
}

/* Same shape, F32 weights.  TF32 tensor cores give a 2-4x speedup on
 * Blackwell with minimal accuracy loss for non-critical ranges. */
static inline int ds4_cuda_matmul_f32(
        float *y, const float *w, const float *x,
        uint64_t in_dim, uint64_t out_dim, uint64_t n_tok) {
    if (!ds4_cuda_cublas_ensure()) return 0;
    cublasSetMathMode(ds4_cuda_cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t s = cublasGemmEx(
        ds4_cuda_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        (int)out_dim, (int)n_tok, (int)in_dim,
        &alpha,
        w, CUDA_R_32F, (int)in_dim,
        x, CUDA_R_32F, (int)in_dim,
        &beta,
        y, CUDA_R_32F, (int)out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    return ds4_cuda_gemm_check(s, "F32", out_dim, n_tok, in_dim);
}

#endif /* DS4_CUDA_GEMM_CUH */
