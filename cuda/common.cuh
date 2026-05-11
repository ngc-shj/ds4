/* =========================================================================
 * cuda/common.cuh - shared CUDA helpers for the DS4 CUDA backend.
 * =========================================================================
 *
 * The CUDA backend mirrors ds4_metal.m on NVIDIA GPUs.  Initial target is the
 * NVIDIA GB10 (DGX Spark, Grace + Blackwell, sm_121) which has a true unified
 * memory architecture: host pointers from cudaMallocManaged are reachable by
 * the GPU over NVLink-C2C without explicit copies.  This is the analogue of
 * Apple Silicon's shared MTLBuffer storage and is what lets ds4.c keep its
 * tensor_contents()-style direct-pointer access pattern.
 *
 * Code that includes this header is compiled by nvcc and therefore C++.
 */
#ifndef DS4_CUDA_COMMON_CUH
#define DS4_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <stdio.h>

#define DS4_CUDA_CHECK(expr)                                                   \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "ds4-cuda: %s failed at %s:%d: %s\n",              \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_e));        \
            return 0;                                                          \
        }                                                                      \
    } while (0)

#define DS4_CUDA_CHECK_VOID(expr)                                              \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "ds4-cuda: %s failed at %s:%d: %s\n",              \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_e));        \
        }                                                                      \
    } while (0)

/* Stream used for all DS4 compute and copy work.  A single stream keeps order
 * with the Metal-style begin/flush/end model the engine uses; switching to
 * multi-stream pipelining is a later optimisation that needs explicit ordering
 * between KV producers and attention consumers. */
extern cudaStream_t ds4_cuda_stream;
extern int          ds4_cuda_initialized;

/* DS4 GGUF quantization tags reused from ds4_metal.m.  Keep in sync with the
 * Metal backend so model_map offsets resolve identically. */
enum {
    DS4_CUDA_TENSOR_Q8_0    = 8,
    DS4_CUDA_TENSOR_Q2_K    = 10,
    DS4_CUDA_TENSOR_Q4_K    = 12,
    DS4_CUDA_TENSOR_IQ2_XXS = 16,
};

static inline int ds4_cuda_ceil_div(int a, int b) { return (a + b - 1) / b; }

#endif /* DS4_CUDA_COMMON_CUH */
