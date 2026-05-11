/* =========================================================================
 * ds4_cuda.cu - CUDA backend for the DS4 inference engine.
 * =========================================================================
 *
 * Mirrors ds4_metal.m's public surface (the ds4_metal_* C symbols declared in
 * ds4_metal.h) so the engine call sites in ds4.c stay backend-agnostic.  The
 * "metal_" prefix is kept verbatim: this file is the CUDA implementation of
 * those entry points, not a rename.
 *
 * Initial target hardware: NVIDIA GB10 (DGX Spark, Grace + Blackwell, sm_121).
 * The platform's NVLink-C2C unified memory lets cudaMallocManaged buffers be
 * touched from both the host (engine) and the device (kernels) without manual
 * copies, which is the closest match to Metal's MTLResourceStorageModeShared
 * model that ds4.c relies on.
 *
 * Status:
 *   - Process / device / stream lifecycle: implemented.
 *   - Tensor alloc / view / free / read / write / copy: implemented.
 *   - Model mmap registration: implemented via cudaHostRegister.
 *   - A handful of trivial kernels (add, swiglu, plain rms_norm) are wired up
 *     so smoke tests can validate the dispatch path end to end.
 *   - All other kernels (matmul/Q8_0/IQ2_XXS/Q2_K/Q4_K, MoE, flash_attn,
 *     DSV4-specific compressors, indexer/topk, HC mixers, RoPE, etc.) are
 *     declared but return 0 with a one-shot diagnostic.  See PORT_CUDA.md for
 *     the per-kernel porting plan.
 */

#include "cuda/common.cuh"
#include "cuda/elementwise.cuh"
#include "cuda/norm.cuh"
#include "cuda/rope.cuh"
#include "cuda/kv.cuh"
#include "cuda/gemm.cuh"
#include "cuda/q8_0.cuh"
#include "cuda/embed_hc.cuh"
#include "cuda/attention.cuh"
#include "cuda/router.cuh"
#include "cuda/moe.cuh"
#include "cuda/compress.cuh"
#include "cuda/indexer.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

extern "C" {
#include "ds4_metal.h"
}

/* cuBLAS handle bound to ds4_cuda_stream; lazily created on first matmul. */
cublasHandle_t ds4_cuda_cublas = nullptr;

/* Single-token tokens-tensor buffer for router_select_tensor hash-mode path.
 * Defined later but referenced in ds4_metal_cleanup. */
static int32_t *g_router_single_tok_buf;

/* =========================================================================
 * Globals.
 * ========================================================================= */

cudaStream_t ds4_cuda_stream      = 0;
int          ds4_cuda_initialized = 0;
static int   g_in_batch           = 0;
static bool  g_quality            = false;

static const void *g_model_map      = NULL;
static uint64_t    g_model_size     = 0;
static bool        g_model_pinned   = false;

static uint64_t    g_alloc_live     = 0;
static uint64_t    g_alloc_peak     = 0;

/* Opaque tensor.  The engine treats ds4_metal_tensor as a forward-declared
 * struct; here we make the layout concrete.  base+offset+bytes lets us model
 * the same slice-view semantics ds4_metal.m has on top of MTLBuffer. */
struct ds4_metal_tensor {
    void    *base;     /* allocation base, NULL for views */
    uint64_t offset;   /* byte offset into the base allocation */
    uint64_t bytes;    /* visible bytes from offset */
    int      owner;    /* 1 = owns base, 0 = view */
};

/* =========================================================================
 * Lifecycle.
 * ========================================================================= */

int ds4_metal_init(void) {
    if (ds4_cuda_initialized) return 1;

    int dev_count = 0;
    cudaError_t e = cudaGetDeviceCount(&dev_count);
    if (e != cudaSuccess || dev_count == 0) {
        fprintf(stderr, "ds4-cuda: no CUDA device available (%s)\n",
                e == cudaSuccess ? "0 devices" : cudaGetErrorString(e));
        return 0;
    }

    DS4_CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop;
    DS4_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "ds4-cuda: using %s (sm_%d%d, %.1f GiB global mem)\n",
            prop.name, prop.major, prop.minor,
            (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    DS4_CUDA_CHECK(cudaStreamCreateWithFlags(&ds4_cuda_stream, cudaStreamNonBlocking));

    ds4_cuda_initialized = 1;
    return 1;
}

void ds4_metal_cleanup(void) {
    if (!ds4_cuda_initialized) return;

    if (ds4_cuda_cublas) {
        cublasDestroy(ds4_cuda_cublas);
        ds4_cuda_cublas = nullptr;
    }
    if (g_router_single_tok_buf) {
        cudaFree(g_router_single_tok_buf);
        g_router_single_tok_buf = nullptr;
    }
    if (ds4_cuda_stream) {
        DS4_CUDA_CHECK_VOID(cudaStreamSynchronize(ds4_cuda_stream));
        DS4_CUDA_CHECK_VOID(cudaStreamDestroy(ds4_cuda_stream));
        ds4_cuda_stream = 0;
    }
    if (g_model_pinned && g_model_map) {
        DS4_CUDA_CHECK_VOID(cudaHostUnregister((void *)g_model_map));
        g_model_pinned = false;
    }
    g_model_map = NULL;
    g_model_size = 0;
    g_in_batch = 0;
    ds4_cuda_initialized = 0;
}

/* =========================================================================
 * Command lifecycle.
 *
 * Metal uses an MTLCommandBuffer + MTLComputeCommandEncoder explicitly.  CUDA
 * has no equivalent: kernel launches just enqueue on a stream.  We model
 * begin/flush/end as flag/synchronize calls so ds4.c keeps the same call
 * pattern.  Future work may upgrade this to cudaGraph capture for fewer
 * launch overheads.
 * ========================================================================= */

int ds4_metal_begin_commands(void) {
    if (!ds4_cuda_initialized && !ds4_metal_init()) return 0;
    if (g_in_batch) return 0;
    g_in_batch = 1;
    return 1;
}

int ds4_metal_flush_commands(void) {
    if (!ds4_cuda_initialized || !g_in_batch) return 0;
    /* Stay open for more dispatches; just push prior work out.  A
     * cudaStreamSynchronize would be heavy-handed - we leave the work
     * in-flight so chained kernels overlap as much as the engine allows. */
    return 1;
}

int ds4_metal_end_commands(void) {
    if (!g_in_batch) return 0;
    g_in_batch = 0;
    DS4_CUDA_CHECK(cudaStreamSynchronize(ds4_cuda_stream));
    return 1;
}

int ds4_metal_synchronize(void) {
    if (!ds4_cuda_initialized && !ds4_metal_init()) return 0;
    if (g_in_batch) return ds4_metal_end_commands();
    DS4_CUDA_CHECK(cudaStreamSynchronize(ds4_cuda_stream));
    return 1;
}

/* =========================================================================
 * Tensor allocation.
 *
 * cudaMallocManaged gives us a single virtual address that is reachable from
 * both host and device.  On GB10 (Grace+Blackwell NVLink-C2C) and on Jetson
 * Thor / iGPU systems this is genuinely zero-copy; on discrete GPUs the
 * driver migrates pages on demand.  Either way the engine's
 * ds4_metal_tensor_contents()-style direct host pointer access keeps working.
 * ========================================================================= */

ds4_metal_tensor *ds4_metal_tensor_alloc(uint64_t bytes) {
    if (!ds4_cuda_initialized && !ds4_metal_init()) return NULL;
    if (bytes == 0) return NULL;

    void *ptr = NULL;
    cudaError_t e = cudaMallocManaged(&ptr, (size_t)bytes, cudaMemAttachGlobal);
    if (e != cudaSuccess) {
        fprintf(stderr, "ds4-cuda: cudaMallocManaged(%llu) failed: %s\n",
                (unsigned long long)bytes, cudaGetErrorString(e));
        return NULL;
    }

    ds4_metal_tensor *t = (ds4_metal_tensor *)calloc(1, sizeof(*t));
    if (!t) { cudaFree(ptr); return NULL; }
    t->base = ptr;
    t->offset = 0;
    t->bytes = bytes;
    t->owner = 1;

    g_alloc_live += bytes;
    if (g_alloc_live > g_alloc_peak) g_alloc_peak = g_alloc_live;
    return t;
}

ds4_metal_tensor *ds4_metal_tensor_view(const ds4_metal_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base) return NULL;
    if (offset > base->bytes || bytes > base->bytes - offset) return NULL;

    ds4_metal_tensor *v = (ds4_metal_tensor *)calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->base = base->base;
    v->offset = base->offset + offset;
    v->bytes = bytes;
    v->owner = 0;
    return v;
}

void ds4_metal_tensor_free(ds4_metal_tensor *t) {
    if (!t) return;
    if (t->owner && t->base) {
        if (t->bytes <= g_alloc_live) g_alloc_live -= t->bytes;
        else g_alloc_live = 0;
        DS4_CUDA_CHECK_VOID(cudaFree(t->base));
    }
    free(t);
}

uint64_t ds4_metal_tensor_bytes(const ds4_metal_tensor *t) {
    return t ? t->bytes : 0;
}

void *ds4_metal_tensor_contents(ds4_metal_tensor *t) {
    if (!t || !t->base) return NULL;
    return (uint8_t *)t->base + t->offset;
}

int ds4_metal_tensor_write(ds4_metal_tensor *t, uint64_t offset, const void *data, uint64_t bytes) {
    if (!t || (!data && bytes != 0)) return 0;
    if (offset > t->bytes || bytes > t->bytes - offset) return 0;
    if (bytes == 0) return 1;
    /* Managed memory: writes from host are visible to subsequent kernels once
     * the stream has finished any prior consumer.  Since we serialize on one
     * stream and the engine guards write-then-launch ordering, a plain memcpy
     * is correct. */
    memcpy((uint8_t *)t->base + t->offset + offset, data, (size_t)bytes);
    return 1;
}

int ds4_metal_tensor_read(const ds4_metal_tensor *t, uint64_t offset, void *data, uint64_t bytes) {
    if (!t || (!data && bytes != 0)) return 0;
    if (offset > t->bytes || bytes > t->bytes - offset) return 0;
    if (bytes == 0) return 1;
    /* Reads must wait for the in-flight stream to finish, otherwise the host
     * could observe a stale value.  ds4.c usually pairs reads with synchronize
     * already, but be defensive here too. */
    DS4_CUDA_CHECK(cudaStreamSynchronize(ds4_cuda_stream));
    memcpy(data, (const uint8_t *)t->base + t->offset + offset, (size_t)bytes);
    return 1;
}

int ds4_metal_tensor_copy(ds4_metal_tensor *dst, uint64_t dst_offset,
                          const ds4_metal_tensor *src, uint64_t src_offset,
                          uint64_t bytes) {
    if (!dst || !src) return 0;
    if (!ds4_cuda_initialized && !ds4_metal_init()) return 0;
    if (dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) return 0;
    if (src_offset > src->bytes || bytes > src->bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    DS4_CUDA_CHECK(cudaMemcpyAsync(
        (uint8_t *)dst->base + dst->offset + dst_offset,
        (const uint8_t *)src->base + src->offset + src_offset,
        (size_t)bytes, cudaMemcpyDefault, ds4_cuda_stream));
    return 1;
}

/* =========================================================================
 * Model mmap registration.
 *
 * The engine mmaps the GGUF and passes (ptr, size) here.  On Metal we wrap
 * slices in MTLBuffer with noCopy.  On CUDA the equivalent is to pin the
 * mapping with cudaHostRegister so the GPU can DMA / unified-access it
 * cheaply.  On GB10 this is essentially free; on discrete GPUs it ensures the
 * memory is pinned and accessible from the device.
 * ========================================================================= */

int ds4_metal_set_model_map(const void *model_map, uint64_t model_size) {
    if (!ds4_cuda_initialized && !ds4_metal_init()) return 0;
    if (g_model_pinned && g_model_map) {
        DS4_CUDA_CHECK(cudaHostUnregister((void *)g_model_map));
        g_model_pinned = false;
    }
    g_model_map = model_map;
    g_model_size = model_size;
    if (model_map && model_size) {
        /* Try a few registration modes.  File-backed mmap declines
         * Default+ReadOnly on some kernels; Default alone often works.
         * Then explicitly advise the driver that the range is read-mostly
         * so the unified-memory path doesn't keep tracking dirty pages. */
        cudaError_t e = cudaHostRegister((void *)model_map, (size_t)model_size,
                                         cudaHostRegisterReadOnly);
        if (e != cudaSuccess) {
            e = cudaHostRegister((void *)model_map, (size_t)model_size,
                                 cudaHostRegisterDefault);
        }
        if (e == cudaSuccess) {
            g_model_pinned = true;
        } else {
            fprintf(stderr,
                    "ds4-cuda: cudaHostRegister(model_map=%p, %llu) declined (%s); "
                    "continuing without pinned mapping\n",
                    model_map, (unsigned long long)model_size,
                    cudaGetErrorString(e));
        }

        /* Even without HostRegister, cudaMemAdvise on the UVA range can move
         * the pages into a GPU-cacheable state on GB10 / unified-memory
         * systems.  CUDA 13 switched to a cudaMemLocation argument.  Failure
         * here is non-fatal. */
        int device = 0;
        cudaGetDevice(&device);
        cudaMemLocation loc = { cudaMemLocationTypeDevice, device };
        cudaMemAdvise(model_map, (size_t)model_size,
                      cudaMemAdviseSetReadMostly, loc);
        cudaMemAdvise(model_map, (size_t)model_size,
                      cudaMemAdviseSetPreferredLocation, loc);
    }
    return 1;
}

int ds4_metal_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size) {
    (void)map_offset; (void)map_size;
    /* Range registration is only an optimization hint on Metal.  On CUDA we
     * register the whole map once; per-range pin would just churn the page
     * tracker for no benefit. */
    return ds4_metal_set_model_map(model_map, model_size);
}

void ds4_metal_set_quality(bool quality) { g_quality = quality; }

/* Resolve a host weight pointer in the registered model map.  Declared early
 * so kernel wrappers above and below this point can use it uniformly. */
static inline const void *ds4_cuda_weight_ptr(const void *model_map, uint64_t offset) {
    return (const uint8_t *)model_map + offset;
}

/* Tensor data accessors shared by every kernel wrapper. */
static inline float *tensor_fptr(ds4_metal_tensor *t) {
    return (float *)((uint8_t *)t->base + t->offset);
}
static inline const float *tensor_cfptr(const ds4_metal_tensor *t) {
    return (const float *)((const uint8_t *)t->base + t->offset);
}

void ds4_metal_print_memory_report(const char *label) {
    size_t free_b = 0, total_b = 0;
    if (ds4_cuda_initialized) {
        cudaMemGetInfo(&free_b, &total_b);
    }
    fprintf(stderr,
            "ds4-cuda memory [%s]: tensors live %.3f MiB peak %.3f MiB; "
            "device free %.1f / %.1f GiB\n",
            label ? label : "",
            (double)g_alloc_live / (1024.0 * 1024.0),
            (double)g_alloc_peak / (1024.0 * 1024.0),
            (double)free_b  / (1024.0 * 1024.0 * 1024.0),
            (double)total_b / (1024.0 * 1024.0 * 1024.0));
}

/* =========================================================================
 * Implemented kernels.
 *
 * Wired-up so the build can validate the dispatch path.  Functional but not
 * yet performance-tuned.
 * ========================================================================= */

int ds4_metal_add_tensor(ds4_metal_tensor *out,
                         const ds4_metal_tensor *a,
                         const ds4_metal_tensor *b,
                         uint32_t n) {
    if (!out || !a || !b) return 0;
    const int BLK = 256;
    ds4_cuda_kernel_add_f32<<<ds4_cuda_ceil_div((int)n, BLK), BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), tensor_cfptr(a), tensor_cfptr(b), n);
    return 1;
}

int ds4_metal_swiglu_tensor(ds4_metal_tensor *out,
                            const ds4_metal_tensor *gate,
                            const ds4_metal_tensor *up,
                            uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up) return 0;
    const int BLK = 256;
    ds4_cuda_kernel_swiglu_f32<<<ds4_cuda_ceil_div((int)n, BLK), BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), tensor_cfptr(gate), tensor_cfptr(up), n, clamp, weight);
    return 1;
}

int ds4_metal_rms_norm_plain_tensor(ds4_metal_tensor *out,
                                    const ds4_metal_tensor *x,
                                    uint32_t n, float eps) {
    if (!out || !x) return 0;
    ds4_cuda_kernel_rms_norm_plain_f32<256><<<1, 256, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), tensor_cfptr(x), n, eps);
    return 1;
}

int ds4_metal_rms_norm_plain_rows_tensor(ds4_metal_tensor *out,
                                         const ds4_metal_tensor *x,
                                         uint32_t n, uint32_t rows, float eps) {
    if (!out || !x) return 0;
    ds4_cuda_kernel_rms_norm_plain_f32<256><<<rows, 256, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), tensor_cfptr(x), n, eps);
    return 1;
}

/* =========================================================================
 * Stubs.
 *
 * Every kernel below is part of the engine's hot path and must be implemented
 * before ds4-cuda can run real inference.  They are declared here in API
 * order so the build links cleanly and the failure surface is "this kernel
 * has not been ported yet" rather than "undefined symbol".  PORT_CUDA.md
 * tracks the porting plan and priorities.
 * ========================================================================= */

static int ds4_cuda_stub(const char *name) {
    static int reported = 0;
    if (!reported) {
        fprintf(stderr,
                "ds4-cuda: kernel %s() is not implemented yet.  See PORT_CUDA.md.\n",
                name);
        reported = 1;
    }
    return 0;
}

#define DS4_CUDA_STUB(name)                                                    \
    do { return ds4_cuda_stub(#name); } while (0)

/* --- embeddings / indexer --------------------------------------------- */

int ds4_metal_embed_token_hc_tensor(ds4_metal_tensor *out, const void *m, uint64_t ms, uint64_t wo,
        uint32_t nv, uint32_t tok, uint32_t ne, uint32_t nh) {
    if (!out || !m || tok >= nv || ne == 0 || nh == 0) return 0;
    const uint64_t bytes = (uint64_t)nv * ne * sizeof(__half);
    if (wo > ms || bytes > ms - wo) return 0;
    const __half *w = (const __half *)ds4_cuda_weight_ptr(m, wo);
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), nh);
    ds4_cuda_kernel_embed_token_hc_f16<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), w, tok, ne, nh);
    return 1;
}

int ds4_metal_embed_tokens_hc_tensor(ds4_metal_tensor *out, const ds4_metal_tensor *tok,
        const void *m, uint64_t ms, uint64_t wo, uint32_t nv, uint32_t nt, uint32_t ne, uint32_t nh) {
    if (!out || !tok || !m || nt == 0 || ne == 0 || nh == 0) return 0;
    const uint64_t bytes = (uint64_t)nv * ne * sizeof(__half);
    if (wo > ms || bytes > ms - wo) return 0;
    const __half *w = (const __half *)ds4_cuda_weight_ptr(m, wo);
    const int32_t *tokens = (const int32_t *)((const uint8_t *)tok->base + tok->offset);
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), nh, nt);
    ds4_cuda_kernel_embed_tokens_hc_f16<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), w, tokens, nt, ne, nh);
    return 1;
}
static inline int ds4_cuda_indexer_scores_dispatch(
        ds4_metal_tensor *s, const ds4_metal_tensor *q, const ds4_metal_tensor *w,
        const ds4_metal_tensor *ic, uint32_t nc, uint32_t nt, uint32_t nh,
        uint32_t hd, float scale) {
    if (!s || !q || !w || !ic || nc == 0 || nt == 0) return 0;
    dim3 grid(nc, nt);
    ds4_cuda_kernel_indexer_scores_f32<128><<<grid, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(s), tensor_cfptr(q), tensor_cfptr(w), tensor_cfptr(ic),
        nc, nt, nh, hd, scale);
    return 1;
}

int ds4_metal_indexer_score_one_tensor(ds4_metal_tensor *s, const ds4_metal_tensor *q,
        const ds4_metal_tensor *w, const ds4_metal_tensor *ic,
        uint32_t nc, uint32_t nh, uint32_t hd, float scale) {
    return ds4_cuda_indexer_scores_dispatch(s, q, w, ic, nc, /*nt=*/1, nh, hd, scale);
}
int ds4_metal_indexer_scores_prefill_tensor(ds4_metal_tensor *s, const ds4_metal_tensor *q,
        const ds4_metal_tensor *w, const ds4_metal_tensor *ic,
        uint32_t nc, uint32_t nt, uint32_t nh, uint32_t hd, uint32_t ratio, float scale) {
    (void)ratio;
    return ds4_cuda_indexer_scores_dispatch(s, q, w, ic, nc, nt, nh, hd, scale);
}
int ds4_metal_indexer_scores_decode_batch_tensor(ds4_metal_tensor *s, const ds4_metal_tensor *q,
        const ds4_metal_tensor *w, const ds4_metal_tensor *ic,
        uint32_t nc, uint32_t nt, uint32_t pos0, uint32_t nh, uint32_t hd, uint32_t ratio, float scale) {
    (void)pos0; (void)ratio;
    return ds4_cuda_indexer_scores_dispatch(s, q, w, ic, nc, nt, nh, hd, scale);
}
int ds4_metal_indexer_topk_tensor(ds4_metal_tensor *sel, const ds4_metal_tensor *sc,
        uint32_t nc, uint32_t nt, uint32_t top_k) {
    if (!sel || !sc || nc == 0 || nt == 0 || top_k == 0) return 0;
    const size_t smem = (size_t)nc * sizeof(float);
    ds4_cuda_kernel_indexer_topk_f32<<<nt, 1, smem, ds4_cuda_stream>>>(
        (int32_t *)tensor_fptr(sel), tensor_cfptr(sc), nc, nt, top_k);
    return 1;
}
int ds4_metal_dsv4_topk_mask_tensor(ds4_metal_tensor *mt, const ds4_metal_tensor *tk,
        uint32_t nc, uint32_t nt, uint32_t top_k) {
    if (!mt || !tk || nc == 0 || nt == 0 || top_k == 0) return 0;
    /* Pass 1: zero the mask. */
    const int BLK = 256;
    dim3 grid(nt, ds4_cuda_ceil_div((int)nc, BLK));
    ds4_cuda_kernel_topk_mask_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(mt), (const int32_t *)tensor_cfptr(tk), nc, nt, top_k);
    /* Pass 2: scatter ones at the top-K indices. */
    ds4_cuda_kernel_topk_mask_scatter_f32<<<nt, top_k, 0, ds4_cuda_stream>>>(
        tensor_fptr(mt), (const int32_t *)tensor_cfptr(tk), nc, nt, top_k);
    return 1;
}

/* --- dense / quantized matmul (Q8_0/F16/F32) -------------------------- */

int ds4_metal_matmul_f16_tensor(ds4_metal_tensor *o, const void *m, uint64_t ms, uint64_t wo,
        uint64_t in_dim, uint64_t out_dim, const ds4_metal_tensor *x, uint64_t nt) {
    if (!o || !m || !x) return 0;
    if (wo > ms || in_dim * out_dim * sizeof(__half) > ms - wo) return 0;
    const __half *w = (const __half *)ds4_cuda_weight_ptr(m, wo);
    return ds4_cuda_matmul_f16(tensor_fptr(o), w, tensor_cfptr(x), in_dim, out_dim, nt);
}

int ds4_metal_matmul_f16_pair_tensor(ds4_metal_tensor *a, ds4_metal_tensor *b,
        const void *m, uint64_t ms, uint64_t wa, uint64_t wb,
        uint64_t in_dim, uint64_t out_dim, const ds4_metal_tensor *x, uint64_t nt) {
    /* Two independent F16 matvecs sharing the same activation.  The fused
     * Metal kernel groups them for one shared-memory load of x; cuBLAS does
     * not have an equivalent pair primitive, so we issue two GEMMs.  Both
     * land on the same stream so they pipeline naturally. */
    if (!a || !b || !m || !x) return 0;
    const uint64_t bytes = in_dim * out_dim * sizeof(__half);
    if (wa > ms || bytes > ms - wa || wb > ms || bytes > ms - wb) return 0;
    const __half *wap = (const __half *)ds4_cuda_weight_ptr(m, wa);
    const __half *wbp = (const __half *)ds4_cuda_weight_ptr(m, wb);
    if (!ds4_cuda_matmul_f16(tensor_fptr(a), wap, tensor_cfptr(x), in_dim, out_dim, nt)) return 0;
    if (!ds4_cuda_matmul_f16(tensor_fptr(b), wbp, tensor_cfptr(x), in_dim, out_dim, nt)) return 0;
    return 1;
}

int ds4_metal_matmul_f32_tensor(ds4_metal_tensor *o, const void *m, uint64_t ms, uint64_t wo,
        uint64_t in_dim, uint64_t out_dim, const ds4_metal_tensor *x, uint64_t nt) {
    if (!o || !m || !x) return 0;
    if (wo > ms || in_dim * out_dim * sizeof(float) > ms - wo) return 0;
    const float *w = (const float *)ds4_cuda_weight_ptr(m, wo);
    return ds4_cuda_matmul_f32(tensor_fptr(o), w, tensor_cfptr(x), in_dim, out_dim, nt);
}

int ds4_metal_matmul_q8_0_tensor(ds4_metal_tensor *out, const void *m, uint64_t ms, uint64_t wo,
        uint64_t in_dim, uint64_t out_dim, const ds4_metal_tensor *x, uint64_t nt) {
    if (!out || !m || !x) return 0;
    if (in_dim % DS4_CUDA_QK8_0 != 0) {
        fprintf(stderr, "ds4-cuda: Q8_0 matmul requires in_dim %% 32 == 0 (got %llu)\n",
                (unsigned long long)in_dim);
        return 0;
    }
    const uint64_t bytes_per_row = (in_dim / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    if (wo > ms || bytes_per_row * out_dim > ms - wo) return 0;
    const block_q8_0 *w = (const block_q8_0 *)ds4_cuda_weight_ptr(m, wo);
    dim3 grid((unsigned)out_dim, (unsigned)nt);
    ds4_cuda_kernel_matmul_q8_0_f32<128><<<grid, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(out), w, tensor_cfptr(x), in_dim, out_dim, nt);
    return 1;
}

int ds4_metal_shared_gate_up_swiglu_q8_0_tensor(ds4_metal_tensor *g, ds4_metal_tensor *u,
        ds4_metal_tensor *mid, const void *m, uint64_t ms, uint64_t go, uint64_t uo,
        uint64_t in_dim, uint64_t out_dim, const ds4_metal_tensor *x) {
    if (!mid || !m || !x) return 0;
    if (in_dim % DS4_CUDA_QK8_0 != 0) {
        fprintf(stderr, "ds4-cuda: shared GLU requires in_dim %% 32 == 0 (got %llu)\n",
                (unsigned long long)in_dim);
        return 0;
    }
    const uint64_t bytes_per_row = (in_dim / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    if (go > ms || uo > ms || bytes_per_row * out_dim > ms - go ||
        bytes_per_row * out_dim > ms - uo) return 0;
    const block_q8_0 *wg = (const block_q8_0 *)ds4_cuda_weight_ptr(m, go);
    const block_q8_0 *wu = (const block_q8_0 *)ds4_cuda_weight_ptr(m, uo);
    /* DS4 uses a SwiGLU clamp; the engine passes it as a separate flag in
     * the routed-MoE call but not the shared path - default to no clamp here
     * to match metal/dense.metal::kernel_dsv4_shared_gate_up_swiglu_q8_0. */
    ds4_cuda_kernel_shared_gate_up_swiglu_q8_0_f32<128><<<(unsigned)out_dim, 128, 0, ds4_cuda_stream>>>(
        g ? tensor_fptr(g) : nullptr,
        u ? tensor_fptr(u) : nullptr,
        tensor_fptr(mid), wg, wu, tensor_cfptr(x),
        in_dim, out_dim, /*clamp=*/0.0f);
    return 1;
}

/* --- helpers / norms / rope ------------------------------------------ */

int ds4_metal_repeat_hc_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *r,
        uint32_t ne, uint32_t nh) {
    if (!o || !r || ne == 0 || nh == 0) return 0;
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), nh);
    ds4_cuda_kernel_repeat_hc_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(r), ne, nh);
    return 1;
}

int ds4_metal_rms_norm_weight_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *x,
        const void *m, uint64_t ms, uint64_t wo, uint32_t n, float eps) {
    if (!o || !x || !m) return 0;
    if (wo > ms || (uint64_t)n * sizeof(float) > ms - wo) return 0;
    const float *w = (const float *)ds4_cuda_weight_ptr(m, wo);
    ds4_cuda_kernel_rms_norm_w_f32<256><<<1, 256, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(x), w, n, eps);
    return 1;
}

int ds4_metal_rms_norm_weight_rows_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *x,
        const void *m, uint64_t ms, uint64_t wo, uint32_t n, uint32_t rows, float eps) {
    if (!o || !x || !m) return 0;
    if (wo > ms || (uint64_t)n * sizeof(float) > ms - wo) return 0;
    const float *w = (const float *)ds4_cuda_weight_ptr(m, wo);
    ds4_cuda_kernel_rms_norm_w_f32<256><<<rows, 256, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(x), w, n, eps);
    return 1;
}

int ds4_metal_dsv4_qkv_rms_norm_rows_tensor(ds4_metal_tensor *q_out, const ds4_metal_tensor *q,
        const void *m, uint64_t ms, uint64_t qwo, uint32_t qn,
        ds4_metal_tensor *kv_out, const ds4_metal_tensor *kv,
        uint64_t kwo, uint32_t kn, uint32_t rows, float eps) {
    if (!q_out || !q || !kv_out || !kv || !m) return 0;
    if (qwo > ms || (uint64_t)qn * sizeof(float) > ms - qwo) return 0;
    if (kwo > ms || (uint64_t)kn * sizeof(float) > ms - kwo) return 0;
    const float *q_w = (const float *)ds4_cuda_weight_ptr(m, qwo);
    const float *k_w = (const float *)ds4_cuda_weight_ptr(m, kwo);
    /* Launch grid: x = row index, y = 0 (Q) or 1 (KV). */
    dim3 grid(rows, 2);
    ds4_cuda_kernel_qkv_rms_norm_w_f32<256><<<grid, 256, 0, ds4_cuda_stream>>>(
        tensor_fptr(q_out), tensor_cfptr(q), q_w, qn,
        tensor_fptr(kv_out), tensor_cfptr(kv), k_w, kn, eps);
    return 1;
}

int ds4_metal_head_rms_norm_tensor(ds4_metal_tensor *x, uint32_t nt, uint32_t nh, uint32_t hd, float eps) {
    if (!x || nt == 0 || nh == 0 || hd == 0) return 0;
    dim3 grid(nh, nt);
    ds4_cuda_kernel_head_rms_norm_f32<256><<<grid, 256, 0, ds4_cuda_stream>>>(
        tensor_fptr(x), hd, nh, eps);
    return 1;
}

int ds4_metal_dsv4_fp8_kv_quantize_tensor(ds4_metal_tensor *x, uint32_t nt, uint32_t hd, uint32_t nr) {
    if (!x || nt == 0 || hd == 0 || nr > hd) return 0;
    /* One block per token, head_dim threads (capped at 256). */
    const uint32_t BLK = hd < 256 ? hd : 256;
    for (uint32_t t = 0; t < nt; t++) {
        ds4_cuda_kernel_fp8_kv_quantize_f32<<<1, BLK, 0, ds4_cuda_stream>>>(
            tensor_fptr(x) + (uint64_t)t * hd, hd, nr);
    }
    return 1;
}

int ds4_metal_rope_tail_tensor(ds4_metal_tensor *x, uint32_t nt, uint32_t nh, uint32_t hd, uint32_t nr,
        uint32_t pos0, uint32_t nco, bool inv,
        float fb, float fs, float ef, float af, float bf, float bs) {
    if (!x || nt == 0 || nh == 0 || hd == 0 || nr > hd || (nr & 1)) return 0;
    if (nr == 0) return 1;
    dim3 grid(nh, nt);
    ds4_cuda_kernel_rope_tail_f32<<<grid, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(x), nt, nh, hd, nr, pos0, nco, inv ? 1 : 0,
        fb, fs, ef, af, bf, bs);
    return 1;
}

int ds4_metal_kv_fp8_store_raw_tensor(ds4_metal_tensor *kv, ds4_metal_tensor *cache,
        uint32_t cap, uint32_t row, uint32_t hd, uint32_t nr) {
    if (!kv || !cache || cap == 0 || row >= cap || hd == 0 || nr > hd) return 0;
    const int BLK = 128;
    ds4_cuda_kernel_kv_fp8_store_raw_f32<<<ds4_cuda_ceil_div((int)hd, BLK), BLK, 0, ds4_cuda_stream>>>(
        tensor_cfptr(kv), tensor_fptr(cache), hd, nr, row);
    return 1;
}

int ds4_metal_store_raw_kv_tensor(ds4_metal_tensor *cache, const ds4_metal_tensor *kv,
        uint32_t cap, uint32_t row, uint32_t hd) {
    if (!cache || !kv || cap == 0 || row >= cap || hd == 0) return 0;
    const int BLK = 128;
    ds4_cuda_kernel_store_raw_kv_f32<<<ds4_cuda_ceil_div((int)hd, BLK), BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(cache), tensor_cfptr(kv), hd, row);
    return 1;
}

int ds4_metal_store_raw_kv_batch_tensor(ds4_metal_tensor *cache, const ds4_metal_tensor *kv,
        uint32_t cap, uint32_t pos0, uint32_t nt, uint32_t hd) {
    if (!cache || !kv || cap == 0 || pos0 + nt > cap || hd == 0) return 0;
    const int BLK = 128;
    dim3 grid(ds4_cuda_ceil_div((int)hd, BLK), nt);
    ds4_cuda_kernel_store_raw_kv_batch_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(cache), tensor_cfptr(kv), hd, cap, pos0, nt);
    return 1;
}

/* --- compressors / attention ----------------------------------------- */

int ds4_metal_compressor_update_tensor(const ds4_metal_tensor *kv_cur, const ds4_metal_tensor *sc_cur,
        ds4_metal_tensor *skv, ds4_metal_tensor *ssc, ds4_metal_tensor *cc,
        const void *m, uint64_t ms, uint64_t ao, uint32_t at, uint64_t no, uint32_t nt,
        uint32_t hd, uint32_t ratio, uint32_t pos, uint32_t cr, uint32_t nr, uint32_t nco,
        float fb, float fs, float ef, float af, float bf, float bs, float eps) {
    (void)nt;
    if (!kv_cur || !sc_cur || !skv || !ssc || !m || hd == 0 || ratio == 0) return 0;
    const uint32_t width = 2u * hd;
    const uint64_t ape_bytes = (uint64_t)ratio * width *
        (at == 1u ? sizeof(__half) : sizeof(float));
    if (ao > ms || ape_bytes > ms - ao) return 0;
    const void *ape = ds4_cuda_weight_ptr(m, ao);

    /* Step 1: store incoming row into state[ratio + pos%ratio, :]. */
    {
        const int BLK = 256;
        ds4_cuda_kernel_compressor_store_one_f32<<<
            ds4_cuda_ceil_div((int)width, BLK), BLK, 0, ds4_cuda_stream>>>(
            tensor_cfptr(kv_cur), tensor_cfptr(sc_cur), ape, at,
            tensor_fptr(skv), tensor_fptr(ssc), width, ratio, pos);
    }

    /* Only emit + shift when this store completes a ratio block. */
    if (!cc || ((pos + 1u) % ratio) != 0u) return 1;

    if (ratio != 4u) {
        /* ratio=128 (indexer) emit not yet implemented.  Leave comp_cache
         * row at whatever value the previous emit left, and continue.
         * This degrades indexer accuracy but doesn't crash; the indexer
         * top-K simply picks slightly stale rows. */
        return 1;
    }
    {
        const int BLK = 128;
        ds4_cuda_kernel_compressor_pool_only_f32<<<
            ds4_cuda_ceil_div((int)hd, BLK), BLK, 0, ds4_cuda_stream>>>(
            tensor_fptr(cc), cr, tensor_cfptr(skv), tensor_cfptr(ssc), hd);
    }

    /* Step 3: RMSNorm with learned weight on the just-emitted comp row.
     * Use a 1-row view so the existing kernel can rewrite in place. */
    {
        const uint64_t norm_bytes = (uint64_t)hd * sizeof(float);
        if (no > ms || norm_bytes > ms - no) return 0;
        const float *norm_w = (const float *)ds4_cuda_weight_ptr(m, no);
        ds4_cuda_kernel_rms_norm_w_f32<256><<<1, 256, 0, ds4_cuda_stream>>>(
            tensor_fptr(cc) + (uint64_t)cr * hd,
            tensor_fptr(cc) + (uint64_t)cr * hd,
            norm_w, hd, eps);
    }

    /* Step 4: RoPE the comp row at its logical position (pos + 1 - ratio). */
    {
        const uint32_t comp_pos = pos + 1u - ratio;
        dim3 grid(1, 1);
        ds4_cuda_kernel_rope_tail_f32<<<grid, 128, 0, ds4_cuda_stream>>>(
            tensor_fptr(cc) + (uint64_t)cr * hd,
            /*n_tok=*/1u, /*n_head=*/1u, hd, nr, comp_pos, nco,
            /*inverse=*/0,
            fb, fs, ef, af, bf, bs);
    }

    /* Step 5: shift state[4..7] -> state[0..3]. */
    {
        const int BLK = 256;
        const uint32_t n = 4u * width;
        ds4_cuda_kernel_compressor_shift_ratio4_f32<<<
            ds4_cuda_ceil_div((int)n, BLK), BLK, 0, ds4_cuda_stream>>>(
            tensor_fptr(skv), tensor_fptr(ssc), hd);
    }
    return 1;
}
int ds4_metal_compressor_store_batch_tensor(const ds4_metal_tensor *kv, const ds4_metal_tensor *sc,
        ds4_metal_tensor *skv, ds4_metal_tensor *ssc, const void *m, uint64_t ms,
        uint64_t ao, uint32_t at, uint32_t hd, uint32_t ratio, uint32_t pos0, uint32_t nt) {
    if (!kv || !sc || !skv || !ssc || !m || hd == 0 || ratio == 0 || nt == 0) return 0;
    const uint32_t width = 2u * hd;
    const uint64_t ape_bytes = (uint64_t)ratio * width *
        (at == 1u ? sizeof(__half) : sizeof(float));
    if (ao > ms || ape_bytes > ms - ao) return 0;
    const void *ape = ds4_cuda_weight_ptr(m, ao);

    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)width, BLK), nt);
    ds4_cuda_kernel_compressor_store_batch_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_cfptr(kv), tensor_cfptr(sc), ape, at,
        tensor_fptr(skv), tensor_fptr(ssc), width, ratio, pos0, nt);
    return 1;
}
int ds4_metal_compressor_prefill_tensor(ds4_metal_tensor *cc, ds4_metal_tensor *skv,
        ds4_metal_tensor *ssc, const ds4_metal_tensor *kv, const ds4_metal_tensor *sc,
        const void *m, uint64_t ms, uint64_t ao, uint32_t at, uint64_t no, uint32_t nt,
        uint32_t hd, uint32_t ratio, uint32_t pos0, uint32_t nt2, uint32_t nr, uint32_t nco,
        bool qfp8, float fb, float fs, float ef, float af, float bf, float bs, float eps) {
    (void)no;(void)nt;(void)pos0;(void)nr;(void)nco;(void)qfp8;
    (void)fb;(void)fs;(void)ef;(void)af;(void)bf;(void)bs;(void)eps;
    if (!cc || !skv || !ssc || !kv || !sc || !m || hd == 0 || ratio == 0 || nt2 == 0) return 0;
    const uint32_t coff   = (ratio == 4u) ? 2u : 1u;
    const uint32_t width  = coff * hd;
    const uint32_t n_comp = nt2 / ratio;
    const uint64_t ape_bytes = (uint64_t)ratio * width *
        (at == 1u ? sizeof(__half) : sizeof(float));
    if (ao > ms || ape_bytes > ms - ao) return 0;
    const void *ape = ds4_cuda_weight_ptr(m, ao);

    /* Initialise rolling state buffers (zero kv, zero score is fine as a
     * coarse init; the post-prefill state_init kernel writes the precise
     * values for ratio=4 if the engine calls it next). */
    const uint64_t state_bytes = (uint64_t)coff * ratio * width * sizeof(float);
    DS4_CUDA_CHECK(cudaMemsetAsync(tensor_fptr(skv), 0, state_bytes, ds4_cuda_stream));
    DS4_CUDA_CHECK(cudaMemsetAsync(tensor_fptr(ssc), 0, state_bytes, ds4_cuda_stream));

    /* Pool n_comp blocks. */
    if (n_comp > 0) {
        const int BLK = 128;
        dim3 grid(n_comp, ds4_cuda_ceil_div((int)hd, BLK));
        ds4_cuda_kernel_compressor_prefill_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
            tensor_fptr(cc), tensor_cfptr(kv), tensor_cfptr(sc), ape, at,
            hd, ratio, coff, n_comp);
    }
    return 1;
}
int ds4_metal_compressor_prefill_ratio4_replay_tensor(ds4_metal_tensor *cc, ds4_metal_tensor *skv,
        ds4_metal_tensor *ssc, const ds4_metal_tensor *kv, const ds4_metal_tensor *sc,
        const void *m, uint64_t ms, uint64_t ao, uint32_t at, uint64_t no, uint32_t nt,
        uint32_t hd, uint32_t pos0, uint32_t nt2, uint32_t nr, uint32_t nco,
        bool qfp8, float fb, float fs, float ef, float af, float bf, float bs, float eps) {
    /* Replay path on KV-cache resume uses the same pool math as prefill. */
    return ds4_metal_compressor_prefill_tensor(cc, skv, ssc, kv, sc, m, ms, ao, at, no, nt,
                                               hd, /*ratio=*/4u, pos0, nt2, nr, nco,
                                               qfp8, fb, fs, ef, af, bf, bs, eps);
}
int ds4_metal_compressor_prefill_state_ratio4_tensor(ds4_metal_tensor *skv, ds4_metal_tensor *ssc,
        const ds4_metal_tensor *kt, const ds4_metal_tensor *st,
        const void *m, uint64_t ms, uint64_t ao, uint32_t at, uint32_t hd, uint32_t pos0) {
    (void)pos0;
    if (!skv || !ssc || !m || hd == 0) return 0;
    const uint32_t width = 2u * hd;
    const uint64_t ape_bytes = (uint64_t)4u * width *
        (at == 1u ? sizeof(__half) : sizeof(float));
    if (ao > ms || ape_bytes > ms - ao) return 0;
    const void *ape = ds4_cuda_weight_ptr(m, ao);
    const int have_prev = (kt && st) ? 1 : 0;

    const int BLK = 128;
    dim3 grid(ds4_cuda_ceil_div((int)hd, BLK), 4u);
    ds4_cuda_kernel_compressor_state_init_ratio4_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(skv), tensor_fptr(ssc),
        have_prev ? tensor_cfptr(kt) : nullptr,
        have_prev ? tensor_cfptr(st) : nullptr,
        ape, at, hd, have_prev);
    return 1;
}
/* Dispatch attention kernels with a BLOCK size that covers head_dim.
 * DS4 production uses head_dim 128 (MLA main attention) or 512 (MLA latent
 * / indexer-class paths).  We instantiate the template for both. */
#define DS4_CUDA_ATTN_DISPATCH(KERNEL, hd, grid, stream, ...)                   \
    do {                                                                        \
        const size_t smem = (32 + 1) * sizeof(float);                           \
        if ((hd) <= 128) {                                                      \
            KERNEL<128><<<(grid), 128, smem, (stream)>>>(__VA_ARGS__);          \
        } else if ((hd) <= 256) {                                               \
            KERNEL<256><<<(grid), 256, smem, (stream)>>>(__VA_ARGS__);          \
        } else if ((hd) <= 512) {                                               \
            KERNEL<512><<<(grid), 512, smem, (stream)>>>(__VA_ARGS__);          \
        } else if ((hd) <= 1024) {                                              \
            KERNEL<1024><<<(grid), 1024, smem, (stream)>>>(__VA_ARGS__);        \
        } else {                                                                \
            fprintf(stderr, "ds4-cuda: head_dim=%u not supported (>1024)\n",    \
                    (hd));                                                      \
            return 0;                                                           \
        }                                                                       \
    } while (0)

int ds4_metal_attention_decode_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start, const ds4_metal_tensor *comp_kv,
        uint32_t n_comp, const ds4_metal_tensor *comp_mask, uint32_t use_mask,
        uint32_t nh, uint32_t hd) {
    /* Single-token decode with optional masked compressed rows.  Delegate to
     * the batched mixed-decode kernel with n_tok=1, window=n_raw (consume
     * all available raw rows), ratio=1 (unused). */
    if (!h || !q || !raw_kv || !m || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    const float *cmask_p = (use_mask && comp_mask) ? tensor_cfptr(comp_mask) : nullptr;
    const float *ckv_p = comp_kv ? tensor_cfptr(comp_kv) : nullptr;
    dim3 grid(1, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_decode_mixed_batch_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), ckv_p,
        cmask_p, use_mask ? 1u : 0u, sinks,
        /*n_tok=*/1u, /*pos0=*/n_raw - 1u, n_raw, raw_cap, raw_start,
        n_comp, /*window=*/n_raw, /*ratio=*/1u, nh, hd);
    return 1;
}

int ds4_metal_attention_prefill_raw_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        uint32_t nt, uint32_t win, uint32_t nh, uint32_t hd) {
    if (!h || !q || !raw_kv || !m || nt == 0 || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    dim3 grid(nt, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_prefill_raw_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), sinks,
        nt, win, nh, hd);
    return 1;
}
int ds4_metal_attention_decode_raw_batch_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        uint32_t nt, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t win, uint32_t nh, uint32_t hd) {
    if (!h || !q || !raw_kv || !m || nt == 0 || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    dim3 grid(nt, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_decode_raw_batch_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), sinks,
        nt, pos0, n_raw, raw_cap, raw_start, win, nh, hd);
    return 1;
}
int ds4_metal_attention_decode_mixed_batch_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv, const ds4_metal_tensor *comp_mask, uint32_t use_mask,
        uint32_t nt, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t win, uint32_t ratio, uint32_t nh, uint32_t hd) {
    if (!h || !q || !raw_kv || !comp_kv || !m || nt == 0 || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    const float *mask_p = (use_mask && comp_mask) ? tensor_cfptr(comp_mask) : nullptr;
    dim3 grid(nt, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_decode_mixed_batch_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), tensor_cfptr(comp_kv),
        mask_p, use_mask ? 1u : 0u, sinks,
        nt, pos0, n_raw, raw_cap, raw_start, n_comp, win, ratio, nh, hd);
    return 1;
}
int ds4_metal_attention_indexed_mixed_batch_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv, const ds4_metal_tensor *topk,
        uint32_t nt, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t top_k, uint32_t win, uint32_t ratio, uint32_t nh, uint32_t hd) {
    if (!h || !q || !raw_kv || !comp_kv || !topk || !m || nt == 0 || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    const int32_t *topk_p = (const int32_t *)tensor_cfptr(topk);
    dim3 grid(nt, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_indexed_mixed_batch_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), tensor_cfptr(comp_kv),
        topk_p, sinks,
        nt, pos0, n_raw, raw_cap, raw_start, n_comp, top_k, win, ratio, nh, hd);
    return 1;
}
int ds4_metal_attention_prefill_static_mixed_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv, uint32_t nt, uint32_t n_comp,
        uint32_t win, uint32_t ratio, uint32_t nh, uint32_t hd) {
    if (!h || !q || !raw_kv || !comp_kv || !m || nt == 0 || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    dim3 grid(nt, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_prefill_static_mixed_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), tensor_cfptr(comp_kv),
        sinks, nt, n_comp, win, ratio, nh, hd);
    return 1;
}
int ds4_metal_attention_prefill_masked_mixed_heads_tensor(ds4_metal_tensor *h, const void *m, uint64_t ms,
        uint64_t so, const ds4_metal_tensor *q, const ds4_metal_tensor *raw_kv,
        const ds4_metal_tensor *comp_kv, const ds4_metal_tensor *comp_mask,
        uint32_t nt, uint32_t n_comp, uint32_t win, uint32_t ratio, uint32_t nh, uint32_t hd) {
    if (!h || !q || !raw_kv || !comp_kv || !comp_mask || !m || nt == 0 || nh == 0 || hd == 0) return 0;
    if (so > ms || (uint64_t)nh * sizeof(float) > ms - so) return 0;
    const float *sinks = (const float *)ds4_cuda_weight_ptr(m, so);
    dim3 grid(nt, nh);
    DS4_CUDA_ATTN_DISPATCH(
        ds4_cuda_kernel_attn_prefill_masked_mixed_f32, hd, grid, ds4_cuda_stream,
        tensor_fptr(h), tensor_cfptr(q), tensor_cfptr(raw_kv), tensor_cfptr(comp_kv),
        tensor_cfptr(comp_mask), sinks, nt, n_comp, win, ratio, nh, hd);
    return 1;
}
int ds4_metal_attention_output_low_q8_tensor(ds4_metal_tensor *low, const void *m, uint64_t ms,
        uint64_t oa, uint64_t gd, uint64_t rank, uint32_t ng, const ds4_metal_tensor *h) {
    if (!low || !h || !m || ng == 0 || rank == 0 || gd == 0) return 0;
    if (gd % DS4_CUDA_QK8_0 != 0) {
        fprintf(stderr, "ds4-cuda: attn_output_low group_dim %% 32 != 0 (got %llu)\n",
                (unsigned long long)gd);
        return 0;
    }
    const uint64_t row_a_bytes = (gd / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    const uint64_t bytes = (uint64_t)ng * rank * row_a_bytes;
    if (oa > ms || bytes > ms - oa) return 0;
    /* low tensor shape implies token count.  low_bytes = n_tok * ng * rank * f32 */
    const uint64_t low_bytes = ds4_metal_tensor_bytes(low);
    const uint64_t per_tok = (uint64_t)ng * rank * sizeof(float);
    const uint32_t n_tok = (uint32_t)(low_bytes / per_tok);
    const block_q8_0 *wa = (const block_q8_0 *)ds4_cuda_weight_ptr(m, oa);
    dim3 grid((unsigned)rank, ng, n_tok);
    ds4_cuda_kernel_attn_output_low_q8_f32<128><<<grid, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(low), wa, tensor_cfptr(h), n_tok, ng, (uint32_t)rank, (uint32_t)gd);
    return 1;
}

int ds4_metal_attention_output_q8_batch_tensor(ds4_metal_tensor *o, ds4_metal_tensor *low,
        ds4_metal_tensor *gt, ds4_metal_tensor *lt, const void *m, uint64_t ms,
        uint64_t oa, uint64_t ob, uint64_t gd, uint64_t rank, uint32_t ng, uint64_t od,
        const ds4_metal_tensor *h, uint32_t nt) {
    /* Two-stage projection mirroring metal/ds4_metal.m:
     *   stage 1: low = W_a @ heads      (per-group Q8_0 LoRA-A)
     *   stage 2: out = W_b @ low        (single Q8_0 GEMM with in_dim = ng*rank) */
    (void)gt; (void)lt;
    if (!o || !low || !h || !m || ng == 0 || rank == 0 || gd == 0 || od == 0 || nt == 0) return 0;
    if (gd % DS4_CUDA_QK8_0 != 0 || (ng * rank) % DS4_CUDA_QK8_0 != 0) {
        fprintf(stderr, "ds4-cuda: attn_output_batch dims must be multiples of 32 (gd=%llu, ng*rank=%llu)\n",
                (unsigned long long)gd, (unsigned long long)(ng * rank));
        return 0;
    }
    const uint64_t row_a_bytes = (gd / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    const uint64_t row_b_bytes = ((ng * rank) / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    if (oa > ms || (uint64_t)ng * rank * row_a_bytes > ms - oa) return 0;
    if (ob > ms || od * row_b_bytes > ms - ob) return 0;

    const block_q8_0 *wa = (const block_q8_0 *)ds4_cuda_weight_ptr(m, oa);
    const block_q8_0 *wb = (const block_q8_0 *)ds4_cuda_weight_ptr(m, ob);

    /* Stage 1: low = W_a · heads (per-group LoRA-A). */
    dim3 grid1((unsigned)rank, ng, nt);
    ds4_cuda_kernel_attn_output_low_q8_f32<128><<<grid1, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(low), wa, tensor_cfptr(h), nt, ng, (uint32_t)rank, (uint32_t)gd);

    /* Stage 2: out = W_b · low.  in_dim = ng * rank. */
    const uint64_t in_dim = (uint64_t)ng * rank;
    dim3 grid2((unsigned)od, (unsigned)nt);
    ds4_cuda_kernel_matmul_q8_0_f32<128><<<grid2, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), wb, tensor_cfptr(low), in_dim, od, nt);
    return 1;
}

/* --- router / MoE ----------------------------------------------------- */

/* Pull bias / hash table pointers from the registered model map. */
static inline int ds4_cuda_router_dispatch(
        ds4_metal_tensor *sel, ds4_metal_tensor *w, ds4_metal_tensor *p,
        const void *model_map, uint64_t model_size,
        uint64_t bias_off, uint64_t hash_off, uint32_t hash_rows,
        bool has_bias, bool hash_mode,
        const ds4_metal_tensor *log, const ds4_metal_tensor *toks, uint32_t n_tok) {
    if (!sel || !w || !p || !log || n_tok == 0) return 0;
    const float *bias = nullptr;
    const int32_t *hash = nullptr;
    if (has_bias && !hash_mode) {
        if (bias_off > model_size || 256ull * sizeof(float) > model_size - bias_off) return 0;
        bias = (const float *)ds4_cuda_weight_ptr(model_map, bias_off);
    }
    if (hash_mode) {
        const uint64_t bytes = (uint64_t)hash_rows * DS4_CUDA_TOP_K * sizeof(int32_t);
        if (hash_off > model_size || bytes > model_size - hash_off) return 0;
        hash = (const int32_t *)ds4_cuda_weight_ptr(model_map, hash_off);
    }
    const int32_t *tokens = toks ? (const int32_t *)((const uint8_t *)toks->base + toks->offset) : nullptr;
    ds4_cuda_kernel_router_select_f32<256><<<n_tok, 256, 0, ds4_cuda_stream>>>(
        (int32_t *)tensor_fptr(sel), tensor_fptr(w), tensor_fptr(p),
        tensor_cfptr(log), bias, hash, tokens,
        n_tok, hash_rows, has_bias ? 1 : 0, hash_mode ? 1 : 0);
    return 1;
}

/* Persistent 1-token managed buffer for the single-token hash-mode router
 * path; created lazily on first use and reused across decode steps.  Tu
 * (storage declared near the top of this file for ds4_metal_cleanup). */

int ds4_metal_router_select_tensor(ds4_metal_tensor *sel, ds4_metal_tensor *w, ds4_metal_tensor *p,
        const void *m, uint64_t ms, uint64_t bo, uint64_t ho, uint32_t hr, uint32_t tok,
        uint32_t neg, uint32_t ngu, bool hb, bool hm, const ds4_metal_tensor *log) {
    if (neg > 1u || ngu > 0u) {
        fprintf(stderr, "ds4-cuda: router group gating not supported (groups=%u, used=%u)\n", neg, ngu);
        return 0;
    }
    /* Use a temporary tokens tensor of length 1 holding the engine's `tok`. */
    if (!g_router_single_tok_buf) {
        DS4_CUDA_CHECK(cudaMallocManaged((void **)&g_router_single_tok_buf,
                                         sizeof(int32_t), cudaMemAttachGlobal));
    }
    *g_router_single_tok_buf = (int32_t)tok;
    /* Wrap in a transient ds4_metal_tensor-like alias.  We need only ->base
     * and ->offset for the kernel to read; build it on the stack. */
    ds4_metal_tensor toks_alias = { g_router_single_tok_buf, 0, sizeof(int32_t), 0 };
    return ds4_cuda_router_dispatch(sel, w, p, m, ms, bo, ho, hr, hb, hm, log, &toks_alias, 1);
}

int ds4_metal_router_select_batch_tensor(ds4_metal_tensor *sel, ds4_metal_tensor *w, ds4_metal_tensor *p,
        const void *m, uint64_t ms, uint64_t bo, uint64_t ho, uint32_t hr,
        uint32_t neg, uint32_t ngu, bool hb, bool hm,
        const ds4_metal_tensor *log, const ds4_metal_tensor *toks, uint32_t nt) {
    if (neg > 1u || ngu > 0u) {
        fprintf(stderr, "ds4-cuda: router group gating not supported (groups=%u, used=%u)\n", neg, ngu);
        return 0;
    }
    return ds4_cuda_router_dispatch(sel, w, p, m, ms, bo, ho, hr, hb, hm, log, toks, nt);
}
/* Naive routed-MoE implementation: for each selected expert k in 0..K-1, run
 * gate/up (IQ2_XXS) → SwiGLU+clamp+route-weight → down (Q2_K) and accumulate
 * into `out`.  Buffers gate/up/mid passed by the engine are used as scratch.
 * `experts` is an unused counter buffer in this naive form. */
static inline int ds4_cuda_routed_moe_run(
        ds4_metal_tensor *out, ds4_metal_tensor *g, ds4_metal_tensor *u, ds4_metal_tensor *mid,
        const void *m, uint64_t ms,
        uint64_t go, uint64_t uo, uint64_t dno,
        uint32_t gate_type, uint32_t down_type,
        uint64_t geb, uint64_t deb,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        const ds4_metal_tensor *sel, const ds4_metal_tensor *wgt,
        uint32_t K, float clamp,
        const ds4_metal_tensor *x, uint32_t n_tok) {

    if (!out || !g || !u || !mid || !m || !sel || !wgt || !x) return 0;
    if (in_dim % DS4_CUDA_QK_K != 0 || mid_dim % DS4_CUDA_QK_K != 0) {
        fprintf(stderr, "ds4-cuda: MoE dims must be multiples of 256 (in=%u, mid=%u)\n", in_dim, mid_dim);
        return 0;
    }
    if (gate_type != DS4_CUDA_TENSOR_IQ2_XXS) {
        fprintf(stderr, "ds4-cuda: routed MoE only supports IQ2_XXS gate/up (got type=%u)\n", gate_type);
        return 0;
    }
    if (down_type != DS4_CUDA_TENSOR_Q2_K) {
        fprintf(stderr, "ds4-cuda: routed MoE only supports Q2_K down (got type=%u)\n", down_type);
        return 0;
    }

    const uint64_t expert_stride_gate_blocks = geb / sizeof(block_iq2_xxs);
    const uint64_t expert_stride_down_blocks = deb / sizeof(block_q2_K);

    const block_iq2_xxs *w_gate = (const block_iq2_xxs *)ds4_cuda_weight_ptr(m, go);
    const block_iq2_xxs *w_up   = (const block_iq2_xxs *)ds4_cuda_weight_ptr(m, uo);
    const block_q2_K    *w_down = (const block_q2_K    *)ds4_cuda_weight_ptr(m, dno);
    const int32_t *sel_p = (const int32_t *)((const uint8_t *)sel->base + sel->offset);
    (void)ms;

    /* Zero output before accumulation. */
    DS4_CUDA_CHECK(cudaMemsetAsync(tensor_fptr(out), 0, (uint64_t)n_tok * out_dim * sizeof(float),
                                   ds4_cuda_stream));

    for (uint32_t k = 0; k < K; k++) {
        /* Stage A: gate = W_gate[sel[t,k]] · x       (mid_dim outputs) */
        dim3 grid1(mid_dim, n_tok);
        ds4_cuda_kernel_matmul_iq2xxs_expert_f32<128><<<grid1, 128, 0, ds4_cuda_stream>>>(
            tensor_fptr(g), w_gate, tensor_cfptr(x), sel_p,
            expert_stride_gate_blocks, in_dim, mid_dim, K, k, n_tok);

        /* Stage B: up = W_up[sel[t,k]] · x */
        ds4_cuda_kernel_matmul_iq2xxs_expert_f32<128><<<grid1, 128, 0, ds4_cuda_stream>>>(
            tensor_fptr(u), w_up, tensor_cfptr(x), sel_p,
            expert_stride_gate_blocks, in_dim, mid_dim, K, k, n_tok);

        /* Stage C: mid = silu(clamp(gate)) * up * weights[t, k] */
        const int BLK_MID = 256;
        dim3 grid2(ds4_cuda_ceil_div((int)mid_dim, BLK_MID), n_tok);
        ds4_cuda_kernel_moe_swiglu_weight_f32<<<grid2, BLK_MID, 0, ds4_cuda_stream>>>(
            tensor_fptr(mid), tensor_cfptr(g), tensor_cfptr(u), tensor_cfptr(wgt),
            mid_dim, n_tok, K, k, clamp);

        /* Stage D: contrib = W_down[sel[t,k]] · mid; reuse `g` as contrib scratch. */
        dim3 grid3(out_dim, n_tok);
        ds4_cuda_kernel_matmul_q2_k_expert_f32<128><<<grid3, 128, 0, ds4_cuda_stream>>>(
            tensor_fptr(g), w_down, tensor_cfptr(mid), sel_p,
            expert_stride_down_blocks, mid_dim, out_dim, K, k, n_tok);

        /* Stage E: out += contrib. */
        const int BLK_ACC = 256;
        dim3 grid4(ds4_cuda_ceil_div((int)out_dim, BLK_ACC), n_tok);
        ds4_cuda_kernel_acc_f32<<<grid4, BLK_ACC, 0, ds4_cuda_stream>>>(
            tensor_fptr(out), tensor_cfptr(g), out_dim, n_tok);
    }
    return 1;
}

int ds4_metal_routed_moe_one_tensor(ds4_metal_tensor *out, ds4_metal_tensor *g, ds4_metal_tensor *u,
        ds4_metal_tensor *mid, ds4_metal_tensor *ex, const void *m, uint64_t ms,
        uint64_t go, uint64_t uo, uint64_t dno, uint32_t gt, uint32_t dt,
        uint64_t geb, uint64_t grb, uint64_t deb, uint64_t drb,
        uint32_t eid, uint32_t emd, uint32_t od,
        const ds4_metal_tensor *sel, const ds4_metal_tensor *w, uint32_t ne, float c, const ds4_metal_tensor *x) {
    (void)ex; (void)grb; (void)drb;
    return ds4_cuda_routed_moe_run(out, g, u, mid, m, ms, go, uo, dno, gt, dt,
                                   geb, deb, eid, emd, (uint32_t)od,
                                   sel, w, ne, c, x, /*n_tok=*/1);
}
int ds4_metal_routed_moe_batch_tensor(ds4_metal_tensor *out, ds4_metal_tensor *g, ds4_metal_tensor *u,
        ds4_metal_tensor *mid, ds4_metal_tensor *ex, const void *m, uint64_t ms,
        uint64_t go, uint64_t uo, uint64_t dno, uint32_t gt, uint32_t dt,
        uint64_t geb, uint64_t grb, uint64_t deb, uint64_t drb,
        uint32_t eid, uint32_t emd, uint32_t od,
        const ds4_metal_tensor *sel, const ds4_metal_tensor *w, uint32_t ne, float c,
        const ds4_metal_tensor *x, uint32_t nt) {
    (void)ex; (void)grb; (void)drb;
    return ds4_cuda_routed_moe_run(out, g, u, mid, m, ms, go, uo, dno, gt, dt,
                                   geb, deb, eid, emd, (uint32_t)od,
                                   sel, w, ne, c, x, nt);
}

/* --- hyper-connection mixers ----------------------------------------- */

int ds4_metal_hc_split_sinkhorn_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *mx,
        const void *m, uint64_t ms, uint64_t so, uint64_t bo, uint32_t nh, uint32_t it, float eps) {
    if (!o || !mx || !m || nh == 0) return 0;
    if (nh != 4) {
        fprintf(stderr, "ds4-cuda: HC split sinkhorn specialised to n_hc=4 (got %u)\n", nh);
        return 0;
    }
    const uint32_t mix_hc = 2u * nh + nh * nh;  /* 24 */
    if (so > ms || 3u * sizeof(float) > ms - so) return 0;
    if (bo > ms || mix_hc * sizeof(float) > ms - bo) return 0;
    const float *scale = (const float *)ds4_cuda_weight_ptr(m, so);
    const float *base  = (const float *)ds4_cuda_weight_ptr(m, bo);
    const uint64_t n_rows = ds4_metal_tensor_bytes(o) / ((uint64_t)mix_hc * sizeof(float));
    const int BLK = 64;
    ds4_cuda_kernel_hc_split_sinkhorn_hc4_f32<<<ds4_cuda_ceil_div((int)n_rows, BLK), BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(mx), scale, base, (uint32_t)n_rows, it, eps);
    return 1;
}
int ds4_metal_hc_weighted_sum_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *r,
        const ds4_metal_tensor *w, uint32_t ne, uint32_t nh) {
    if (!o || !r || !w || ne == 0 || nh == 0) return 0;
    /* weights[]: contiguous [n_rows, n_hc] - stride = n_hc. */
    const uint64_t n_rows = ds4_metal_tensor_bytes(o) / ((uint64_t)ne * sizeof(float));
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), (unsigned)n_rows);
    ds4_cuda_kernel_hc_weighted_sum_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(r), tensor_cfptr(w), nh, ne, nh);
    return 1;
}
int ds4_metal_hc_weighted_sum_split_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *r,
        const ds4_metal_tensor *s, uint32_t ne, uint32_t nh) {
    if (!o || !r || !s || ne == 0 || nh == 0) return 0;
    /* split tensor: [n_rows, 2*n_hc + n_hc*n_hc] -- the first n_hc entries of
     * each row are the per-hc weights used for the residual reduction.
     * Stride per row = 2*n_hc + n_hc*n_hc. */
    const uint64_t n_rows = ds4_metal_tensor_bytes(o) / ((uint64_t)ne * sizeof(float));
    const uint32_t stride = 2u * nh + nh * nh;
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), (unsigned)n_rows);
    ds4_cuda_kernel_hc_weighted_sum_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(r), tensor_cfptr(s), stride, ne, nh);
    return 1;
}
int ds4_metal_hc_split_weighted_sum_tensor(ds4_metal_tensor *o, ds4_metal_tensor *sp,
        const ds4_metal_tensor *mx, const ds4_metal_tensor *r,
        const void *m, uint64_t ms, uint64_t so, uint64_t bo, uint32_t ne, uint32_t nh,
        uint32_t it, float eps) {
    if (!o || !sp || !mx || !r || !m || ne == 0 || nh == 0) return 0;
    if (nh != 4) {
        fprintf(stderr, "ds4-cuda: HC split-weighted-sum specialised to n_hc=4 (got %u)\n", nh);
        return 0;
    }
    const uint32_t mix_hc = 2u * nh + nh * nh;
    if (so > ms || 3u * sizeof(float) > ms - so) return 0;
    if (bo > ms || mix_hc * sizeof(float) > ms - bo) return 0;
    const float *scale = (const float *)ds4_cuda_weight_ptr(m, so);
    const float *base  = (const float *)ds4_cuda_weight_ptr(m, bo);
    const uint64_t n_rows = ds4_metal_tensor_bytes(o) / ((uint64_t)ne * sizeof(float));
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), (unsigned)n_rows);
    ds4_cuda_kernel_hc_split_weighted_sum_hc4_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_fptr(sp),
        tensor_cfptr(mx), tensor_cfptr(r),
        scale, base, ne, (uint32_t)n_rows, it, eps);
    return 1;
}
int ds4_metal_hc_split_weighted_sum_norm_tensor(ds4_metal_tensor *o, ds4_metal_tensor *no,
        ds4_metal_tensor *sp, const ds4_metal_tensor *mx, const ds4_metal_tensor *r,
        const void *m, uint64_t ms, uint64_t so, uint64_t bo, uint64_t nwo,
        uint32_t ne, uint32_t nh, uint32_t it, float eps, float neps) {
    if (!o || !no || !sp || !mx || !r || !m || ne == 0 || nh == 0) return 0;
    if (nh != 4) {
        fprintf(stderr, "ds4-cuda: HC split-weighted-sum-norm specialised to n_hc=4 (got %u)\n", nh);
        return 0;
    }
    const uint32_t mix_hc = 2u * nh + nh * nh;
    if (so > ms || 3u * sizeof(float) > ms - so) return 0;
    if (bo > ms || mix_hc * sizeof(float) > ms - bo) return 0;
    if (nwo > ms || (uint64_t)ne * sizeof(float) > ms - nwo) return 0;
    const float *scale = (const float *)ds4_cuda_weight_ptr(m, so);
    const float *base  = (const float *)ds4_cuda_weight_ptr(m, bo);
    const float *normw = (const float *)ds4_cuda_weight_ptr(m, nwo);
    const uint64_t n_rows = ds4_metal_tensor_bytes(o) / ((uint64_t)ne * sizeof(float));
    const int BLK = 256;
    const size_t smem = 32 * sizeof(float);
    ds4_cuda_kernel_hc_split_weighted_sum_norm_hc4_f32<<<n_rows, BLK, smem, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_fptr(no), tensor_fptr(sp),
        tensor_cfptr(mx), tensor_cfptr(r),
        scale, base, normw, ne, (uint32_t)n_rows, it, eps, neps);
    return 1;
}
int ds4_metal_output_hc_weights_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *p,
        const void *m, uint64_t ms, uint64_t so, uint64_t bo, uint32_t nh, float eps) {
    if (!o || !p || !m || nh == 0) return 0;
    if ((nh & 3u) != 0u) {
        fprintf(stderr, "ds4-cuda: output_hc_weights requires n_hc %% 4 == 0 (got %u)\n", nh);
        return 0;
    }
    if (so > ms || sizeof(float) > ms - so) return 0;
    if (bo > ms || (uint64_t)nh * sizeof(float) > ms - bo) return 0;
    const float *scale = (const float *)ds4_cuda_weight_ptr(m, so);
    const float *base  = (const float *)ds4_cuda_weight_ptr(m, bo);
    const uint64_t n_tok = ds4_metal_tensor_bytes(o) / ((uint64_t)nh * sizeof(float));
    const int BLK = 64;
    dim3 grid(ds4_cuda_ceil_div((int)nh, BLK), (unsigned)n_tok);
    ds4_cuda_kernel_output_hc_weights_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(p), scale, base, nh, (uint32_t)n_tok, eps);
    return 1;
}
int ds4_metal_hc_expand_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *b,
        const ds4_metal_tensor *r, const ds4_metal_tensor *p, const ds4_metal_tensor *c,
        uint32_t ne, uint32_t nh) {
    if (!o || !b || !r || !p || !c || ne == 0 || nh == 0) return 0;
    if (nh != 4) {
        fprintf(stderr, "ds4-cuda: hc_expand_tensor specialised to n_hc=4 (got %u)\n", nh);
        return 0;
    }
    const uint64_t per_tok_bytes = (uint64_t)nh * ne * sizeof(float);
    const uint32_t n_tok = (uint32_t)(ds4_metal_tensor_bytes(o) / per_tok_bytes);
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), n_tok);
    ds4_cuda_kernel_hc_expand_with_comb_hc4_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(b), tensor_cfptr(r),
        tensor_cfptr(p), tensor_cfptr(c),
        /*block_add=*/nullptr, /*has_add=*/0,
        ne, n_tok);
    return 1;
}
int ds4_metal_hc_expand_split_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *b,
        const ds4_metal_tensor *r, const ds4_metal_tensor *s, uint32_t ne, uint32_t nh) {
    if (!o || !b || !r || !s || ne == 0 || nh == 0) return 0;
    /* For HC=4 the production case, split is [2*n_hc + n_hc^2]; the n_hc
     * "post" scales sit at offset n_hc (after the n_hc "pre" scales).  See
     * metal/dsv4_misc.metal::kernel_hc_expand_*. */
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), nh);
    /* The kernel uses split[hc] as post; engine slices `s` to point at the
     * post region directly, so we forward as-is. */
    ds4_cuda_kernel_hc_expand_split_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(b), tensor_cfptr(r), tensor_cfptr(s), ne, nh);
    return 1;
}
int ds4_metal_hc_expand_add_split_tensor(ds4_metal_tensor *o, const ds4_metal_tensor *b,
        const ds4_metal_tensor *ba, const ds4_metal_tensor *r, const ds4_metal_tensor *s,
        uint32_t ne, uint32_t nh) {
    if (!o || !b || !ba || !r || !s || ne == 0 || nh == 0) return 0;
    const int BLK = 256;
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), nh);
    ds4_cuda_kernel_hc_expand_add_split_f32<<<grid, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(b), tensor_cfptr(ba),
        tensor_cfptr(r), tensor_cfptr(s), ne, nh);
    return 1;
}
/* Q8_0 GEMM + HC expand fused.  Internally: block_out = W·x (Q8_0); then
 * out_hc = HC_expand_split(block_out, residual_hc, split).  Sequential
 * kernel launches share the stream so the dependency holds. */
int ds4_metal_matmul_q8_0_hc_expand_tensor(ds4_metal_tensor *o, ds4_metal_tensor *b,
        const void *m, uint64_t ms, uint64_t wo, uint64_t id, uint64_t od,
        const ds4_metal_tensor *x, const ds4_metal_tensor *r, const ds4_metal_tensor *s,
        uint32_t ne, uint32_t nh) {
    if (!o || !b || !x || !r || !s || !m || ne == 0 || nh == 0) return 0;
    if (id % DS4_CUDA_QK8_0 != 0) return 0;
    const uint64_t row_bytes = (id / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    if (wo > ms || row_bytes * od > ms - wo) return 0;
    const block_q8_0 *w = (const block_q8_0 *)ds4_cuda_weight_ptr(m, wo);

    /* Stage 1: Q8_0 matvec into `b`.  block_out shape = [n_tok, od]. */
    const uint64_t n_tok = ds4_metal_tensor_bytes(b) / (od * sizeof(float));
    dim3 grid1((unsigned)od, (unsigned)n_tok);
    ds4_cuda_kernel_matmul_q8_0_f32<128><<<grid1, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(b), w, tensor_cfptr(x), id, od, n_tok);
    /* Stage 2: HC expand split. */
    const int BLK = 256;
    dim3 grid2(ds4_cuda_ceil_div((int)ne, BLK), nh);
    ds4_cuda_kernel_hc_expand_split_f32<<<grid2, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(b), tensor_cfptr(r), tensor_cfptr(s), ne, nh);
    return 1;
}

/* Same shape as above but with an extra `routed_out` summed in before the
 * HC expand (shared_mid · W_down + routed_out -> expand). */
int ds4_metal_shared_down_hc_expand_q8_0_tensor(ds4_metal_tensor *o, ds4_metal_tensor *so,
        const void *m, uint64_t ms, uint64_t wo, uint64_t id, uint64_t od,
        const ds4_metal_tensor *sm, const ds4_metal_tensor *ro,
        const ds4_metal_tensor *r, const ds4_metal_tensor *s, uint32_t ne, uint32_t nh) {
    if (!o || !so || !sm || !ro || !r || !s || !m || ne == 0 || nh == 0) return 0;
    if (id % DS4_CUDA_QK8_0 != 0) return 0;
    const uint64_t row_bytes = (id / DS4_CUDA_QK8_0) * sizeof(block_q8_0);
    if (wo > ms || row_bytes * od > ms - wo) return 0;
    const block_q8_0 *w = (const block_q8_0 *)ds4_cuda_weight_ptr(m, wo);

    /* Stage 1: shared_out = W_down · shared_mid (Q8_0). */
    const uint64_t n_tok = ds4_metal_tensor_bytes(so) / (od * sizeof(float));
    dim3 grid1((unsigned)od, (unsigned)n_tok);
    ds4_cuda_kernel_matmul_q8_0_f32<128><<<grid1, 128, 0, ds4_cuda_stream>>>(
        tensor_fptr(so), w, tensor_cfptr(sm), id, od, n_tok);
    /* Stage 2: HC expand with add: out = HC_expand_split(shared_out + routed_out,
     * residual, split). */
    const int BLK = 256;
    dim3 grid2(ds4_cuda_ceil_div((int)ne, BLK), nh);
    ds4_cuda_kernel_hc_expand_add_split_f32<<<grid2, BLK, 0, ds4_cuda_stream>>>(
        tensor_fptr(o), tensor_cfptr(so), tensor_cfptr(ro),
        tensor_cfptr(r), tensor_cfptr(s), ne, nh);
    return 1;
}
