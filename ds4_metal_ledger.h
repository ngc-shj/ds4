/*
 * Decode ledger: attribute every command buffer, encoder and dispatch to a
 * token, and record the CPU-side instants the GPU tables cannot show.
 *
 * Enabled by DS4_METAL_LEDGER=<path>; entirely inert otherwise.
 *
 * Two things this is not. It is not a profiler: no interval is timed here that
 * the trace already times, and no encoder is split or reordered to make a
 * measurement sharper, because splitting encoders is how the previous
 * instrument changed the thing it was measuring. It is not a sampler either:
 * every command buffer and every dispatch is recorded, so expected counts can
 * be checked against observed ones instead of scaled up from a sample.
 *
 * What it adds:
 *   - a unique id and a token/layer/role/queue label on every command buffer,
 *     and a unique id plus kernel composition on every encoder, so the GPU
 *     rows in a Metal System Trace can be attributed without guessing;
 *   - os_signpost intervals on Points of Interest for the token window and for
 *     CPU waits, because a wait produces no GPU interval at all and cannot be
 *     recovered from the GPU tables;
 *   - GPUStartTime/GPUEndTime from every completed handler, to cross-check the
 *     trace against the Metal API on the same run.
 *
 * The attribution context is thread-local. A service thread must be told which
 * token and layer its job belongs to (ds4_gpu_ledger_push), never read a
 * global "current token", which would race with the decode thread.
 */
#ifndef DS4_METAL_LEDGER_H
#define DS4_METAL_LEDGER_H

#include <os/signpost.h>
#include <os/lock.h>
#include <mach/mach_time.h>
#include <stdatomic.h>

/* ---- public API, called from ds4_metal.m and ds4.c ---- */
void ds4_gpu_ledger_probe(id<MTLDevice> device);
void ds4_gpu_ledger_note_queue(id<MTLCommandQueue> q, const char *role);
int  ds4_gpu_ledger_active(void);
void ds4_gpu_ledger_token_begin(unsigned long long pos);
void ds4_gpu_ledger_token_end(void);
void ds4_gpu_ledger_layer(int il);
void ds4_gpu_ledger_role(const char *role);
void ds4_gpu_ledger_push(unsigned long long token, int layer, const char *role);
void ds4_gpu_ledger_pop(void);
void ds4_gpu_ledger_mark(const char *name, const char *detail);
void ds4_gpu_ledger_wait_begin(const char *what);
void ds4_gpu_ledger_wait_end(const char *what);
void ds4_gpu_ledger_dump(void);

/* ---- record stream ---- */
enum {
    DS4_LR_CB_CREATE = 1, DS4_LR_CB_COMMIT, DS4_LR_CB_DONE,
    DS4_LR_ENC, DS4_LR_BLIT, DS4_LR_WAIT_EVENT, DS4_LR_SIGNAL_EVENT,
    DS4_LR_TOKEN_BEGIN, DS4_LR_TOKEN_END, DS4_LR_CPU_WAIT,
};

typedef struct {
    uint8_t  kind;
    uint8_t  enc_kind;       /* 0 = compute, 1 = blit */
    uint16_t name_id;        /* interned: role, kernel composition, wait name */
    int32_t  layer;
    uint64_t token;
    uint64_t id;             /* command buffer or encoder sequence number */
    uint64_t parent;         /* encoder -> its command buffer */
    uint64_t t_ns;           /* mach_absolute_time, converted to ns */
    uint64_t a, b;           /* kind-specific, see the dump */
} ds4_ledger_rec;

#define DS4_LEDGER_MAX_RECS  (4u * 1024u * 1024u)
#define DS4_LEDGER_MAX_NAMES 4096u
#define DS4_LEDGER_ENC_KERNELS 32u

typedef struct {
    const void *enc;
    uint64_t seq;
    uint64_t cb_seq;
    uint64_t token;
    int32_t  layer;
    uint32_t idx_in_cb;
    uint8_t  kind;
    uint16_t n_kernel;
    uint16_t kernel_name[DS4_LEDGER_ENC_KERNELS];
    uint32_t kernel_count[DS4_LEDGER_ENC_KERNELS];
    uint32_t n_dispatch;
    uint64_t bytes;
    uint64_t t_create;
} ds4_ledger_enc;

typedef struct {
    const void *cb;
    uint64_t seq;
    uint64_t token;
    int32_t  layer;
    uint16_t role;
    uint16_t queue;
    uint32_t n_enc;
    uint8_t  done;
    uint64_t t_create;
} ds4_ledger_cb;

static int g_ledger_on;
static FILE *g_ledger_out;
static const char *g_ledger_path;
static os_log_t g_ledger_poi;
static mach_timebase_info_data_t g_ledger_tb;
static _Atomic uint64_t g_ledger_cb_seq;
static _Atomic uint64_t g_ledger_enc_seq;
static _Atomic uint32_t g_ledger_n_rec;
static ds4_ledger_rec *g_ledger_recs;
static _Atomic uint64_t g_ledger_dropped;
static uint64_t g_ledger_token_limit;   /* dump after this many tokens, 0 = never */
static _Atomic uint64_t g_ledger_tokens_seen;

/* Attribution is per thread: the decode thread owns one token at a time, and
 * anything else (service thread, pacer, keepalive) carries its own. */
typedef struct { uint64_t token; uint64_t pos; int32_t layer; const char *role; } ds4_ledger_ctx;
static _Thread_local ds4_ledger_ctx g_ledger_ctx = { 0, 0, -1, "pre-decode" };
static _Atomic uint64_t g_ledger_token_seq;
static _Thread_local ds4_ledger_ctx g_ledger_saved;
static _Thread_local os_signpost_id_t g_ledger_tok_sid;

static uint64_t ds4_ledger_now(void) {
    return mach_absolute_time() * g_ledger_tb.numer / g_ledger_tb.denom;
}

/* ---- interning ------------------------------------------------------- */
static os_unfair_lock g_ledger_name_lock = OS_UNFAIR_LOCK_INIT;
static char *g_ledger_names[DS4_LEDGER_MAX_NAMES];
static uint32_t g_ledger_n_names;

static uint16_t ds4_ledger_intern(const char *s) {
    if (!s) s = "?";
    os_unfair_lock_lock(&g_ledger_name_lock);
    for (uint32_t i = 0; i < g_ledger_n_names; i++) {
        if (strcmp(g_ledger_names[i], s) == 0) {
            os_unfair_lock_unlock(&g_ledger_name_lock);
            return (uint16_t)i;
        }
    }
    uint16_t id = 0;
    if (g_ledger_n_names < DS4_LEDGER_MAX_NAMES) {
        g_ledger_names[g_ledger_n_names] = strdup(s);
        id = (uint16_t)g_ledger_n_names++;
    }
    os_unfair_lock_unlock(&g_ledger_name_lock);
    return id;
}

static ds4_ledger_rec *ds4_ledger_claim(void) {
    const uint32_t i = atomic_fetch_add_explicit(&g_ledger_n_rec, 1u, memory_order_relaxed);
    if (i >= DS4_LEDGER_MAX_RECS) {
        atomic_fetch_add_explicit(&g_ledger_dropped, 1u, memory_order_relaxed);
        return NULL;
    }
    return &g_ledger_recs[i];
}

/* ---- pipeline names: written rarely, read once per dispatch ---------- */
#define DS4_LEDGER_PSO_SLOTS 8192u
typedef struct { _Atomic(const void *) pso; uint16_t name; } ds4_ledger_pso_slot;
static ds4_ledger_pso_slot g_ledger_psos[DS4_LEDGER_PSO_SLOTS];
static os_unfair_lock g_ledger_pso_lock = OS_UNFAIR_LOCK_INIT;

static uint32_t ds4_ledger_ptr_hash(const void *p) {
    uint64_t x = (uint64_t)(uintptr_t)p;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    return (uint32_t)x;
}

static void ds4_ledger_note_pso(const void *pso, const char *name) {
    if (!g_ledger_on || !pso) return;
    const uint16_t id = ds4_ledger_intern(name);
    os_unfair_lock_lock(&g_ledger_pso_lock);
    uint32_t i = ds4_ledger_ptr_hash(pso) & (DS4_LEDGER_PSO_SLOTS - 1u);
    for (uint32_t n = 0; n < DS4_LEDGER_PSO_SLOTS; n++, i = (i + 1u) & (DS4_LEDGER_PSO_SLOTS - 1u)) {
        const void *cur = atomic_load_explicit(&g_ledger_psos[i].pso, memory_order_relaxed);
        if (cur == pso) break;
        if (cur == NULL) {
            g_ledger_psos[i].name = id;
            atomic_store_explicit(&g_ledger_psos[i].pso, pso, memory_order_release);
            break;
        }
    }
    os_unfair_lock_unlock(&g_ledger_pso_lock);
}

static uint16_t ds4_ledger_pso_name(const void *pso) {
    uint32_t i = ds4_ledger_ptr_hash(pso) & (DS4_LEDGER_PSO_SLOTS - 1u);
    for (uint32_t n = 0; n < 64u; n++, i = (i + 1u) & (DS4_LEDGER_PSO_SLOTS - 1u)) {
        const void *cur = atomic_load_explicit(&g_ledger_psos[i].pso, memory_order_acquire);
        if (cur == pso) return g_ledger_psos[i].name;
        if (cur == NULL) break;
    }
    return ds4_ledger_intern("unnamed-pipeline");
}

/* ---- side tables for live command buffers and encoders --------------- */
#define DS4_LEDGER_CB_SLOTS  4096u
#define DS4_LEDGER_ENC_SLOTS 4096u
static ds4_ledger_cb  g_ledger_cbs[DS4_LEDGER_CB_SLOTS];
static ds4_ledger_enc g_ledger_encs[DS4_LEDGER_ENC_SLOTS];
static os_unfair_lock g_ledger_tab_lock = OS_UNFAIR_LOCK_INIT;
/* Consecutive dispatches land on the same encoder, so remember the last one
 * and keep the table lock off the per-dispatch path. */
static _Thread_local ds4_ledger_enc *g_ledger_enc_cache;

#define DS4_LEDGER_DEAD ((const void *)-1)

static ds4_ledger_cb *ds4_ledger_cb_find(const void *cb, int create) {
    uint32_t i = ds4_ledger_ptr_hash(cb) & (DS4_LEDGER_CB_SLOTS - 1u);
    ds4_ledger_cb *reuse = NULL;
    for (uint32_t n = 0; n < DS4_LEDGER_CB_SLOTS; n++, i = (i + 1u) & (DS4_LEDGER_CB_SLOTS - 1u)) {
        if (g_ledger_cbs[i].cb == cb) return &g_ledger_cbs[i];
        if (g_ledger_cbs[i].done && !reuse) reuse = &g_ledger_cbs[i];
        if (g_ledger_cbs[i].cb == NULL) {
            if (!create) return NULL;
            ds4_ledger_cb *slot = reuse ? reuse : &g_ledger_cbs[i];
            memset(slot, 0, sizeof(*slot));
            slot->cb = cb;
            return slot;
        }
    }
    if (create && reuse) { memset(reuse, 0, sizeof(*reuse)); reuse->cb = cb; return reuse; }
    return NULL;
}

static ds4_ledger_enc *ds4_ledger_enc_find(const void *enc, int create) {
    uint32_t i = ds4_ledger_ptr_hash(enc) & (DS4_LEDGER_ENC_SLOTS - 1u);
    ds4_ledger_enc *reuse = NULL;
    for (uint32_t n = 0; n < DS4_LEDGER_ENC_SLOTS; n++, i = (i + 1u) & (DS4_LEDGER_ENC_SLOTS - 1u)) {
        if (g_ledger_encs[i].enc == enc) return &g_ledger_encs[i];
        if (g_ledger_encs[i].enc == DS4_LEDGER_DEAD && !reuse) reuse = &g_ledger_encs[i];
        if (g_ledger_encs[i].enc == NULL) {
            if (!create) return NULL;
            ds4_ledger_enc *slot = reuse ? reuse : &g_ledger_encs[i];
            memset(slot, 0, sizeof(*slot));
            slot->enc = enc;
            return slot;
        }
    }
    if (create && reuse) { memset(reuse, 0, sizeof(*reuse)); reuse->enc = enc; return reuse; }
    return NULL;
}


/* ---- swizzling ------------------------------------------------------- */

@interface NSObject (DS4LedgerSwizzle)
- (id)ds4_lg_newComputePipelineStateWithFunction:(id<MTLFunction>)fn
                                           error:(NSError * __autoreleasing *)error NS_RETURNS_RETAINED;
- (id)ds4_lg_newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)d
                                            options:(MTLPipelineOption)o
                                         reflection:(MTLAutoreleasedComputePipelineReflection *)r
                                              error:(NSError * __autoreleasing *)error NS_RETURNS_RETAINED;
- (id)ds4_lg_commandBuffer;
- (id)ds4_lg_commandBufferWithUnretainedReferences;
- (id)ds4_lg_computeCommandEncoder;
- (id)ds4_lg_computeCommandEncoderWithDispatchType:(MTLDispatchType)t;
- (id)ds4_lg_computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)d;
- (id)ds4_lg_blitCommandEncoder;
- (id)ds4_lg_blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor *)d;
- (void)ds4_lg_commit;
- (void)ds4_lg_waitUntilCompleted;
- (void)ds4_lg_encodeWaitForEvent:(id<MTLEvent>)e value:(uint64_t)v;
- (void)ds4_lg_encodeSignalEvent:(id<MTLEvent>)e value:(uint64_t)v;
- (void)ds4_lg_setComputePipelineState:(id<MTLComputePipelineState>)pso;
- (void)ds4_lg_dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt;
- (void)ds4_lg_dispatchThreads:(MTLSize)th threadsPerThreadgroup:(MTLSize)tpt;
- (void)ds4_lg_copyFromBuffer:(id<MTLBuffer>)src sourceOffset:(NSUInteger)so
                     toBuffer:(id<MTLBuffer>)dst destinationOffset:(NSUInteger)dof
                         size:(NSUInteger)size;
- (void)ds4_lg_endEncoding;
@end

/* Swizzle where the method is actually implemented, not where it is first
 * seen. Metal's compute and blit encoder classes share ancestors: installing a
 * replacement on a subclass while the original lives on a shared superclass
 * makes the *other* subclass resolve our replacement selector to the first
 * subclass's original, so a blit encoder ends up in the compute encoder's
 * endEncoding. That crashes, and it crashes inside Metal, a long way from the
 * cause. */
static Class ds4_ledger_impl_class(Class cls, SEL sel) {
    for (Class c = cls; c; c = class_getSuperclass(c)) {
        unsigned n = 0;
        Method *ms = class_copyMethodList(c, &n);
        int here = 0;
        for (unsigned i = 0; i < n; i++) if (method_getName(ms[i]) == sel) { here = 1; break; }
        free(ms);
        if (here) return c;
    }
    return cls;
}

static struct { Class cls; SEL sel; } g_ledger_swizzled[64];
static uint32_t g_ledger_n_swizzled;

static void ds4_ledger_swizzle(Class cls, SEL orig, SEL repl) {
    if (!cls) return;
    Class target = ds4_ledger_impl_class(cls, orig);
    for (uint32_t i = 0; i < g_ledger_n_swizzled; i++)
        if (g_ledger_swizzled[i].cls == target && g_ledger_swizzled[i].sel == orig) return;
    Method a = class_getInstanceMethod(target, orig);
    Method src = class_getInstanceMethod([NSObject class], repl);
    if (!a || !src) return;
    /* Give the target class its own copy of the replacement before exchanging.
     * Exchanging against the category's method on NSObject rewrites it for
     * every class in the process, so the next class to hook the same selector
     * inherits this class's original implementation: a blit encoder then ends
     * up running the compute encoder's endEncoding, and Metal faults far from
     * the cause. */
    class_addMethod(target, repl, method_getImplementation(src), method_getTypeEncoding(a));
    Method b = class_getInstanceMethod(target, repl);
    if (!b || b == src) return;
    if (g_ledger_n_swizzled < 64) {
        g_ledger_swizzled[g_ledger_n_swizzled].cls = target;
        g_ledger_swizzled[g_ledger_n_swizzled++].sel = orig;
    }
    method_exchangeImplementations(a, b);
}

static Class g_ledger_hooked_cls[32];
static uint32_t g_ledger_n_hooked_cls;

static int ds4_ledger_class_seen(Class c) {
    for (uint32_t i = 0; i < g_ledger_n_hooked_cls; i++) if (g_ledger_hooked_cls[i] == c) return 1;
    if (g_ledger_n_hooked_cls < 32) g_ledger_hooked_cls[g_ledger_n_hooked_cls++] = c;
    return 0;
}

/* The chain starts here: until the queue's creator is swizzled, nothing else
 * is ever reached. */
static void ds4_ledger_hook_queue_class(id q) {
    Class c = object_getClass(q);
    if (ds4_ledger_class_seen(c)) return;
    ds4_ledger_swizzle(c, @selector(commandBuffer), @selector(ds4_lg_commandBuffer));
    ds4_ledger_swizzle(c, @selector(commandBufferWithUnretainedReferences),
                          @selector(ds4_lg_commandBufferWithUnretainedReferences));
}

static void ds4_ledger_hook_cb_class(id cb) {
    Class c = object_getClass(cb);
    if (ds4_ledger_class_seen(c)) return;
    ds4_ledger_swizzle(c, @selector(computeCommandEncoder), @selector(ds4_lg_computeCommandEncoder));
    ds4_ledger_swizzle(c, @selector(computeCommandEncoderWithDispatchType:), @selector(ds4_lg_computeCommandEncoderWithDispatchType:));
    ds4_ledger_swizzle(c, @selector(computeCommandEncoderWithDescriptor:), @selector(ds4_lg_computeCommandEncoderWithDescriptor:));
    ds4_ledger_swizzle(c, @selector(blitCommandEncoder), @selector(ds4_lg_blitCommandEncoder));
    ds4_ledger_swizzle(c, @selector(blitCommandEncoderWithDescriptor:), @selector(ds4_lg_blitCommandEncoderWithDescriptor:));
    ds4_ledger_swizzle(c, @selector(commit), @selector(ds4_lg_commit));
    ds4_ledger_swizzle(c, @selector(waitUntilCompleted), @selector(ds4_lg_waitUntilCompleted));
    ds4_ledger_swizzle(c, @selector(encodeWaitForEvent:value:), @selector(ds4_lg_encodeWaitForEvent:value:));
    ds4_ledger_swizzle(c, @selector(encodeSignalEvent:value:), @selector(ds4_lg_encodeSignalEvent:value:));
}

static void ds4_ledger_hook_compute_encoder_class(id enc) {
    Class c = object_getClass(enc);
    if (ds4_ledger_class_seen(c)) return;
    ds4_ledger_swizzle(c, @selector(setComputePipelineState:), @selector(ds4_lg_setComputePipelineState:));
    ds4_ledger_swizzle(c, @selector(dispatchThreadgroups:threadsPerThreadgroup:), @selector(ds4_lg_dispatchThreadgroups:threadsPerThreadgroup:));
    ds4_ledger_swizzle(c, @selector(dispatchThreads:threadsPerThreadgroup:), @selector(ds4_lg_dispatchThreads:threadsPerThreadgroup:));
    ds4_ledger_swizzle(c, @selector(endEncoding), @selector(ds4_lg_endEncoding));
}

static void ds4_ledger_hook_blit_encoder_class(id enc) {
    Class c = object_getClass(enc);
    if (ds4_ledger_class_seen(c)) return;
    ds4_ledger_swizzle(c, @selector(copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:),
                          @selector(ds4_lg_copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:));
    ds4_ledger_swizzle(c, @selector(endEncoding), @selector(ds4_lg_endEncoding));
}

/* ---- queue roles ----------------------------------------------------- */
static os_unfair_lock g_ledger_q_lock = OS_UNFAIR_LOCK_INIT;
static const void *g_ledger_q_ptr[16];
static uint16_t g_ledger_q_name[16];
static uint32_t g_ledger_n_q;

void ds4_gpu_ledger_note_queue(id<MTLCommandQueue> q, const char *role) {
    if (!g_ledger_on || !q) return;
    const uint16_t id = ds4_ledger_intern(role);
    os_unfair_lock_lock(&g_ledger_q_lock);
    for (uint32_t i = 0; i < g_ledger_n_q; i++)
        if (g_ledger_q_ptr[i] == (__bridge const void *)q) { g_ledger_q_name[i] = id; goto done; }
    if (g_ledger_n_q < 16) {
        g_ledger_q_ptr[g_ledger_n_q] = (__bridge const void *)q;
        g_ledger_q_name[g_ledger_n_q++] = id;
    }
done:
    os_unfair_lock_unlock(&g_ledger_q_lock);
    if (!q.label) q.label = [NSString stringWithFormat:@"ds4-q-%s", role];
    ds4_ledger_hook_queue_class(q);
}

static uint16_t ds4_ledger_queue_name(const void *q) {
    os_unfair_lock_lock(&g_ledger_q_lock);
    uint16_t id = ds4_ledger_intern("queue-unregistered");
    for (uint32_t i = 0; i < g_ledger_n_q; i++)
        if (g_ledger_q_ptr[i] == q) { id = g_ledger_q_name[i]; break; }
    os_unfair_lock_unlock(&g_ledger_q_lock);
    return id;
}

/* ---- command buffer creation ---------------------------------------- */
static id ds4_ledger_adopt_cb(id cbid, id queue) {
    if (!cbid) return cbid;
    id<MTLCommandBuffer> cb = cbid;
    ds4_ledger_hook_cb_class(cb);
    const uint64_t seq = atomic_fetch_add(&g_ledger_cb_seq, 1u) + 1u;
    const ds4_ledger_ctx ctx = g_ledger_ctx;
    const uint16_t qname = ds4_ledger_queue_name((__bridge const void *)queue);
    const uint16_t role = ds4_ledger_intern(ctx.role ? ctx.role : "unset");

    os_unfair_lock_lock(&g_ledger_tab_lock);
    ds4_ledger_cb *rec = ds4_ledger_cb_find((__bridge const void *)cb, 1);
    if (rec) {
        rec->seq = seq; rec->token = ctx.token; rec->layer = ctx.layer;
        rec->role = role; rec->queue = qname; rec->n_enc = 0;
        rec->t_create = ds4_ledger_now();
    }
    os_unfair_lock_unlock(&g_ledger_tab_lock);

    cb.label = [NSString stringWithFormat:@"ds4/cb=%llu tok=%llu lay=%d role=%s q=%s",
                (unsigned long long)seq, (unsigned long long)ctx.token, ctx.layer,
                g_ledger_names[role], g_ledger_names[qname]];

    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) {
        memset(r, 0, sizeof(*r));
        r->kind = DS4_LR_CB_CREATE; r->id = seq; r->token = ctx.token;
        r->layer = ctx.layer; r->name_id = role; r->t_ns = ds4_ledger_now();
        r->a = qname;
    }
    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        ds4_ledger_rec *f = ds4_ledger_claim();
        if (!f) return;
        memset(f, 0, sizeof(*f));
        f->kind = DS4_LR_CB_DONE; f->id = seq; f->token = ctx.token; f->layer = ctx.layer;
        f->t_ns = ds4_ledger_now();
        f->a = (uint64_t)(done.GPUStartTime * 1e9);
        f->b = (uint64_t)(done.GPUEndTime * 1e9);
        os_unfair_lock_lock(&g_ledger_tab_lock);
        ds4_ledger_cb *slot = ds4_ledger_cb_find((__bridge const void *)done, 0);
        if (slot) slot->done = 1;
        os_unfair_lock_unlock(&g_ledger_tab_lock);
    }];
    return cbid;
}

/* ---- encoder creation ------------------------------------------------ */
static id ds4_ledger_adopt_enc(id encid, id cbid, uint8_t kind) {
    if (!encid) return encid;
    const uint64_t seq = atomic_fetch_add(&g_ledger_enc_seq, 1u) + 1u;
    os_unfair_lock_lock(&g_ledger_tab_lock);
    ds4_ledger_cb *cbrec = ds4_ledger_cb_find((__bridge const void *)cbid, 0);
    ds4_ledger_enc *e = ds4_ledger_enc_find((__bridge const void *)encid, 1);
    if (e) {
        e->seq = seq;
        e->cb_seq = cbrec ? cbrec->seq : 0;
        e->token = cbrec ? cbrec->token : g_ledger_ctx.token;
        e->layer = cbrec ? cbrec->layer : g_ledger_ctx.layer;
        e->idx_in_cb = cbrec ? cbrec->n_enc++ : 0;
        e->kind = kind;
        e->t_create = ds4_ledger_now();
    }
    os_unfair_lock_unlock(&g_ledger_tab_lock);
    g_ledger_enc_cache = e;
    if (kind == 0) ds4_ledger_hook_compute_encoder_class(encid);
    else           ds4_ledger_hook_blit_encoder_class(encid);
    return encid;
}

static ds4_ledger_enc *ds4_ledger_enc_for(const void *enc) {
    ds4_ledger_enc *e = g_ledger_enc_cache;
    if (e && e->enc == enc) return e;
    os_unfair_lock_lock(&g_ledger_tab_lock);
    e = ds4_ledger_enc_find(enc, 0);
    os_unfair_lock_unlock(&g_ledger_tab_lock);
    if (e) g_ledger_enc_cache = e;
    return e;
}

/* The pipeline a dispatch will use is whatever was last bound on this thread
 * for this encoder; Metal requires encoding to be single-threaded per encoder,
 * so a thread-local is exact here. */
static _Thread_local uint16_t g_ledger_cur_kernel;

@implementation NSObject (DS4LedgerSwizzle)

- (id)ds4_lg_newComputePipelineStateWithFunction:(id<MTLFunction>)fn error:(NSError * __autoreleasing *)error {
    id pso = [self ds4_lg_newComputePipelineStateWithFunction:fn error:error];
    if (pso) ds4_ledger_note_pso((__bridge const void *)pso, [[fn name] UTF8String]);
    return pso;
}
- (id)ds4_lg_newComputePipelineStateWithDescriptor:(MTLComputePipelineDescriptor *)d
                                            options:(MTLPipelineOption)o
                                         reflection:(MTLAutoreleasedComputePipelineReflection *)r
                                              error:(NSError * __autoreleasing *)error {
    id pso = [self ds4_lg_newComputePipelineStateWithDescriptor:d options:o reflection:r error:error];
    if (pso) ds4_ledger_note_pso((__bridge const void *)pso, [[[d computeFunction] name] UTF8String]);
    return pso;
}
- (id)ds4_lg_commandBuffer {
    return ds4_ledger_adopt_cb([self ds4_lg_commandBuffer], self);
}
- (id)ds4_lg_commandBufferWithUnretainedReferences {
    return ds4_ledger_adopt_cb([self ds4_lg_commandBufferWithUnretainedReferences], self);
}
- (id)ds4_lg_computeCommandEncoder {
    return ds4_ledger_adopt_enc([self ds4_lg_computeCommandEncoder], self, 0);
}
- (id)ds4_lg_computeCommandEncoderWithDispatchType:(MTLDispatchType)t {
    return ds4_ledger_adopt_enc([self ds4_lg_computeCommandEncoderWithDispatchType:t], self, 0);
}
- (id)ds4_lg_computeCommandEncoderWithDescriptor:(MTLComputePassDescriptor *)d {
    return ds4_ledger_adopt_enc([self ds4_lg_computeCommandEncoderWithDescriptor:d], self, 0);
}
- (id)ds4_lg_blitCommandEncoder {
    return ds4_ledger_adopt_enc([self ds4_lg_blitCommandEncoder], self, 1);
}
- (id)ds4_lg_blitCommandEncoderWithDescriptor:(MTLBlitPassDescriptor *)d {
    return ds4_ledger_adopt_enc([self ds4_lg_blitCommandEncoderWithDescriptor:d], self, 1);
}
- (void)ds4_lg_commit {
    os_unfair_lock_lock(&g_ledger_tab_lock);
    ds4_ledger_cb *rec = ds4_ledger_cb_find((__bridge const void *)self, 0);
    const uint64_t seq = rec ? rec->seq : 0;
    const uint64_t tok = rec ? rec->token : 0;
    const int32_t lay = rec ? rec->layer : -1;
    const uint32_t nenc = rec ? rec->n_enc : 0;
    os_unfair_lock_unlock(&g_ledger_tab_lock);
    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) {
        memset(r, 0, sizeof(*r));
        r->kind = DS4_LR_CB_COMMIT; r->id = seq; r->token = tok; r->layer = lay;
        r->t_ns = ds4_ledger_now(); r->a = nenc;
    }
    [self ds4_lg_commit];
}
- (void)ds4_lg_waitUntilCompleted {
    os_unfair_lock_lock(&g_ledger_tab_lock);
    ds4_ledger_cb *rec = ds4_ledger_cb_find((__bridge const void *)self, 0);
    const uint64_t seq = rec ? rec->seq : 0;
    os_unfair_lock_unlock(&g_ledger_tab_lock);
    const uint64_t t0 = ds4_ledger_now();
    os_signpost_id_t sid = os_signpost_id_generate(g_ledger_poi);
    os_signpost_interval_begin(g_ledger_poi, sid, "cpu-wait", "cb=%llu", (unsigned long long)seq);
    [self ds4_lg_waitUntilCompleted];
    os_signpost_interval_end(g_ledger_poi, sid, "cpu-wait", "cb=%llu", (unsigned long long)seq);
    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) {
        memset(r, 0, sizeof(*r));
        r->kind = DS4_LR_CPU_WAIT; r->id = seq; r->token = g_ledger_ctx.token;
        r->layer = g_ledger_ctx.layer; r->name_id = ds4_ledger_intern("waitUntilCompleted");
        r->t_ns = t0; r->a = ds4_ledger_now();
    }
}
- (void)ds4_lg_encodeWaitForEvent:(id<MTLEvent>)e value:(uint64_t)v {
    os_unfair_lock_lock(&g_ledger_tab_lock);
    ds4_ledger_cb *rec = ds4_ledger_cb_find((__bridge const void *)self, 0);
    const uint64_t seq = rec ? rec->seq : 0;
    os_unfair_lock_unlock(&g_ledger_tab_lock);
    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) {
        memset(r, 0, sizeof(*r));
        r->kind = DS4_LR_WAIT_EVENT; r->id = seq; r->token = g_ledger_ctx.token;
        r->layer = g_ledger_ctx.layer; r->t_ns = ds4_ledger_now(); r->a = v;
        r->name_id = ds4_ledger_intern([[e label] UTF8String] ?: "event");
    }
    [self ds4_lg_encodeWaitForEvent:e value:v];
}
- (void)ds4_lg_encodeSignalEvent:(id<MTLEvent>)e value:(uint64_t)v {
    os_unfair_lock_lock(&g_ledger_tab_lock);
    ds4_ledger_cb *rec = ds4_ledger_cb_find((__bridge const void *)self, 0);
    const uint64_t seq = rec ? rec->seq : 0;
    os_unfair_lock_unlock(&g_ledger_tab_lock);
    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) {
        memset(r, 0, sizeof(*r));
        r->kind = DS4_LR_SIGNAL_EVENT; r->id = seq; r->token = g_ledger_ctx.token;
        r->layer = g_ledger_ctx.layer; r->t_ns = ds4_ledger_now(); r->a = v;
        r->name_id = ds4_ledger_intern([[e label] UTF8String] ?: "event");
    }
    [self ds4_lg_encodeSignalEvent:e value:v];
}
- (void)ds4_lg_setComputePipelineState:(id<MTLComputePipelineState>)pso {
    g_ledger_cur_kernel = ds4_ledger_pso_name((__bridge const void *)pso);
    [self ds4_lg_setComputePipelineState:pso];
}
- (void)ds4_lg_dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {
    ds4_ledger_enc *e = ds4_ledger_enc_for((__bridge const void *)self);
    if (e) {
        e->n_dispatch++;
        uint16_t k = g_ledger_cur_kernel, i = 0;
        for (; i < e->n_kernel; i++) if (e->kernel_name[i] == k) break;
        if (i == e->n_kernel && e->n_kernel < DS4_LEDGER_ENC_KERNELS) {
            e->kernel_name[i] = k; e->kernel_count[i] = 0; e->n_kernel++;
        }
        if (i < DS4_LEDGER_ENC_KERNELS) e->kernel_count[i]++;
    }
    [self ds4_lg_dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
}
- (void)ds4_lg_dispatchThreads:(MTLSize)th threadsPerThreadgroup:(MTLSize)tpt {
    ds4_ledger_enc *e = ds4_ledger_enc_for((__bridge const void *)self);
    if (e) {
        e->n_dispatch++;
        uint16_t k = g_ledger_cur_kernel, i = 0;
        for (; i < e->n_kernel; i++) if (e->kernel_name[i] == k) break;
        if (i == e->n_kernel && e->n_kernel < DS4_LEDGER_ENC_KERNELS) {
            e->kernel_name[i] = k; e->kernel_count[i] = 0; e->n_kernel++;
        }
        if (i < DS4_LEDGER_ENC_KERNELS) e->kernel_count[i]++;
    }
    [self ds4_lg_dispatchThreads:th threadsPerThreadgroup:tpt];
}
- (void)ds4_lg_copyFromBuffer:(id<MTLBuffer>)src sourceOffset:(NSUInteger)so
                     toBuffer:(id<MTLBuffer>)dst destinationOffset:(NSUInteger)dof
                         size:(NSUInteger)size {
    ds4_ledger_enc *e = ds4_ledger_enc_for((__bridge const void *)self);
    if (e) { e->n_dispatch++; e->bytes += size; }
    [self ds4_lg_copyFromBuffer:src sourceOffset:so toBuffer:dst destinationOffset:dof size:size];
}
- (void)ds4_lg_endEncoding {
    ds4_ledger_enc *e = ds4_ledger_enc_for((__bridge const void *)self);
    if (e) {
        char comp[768]; size_t used = 0;
        if (e->kind == 1) {
            snprintf(comp, sizeof(comp), "copy");
        } else if (e->n_kernel == 0) {
            snprintf(comp, sizeof(comp), "none");
        } else {
            for (uint16_t i = 0; i < e->n_kernel && used + 1 < sizeof(comp); i++)
                used += (size_t)snprintf(comp + used, sizeof(comp) - used, "%s%s*%u",
                                         i ? "+" : "", g_ledger_names[e->kernel_name[i]],
                                         e->kernel_count[i]);
        }
        /* An encoder that ran more than one kernel is reported as mixed. It is
         * not split to make the label sharper: splitting it would change the
         * submission shape this ledger exists to measure. */
        const char *stage = (e->kind == 1) ? "blit"
                          : (e->n_kernel == 1) ? "single"
                          : (e->n_kernel == 0) ? "empty" : "mixed";
        ((id<MTLCommandEncoder>)self).label =
            [NSString stringWithFormat:@"enc=%llu i=%u kind=%s stage=%s n=%u k=%s",
             (unsigned long long)e->seq, e->idx_in_cb, e->kind ? "blit" : "compute",
             stage, e->n_dispatch, comp];
        ds4_ledger_rec *r = ds4_ledger_claim();
        if (r) {
            memset(r, 0, sizeof(*r));
            r->kind = e->kind ? DS4_LR_BLIT : DS4_LR_ENC;
            r->enc_kind = e->kind; r->id = e->seq; r->parent = e->cb_seq;
            r->token = e->token; r->layer = e->layer;
            r->name_id = ds4_ledger_intern(comp);
            r->t_ns = e->t_create; r->a = e->n_dispatch; r->b = e->bytes;
        }
        os_unfair_lock_lock(&g_ledger_tab_lock);
        e->enc = DS4_LEDGER_DEAD;
        os_unfair_lock_unlock(&g_ledger_tab_lock);
        if (g_ledger_enc_cache == e) g_ledger_enc_cache = NULL;
    }
    [self ds4_lg_endEncoding];
}
@end

/* ---- public API ------------------------------------------------------ */

int ds4_gpu_ledger_active(void) { return g_ledger_on; }

void ds4_gpu_ledger_probe(id<MTLDevice> device) {
    static int probed;
    if (probed) return;
    probed = 1;
    const char *path = getenv("DS4_METAL_LEDGER");
    if (!path || !path[0] || !device) return;
    if (getenv("DS4_METAL_ENCODER_TIMELINE")) {
        fprintf(stderr, "ds4: DS4_METAL_LEDGER and DS4_METAL_ENCODER_TIMELINE hook the same "
                        "selectors; run one at a time. Ledger not started.\n");
        return;
    }
    g_ledger_recs = calloc(DS4_LEDGER_MAX_RECS, sizeof(ds4_ledger_rec));
    if (!g_ledger_recs) return;
    mach_timebase_info(&g_ledger_tb);
    g_ledger_path = strdup(path);
    g_ledger_poi = os_log_create("com.ds4.ledger", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    const char *lim = getenv("DS4_METAL_LEDGER_TOKENS");
    g_ledger_token_limit = lim && lim[0] ? strtoull(lim, NULL, 10) : 0;
    g_ledger_on = 1;
    Class dev = object_getClass(device);
    ds4_ledger_swizzle(dev, @selector(newComputePipelineStateWithFunction:error:),
                            @selector(ds4_lg_newComputePipelineStateWithFunction:error:));
    ds4_ledger_swizzle(dev, @selector(newComputePipelineStateWithDescriptor:options:reflection:error:),
                            @selector(ds4_lg_newComputePipelineStateWithDescriptor:options:reflection:error:));
    atexit(ds4_gpu_ledger_dump);
    fprintf(stderr, "ds4: decode ledger -> %s (pid %d, timebase %u/%u, limit %llu tokens)\n",
            g_ledger_path, (int)getpid(), g_ledger_tb.numer, g_ledger_tb.denom,
            (unsigned long long)g_ledger_token_limit);
}

void ds4_gpu_ledger_token_begin(unsigned long long pos) {
    if (!g_ledger_on) return;
    /* pos restarts with every session, so two tokens in one run share it. The
     * ledger's identity is a counter; pos is carried alongside for context. */
    g_ledger_ctx.token = atomic_fetch_add(&g_ledger_token_seq, 1u) + 1u;
    g_ledger_ctx.pos = pos;
    g_ledger_ctx.layer = -1;
    g_ledger_ctx.role = "decode";
    g_ledger_tok_sid = os_signpost_id_generate(g_ledger_poi);
    os_signpost_interval_begin(g_ledger_poi, g_ledger_tok_sid, "token", "tok=%llu pos=%llu",
                               (unsigned long long)g_ledger_ctx.token, pos);
    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) { memset(r, 0, sizeof(*r)); r->kind = DS4_LR_TOKEN_BEGIN;
             r->token = g_ledger_ctx.token; r->layer = -1; r->a = pos;
             r->t_ns = ds4_ledger_now(); }
}

void ds4_gpu_ledger_token_end(void) {
    if (!g_ledger_on) return;
    ds4_ledger_rec *r = ds4_ledger_claim();
    if (r) { memset(r, 0, sizeof(*r)); r->kind = DS4_LR_TOKEN_END;
             r->token = g_ledger_ctx.token; r->layer = -1; r->a = g_ledger_ctx.pos;
             r->t_ns = ds4_ledger_now(); }
    os_signpost_interval_end(g_ledger_poi, g_ledger_tok_sid, "token", "tok=%llu pos=%llu",
                             (unsigned long long)g_ledger_ctx.token,
                             (unsigned long long)g_ledger_ctx.pos);
    g_ledger_ctx.layer = -1;
    g_ledger_ctx.role = "between-tokens";
    const uint64_t seen = atomic_fetch_add(&g_ledger_tokens_seen, 1u) + 1u;
    if (g_ledger_token_limit && seen == g_ledger_token_limit) ds4_gpu_ledger_dump();
}

void ds4_gpu_ledger_layer(int il) { if (g_ledger_on) g_ledger_ctx.layer = il; }
void ds4_gpu_ledger_role(const char *role) { if (g_ledger_on) g_ledger_ctx.role = role; }

void ds4_gpu_ledger_push(unsigned long long token, int layer, const char *role) {
    if (!g_ledger_on) return;
    g_ledger_saved = g_ledger_ctx;
    g_ledger_ctx.token = token;
    g_ledger_ctx.layer = layer;
    g_ledger_ctx.role = role;
}
void ds4_gpu_ledger_pop(void) { if (g_ledger_on) g_ledger_ctx = g_ledger_saved; }

void ds4_gpu_ledger_mark(const char *name, const char *detail) {
    if (!g_ledger_on) return;
    os_signpost_event_emit(g_ledger_poi, g_ledger_tok_sid, "mark",
                           "tok=%llu what=%s %s", (unsigned long long)g_ledger_ctx.token,
                           name, detail ? detail : "");
}

void ds4_gpu_ledger_wait_begin(const char *what) {
    if (!g_ledger_on) return;
    os_signpost_interval_begin(g_ledger_poi, g_ledger_tok_sid, "host-wait",
                               "tok=%llu what=%s", (unsigned long long)g_ledger_ctx.token, what);
}
void ds4_gpu_ledger_wait_end(const char *what) {
    if (!g_ledger_on) return;
    os_signpost_interval_end(g_ledger_poi, g_ledger_tok_sid, "host-wait",
                             "tok=%llu what=%s", (unsigned long long)g_ledger_ctx.token, what);
}

/* Records are kept in memory and written once, so that the hot path never
 * touches a FILE* and the ledger cannot serialise the decode thread behind
 * its own logging. */
void ds4_gpu_ledger_dump(void) {
    static _Atomic int dumped;
    int expect = 0;
    if (!g_ledger_on || !atomic_compare_exchange_strong(&dumped, &expect, 1)) return;
    g_ledger_out = fopen(g_ledger_path, "w");
    if (!g_ledger_out) {
        fprintf(stderr, "ds4: cannot write ledger %s\n", g_ledger_path);
        return;
    }
    uint32_t n = atomic_load(&g_ledger_n_rec);
    const uint64_t dropped = atomic_load(&g_ledger_dropped);
    if (n > DS4_LEDGER_MAX_RECS) n = DS4_LEDGER_MAX_RECS;
    fprintf(g_ledger_out, "# ds4 decode ledger v1 pid=%d timebase=%u/%u\n",
            (int)getpid(), g_ledger_tb.numer, g_ledger_tb.denom);
    fprintf(g_ledger_out, "# records=%u dropped=%llu cbs=%llu encoders=%llu\n",
            n, (unsigned long long)dropped,
            (unsigned long long)atomic_load(&g_ledger_cb_seq),
            (unsigned long long)atomic_load(&g_ledger_enc_seq));
    fprintf(g_ledger_out, "# TB  tok t_ns pos                      token begin\n");
    fprintf(g_ledger_out, "# TE  tok t_ns pos                      token end\n");
    fprintf(g_ledger_out, "# CC  cb tok lay t_ns role queue        command buffer created\n");
    fprintf(g_ledger_out, "# CM  cb tok lay t_ns n_encoders        committed\n");
    fprintf(g_ledger_out, "# CD  cb tok lay t_ns gpu_start gpu_end completed (Metal API times)\n");
    fprintf(g_ledger_out, "# EN  enc cb tok lay t_create n_dispatch kernels   compute encoder\n");
    fprintf(g_ledger_out, "# BL  enc cb tok lay t_create n_ops bytes          blit encoder\n");
    fprintf(g_ledger_out, "# EW/ES cb tok lay t_ns value event                encoded wait/signal\n");
    fprintf(g_ledger_out, "# HW  cb tok lay t_begin t_end what                host wait\n");
    for (uint32_t i = 0; i < n; i++) {
        const ds4_ledger_rec *r = &g_ledger_recs[i];
        const char *nm = (r->name_id < g_ledger_n_names) ? g_ledger_names[r->name_id] : "?";
        switch (r->kind) {
        case DS4_LR_TOKEN_BEGIN:
            fprintf(g_ledger_out, "TB %llu %llu %llu\n", (unsigned long long)r->token,
                    (unsigned long long)r->t_ns, (unsigned long long)r->a); break;
        case DS4_LR_TOKEN_END:
            fprintf(g_ledger_out, "TE %llu %llu %llu\n", (unsigned long long)r->token,
                    (unsigned long long)r->t_ns, (unsigned long long)r->a); break;
        case DS4_LR_CB_CREATE:
            fprintf(g_ledger_out, "CC %llu %llu %d %llu %s %s\n",
                    (unsigned long long)r->id, (unsigned long long)r->token, r->layer,
                    (unsigned long long)r->t_ns, nm,
                    (r->a < g_ledger_n_names) ? g_ledger_names[r->a] : "?"); break;
        case DS4_LR_CB_COMMIT:
            fprintf(g_ledger_out, "CM %llu %llu %d %llu %llu\n",
                    (unsigned long long)r->id, (unsigned long long)r->token, r->layer,
                    (unsigned long long)r->t_ns, (unsigned long long)r->a); break;
        case DS4_LR_CB_DONE:
            fprintf(g_ledger_out, "CD %llu %llu %d %llu %llu %llu\n",
                    (unsigned long long)r->id, (unsigned long long)r->token, r->layer,
                    (unsigned long long)r->t_ns, (unsigned long long)r->a,
                    (unsigned long long)r->b); break;
        case DS4_LR_ENC:
            fprintf(g_ledger_out, "EN %llu %llu %llu %d %llu %llu %s\n",
                    (unsigned long long)r->id, (unsigned long long)r->parent,
                    (unsigned long long)r->token, r->layer, (unsigned long long)r->t_ns,
                    (unsigned long long)r->a, nm); break;
        case DS4_LR_BLIT:
            fprintf(g_ledger_out, "BL %llu %llu %llu %d %llu %llu %llu\n",
                    (unsigned long long)r->id, (unsigned long long)r->parent,
                    (unsigned long long)r->token, r->layer, (unsigned long long)r->t_ns,
                    (unsigned long long)r->a, (unsigned long long)r->b); break;
        case DS4_LR_WAIT_EVENT:
        case DS4_LR_SIGNAL_EVENT:
            fprintf(g_ledger_out, "%s %llu %llu %d %llu %llu %s\n",
                    r->kind == DS4_LR_WAIT_EVENT ? "EW" : "ES",
                    (unsigned long long)r->id, (unsigned long long)r->token, r->layer,
                    (unsigned long long)r->t_ns, (unsigned long long)r->a, nm); break;
        case DS4_LR_CPU_WAIT:
            fprintf(g_ledger_out, "HW %llu %llu %d %llu %llu %s\n",
                    (unsigned long long)r->id, (unsigned long long)r->token, r->layer,
                    (unsigned long long)r->t_ns, (unsigned long long)r->a, nm); break;
        default: break;
        }
    }
    fclose(g_ledger_out);
    g_ledger_out = NULL;
    fprintf(stderr, "ds4: ledger written: %u records, %llu dropped -> %s\n",
            n, (unsigned long long)dropped, g_ledger_path);
}

#endif /* DS4_METAL_LEDGER_H */
