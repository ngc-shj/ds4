/*
 * The Metal 4 backend: command contexts, each a graph of what it reads and
 * writes, submitted on one queue.
 *
 * Selected with the process's backend (g_backend == 4), it carries the whole
 * of a V4.1 request, and no Metal 3 queue exists.
 *
 * A context (DS4M4Ctx) is one submission in flight and owns everything that
 * submission needs - allocators and command buffers, argument tables, the
 * constant scratch, the residency set, its stamp and commit feedback - and
 * nothing is shared between two contexts. The engine's batch handle (DS4Cmd)
 * names one.
 *
 * What the engine encodes into a context is not encoded yet. Each dispatch or
 * copy becomes a node: its pipeline, bindings, constants and grid, and the
 * byte ranges it reads and writes. Which binding is read and which written
 * comes from the pipeline's own reflection. A binding's range is what its
 * call site states (ds4_enc_buf_n); one that states nothing runs from its
 * offset to the end of its buffer, and those are counted by call site so they
 * can be driven to none. A copy covers what it copies, a gated dispatch reads
 * its twelve bytes of grid, and a resource declared with useResource is read
 * or written whole. Views of the mapped model are read-only pages and carry no
 * range. Each written range bumps its buffer's generation, and every range
 * records the generation it saw.
 *
 * At submission the graph is encoded. A barrier goes in only in front of a
 * node whose ranges conflict with something since the last barrier - read
 * after write, write after read, write after write - so the engine's own
 * barriers and section ends order nothing here; they are counted. A write to
 * a buffer the queue holds resident is fenced on both sides, because kernels
 * reach those by raw address without binding them. A large graph can be cut
 * at barrier points into parts encoded on worker threads, each into its own
 * command buffer with its own table, and sent together with commit:count:.
 *
 * Every command buffer opens with a queue-stage barrier and closes with one:
 * contexts committed one after another run in order, and what the host reads
 * after a completion is visible.
 */
#ifndef DS4_METAL_MTL4_H
#define DS4_METAL_MTL4_H

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000

/* Contexts that may be recording or in flight at once. The engine keeps one
 * segment prepared behind the running one, so two are live; the gate's miss
 * ring is four deep, which is the ceiling on how far ahead it could ever go. */
#define DS4_MTL4_PIPE          4u
#define DS4_MTL4_SCRATCH_BYTES (8u << 20)
#define DS4_MTL4_SCRATCH_ALIGN 256u
#define DS4_MTL4_MAX_BINDINGS  DS4_ENC_SLOTS
/* Parts a graph may be encoded in, and the smallest part worth a thread. */
#define DS4_MTL4_PARTS         4u
#define DS4_MTL4_PART_MIN      1024u

int ds4_gpu_flush_commands(void);
int ds4_gpu_wait_committed(void);
static void ds4_gpu_close_batch_encoder(void);

/* Device flushes to the GPU coherence point; ResourceAlias additionally
 * promises that aliased virtual addresses agree, which is a different
 * guarantee. This engine reaches expert weights through raw addresses into
 * views made with newBufferWithBytesNoCopy, so Device alone does not imply it.
 * DS4_METAL_V41_MTL4_ALIAS=0 drops it, which is the control. */
API_AVAILABLE(macos(26.0))
static MTL4VisibilityOptions ds4_mtl4_visibility(void) {
    static int checked; static MTL4VisibilityOptions v;
    if (!checked) {
        checked = 1;
        const char *e = getenv("DS4_METAL_V41_MTL4_ALIAS");
        v = (e && e[0] && atoi(e) == 0)
            ? MTL4VisibilityOptionDevice
            : (MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias);
    }
    return v;
}

/* ---- pipelines: which bindings they read and which they write ----------- */

/* Filled where pipelines are made (ds4_gpu_new_pso), read at every dispatch.
 * Written under a lock, read without one: an entry's masks are stored before
 * its key is published. A pipeline with no entry is taken as reading and
 * writing every binding. */
#define DS4_PSO_ACCESS_CAP 8192u
typedef struct {
    _Atomic(const void *) pso;
    uint32_t read, write;
} ds4_pso_access_entry;
static ds4_pso_access_entry g_pso_access[DS4_PSO_ACCESS_CAP];
static pthread_mutex_t g_pso_access_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long g_pso_access_full, g_pso_access_unknown;

static inline uint32_t ds4_ptr_slot(const void *p, uint32_t cap) {
    const uintptr_t k = (uintptr_t)p;
    return (uint32_t)((k >> 4) ^ (k >> 17)) & (cap - 1u);
}

static void ds4_pso_access_note(id pso, MTLComputePipelineReflection *refl) {
    if (!pso || !refl) return;
    uint32_t r = 0, w = 0;
    for (id<MTLBinding> b in refl.bindings) {
        if (b.type != MTLBindingTypeBuffer || b.index >= DS4_MTL4_MAX_BINDINGS) continue;
        const uint32_t bit = 1u << b.index;
        switch (b.access) {
        case MTLBindingAccessReadOnly: r |= bit; break;
        case MTLBindingAccessWriteOnly: w |= bit; break;
        default: r |= bit; w |= bit; break;
        }
    }
    const void *key = (__bridge const void *)pso;
    pthread_mutex_lock(&g_pso_access_mu);
    uint32_t s = ds4_ptr_slot(key, DS4_PSO_ACCESS_CAP);
    for (uint32_t n = 0; n < DS4_PSO_ACCESS_CAP; n++, s = (s + 1u) & (DS4_PSO_ACCESS_CAP - 1u)) {
        const void *k = atomic_load_explicit(&g_pso_access[s].pso, memory_order_relaxed);
        if (k && k != key) continue;
        g_pso_access[s].read = r;
        g_pso_access[s].write = w;
        atomic_store_explicit(&g_pso_access[s].pso, key, memory_order_release);
        pthread_mutex_unlock(&g_pso_access_mu);
        return;
    }
    g_pso_access_full++;
    pthread_mutex_unlock(&g_pso_access_mu);
}

static int ds4_pso_access(const void *key, uint32_t *r, uint32_t *w) {
    uint32_t s = ds4_ptr_slot(key, DS4_PSO_ACCESS_CAP);
    for (uint32_t n = 0; n < DS4_PSO_ACCESS_CAP; n++, s = (s + 1u) & (DS4_PSO_ACCESS_CAP - 1u)) {
        const void *k = atomic_load_explicit(&g_pso_access[s].pso, memory_order_acquire);
        if (!k) return 0;
        if (k == key) { *r = g_pso_access[s].read; *w = g_pso_access[s].write; return 1; }
    }
    return 0;
}

/* ---- buffers nothing can write ------------------------------------------- */

/* Views of the mapped model are read-only pages: a kernel whose weight
 * argument is not declared const reflects as read-write, but a write there
 * would fault, so such a binding can conflict with nothing and carries no
 * range. Decided once a buffer, from the page protection itself. */
static NSMapTable *g_m4_readonly;     /* buffer (weak) -> @YES / @NO */
static int ds4_m4_readonly(id<MTLBuffer> b) {
    if (b.storageMode != MTLStorageModeShared) return 0;
    if (!g_m4_readonly) g_m4_readonly = [NSMapTable weakToStrongObjectsMapTable];
    NSNumber *v = [g_m4_readonly objectForKey:b];
    if (!v) {
        mach_vm_address_t addr = (mach_vm_address_t)(uintptr_t)[b contents];
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        int ro = 0;
        if (mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64,
                           (vm_region_info_t)&info, &count, &object) == KERN_SUCCESS)
            ro = (info.protection & VM_PROT_WRITE) == 0;
        v = @(ro);
        [g_m4_readonly setObject:v forKey:b];
    }
    return v.boolValue;
}

/* ---- the graph ------------------------------------------------------------ */

typedef struct {
    const void *buf;
    uint64_t lo, hi;
    uint32_t gen;               /* the buffer's generation this range saw */
    uint8_t write;
} ds4_m4_range;

typedef struct {
    uint8_t kind;
    uint8_t nbind;              /* table slots 0..nbind-1 this node sets */
    uint8_t tgmem_set;
    uint8_t barrier;            /* filled at submission */
    uint8_t fence;              /* orders everything around it; see below */
    uint32_t tgmem[4];
    const void *pso;            /* held by the context's pso set */
    uint32_t bind_at;           /* into the context's address pool */
    uint32_t range_at, n_range; /* into the context's range pool */
    MTLSize a, b;               /* grid or thread count, threadgroup */
    uint64_t indirect;          /* GPU address of the indirect arguments */
    const void *src, *dst;      /* a copy; held resident by the context */
    uint64_t soff, doff, size;
} ds4_m4_node;

/* A buffer's generation: how many writes to it have been recorded. The
 * direct-mapped table forgets a buffer when another lands on its slot, which
 * restarts its count - the number labels a version, it does not order one. */
#define DS4_M4_GEN_SLOTS 4096u
static struct { const void *buf; uint32_t gen; } g_m4_gen[DS4_M4_GEN_SLOTS];
static uint32_t ds4_m4_gen(const void *buf, int bump) {
    const uint32_t s = ds4_ptr_slot(buf, DS4_M4_GEN_SLOTS);
    if (g_m4_gen[s].buf != buf) { g_m4_gen[s].buf = buf; g_m4_gen[s].gen = 0; }
    if (bump) g_m4_gen[s].gen++;
    return g_m4_gen[s].gen;
}

/* Bindings recorded with no stated length, by ds4_metal.m line: the call
 * sites whose range still runs to the end of the buffer. */
#define DS4_M4_CENSUS_LINES 65536u
static unsigned long long g_m4_coarse_line[DS4_M4_CENSUS_LINES];

/* ---- one context ------------------------------------------------------- */

/* A direct-mapped memo of which resources a context has marked resident in
 * its current generation. The engine binds the same few hundred buffers
 * thousands of times a token; a hash-table probe and three message sends for
 * each of those binds was host encode time the Metal 3 encoder does not
 * spend. A hit is exact: the entry is written only after the mark, and the
 * context holds every resource it marked until it completes, so an address in
 * the memo cannot be reused by another object within the same generation. A
 * miss just falls through to the set. */
#define DS4_MTL4_MARK_MEMO 4096u

/* One submission in flight, and everything it needs: nothing here is shared
 * with another context, and nothing a context owns is reachable except
 * through it. */
API_AVAILABLE(macos(26.0))
@interface DS4M4Ctx : NSObject {
@public
    id<MTL4CommandAllocator> alloc[DS4_MTL4_PARTS];
    id<MTL4CommandBuffer> cb[DS4_MTL4_PARTS];
    /* A table a part: Metal snapshots a table at each dispatch, so one serves
     * a whole part, but parts are encoded at once on different threads. */
    id<MTL4ArgumentTable> table[DS4_MTL4_PARTS];
    /* setBytes has no inline path in Metal 4: constants are copied here and
     * bound by address. Reused only once this context has completed. */
    id<MTLBuffer> scratch;
    NSUInteger scratch_used;
    /* What this context reaches, resident for exactly as long as it is in
     * flight: filled while it is recorded, committed with it, emptied once
     * its feedback says it completed. */
    id<MTLResidencySet> resset;
    NSHashTable *resident;
    NSHashTable *psos;
    unsigned long long res_bytes;
    /* Whether anything has been recorded since it was begun. A context with
     * nothing in it is not sent: it stays open and takes the next batch. */
    BOOL used;
    /* The event value that says this context is done. 0 while recording. */
    uint64_t stamp;
    unsigned parts;             /* command buffers its last submission used */
    /* The commit feedback for this context's stamp. Written on the feedback
     * queue; `fb_stamp` is stored last and publishes the rest. */
    _Atomic uint64_t fb_stamp;
    _Atomic int fb_failed;
    double gpu_start, gpu_end;
    /* The generation of what is marked resident, and the memo of it. 0 is
     * never a live generation, so a zeroed memo entry never hits. */
    uint32_t gen;
    struct { const void *res; uint32_t gen; } memo[DS4_MTL4_MARK_MEMO];
    /* The graph. */
    ds4_m4_node *nodes;
    uint32_t n_nodes, cap_nodes;
    uint64_t *addrs;
    uint32_t n_addrs, cap_addrs;
    ds4_m4_range *ranges;
    uint32_t n_ranges, cap_ranges;
    const void *last_pso;
    /* The next node must wait for everything before it: set after a fence. */
    uint8_t fence_next;
    /* The encoder timeline (DS4_METAL_ENCODER_TIMELINE), when it is on: a
     * record a node, and a heap its start and end timestamps go into. */
    ds4_timeline_rec *tl;
    uint32_t tl_cap;
    id<MTL4CounterHeap> tl_heap;
    NSUInteger tl_heap_cap;
    uint64_t tl_seq;
    /* The decode ledger's record of this submission (DS4_METAL_LEDGER). */
    ds4_ledger_m4 ledger;
}
/* The engine's handle for this context, and its recorder. */
@property (nonatomic, strong) DS4Cmd *cmd;
@property (nonatomic, strong) DS4Enc *enc;
@end

@implementation DS4M4Ctx
- (void)dealloc {
    free(nodes);
    free(addrs);
    free(ranges);
    free(tl);
}
@end

static id<MTL4CommandQueue> g_mtl4_queue API_AVAILABLE(macos(26.0));
static NSMutableArray *g_mtl4_pool;          /* DS4M4Ctx, all of them */
/* The context the engine is recording into: a cursor, not an owner. */
static DS4M4Ctx *g_mtl4_cur API_AVAILABLE(macos(26.0));
static id<MTLSharedEvent> g_mtl4_event;
/* Defined with the Metal 3 wait below; the GPU's own span of what a wait
 * covered, from each context's feedback, goes into the same accumulator. */
static double g_gpu_span_accum_ms;
static uint64_t g_gpu_span_cbs;
static uint64_t g_mtl4_stamp;                /* last value handed to a commit */
static uint64_t g_mtl4_done;                 /* last value waited for */

static unsigned long long g_mtl4_dispatches, g_mtl4_segments;
/* Contexts actually sent, against contexts begun: the two differ by the ones
 * an abort discards, and reporting begins as though they were submissions is
 * how a submission count stops being one. */
static unsigned long long g_mtl4_commits;
unsigned long long ds4_gpu_backend_m3_fallback(void);

/* Contexts sent and dispatches recorded on this backend, and Metal 3 work
 * noted while it was selected - which is the number that has to be zero. */
void ds4_gpu_mtl4_counts(unsigned long long *segments,
                         unsigned long long *dispatches,
                         unsigned long long *m3_fallback) {
    if (segments) *segments = g_mtl4_commits;
    if (dispatches) *dispatches = g_mtl4_dispatches;
    if (m3_fallback) *m3_fallback = ds4_gpu_backend_m3_fallback();
}

/* ---- residency --------------------------------------------------------- */

/* Metal 4 tracks no residency implicitly. What a context binds or declares
 * goes into that context's own set - what Metal 3 does implicitly for a
 * command buffer. The model and the expert cache live in the process-lifetime
 * sets on the queue; everything else - a session's arena, a transient, a
 * tensor freed since - is resident only while a context that uses it is in
 * flight. A single long-lived set held a session's gigabytes past the session,
 * and the next request's allocations did not fit. */
static unsigned long long g_mtl4_res_adds, g_mtl4_res_releases, g_mtl4_res_peak;

API_AVAILABLE(macos(26.0))
static void ds4_mtl4_mark_resident(DS4M4Ctx *c, id<MTLResource> r) {
    if (!r || !c) return;
    const void *key = (__bridge const void *)r;
    const uintptr_t k = (uintptr_t)key;
    const uint32_t slot = (uint32_t)((k >> 4) ^ (k >> 16)) & (DS4_MTL4_MARK_MEMO - 1u);
    if (c->memo[slot].res == key && c->memo[slot].gen == c->gen) return;
    if (![c->resident containsObject:r]) {
        [c->resident addObject:r];
        [c->resset addAllocation:r];
        c->res_bytes += [r allocatedSize];
        g_mtl4_res_adds++;
    }
    c->memo[slot].res = key;
    c->memo[slot].gen = c->gen;
}

/* Residency sets the engine attaches to its queue: the model and the expert
 * cache, reached by raw address from the masked kernels - nothing else
 * declares them. A submission without these faults in AGX on its first
 * dispatch. */
static NSMutableArray *g_mtl4_queue_sets;
/* How many of g_mtl4_queue_sets, from the front, the queue holds. */
static NSUInteger g_mtl4_sets_attached_n;
API_AVAILABLE(macos(26.0)) static void ds4_mtl4_attach_sets(void);

void ds4_gpu_queue_residency_added(id set) {
    if (!set) return;
    if (!g_mtl4_queue_sets) g_mtl4_queue_sets = [NSMutableArray array];
    if (![g_mtl4_queue_sets containsObject:set]) [g_mtl4_queue_sets addObject:set];
    if (@available(macOS 26.0, *)) { if (g_mtl4_queue) ds4_mtl4_attach_sets(); }
}

/* A set dropped - a model released, an expert table freed. The queue lets go
 * of it too, or it would hold what the set covers resident for the life of
 * the process. */
void ds4_gpu_queue_residency_removed(id set) {
    if (!set || !g_mtl4_queue_sets) return;
    const NSUInteger i = [g_mtl4_queue_sets indexOfObject:set];
    if (i == NSNotFound) return;
    if (@available(macOS 26.0, *)) {
        if (g_mtl4_queue && i < g_mtl4_sets_attached_n) {
            [g_mtl4_queue removeResidencySet:set];
            g_mtl4_sets_attached_n--;
        }
    }
    [g_mtl4_queue_sets removeObjectAtIndex:i];
}

/* Sets attached to the Metal 4 queue, counted: a set the engine has and the
 * queue lacks makes what it holds unreadable without a fault. */
static unsigned long long g_mtl4_sets_attached;
unsigned long long ds4_gpu_mtl4_sets_attached(void) { return g_mtl4_sets_attached; }

static void ds4_mtl4_attach_sets(void) {
    if (!g_mtl4_queue) return;
    while (g_mtl4_sets_attached_n < g_mtl4_queue_sets.count) {
        [g_mtl4_queue addResidencySet:g_mtl4_queue_sets[g_mtl4_sets_attached_n]];
        g_mtl4_sets_attached_n++;
        g_mtl4_sets_attached++;
    }
}

/* Whether a buffer is held by one of the queue's sets: the model views, the
 * expert cache's slabs and tables. Kernels reach those by raw address from
 * tables without binding them, so a write to one cannot be ordered by ranges
 * alone; it is fenced instead (see ds4_m4_dispatch). */
static int ds4_m4_queue_resident(id<MTLBuffer> b) {
    if (@available(macOS 26.0, *)) {
        for (id<MTLResidencySet> set in g_mtl4_queue_sets)
            if ([set containsAllocation:b]) return 1;
    }
    return 0;
}

/* The shape of what this backend submits, counted by kind so a token can be
 * priced by structure. Plain counters. */
enum {
    DS4_M4S_CONTEXTS = 0,       /* contexts submitted */
    DS4_M4S_BUFFERS,            /* command buffers they were encoded in */
    DS4_M4S_SPLIT,              /* contexts encoded in more than one part */
    DS4_M4S_NODES,
    DS4_M4S_DISPATCH,
    DS4_M4S_COPY,
    DS4_M4S_BAR_GRAPH,          /* barriers the graph put in */
    DS4_M4S_BAR_QUEUE,          /* queue barriers opening and closing buffers */
    DS4_M4S_ENGINE_BARRIER,     /* barriers the engine asked for: counted only */
    DS4_M4S_SECTIONS,           /* sections the engine ended: counted only */
    DS4_M4S_SET_ADDRESS,        /* buffer bindings recorded */
    DS4_M4S_SET_BYTES,          /* constant bindings recorded */
    DS4_M4S_BYTES_COPIED,       /* bytes copied into the constant scratch */
    DS4_M4S_WAIT_CALLS,         /* waits that actually blocked */
    DS4_M4S_WAIT_NS,            /* host time blocked in them */
    DS4_M4S_COARSE,             /* bindings recorded with no stated length */
    DS4_M4S_FENCES,             /* writes to queue-resident buffers, fenced */
    DS4_M4S_KINDS
};
static unsigned long long g_m4s[DS4_M4S_KINDS];
void ds4_gpu_mtl4_structure(unsigned long long out[16]);
void ds4_gpu_mtl4_structure(unsigned long long out[16]) {
    for (int i = 0; i < 16; i++) out[i] = g_m4s[i];
}
unsigned long long ds4_gpu_mtl4_fences(void) { return g_m4s[DS4_M4S_FENCES]; }

/* The call sites whose bindings still carry no length, most used first,
 * counted since the process started. */
void ds4_gpu_mtl4_coarse_report(FILE *f, unsigned max_lines);
void ds4_gpu_mtl4_coarse_report(FILE *f, unsigned max_lines) {
    unsigned long long total = 0, sites = 0;
    for (unsigned i = 0; i < DS4_M4_CENSUS_LINES; i++)
        if (g_m4_coarse_line[i]) { total += g_m4_coarse_line[i]; sites++; }
    fprintf(f, "ds4: Metal 4 bindings with no stated length: %llu at %llu call sites\n",
            total, sites);
    if (!sites) return;
    unsigned long long *left = malloc(sizeof(g_m4_coarse_line));
    if (!left) return;
    memcpy(left, g_m4_coarse_line, sizeof(g_m4_coarse_line));
    for (unsigned shown = 0; shown < max_lines; shown++) {
        unsigned best = 0;
        unsigned long long n = 0;
        for (unsigned i = 0; i < DS4_M4_CENSUS_LINES; i++)
            if (left[i] > n) { n = left[i]; best = i; }
        if (!n) break;
        fprintf(f, "ds4:   ds4_metal.m:%u %llu\n", best, n);
        left[best] = 0;
    }
    free(left);
}

/* ---- recording ------------------------------------------------------------ */

static int ds4_m4_grow(void **p, uint32_t *cap, uint32_t need, size_t elem) {
    if (need <= *cap) return 1;
    uint32_t n = *cap ? *cap : 256u;
    while (n < need) n *= 2u;
    void *q = realloc(*p, (size_t)n * elem);
    if (!q) return 0;
    *p = q;
    *cap = n;
    return 1;
}

static void ds4_m4_pso(DS4Enc *e, id<MTLComputePipelineState> pso) {
    e->pso = pso;
}

static void ds4_m4_buf(DS4Enc *e, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger length,
                       NSUInteger index, uint32_t line) {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 buffer index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    g_m4s[DS4_M4S_SET_ADDRESS]++;
    const uint32_t bit = 1u << index;
    e->constant &= ~bit;
    /* setBuffer:nil clears the slot in Metal 3; so does this. */
    if (!buffer) {
        e->bound &= ~bit;
        e->buf[index] = nil;
        e->addr[index] = 0;
        return;
    }
    if (@available(macOS 26.0, *)) ds4_mtl4_mark_resident(e->m4, buffer);
    e->bound |= bit;
    e->buf[index] = buffer;
    e->off[index] = offset;
    e->len[index] = length;
    e->line[index] = line;
    e->addr[index] = buffer.gpuAddress + offset;
}

static void ds4_m4_bytes(DS4Enc *e, const void *bytes, NSUInteger length, NSUInteger index) {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 bytes index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    if (@available(macOS 26.0, *)) {
        DS4M4Ctx *c = e->m4;
        const NSUInteger off = (c->scratch_used + DS4_MTL4_SCRATCH_ALIGN - 1u) &
                               ~(NSUInteger)(DS4_MTL4_SCRATCH_ALIGN - 1u);
        if (off + length > DS4_MTL4_SCRATCH_BYTES) {
            /* Binding nothing here would be a wrong answer with no symptom. */
            fprintf(stderr, "ds4: Metal 4 scratch exhausted at %lu + %lu bytes\n",
                    (unsigned long)off, (unsigned long)length);
            abort();
        }
        g_m4s[DS4_M4S_SET_BYTES]++;
        g_m4s[DS4_M4S_BYTES_COPIED] += length;
        memcpy((char *)[c->scratch contents] + off, bytes, length);
        c->scratch_used = off + length;
        const uint32_t bit = 1u << index;
        e->bound |= bit;
        e->constant |= bit;
        e->buf[index] = nil;
        e->addr[index] = c->scratch.gpuAddress + off;
    }
}

static void ds4_m4_tgmem(DS4Enc *e, NSUInteger length, NSUInteger index) {
    if (index >= 4) {
        fprintf(stderr, "ds4: Metal 4 threadgroup memory index %lu\n", (unsigned long)index);
        abort();
    }
    e->tgmem[index] = (uint32_t)length;
    e->tgmem_set |= (uint8_t)(1u << index);
}

static void ds4_m4_use(DS4Enc *e, id<MTLResource> resource, MTLResourceUsage usage) {
    if (!resource) return;
    if (@available(macOS 26.0, *)) ds4_mtl4_mark_resident(e->m4, resource);
    /* Only buffers carry a range. The engine declares no textures. */
    if (![resource conformsToProtocol:@protocol(MTLBuffer)]) return;
    if (e->n_use == sizeof(e->use) / sizeof(e->use[0])) {
        fprintf(stderr, "ds4: Metal 4 more than %zu resources declared for one dispatch\n",
                sizeof(e->use) / sizeof(e->use[0]));
        abort();
    }
    e->use[e->n_use] = resource;
    e->use_usage[e->n_use] = usage;
    e->n_use++;
}

/* On Metal 4 the engine's barriers order nothing: the graph does, from what
 * each dispatch reads and writes. They are counted, not encoded. */
static void ds4_m4_engine_barrier(DS4Enc *e) {
    (void)e;
    g_m4s[DS4_M4S_ENGINE_BARRIER]++;
}

API_AVAILABLE(macos(26.0))
static void ds4_m4_add_range(DS4M4Ctx *c, const void *buf, uint64_t lo, uint64_t hi, int write) {
    if (!ds4_m4_grow((void **)&c->ranges, &c->cap_ranges, c->n_ranges + 1u, sizeof(ds4_m4_range))) {
        fprintf(stderr, "ds4: Metal 4 range pool allocation failed\n");
        abort();
    }
    ds4_m4_range *r = &c->ranges[c->n_ranges++];
    r->buf = buf;
    r->lo = lo;
    r->hi = hi;
    r->write = (uint8_t)write;
    r->gen = ds4_m4_gen(buf, write);
}

API_AVAILABLE(macos(26.0))
static ds4_m4_node *ds4_m4_new_node(DS4M4Ctx *c, int kind) {
    if (!ds4_m4_grow((void **)&c->nodes, &c->cap_nodes, c->n_nodes + 1u, sizeof(ds4_m4_node))) {
        fprintf(stderr, "ds4: Metal 4 node pool allocation failed\n");
        abort();
    }
    ds4_m4_node *n = &c->nodes[c->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->kind = (uint8_t)kind;
    if (g_timeline_enabled) {
        if (!ds4_m4_grow((void **)&c->tl, &c->tl_cap, c->n_nodes, sizeof(ds4_timeline_rec))) {
            fprintf(stderr, "ds4: Metal 4 timeline allocation failed\n");
            abort();
        }
        memset(&c->tl[c->n_nodes - 1u], 0, sizeof(ds4_timeline_rec));
    }
    n->range_at = c->n_ranges;
    n->fence = c->fence_next;
    c->fence_next = 0;
    c->used = YES;
    g_m4s[DS4_M4S_NODES]++;
    return n;
}

/* A write to a buffer the queue holds resident: whatever reads it may do so
 * by raw address, unbound, so the node waits for everything before it and
 * everything after it waits for the node. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_fence_writes(DS4M4Ctx *c, ds4_m4_node *n) {
    for (uint32_t k = 0; k < n->n_range; k++) {
        const ds4_m4_range *r = &c->ranges[n->range_at + k];
        if (!r->write || !ds4_m4_queue_resident((__bridge id<MTLBuffer>)r->buf)) continue;
        n->fence = 1;
        c->fence_next = 1;
        g_m4s[DS4_M4S_FENCES]++;
        return;
    }
}

/* What the timeline writes for a node: its kernel, its grid, and the code
 * that encoded it. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_timeline_note(DS4M4Ctx *c, id pso, MTLSize a, MTLSize b, int threads,
                                 uintptr_t caller) {
    ds4_timeline_rec *r = &c->tl[c->n_nodes - 1u];
    r->caller = caller;
    r->n_dispatch = 1;
    if (threads) {
        r->tg[0] = (uint32_t)((a.width + b.width - 1) / (b.width ? b.width : 1));
        r->tg[1] = (uint32_t)((a.height + b.height - 1) / (b.height ? b.height : 1));
        r->tg[2] = (uint32_t)((a.depth + b.depth - 1) / (b.depth ? b.depth : 1));
    } else {
        r->tg[0] = (uint32_t)a.width; r->tg[1] = (uint32_t)a.height; r->tg[2] = (uint32_t)a.depth;
    }
    r->tpt[0] = (uint32_t)b.width; r->tpt[1] = (uint32_t)b.height; r->tpt[2] = (uint32_t)b.depth;
    NSString *name = nil;
    if (pso) {
        pthread_mutex_lock(&g_timeline_mutex);
        if (g_timeline_pso_names) name = g_timeline_pso_names[@((uintptr_t)(__bridge void *)pso)];
        pthread_mutex_unlock(&g_timeline_mutex);
    }
    snprintf(r->kernel, sizeof(r->kernel), "%s", pso ? (name ? name.UTF8String : "?") : "copy");
}

static void ds4_m4_dispatch(DS4Enc *e, int kind, MTLSize a, MTLSize b,
                            id<MTLBuffer> indirect, NSUInteger indirect_offset) {
    if (@available(macOS 26.0, *)) {
        DS4M4Ctx *c = e->m4;
        if (!e->pso) {
            fprintf(stderr, "ds4: Metal 4 dispatch with no pipeline\n");
            g_mtl4_encode_error = 1;
            return;
        }
        ds4_m4_node *n = ds4_m4_new_node(c, kind);
        const void *pso = (__bridge const void *)e->pso;
        n->pso = pso;
        if (pso != c->last_pso) { [c->psos addObject:e->pso]; c->last_pso = pso; }
        n->a = a;
        n->b = b;
        n->tgmem_set = e->tgmem_set;
        memcpy(n->tgmem, e->tgmem, sizeof(n->tgmem));
        const uint32_t nbind = e->bound ? 32u - (uint32_t)__builtin_clz(e->bound) : 0u;
        n->nbind = (uint8_t)nbind;
        if (!ds4_m4_grow((void **)&c->addrs, &c->cap_addrs, c->n_addrs + nbind, sizeof(uint64_t))) {
            fprintf(stderr, "ds4: Metal 4 binding pool allocation failed\n");
            abort();
        }
        n->bind_at = c->n_addrs;
        for (uint32_t i = 0; i < nbind; i++)
            c->addrs[c->n_addrs++] = (e->bound & (1u << i)) ? e->addr[i] : 0;
        uint32_t rd = ~0u, wr = ~0u;
        if (!ds4_pso_access(pso, &rd, &wr)) g_pso_access_unknown++;
        for (uint32_t m = e->bound & ~e->constant & (rd | wr); m; m &= m - 1u) {
            const uint32_t i = (uint32_t)__builtin_ctz(m);
            id<MTLBuffer> bb = e->buf[i];
            if (ds4_m4_readonly(bb)) continue;
            const void *buf = (__bridge const void *)bb;
            uint64_t hi = bb.length;
            NSUInteger len = e->len[i];
            /* No length from the call site: the tensor that starts there. */
            if (!len) len = ds4_extent_at(buf, e->off[i]);
            if (len) {
                hi = e->off[i] + len;
            } else {
                g_m4s[DS4_M4S_COARSE]++;
                if (e->line[i] < DS4_M4_CENSUS_LINES) g_m4_coarse_line[e->line[i]]++;
            }
            if (rd & (1u << i)) ds4_m4_add_range(c, buf, e->off[i], hi, 0);
            if (wr & (1u << i)) ds4_m4_add_range(c, buf, e->off[i], hi, 1);
        }
        for (uint32_t u = 0; u < e->n_use; u++) {
            id<MTLBuffer> ub = (id<MTLBuffer>)e->use[u];
            if (ds4_m4_readonly(ub)) continue;
            const void *buf = (__bridge const void *)ub;
            if (e->use_usage[u] & MTLResourceUsageRead) ds4_m4_add_range(c, buf, 0, ub.length, 0);
            if (e->use_usage[u] & MTLResourceUsageWrite) ds4_m4_add_range(c, buf, 0, ub.length, 1);
        }
        e->n_use = 0;
        if (kind == DS4_M4_NODE_INDIRECT) {
            ds4_mtl4_mark_resident(c, indirect);
            n->indirect = indirect.gpuAddress + indirect_offset;
            ds4_m4_add_range(c, (__bridge const void *)indirect,
                             indirect_offset, indirect_offset + 3u * sizeof(uint32_t), 0);
        }
        n->n_range = c->n_ranges - n->range_at;
        ds4_m4_fence_writes(c, n);
        if (g_timeline_enabled)
            ds4_m4_timeline_note(c, e->pso, a, b, kind == DS4_M4_NODE_THREADS,
                                 (uintptr_t)__builtin_return_address(0));
        g_mtl4_dispatches++;
        g_m4s[DS4_M4S_DISPATCH]++;
    }
}

/* The engine ended a section. A fresh Metal 3 encoder starts with nothing
 * bound, and the engine relies on nothing carrying over; the order between
 * sections is the graph's. */
static void ds4_m4_end(DS4Enc *e) {
    e->ended = YES;
    e->bound = e->constant = 0;
    e->tgmem_set = 0;
    e->n_use = 0;
    e->pso = nil;
    g_m4s[DS4_M4S_SECTIONS]++;
}

/* A buffer copy, as a node like a dispatch. */
API_AVAILABLE(macos(26.0))
static int ds4_mtl4_copy(id<MTLBuffer> src, NSUInteger soff,
                         id<MTLBuffer> dst, NSUInteger doff, NSUInteger size) {
    DS4M4Ctx *c = g_mtl4_cur;
    if (!c) return 0;
    if (!src || !dst) {
        g_mtl4_encode_error = 1;
        return 0;
    }
    ds4_mtl4_mark_resident(c, src);
    ds4_mtl4_mark_resident(c, dst);
    ds4_m4_node *n = ds4_m4_new_node(c, DS4_M4_NODE_COPY);
    n->src = (__bridge const void *)src;
    n->dst = (__bridge const void *)dst;
    n->soff = soff;
    n->doff = doff;
    n->size = size;
    if (!ds4_m4_readonly(src)) ds4_m4_add_range(c, n->src, soff, soff + size, 0);
    ds4_m4_add_range(c, n->dst, doff, doff + size, 1);
    n->n_range = c->n_ranges - n->range_at;
    ds4_m4_fence_writes(c, n);
    if (g_timeline_enabled)
        ds4_m4_timeline_note(c, nil, MTLSizeMake(0, 0, 0), MTLSizeMake(0, 0, 0), 0,
                             (uintptr_t)__builtin_return_address(0));
    g_m4s[DS4_M4S_COPY]++;
    return 1;
}

/* ---- barriers from the graph -------------------------------------------- */

/* Marks every node that has to wait for something recorded before it since
 * the last barrier: it reads what that wrote, or writes what that read or
 * wrote. A fenced node waits for everything, and so does the node after it.
 * Returns the number of barriers. */
API_AVAILABLE(macos(26.0))
static uint32_t ds4_m4_place_barriers(DS4M4Ctx *c) {
    const ds4_m4_range **pend = NULL;
    uint32_t np = 0, cap = 0, bars = 0;
    for (uint32_t i = 0; i < c->n_nodes; i++) {
        ds4_m4_node *n = &c->nodes[i];
        int conflict = n->fence;
        for (uint32_t k = 0; k < n->n_range && !conflict; k++) {
            const ds4_m4_range *x = &c->ranges[n->range_at + k];
            for (uint32_t j = 0; j < np; j++) {
                const ds4_m4_range *y = pend[j];
                if ((x->write || y->write) && y->buf == x->buf && y->lo < x->hi && x->lo < y->hi) {
                    conflict = 1;
                    break;
                }
            }
        }
        n->barrier = (uint8_t)(conflict && i > 0);
        if (n->barrier) bars++;
        if (conflict) np = 0;
        if (np + n->n_range > cap) {
            cap = (np + n->n_range) * 2u + 64u;
            pend = realloc(pend, cap * sizeof(*pend));
            if (!pend) { fprintf(stderr, "ds4: Metal 4 pending set allocation failed\n"); abort(); }
        }
        for (uint32_t k = 0; k < n->n_range; k++) pend[np++] = &c->ranges[n->range_at + k];
    }
    free(pend);
    return bars;
}

/* ---- encoding ------------------------------------------------------------- */

API_AVAILABLE(macos(26.0))
static void ds4_m4_encode_part(DS4M4Ctx *c, unsigned part, uint32_t from, uint32_t to) {
    id<MTL4CommandBuffer> cb = c->cb[part];
    id<MTL4ArgumentTable> table = c->table[part];
    [c->alloc[part] reset];
    [cb beginCommandBufferWithAllocator:c->alloc[part]];
    [cb useResidencySet:c->resset];
    id<MTL4ComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setArgumentTable:table];
    /* Nothing bound, as a fresh Metal 3 encoder has. */
    uint64_t cur[DS4_MTL4_MAX_BINDINGS];
    for (uint32_t k = 0; k < DS4_MTL4_MAX_BINDINGS; k++) {
        [table setAddress:0 atIndex:k];
        cur[k] = 0;
    }
    const void *cur_pso = NULL;
    const MTL4VisibilityOptions vis = ds4_mtl4_visibility();
    /* Ordered after everything the queue holds ahead of it: the contexts
     * committed before this one, and the part before this one. */
    [enc barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageAll visibilityOptions:vis];
    for (uint32_t i = from; i < to; i++) {
        const ds4_m4_node *n = &c->nodes[i];
        /* Every stage on both sides: what this encoder holds is not only
         * dispatches - a copy is blit-stage work, and an indirect dispatch
         * fetches its grid as arguments. */
        if (n->barrier && i != from)
            [enc barrierAfterEncoderStages:MTLStageAll beforeEncoderStages:MTLStageAll
                         visibilityOptions:vis];
        if (c->tl_heap)
            [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                      intoHeap:c->tl_heap atIndex:2u * i];
        if (n->kind == DS4_M4_NODE_COPY) {
            [enc copyFromBuffer:(__bridge id<MTLBuffer>)n->src sourceOffset:n->soff
                       toBuffer:(__bridge id<MTLBuffer>)n->dst destinationOffset:n->doff
                           size:n->size];
            if (c->tl_heap)
                [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                          intoHeap:c->tl_heap atIndex:2u * i + 1u];
            continue;
        }
        if (n->pso != cur_pso) {
            [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)n->pso];
            cur_pso = n->pso;
        }
        const uint64_t *addr = &c->addrs[n->bind_at];
        for (uint32_t k = 0; k < DS4_MTL4_MAX_BINDINGS; k++) {
            const uint64_t v = k < n->nbind ? addr[k] : 0;
            if (v != cur[k]) { [table setAddress:v atIndex:k]; cur[k] = v; }
        }
        for (uint32_t k = 0; k < 4; k++)
            if (n->tgmem_set & (1u << k)) [enc setThreadgroupMemoryLength:n->tgmem[k] atIndex:k];
        switch (n->kind) {
        case DS4_M4_NODE_GROUPS:
            [enc dispatchThreadgroups:n->a threadsPerThreadgroup:n->b];
            break;
        case DS4_M4_NODE_THREADS:
            [enc dispatchThreads:n->a threadsPerThreadgroup:n->b];
            break;
        case DS4_M4_NODE_INDIRECT:
            [enc dispatchThreadgroupsWithIndirectBuffer:n->indirect threadsPerThreadgroup:n->b];
            break;
        }
        if (c->tl_heap)
            [enc writeTimestampWithGranularity:MTL4TimestampGranularityPrecise
                                      intoHeap:c->tl_heap atIndex:2u * i + 1u];
    }
    /* What this part wrote is visible to what follows it on the queue, and to
     * the host once the completion is seen. */
    [enc barrierAfterStages:MTLStageAll beforeQueueStages:MTLStageAll visibilityOptions:vis];
    [enc endEncoding];
    [cb endCommandBuffer];
}

/* How many parts a graph is cut into: none below DS4_MTL4_PART_MIN nodes,
 * then a part per DS4_MTL4_PART_MIN up to DS4_MTL4_PARTS, cut only where the
 * graph has a barrier. ds4_gpu_test_mtl4_parts forces a count regardless of
 * size, so a test can put every context through the split. */
/* Held at one part while the kernel panic seen with the full expert address
 * table is isolated: the binary that took the machine down had both. */
static unsigned g_mtl4_parts_max = 1u;
static unsigned g_mtl4_parts_forced;
void ds4_gpu_test_mtl4_parts(unsigned parts);
void ds4_gpu_test_mtl4_parts(unsigned parts) {
    g_mtl4_parts_forced = parts > DS4_MTL4_PARTS ? DS4_MTL4_PARTS : parts;
}

/* Where the graph is cut. Returns the number of parts; bounds[p] is where
 * part p starts, bounds[parts] the end. */
API_AVAILABLE(macos(26.0))
static unsigned ds4_m4_split(DS4M4Ctx *c, uint32_t bounds[DS4_MTL4_PARTS + 1u]) {
    unsigned want = c->n_nodes / DS4_MTL4_PART_MIN;
    if (want > g_mtl4_parts_max) want = g_mtl4_parts_max;
    if (g_mtl4_parts_forced) want = g_mtl4_parts_forced;
    if (want < 1u) want = 1u;
    bounds[0] = 0;
    unsigned parts = 1;
    for (unsigned p = 1; p < want; p++) {
        uint32_t at = (uint32_t)((uint64_t)c->n_nodes * p / want);
        while (at < c->n_nodes && !c->nodes[at].barrier) at++;
        if (at >= c->n_nodes || at <= bounds[parts - 1]) break;
        bounds[parts++] = at;
    }
    bounds[parts] = c->n_nodes;
    return parts;
}

/* ---- completion ----------------------------------------------------------- */

/* The commit feedback, on the context that carried the stamp. Without it a
 * submission the GPU failed - out of memory, a fault - completes like any
 * other and the host reads what was left there. The event says a submission
 * finished, not that it succeeded, so a stamp counts as done only once its
 * own feedback has arrived and said so; feedback that has not arrived is a
 * failure, never a success. */
/* How far ahead of the last wait stamps may run: a wait covers at most this
 * many submissions. */
#define DS4_MTL4_FB_RING 256u
/* Where the GPU spans already counted end. */
static double g_mtl4_span_last_end;
/* The expert cache's logical batch sequence each stamp carried. */
static uint64_t g_mtl4_stamp_seq[DS4_MTL4_FB_RING];
static _Atomic unsigned long long g_mtl4_fb_received, g_mtl4_fb_errors;

/* Test hook: the next feedback to arrive reports a failure. The GPU cannot be
 * made to fail on demand, so this is where the host path after a real failure
 * is exercised - the wait, end_commands, the executor, the request. */
static _Atomic int g_mtl4_inject;
void ds4_gpu_mtl4_inject_failure(void) { atomic_store(&g_mtl4_inject, 1); }

API_AVAILABLE(macos(26.0))
static void ds4_mtl4_fb_note(DS4M4Ctx *c, uint64_t stamp, NSError *error,
                             double gpu_start, double gpu_end) {
    c->gpu_start = gpu_start;
    c->gpu_end = gpu_end;
    if (!error && atomic_exchange(&g_mtl4_inject, 0))
        error = [NSError errorWithDomain:@"ds4.injected" code:1
                                userInfo:@{NSLocalizedDescriptionKey: @"injected failure"}];
    if (error) {
        fprintf(stderr, "ds4: Metal 4 submission %llu failed: %s\n",
                (unsigned long long)stamp, error.description.UTF8String);
        atomic_fetch_add(&g_mtl4_fb_errors, 1u);
    }
    atomic_store(&c->fb_failed, error ? 1 : 0);
    atomic_store(&c->fb_stamp, stamp);       /* last: publishes the rest */
    atomic_fetch_add(&g_mtl4_fb_received, 1u);
}

/* A completed context lets go of what it held: emptied and committed now, not
 * when the context is next reused, so a freed session is not held resident by
 * a context that happens to sit idle in the pool. */
static void ds4_mtl4_release_completed(void) API_AVAILABLE(macos(26.0));
static void ds4_mtl4_release_completed(void) {
    for (DS4M4Ctx *c in g_mtl4_pool) {
        if (c == g_mtl4_cur || c->stamp == 0 || c->stamp > g_mtl4_done) continue;
        if (c->resident.count) {
            g_mtl4_res_releases += c->resident.count;
            [c->resset removeAllAllocations];
            [c->resset commit];
            [c->resident removeAllObjects];
            c->res_bytes = 0;
        }
        [c->psos removeAllObjects];
    }
}

/* A completed context's timeline, in the Metal 3 timeline's format: a "B" line
 * for the submission and an "E" line a node, times in ns. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_timeline_resolve(DS4M4Ctx *c) {
    if (!g_timeline_file || !c->n_nodes) return;
    NSData *d = [c->tl_heap resolveCounterRange:NSMakeRange(0, 2u * c->n_nodes)];
    if (!d || d.length < 2u * c->n_nodes * sizeof(MTL4TimestampHeapEntry)) return;
    const MTL4TimestampHeapEntry *ts = (const MTL4TimestampHeapEntry *)d.bytes;
    static double ns_per_tick;
    if (ns_per_tick == 0.0) {
        const uint64_t hz = [g_device queryTimestampFrequency];
        ns_per_tick = hz ? 1e9 / (double)hz : 1.0;
    }
    pthread_mutex_lock(&g_timeline_mutex);
    fprintf(g_timeline_file, "B %llu %u %.0f %.0f\n", (unsigned long long)c->tl_seq, c->n_nodes,
            c->gpu_start * 1e9, c->gpu_end * 1e9);
    uint64_t prev_end = 0;
    for (uint32_t i = 0; i < c->n_nodes; i++) {
        const uint64_t start = (uint64_t)((double)ts[2u * i].timestamp * ns_per_tick);
        const uint64_t end = (uint64_t)((double)ts[2u * i + 1u].timestamp * ns_per_tick);
        const ds4_timeline_rec *r = &c->tl[i];
        const double dur_us = end >= start ? (double)(end - start) / 1000.0 : -1.0;
        const double gap_us = prev_end && start >= prev_end ? (double)(start - prev_end) / 1000.0 : 0.0;
        fprintf(g_timeline_file, "E %llu %u %llu %llu %.2f %.2f 0x%llx %u %ux%ux%u %ux%ux%u %s\n",
                (unsigned long long)c->tl_seq, i, (unsigned long long)start,
                (unsigned long long)end, dur_us, gap_us,
                (unsigned long long)(r->caller - (uintptr_t)_dyld_get_image_vmaddr_slide(0)),
                r->n_dispatch, r->tg[0], r->tg[1], r->tg[2], r->tpt[0], r->tpt[1], r->tpt[2],
                r->kernel[0] ? r->kernel : "?");
        if (end) prev_end = end;
    }
    fflush(g_timeline_file);
    pthread_mutex_unlock(&g_timeline_mutex);
}

/* Wait until every submission up to `value` has completed and reported.
 * Returns 0 if any of them failed, or if a report did not arrive; in the second
 * case nothing is marked done, so a later wait looks again. */
static int ds4_mtl4_wait_to(uint64_t value) {
    if (value <= g_mtl4_done) {
        if (!g_mtl4_encode_error) return 1;
        fprintf(stderr, "ds4: Metal 4 batch was short of commands it was given; failing it\n");
        g_mtl4_encode_error = 0;
        return 0;
    }
    if (value - g_mtl4_done > DS4_MTL4_FB_RING) {
        fprintf(stderr, "ds4: Metal 4 wait spans %llu submissions, past the feedback ring\n",
                (unsigned long long)(value - g_mtl4_done));
        return 0;
    }
    DS4_M4_NOTE("wait", DS4_BK_WAIT);
    const uint64_t w0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    const BOOL ok = [g_mtl4_event waitUntilSignaledValue:value timeoutMS:600000];
    g_m4s[DS4_M4S_WAIT_CALLS]++;
    g_m4s[DS4_M4S_WAIT_NS] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - w0;
    if (!ok) {
        fprintf(stderr, "ds4: Metal 4 submission timed out\n");
        return 0;
    }
    int good = 1;
    for (uint64_t st = g_mtl4_done + 1u; st <= value; st++) {
        /* The context that carried the stamp: it is not reused before its
         * stamp has been waited for, so it is still in the pool. */
        DS4M4Ctx *slot = nil;
        if (@available(macOS 26.0, *))
            for (DS4M4Ctx *c in g_mtl4_pool) if (c->stamp == st) { slot = c; break; }
        if (!slot) {
            fprintf(stderr, "ds4: Metal 4 stamp %llu has no context\n", (unsigned long long)st);
            return 0;
        }
        /* The feedback is delivered on its own queue; the event can be seen
         * first. It follows within microseconds, and a usleep(10) here slept
         * tens of them per segment - about seventeen times a token - so the
         * first few milliseconds are spun rather than slept. */
        if (atomic_load(&slot->fb_stamp) != st) {
            const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            uint64_t t = t0;
            while (atomic_load(&slot->fb_stamp) != st && t - t0 < 5000000ull) {
                __builtin_arm_yield();
                t = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            }
            for (int nap = 0; atomic_load(&slot->fb_stamp) != st && nap < 500000; nap++)
                usleep(10);
            g_mtl4_fb_late++;
            g_mtl4_fb_late_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        }
        if (atomic_load(&slot->fb_stamp) != st) {
            fprintf(stderr, "ds4: Metal 4 submission %llu completed without feedback\n",
                    (unsigned long long)st);
            return 0;
        }
        if (atomic_load(&slot->fb_failed)) good = 0;
        if (g_timeline_enabled && slot->tl_heap) ds4_m4_timeline_resolve(slot);
        /* The union of the submissions' spans, not their sum: with the next
         * one committed before the current one is read, their spans overlap.
         * Stamps are waited in order. */
        if (slot->gpu_end > g_mtl4_span_last_end) {
            const double from = slot->gpu_start > g_mtl4_span_last_end ? slot->gpu_start
                                                                       : g_mtl4_span_last_end;
            g_gpu_span_accum_ms += (slot->gpu_end - from) * 1000.0;
            g_mtl4_span_last_end = slot->gpu_end;
        }
        g_gpu_span_cbs++;
    }
    g_mtl4_done = value;
    if (g_mtl4_encode_error) {
        fprintf(stderr, "ds4: Metal 4 batch was short of commands it was given; failing it\n");
        g_mtl4_encode_error = 0;
        good = 0;
    }
    if (@available(macOS 26.0, *)) ds4_mtl4_release_completed();
    return good;
}

/* Everything committed and not yet waited for. */
static int ds4_mtl4_wait_all(void) {
    if (@available(macOS 26.0, *)) return ds4_mtl4_wait_to(g_mtl4_stamp);
    return 1;
}

/* ---- the pool ---------------------------------------------------------- */

/* DS4_METAL_V41_MTL4=2 sends greedy decode through the verify executor. The
 * backend itself is not chosen here: V4.1 always runs on Metal 4. */
static int ds4_mtl4_level(void) {
    static int checked, level;
    if (!checked) {
        const char *e = getenv("DS4_METAL_V41_MTL4");
        checked = 1;
        level = e && e[0] ? atoi(e) : 0;
    }
    return level;
}

static int ds4_mtl4_on(void) { return g_backend == 4; }

API_AVAILABLE(macos(26.0))
static DS4M4Ctx *ds4_m4_ctx_new(void) {
    DS4M4Ctx *c = [DS4M4Ctx new];
    MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
    d.maxBufferBindCount = DS4_MTL4_MAX_BINDINGS;
    /* Unset slots of a table are otherwise undefined, which is the opposite
     * of a fresh Metal 3 encoder. */
    d.initializeBindings = YES;
    for (unsigned p = 0; p < DS4_MTL4_PARTS; p++) {
        c->alloc[p] = [g_device newCommandAllocator];
        c->cb[p] = [g_device newCommandBuffer];
        NSError *err = nil;
        c->table[p] = [g_device newArgumentTableWithDescriptor:d error:&err];
        if (!c->alloc[p] || !c->cb[p] || !c->table[p]) {
            fprintf(stderr, "ds4: Metal 4 context part %u unavailable: %s\n", p,
                    err ? err.description.UTF8String : "allocator or command buffer");
            return nil;
        }
    }
    c->scratch = [g_device newBufferWithLength:DS4_MTL4_SCRATCH_BYTES
                                       options:MTLResourceStorageModeShared];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.label = @"ds4_mtl4_context";
    NSError *rerr = nil;
    c->resset = [g_device newResidencySetWithDescriptor:rd error:&rerr];
    c->resident = [NSHashTable hashTableWithOptions:NSPointerFunctionsStrongMemory |
                                                    NSPointerFunctionsObjectPointerPersonality];
    c->psos = [NSHashTable hashTableWithOptions:NSPointerFunctionsStrongMemory |
                                                NSPointerFunctionsObjectPointerPersonality];
    if (!c->scratch || !c->resset) return nil;
    c->scratch.label = @"ds4_mtl4_scratch";
    c.cmd = [DS4Cmd new];
    c.cmd->m4 = c;
    c.enc = [DS4Enc new];
    c.enc->m4 = c;
    return c;
}

static int ds4_mtl4_ready(void) API_AVAILABLE(macos(26.0));
static int ds4_mtl4_ready(void) {
    if (![g_device supportsFamily:MTLGPUFamilyMetal4]) return 0;
    if (g_mtl4_queue) return 1;
    g_mtl4_queue = [g_device newMTL4CommandQueue];
    g_mtl4_event = [g_device newSharedEvent];
    if (!g_mtl4_queue || !g_mtl4_event) {
        fprintf(stderr, "ds4: Metal 4 queue unavailable\n");
        g_mtl4_queue = nil;
        return 0;
    }
    g_mtl4_pool = [NSMutableArray array];
    for (unsigned i = 0; i < DS4_MTL4_PIPE; i++) {
        DS4M4Ctx *c = ds4_m4_ctx_new();
        if (!c) {
            fprintf(stderr, "ds4: Metal 4 command context unavailable\n");
            g_mtl4_queue = nil;
            g_mtl4_pool = nil;
            return 0;
        }
        [g_mtl4_pool addObject:c];
    }
    ds4_mtl4_attach_sets();
    return 1;
}

/* A context whose work has completed: its allocators may be reset and its
 * scratch reused. */
API_AVAILABLE(macos(26.0))
static DS4M4Ctx *ds4_mtl4_take(void) {
    for (DS4M4Ctx *c in g_mtl4_pool)
        if (c != g_mtl4_cur && c->stamp <= g_mtl4_done) return c;
    /* Every context is still out. Wait for everything: the pipeline is two
     * deep and the pool is four, so this is a guard rather than a path. */
    if (!ds4_mtl4_wait_all()) return nil;
    for (DS4M4Ctx *c in g_mtl4_pool)
        if (c != g_mtl4_cur && c->stamp <= g_mtl4_done) return c;
    return nil;
}

/* Back to empty: no nodes, no constants, nothing bound. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_ctx_clear(DS4M4Ctx *c) {
    c->scratch_used = 0;
    c->stamp = 0;
    c->used = NO;
    c->n_nodes = c->n_addrs = c->n_ranges = 0;
    c->last_pso = NULL;
    c->fence_next = 0;
    DS4Enc *e = c.enc;
    e->ended = NO;
    e->concurrent = NO;
    e->bound = e->constant = 0;
    e->tgmem_set = 0;
    e->n_use = 0;
    e->pso = nil;
}

static int ds4_mtl4_begin(void) {
    if (@available(macOS 26.0, *)) {
        if (!ds4_mtl4_on()) return 0;
        if (g_mtl4_cur) return 1;
        if (!ds4_mtl4_ready()) return 0;
        DS4M4Ctx *c = ds4_mtl4_take();
        if (!c) return 0;
        ds4_m4_ctx_clear(c);
        c->ledger = ds4_gpu_ledger_m4_begin();
        g_mtl4_cur = c;
        /* 0 is never a live generation, so a zeroed memo entry never hits. */
        if (++c->gen == 0) ++c->gen;
        /* Proof the path ran, not that it was asked for. Byte identity from a
         * backend that silently fell back is byte identity from the backend it
         * fell back to. */
        g_mtl4_segments++;
        return 1;
    }
    return 0;
}

/* Encode the open context's graph and send it, without waiting. */
static int ds4_mtl4_commit(void) {
    if (@available(macOS 26.0, *)) {
        DS4M4Ctx *c = g_mtl4_cur;
        if (!c) return 0;
        g_m4s[DS4_M4S_BAR_GRAPH] += ds4_m4_place_barriers(c);
        /* The constant scratch is reached only through the table, so it is
         * declared here; the set is complete once the graph is. */
        if (c->scratch_used) ds4_mtl4_mark_resident(c, c->scratch);
        [c->resset commit];
        {
            unsigned long long held = 0;
            for (DS4M4Ctx *p in g_mtl4_pool) held += p->res_bytes;
            if (held > g_mtl4_res_peak) g_mtl4_res_peak = held;
        }
        if (g_timeline_enabled && c->n_nodes) {
            if (!c->tl_heap || c->tl_heap_cap < 2u * c->n_nodes) {
                MTL4CounterHeapDescriptor *hd = [MTL4CounterHeapDescriptor new];
                hd.type = MTL4CounterHeapTypeTimestamp;
                hd.count = 2u * c->n_nodes;
                NSError *herr = nil;
                c->tl_heap = [g_device newCounterHeapWithDescriptor:hd error:&herr];
                c->tl_heap_cap = c->tl_heap ? hd.count : 0;
                if (!c->tl_heap)
                    fprintf(stderr, "ds4: Metal 4 timeline heap: %s\n", herr.description.UTF8String);
            }
            c->tl_seq = ++g_timeline_seq;
        } else if (c->tl_heap) {
            c->tl_heap = nil;
            c->tl_heap_cap = 0;
        }
        uint32_t bounds[DS4_MTL4_PARTS + 1u];
        const unsigned parts = ds4_m4_split(c, bounds);
        if (parts == 1) {
            ds4_m4_encode_part(c, 0, 0, c->n_nodes);
        } else {
            const uint32_t *bp = bounds;
            dispatch_apply(parts, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                           ^(size_t p) {
                ds4_m4_encode_part(c, (unsigned)p, bp[p], bp[p + 1]);
            });
            g_m4s[DS4_M4S_SPLIT]++;
        }
        c->parts = parts;
        g_m4s[DS4_M4S_BAR_QUEUE] += 2u * parts;
        if (ds4_gpu_ledger_active()) {
            const void **psos = calloc(c->n_nodes ? c->n_nodes : 1, sizeof(*psos));
            uint32_t copies = 0;
            uint64_t copy_bytes = 0;
            for (uint32_t i = 0; i < c->n_nodes; i++) {
                psos[i] = c->nodes[i].pso;
                if (c->nodes[i].kind == DS4_M4_NODE_COPY) { copies++; copy_bytes += c->nodes[i].size; }
            }
            ds4_gpu_ledger_m4_commit(&c->ledger, psos, c->n_nodes, copies, copy_bytes);
            free(psos);
        }
        const uint64_t stamp = g_mtl4_stamp + 1u;
        const ds4_ledger_m4 ledger = c->ledger;
        MTL4CommitOptions *opts = [MTL4CommitOptions new];
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
            ds4_gpu_ledger_m4_done(&ledger, fb.GPUStartTime, fb.GPUEndTime);
            ds4_mtl4_fb_note(c, stamp, fb.error, fb.GPUStartTime, fb.GPUEndTime);
        }];
        id<MTL4CommandBuffer> list[DS4_MTL4_PARTS];
        for (unsigned p = 0; p < parts; p++) list[p] = c->cb[p];
        DS4_M4_NOTE("segment", DS4_BK_COMMIT);
        [g_mtl4_queue commit:list count:parts options:opts];
        c->stamp = ++g_mtl4_stamp;
        g_mtl4_commits++;
        g_m4s[DS4_M4S_CONTEXTS]++;
        g_m4s[DS4_M4S_BUFFERS] += parts;
        [g_mtl4_queue signalEvent:g_mtl4_event value:c->stamp];
        g_mtl4_cur = nil;
        return 1;
    }
    return 0;
}

/* Throw away what is recorded and not committed. The engine drops a prepared
 * segment when the one before it aborted - about thirty-two times a token at
 * forty gated layers. Nothing of it was encoded yet, so there is no command
 * buffer to end and nothing to release but the recording and its constants. */
static void ds4_mtl4_discard(void) {
    if (@available(macOS 26.0, *)) {
        DS4M4Ctx *c = g_mtl4_cur;
        if (!c) return;
        ds4_m4_ctx_clear(c);
    }
}

/* The open context's recorder. A section the engine ended starts over with
 * nothing bound (ds4_m4_end); the order between sections is the graph's. */
API_AVAILABLE(macos(26.0))
static DS4Enc *ds4_mtl4_encoder(void) {
    DS4M4Ctx *c = g_mtl4_cur;
    if (!c) return nil;
    DS4Enc *e = c.enc;
    e->ended = NO;
    e->concurrent = g_batch_encoder_concurrent;
    return e;
}

/* ---- what the engine calls --------------------------------------------- */

/* Whether the Metal 4 backend is on for this process: the verify executor runs
 * only under it, so it asks rather than falling back to the Metal 3 path. */
int ds4_gpu_mtl4_enabled(void) {
    return ds4_mtl4_on() && ds4_mtl4_ready();
}
/* Production decode goes through the executor only when asked: at a commit
 * depth of 1.1 a block it computes several nodes to advance about one token. */
int ds4_gpu_mtl4_executor_enabled(void) { return ds4_mtl4_level() >= 2 && ds4_mtl4_ready(); }

/* commits, feedback received, feedback errors, residency entries and bytes held
 * by buffers in flight, adds, releases, peak bytes held. */
void ds4_gpu_mtl4_health(unsigned long long out[8]) {
    out[0] = g_mtl4_commits;
    out[1] = atomic_load(&g_mtl4_fb_received);
    out[2] = atomic_load(&g_mtl4_fb_errors);
    unsigned long long entries = 0, bytes = 0;
    if (@available(macOS 26.0, *))
        for (DS4M4Ctx *c in g_mtl4_pool) { entries += c->resident.count; bytes += c->res_bytes; }
    out[3] = entries;
    out[4] = bytes;
    out[5] = g_mtl4_res_adds;
    out[6] = g_mtl4_res_releases;
    out[7] = g_mtl4_res_peak;
}

/* The end of the gated region waits for what the region committed and leaves
 * the open batch open: the token goes on encoding into it. */
int ds4_gpu_mtl4_region_end(void) {
    if (!ds4_mtl4_on()) return 1;
    return ds4_mtl4_wait_all();
}

/* ---- the batch, for ds4_metal.m's begin/flush/end ---------------------- */

/* The engine's handle for the open context: what g_batch_cmd holds under
 * this backend. */
static DS4Cmd *ds4_mtl4_batch_cmd(void) {
    if (@available(macOS 26.0, *)) return g_mtl4_cur ? g_mtl4_cur.cmd : nil;
    return nil;
}

/* Whether the open buffer has anything in it. */
static int ds4_mtl4_cur_used(void) {
    if (@available(macOS 26.0, *)) return g_mtl4_cur && g_mtl4_cur->used;
    return 0;
}

/* Send the open buffer if it has work, and open the next one. Returns 0 if a
 * buffer could not be had; the commit itself reports through its feedback. */
static int ds4_mtl4_rotate(void) {
    if (@available(macOS 26.0, *)) {
        if (ds4_mtl4_cur_used() && !ds4_mtl4_commit()) return 0;
        return ds4_mtl4_begin();
    }
    return 0;
}

/* The stamp everything encoded so far will have completed by, once the open
 * buffer is sent: what a "signal here" means on a queue whose buffers are
 * the unit of completion. */
static uint64_t ds4_mtl4_last_stamp(void) { return g_mtl4_stamp; }

/* ---- small submissions from other threads ------------------------------
 *
 * The queue keepalive and the GPU pacer each run one small kernel from their
 * own thread, over and over. Each gets its own allocator, buffer, table,
 * residency and event, so nothing it does touches the engine's pool or
 * stamps; the allocator is reset only once the previous kernel has run. */
API_AVAILABLE(macos(26.0))
@interface DS4MTL4Side : NSObject
@property (nonatomic, strong) id<MTL4CommandQueue> queue;
@property (nonatomic, strong) id<MTL4CommandAllocator> allocator;
@property (nonatomic, strong) id<MTL4CommandBuffer> cb;
@property (nonatomic, strong) id<MTL4ArgumentTable> table;
@property (nonatomic, strong) id<MTLResidencySet> res;
@property (nonatomic, strong) id<MTLSharedEvent> event;
@property (nonatomic) uint64_t sent;
@end
@implementation DS4MTL4Side
@end

API_AVAILABLE(macos(26.0))
static DS4MTL4Side *ds4_mtl4_side_new(id<MTL4CommandQueue> queue, id<MTLBuffer> b0, id<MTLBuffer> b1,
                                      NSString *label) {
    if (!queue || !b0 || !b1) return nil;
    DS4MTL4Side *s = [DS4MTL4Side new];
    s.queue = queue;
    s.allocator = [g_device newCommandAllocator];
    s.cb = [g_device newCommandBuffer];
    MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
    d.maxBufferBindCount = 2;
    d.initializeBindings = YES;
    s.table = [g_device newArgumentTableWithDescriptor:d error:nil];
    MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
    rd.label = label;
    s.res = [g_device newResidencySetWithDescriptor:rd error:nil];
    s.event = [g_device newSharedEvent];
    if (!s.allocator || !s.cb || !s.table || !s.res || !s.event) return nil;
    [s.res addAllocation:b0];
    [s.res addAllocation:b1];
    [s.res commit];
    return s;
}

/* One dispatch of `groups` threadgroups; with `wait`, until it has run. */
API_AVAILABLE(macos(26.0))
static int ds4_mtl4_side_run(DS4MTL4Side *s, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> b0, id<MTLBuffer> b1, NSUInteger groups, int wait) {
    if (!s || !pipeline) return 0;
    if (s.sent && ![s.event waitUntilSignaledValue:s.sent timeoutMS:10000]) return 0;
    [s.allocator reset];
    [s.cb beginCommandBufferWithAllocator:s.allocator];
    [s.cb useResidencySet:s.res];
    id<MTL4ComputeCommandEncoder> e = [s.cb computeCommandEncoder];
    [e setArgumentTable:s.table];
    [e setComputePipelineState:pipeline];
    [s.table setAddress:b0.gpuAddress atIndex:0];
    [s.table setAddress:b1.gpuAddress atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [s.cb endCommandBuffer];
    id<MTL4CommandBuffer> one = s.cb;
    [s.queue commit:&one count:1];
    s.sent++;
    [s.queue signalEvent:s.event value:s.sent];
    if (wait && ![s.event waitUntilSignaledValue:s.sent timeoutMS:10000]) return 0;
    return 1;
}

/* ---- the queue keepalive ----------------------------------------------- */

/* The keepalive exists because a queue left idle for about three seconds
 * loses the process's GPU mappings and the next buffer on it starts 600-800
 * ms late, and work on other queues does not prevent it. With this backend
 * selected the queue doing the work is this one, so the keepalive's kernel
 * goes here. Returns 0 if it could not submit. */
static DS4MTL4Side *g_mtl4_keepalive API_AVAILABLE(macos(26.0));

int ds4_mtl4_keepalive_once(id<MTLComputePipelineState> pipeline, id<MTLBuffer> scratch,
                            id<MTLBuffer> args) API_AVAILABLE(macos(26.0));
int ds4_mtl4_keepalive_once(id<MTLComputePipelineState> pipeline, id<MTLBuffer> scratch,
                            id<MTLBuffer> args) {
    if (!g_mtl4_queue) return 0;
    if (!g_mtl4_keepalive)
        g_mtl4_keepalive = ds4_mtl4_side_new(g_mtl4_queue, scratch, args, @"ds4_mtl4_keepalive");
    return ds4_mtl4_side_run(g_mtl4_keepalive, pipeline, scratch, args, 1, 0);
}

/* The GPU pacer on this backend: its own Metal 4 queue, as the Metal 3 pacer
 * has its own queue, so its kernels sit beside the engine's rather than in
 * its order. Created by the pacer thread and used only there. */
void *ds4_mtl4_pacer_new(id<MTLBuffer> scratch, id<MTLBuffer> args) API_AVAILABLE(macos(26.0));
void *ds4_mtl4_pacer_new(id<MTLBuffer> scratch, id<MTLBuffer> args) {
    id<MTL4CommandQueue> q = [g_device newMTL4CommandQueue];
    if (!q) return NULL;
    DS4MTL4Side *s = ds4_mtl4_side_new(q, scratch, args, @"ds4_mtl4_pacer");
    return s ? (__bridge_retained void *)s : NULL;
}
int ds4_mtl4_pacer_run(void *pacer, id<MTLComputePipelineState> pipeline, id<MTLBuffer> scratch,
                       id<MTLBuffer> args, NSUInteger groups) API_AVAILABLE(macos(26.0));
int ds4_mtl4_pacer_run(void *pacer, id<MTLComputePipelineState> pipeline, id<MTLBuffer> scratch,
                       id<MTLBuffer> args, NSUInteger groups) {
    return ds4_mtl4_side_run((__bridge DS4MTL4Side *)pacer, pipeline, scratch, args, groups, 1);
}
void ds4_mtl4_pacer_free(void *pacer) API_AVAILABLE(macos(26.0));
void ds4_mtl4_pacer_free(void *pacer) {
    if (pacer) { DS4MTL4Side *s = (__bridge_transfer DS4MTL4Side *)pacer; (void)s; }
}

/* ---- the full expert address table's prewire ---------------------------

/* Every expert buffer the table names copied once into a scratch buffer, so
 * its pages are wired before decode rather than during it. Its own command
 * buffer, committed behind whatever is in flight and ahead of the context
 * being recorded, and waited for. The copies all land in one scratch buffer;
 * what they leave there is never read. */
static int ds4_mtl4_prewire(NSArray<id<MTLBuffer>> *buffers, id<MTLBuffer> scratch) {
    if (@available(macOS 26.0, *)) {
        if (!g_mtl4_queue || !scratch) return 0;
        id<MTL4CommandAllocator> alloc = [g_device newCommandAllocator];
        id<MTL4CommandBuffer> cb = [g_device newCommandBuffer];
        id<MTLSharedEvent> done = [g_device newSharedEvent];
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.label = @"ds4_mtl4_prewire";
        id<MTLResidencySet> res = [g_device newResidencySetWithDescriptor:rd error:nil];
        if (!alloc || !cb || !done || !res) return 0;
        [res addAllocation:scratch];
        for (id<MTLBuffer> b in buffers) [res addAllocation:b];
        [res commit];
        [cb beginCommandBufferWithAllocator:alloc];
        [cb useResidencySet:res];
        id<MTL4ComputeCommandEncoder> e = [cb computeCommandEncoder];
        for (id<MTLBuffer> b in buffers) {
            if ([b length] == 0) continue;
            [e copyFromBuffer:b sourceOffset:0 toBuffer:scratch destinationOffset:0 size:[b length]];
        }
        [e endEncoding];
        [cb endCommandBuffer];
        id<MTL4CommandBuffer> one = cb;
        DS4_M4_NOTE("full_expert_addr_prewire", DS4_BK_COMMIT);
        [g_mtl4_queue commit:&one count:1];
        [g_mtl4_queue signalEvent:done value:1];
        DS4_M4_NOTE("full_expert_addr_prewire", DS4_BK_WAIT);
        return [done waitUntilSignaledValue:1 timeoutMS:600000] ? 1 : 0;
    }
    return 0;
}

/* ---- teardown ----------------------------------------------------------- */

/* What ds4_gpu_cleanup has to undo on this side, so an engine opened after it
 * starts from nothing: everything sent is waited for, every buffer lets go of
 * what it held, the queue lets go of the process sets, and the queue, table,
 * pool, event and the keepalive's objects are dropped - they belong to the
 * device being released. Stamps stay monotonic across engines, so a feedback
 * slot left from the last one can never read as the next one's. The keepalive
 * thread has been stopped before this runs. Returns 0 if a wait failed; the
 * teardown happens either way. */
static int ds4_mtl4_shutdown(void) {
    int ok = 1;
    if (@available(macOS 26.0, *)) {
        if (g_mtl4_cur) {
            if (g_mtl4_cur->used) { if (!ds4_mtl4_commit()) ok = 0; }
            else g_mtl4_cur = nil;
        }
        if (g_mtl4_queue && !ds4_mtl4_wait_all()) ok = 0;
        for (DS4M4Ctx *b in g_mtl4_pool) {
            [b->resset removeAllAllocations];
            [b->resset commit];
            [b->resident removeAllObjects];
            [b->psos removeAllObjects];
            b->res_bytes = 0;
        }
        if (g_mtl4_queue) {
            for (NSUInteger i = 0; i < g_mtl4_sets_attached_n; i++)
                [g_mtl4_queue removeResidencySet:g_mtl4_queue_sets[i]];
        }
        g_mtl4_sets_attached_n = 0;
        [g_mtl4_queue_sets removeAllObjects];
        g_mtl4_cur = nil;
        g_mtl4_pool = nil;
        g_mtl4_queue = nil;
        g_mtl4_event = nil;
        g_mtl4_keepalive = nil;
        memset(g_m4_gen, 0, sizeof(g_m4_gen));
        g_mtl4_encode_error = 0;
    }
    return ok;
}

#endif  /* macOS 26 SDK */
#endif  /* DS4_METAL_MTL4_H */
