/*
 * The Metal 4 backend: every command batch as command buffers on one queue.
 *
 * Selected with the process's backend (g_backend == 4), it carries the whole
 * request - prefill, the embedding, the gated layers, the ungated tail, the
 * head - and the Metal 3 queue carries nothing. The batch the engine opens
 * with ds4_gpu_begin_commands is a buffer from the pool below, and the
 * engine's batch identity (g_batch_cmd) is that buffer object, so a Metal 3
 * call made on it by mistake stops with an unrecognised selector instead of
 * running somewhere unordered. What follows was written when this backend
 * owned only the gated region, and still holds for every segment.
 *
 * This is not the earlier shim, and the difference is the submission, not the
 * encoding. That one left Metal 3 owning the token and handed single segments
 * across, so each segment paid a flush and a wait on both sides; every
 * dispatch carried a barrier; and anything the engine put in the Metal 3
 * buffer during a Metal 4 segment - a copy, an event - ran after that
 * segment's compute rather than in place, because two queues have no ordering
 * of their own. Arranged that way Metal 4 cannot win, and its numbers said
 * nothing about Metal 4.
 *
 * What this does instead:
 *
 *   one queue owns the region   The gated region runs entirely on the Metal 4
 *                               queue. The Metal 3 batch buffer stays open as
 *                               the identity everything else is keyed on - the
 *                               expert cache's bookkeeping, the GPU span - and
 *                               carries no dispatches. Whether it stays empty
 *                               is counted rather than assumed.
 *   a buffer a segment          Each segment takes its own allocator and
 *                               command buffer from a pool, so the segment the
 *                               engine encodes ahead is built into its own
 *                               buffer while the one before it runs, and an
 *                               allocator is reset only once its buffer has
 *                               completed.
 *   one argument table          Metal snapshots the table at the dispatch, not
 *                               at execution, so one table serves every
 *                               dispatch. The shim allocated one per encoder
 *                               reopen, which is twice a layer for nothing.
 *
 * Still deliberately slow in two places, because the answer comes first: a
 * barrier after every dispatch, and one commit a segment rather than
 * commit:count: over a batch. Both express the Metal 3 order rather than
 * improve on it, and both are the next things to relax - separately, each
 * against a byte comparison, so a wrong answer names one change.
 */
#ifndef DS4_METAL_MTL4_H
#define DS4_METAL_MTL4_H

#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000

/* Buffers that may be encoding or in flight at once. The engine keeps one
 * segment prepared behind the running one, so two are live; the gate's miss
 * ring is four deep, which is the ceiling on how far ahead it could ever go. */
#define DS4_MTL4_PIPE          4u
#define DS4_MTL4_SCRATCH_BYTES (8u << 20)
#define DS4_MTL4_SCRATCH_ALIGN 256u
/* MTL4ArgumentTableDescriptor caps maxBufferBindCount at 31. */
#define DS4_MTL4_MAX_BINDINGS  31u

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
    /* The commit feedback for this context's stamp. Written on the feedback
     * queue; `fb_stamp` is stored last and publishes the rest. */
    _Atomic uint64_t fb_stamp;
    _Atomic int fb_failed;
    double gpu_start, gpu_end;
    /* The generation of what is marked resident, and the memo of it. 0 is
     * never a live generation, so a zeroed memo entry never hits. */
    uint32_t gen;
    struct { const void *res; uint32_t gen; } memo[DS4_MTL4_MARK_MEMO];
}
@property (nonatomic, strong) id<MTL4CommandAllocator> allocator;
@property (nonatomic, strong) id<MTL4CommandBuffer> cb;
/* setBytes has no inline path in Metal 4, so the bytes are copied into a
 * scratch buffer and the table is given that address. The copy has to outlive
 * encoding, so the scratch belongs to the command buffer and is reused only
 * once that buffer has completed. */
@property (nonatomic, strong) id<MTLBuffer> scratch;
@property (nonatomic) NSUInteger scratchUsed;
/* Encoders opened into this context, held until it completes. */
@property (nonatomic, strong) NSMutableArray *encoders;
/* Whether anything has been encoded since it was begun. A buffer with
 * nothing in it is not sent: it stays open and takes the next batch. */
@property (nonatomic) BOOL used;
/* The event value that says this buffer is done. 0 while it is being encoded. */
@property (nonatomic) uint64_t stamp;
/* What this buffer reaches, resident for exactly as long as it is in flight:
 * filled while it is encoded, committed with it, emptied once its feedback
 * says it completed. */
@property (nonatomic, strong) id<MTLResidencySet> resset;
@property (nonatomic, strong) NSHashTable *resident;
@property (nonatomic) unsigned long long resBytes;
/* The table every dispatch in this context binds through. Metal snapshots a
 * table at the dispatch, not at execution, so one table serves the whole
 * context - but it is this context's, and the next context encoding while
 * this one runs has its own. */
@property (nonatomic, strong) id<MTL4ArgumentTable> table;
/* The engine's handle for this context, and its encoder. */
@property (nonatomic, strong) DS4Cmd *cmd;
@property (nonatomic, strong) DS4Enc *enc;
@end

@implementation DS4M4Ctx
@end

static id<MTL4CommandQueue> g_mtl4_queue API_AVAILABLE(macos(26.0));
static NSMutableArray *g_mtl4_pool;          /* DS4M4Ctx, all of them */
/* The context the engine is encoding into: a cursor, not an owner. */
static DS4M4Ctx *g_mtl4_cur API_AVAILABLE(macos(26.0));
static id<MTLSharedEvent> g_mtl4_event;
/* Defined with the Metal 3 wait below; the GPU's own span of what a wait
 * covered, from each buffer's feedback, goes into the same accumulator. */
static double g_gpu_span_accum_ms;
static uint64_t g_gpu_span_cbs;
static uint64_t g_mtl4_stamp;                /* last value handed to a commit */
static uint64_t g_mtl4_done;                 /* last value waited for */

static unsigned long long g_mtl4_dispatches, g_mtl4_segments;
/* Buffers actually sent, against segments begun: the two differ by the ones
 * an abort discards, and reporting begins as though they were submissions is
 * how a submission count stops being one. */
static unsigned long long g_mtl4_commits;
unsigned long long ds4_gpu_backend_m3_fallback(void);

/* Buffers sent and dispatches encoded on this backend, and Metal 3 work noted
 * while it was selected - which is the number that has to be zero. */
void ds4_gpu_mtl4_counts(unsigned long long *segments,
                         unsigned long long *dispatches,
                         unsigned long long *m3_fallback) {
    if (segments) *segments = g_mtl4_commits;
    if (dispatches) *dispatches = g_mtl4_dispatches;
    if (m3_fallback) *m3_fallback = ds4_gpu_backend_m3_fallback();
}

/* ---- residency --------------------------------------------------------- */

/* Metal 4 tracks no residency implicitly. What the engine binds or marks with
 * useResource while a buffer is encoded goes into that buffer's own set - what
 * Metal 3 does implicitly for a command buffer. The model and the expert cache
 * live in the process-lifetime sets on the queue; everything else - a
 * session's arena, a transient, a tensor freed since - is resident only while a
 * buffer that uses it is in flight. A single long-lived set held a session's
 * gigabytes past the session, and the next request's allocations did not fit. */
static unsigned long long g_mtl4_res_adds, g_mtl4_res_releases, g_mtl4_res_peak;

API_AVAILABLE(macos(26.0))
static void ds4_mtl4_mark_resident(DS4M4Ctx *c, id<MTLResource> r) {
    if (!r || !c) return;
    const void *key = (__bridge const void *)r;
    const uintptr_t k = (uintptr_t)key;
    const uint32_t slot = (uint32_t)((k >> 4) ^ (k >> 16)) & (DS4_MTL4_MARK_MEMO - 1u);
    if (c->memo[slot].res == key && c->memo[slot].gen == c->gen) return;
    if (!c.resset) return;
    if (![c.resident containsObject:r]) {
        [c.resident addObject:r];
        [c.resset addAllocation:r];
        c.resBytes += [r allocatedSize];
        g_mtl4_res_adds++;
    }
    c->memo[slot].res = key;
    c->memo[slot].gen = c->gen;
}

/* Residency sets the engine attaches to the Metal 3 queue. The Metal 4 queue
 * inherits none of them, and what they hold is the model and the expert cache,
 * reached by raw address from the masked kernels - nothing else declares them.
 * A segment without these faults in AGX on its first dispatch. */
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

/* The Metal 3 side dropped a set - a model released, an expert table freed.
 * The Metal 4 queue lets go of it too, or it would hold what the set covers
 * resident for the life of the process. */
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

/* Sets attached to the Metal 4 queue, counted: a set the Metal 3 queue has and
 * this one lacks makes what it holds unreadable here without a fault. */
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

/* The shape of what this backend submits, counted by kind so a token can be
 * priced by structure: how many encoders it opens and of which kind, every
 * barrier by the reason it was inserted, the argument-table traffic, and the
 * host's waits and how long they took. Plain counters. */
enum {
    DS4_M4S_ENC_SERIAL = 0,     /* encoders opened serial */
    DS4_M4S_ENC_CONCURRENT,     /* encoders opened for a concurrent section */
    DS4_M4S_BAR_DISPATCH,       /* barrier after a dispatch in a serial encoder */
    DS4_M4S_BAR_COPY,           /* barriers either side of a copy */
    DS4_M4S_BAR_ENGINE,         /* the engine's own memoryBarrier calls */
    DS4_M4S_BAR_OPEN,           /* queue-stage barrier opening an encoder */
    DS4_M4S_BAR_END,            /* queue-facing barrier ending an encoder */
    DS4_M4S_TABLE_CLEAR,        /* whole-table clears (31 slots each) */
    DS4_M4S_SET_ADDRESS,        /* setBuffer bindings */
    DS4_M4S_SET_BYTES,          /* setBytes bindings */
    DS4_M4S_BYTES_COPIED,       /* bytes copied into the constant scratch */
    DS4_M4S_DISPATCH,
    DS4_M4S_COPY,
    DS4_M4S_WAIT_CALLS,         /* waits that actually blocked */
    DS4_M4S_WAIT_NS,            /* host time blocked in them */
    DS4_M4S_BAR_SECTION,        /* in-encoder barrier where a section ended */
    DS4_M4S_KINDS
};
static unsigned long long g_m4s[DS4_M4S_KINDS];
void ds4_gpu_mtl4_structure(unsigned long long out[DS4_M4S_KINDS]);
void ds4_gpu_mtl4_structure(unsigned long long out[DS4_M4S_KINDS]) {
    for (int i = 0; i < DS4_M4S_KINDS; i++) out[i] = g_m4s[i];
}

/* The engine's encoder calls on this backend. Metal 4 runs the dispatches of
 * one encoder concurrently unless a barrier says otherwise, the opposite
 * default from Metal 3; a serial section puts the Metal 3 order back with a
 * barrier after every dispatch. */

API_AVAILABLE(macos(26.0))
static inline id<MTL4ComputeCommandEncoder> ds4_m4_enc(DS4Enc *e) {
    return (id<MTL4ComputeCommandEncoder>)e->m4enc;
}
API_AVAILABLE(macos(26.0))
static inline id<MTL4ArgumentTable> ds4_m4_table(DS4Enc *e) {
    return (id<MTL4ArgumentTable>)e->m4table;
}

static void ds4_m4_pso(DS4Enc *e, id<MTLComputePipelineState> pso) {
    if (@available(macOS 26.0, *)) [ds4_m4_enc(e) setComputePipelineState:pso];
}

static void ds4_m4_buf(DS4Enc *e, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 buffer index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    /* setBuffer:nil clears the slot in Metal 3; skipping the call would leave
     * whatever the slot held. */
    g_m4s[DS4_M4S_SET_ADDRESS]++;
    if (@available(macOS 26.0, *)) {
        if (!buffer) { [ds4_m4_table(e) setAddress:0 atIndex:index]; return; }
        ds4_mtl4_mark_resident(e->m4, buffer);
        [ds4_m4_table(e) setAddress:buffer.gpuAddress + offset atIndex:index];
    }
}

static void ds4_m4_bytes(DS4Enc *e, const void *bytes, NSUInteger length, NSUInteger index) {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 bytes index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    if (@available(macOS 26.0, *)) {
        DS4M4Ctx *b = e->m4;
        id<MTLBuffer> s = b.scratch;
        const NSUInteger off = (b.scratchUsed + DS4_MTL4_SCRATCH_ALIGN - 1u) &
                               ~(NSUInteger)(DS4_MTL4_SCRATCH_ALIGN - 1u);
        if (!s || off + length > DS4_MTL4_SCRATCH_BYTES) {
            /* Binding nothing here would be a wrong answer with no symptom. */
            fprintf(stderr, "ds4: Metal 4 scratch exhausted at %lu + %lu bytes\n",
                    (unsigned long)off, (unsigned long)length);
            abort();
        }
        g_m4s[DS4_M4S_SET_BYTES]++;
        g_m4s[DS4_M4S_BYTES_COPIED] += length;
        memcpy((char *)[s contents] + off, bytes, length);
        b.scratchUsed = off + length;
        /* The scratch carries every kernel's constant arguments. Nothing else
         * declares it, and a buffer that is not resident reads as whatever the
         * page holds. */
        ds4_mtl4_mark_resident(b, s);
        [ds4_m4_table(e) setAddress:s.gpuAddress + off atIndex:index];
    }
}

static void ds4_m4_tgmem(DS4Enc *e, NSUInteger length, NSUInteger index) {
    if (@available(macOS 26.0, *)) [ds4_m4_enc(e) setThreadgroupMemoryLength:length atIndex:index];
}

static void ds4_m4_use(DS4Enc *e, id<MTLResource> resource, MTLResourceUsage usage) {
    if (@available(macOS 26.0, *)) ds4_mtl4_mark_resident(e->m4, resource);
}

/* memoryBarrier becomes barrierAfterEncoderStages:, the barrier that orders
 * dispatches inside this encoder. barrierAfterStages:beforeQueueStages: is a
 * different call that orders against other queue work and expresses nothing
 * here; it measures fast because of that, and using it for a dependency edge
 * would be quietly wrong.
 *
 * Every stage on both sides, deliberately: what this encoder contains is not
 * only dispatches - a buffer copy is blit-stage work, and an indirect dispatch
 * fetches its grid as arguments. Naming MTLStageDispatch alone leaves both of
 * those unordered against the dispatches around them. */
static void ds4_m4_barrier_for(DS4Enc *e, int kind) {
    g_m4s[kind]++;
    if (@available(macOS 26.0, *))
        [ds4_m4_enc(e) barrierAfterEncoderStages:MTLStageAll
                              beforeEncoderStages:MTLStageAll
                                visibilityOptions:ds4_mtl4_visibility()];
}

static void ds4_m4_engine_barrier(DS4Enc *e) { ds4_m4_barrier_for(e, DS4_M4S_BAR_ENGINE); }

static void ds4_m4_dispatch(DS4Enc *e, int kind, MTLSize a, MTLSize b,
                            id<MTLBuffer> indirect, NSUInteger indirect_offset) {
    if (@available(macOS 26.0, *)) {
        g_mtl4_dispatches++;
        g_m4s[DS4_M4S_DISPATCH]++;
        id<MTL4ComputeCommandEncoder> enc = ds4_m4_enc(e);
        switch (kind) {
        case DS4_M4_NODE_GROUPS:
            [enc dispatchThreadgroups:a threadsPerThreadgroup:b];
            break;
        case DS4_M4_NODE_THREADS:
            [enc dispatchThreads:a threadsPerThreadgroup:b];
            break;
        case DS4_M4_NODE_INDIRECT:
            ds4_mtl4_mark_resident(e->m4, indirect);
            [enc dispatchThreadgroupsWithIndirectBuffer:indirect.gpuAddress + indirect_offset
                                  threadsPerThreadgroup:b];
            break;
        }
        if (e->serial) ds4_m4_barrier_for(e, DS4_M4S_BAR_DISPATCH);
    }
}

static void ds4_m4_sample_counters(DS4Enc *e) {
    /* The timeline diagnostic has no Metal 4 form yet; off here rather than
     * silently sampling nothing. */
    (void)e;
}

/* The engine ends its encoder at every section: a concurrent section and the
 * serial run after it, twice a layer. On Metal 3 that is a new encoder, and a
 * new Metal 4 encoder has to be ordered against the queue - a queue-stage
 * barrier to open it and another to end it, which wait on everything the GPU
 * holds, earlier buffers included. Measured, those two barriers at about two
 * hundred sections a token were ~6.5 ms of GPU time. Inside one buffer the
 * only order that section boundary needs is "what came before, before what
 * comes after", which one barrier inside the encoder says. So the engine's
 * end marks the section, and the next section opens with that barrier on the
 * same encoder. */
static void ds4_m4_end(DS4Enc *e) { e->ended = YES; }

/* A buffer copy in the compute encoder. Metal 3 needs a blit encoder for it,
 * which means closing the compute one; here it is a command like a dispatch,
 * and Metal 4 runs the commands of an encoder concurrently. The engine issues
 * this where Metal 3 gave it a whole encoder boundary, so it is ordered on
 * both sides. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_copy_in(DS4Enc *e, id<MTLBuffer> src, NSUInteger soff,
                           id<MTLBuffer> dst, NSUInteger doff, NSUInteger size) {
    ds4_mtl4_mark_resident(e->m4, src);
    ds4_mtl4_mark_resident(e->m4, dst);
    g_m4s[DS4_M4S_COPY]++;
    if (e->serial) ds4_m4_barrier_for(e, DS4_M4S_BAR_COPY);
    [ds4_m4_enc(e) copyFromBuffer:src sourceOffset:soff
                         toBuffer:dst destinationOffset:doff size:size];
    if (e->serial) ds4_m4_barrier_for(e, DS4_M4S_BAR_COPY);
}

/* The real end, when the context is committed or dropped. Flush what this
 * encoder wrote out to the coherence point: the host reads the gate's miss
 * and route words straight out of shared buffers once the buffer has
 * completed, and the next buffer's first encoder reads what this one
 * produced. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_finish(DS4Enc *e) {
    if (e->finished) return;
    e->finished = YES;
    e->ended = YES;
    g_m4s[DS4_M4S_BAR_END]++;
    [ds4_m4_enc(e) barrierAfterStages:MTLStageAll
                    beforeQueueStages:MTLStageAll
                    visibilityOptions:ds4_mtl4_visibility()];
    [ds4_m4_enc(e) endEncoding];
}

/* A new section on the same encoder: everything encoded so far completes and
 * is visible before anything encoded next starts, and the bindings start
 * clear, as they do for a fresh Metal 3 encoder. */
API_AVAILABLE(macos(26.0))
static void ds4_m4_begin_section(DS4Enc *e, BOOL serial) {
    g_m4s[DS4_M4S_BAR_SECTION]++;
    [ds4_m4_enc(e) barrierAfterEncoderStages:MTLStageAll
                         beforeEncoderStages:MTLStageAll
                           visibilityOptions:ds4_mtl4_visibility()];
    g_m4s[DS4_M4S_TABLE_CLEAR]++;
    for (NSUInteger i = 0; i < DS4_MTL4_MAX_BINDINGS; i++)
        [ds4_m4_table(e) setAddress:0 atIndex:i];
    e->serial = serial;
    g_m4s[serial ? DS4_M4S_ENC_SERIAL : DS4_M4S_ENC_CONCURRENT]++;
    e->ended = NO;
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
    MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
    d.maxBufferBindCount = DS4_MTL4_MAX_BINDINGS;
    /* A new Metal 3 encoder starts with nil bindings; without this the unset
     * slots of a table are undefined, which is the opposite of the backend
     * being replaced. */
    d.initializeBindings = YES;
    g_mtl4_pool = [NSMutableArray array];
    for (unsigned i = 0; i < DS4_MTL4_PIPE; i++) {
        DS4M4Ctx *b = [DS4M4Ctx new];
        NSError *err = nil;
        b.table = [g_device newArgumentTableWithDescriptor:d error:&err];
        if (!b.table) {
            fprintf(stderr, "ds4: Metal 4 argument table: %s\n", err.description.UTF8String);
            g_mtl4_queue = nil;
            g_mtl4_pool = nil;
            return 0;
        }
        b.allocator = [g_device newCommandAllocator];
        b.cb = [g_device newCommandBuffer];
        b.scratch = [g_device newBufferWithLength:DS4_MTL4_SCRATCH_BYTES
                                          options:MTLResourceStorageModeShared];
        b.encoders = [NSMutableArray array];
        {
            MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
            rd.label = @"ds4_mtl4_buffer";
            NSError *rerr = nil;
            b.resset = [g_device newResidencySetWithDescriptor:rd error:&rerr];
            b.resident = [NSHashTable hashTableWithOptions:NSPointerFunctionsStrongMemory |
                                                          NSPointerFunctionsObjectPointerPersonality];
        }
        if (!b.allocator || !b.cb || !b.scratch || !b.resset) {
            fprintf(stderr, "ds4: Metal 4 command pool unavailable\n");
            g_mtl4_queue = nil;
            g_mtl4_pool = nil;
            return 0;
        }
        b.scratch.label = @"ds4_mtl4_scratch";
        b.cmd = [DS4Cmd new];
        b.cmd->m4 = b;
        b.enc = [DS4Enc new];
        b.enc->m4 = b;
        [g_mtl4_pool addObject:b];
    }
    ds4_mtl4_attach_sets();
    return 1;
}

/* A completed buffer lets go of what it held: emptied and committed now, not
 * when the buffer is next reused, so a freed session is not held resident by
 * a buffer that happens to sit idle in the pool. */
static void ds4_mtl4_release_completed(void) API_AVAILABLE(macos(26.0));
static void ds4_mtl4_release_completed(void) {
    for (DS4M4Ctx *b in g_mtl4_pool) {
        if (b == g_mtl4_cur || b.stamp == 0 || b.stamp > g_mtl4_done || !b.resident.count)
            continue;
        g_mtl4_res_releases += b.resident.count;
        [b.resset removeAllAllocations];
        [b.resset commit];
        [b.resident removeAllObjects];
        b.resBytes = 0;
    }
}

/* The commit feedback, a slot a stamp. Without it a buffer the GPU failed - out
 * of memory, a fault - completes like any other and the host reads what was
 * left there. The event says a buffer finished, not that it succeeded, so a
 * stamp counts as done only once its own feedback has arrived and said so;
 * feedback that has not arrived is a failure, never a success. */
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
        fprintf(stderr, "ds4: Metal 4 command buffer %llu failed: %s\n",
                (unsigned long long)stamp, error.description.UTF8String);
        atomic_fetch_add(&g_mtl4_fb_errors, 1u);
    }
    atomic_store(&c->fb_failed, error ? 1 : 0);
    atomic_store(&c->fb_stamp, stamp);       /* last: publishes the rest */
    atomic_fetch_add(&g_mtl4_fb_received, 1u);
}

/* Wait until every buffer committed up to `value` has completed and reported.
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
        fprintf(stderr, "ds4: Metal 4 wait spans %llu buffers, past the feedback ring\n",
                (unsigned long long)(value - g_mtl4_done));
        return 0;
    }
    DS4_M4_NOTE("wait", DS4_BK_WAIT);
    const uint64_t w0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    const BOOL ok = [g_mtl4_event waitUntilSignaledValue:value timeoutMS:600000];
    g_m4s[DS4_M4S_WAIT_CALLS]++;
    g_m4s[DS4_M4S_WAIT_NS] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - w0;
    if (!ok) {
        fprintf(stderr, "ds4: Metal 4 command buffer timed out\n");
        return 0;
    }
    int good = 1;
    for (uint64_t st = g_mtl4_done + 1u; st <= value; st++) {
        /* The context that carried the stamp: it is not reused before its
         * stamp has been waited for, so it is still in the pool. */
        DS4M4Ctx *slot = nil;
        if (@available(macOS 26.0, *))
            for (DS4M4Ctx *c in g_mtl4_pool) if (c.stamp == st) { slot = c; break; }
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
            fprintf(stderr, "ds4: Metal 4 command buffer %llu completed without feedback\n",
                    (unsigned long long)st);
            return 0;
        }
        if (atomic_load(&slot->fb_failed)) good = 0;
        /* The union of the buffers' spans, not their sum: with the next
         * segment committed before the current one is read, one buffer's
         * span starts while the previous one is still running, and adding
         * both counted the overlap twice. Stamps are waited in order. */
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

/* A buffer whose work has completed: its allocator may be reset and its
 * scratch reused. */
API_AVAILABLE(macos(26.0))
static DS4M4Ctx *ds4_mtl4_take(void) {
    for (DS4M4Ctx *b in g_mtl4_pool)
        if (b != g_mtl4_cur && b.stamp <= g_mtl4_done) return b;
    /* Every buffer is still out. Wait for the oldest thing there is, which is
     * everything: the pipeline is two deep and the pool is four, so this is a
     * guard rather than a path. */
    if (!ds4_mtl4_wait_all()) return nil;
    for (DS4M4Ctx *b in g_mtl4_pool)
        if (b != g_mtl4_cur && b.stamp <= g_mtl4_done) return b;
    return nil;
}

static int ds4_mtl4_begin(void) {
    if (@available(macOS 26.0, *)) {
        if (!ds4_mtl4_on()) return 0;
        if (g_mtl4_cur) return 1;
        if (!ds4_mtl4_ready()) return 0;
        DS4M4Ctx *b = ds4_mtl4_take();
        if (!b) return 0;
        [b.allocator reset];
        [b.cb beginCommandBufferWithAllocator:b.allocator];
        b.scratchUsed = 0;
        b.stamp = 0;
        b.used = NO;
        [b.encoders removeAllObjects];
        DS4Enc *e = b.enc;
        e->m4enc = nil;
        e->m4table = b.table;
        e->ended = e->finished = e->serial = e->concurrent = NO;
        g_mtl4_cur = b;
        /* 0 is never a live generation, so a zeroed memo entry never hits. */
        if (++b->gen == 0) ++b->gen;
        /* Proof the path ran, not that it was asked for. Byte identity from a
         * backend that silently fell back is byte identity from the backend it
         * fell back to. */
        g_mtl4_segments++;
        return 1;
    }
    return 0;
}

API_AVAILABLE(macos(26.0))
static void ds4_mtl4_close_encoder(void) {
    DS4Enc *e = g_mtl4_cur.enc;
    if (!e || !e->m4enc) return;
    ds4_m4_finish(e);
    /* Held until the context completes: the GPU reads its bindings through
     * objects this owns. */
    [g_mtl4_cur.encoders addObject:e->m4enc];
    e->m4enc = nil;
}

/* End the open buffer and send it, without waiting. */
static int ds4_mtl4_commit(void) {
    if (@available(macOS 26.0, *)) {
        if (!g_mtl4_cur) return 0;
        DS4M4Ctx *b = g_mtl4_cur;
        ds4_mtl4_close_encoder();
        /* Residency is declared once the set is complete. Declaring it before
         * encoding looked tidier and was empty - the scratch does not exist
         * until the first setBytes - so the buffer carrying every kernel's
         * constant arguments was never made resident. */
        [b.resset commit];
        [b.cb useResidencySet:b.resset];
        {
            unsigned long long held = 0;
            for (DS4M4Ctx *p in g_mtl4_pool) held += p.resBytes;
            if (held > g_mtl4_res_peak) g_mtl4_res_peak = held;
        }
        [b.cb endCommandBuffer];
        id<MTL4CommandBuffer> one = b.cb;
        const uint64_t stamp = g_mtl4_stamp + 1u;
        MTL4CommitOptions *opts = [MTL4CommitOptions new];
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
            ds4_mtl4_fb_note(b, stamp, fb.error, fb.GPUStartTime, fb.GPUEndTime);
        }];
        DS4_M4_NOTE("segment", DS4_BK_COMMIT);
        [g_mtl4_queue commit:&one count:1 options:opts];
        b.stamp = ++g_mtl4_stamp;
        g_mtl4_commits++;
        [g_mtl4_queue signalEvent:g_mtl4_event value:b.stamp];
        g_mtl4_cur = nil;
        return 1;
    }
    return 0;
}

/* Throw away what is encoded and not committed. The engine drops a prepared
 * segment when the one before it aborted; nothing has reached the GPU.
 *
 * The buffer is ended and begun again on the same allocator rather than
 * handed back to the pool. Returning it would mean resetting that allocator
 * before the next begin, and a command buffer that ended without being
 * committed still holds storage the allocator hands back at that begin: with
 * the allocator reset in between, the generation it is looking for is gone
 * and the driver faults inside beginCommandBufferWithAllocator. Ending and
 * re-beginning returns the storage in the same generation, which is what the
 * abort path needs - and at forty layers it is about thirty-two aborts a
 * token, so this is the common case, not the corner. */
static void ds4_mtl4_discard(void) {
    if (@available(macOS 26.0, *)) {
        if (!g_mtl4_cur) return;
        DS4M4Ctx *b = g_mtl4_cur;
        ds4_mtl4_close_encoder();
        [b.cb endCommandBuffer];
        [b.encoders removeAllObjects];
        [b.cb beginCommandBufferWithAllocator:b.allocator];
        b.scratchUsed = 0;
        b.stamp = 0;
        b.used = NO;
    }
}

/* DS4_METAL_V41_MTL4_SERIAL=1 puts a barrier after every dispatch in a
 * concurrent section too, which is the control. */
static int ds4_mtl4_serial(void) {
    static int checked, serial;
    if (!checked) {
        const char *e = getenv("DS4_METAL_V41_MTL4_SERIAL");
        checked = 1;
        serial = e && e[0] ? atoi(e) != 0 : 0;
    }
    return serial;
}

API_AVAILABLE(macos(26.0))
static DS4Enc *ds4_mtl4_encoder(void) {
    if (!g_mtl4_cur) return nil;
    DS4Enc *de = g_mtl4_cur.enc;
    /* The engine closes and reopens the batch encoder inside a segment - a
     * concurrent section does it twice a layer - so an encoder that has been
     * ended must not be handed back. */
    if (de->m4enc && !de->ended) return de;
    if (de->m4enc && !de->finished) {
        ds4_m4_begin_section(de, ds4_mtl4_serial() || !g_batch_encoder_concurrent);
        de->concurrent = g_batch_encoder_concurrent;
        return de;
    }
    id<MTL4ComputeCommandEncoder> e = [g_mtl4_cur.cb computeCommandEncoder];
    if (!e) {
        fprintf(stderr, "ds4: Metal 4 encoder unavailable\n");
        g_mtl4_encode_error = 1;
        return nil;
    }
    g_mtl4_cur.used = YES;
    [e setArgumentTable:g_mtl4_cur.table];
    /* Metal 3 serialises one compute encoder after another; Metal 4 does not,
     * and an encoder opened after another in the same buffer runs beside it
     * unless told otherwise. This is what the queue-stage barrier is for: it
     * orders against work outside this encoder, which is the previous
     * encoder's. Without it the answer came out wrong rather than late. */
    g_m4s[DS4_M4S_BAR_OPEN]++;
    g_m4s[DS4_M4S_TABLE_CLEAR]++;
    [e barrierAfterQueueStages:MTLStageAll
                  beforeStages:MTLStageAll
             visibilityOptions:ds4_mtl4_visibility()];
    /* A fresh Metal 3 encoder starts with nil bindings, and the engine relies
     * on nothing carrying over. The table outlives the encoder, so it is
     * cleared here. */
    for (NSUInteger i = 0; i < DS4_MTL4_MAX_BINDINGS; i++)
        [g_mtl4_cur.table setAddress:0 atIndex:i];
    de->m4enc = e;
    de->m4table = g_mtl4_cur.table;
    de->ended = de->finished = NO;
    de->concurrent = g_batch_encoder_concurrent;
    /* A barrier after every dispatch reproduces the serial Metal 3 encoder,
     * and for a serial encoder that is what the engine asked for. A
     * concurrent section is the opposite: the engine opened it because the
     * dispatches in a level are independent, and it puts an explicit barrier
     * at each level break itself. Serialising those here threw away the
     * parallelism the section exists to express, and left Metal 4 running a
     * graph more serial than the one Metal 3 runs.
     *
     * DS4_METAL_V41_MTL4_SERIAL=1 puts it back, which is the control. */
    {
        de->serial = ds4_mtl4_serial() || !g_batch_encoder_concurrent;
        g_m4s[de->serial ? DS4_M4S_ENC_SERIAL : DS4_M4S_ENC_CONCURRENT]++;
    }
    return de;
}

/* Whether the Metal 4 backend took this copy. The engine falls back to its
 * blit encoder when it did not. */
API_AVAILABLE(macos(26.0))
static int ds4_mtl4_copy(id<MTLBuffer> src, NSUInteger soff,
                         id<MTLBuffer> dst, NSUInteger doff, NSUInteger size) {
    if (!g_mtl4_cur) return 0;
    DS4Enc *e = ds4_mtl4_encoder();
    if (!e) return 0;
    if (!src || !dst) {
        g_mtl4_encode_error = 1;
        return 0;
    }
    ds4_m4_copy_in(e, src, soff, dst, doff, size);
    return 1;
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
        for (DS4M4Ctx *b in g_mtl4_pool) { entries += b.resident.count; bytes += b.resBytes; }
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
    if (@available(macOS 26.0, *)) return g_mtl4_cur && g_mtl4_cur.used;
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
            ds4_mtl4_close_encoder();
            if (g_mtl4_cur.used) { if (!ds4_mtl4_commit()) ok = 0; }
            else {
                [g_mtl4_cur.cb endCommandBuffer];
                g_mtl4_cur = nil;
            }
        }
        if (g_mtl4_queue && !ds4_mtl4_wait_all()) ok = 0;
        for (DS4M4Ctx *b in g_mtl4_pool) {
            [b.resset removeAllAllocations];
            [b.resset commit];
            [b.resident removeAllObjects];
            b.resBytes = 0;
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
        g_mtl4_encode_error = 0;
    }
    return ok;
}

#endif  /* macOS 26 SDK */
#endif  /* DS4_METAL_MTL4_H */
