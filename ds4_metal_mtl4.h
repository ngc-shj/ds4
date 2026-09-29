/*
 * The Metal 4 backend: every command batch as command buffers on one queue.
 *
 * Selected with the process's backend (g_backend == 4), it carries the whole
 * request - prefill, the embedding, the gated layers, the ungated tail, the
 * head - and the Metal 3 queue carries nothing. The batch the engine opens
 * with ds4_gpu_begin_commands is a buffer from the pool below, and the
 * engine's batch identity (g_batch_cb) is that buffer object, so a Metal 3
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

/* ---- one buffer of the pipeline ---------------------------------------- */

API_AVAILABLE(macos(26.0))
@interface DS4MTL4Buffer : NSObject
@property (nonatomic, strong) id<MTL4CommandAllocator> allocator;
@property (nonatomic, strong) id<MTL4CommandBuffer> cb;
/* setBytes has no inline path in Metal 4, so the bytes are copied into a
 * scratch buffer and the table is given that address. The copy has to outlive
 * encoding, so the scratch belongs to the command buffer and is reused only
 * once that buffer has completed. */
@property (nonatomic, strong) id<MTLBuffer> scratch;
@property (nonatomic) NSUInteger scratchUsed;
/* Encoders opened into this buffer, held until it completes. */
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
@end

@implementation DS4MTL4Buffer
@end

/* ---- the encoder ------------------------------------------------------- */

API_AVAILABLE(macos(26.0))
@interface DS4MTL4Encoder : NSObject <DS4Encoder>
@property (nonatomic, strong) id<MTL4ComputeCommandEncoder> enc;
@property (nonatomic, strong) id<MTL4ArgumentTable> table;
@property (nonatomic, weak) DS4MTL4Buffer *owner;
/* Metal 4 runs the dispatches of one encoder concurrently unless a barrier
 * says otherwise, which is the opposite default from the backend this
 * replaces. Phase 1 puts the Metal 3 order back, explicitly. */
@property (nonatomic) BOOL barrierEveryDispatch;
@property (nonatomic) BOOL ended;
/* Metal 4 encodes a buffer copy into the compute encoder. Metal 3 needs a
 * blit encoder for it, which means closing the compute one - a pipeline
 * flush the engine pays several times a token. This is not on DS4Encoder:
 * that protocol is the set of selectors MTLComputeCommandEncoder already
 * answers, which is what lets the Metal 3 path be the encoder itself. */
- (void)copyFromBuffer:(id<MTLBuffer>)src sourceOffset:(NSUInteger)soff
              toBuffer:(id<MTLBuffer>)dst destinationOffset:(NSUInteger)doff
                  size:(NSUInteger)size;
@end

static id<MTL4CommandQueue> g_mtl4_queue API_AVAILABLE(macos(26.0));
static id<MTL4ArgumentTable> g_mtl4_table API_AVAILABLE(macos(26.0));
static NSMutableArray *g_mtl4_pool;          /* DS4MTL4Buffer, all of them */
static DS4MTL4Encoder *g_mtl4_enc API_AVAILABLE(macos(26.0));
static DS4MTL4Buffer *g_mtl4_cur API_AVAILABLE(macos(26.0));
static id<MTLSharedEvent> g_mtl4_event;
/* Defined with the Metal 3 wait below; the GPU's own span of what a wait
 * covered, from each buffer's feedback, goes into the same accumulator. */
static double g_gpu_span_accum_ms;
static uint64_t g_gpu_span_cbs;
static uint64_t g_mtl4_stamp;                /* last value handed to a commit */
static uint64_t g_mtl4_done;                 /* last value waited for */
static int g_mtl4_region;                    /* inside the gated region */

static unsigned long long g_mtl4_dispatches, g_mtl4_segments, g_mtl4_copies_m3;
/* Buffers actually sent, against segments begun: the two differ by the ones
 * an abort discards, and reporting begins as though they were submissions is
 * how a submission count stops being one. */
static unsigned long long g_mtl4_commits;
extern unsigned long long g_ds4_m3_commits, g_ds4_m3_skipped;

/* Counted rather than assumed: work the engine puts in the Metal 3 batch
 * buffer while a Metal 4 buffer is open runs after it, not in place. The
 * design says there is none; this is what says so about a run. */
void ds4_gpu_mtl4_note_metal3_work(const char *what);
static int ds4_mtl4_open(void);

void ds4_gpu_mtl4_counts(unsigned long long *segments,
                         unsigned long long *dispatches,
                         unsigned long long *m3_work,
                         unsigned long long *m3_sent,
                         unsigned long long *m3_skipped) {
    if (segments) *segments = g_mtl4_commits;
    if (dispatches) *dispatches = g_mtl4_dispatches;
    if (m3_work) *m3_work = g_mtl4_copies_m3;
    if (m3_sent) *m3_sent = g_ds4_m3_commits;
    if (m3_skipped) *m3_skipped = g_ds4_m3_skipped;
}

void ds4_gpu_mtl4_note_metal3_work(const char *what) {
    if (@available(macOS 26.0, *)) {
        if (!ds4_mtl4_open()) return;
        g_mtl4_copies_m3++;
        /* Said once a site: the whole point is which of them it is. */
        enum { SITES = 8 };
        static const char *seen[SITES];
        static unsigned n_seen;
        const char *name = what ? what : "(unlabelled)";
        for (unsigned i = 0; i < n_seen; i++)
            if (seen[i] == name || strcmp(seen[i], name) == 0) return;
        if (n_seen < SITES) seen[n_seen++] = name;
        fprintf(stderr, "ds4: Metal 3 work \"%s\" encoded inside a Metal 4 segment; "
                        "it runs after that segment, not in place\n", name);
    }
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

static void ds4_mtl4_mark_resident(id<MTLResource> r) API_AVAILABLE(macos(26.0));
/* The open buffer's generation, and a direct-mapped memo of which resources
 * were marked in it. The engine binds the same few hundred buffers thousands
 * of times a token; a hash-table probe and three message sends for each of
 * those binds was host encode time the Metal 3 encoder does not spend. A hit
 * is exact: the entry is written only after the mark, and the buffer holds
 * every resource it marked until it completes, so an address in the memo
 * cannot be reused by another object within the same generation. A miss just
 * falls through to the table. */
static uint32_t g_mtl4_gen;
#define DS4_MTL4_MARK_MEMO 4096u
static struct { const void *res; uint32_t gen; } g_mtl4_mark_memo[DS4_MTL4_MARK_MEMO];

static void ds4_mtl4_mark_resident(id<MTLResource> r) {
    if (!r) return;
    const void *key = (__bridge const void *)r;
    const uintptr_t k = (uintptr_t)key;
    const uint32_t slot = (uint32_t)((k >> 4) ^ (k >> 16)) & (DS4_MTL4_MARK_MEMO - 1u);
    if (g_mtl4_mark_memo[slot].res == key && g_mtl4_mark_memo[slot].gen == g_mtl4_gen) return;
    DS4MTL4Buffer *b = g_mtl4_cur;
    if (!b || !b.resset) return;
    if (![b.resident containsObject:r]) {
        [b.resident addObject:r];
        [b.resset addAllocation:r];
        b.resBytes += [r allocatedSize];
        g_mtl4_res_adds++;
    }
    g_mtl4_mark_memo[slot].res = key;
    g_mtl4_mark_memo[slot].gen = g_mtl4_gen;
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

@implementation DS4MTL4Encoder

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
    [_enc setComputePipelineState:state];
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 buffer index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    /* setBuffer:nil clears the slot in Metal 3; skipping the call would leave
     * whatever the slot held. */
    if (!buffer) { [_table setAddress:0 atIndex:index]; return; }
    ds4_mtl4_mark_resident(buffer);
    [_table setAddress:buffer.gpuAddress + offset atIndex:index];
}

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 bytes index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    DS4MTL4Buffer *b = g_mtl4_cur;
    id<MTLBuffer> s = b.scratch;
    const NSUInteger off = (b.scratchUsed + DS4_MTL4_SCRATCH_ALIGN - 1u) &
                           ~(NSUInteger)(DS4_MTL4_SCRATCH_ALIGN - 1u);
    if (!s || off + length > DS4_MTL4_SCRATCH_BYTES) {
        /* Binding nothing here would be a wrong answer with no symptom. */
        fprintf(stderr, "ds4: Metal 4 scratch exhausted at %lu + %lu bytes\n",
                (unsigned long)off, (unsigned long)length);
        abort();
    }
    memcpy((char *)[s contents] + off, bytes, length);
    b.scratchUsed = off + length;
    /* The scratch carries every kernel's constant arguments. Nothing else
     * declares it, and a buffer that is not resident reads as whatever the
     * page holds. */
    ds4_mtl4_mark_resident(s);
    [_table setAddress:s.gpuAddress + off atIndex:index];
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
    [_enc setThreadgroupMemoryLength:length atIndex:index];
}

- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage {
    (void)usage;
    ds4_mtl4_mark_resident(resource);
}

- (void)useResources:(const id<MTLResource> *)resources count:(NSUInteger)count
               usage:(MTLResourceUsage)usage {
    (void)usage;
    for (NSUInteger i = 0; i < count; i++) ds4_mtl4_mark_resident(resources[i]);
}

/* memoryBarrier becomes barrierAfterEncoderStages:, the barrier that orders
 * dispatches inside this encoder. barrierAfterStages:beforeQueueStages: is a
 * different call that orders against other queue work and expresses nothing
 * here; it measures fast because of that, and using it for a dependency edge
 * would be quietly wrong. */
/* Every stage on both sides, deliberately. A Metal 3 compute encoder orders
 * what it contains implicitly, and what this encoder contains is not only
 * dispatches: a buffer copy is blit-stage work, and an indirect dispatch
 * fetches its grid as arguments. Naming MTLStageDispatch alone - which is
 * what the shim did - leaves both of those unordered against the dispatches
 * around them, and an answer wrong for that reason does not move when the
 * barriers inside the encoder are rearranged, because the edge that is
 * missing was never a dispatch-to-dispatch edge. Narrowing the mask is for
 * the DAG, one edge at a time, against a byte comparison. */
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
    (void)scope;
    [_enc barrierAfterEncoderStages:MTLStageAll
                    beforeEncoderStages:MTLStageAll
                      visibilityOptions:ds4_mtl4_visibility()];
}

- (void)memoryBarrierWithResources:(const id<MTLResource> *)resources count:(NSUInteger)count {
    for (NSUInteger i = 0; i < count; i++) ds4_mtl4_mark_resident(resources[i]);
    [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {
    g_mtl4_dispatches++;
    [_enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
    if (_barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)tpt {
    ds4_mtl4_mark_resident(indirectBuffer);
    g_mtl4_dispatches++;
    [_enc dispatchThreadgroupsWithIndirectBuffer:indirectBuffer.gpuAddress + offset
                               threadsPerThreadgroup:tpt];
    if (_barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tpt {
    g_mtl4_dispatches++;
    [_enc dispatchThreads:threads threadsPerThreadgroup:tpt];
    if (_barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)copyFromBuffer:(id<MTLBuffer>)src sourceOffset:(NSUInteger)soff
              toBuffer:(id<MTLBuffer>)dst destinationOffset:(NSUInteger)doff
                  size:(NSUInteger)size {
    ds4_mtl4_mark_resident(src);
    ds4_mtl4_mark_resident(dst);
    /* A copy is a command in this encoder like a dispatch, and Metal 4 runs
     * the commands of an encoder concurrently. The engine issues this where
     * Metal 3 gave it a whole encoder boundary, so it is ordered on both
     * sides here. */
    if (_barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [_enc copyFromBuffer:src sourceOffset:soff
                    toBuffer:dst destinationOffset:doff size:size];
    if (_barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier {
    /* The timeline diagnostic has no Metal 4 form yet; off here rather than
     * silently sampling nothing. */
    (void)sampleBuffer; (void)sampleIndex; (void)barrier;
}

- (void)endEncoding {
    if (self.ended) return;
    self.ended = YES;
    /* Flush what this encoder wrote out to the coherence point before it
     * ends. Metal 3 does this at the end of a command buffer; Metal 4 says
     * only what you ask it to say. The host reads the gate's miss and route
     * words straight out of shared buffers once the buffer has completed, and
     * the next encoder in the buffer reads what this one produced - neither
     * edge is a dispatch-to-dispatch edge inside one encoder, which is why
     * rearranging those never moved the answer. */
    [_enc barrierAfterStages:MTLStageAll
               beforeQueueStages:MTLStageAll
               visibilityOptions:ds4_mtl4_visibility()];
    [_enc endEncoding];
}

- (MTLDispatchType)dispatchType { return MTLDispatchTypeConcurrent; }
- (NSString *)label { return _enc.label; }
- (void)setLabel:(NSString *)label { _enc.label = label; }

@end

/* ---- the pool ---------------------------------------------------------- */

/* DS4_METAL_V41_MTL4: 1 is the Metal 4 backend, 2 also sends greedy decode
 * through the verify executor. */
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


/* Whether a Metal 4 command buffer is open and taking the engine's work. */
static int ds4_mtl4_open(void) {
    if (@available(macOS 26.0, *)) return g_mtl4_cur != nil;
    return 0;
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
    MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
    d.maxBufferBindCount = DS4_MTL4_MAX_BINDINGS;
    /* A new Metal 3 encoder starts with nil bindings; without this the unset
     * slots of a table are undefined, which is the opposite of the backend
     * being replaced. The table is snapshotted at each dispatch, so one table
     * serves the whole run - but it has to start clean. */
    d.initializeBindings = YES;
    NSError *err = nil;
    g_mtl4_table = [g_device newArgumentTableWithDescriptor:d error:&err];
    if (!g_mtl4_table) {
        fprintf(stderr, "ds4: Metal 4 argument table: %s\n", err.description.UTF8String);
        g_mtl4_queue = nil;
        return 0;
    }
    g_mtl4_pool = [NSMutableArray array];
    for (unsigned i = 0; i < DS4_MTL4_PIPE; i++) {
        DS4MTL4Buffer *b = [DS4MTL4Buffer new];
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
    for (DS4MTL4Buffer *b in g_mtl4_pool) {
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
#define DS4_MTL4_FB_RING 256u
typedef struct {
    _Atomic uint64_t stamp;     /* the stamp this slot last reported */
    _Atomic int failed;
    double gpu_ms;              /* GPUEndTime - GPUStartTime, published by stamp */
} ds4_mtl4_fb_slot;
static ds4_mtl4_fb_slot g_mtl4_fb[DS4_MTL4_FB_RING];
static _Atomic unsigned long long g_mtl4_fb_received, g_mtl4_fb_errors;

/* Test hook: the next feedback to arrive reports a failure. The GPU cannot be
 * made to fail on demand, so this is where the host path after a real failure
 * is exercised - the wait, end_commands, the executor, the request. */
static _Atomic int g_mtl4_inject;
void ds4_gpu_mtl4_inject_failure(void) { atomic_store(&g_mtl4_inject, 1); }

static void ds4_mtl4_fb_note(uint64_t stamp, NSError *error, double gpu_ms) {
    ds4_mtl4_fb_slot *slot = &g_mtl4_fb[stamp % DS4_MTL4_FB_RING];
    slot->gpu_ms = gpu_ms > 0.0 ? gpu_ms : 0.0;
    if (!error && atomic_exchange(&g_mtl4_inject, 0))
        error = [NSError errorWithDomain:@"ds4.injected" code:1
                                userInfo:@{NSLocalizedDescriptionKey: @"injected failure"}];
    if (error) {
        fprintf(stderr, "ds4: Metal 4 command buffer %llu failed: %s\n",
                (unsigned long long)stamp, error.description.UTF8String);
        atomic_fetch_add(&g_mtl4_fb_errors, 1u);
    }
    atomic_store(&slot->failed, error ? 1 : 0);
    atomic_store(&slot->stamp, stamp);       /* last: publishes `failed` */
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
    const BOOL ok = [g_mtl4_event waitUntilSignaledValue:value timeoutMS:600000];
    if (!ok) {
        fprintf(stderr, "ds4: Metal 4 command buffer timed out\n");
        return 0;
    }
    int good = 1;
    for (uint64_t st = g_mtl4_done + 1u; st <= value; st++) {
        ds4_mtl4_fb_slot *slot = &g_mtl4_fb[st % DS4_MTL4_FB_RING];
        /* The feedback is delivered on its own queue; the event can be seen
         * first. It follows within microseconds, and a usleep(10) here slept
         * tens of them per segment - about seventeen times a token - so the
         * first few milliseconds are spun rather than slept. */
        if (atomic_load(&slot->stamp) != st) {
            const uint64_t t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            uint64_t t = t0;
            while (atomic_load(&slot->stamp) != st && t - t0 < 5000000ull) {
                __builtin_arm_yield();
                t = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            }
            for (int nap = 0; atomic_load(&slot->stamp) != st && nap < 500000; nap++)
                usleep(10);
            g_mtl4_fb_late++;
            g_mtl4_fb_late_ns += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t0;
        }
        if (atomic_load(&slot->stamp) != st) {
            fprintf(stderr, "ds4: Metal 4 command buffer %llu completed without feedback\n",
                    (unsigned long long)st);
            return 0;
        }
        if (atomic_load(&slot->failed)) good = 0;
        g_gpu_span_accum_ms += slot->gpu_ms;
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
static DS4MTL4Buffer *ds4_mtl4_take(void) {
    for (DS4MTL4Buffer *b in g_mtl4_pool)
        if (b != g_mtl4_cur && b.stamp <= g_mtl4_done) return b;
    /* Every buffer is still out. Wait for the oldest thing there is, which is
     * everything: the pipeline is two deep and the pool is four, so this is a
     * guard rather than a path. */
    if (!ds4_mtl4_wait_all()) return nil;
    for (DS4MTL4Buffer *b in g_mtl4_pool)
        if (b != g_mtl4_cur && b.stamp <= g_mtl4_done) return b;
    return nil;
}

static int ds4_mtl4_begin(void) {
    if (@available(macOS 26.0, *)) {
        if (!ds4_mtl4_on()) return 0;
        if (g_mtl4_cur) return 1;
        if (!ds4_mtl4_ready()) return 0;
        DS4MTL4Buffer *b = ds4_mtl4_take();
        if (!b) return 0;
        [b.allocator reset];
        [b.cb beginCommandBufferWithAllocator:b.allocator];
        b.scratchUsed = 0;
        b.stamp = 0;
        b.used = NO;
        [b.encoders removeAllObjects];
        g_mtl4_cur = b;
        /* 0 is never a live generation, so a zeroed memo entry never hits. */
        if (++g_mtl4_gen == 0) ++g_mtl4_gen;
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
    if (!g_mtl4_enc) return;
    [g_mtl4_enc endEncoding];
    /* Held until the buffer completes: the GPU reads its bindings through
     * objects this owns. */
    [g_mtl4_cur.encoders addObject:g_mtl4_enc];
    g_mtl4_enc = nil;
}

/* End the open buffer and send it, without waiting. */
static int ds4_mtl4_commit(void) {
    if (@available(macOS 26.0, *)) {
        if (!g_mtl4_cur) return 0;
        DS4MTL4Buffer *b = g_mtl4_cur;
        ds4_mtl4_close_encoder();
        /* Residency is declared once the set is complete. Declaring it before
         * encoding looked tidier and was empty - the scratch does not exist
         * until the first setBytes - so the buffer carrying every kernel's
         * constant arguments was never made resident. */
        [b.resset commit];
        [b.cb useResidencySet:b.resset];
        {
            unsigned long long held = 0;
            for (DS4MTL4Buffer *p in g_mtl4_pool) held += p.resBytes;
            if (held > g_mtl4_res_peak) g_mtl4_res_peak = held;
        }
        [b.cb endCommandBuffer];
        id<MTL4CommandBuffer> one = b.cb;
        const uint64_t stamp = g_mtl4_stamp + 1u;
        MTL4CommitOptions *opts = [MTL4CommitOptions new];
        [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) {
            ds4_mtl4_fb_note(stamp, fb.error, (fb.GPUEndTime - fb.GPUStartTime) * 1000.0);
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
        DS4MTL4Buffer *b = g_mtl4_cur;
        ds4_mtl4_close_encoder();
        [b.cb endCommandBuffer];
        [b.encoders removeAllObjects];
        [b.cb beginCommandBufferWithAllocator:b.allocator];
        b.scratchUsed = 0;
        b.stamp = 0;
        b.used = NO;
    }
}

API_AVAILABLE(macos(26.0))
static id<DS4Encoder> ds4_mtl4_encoder(void) {
    if (!g_mtl4_cur) return nil;
    /* The engine closes and reopens the batch encoder inside a segment - a
     * concurrent section does it twice a layer - so an encoder that has been
     * ended must not be handed back. */
    if (g_mtl4_enc && !g_mtl4_enc.ended) return (id<DS4Encoder>)g_mtl4_enc;
    if (g_mtl4_enc) ds4_mtl4_close_encoder();
    id<MTL4ComputeCommandEncoder> e = [g_mtl4_cur.cb computeCommandEncoder];
    if (!e) {
        fprintf(stderr, "ds4: Metal 4 encoder unavailable\n");
        g_mtl4_encode_error = 1;
        return nil;
    }
    g_mtl4_cur.used = YES;
    [e setArgumentTable:g_mtl4_table];
    /* Metal 3 serialises one compute encoder after another; Metal 4 does not,
     * and an encoder opened after another in the same buffer runs beside it
     * unless told otherwise. This is what the queue-stage barrier is for: it
     * orders against work outside this encoder, which is the previous
     * encoder's. Without it the answer came out wrong rather than late. */
    [e barrierAfterQueueStages:MTLStageAll
                  beforeStages:MTLStageAll
             visibilityOptions:ds4_mtl4_visibility()];
    /* A fresh Metal 3 encoder starts with nil bindings, and the engine relies
     * on nothing carrying over. The table is shared, so it is cleared here. */
    for (NSUInteger i = 0; i < DS4_MTL4_MAX_BINDINGS; i++)
        [g_mtl4_table setAddress:0 atIndex:i];
    g_mtl4_enc = [DS4MTL4Encoder new];
    g_mtl4_enc.enc = e;
    g_mtl4_enc.table = g_mtl4_table;
    g_mtl4_enc.owner = g_mtl4_cur;
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
        static int checked, serial;
        if (!checked) {
            const char *e = getenv("DS4_METAL_V41_MTL4_SERIAL");
            checked = 1;
            serial = e && e[0] ? atoi(e) != 0 : 0;
        }
        g_mtl4_enc.barrierEveryDispatch = serial || !g_batch_encoder_concurrent;
    }
    return (id<DS4Encoder>)g_mtl4_enc;
}

/* Whether the Metal 4 backend took this copy. The engine falls back to its
 * blit encoder when it did not. */
API_AVAILABLE(macos(26.0))
static int ds4_mtl4_copy(id<MTLBuffer> src, NSUInteger soff,
                         id<MTLBuffer> dst, NSUInteger doff, NSUInteger size) {
    if (!g_mtl4_cur) return 0;
    id<DS4Encoder> e = ds4_mtl4_encoder();
    if (!e) return 0;
    if (!src || !dst) {
        g_mtl4_encode_error = 1;
        return 0;
    }
    [(DS4MTL4Encoder *)e copyFromBuffer:src sourceOffset:soff
                               toBuffer:dst destinationOffset:doff size:size];
    return 1;
}

/* ---- what the engine calls --------------------------------------------- */

/* The gated region is a span of the token the engine marks; with the whole
 * batch already on this backend there is no seam to drain at its edges, so
 * entering it only says so. */
void ds4_gpu_mtl4_region_begin(void) {
    if (!ds4_mtl4_on()) return;
    g_mtl4_region = 1;
}

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
        for (DS4MTL4Buffer *b in g_mtl4_pool) { entries += b.resident.count; bytes += b.resBytes; }
    out[3] = entries;
    out[4] = bytes;
    out[5] = g_mtl4_res_adds;
    out[6] = g_mtl4_res_releases;
    out[7] = g_mtl4_res_peak;
}

/* Leaving the region waits for what the region committed and leaves the open
 * batch open: the token goes on encoding into it, as it does on Metal 3,
 * where the region's edge is not a drain either. */
int ds4_gpu_mtl4_region_end(void) {
    if (!ds4_mtl4_on()) return 1;
    g_mtl4_region = 0;
    return ds4_mtl4_wait_all();
}

/* ---- the batch, for ds4_metal.m's begin/flush/end ---------------------- */

/* The buffer the engine's batch is: what g_batch_cb holds under this backend. */
static id ds4_mtl4_batch_identity(void) {
    if (@available(macOS 26.0, *)) return g_mtl4_cur;
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

/* ---- the queue keepalive ----------------------------------------------- */

/* The keepalive exists because a queue left idle for about three seconds
 * loses the process's GPU mappings and the next buffer on it starts 600-800
 * ms late, and work on other queues does not prevent it. With this backend
 * selected the queue doing the work is this one, so the keepalive's kernel
 * goes here. It runs on its own thread, so it has its own allocator, buffer,
 * table and residency, and its own event: nothing it does touches the
 * engine's pool or stamps. Returns 0 if it could not submit. */
int ds4_mtl4_keepalive_once(id<MTLComputePipelineState> pipeline, id<MTLBuffer> scratch,
                            id<MTLBuffer> args) API_AVAILABLE(macos(26.0));
int ds4_mtl4_keepalive_once(id<MTLComputePipelineState> pipeline, id<MTLBuffer> scratch,
                            id<MTLBuffer> args) {
    static id<MTL4CommandAllocator> alloc;
    static id<MTL4CommandBuffer> cb;
    static id<MTL4ArgumentTable> table;
    static id<MTLResidencySet> res;
    static id<MTLSharedEvent> ev;
    static uint64_t sent;
    id<MTL4CommandQueue> q = g_mtl4_queue;
    if (!q || !pipeline || !scratch || !args) return 0;
    if (!alloc) {
        alloc = [g_device newCommandAllocator];
        cb = [g_device newCommandBuffer];
        MTL4ArgumentTableDescriptor *d = [MTL4ArgumentTableDescriptor new];
        d.maxBufferBindCount = 2;
        d.initializeBindings = YES;
        table = [g_device newArgumentTableWithDescriptor:d error:nil];
        MTLResidencySetDescriptor *rd = [MTLResidencySetDescriptor new];
        rd.label = @"ds4_mtl4_keepalive";
        res = [g_device newResidencySetWithDescriptor:rd error:nil];
        ev = [g_device newSharedEvent];
        if (!alloc || !cb || !table || !res || !ev) { alloc = nil; return 0; }
        [res addAllocation:scratch];
        [res addAllocation:args];
        [res commit];
    }
    /* The allocator is reset only once the previous kernel has run. */
    if (sent && ![ev waitUntilSignaledValue:sent timeoutMS:10000]) return 0;
    [alloc reset];
    [cb beginCommandBufferWithAllocator:alloc];
    [cb useResidencySet:res];
    id<MTL4ComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setArgumentTable:table];
    [e setComputePipelineState:pipeline];
    [table setAddress:scratch.gpuAddress atIndex:0];
    [table setAddress:args.gpuAddress atIndex:1];
    [e dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [e endEncoding];
    [cb endCommandBuffer];
    id<MTL4CommandBuffer> one = cb;
    [q commit:&one count:1];
    [q signalEvent:ev value:++sent];
    return 1;
}

#endif  /* macOS 26 SDK */
#endif  /* DS4_METAL_MTL4_H */
