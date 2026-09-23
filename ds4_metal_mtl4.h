/*
 * The Metal 4 backend: a token's segments as command buffers on one queue.
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
/* The event value that says this buffer is done. 0 while it is being encoded. */
@property (nonatomic) uint64_t stamp;
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
static uint64_t g_mtl4_stamp;                /* last value handed to a commit */
static uint64_t g_mtl4_done;                 /* last value waited for */
static int g_mtl4_region;                    /* inside the gated region */

static unsigned long long g_mtl4_dispatches, g_mtl4_segments, g_mtl4_copies_m3;

/* Counted rather than assumed: work the engine puts in the Metal 3 batch
 * buffer while a Metal 4 buffer is open runs after it, not in place. The
 * design says there is none; this is what says so about a run. */
void ds4_gpu_mtl4_note_metal3_work(const char *what);
static int ds4_mtl4_open(void);

void ds4_gpu_mtl4_counts(unsigned long long *segments,
                         unsigned long long *dispatches,
                         unsigned long long *m3_work) {
    if (segments) *segments = g_mtl4_segments;
    if (dispatches) *dispatches = g_mtl4_dispatches;
    if (m3_work) *m3_work = g_mtl4_copies_m3;
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

/* Metal 4 tracks no residency implicitly. What the engine marks with
 * useResource, and every buffer it binds, is collected here. The set is
 * long-lived: it holds the model and the expert cache, which do not come and
 * go, so it is committed when it has grown rather than once a segment. */
static id<MTLResidencySet> g_mtl4_residency;
static NSMutableSet *g_mtl4_resident;
static BOOL g_mtl4_resident_dirty;

static void ds4_mtl4_mark_resident(id<MTLResource> r) {
    if (!r) return;
    if (!g_mtl4_resident) g_mtl4_resident = [NSMutableSet set];
    if ([g_mtl4_resident containsObject:r]) return;
    [g_mtl4_resident addObject:r];
    if (!g_mtl4_residency) {
        MTLResidencySetDescriptor *d = [MTLResidencySetDescriptor new];
        NSError *err = nil;
        g_mtl4_residency = [g_device newResidencySetWithDescriptor:d error:&err];
        if (!g_mtl4_residency) {
            fprintf(stderr, "ds4: Metal 4 residency set: %s\n",
                    err.description.UTF8String);
            return;
        }
    }
    [g_mtl4_residency addAllocation:r];
    g_mtl4_resident_dirty = YES;
}

/* Residency sets the engine attaches to the Metal 3 queue. The Metal 4 queue
 * inherits none of them, and what they hold is the model and the expert cache,
 * reached by raw address from the masked kernels - nothing else declares them.
 * A segment without these faults in AGX on its first dispatch. */
static NSMutableArray *g_mtl4_queue_sets;
API_AVAILABLE(macos(26.0)) static void ds4_mtl4_attach_sets(void);

void ds4_gpu_queue_residency_added(id set) {
    if (!set) return;
    if (!g_mtl4_queue_sets) g_mtl4_queue_sets = [NSMutableArray array];
    if (![g_mtl4_queue_sets containsObject:set]) [g_mtl4_queue_sets addObject:set];
    if (@available(macOS 26.0, *)) { if (g_mtl4_queue) ds4_mtl4_attach_sets(); }
}

static void ds4_mtl4_attach_sets(void) {
    static NSUInteger attached;
    if (!g_mtl4_queue) return;
    while (attached < g_mtl4_queue_sets.count) {
        [g_mtl4_queue addResidencySet:g_mtl4_queue_sets[attached]];
        attached++;
    }
}

@implementation DS4MTL4Encoder

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
    [self.enc setComputePipelineState:state];
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 buffer index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    /* setBuffer:nil clears the slot in Metal 3; skipping the call would leave
     * whatever the slot held. */
    if (!buffer) { [self.table setAddress:0 atIndex:index]; return; }
    ds4_mtl4_mark_resident(buffer);
    [self.table setAddress:buffer.gpuAddress + offset atIndex:index];
}

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    if (index >= DS4_MTL4_MAX_BINDINGS) {
        fprintf(stderr, "ds4: Metal 4 bytes index %lu past the table's %u\n",
                (unsigned long)index, DS4_MTL4_MAX_BINDINGS);
        abort();
    }
    DS4MTL4Buffer *b = self.owner;
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
    [self.table setAddress:s.gpuAddress + off atIndex:index];
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
    [self.enc setThreadgroupMemoryLength:length atIndex:index];
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
    [self.enc barrierAfterEncoderStages:MTLStageAll
                    beforeEncoderStages:MTLStageAll
                      visibilityOptions:ds4_mtl4_visibility()];
}

- (void)memoryBarrierWithResources:(const id<MTLResource> *)resources count:(NSUInteger)count {
    for (NSUInteger i = 0; i < count; i++) ds4_mtl4_mark_resident(resources[i]);
    [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {
    g_mtl4_dispatches++;
    [self.enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
    if (self.barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)tpt {
    ds4_mtl4_mark_resident(indirectBuffer);
    g_mtl4_dispatches++;
    [self.enc dispatchThreadgroupsWithIndirectBuffer:indirectBuffer.gpuAddress + offset
                               threadsPerThreadgroup:tpt];
    if (self.barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tpt {
    g_mtl4_dispatches++;
    [self.enc dispatchThreads:threads threadsPerThreadgroup:tpt];
    if (self.barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
    if (self.barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
    [self.enc copyFromBuffer:src sourceOffset:soff
                    toBuffer:dst destinationOffset:doff size:size];
    if (self.barrierEveryDispatch) [self memoryBarrierWithScope:MTLBarrierScopeBuffers];
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
    [self.enc barrierAfterStages:MTLStageAll
               beforeQueueStages:MTLStageAll
               visibilityOptions:ds4_mtl4_visibility()];
    [self.enc endEncoding];
}

- (MTLDispatchType)dispatchType { return MTLDispatchTypeConcurrent; }
- (NSString *)label { return self.enc.label; }
- (void)setLabel:(NSString *)label { self.enc.label = label; }

@end

/* ---- the pool ---------------------------------------------------------- */

static int ds4_mtl4_on(void) {
    static int checked, on;
    if (!checked) {
        const char *e = getenv("DS4_METAL_V41_MTL4");
        checked = 1;
        on = e && e[0] ? atoi(e) != 0 : 0;
    }
    return on;
}

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
        if (!b.allocator || !b.cb || !b.scratch) {
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

/* Wait until every buffer committed up to `value` has completed. One queue in
 * order, so the newest stamp implies the rest. */
static int ds4_mtl4_wait_to(uint64_t value) {
    if (value <= g_mtl4_done) return 1;
    const BOOL ok = [g_mtl4_event waitUntilSignaledValue:value timeoutMS:600000];
    if (!ok) {
        fprintf(stderr, "ds4: Metal 4 command buffer timed out\n");
        return 0;
    }
    g_mtl4_done = value;
    return 1;
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
        if (!g_mtl4_region || !ds4_mtl4_on()) return 0;
        if (g_mtl4_cur) return 1;
        if (!ds4_mtl4_ready()) return 0;
        DS4MTL4Buffer *b = ds4_mtl4_take();
        if (!b) return 0;
        [b.allocator reset];
        [b.cb beginCommandBufferWithAllocator:b.allocator];
        b.scratchUsed = 0;
        b.stamp = 0;
        [b.encoders removeAllObjects];
        g_mtl4_cur = b;
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
        if (g_mtl4_residency && g_mtl4_resident_dirty) {
            [g_mtl4_residency commit];
            g_mtl4_resident_dirty = NO;
        }
        if (g_mtl4_residency) [b.cb useResidencySet:g_mtl4_residency];
        [b.cb endCommandBuffer];
        id<MTL4CommandBuffer> one = b.cb;
        [g_mtl4_queue commit:&one count:1];
        b.stamp = ++g_mtl4_stamp;
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
        return nil;
    }
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
    g_mtl4_enc.barrierEveryDispatch = YES;
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
    [(DS4MTL4Encoder *)e copyFromBuffer:src sourceOffset:soff
                               toBuffer:dst destinationOffset:doff size:size];
    return 1;
}

/* ---- what the engine calls --------------------------------------------- */

/* The gated region is where this backend owns the token. Outside it the
 * ordinary Metal 3 path runs, and the boundary is a drain the engine already
 * performs, so the two never hold work at the same time. */
void ds4_gpu_mtl4_region_begin(void) {
    if (!ds4_mtl4_on()) return;
    if (@available(macOS 26.0, *)) {
        /* A token reaches here with a batch already open and work in it - the
         * embedding and everything ahead of the gated region. Opening a Metal
         * 4 buffer over the top of that would put the region's first segment
         * on a queue that has no ordering against it, and the Metal 3 buffer
         * is committed after, so its work would land after the segment that
         * reads it. Drain first: this is the seam between the two backends,
         * and it is a wait, not a barrier, because nothing else orders two
         * queues. */
        if (g_batch_cb && g_batch_has_work) {
            if (!ds4_gpu_flush_commands()) return;
            if (!ds4_gpu_wait_committed()) return;
        }
        ds4_gpu_close_batch_encoder();
        g_mtl4_region = 1;
        (void)ds4_mtl4_begin();
    }
}

void ds4_gpu_mtl4_region_end(void) {
    if (@available(macOS 26.0, *)) {
        if (g_mtl4_cur) { (void)ds4_mtl4_commit(); }
        (void)ds4_mtl4_wait_all();
        g_mtl4_region = 0;
    }
}

#endif  /* macOS 26 SDK */
#endif  /* DS4_METAL_MTL4_H */
