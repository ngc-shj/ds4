/*
 * What a backend was told, written down, so two backends can be compared
 * without guessing which of them is wrong.
 *
 * Ordering and visibility have been rearranged in every shape available and
 * the answer does not move, which leaves what the dispatches read. That is
 * not a thing to reason about: it is a thing to record. This wraps whatever
 * encoder the engine is handed - the Metal 3 encoder itself, or the Metal 4
 * one - and writes one line a call, in encode order, with addresses rather
 * than object identities so the two runs are comparable.
 *
 * DS4_METAL_V41_ENC_TRACE=<path> turns it on. It is off otherwise and costs
 * a branch. The file is closed after DS4_ENC_TRACE_MAX lines so a long run
 * does not produce a file nobody can diff.
 */
#ifndef DS4_METAL_ENC_TRACE_H
#define DS4_METAL_ENC_TRACE_H

#define DS4_ENC_TRACE_MAX 400000u

static FILE *g_enc_trace_fp;
static unsigned long long g_enc_trace_lines;
static int g_enc_trace_tried;

static FILE *ds4_enc_trace_file(void) {
    if (!g_enc_trace_tried) {
        const char *p = getenv("DS4_METAL_V41_ENC_TRACE");
        g_enc_trace_tried = 1;
        if (p && p[0]) g_enc_trace_fp = fopen(p, "wb");
    }
    return g_enc_trace_fp;
}

static void ds4_enc_trace(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void ds4_enc_trace(const char *fmt, ...) {
    FILE *f = ds4_enc_trace_file();
    if (!f) return;
    if (++g_enc_trace_lines > DS4_ENC_TRACE_MAX) {
        if (g_enc_trace_fp) { fclose(g_enc_trace_fp); g_enc_trace_fp = nil; }
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
}

/* A pipeline has no stable name across runs unless it was labelled, so it is
 * numbered by the order it first appears. Both runs encode the same graph in
 * the same order, so the numbers line up - and where they stop lining up is
 * itself the answer. */
#define DS4_ENC_TRACE_PSOS 512u
static const void *g_enc_trace_pso[DS4_ENC_TRACE_PSOS];
static unsigned g_enc_trace_n_pso;

static unsigned ds4_enc_trace_pso_id(id<MTLComputePipelineState> s) {
    const void *k = (__bridge const void *)s;
    for (unsigned i = 0; i < g_enc_trace_n_pso; i++)
        if (g_enc_trace_pso[i] == k) return i;
    if (g_enc_trace_n_pso < DS4_ENC_TRACE_PSOS) {
        g_enc_trace_pso[g_enc_trace_n_pso] = k;
        return g_enc_trace_n_pso++;
    }
    return DS4_ENC_TRACE_PSOS;
}

/* A buffer is named by the order it first appears, not by its address: the
 * addresses are not the same from run to run, and a diff of two runs is the
 * whole point. The offset is kept apart from the name so that two bindings
 * into the same buffer still compare as such. */
#define DS4_ENC_TRACE_BUFS 4096u
static const void *g_enc_trace_buf[DS4_ENC_TRACE_BUFS];
static unsigned g_enc_trace_n_buf;

static unsigned ds4_enc_trace_buf_id(id<MTLBuffer> b) {
    if (!b) return 0;
    const void *k = (__bridge const void *)b;
    for (unsigned i = 0; i < g_enc_trace_n_buf; i++)
        if (g_enc_trace_buf[i] == k) return i + 1u;
    if (g_enc_trace_n_buf < DS4_ENC_TRACE_BUFS) {
        g_enc_trace_buf[g_enc_trace_n_buf] = k;
        return ++g_enc_trace_n_buf;
    }
    return DS4_ENC_TRACE_BUFS + 1u;
}

static uint64_t ds4_enc_trace_hash(const void *p, NSUInteger n) {
    const unsigned char *b = (const unsigned char *)p;
    uint64_t h = 1469598103934665603ull;
    for (NSUInteger i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

@interface DS4TraceEncoder : NSObject <DS4Encoder>
@property (nonatomic, strong) id<DS4Encoder> inner;
@end

@implementation DS4TraceEncoder

- (void)setComputePipelineState:(id<MTLComputePipelineState>)state {
    ds4_enc_trace("pso %u %s\n", ds4_enc_trace_pso_id(state),
                  state.label ? state.label.UTF8String : "-");
    [self.inner setComputePipelineState:state];
}

- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
    /* The address, not the object: Metal 3 binds a buffer and an offset and
     * Metal 4 binds a GPU address, and the question is whether the dispatch
     * ends up reading the same bytes. */
    ds4_enc_trace("buf %lu b%u+%lu\n", (unsigned long)index,
                  ds4_enc_trace_buf_id(buffer), (unsigned long)(buffer ? offset : 0));
    [self.inner setBuffer:buffer offset:offset atIndex:index];
}

- (void)setBytes:(const void *)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
    ds4_enc_trace("byt %lu %lu %llx\n", (unsigned long)index, (unsigned long)length,
                  (unsigned long long)ds4_enc_trace_hash(bytes, length));
    [self.inner setBytes:bytes length:length atIndex:index];
}

- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
    ds4_enc_trace("tgm %lu %lu\n", (unsigned long)index, (unsigned long)length);
    [self.inner setThreadgroupMemoryLength:length atIndex:index];
}

- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage {
    [self.inner useResource:resource usage:usage];
}

- (void)useResources:(const id<MTLResource> *)resources count:(NSUInteger)count
               usage:(MTLResourceUsage)usage {
    [self.inner useResources:resources count:count usage:usage];
}

- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
    ds4_enc_trace("bar %lu\n", (unsigned long)scope);
    [self.inner memoryBarrierWithScope:scope];
}

- (void)memoryBarrierWithResources:(const id<MTLResource> *)resources count:(NSUInteger)count {
    ds4_enc_trace("barr %lu\n", (unsigned long)count);
    [self.inner memoryBarrierWithResources:resources count:count];
}

- (void)dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {
    ds4_enc_trace("dtg %lu.%lu.%lu %lu.%lu.%lu\n",
                  (unsigned long)tg.width, (unsigned long)tg.height, (unsigned long)tg.depth,
                  (unsigned long)tpt.width, (unsigned long)tpt.height, (unsigned long)tpt.depth);
    [self.inner dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
}

- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirectBuffer
                          indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)tpt {
    ds4_enc_trace("did b%u+%lu %lu.%lu.%lu\n",
                  ds4_enc_trace_buf_id(indirectBuffer), (unsigned long)offset,
                  (unsigned long)tpt.width, (unsigned long)tpt.height, (unsigned long)tpt.depth);
    [self.inner dispatchThreadgroupsWithIndirectBuffer:indirectBuffer
                                  indirectBufferOffset:offset
                                 threadsPerThreadgroup:tpt];
}

- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)tpt {
    ds4_enc_trace("dth %lu.%lu.%lu %lu.%lu.%lu\n",
                  (unsigned long)threads.width, (unsigned long)threads.height,
                  (unsigned long)threads.depth,
                  (unsigned long)tpt.width, (unsigned long)tpt.height, (unsigned long)tpt.depth);
    [self.inner dispatchThreads:threads threadsPerThreadgroup:tpt];
}

- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)sampleBuffer
                 atSampleIndex:(NSUInteger)sampleIndex
                   withBarrier:(BOOL)barrier {
    [self.inner sampleCountersInBuffer:sampleBuffer atSampleIndex:sampleIndex withBarrier:barrier];
}

- (void)endEncoding {
    ds4_enc_trace("end\n");
    [self.inner endEncoding];
}

- (MTLDispatchType)dispatchType { return self.inner.dispatchType; }
- (NSString *)label { return self.inner.label; }
- (void)setLabel:(NSString *)label { self.inner.label = label; }

@end

static id<DS4Encoder> ds4_enc_trace_wrap(id<DS4Encoder> enc) {
    if (!enc || !ds4_enc_trace_file()) return enc;
    DS4TraceEncoder *t = [DS4TraceEncoder new];
    t.inner = enc;
    ds4_enc_trace("enc\n");
    return (id<DS4Encoder>)t;
}

#endif  /* DS4_METAL_ENC_TRACE_H */
