/*
 * The command buffer and the encoder, as the engine holds them.
 *
 * The engine used to hold an id<MTLCommandBuffer> and an encoder that answered
 * MTLComputeCommandEncoder's selectors, and the Metal 4 backend dressed itself
 * up as both: its pool buffer was cast to a command buffer, and a shim object
 * took the encoder's messages and turned them into argument-table writes in
 * call order. What the engine meant by a dispatch - which bytes it reads and
 * which it writes - was nowhere, so the only safe order was the one the calls
 * came in.
 *
 * Here both are plain handles with a C interface:
 *
 *   DS4Cmd   a Metal 3 command buffer (m3), or a Metal 4 command context (m4)
 *   DS4Enc   a Metal 3 compute encoder (m3), or the context's recorder (m4)
 *
 * ds4_enc_* calls go straight to the Metal 3 encoder, or to the Metal 4
 * context the handle names (ds4_metal_mtl4.h).
 */
#ifndef DS4_METAL_CMD_H
#define DS4_METAL_CMD_H

@class DS4M4Ctx;

#ifdef DS4_CMD_TYPECHECK
/* A build that only type-checks: as a C struct, an id<MTLCommandBuffer>
 * passed where a DS4Cmd is expected is an error rather than silence. */
typedef struct DS4Cmd { id<MTLCommandBuffer> m3; __unsafe_unretained DS4M4Ctx *m4; } DS4Cmd;
#else
@interface DS4Cmd : NSObject {
@public
    id<MTLCommandBuffer> m3;
    __unsafe_unretained DS4M4Ctx *m4;
}
@end
@implementation DS4Cmd
@end
#endif

#ifdef DS4_CMD_TYPECHECK
typedef struct DS4Enc {

    id<MTLComputeCommandEncoder> m3;
    /* Metal 4: the context this records into, and its encoder and table. */
    __unsafe_unretained DS4M4Ctx *m4;
    id m4enc;                   /* id<MTL4ComputeCommandEncoder> */
    id m4table;                 /* id<MTL4ArgumentTable> */
    /* The engine ended its section; what it encodes next is ordered after. */
    BOOL ended;
    /* The Metal 4 encoder itself has ended; only the commit does that. */
    BOOL finished;
    /* A barrier after every dispatch: a serial section. */
    BOOL serial;
    BOOL concurrent;
} DS4Enc;
#else
@interface DS4Enc : NSObject {
@public
    id<MTLComputeCommandEncoder> m3;
    /* Metal 4: the context this records into, and its encoder and table. */
    __unsafe_unretained DS4M4Ctx *m4;
    id m4enc;                   /* id<MTL4ComputeCommandEncoder> */
    id m4table;                 /* id<MTL4ArgumentTable> */
    /* The engine ended its section; what it encodes next is ordered after. */
    BOOL ended;
    /* The Metal 4 encoder itself has ended; only the commit does that. */
    BOOL finished;
    /* A barrier after every dispatch: a serial section. */
    BOOL serial;
    BOOL concurrent;
}
@end
@implementation DS4Enc
@end
#endif

/* The Metal 4 side, in ds4_metal_mtl4.h. */
static void ds4_m4_pso(DS4Enc *e, id<MTLComputePipelineState> pso);
static void ds4_m4_buf(DS4Enc *e, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index);
static void ds4_m4_bytes(DS4Enc *e, const void *bytes, NSUInteger length, NSUInteger index);
static void ds4_m4_tgmem(DS4Enc *e, NSUInteger length, NSUInteger index);
static void ds4_m4_use(DS4Enc *e, id<MTLResource> resource, MTLResourceUsage usage);
static void ds4_m4_engine_barrier(DS4Enc *e);
static void ds4_m4_dispatch(DS4Enc *e, int kind, MTLSize a, MTLSize b,
                            id<MTLBuffer> indirect, NSUInteger indirect_offset);
static void ds4_m4_end(DS4Enc *e);

enum { DS4_M4_NODE_GROUPS = 0, DS4_M4_NODE_THREADS, DS4_M4_NODE_INDIRECT };

static inline void ds4_enc_pso(DS4Enc *e, id<MTLComputePipelineState> pso) {
    if (g_enc_trace_fp)
        ds4_enc_trace("pso %u %s\n", ds4_enc_trace_pso_id(pso),
                      pso.label ? pso.label.UTF8String : "-");
    if (e->m3) { [e->m3 setComputePipelineState:pso]; return; }
    ds4_m4_pso(e, pso);
}

static inline void ds4_enc_buf(DS4Enc *e, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
    /* The address, not the object: the question a trace answers is whether a
     * dispatch ends up reading the same bytes on both backends. */
    if (g_enc_trace_fp)
        ds4_enc_trace("buf %lu b%u+%lu\n", (unsigned long)index,
                      ds4_enc_trace_buf_id(buffer), (unsigned long)(buffer ? offset : 0));
    if (e->m3) { [e->m3 setBuffer:buffer offset:offset atIndex:index]; return; }
    ds4_m4_buf(e, buffer, offset, index);
}

static inline void ds4_enc_bytes(DS4Enc *e, const void *bytes, NSUInteger length, NSUInteger index) {
    if (g_enc_trace_fp)
        ds4_enc_trace("byt %lu %lu %llx\n", (unsigned long)index, (unsigned long)length,
                      (unsigned long long)ds4_enc_trace_hash(bytes, length));
    if (e->m3) { [e->m3 setBytes:bytes length:length atIndex:index]; return; }
    ds4_m4_bytes(e, bytes, length, index);
}

static inline void ds4_enc_tgmem(DS4Enc *e, NSUInteger length, NSUInteger index) {
    if (g_enc_trace_fp)
        ds4_enc_trace("tgm %lu %lu\n", (unsigned long)index, (unsigned long)length);
    if (e->m3) { [e->m3 setThreadgroupMemoryLength:length atIndex:index]; return; }
    ds4_m4_tgmem(e, length, index);
}

static inline void ds4_enc_use(DS4Enc *e, id<MTLResource> resource, MTLResourceUsage usage) {
    if (e->m3) { [e->m3 useResource:resource usage:usage]; return; }
    ds4_m4_use(e, resource, usage);
}

static inline void ds4_enc_uses(DS4Enc *e, const id<MTLResource> *resources, NSUInteger count,
                                MTLResourceUsage usage) {
    if (e->m3) { [e->m3 useResources:resources count:count usage:usage]; return; }
    for (NSUInteger i = 0; i < count; i++) ds4_m4_use(e, resources[i], usage);
}

static inline void ds4_enc_barrier_scope(DS4Enc *e, MTLBarrierScope scope) {
    if (g_enc_trace_fp) ds4_enc_trace("bar %lu\n", (unsigned long)scope);
    if (e->m3) { [e->m3 memoryBarrierWithScope:scope]; return; }
    ds4_m4_engine_barrier(e);
}

static inline void ds4_enc_barrier_resources(DS4Enc *e, const id<MTLResource> *resources,
                                             NSUInteger count) {
    if (g_enc_trace_fp) ds4_enc_trace("barr %lu\n", (unsigned long)count);
    if (e->m3) { [e->m3 memoryBarrierWithResources:resources count:count]; return; }
    ds4_m4_engine_barrier(e);
}

static inline void ds4_enc_dispatch_raw(DS4Enc *e, MTLSize tg, MTLSize tpt) {
    if (g_enc_trace_fp)
        ds4_enc_trace("dtg %lu.%lu.%lu %lu.%lu.%lu\n",
                      (unsigned long)tg.width, (unsigned long)tg.height, (unsigned long)tg.depth,
                      (unsigned long)tpt.width, (unsigned long)tpt.height, (unsigned long)tpt.depth);
    if (e->m3) { [e->m3 dispatchThreadgroups:tg threadsPerThreadgroup:tpt]; return; }
    ds4_m4_dispatch(e, DS4_M4_NODE_GROUPS, tg, tpt, nil, 0);
}

static inline void ds4_enc_dispatch_threads(DS4Enc *e, MTLSize threads, MTLSize tpt) {
    if (g_enc_trace_fp)
        ds4_enc_trace("dth %lu.%lu.%lu %lu.%lu.%lu\n",
                      (unsigned long)threads.width, (unsigned long)threads.height,
                      (unsigned long)threads.depth,
                      (unsigned long)tpt.width, (unsigned long)tpt.height, (unsigned long)tpt.depth);
    if (e->m3) { [e->m3 dispatchThreads:threads threadsPerThreadgroup:tpt]; return; }
    ds4_m4_dispatch(e, DS4_M4_NODE_THREADS, threads, tpt, nil, 0);
}

static inline void ds4_enc_dispatch_indirect(DS4Enc *e, id<MTLBuffer> indirect, NSUInteger offset,
                                             MTLSize tpt) {
    if (g_enc_trace_fp)
        ds4_enc_trace("did b%u+%lu %lu.%lu.%lu\n",
                      ds4_enc_trace_buf_id(indirect), (unsigned long)offset,
                      (unsigned long)tpt.width, (unsigned long)tpt.height, (unsigned long)tpt.depth);
    if (e->m3) {
        [e->m3 dispatchThreadgroupsWithIndirectBuffer:indirect indirectBufferOffset:offset
                                threadsPerThreadgroup:tpt];
        return;
    }
    ds4_m4_dispatch(e, DS4_M4_NODE_INDIRECT, MTLSizeMake(0, 0, 0), tpt, indirect, offset);
}

static inline void ds4_enc_end(DS4Enc *e) {
    if (g_enc_trace_fp) ds4_enc_trace("end\n");
    if (e->m3) { [e->m3 endEncoding]; return; }
    ds4_m4_end(e);
}

/* Counter sampling is a Metal 3 encoder facility; see ds4_metal_mtl4.h for
 * what the Metal 4 context does instead. */
static void ds4_m4_sample_counters(DS4Enc *e);
static inline void ds4_enc_sample_counters(DS4Enc *e, id<MTLCounterSampleBuffer> sb,
                                           NSUInteger index, BOOL barrier) {
    if (e->m3) { [e->m3 sampleCountersInBuffer:sb atSampleIndex:index withBarrier:barrier]; return; }
    ds4_m4_sample_counters(e);
}

static inline MTLDispatchType ds4_enc_dispatch_type(DS4Enc *e) {
    if (e->m3) return e->m3.dispatchType;
    return e->concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
}

static inline DS4Enc *ds4_enc_wrap_m3(id<MTLComputeCommandEncoder> m3) {
    if (!m3) return nil;
    DS4Enc *e = [DS4Enc new];
    e->m3 = m3;
    if (ds4_enc_trace_file()) ds4_enc_trace("enc\n");
    return e;
}

static inline DS4Cmd *ds4_cmd_wrap_m3(id<MTLCommandBuffer> m3) {
    if (!m3) return nil;
    DS4Cmd *c = [DS4Cmd new];
    c->m3 = m3;
    return c;
}

#endif  /* DS4_METAL_CMD_H */
