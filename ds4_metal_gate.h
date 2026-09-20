/*
 * Abort gate: making the dispatches after a failed layer cost nothing.
 *
 * A decode layer can only run its routed experts if every selected expert has
 * an address in the layer's table, and that is known first on the GPU. The
 * host learns it today by committing, waiting and reading back, forty times a
 * token. An `MTLSharedEvent` cannot replace that: the wait is a command,
 * written into the buffer before anyone has looked, so it costs the same on a
 * layer that hits as on one that misses.
 *
 * What is decided after the GPU has looked is the completion of a command
 * buffer. So: submit the segment, let each layer check its own route, and on a
 * miss record it and switch the rest of the segment off. The buffer completes,
 * the host reads the record - completion is an ordering point, so the record
 * is visible - brings the expert in with the loader it already has, and
 * submits a continuation from the failed layer's routed stage.
 *
 * "Switch the rest off" has to mean the dispatches do not run, not that their
 * kernels return early: a few thousand dispatches still cost their issue. So
 * every gated dispatch reads its threadgroup count from a table in device
 * memory, and the layer that aborts zeroes the rest of that table. A dispatch
 * whose indirect arguments are zero launches nothing and writes nothing.
 * `proto/gateprobe.m` checks that a kernel's write to those arguments is seen
 * by dispatches later in the same encoder: 256 dispatches, gate at 40, 100
 * rounds, nothing ran past the gate and nothing before it was lost.
 *
 * Off unless DS4_METAL_V41_ABORT_GATE is set.
 */
#ifndef DS4_METAL_GATE_H
#define DS4_METAL_GATE_H

#include <stdatomic.h>

/* One slot per gated dispatch. A token issues about 2,700. */
#define DS4_GATE_MAX_SLOTS 65536u
#define DS4_GATE_MAX_LAYER 64u
#define DS4_GATE_MISS_WORDS 16u

void ds4_gpu_gate_probe(id<MTLDevice> device);
int  ds4_gpu_gate_layers(void);                 /* 0 when off */
void ds4_gpu_gate_segment_begin(void);
void ds4_gpu_gate_segment_end(void);
void ds4_gpu_gate_layer_begin(uint32_t il);
int  ds4_gpu_gate_encode_validate(const void *selected_tensor, uint32_t layer,
                                  uint32_t n_sel, uint32_t n_total);   /* in ds4_metal.m */
uint32_t ds4_gpu_gate_slots_used(void);
int  ds4_gpu_gate_read_miss(uint32_t *layer, uint32_t *mask, int32_t *ids, uint32_t *n);
unsigned long long ds4_gpu_gate_ungatable(void);
unsigned long long ds4_gpu_gate_slots_encoded_after_abort(void);  /* NOT launches */
unsigned long long ds4_gpu_gate_slots_left_live(void);            /* gate failures */

static int        g_gate_layers;
static id<MTLBuffer> g_gate_grid;
static id<MTLBuffer> g_gate_miss;
static _Atomic uint32_t g_gate_next_slot;
static uint32_t   g_gate_layer_base[DS4_GATE_MAX_LAYER];
static _Atomic unsigned long long g_gate_ungatable;
static _Atomic int g_gate_active;             /* dispatches are being gated now */
static id<MTLComputePipelineState> g_gate_validate_pso;

/* The decode thread is the only one that encodes gated work, so the slot
 * cursor does not have to be shared with anything else. */
int ds4_gpu_gate_layers(void) { return g_gate_layers; }
unsigned long long ds4_gpu_gate_ungatable(void) {
    return atomic_load_explicit(&g_gate_ungatable, memory_order_relaxed);
}
uint32_t ds4_gpu_gate_slots_used(void) {
    return atomic_load_explicit(&g_gate_next_slot, memory_order_relaxed);
}


/* ---- turning a dispatch into a gated one ------------------------------- */

@interface NSObject (DS4GateSwizzle)
- (void)ds4_gate_dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt;
- (void)ds4_gate_dispatchThreads:(MTLSize)th threadsPerThreadgroup:(MTLSize)tpt;
@end

static Class g_gate_hooked_cls[8];
static uint32_t g_gate_n_hooked;

/* Called for every encoder the decode path makes. The class set dedups, so a
 * second class - Metal has more than one - is hooked rather than missed. */
static void ds4_gate_hook_encoder(id enc) {
    if (!enc) return;
    Class c = object_getClass(enc);
    for (uint32_t i = 0; i < g_gate_n_hooked; i++) if (g_gate_hooked_cls[i] == c) return;
    if (g_gate_n_hooked < 8) g_gate_hooked_cls[g_gate_n_hooked++] = c;
    /* Shares the ledger's swizzle, which installs on the class that actually
     * implements the selector and gives that class its own copy of the
     * replacement first. Exchanging against the category's method on NSObject
     * rewrites it for every class in the process. */
    ds4_ledger_swizzle(c, @selector(dispatchThreadgroups:threadsPerThreadgroup:),
                          @selector(ds4_gate_dispatchThreadgroups:threadsPerThreadgroup:));
    ds4_ledger_swizzle(c, @selector(dispatchThreads:threadsPerThreadgroup:),
                          @selector(ds4_gate_dispatchThreads:threadsPerThreadgroup:));
}

/* Takes the next slot, writes the real grid into it, and returns the byte
 * offset the dispatch should read its arguments from. */
static int ds4_gate_take_slot(MTLSize tg, NSUInteger *offset) {
    const uint32_t i = atomic_fetch_add_explicit(&g_gate_next_slot, 1u, memory_order_relaxed);
    if (i >= DS4_GATE_MAX_SLOTS) {
        atomic_fetch_add_explicit(&g_gate_ungatable, 1u, memory_order_relaxed);
        return 0;
    }
    uint32_t *grid = (uint32_t *)[g_gate_grid contents];
    grid[3u * i + 0u] = (uint32_t)tg.width;
    grid[3u * i + 1u] = (uint32_t)tg.height;
    grid[3u * i + 2u] = (uint32_t)tg.depth;
    *offset = (NSUInteger)i * 3u * sizeof(uint32_t);
    return 1;
}

@implementation NSObject (DS4GateSwizzle)
- (void)ds4_gate_dispatchThreadgroups:(MTLSize)tg threadsPerThreadgroup:(MTLSize)tpt {
    NSUInteger off = 0;
    if (atomic_load_explicit(&g_gate_active, memory_order_relaxed) &&
        ds4_gate_take_slot(tg, &off)) {
        [(id<MTLComputeCommandEncoder>)self dispatchThreadgroupsWithIndirectBuffer:g_gate_grid
                                                             indirectBufferOffset:off
                                                            threadsPerThreadgroup:tpt];
        return;
    }
    [self ds4_gate_dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
}
- (void)ds4_gate_dispatchThreads:(MTLSize)th threadsPerThreadgroup:(MTLSize)tpt {
    /* Non-uniform dispatch has no indirect form. Where the grid divides evenly
     * the two forms are the same work, so it can still be gated; where it does
     * not, the partial threadgroup would be lost and the call is counted
     * instead. A segment that reports any of these is not gated and its result
     * does not count. */
    if (atomic_load_explicit(&g_gate_active, memory_order_relaxed)) {
        const NSUInteger w = tpt.width ? tpt.width : 1, h = tpt.height ? tpt.height : 1,
                         d = tpt.depth ? tpt.depth : 1;
        NSUInteger off = 0;
        if (th.width % w == 0 && th.height % h == 0 && th.depth % d == 0 &&
            ds4_gate_take_slot(MTLSizeMake(th.width / w, th.height / h, th.depth / d), &off)) {
            [(id<MTLComputeCommandEncoder>)self dispatchThreadgroupsWithIndirectBuffer:g_gate_grid
                                                                 indirectBufferOffset:off
                                                                threadsPerThreadgroup:tpt];
            return;
        }
        atomic_fetch_add_explicit(&g_gate_ungatable, 1u, memory_order_relaxed);
    }
    [self ds4_gate_dispatchThreads:th threadsPerThreadgroup:tpt];
}
@end

/* ---- lifecycle --------------------------------------------------------- */

void ds4_gpu_gate_probe(id<MTLDevice> device) {
    static int probed;
    if (probed) return;
    probed = 1;
    const char *env = getenv("DS4_METAL_V41_ABORT_GATE");
    if (!env || !env[0] || !device) return;
    if (getenv("DS4_METAL_LEDGER") || getenv("DS4_METAL_ENCODER_TIMELINE")) {
        fprintf(stderr, "ds4: the abort gate and the ledger hook the same selector; "
                        "run one at a time. Gate not started.\n");
        return;
    }
    long n = strtol(env, NULL, 10);
    if (n <= 0) return;
    if (n > (long)DS4_GATE_MAX_LAYER) n = DS4_GATE_MAX_LAYER;
    g_gate_grid = [device newBufferWithLength:DS4_GATE_MAX_SLOTS * 3u * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
    g_gate_miss = [device newBufferWithLength:DS4_GATE_MISS_WORDS * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
    if (!g_gate_grid || !g_gate_miss) { g_gate_grid = nil; g_gate_miss = nil; return; }
    g_gate_grid.label = @"ds4_gate_grid";
    g_gate_miss.label = @"ds4_gate_miss";
    g_gate_layers = (int)n;
    fprintf(stderr, "ds4: abort gate on for the first %d layers\n", g_gate_layers);
}

void ds4_gpu_gate_segment_begin(void) {
    if (!g_gate_layers) return;
    atomic_store_explicit(&g_gate_next_slot, 0u, memory_order_relaxed);
    memset([g_gate_miss contents], 0, DS4_GATE_MISS_WORDS * sizeof(uint32_t));
    memset(g_gate_layer_base, 0, sizeof(g_gate_layer_base));
    atomic_store_explicit(&g_gate_active, 1, memory_order_relaxed);
}

void ds4_gpu_gate_segment_end(void) {
    if (!g_gate_layers) return;
    atomic_store_explicit(&g_gate_active, 0, memory_order_relaxed);
}

void ds4_gpu_gate_layer_begin(uint32_t il) {
    if (!g_gate_layers || il >= DS4_GATE_MAX_LAYER) return;
    g_gate_layer_base[il] = atomic_load_explicit(&g_gate_next_slot, memory_order_relaxed);
}

/* How many dispatches were encoded behind the layer that aborted. This is a
 * count of slots taken, NOT of kernels that ran: nothing here observes a
 * launch, and with one gated layer it is zero only because there is nothing
 * behind it. It cannot be used to explain a wrong answer. */
unsigned long long ds4_gpu_gate_slots_encoded_after_abort(void) {
    if (!g_gate_layers || !g_gate_miss) return 0;
    const uint32_t *m = (const uint32_t *)[g_gate_miss contents];
    if (m[0] == 0u) return 0;
    const uint32_t aborted = m[0] - 1u;
    const uint32_t used = atomic_load_explicit(&g_gate_next_slot, memory_order_relaxed);
    const uint32_t base = (aborted + 1u < DS4_GATE_MAX_LAYER && g_gate_layer_base[aborted + 1u])
                          ? g_gate_layer_base[aborted + 1u] : used;
    return used > base ? used - base : 0;
}

/* What the gate is supposed to guarantee, read back after the buffer has
 * completed: every slot behind the aborting layer should be zero. A slot that
 * is not is a dispatch the gate did not switch off - either the validator's
 * write did not reach it, or the host rewrote the slot after the GPU zeroed
 * it. This is observable, and it is the number that may be used to explain a
 * wrong answer. */
unsigned long long ds4_gpu_gate_slots_left_live(void) {
    if (!g_gate_layers || !g_gate_miss || !g_gate_grid) return 0;
    const uint32_t *m = (const uint32_t *)[g_gate_miss contents];
    if (m[0] == 0u) return 0;
    const uint32_t aborted = m[0] - 1u;
    const uint32_t used = atomic_load_explicit(&g_gate_next_slot, memory_order_relaxed);
    const uint32_t base = (aborted + 1u < DS4_GATE_MAX_LAYER && g_gate_layer_base[aborted + 1u])
                          ? g_gate_layer_base[aborted + 1u] : used;
    const uint32_t *grid = (const uint32_t *)[g_gate_grid contents];
    unsigned long long live = 0;
    for (uint32_t i = base; i < used; i++)
        if (grid[3u * i] | grid[3u * i + 1u] | grid[3u * i + 2u]) live++;
    return live;
}

int ds4_gpu_gate_read_miss(uint32_t *layer, uint32_t *mask, int32_t *ids, uint32_t *n) {
    if (!g_gate_layers || !g_gate_miss) return 0;
    const uint32_t *m = (const uint32_t *)[g_gate_miss contents];
    if (m[0] == 0u) return 0;
    if (layer) *layer = m[0] - 1u;
    if (mask) *mask = m[1];
    const uint32_t count = m[2] < DS4_GATE_MISS_WORDS - 3u ? m[2] : DS4_GATE_MISS_WORDS - 3u;
    if (n) *n = count;
    if (ids) for (uint32_t k = 0; k < count; k++) ids[k] = (int32_t)m[3 + k];
    return 1;
}

#endif /* DS4_METAL_GATE_H */
