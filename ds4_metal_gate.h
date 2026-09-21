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
int  ds4_gpu_gate_read_route(uint32_t layer, int32_t *ids, uint32_t *n);
void ds4_gpu_gate_clear_route(void);
unsigned long long ds4_gpu_gate_ungatable(void);
unsigned long long ds4_gpu_gate_slots_encoded_after_abort(void);  /* NOT launches */
unsigned long long ds4_gpu_gate_slots_total(void);   /* slots taken, all segments */
unsigned long long ds4_gpu_gate_slots_left_live(void);            /* gate failures */
void ds4_gpu_gate_print_ungatable(void);
void ds4_gpu_gate_print_sites(unsigned long long tokens);

static int        g_gate_layers;
static id<MTLBuffer> g_gate_grid;
static id<MTLBuffer> g_gate_miss;
static id<MTLBuffer> g_gate_route;     /* {valid, layer, ids[6]} a layer */
static _Atomic uint32_t g_gate_next_slot;
static _Atomic unsigned long long g_gate_slots_total;
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


/* ---- the gated dispatch -------------------------------------------------
 * Every dispatch on the decode path is sent as ds4_dispatch: rather than
 * dispatchThreadgroups:, so the gate is a call the code makes rather than a
 * method the runtime was rewritten behind. With the gate off these forward one
 * message; with it on, each takes a slot in the grid table and goes out
 * indirect, so the layer that aborts can zero the rest.
 */
/* Which call sites issue a dispatch the gate cannot switch off, and with what
 * shape. Unslid addresses, to be symbolised with atos. */
#define DS4_GATE_UNGATABLE_SITES 16u
static struct { uintptr_t caller; uint64_t count; uint32_t w, h, d, tw, th_, td; }
    g_gate_ungatable_sites[DS4_GATE_UNGATABLE_SITES];
static uint32_t g_gate_n_ungatable_sites;
static os_unfair_lock g_gate_site_lock = OS_UNFAIR_LOCK_INIT;

static void ds4_gate_note_ungatable(uintptr_t caller, MTLSize th, MTLSize tpt) {
    const uintptr_t unslid = caller - (uintptr_t)_dyld_get_image_vmaddr_slide(0);
    os_unfair_lock_lock(&g_gate_site_lock);
    for (uint32_t i = 0; i < g_gate_n_ungatable_sites; i++)
        if (g_gate_ungatable_sites[i].caller == unslid) {
            g_gate_ungatable_sites[i].count++;
            os_unfair_lock_unlock(&g_gate_site_lock);
            return;
        }
    if (g_gate_n_ungatable_sites < DS4_GATE_UNGATABLE_SITES) {
        const uint32_t i = g_gate_n_ungatable_sites++;
        g_gate_ungatable_sites[i].caller = unslid;
        g_gate_ungatable_sites[i].count = 1;
        g_gate_ungatable_sites[i].w = (uint32_t)th.width;
        g_gate_ungatable_sites[i].h = (uint32_t)th.height;
        g_gate_ungatable_sites[i].d = (uint32_t)th.depth;
        g_gate_ungatable_sites[i].tw = (uint32_t)tpt.width;
        g_gate_ungatable_sites[i].th_ = (uint32_t)tpt.height;
        g_gate_ungatable_sites[i].td = (uint32_t)tpt.depth;
    }
    os_unfair_lock_unlock(&g_gate_site_lock);
}

/* Where a token's gated dispatches come from. Off unless
 * DS4_METAL_V41_GATE_CENSUS is set: it is a lock and a scan on a path that
 * runs three thousand times a token, so it prices itself into any run that
 * has it on. The count is what a fusion is worth - a call site issuing six
 * dispatches a layer is forty-eight a token, and the indirect form makes each
 * of those 2.25 us of GPU time, not the 0.70 us a plain dispatch costs. */
#define DS4_GATE_SITES 192u
/* Two frames, because the encoder that issues the dispatch is shared: knowing
 * that 759 of them come from the quantize encoder does not say which of its
 * nineteen call sites in a layer they are, and the fusion is chosen per call
 * site. */
static struct { uintptr_t caller, parent; uint64_t count; uint32_t w, h, d, tw, th_, td; }
    g_gate_sites[DS4_GATE_SITES];
static uint32_t g_gate_n_sites;

static int ds4_gate_census_on(void) {
    static int checked, on;
    if (!checked) { checked = 1; on = getenv("DS4_METAL_V41_GATE_CENSUS") != NULL; }
    return on;
}

static void ds4_gate_note_slot(uintptr_t caller, uintptr_t parent,
                               MTLSize tg, MTLSize tpt) {
    const uintptr_t slide = (uintptr_t)_dyld_get_image_vmaddr_slide(0);
    const uintptr_t unslid = caller - slide;
    const uintptr_t punslid = parent ? parent - slide : 0;
    os_unfair_lock_lock(&g_gate_site_lock);
    for (uint32_t i = 0; i < g_gate_n_sites; i++)
        if (g_gate_sites[i].caller == unslid && g_gate_sites[i].parent == punslid) {
            g_gate_sites[i].count++;
            os_unfair_lock_unlock(&g_gate_site_lock);
            return;
        }
    if (g_gate_n_sites < DS4_GATE_SITES) {
        const uint32_t i = g_gate_n_sites++;
        g_gate_sites[i].caller = unslid;
        g_gate_sites[i].parent = punslid;
        g_gate_sites[i].count = 1;
        g_gate_sites[i].w = (uint32_t)tg.width;
        g_gate_sites[i].h = (uint32_t)tg.height;
        g_gate_sites[i].d = (uint32_t)tg.depth;
        g_gate_sites[i].tw = (uint32_t)tpt.width;
        g_gate_sites[i].th_ = (uint32_t)tpt.height;
        g_gate_sites[i].td = (uint32_t)tpt.depth;
    }
    os_unfair_lock_unlock(&g_gate_site_lock);
}

void ds4_gpu_gate_print_sites(unsigned long long tokens) {
    if (!ds4_gate_census_on() || tokens == 0) return;
    os_unfair_lock_lock(&g_gate_site_lock);
    /* Heaviest first, by repeatedly taking the largest remaining count and
     * clearing it. The table is rebuilt from scratch for the next window, so
     * every figure printed belongs to the window named beside it. */
    for (;;) {
        uint32_t best = DS4_GATE_SITES;
        uint64_t best_c = 0;
        for (uint32_t i = 0; i < g_gate_n_sites; i++)
            if (g_gate_sites[i].count > best_c) { best = i; best_c = g_gate_sites[i].count; }
        if (best == DS4_GATE_SITES) break;
        fprintf(stderr, "ds4: gated dispatch site 0x%llx from 0x%llx: %.1f a token, "
                        "groups %ux%ux%u of %ux%ux%u\n",
                (unsigned long long)g_gate_sites[best].caller,
                (unsigned long long)g_gate_sites[best].parent,
                (double)best_c / (double)tokens,
                g_gate_sites[best].w, g_gate_sites[best].h, g_gate_sites[best].d,
                g_gate_sites[best].tw, g_gate_sites[best].th_, g_gate_sites[best].td);
        g_gate_sites[best].count = 0;
    }
    g_gate_n_sites = 0;
    os_unfair_lock_unlock(&g_gate_site_lock);
}

void ds4_gpu_gate_print_ungatable(void) {
    os_unfair_lock_lock(&g_gate_site_lock);
    for (uint32_t i = 0; i < g_gate_n_ungatable_sites; i++)
        fprintf(stderr, "ds4: ungatable dispatch at 0x%llx: %llu times, "
                        "threads %ux%ux%u per group %ux%ux%u\n",
                (unsigned long long)g_gate_ungatable_sites[i].caller,
                (unsigned long long)g_gate_ungatable_sites[i].count,
                g_gate_ungatable_sites[i].w, g_gate_ungatable_sites[i].h,
                g_gate_ungatable_sites[i].d, g_gate_ungatable_sites[i].tw,
                g_gate_ungatable_sites[i].th_, g_gate_ungatable_sites[i].td);
    os_unfair_lock_unlock(&g_gate_site_lock);
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

/* The gate itself: every dispatch on the decode path comes through here, so
 * it is a call the code makes rather than a method the runtime was rewritten
 * behind. With the gate off this is one message; with it on, the dispatch
 * takes a slot in the grid table and goes out indirect, so the layer that
 * aborts can zero the rest. */
static void ds4_gpu_dispatch(id<MTLComputeCommandEncoder> enc, MTLSize tg, MTLSize tpt) {
    NSUInteger off = 0;
    if (g_gate_layers && atomic_load_explicit(&g_gate_active, memory_order_relaxed) &&
        ds4_gate_take_slot(tg, &off)) {
        if (ds4_gate_census_on())
            ds4_gate_note_slot((uintptr_t)__builtin_return_address(0),
                               (uintptr_t)__builtin_return_address(1), tg, tpt);
        [enc dispatchThreadgroupsWithIndirectBuffer:g_gate_grid
                               indirectBufferOffset:off
                              threadsPerThreadgroup:tpt];
        return;
    }
    [enc dispatchThreadgroups:tg threadsPerThreadgroup:tpt];
}

/* Non-uniform dispatch has no indirect form. The decode path has none left -
 * both of its call sites were rounded up to whole threadgroups, their kernels
 * already bounds-checking - so one arriving here while a segment is gated is
 * counted and named rather than left running after an abort. */
static void ds4_gpu_dispatch_threads(id<MTLComputeCommandEncoder> enc, MTLSize th, MTLSize tpt) {
    if (g_gate_layers && atomic_load_explicit(&g_gate_active, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&g_gate_ungatable, 1u, memory_order_relaxed);
        ds4_gate_note_ungatable((uintptr_t)__builtin_return_address(0), th, tpt);
    }
    [enc dispatchThreads:th threadsPerThreadgroup:tpt];
}

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
    g_gate_route = [device newBufferWithLength:DS4_GATE_MAX_LAYER * 8u * sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
    if (!g_gate_grid || !g_gate_miss || !g_gate_route) {
        g_gate_grid = nil; g_gate_miss = nil; g_gate_route = nil; return;
    }
    g_gate_route.label = @"ds4_gate_route_log";
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
    /* The slot counter is per segment; a token is several segments. Carry the
     * total so the indirect form's cost can be priced against a count that was
     * observed rather than one estimated from the call sites. */
    atomic_fetch_add_explicit(&g_gate_slots_total,
                              atomic_load_explicit(&g_gate_next_slot, memory_order_relaxed),
                              memory_order_relaxed);
}

unsigned long long ds4_gpu_gate_slots_total(void) {
    return atomic_load_explicit(&g_gate_slots_total, memory_order_relaxed);
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

/* The route log is per token, not per segment: a continuation re-runs a layer
 * and would otherwise account for it twice. */
int ds4_gpu_gate_read_route(uint32_t layer, int32_t *ids, uint32_t *n) {
    if (!g_gate_layers || !g_gate_route || layer >= DS4_GATE_MAX_LAYER) return 0;
    const uint32_t *slot = (const uint32_t *)[g_gate_route contents] + 8u * layer;
    if (slot[0] == 0u) return 0;
    if (n) *n = 6u;
    if (ids) for (uint32_t k = 0; k < 6u; k++) ids[k] = (int32_t)slot[2u + k];
    return 1;
}

void ds4_gpu_gate_clear_route(void) {
    if (g_gate_route) memset([g_gate_route contents], 0,
                             DS4_GATE_MAX_LAYER * 8u * sizeof(uint32_t));
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
