/*
 * Where the command-buffer envelope goes.
 *
 * A token's 34.78 ms of envelope is the sum of GPUEndTime - GPUStartTime over
 * the buffers a drain waits on. It contains the kernels and the gaps between
 * encoders, and nothing so far separates them. Splitting it by encoder would
 * mean splitting the encoders, which changes the submission shape this is
 * supposed to measure.
 *
 * So: timestamps at dispatch boundaries, taken only where the stage changes -
 * about six a layer rather than one per dispatch - and with barrier NO, so the
 * sampling does not serialise what it is timing. The stage is set by the
 * decode path around the work it already separates.
 *
 * Off unless DS4_METAL_V41_STAGE_SAMPLE is set.
 */
#ifndef DS4_METAL_STAGE_H
#define DS4_METAL_STAGE_H

enum {
    DS4_STAGE_OTHER = 0,
    DS4_STAGE_ATTENTION,
    DS4_STAGE_ROUTE_SHARED,
    DS4_STAGE_ROUTED_GATE_UP,
    DS4_STAGE_ROUTED_DOWN,
    DS4_STAGE_FINISH,
    DS4_STAGE_COUNT
};

static const char *const g_stage_names[DS4_STAGE_COUNT] = {
    "other", "attention", "router+shared", "routed gate/up", "routed down", "finish/hc/norm"
};

void ds4_gpu_stage(int stage);
void ds4_gpu_stage_probe(id<MTLDevice> device);
int  ds4_gpu_stage_enabled(void);
void ds4_gpu_stage_report(uint64_t tokens);

#define DS4_STAGE_MAX_SAMPLES 4096u

static int g_stage_on;
static id<MTLCounterSampleBuffer> g_stage_sb;
static id<MTLCounterSet> g_stage_counter_set;
static _Thread_local int g_stage_cur = DS4_STAGE_OTHER;
static uint32_t g_stage_n;                       /* samples taken this buffer */
static uint8_t  g_stage_of[DS4_STAGE_MAX_SAMPLES];
static double   g_stage_ms[DS4_STAGE_COUNT];
static double   g_stage_covered_ms;
static uint64_t g_stage_buffers;

int ds4_gpu_stage_enabled(void) { return g_stage_on; }
void ds4_gpu_stage(int stage) { g_stage_cur = stage; }

void ds4_gpu_stage_probe(id<MTLDevice> device) {
    static int probed;
    if (probed) return;
    probed = 1;
    if (!getenv("DS4_METAL_V41_STAGE_SAMPLE") || !device) return;
    for (id<MTLCounterSet> cs in [device counterSets])
        if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) g_stage_counter_set = cs;
    if (!g_stage_counter_set ||
        ![device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        fprintf(stderr, "ds4: no dispatch-boundary counter sampling on this device\n");
        return;
    }
    MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = g_stage_counter_set;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = DS4_STAGE_MAX_SAMPLES;
    NSError *e = nil;
    g_stage_sb = [device newCounterSampleBufferWithDescriptor:d error:&e];
    if (!g_stage_sb) {
        fprintf(stderr, "ds4: stage sample buffer failed: %s\n",
                [[e localizedDescription] UTF8String]);
        return;
    }
    g_stage_on = 1;
    fprintf(stderr, "ds4: stage sampling on, %u samples a buffer\n", DS4_STAGE_MAX_SAMPLES);
}

/* Called from the dispatch helper when the stage has changed. */
static void ds4_stage_mark(id<MTLComputeCommandEncoder> enc) {
    if (!g_stage_on || g_stage_n >= DS4_STAGE_MAX_SAMPLES) return;
    g_stage_of[g_stage_n] = (uint8_t)g_stage_cur;
    [enc sampleCountersInBuffer:g_stage_sb atSampleIndex:g_stage_n withBarrier:NO];
    g_stage_n++;
}

/* Called once a command buffer has completed: the interval between one sample
 * and the next belongs to the stage the first was taken in. Whatever the
 * samples do not cover is the buffer's own gaps, and is reported as such. */
static void ds4_stage_resolve(void) {
    if (!g_stage_on || g_stage_n < 2u) { g_stage_n = 0; return; }
    NSData *data = [g_stage_sb resolveCounterRange:NSMakeRange(0, g_stage_n)];
    if (data && [data length] >= g_stage_n * sizeof(uint64_t)) {
        const uint64_t *ts = (const uint64_t *)[data bytes];
        for (uint32_t i = 0; i + 1u < g_stage_n; i++) {
            if (ts[i + 1u] <= ts[i]) continue;
            const double ms = (double)(ts[i + 1u] - ts[i]) / 1e6;
            g_stage_ms[g_stage_of[i]] += ms;
            g_stage_covered_ms += ms;
        }
        g_stage_buffers++;
    }
    g_stage_n = 0;
}

void ds4_gpu_stage_report(uint64_t tokens) {
    if (!g_stage_on || tokens == 0) return;
    fprintf(stderr, "ds4: stages over %llu tokens, %llu buffers:\n",
            (unsigned long long)tokens, (unsigned long long)g_stage_buffers);
    for (int i = 0; i < DS4_STAGE_COUNT; i++)
        fprintf(stderr, "ds4:   %-16s %6.2f ms a token\n",
                g_stage_names[i], g_stage_ms[i] / (double)tokens);
    fprintf(stderr, "ds4:   %-16s %6.2f ms a token (sampled span)\n",
            "covered", g_stage_covered_ms / (double)tokens);
}

#endif /* DS4_METAL_STAGE_H */
