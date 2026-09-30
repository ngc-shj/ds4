/*
 * What a backend was told, written down, so two backends can be compared
 * without guessing which of them is wrong.
 *
 * Ordering and visibility have been rearranged in every shape available and
 * the answer does not move, which leaves what the dispatches read. That is
 * not a thing to reason about: it is a thing to record. Every encoder call
 * the engine makes, on either backend, writes one line, in encode order, with
 * addresses rather than object identities so the two runs are comparable.
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
/* DS4_METAL_V41_ENC_TRACE_FROM_MARK=1 drops everything before the first mark,
 * so a trace of one verify block is not spent on the prefill that precedes it. */
static int g_enc_trace_armed;

static void ds4_enc_trace(const char *fmt, ...) {
    FILE *f = ds4_enc_trace_file();
    if (!f) return;
    if (!g_enc_trace_armed && getenv("DS4_METAL_V41_ENC_TRACE_FROM_MARK") &&
        strncmp(fmt, "mark ", 5) != 0) return;
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

/* The calls themselves are written in ds4_metal_cmd.h, where every encoder
 * call on either backend goes through. */

#endif  /* DS4_METAL_ENC_TRACE_H */
