/* The long prompt's prefill tail, two ways, in one process.
 *
 *   test_deepseek41_tailab <model.gguf> <story.txt> [rounds] [cache_experts] [rows_a rows_b]
 *
 * With rows_a and rows_b both arms are batched, in steps of that many rows
 * (2..800, through ds4_test_v41_tail_rows); without them B is token-major. A
 * further argument "em" makes B the expert-major routed gate/up
 * (ds4_gpu_moe_expert_major_override), A the default pair-major one; rows and
 * "em" combine.
 *
 * Prefills the first 4096 tokens of the backend test's long prompt once and
 * keeps the session as a snapshot. Each arm then restores that snapshot and
 * syncs the whole prompt, so only the 699-row tail runs, and is timed from the
 * sync's start to its return; restoring and hashing are outside the timed
 * span. Arm A is the production tail. Arm B is the token-major tail the
 * engine's ds4_test_v41_token_major_tail hook brings back - the path before
 * the batched tail - for the comparison; a build without the hook runs A
 * twice over, which measures the spread of the ruler itself.
 *
 * A session snapshot does not hold the expert cache, so an arm warms the cache
 * for the next: both arms run once first, unmeasured, and the rounds then
 * alternate A B B A and B A A B. This compares compute on a warm cache; a cold
 * time to first token is a separate, one-process-per-arm measurement.
 *
 * Per arm: the wall, the logits after the tail and the saved session payload
 * (KV and carried state), hashed, and the expert cache's hits, misses, loads
 * and evictions during it. Every arm's hashes must equal the first A's. The
 * rows each tail ran token-major are in the engine's own "prefill tail" line
 * on stderr, and the distinct experts the first tail touched - its I/O bound,
 * read once each - are counted. */
#include "../ds4.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ds4_test_v41_token_major_tail(int on) __attribute__((weak));
void ds4_test_v41_tail_rows(uint32_t rows) __attribute__((weak));
void ds4_gpu_moe_expert_major_override(int mode) __attribute__((weak));
uint64_t ds4_gpu_moe_expert_major_dispatches(void) __attribute__((weak));
static uint32_t g_rows[2];
static int g_em, g_blk;
/* "benv:NAME=VALUE": an engine switch read at every call, set only while arm B
 * runs (both arms otherwise as asked). */
static char g_benv_name[128], g_benv_value[128];
/* "blk": arm B runs the block executor's batched attention, through the
 * switches the engine reads at every step (DS4_METAL_V41_BLOCK and its
 * ATTN_K / ATTN_RAW parts), set only while that arm runs. */
static void block_switches(int on) {
    static const char *names[] = { "DS4_METAL_V41_BLOCK", "DS4_METAL_V41_BLOCK_ATTN_K",
                                    "DS4_METAL_V41_BLOCK_ATTN_RAW", "DS4_METAL_V41_ROW_GATHER",
                                    "DS4_METAL_V41_BLOCK_KV", "DS4_METAL_V41_BLOCK_LEVELS",
                                    "DS4_METAL_V41_BLOCK_LOW_BATCH" };
    for (int i = 0; i < 7; i++) {
        if (on && (g_blk & (1 << i))) setenv(names[i], "1", 1); else unsetenv(names[i]);
    }
}
void ds4_gpu_stream_expert_cache_counters(uint64_t *hits, uint64_t *misses, uint64_t *evictions,
                                          uint32_t *entries, uint32_t *budget);
void ds4_gpu_expert_directory_counts(unsigned long long out[8]);
void ds4_test_moe_host_ms(double out[8]) __attribute__((weak));
void ds4_test_expert_read_cached(int on) __attribute__((weak));
void ds4_test_v41_tail_block(int on) __attribute__((weak));
/* "prev": arm B is the batched tail before the wide step - 32-row steps, no
 * block attention - arm A production; both read experts through the mapped
 * descriptor. */
static int g_prev;
/* "q8": arm B groups two output rows a threadgroup in the four-row Q8_0 tile,
 * not eight. */
void ds4_test_v41_q8_nr8(int on) __attribute__((weak));
int ds4_test_engine_q8_tiles(ds4_engine *e);
int ds4_test_v41_carry_ms(ds4_session *s, double out[2]);
static int g_q8;
/* "full": each arm prefills the whole prompt from position 0 in a new session
 * (the 4096-row layer-major prefix and the tail), not the tail from the
 * snapshot. */
static int g_full;
/* "gu2": arm B encodes the wide step's routed gate/up twice - what it costs. */
void ds4_test_moe_gateup_reps(uint32_t n) __attribute__((weak));
static int g_gu2;
/* "il" (with "full"): arm B runs the prefix and then the tail, not layer by
 * layer. */
void ds4_test_v41_interleave(int on) __attribute__((weak));
static int g_il;
/* "f32x2": arm B encodes every F32-weight batch projection twice. */
void ds4_test_v41_f32_reps(uint32_t n) __attribute__((weak));
static int g_f32;
/* "abl:routed" / "abl:attn" (with "full"): the whole-prompt exact replay -
 * the reference warm-up captures the stage's outputs, then A runs the stage
 * and copies them over its own (control), B skips it and copies them in
 * (replay); B - A is what the stage costs in the prompt. */
void ds4_test_v41_pabl(int stage, int mode) __attribute__((weak));
static int g_pabl, g_pabl_runs;
/* "abl3" (with "full"): one capture of the pre-attention mixes and norms and
 * the router's selection, then four arms in the order C P R PR PR R P C -
 * control, pre-attention skipped, router selection skipped, both - each
 * restoring both stages' outputs, so the arms differ only in what they skip;
 * and the indexer top-k's sort dispatches each arm encodes. */
void ds4_test_v41_pabl_set(int stages, int mode, int skip) __attribute__((weak));
void ds4_test_topk_counts(unsigned long long out[3]) __attribute__((weak));
static int g_abl3;
/* "cool:N": sleep N seconds before each arm, outside the timed span - a
 * row of whole-prompt runs heats the machine past any order's cancelling. */
static unsigned g_cool;
/* "topk1" (with "full"): arm B selects the indexer's top-k a row at a time,
 * A batches the wide rows; both print the rows each way and the sort
 * dispatches. */
void ds4_test_v41_topk_batch(int on) __attribute__((weak));
void ds4_test_v41_topk_rows(unsigned long long out[2]) __attribute__((weak));
static int g_topk1;
/* "eb" (with "full"): arm B runs layer 20's block selection and the later
 * layers' filter; A takes the fill when every block is kept. */
void ds4_test_v41_every_block(int on) __attribute__((weak));
void ds4_test_v41_every_block_counts(unsigned long long out[6]) __attribute__((weak));
static int g_eb;
/* "attnrows": arm B takes the block attention levels a row at a time, A in
 * row tiles. "attnchk": every arm runs the tiles and then the rows and
 * compares every word of the heads (compare slot 11). */
void ds4_test_v41_attn_rows(int on) __attribute__((weak));
void ds4_test_v41_attn_rows_check(int on) __attribute__((weak));
void ds4_test_v41_attn_rows_counts(unsigned long long out[2]) __attribute__((weak));
void ds4_gpu_dsv41_compare_counts(uint32_t out[16]) __attribute__((weak));
static int g_attn;
/* "append:N:M": sync the prompt's first N tokens, then append the next M with
 * the attention check on - an append at a short position, where rows have
 * different compressed key counts - print the rows and the differing words,
 * and stop. */
static uint32_t g_app_n, g_app_m;
/* "pkg" (with "full"): the package of exact changes as one arm - A with the
 * every-block selection and the attention row tiles, B with neither. */
static int g_pkg;
/* "mt:GU:DN" (with "full"): arm A runs the prefill's packed routed matmuls
 * with gate/up tile GU and down tile DN (M * 1000 + N, 0 = 64 x 32), B with
 * production's 64 x 32. "mtchk:GU:DN": every arm runs the tiles and then the
 * 64 x 32 kernel, and compares every word (compare slots 12 gate, 13 up,
 * 14 down). */
void ds4_test_moe_packed_tiles(uint32_t gate_up, uint32_t down, int check) __attribute__((weak));
uint64_t ds4_test_moe_packed_tile_layers(void) __attribute__((weak));
static uint32_t g_mt_gu, g_mt_dn;
/* "preps:S[:N]" (with "full"): arm B encodes stage S of the prefill's packed
 * routed MoE N times (default 2) (1 map and pack, 2 gate, 3 up, 4 SwiGLU, 5 down, 6 sum). */
void ds4_test_moe_packed_reps(int stage, uint32_t n) __attribute__((weak));
static int g_preps;
static unsigned g_preps_n = 2;
static int g_mt;
/* "mp" (with "full"): arm A runs a 4096-row prefix's two chunks as one
 * superbatch (every row-independent step once over both, attention a chunk at
 * a time), B a chunk at a time; both print the layers run so. */
void ds4_test_v41_superbatch(int on) __attribute__((weak));
unsigned long long ds4_test_v41_superbatch_layers(void) __attribute__((weak));
static int g_mp;
/* "drop:<pagecache tool>": arm B starts with the model file dropped from the
 * file cache, outside the timed span - what the arm's reads cost from the SSD. */
static const char *g_drop_tool;
static const char *g_model_path;
static ds4_engine *g_engine;
void ds4_test_expert_touch(int on);
unsigned ds4_test_expert_touch_count(void);

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int payload_hash(ds4_session *s, uint64_t *hash) {
    char err[256];
    FILE *fp = tmpfile();
    if (!fp) return 0;
    if (ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) { fclose(fp); return 0; }
    rewind(fp);
    uint64_t h = 1469598103934665603ull;
    unsigned char chunk[1 << 16];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) h = fnv(h, chunk, n);
    fclose(fp);
    *hash = h;
    return 1;
}

typedef struct { double wall; uint64_t logits, state, hits, misses, evictions, em; unsigned long long loads; double host[8]; } arm_result;

static int run_arm(ds4_session **sp, const ds4_session_snapshot *snap, const ds4_tokens *prompt,
                   int b, float *buf, int vocab, arm_result *r) {
    char err[256];
    if (g_cool) sleep(g_cool);
    if (g_full) {
        ds4_session_free(*sp);
        *sp = NULL;
        if (ds4_session_create(sp, g_engine, 8192) != 0) return 0;
    } else if (ds4_session_load_snapshot(*sp, snap, err, sizeof(err)) != 0) {
        fprintf(stderr, "snapshot load: %s\n", err);
        return 0;
    }
    if (g_em) ds4_gpu_moe_expert_major_override(b);
    if (g_blk) block_switches(b);
    if (g_q8) ds4_test_v41_q8_nr8(!b);
    if (g_gu2) ds4_test_moe_gateup_reps(b ? 2u : 1u);
    if (g_il) ds4_test_v41_interleave(!b);
    if (g_f32) ds4_test_v41_f32_reps(b ? 2u : 1u);
    if (g_topk1) ds4_test_v41_topk_batch(!b);
    if (g_pkg) { ds4_test_v41_every_block(!b); ds4_test_v41_attn_rows(!b); }
    unsigned long long pk_eb0[6] = {0}, pk_ar0[2] = {0};
    if (g_pkg) { ds4_test_v41_every_block_counts(pk_eb0); ds4_test_v41_attn_rows_counts(pk_ar0); }
    if (g_eb) ds4_test_v41_every_block(!b);
    if (g_mp) ds4_test_v41_superbatch(!b);
    const unsigned long long mp0 = g_mp ? ds4_test_v41_superbatch_layers() : 0;
    if (g_preps) ds4_test_moe_packed_reps(g_preps, b ? g_preps_n : 1u);
    uint64_t mt0 = 0;
    uint32_t mc0[16] = {0};
    if (g_mt) {
        const int on = g_mt == 2 || !b;
        ds4_test_moe_packed_tiles(on ? g_mt_gu : 0, on ? g_mt_dn : 0, g_mt == 2);
        mt0 = ds4_test_moe_packed_tile_layers();
        ds4_gpu_dsv41_compare_counts(mc0);
    }
    unsigned long long ek0[3] = {0}, eb0[6] = {0};
    if (g_eb) { ds4_test_topk_counts(ek0); ds4_test_v41_every_block_counts(eb0); }
    unsigned long long ar0[2] = {0};
    uint32_t ac0[16] = {0};
    if (g_attn) {
        ds4_test_v41_attn_rows(g_attn == 2 || !b);
        ds4_test_v41_attn_rows_check(g_attn == 2);
        ds4_test_v41_attn_rows_counts(ar0);
        ds4_gpu_dsv41_compare_counts(ac0);
    }
    unsigned long long tk0[3] = {0}, tr0[2] = {0};
    if (g_topk1) { ds4_test_topk_counts(tk0); ds4_test_v41_topk_rows(tr0); }
    if (g_pabl) ds4_test_v41_pabl(g_pabl, g_pabl_runs++ == 0 ? 1 : b ? 3 : 2);
    if (g_drop_tool && b) {
        char cmd[2048];
        snprintf(cmd, sizeof(cmd), "'%s' '%s' drop 1>&2", g_drop_tool, g_model_path);
        if (system(cmd) != 0) return 0;
    }
    if (g_prev) {
        ds4_test_expert_read_cached(1);
        ds4_test_v41_tail_block(!b);
        ds4_test_v41_tail_rows(b ? 32u : 0u);
    }
    if (g_benv_name[0]) { if (b) setenv(g_benv_name, g_benv_value, 1); else unsetenv(g_benv_name); }
    if (g_rows[0] && ds4_test_v41_tail_rows) ds4_test_v41_tail_rows(g_rows[b]);
    else if (!g_em && !g_prev && !g_q8 && !g_gu2 && !g_drop_tool && !g_il && !g_f32 && !g_pabl && !g_topk1 && !g_eb && !g_attn && !g_pkg && !g_mt && !g_preps && !g_mp && ds4_test_v41_token_major_tail) ds4_test_v41_token_major_tail(b);
    ds4_session *s = *sp;
    uint64_t h0, m0, e0, h1, m1, e1;
    const uint64_t em0 = ds4_gpu_moe_expert_major_dispatches ? ds4_gpu_moe_expert_major_dispatches() : 0;
    unsigned long long d0[8], d1[8];
    ds4_gpu_stream_expert_cache_counters(&h0, &m0, &e0, NULL, NULL);
    ds4_gpu_expert_directory_counts(d0);
    double h0ms[8] = {0}, h1ms[8] = {0};
    if (ds4_test_moe_host_ms) ds4_test_moe_host_ms(h0ms);
    const double t0 = now_s();
    const int rc = ds4_session_sync(s, prompt, err, sizeof(err));
    const double t1 = now_s();
    ds4_gpu_stream_expert_cache_counters(&h1, &m1, &e1, NULL, NULL);
    ds4_gpu_expert_directory_counts(d1);
    if (ds4_test_moe_host_ms) ds4_test_moe_host_ms(h1ms);
    for (int k = 0; k < 8; k++) r->host[k] = (h1ms[k] - h0ms[k]) / 1000.0;
    if (ds4_test_v41_token_major_tail) ds4_test_v41_token_major_tail(0);
    if (ds4_test_v41_tail_rows) ds4_test_v41_tail_rows(0);
    if (g_em) ds4_gpu_moe_expert_major_override(-1);
    if (g_blk) block_switches(0);
    if (g_prev) { ds4_test_expert_read_cached(1); ds4_test_v41_tail_block(1); }
    if (g_q8) ds4_test_v41_q8_nr8(1);
    if (g_gu2) ds4_test_moe_gateup_reps(1);
    if (g_il) ds4_test_v41_interleave(1);
    if (g_f32) ds4_test_v41_f32_reps(1);
    if (g_mp) {
        printf("mp       arm %c: %llu layers superbatched\n", b ? 'B' : 'A',
               ds4_test_v41_superbatch_layers() - mp0);
        ds4_test_v41_superbatch(1);
    }
    if (g_preps) {
        ds4_test_moe_packed_reps(g_preps, 1u);
        printf("preps    arm %c: stage %d encoded %u times\n", b ? 'B' : 'A', g_preps, b ? g_preps_n : 1u);
    }
    if (g_mt) {
        uint32_t mc1[16];
        ds4_gpu_dsv41_compare_counts(mc1);
        const uint64_t mt1 = ds4_test_moe_packed_tile_layers();
        ds4_test_moe_packed_tiles(0, 0, 0);
        printf("mt       arm %c: %llu layer-chunks tiled; words differing gate %u up %u down %u\n",
               b ? 'B' : 'A', (unsigned long long)(mt1 - mt0), mc1[12] - mc0[12],
               mc1[13] - mc0[13], mc1[14] - mc0[14]);
    }
    if (g_eb) {
        unsigned long long ek1[3], eb1[6];
        ds4_test_topk_counts(ek1);
        ds4_test_v41_every_block_counts(eb1);
        ds4_test_v41_every_block(1);
        printf("eb       arm %c: layer-20 rows fast/fallback prefix %llu/%llu, tail %llu/%llu, "
               "other %llu/%llu; %llu top-k calls, %llu sort dispatches\n", b ? 'B' : 'A',
               eb1[0] - eb0[0], eb1[1] - eb0[1], eb1[2] - eb0[2], eb1[3] - eb0[3],
               eb1[4] - eb0[4], eb1[5] - eb0[5], ek1[0] - ek0[0], ek1[1] - ek0[1]);
    }
    if (g_attn) {
        unsigned long long ar1[2];
        uint32_t ac1[16];
        ds4_test_v41_attn_rows_counts(ar1);
        ds4_gpu_dsv41_compare_counts(ac1);
        ds4_test_v41_attn_rows(1);
        ds4_test_v41_attn_rows_check(0);
        printf("attn     arm %c: %llu rows tiled, %llu a row; heads words differing %u\n",
               b ? 'B' : 'A', ar1[0] - ar0[0], ar1[1] - ar0[1], ac1[11] - ac0[11]);
    }
    if (g_pkg) {
        unsigned long long eb1[6], ar1[2];
        ds4_test_v41_every_block_counts(eb1);
        ds4_test_v41_attn_rows_counts(ar1);
        ds4_test_v41_every_block(1);
        ds4_test_v41_attn_rows(1);
        printf("pkg      arm %c: layer-20 fast/fallback prefix %llu/%llu tail %llu/%llu other "
               "%llu/%llu; attention rows tiled %llu, a row %llu\n", b ? 'B' : 'A',
               eb1[0] - pk_eb0[0], eb1[1] - pk_eb0[1], eb1[2] - pk_eb0[2], eb1[3] - pk_eb0[3],
               eb1[4] - pk_eb0[4], eb1[5] - pk_eb0[5], ar1[0] - pk_ar0[0], ar1[1] - pk_ar0[1]);
    }
    if (g_topk1) {
        unsigned long long tk1[3], tr1[2];
        ds4_test_topk_counts(tk1); ds4_test_v41_topk_rows(tr1);
        ds4_test_v41_topk_batch(1);
        printf("topk     arm %c: %llu rows batched, %llu a row; %llu top-k calls, %llu sort "
               "dispatches\n", b ? 'B' : 'A', tr1[0] - tr0[0], tr1[1] - tr0[1],
               tk1[0] - tk0[0], tk1[1] - tk0[1]);
    }
    if (g_pabl) ds4_test_v41_pabl(0, 0);
    if (g_benv_name[0]) unsetenv(g_benv_name);
    if (rc != 0) { fprintf(stderr, "sync: %s\n", err); return 0; }
    if (ds4_session_copy_logits(s, buf, vocab) != vocab) return 0;
    r->wall = t1 - t0;
    r->logits = fnv(1469598103934665603ull, buf, (size_t)vocab * sizeof(float));
    r->hits = h1 - h0; r->misses = m1 - m0; r->evictions = e1 - e0; r->loads = d1[2] - d0[2];
    r->em = (ds4_gpu_moe_expert_major_dispatches ? ds4_gpu_moe_expert_major_dispatches() : 0) - em0;
    return payload_hash(s, &r->state);
}

static volatile sig_atomic_t g_stop;
static void on_term(int sig) { (void)sig; g_stop = 1; }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf story.txt [rounds] [cache_experts]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGTERM, on_term);
    const int rounds = argc > 3 ? atoi(argv[3]) : 2;
    const unsigned cache = argc > 4 ? (unsigned)atoi(argv[4]) : 10268;
    /* [rows_a rows_b] [em]: either or both. */
    for (int a = 5; a < argc; a++) {
        /* blk: BLOCK, ATTN_K and ATTN_RAW; blkN: the switches whose bits are set
         * in N - 1 BLOCK, 2 ATTN_K, 4 ATTN_RAW, 8 ROW_GATHER, 16 BLOCK_KV,
         * 32 BLOCK_LEVELS, 64 BLOCK_LOW_BATCH. */
        if (strcmp(argv[a], "q8") == 0) {
            if (!ds4_test_v41_q8_nr8) return 2;
            g_q8 = 1;
        } else if (strncmp(argv[a], "abl:", 4) == 0) {
            if (!ds4_test_v41_pabl) return 2;
            g_pabl = strcmp(argv[a] + 4, "routed") == 0 ? 1 :
                     strcmp(argv[a] + 4, "attn") == 0 ? 2 : 0;
            if (!g_pabl) return 2;
        } else if (strncmp(argv[a], "cool:", 5) == 0) {
            g_cool = (unsigned)atoi(argv[a] + 5);
        } else if (strcmp(argv[a], "eb") == 0) {
            if (!ds4_test_v41_every_block || !ds4_test_v41_every_block_counts || !ds4_test_topk_counts)
                return 2;
            g_eb = 1;
        } else if (strncmp(argv[a], "append:", 7) == 0) {
            if (sscanf(argv[a] + 7, "%u:%u", &g_app_n, &g_app_m) != 2 || !g_app_n || !g_app_m)
                return 2;
            if (!ds4_test_v41_attn_rows_check || !ds4_test_v41_attn_rows_counts ||
                !ds4_gpu_dsv41_compare_counts) return 2;
        } else if (strcmp(argv[a], "attnrows") == 0 || strcmp(argv[a], "attnchk") == 0) {
            if (!ds4_test_v41_attn_rows || !ds4_test_v41_attn_rows_check ||
                !ds4_test_v41_attn_rows_counts || !ds4_gpu_dsv41_compare_counts) return 2;
            g_attn = argv[a][4] == 'c' ? 2 : 1;
        } else if (strcmp(argv[a], "mp") == 0) {
            if (!ds4_test_v41_superbatch || !ds4_test_v41_superbatch_layers) return 2;
            g_mp = 1;
        } else if (strncmp(argv[a], "preps:", 6) == 0) {
            if (!ds4_test_moe_packed_reps) return 2;
            g_preps = atoi(argv[a] + 6);
            const char *n = strchr(argv[a] + 6, ':');
            if (n) g_preps_n = (unsigned)atoi(n + 1);
            if (g_preps_n < 2 || g_preps_n > 16) return 2;
            if (g_preps < 1 || g_preps > 6) return 2;
        } else if (strncmp(argv[a], "mt:", 3) == 0 || strncmp(argv[a], "mtchk:", 6) == 0) {
            if (!ds4_test_moe_packed_tiles || !ds4_test_moe_packed_tile_layers ||
                !ds4_gpu_dsv41_compare_counts) return 2;
            g_mt = argv[a][2] == 'c' ? 2 : 1;
            if (sscanf(strchr(argv[a], ':') + 1, "%u:%u", &g_mt_gu, &g_mt_dn) != 2) return 2;
        } else if (strcmp(argv[a], "pkg") == 0) {
            if (!ds4_test_v41_every_block || !ds4_test_v41_attn_rows) return 2;
            g_pkg = 1;
        } else if (strcmp(argv[a], "topk1") == 0) {
            if (!ds4_test_v41_topk_batch || !ds4_test_v41_topk_rows || !ds4_test_topk_counts)
                return 2;
            g_topk1 = 1;
        } else if (strcmp(argv[a], "abl3") == 0 || strcmp(argv[a], "run1") == 0 ||
                   strcmp(argv[a], "run2") == 0) {
            if (!ds4_test_v41_pabl_set || !ds4_test_topk_counts) return 2;
            g_abl3 = argv[a][0] == 'a' ? 3 : argv[a][3] - '0';
        } else if (strcmp(argv[a], "f32x2") == 0) {
            if (!ds4_test_v41_f32_reps) return 2;
            g_f32 = 1;
        } else if (strcmp(argv[a], "il") == 0) {
            if (!ds4_test_v41_interleave) return 2;
            g_il = 1;
        } else if (strcmp(argv[a], "gu2") == 0) {
            if (!ds4_test_moe_gateup_reps) return 2;
            g_gu2 = 1;
        } else if (strncmp(argv[a], "drop:", 5) == 0) {
            g_drop_tool = argv[a] + 5;
        } else if (strcmp(argv[a], "full") == 0) {
            g_full = 1;
        } else if (strcmp(argv[a], "prev") == 0) {
            if (!ds4_test_expert_read_cached || !ds4_test_v41_tail_block || !ds4_test_v41_tail_rows)
                return 2;
            g_prev = 1;
        } else if (strncmp(argv[a], "benv:", 5) == 0) {
            const char *eq = strchr(argv[a] + 5, '=');
            if (!eq || (size_t)(eq - argv[a] - 5) >= sizeof(g_benv_name)) return 2;
            memcpy(g_benv_name, argv[a] + 5, (size_t)(eq - argv[a] - 5));
            snprintf(g_benv_value, sizeof(g_benv_value), "%s", eq + 1);
        } else if (strncmp(argv[a], "blk", 3) == 0) {
            g_blk = argv[a][3] ? atoi(argv[a] + 3) : 7;
        } else if (strcmp(argv[a], "em") == 0) {
            if (!ds4_gpu_moe_expert_major_override) return 2;
            g_em = 1;
        } else if (a + 1 < argc) {
            g_rows[0] = (uint32_t)atoi(argv[a]); g_rows[1] = (uint32_t)atoi(argv[a + 1]); a++;
        }
    }
    FILE *f = fopen(argv[2], "rb");
    if (!f) return 2;
    char *story = calloc(1, 22000 + 64);
    const size_t n = fread(story, 1, 22000, f);
    fclose(f);
    strcpy(story + n, "\n\nSummarize the story so far.");

    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = cache,
        .context_size = 8192, .placement_ctx_hint = 8192,
        .placement_session_count_hint = 2, .power_percent = 100,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 2;
    g_engine = engine;
    g_model_path = argv[1];
    /* "tiles": only the Q8_0 row-tile check on the model's weights. */
    for (int a = 5; a < argc; a++)
        if (strcmp(argv[a], "tiles") == 0) {
            const int bad = ds4_test_engine_q8_tiles(engine);
            printf("tiles    %d combination(s) differ from the single-row arithmetic -> %s\n",
                   bad, bad == 0 ? "PASS" : "FAIL");
            ds4_engine_close(engine);
            return bad == 0 ? 0 : 1;
        }
    const int vocab = ds4_engine_vocab_size(engine);
    float *buf = malloc((size_t)vocab * sizeof(float));
    ds4_tokens prompt = {0}, prefix = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", story);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    ds4_session *s = NULL;
    char err[256] = "";
    int ok = ds4_session_create(&s, engine, 8192) == 0 && prompt.len > 4096;
    if (ok && g_app_n) {
        ds4_tokens head = {0}, both = {0};
        for (uint32_t i = 0; i < g_app_n && (int)i < prompt.len; i++) ds4_tokens_push(&head, prompt.v[i]);
        for (uint32_t i = 0; i < g_app_n + g_app_m && (int)i < prompt.len; i++)
            ds4_tokens_push(&both, prompt.v[i]);
        ok = ds4_session_sync(s, &head, err, sizeof(err)) == 0;
        unsigned long long r0[2], r1[2];
        uint32_t c0[16], c1[16];
        ds4_test_v41_attn_rows_counts(r0);
        ds4_gpu_dsv41_compare_counts(c0);
        ds4_test_v41_attn_rows_check(1);
        ok = ok && ds4_session_sync(s, &both, err, sizeof(err)) == 0;
        ds4_test_v41_attn_rows_check(0);
        ds4_test_v41_attn_rows_counts(r1);
        ds4_gpu_dsv41_compare_counts(c1);
        printf("append   %u then %u tokens: %llu rows tiled, %llu a row; heads words differing %u%s\n",
               g_app_n, g_app_m, r1[0] - r0[0], r1[1] - r0[1], c1[11] - c0[11],
               ok ? "" : " (sync failed)");
        ds4_tokens_free(&head); ds4_tokens_free(&both);
        return ok && c1[11] == c0[11] ? 0 : 1;
    }
    for (int i = 0; ok && i < 4096; i++) ds4_tokens_push(&prefix, prompt.v[i]);
    const double p0 = now_s();
    ok = ok && ds4_session_sync(s, &prefix, err, sizeof(err)) == 0;
    const double p1 = now_s();
    /* "carry": time the wide prefix's carry copies on this session, then stop. */
    for (int a = 5; ok && a < argc; a++)
        if (strcmp(argv[a], "carry") == 0) {
            double ms[2] = {0, 0};
            const int rc = ds4_test_v41_carry_ms(s, ms);
            printf("carry    2048 rows: store %.3f ms, load %.3f ms a call (rc %d)\n",
                   ms[0], ms[1], rc);
            return rc == 0 ? 0 : 1;
        }
    ds4_session_snapshot snap = {0};
    ok = ok && ds4_session_save_snapshot(s, &snap, err, sizeof(err)) == 0;
    printf("prefix   4096 rows layer-major in %.3f s (cold), %.2f ms a row; tail %d rows; "
           "%s%s\n", p1 - p0, (p1 - p0) * 1000.0 / 4096.0, prompt.len - 4096,
           g_q8 ? "B = Q8_0 four-row tile, two outputs a threadgroup" :
           g_prev ? "B = the batched tail before (32 rows, no block attention, cached reads)" :
           g_blk ? "B = block executor attention" :
           g_em && g_rows[0] ? "rows as asked, B expert-major" :
           g_em ? "B = expert-major gate/up" : g_rows[0] ? "A and B batched, rows as asked" :
           ds4_test_v41_token_major_tail ? "B = token-major" : "none (A twice: the ruler's spread)",
           g_full ? "; arms prefill the whole prompt" : "");

    arm_result ref = {0}, r;
    int bad = 0;
    if (ok && g_abl3) {
        /* run1: routed and shared expert; run2: the prefix's attention (with
         * q_b), the q_a/kv projections, the tail's attention (with q_b). Each
         * against the one control of its own run, in mirrored order. */
        enum { ROUTED = 1, ATTN = 2, PRE = 4, ROUTER = 8, SHARED = 16, PROJ = 32, TATTN = 64 };
        static const int skip3[4] = { 0, PRE, ROUTER, PRE | ROUTER };
        static const int skip1[4] = { 0, ROUTED, SHARED, 0 };
        static const int skip2[4] = { 0, ATTN, TATTN, PROJ };
        static const char *const name3[4] = { "C ", "P ", "R ", "PR" };
        static const char *const name1[4] = { "C ", "Ro", "Sh", "" };
        static const char *const name2[4] = { "C ", "At", "Ta", "Pj" };
        const int *skip = g_abl3 == 1 ? skip1 : g_abl3 == 2 ? skip2 : skip3;
        const char *const *name = g_abl3 == 1 ? name1 : g_abl3 == 2 ? name2 : name3;
        const int stages = g_abl3 == 1 ? ROUTED | SHARED : g_abl3 == 2 ? ATTN | TATTN | PROJ :
                           PRE | ROUTER;
        const int arms = g_abl3 == 1 ? 3 : 4;
        static const int order4[8] = { 0, 1, 2, 3, 3, 2, 1, 0 };
        static const int order3[6] = { 0, 1, 2, 2, 1, 0 };
        const int *order = arms == 3 ? order3 : order4;
        double wall[4] = {0};
        ds4_test_v41_pabl_set(stages, 1, 0);
        ok = run_arm(&s, &snap, &prompt, 0, buf, vocab, &ref);
        ds4_test_v41_pabl_set(0, 0, 0);
        printf("capture  %.3f s; reference logits %016llx state %016llx\n", ref.wall,
               (unsigned long long)ref.logits, (unsigned long long)ref.state);
        for (int j = 0; ok && !g_stop && j < 2 * arms; j++) {
            const int k = order[j];
            unsigned long long t0[3], t1[3];
            ds4_test_topk_counts(t0);
            ds4_test_v41_pabl_set(stages, k ? 3 : 2, skip[k]);
            ok = run_arm(&s, &snap, &prompt, 0, buf, vocab, &r);
            ds4_test_v41_pabl_set(0, 0, 0);
            ds4_test_topk_counts(t1);
            if (!ok) break;
            const int same = r.logits == ref.logits && r.state == ref.state;
            bad += !same;
            wall[k] += r.wall / 2.0;
            printf("arm %s   %.3f s  %s | top-k %llu calls, %llu sort dispatches, %llu rows\n",
                   name[k], r.wall, same ? "same" : "DIFFERENT", t1[0] - t0[0], t1[1] - t0[1],
                   t1[2] - t0[2]);
        }
        if (ok && g_abl3 != 3) {
            printf("run%d     C %.3f s", g_abl3, wall[0]);
            for (int k = 1; k < arms; k++)
                printf("; %s %.3f s, C - %s %+.3f s", name[k], wall[k], name[k], wall[0] - wall[k]);
            printf("; %d arm(s) differing -> %s\n", bad, bad ? "FAIL" : "PASS");
        } else if (ok)
            printf("abl3     C %.3f, P %.3f, R %.3f, PR %.3f s: pre-attention %.3f, router "
                   "selection %.3f, both %.3f (sum %.3f) s; %d arm(s) differing -> %s\n",
                   wall[0], wall[1], wall[2], wall[3], wall[0] - wall[1], wall[0] - wall[2],
                   wall[0] - wall[3], (wall[0] - wall[1]) + (wall[0] - wall[2]), bad,
                   bad ? "FAIL" : "PASS");
        goto cleanup;
    }
    double sum[2] = {0, 0};
    int cnt[2] = {0, 0};
    /* Warm-up: each arm once, unmeasured; the first A is the reference. */
    ds4_test_expert_touch(1);
    ok = ok && run_arm(&s, &snap, &prompt, 0, buf, vocab, &ref);
    const unsigned touched = ds4_test_expert_touch_count();
    ds4_test_expert_touch(0);
    printf("touched  the tail used %u distinct experts of %d (%.1f GiB at 9.49 MiB each); "
           "%llu loads, %llu misses\n", touched, 40 * 384, touched * 9.49 / 1024.0,
           ref.loads, (unsigned long long)ref.misses);
    ok = ok && run_arm(&s, &snap, &prompt, 1, buf, vocab, &r);
    if (ok && (r.logits != ref.logits || r.state != ref.state)) bad++;
    printf("host     A: drain %.2f, prepare %.2f (load prepare %.2f, pread %.2f, install %.2f, "
           "slot buffers %.2f, readahead %.2f, reuse scan %.2f) s\n", ref.host[0], ref.host[1],
           ref.host[2], ref.host[3], ref.host[4], ref.host[5], ref.host[6], ref.host[7]);
    printf("host     B: drain %.2f, prepare %.2f (load prepare %.2f, pread %.2f, install %.2f, "
           "slot buffers %.2f, readahead %.2f, reuse scan %.2f) s\n", r.host[0], r.host[1],
           r.host[2], r.host[3], r.host[4], r.host[5], r.host[6], r.host[7]);
    printf("warm-up  A %.3f s, B %.3f s; reference logits %016llx state %016llx\n", ref.wall,
           r.wall, (unsigned long long)ref.logits, (unsigned long long)ref.state);
    static const int order[2][4] = { {0, 1, 1, 0}, {1, 0, 0, 1} };
    for (int k = 0; ok && !g_stop && k < rounds; k++) {
        for (int j = 0; ok && j < 4; j++) {
            const int b = order[k & 1][j];
            ok = run_arm(&s, &snap, &prompt, b, buf, vocab, &r);
            if (!ok) break;
            const int same = r.logits == ref.logits && r.state == ref.state;
            bad += !same;
            sum[b] += r.wall; cnt[b]++;
            printf("round %d %c  %.3f s  logits %016llx state %016llx %s | hits %llu misses %llu "
                   "loads %llu evictions %llu, expert-major dispatches %llu\n", k, b ? 'B' : 'A', r.wall,
                   (unsigned long long)r.logits, (unsigned long long)r.state,
                   same ? "same" : "DIFFERENT", (unsigned long long)r.hits,
                   (unsigned long long)r.misses, r.loads, (unsigned long long)r.evictions,
                   (unsigned long long)r.em);
        }
    }
    if (ok && cnt[0] && cnt[1])
        printf("tail     A mean %.3f s, B mean %.3f s, B - A %+.3f s over %d rounds; %d arm(s) "
               "differing from the reference -> %s\n", sum[0] / cnt[0], sum[1] / cnt[1],
               sum[1] / cnt[1] - sum[0] / cnt[0], rounds, bad, bad ? "FAIL" : "PASS");
cleanup:
    if (!ok) printf("failed   %s\n", err);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    ds4_tokens_free(&prefix);
    free(buf);
    free(story);
    ds4_engine_close(engine);
    return ok && !bad && !g_stop ? 0 : 1;
}
