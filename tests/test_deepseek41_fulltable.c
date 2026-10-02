/* Pinned expert windows in the one expert directory, brought up in stages.
 *
 *   test_deepseek41_fulltable <model.gguf> <window> [steps] [cache_experts] [forced] [rows]
 *
 * Run through bench/gpurun.sh with the production environment plus
 * DS4_METAL_ENABLE_V41_STREAMING_SESSION_BATCH=1, and - for the run under
 * test - DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE=1 and
 * DS4_METAL_FULL_ADDR_LAYERS=<n>. The same command without those two is the
 * oracle: where a pinned expert's weights live changes nothing the model
 * computes, so the two must agree on every token and every logit.
 *
 * `window` experts of each asked layer are pinned at open (0: a whole layer).
 * Two sessions decode together through the native batch for `steps` steps.
 * With `forced` (the default) every layer's routing is replaced, in the same
 * block, by experts 0..5 for the first row and 378..383 for the second, so a
 * window covers the first row's and the cache serves the second's - and the
 * directory's counts over the decode can be checked against what that implies:
 * a hit on a pinned entry for every forced expert the directory holds pinned,
 * a cache hit or a load for every other, no table slot that is not its entry's
 * address, no missing expert with a slot, no selected expert left missing.
 *
 * With rows 1 a single session decodes one row at a time instead: with the
 * gate off that is the single-row routed path, and a forced routing goes in as
 * the host's own choice of ids - staged and copied into the device's buffer by
 * a node - which the staging counts report. The row takes 0..2 and 378..380, so
 * one dispatch reads a window's pinned experts and the cache's together.
 *
 * Prints the device's holdings at open and after close, the directory check,
 * and the token ids, every logit of both sessions after every step, and both
 * saved states, hashed. Exits 1 if a check fails. */
#include "../ds4.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ds4_gpu_test_full_addr_expert_limit(uint32_t limit);
void ds4_gpu_test_force_selection(const int32_t *ids, uint32_t rows);
int ds4_gpu_test_expert_state(uint32_t layer, uint32_t expert);
void ds4_gpu_expert_directory_counts(unsigned long long out[8]);
void ds4_gpu_test_lifecycle(unsigned long long out[14]);
void ds4_gpu_memory_budget_counts(unsigned long long out[6]);
void ds4_gpu_mtl4_health(unsigned long long out[8]);
void ds4_gpu_stage_counts(unsigned long long out[5]);

enum { DIR_PINNED = 3, LAYERS = 40 };

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
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

static void holdings(const char *when) {
    unsigned long long l[14], b[6], h[8];
    ds4_gpu_test_lifecycle(l);
    ds4_gpu_memory_budget_counts(b);
    ds4_gpu_mtl4_health(h);
    const double G = 1073741824.0;
    printf("%-8s allocated %.2f GiB (budget %.2f, peak %.2f) | cache %llu entries in %llu slots | "
           "expert set %llu, queue sets %llu, contexts %llu | residency held %llu\n",
           when, l[0] / G, b[0] / G, b[2] / G, l[8], l[9], l[4], l[6], l[7], h[3]);
}

static volatile sig_atomic_t g_stop;
static void on_term(int sig) { (void)sig; g_stop = 1; }

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf window [steps] [cache_experts] [forced]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGTERM, on_term);
    const uint32_t window = (uint32_t)atoi(argv[2]);
    const int steps = argc > 3 ? atoi(argv[3]) : 16;
    const unsigned cache = argc > 4 ? (unsigned)atoi(argv[4]) : 10268;
    const int forced = argc > 5 ? atoi(argv[5]) != 0 : 1;
    const int rows = argc > 6 && atoi(argv[6]) == 1 ? 1 : 2;
    ds4_gpu_test_full_addr_expert_limit(window);
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = cache,
        .context_size = 4096, .placement_ctx_hint = 4096,
        .placement_session_count_hint = 2, .power_percent = 100,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 2;
    holdings("opened");
    const int vocab = ds4_engine_vocab_size(engine);
    float *buf = malloc((size_t)vocab * sizeof(float));
    ds4_tokens pa = {0}, pb = {0};
    ds4_chat_begin(engine, &pa);
    ds4_chat_append_message(engine, &pa, "user", "Explain how a B-tree insert works.");
    ds4_chat_append_assistant_prefix(engine, &pa, DS4_THINK_NONE);
    ds4_chat_begin(engine, &pb);
    ds4_chat_append_message(engine, &pb, "user", "Prove that the square root of two is irrational.");
    ds4_chat_append_assistant_prefix(engine, &pb, DS4_THINK_NONE);
    ds4_session *a = NULL, *b = NULL;
    char err[256] = "";
    int ok = ds4_session_create(&a, engine, 4096) == 0 && ds4_session_create(&b, engine, 4096) == 0 &&
             ds4_session_sync(a, &pa, err, sizeof(err)) == 0 &&
             ds4_session_sync(b, &pb, err, sizeof(err)) == 0;

    /* The forced routing, and what the directory must report for it: the
     * entries pinned now stay pinned, so a forced expert is a pinned hit in
     * every step exactly when its entry is pinned here. */
    /* A single row takes three of each, so one dispatch reads pinned and
     * cached experts together. */
    int32_t ids[12];
    for (int k = 0; k < 6; k++) { ids[k] = k; ids[6 + k] = 378 + k; }
    if (rows == 1) for (int k = 3; k < 6; k++) ids[k] = 375 + k;
    const int n_ids = rows == 1 ? 6 : 12;
    unsigned long long want_pinned = 0, pinned_entries = 0;
    for (uint32_t l = 0; l < LAYERS; l++)
        for (uint32_t x = 0; x < 384; x++) {
            const int st = ds4_gpu_test_expert_state(l, x);
            if (st != DIR_PINNED) continue;
            pinned_entries++;
            for (int k = 0; k < n_ids; k++) if ((uint32_t)ids[k] == x) want_pinned++;
        }
    printf("window   %u asked; %llu entries pinned; forced routing %s\n", window, pinned_entries,
           !forced ? "off" : rows == 1 ? "0..2 and 378..380" : "rows 0..5 and 378..383");
    if (forced) ds4_gpu_test_force_selection(ids, (uint32_t)rows);
    unsigned long long d0[8], d1[8], s0[5], s1[5];
    ds4_gpu_expert_directory_counts(d0);
    ds4_gpu_stage_counts(s0);

    uint64_t ha = 1469598103934665603ull, hb = ha, la = ha, lb = ha;
    int n = 0;
    for (; ok && !g_stop && n < steps; n++) {
        ds4_decode_item items[2] = { { a, ds4_session_argmax(a) }, { b, ds4_session_argmax(b) } };
        ha = fnv(ha, &items[0].token, sizeof(int));
        hb = fnv(hb, &items[1].token, sizeof(int));
        ok = rows == 1 ? ds4_session_eval(a, items[0].token, err, sizeof(err)) == 0
                       : ds4_sessions_eval_batch(items, 2, err, sizeof(err)) == 0;
        if (ok && ds4_session_copy_logits(a, buf, vocab) == vocab) la = fnv(la, buf, (size_t)vocab * sizeof(float));
        if (ok && ds4_session_copy_logits(b, buf, vocab) == vocab) lb = fnv(lb, buf, (size_t)vocab * sizeof(float));
    }
    ds4_gpu_expert_directory_counts(d1);
    ds4_gpu_stage_counts(s1);
    ds4_gpu_test_force_selection(NULL, 0);
    if (!ok) printf("failed   step %d: %s\n", n, err);

    const unsigned long long pinned_hits = d1[0] - d0[0], cache_hits = d1[1] - d0[1];
    const unsigned long long loads = d1[2] - d0[2], mismatch = d1[3] - d0[3];
    const unsigned long long nonzero = d1[4] - d0[4], missing = d1[5] - d0[5];
    const unsigned long long want_pin = forced ? want_pinned * (unsigned long long)n : 0;
    const unsigned long long want_rest = forced ? (unsigned long long)n_ids * LAYERS * (unsigned long long)n - want_pin : 0;
    int dir_ok = mismatch == 0 && nonzero == 0 && missing == 0 && d1[6] == pinned_entries;
    if (forced) dir_ok = dir_ok && pinned_hits == want_pin && cache_hits + loads == want_rest;
    /* A single row goes through the staged path instead: every layer of every
     * step stages its forced ids once, four copies a compact row and one an id
     * row, and nothing is refused. */
    const unsigned long long st_compact = s1[0] - s0[0], st_ids = s1[1] - s0[1];
    const unsigned long long st_rej = s1[2] - s0[2], st_copies = s1[3] - s0[3];
    if (rows == 1) {
        /* And since the process began: compact rows are only taken before the
         * split path is ready, in the first tokens, outside the decode above. */
        const int st_ok = st_rej == 0 && st_copies == 4ull * st_compact + st_ids &&
                          s1[2] == 0 && s1[3] == 4ull * s1[0] + s1[1] &&
                          (!forced || st_ids == (unsigned long long)LAYERS * (unsigned long long)n);
        printf("staging  over %d steps: %llu compact rows, %llu id rows (want %llu), %llu refused, "
               "%llu copy nodes (want %llu) -> %s\n", n, st_compact, st_ids,
               forced ? (unsigned long long)LAYERS * (unsigned long long)n : st_ids, st_rej,
               st_copies, 4ull * st_compact + st_ids, st_ok ? "PASS" : "FAIL");
        printf("staging  since open: %llu compact rows, %llu id rows, %llu refused, %llu copy nodes "
               "(want %llu)\n", s1[0], s1[1], s1[2], s1[3], 4ull * s1[0] + s1[1]);
        dir_ok = dir_ok && st_ok;
    }
    printf("directory over %d steps: pinned hits %llu (want %llu), cache hits %llu + loads %llu "
           "(want %llu); address mismatches %llu, missing with a slot %llu, selected and missing "
           "%llu; %llu entries still pinned -> %s\n", n, pinned_hits, want_pin, cache_hits, loads,
           want_rest, mismatch, nonzero, missing, d1[6], dir_ok ? "PASS" : "FAIL");

    uint64_t sa = 0, sb = 0;
    const int states = a && payload_hash(a, &sa) && b && payload_hash(b, &sb);
    printf("ids      %d steps: a %016llx b %016llx | logits a %016llx b %016llx | state a %016llx "
           "b %016llx%s\n", n, (unsigned long long)ha, (unsigned long long)hb,
           (unsigned long long)la, (unsigned long long)lb, (unsigned long long)sa,
           (unsigned long long)sb, states ? "" : " STATE-FAILED");
    ds4_session_free(a);
    ds4_session_free(b);
    ds4_tokens_free(&pa);
    ds4_tokens_free(&pb);
    free(buf);
    ds4_engine_close(engine);
    holdings("closed");
    if (g_stop) return 4;
    return ok && dir_ok && states ? 0 : 1;
}
