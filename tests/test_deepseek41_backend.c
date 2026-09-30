/* The V4.1 Metal 4 backend on the real model, in one process.
 *
 *   test_deepseek41_backend <model.gguf> [story.txt]
 *
 * Run with the production environment (bench/m4prod.sh's), so the gated
 * decode, the expert cache and its repairs are the ones the server runs.
 *
 * Two sessions take the same prompt and step together, one evaluation each in
 * turn, and after every evaluation their logits are compared whole with
 * memcmp: two sessions sharing the process's scratch, expert cache and queue
 * must not see each other. Each case prints the hash of its token ids, which
 * the harness compares with the hashes the Metal 3 oracle produced before it
 * was removed (bench/m4inproc.sh). The cases cover a short prompt decoded past
 * the raw KV ring and the compression boundaries, and a long prompt whose
 * prefill crosses the prefill chunk.
 *
 * Then the failure path: a Metal 4 command buffer reported failed must fail
 * the evaluation that owned it, and a fresh session straight after must give
 * the reference tokens. Last, a whole request's worth of Metal 4 work must
 * have noted no Metal 3 fallback. */
#include "../ds4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ds4_gpu_mtl4_inject_failure(void);
unsigned long long ds4_gpu_backend_m3_fallback(void);
void ds4_gpu_backend_m3_after_selection(unsigned long long out[4]);
void ds4_gpu_mtl4_counts(unsigned long long *segments, unsigned long long *dispatches,
                         unsigned long long *m3_fallback);
void ds4_gpu_mtl4_structure(unsigned long long out[16]);
int ds4_gpu_warm_command_queue(void);
unsigned long long ds4_gpu_full_addr_table_uses(void);
void ds4_gpu_stream_expert_cache_counters(uint64_t *hits, uint64_t *misses, uint64_t *evictions,
                                          uint32_t *entries, uint32_t *budget);
int ds4_gpu_test_warm_model_views(void);
void ds4_gpu_backend_report(const char *tag, unsigned long long tokens);
void ds4_gpu_mtl4_health(unsigned long long out[8]);

static int g_failures;

#define CHECK(cond, ...) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); g_failures++; } \
} while (0)

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

static void chat_prompt(ds4_engine *e, const char *text, ds4_tokens *out) {
    ds4_chat_begin(e, out);
    ds4_chat_append_message(e, out, "user", text);
    ds4_chat_append_assistant_prefix(e, out, DS4_THINK_NONE);
}

/* Logits of both sessions, compared whole. Returns 1 if identical. */
static int same_logits(ds4_session *a, ds4_session *b, float *la, float *lb, int vocab) {
    const int na = ds4_session_copy_logits(a, la, vocab);
    const int nb = ds4_session_copy_logits(b, lb, vocab);
    return na == vocab && nb == vocab && memcmp(la, lb, (size_t)vocab * sizeof(float)) == 0;
}

/* One prompt, two sessions in step. Writes the first session's token ids to ref. */
static void paired(ds4_engine *e, const char *name, const char *text, int steps,
                   int vocab, int *ref) {
    ds4_tokens prompt = {0};
    chat_prompt(e, text, &prompt);
    ds4_session *a = NULL, *b = NULL;
    char err[256];
    CHECK(ds4_session_create(&a, e, 8192) == 0 && ds4_session_create(&b, e, 8192) == 0,
          "%s: session create", name);
    if (!a || !b) return;
    float *la = malloc((size_t)vocab * sizeof(float));
    float *lb = malloc((size_t)vocab * sizeof(float));
    CHECK(ds4_session_sync(a, &prompt, err, sizeof(err)) == 0, "%s: prefill A: %s", name, err);
    CHECK(ds4_session_sync(b, &prompt, err, sizeof(err)) == 0, "%s: prefill B: %s", name, err);
    int diff_steps = 0, first_diff = -1, token_diff = -1;
    if (!same_logits(a, b, la, lb, vocab)) { diff_steps++; first_diff = 0; }
    uint64_t ha = 1469598103934665603ull, hb = ha;
    int n = 0;
    for (; n < steps; n++) {
        const int ta = ds4_session_argmax(a), tb = ds4_session_argmax(b);
        if (ref) ref[n] = ta;
        ha = fnv(ha, &ta, sizeof(ta));
        hb = fnv(hb, &tb, sizeof(tb));
        if (ta != tb && token_diff < 0) token_diff = n;
            if (ds4_session_eval(a, ta, err, sizeof(err)) != 0) {
            CHECK(0, "%s: eval A at %d: %s", name, n, err);
            break;
        }
            if (ds4_session_eval(b, tb, err, sizeof(err)) != 0) {
            CHECK(0, "%s: eval B at %d: %s", name, n, err);
            break;
        }
        if (!same_logits(a, b, la, lb, vocab)) {
            diff_steps++;
            if (first_diff < 0) first_diff = n + 1;
        }
    }
    printf("%-10s prompt %5d tokens, %5d steps: logits differ at %d of %d evaluations "
           "(first %d), tokens first differ at %d, ids %016llx / %016llx\n",
           name, prompt.len, n, diff_steps, n + 1, first_diff, token_diff,
           (unsigned long long)ha, (unsigned long long)hb);
    CHECK(diff_steps == 0 && token_diff < 0 && ha == hb, "%s: sessions disagree", name);
    free(la); free(lb);
    ds4_session_free(a);
    ds4_session_free(b);
    ds4_tokens_free(&prompt);
}

/* A reported failure fails its evaluation, and the next session is clean. */
static void injected(ds4_engine *e, const char *text, const int *ref, int nref) {
    ds4_tokens prompt = {0};
    chat_prompt(e, text, &prompt);
    char err[256];
    const unsigned long long fb0 = ds4_gpu_backend_m3_fallback();
    ds4_session *c = NULL;
    CHECK(ds4_session_create(&c, e, 8192) == 0, "inject: session create");
    CHECK(ds4_session_sync(c, &prompt, err, sizeof(err)) == 0, "inject: prefill: %s", err);
    int t = ds4_session_argmax(c);
    for (int i = 0; i < 8; i++) {
        CHECK(ds4_session_eval(c, t, err, sizeof(err)) == 0, "inject: eval %d: %s", i, err);
        t = ds4_session_argmax(c);
    }
    ds4_gpu_mtl4_inject_failure();
    err[0] = '\0';
    const int rc = ds4_session_eval(c, t, err, sizeof(err));
    printf("injected   decode failure: eval returned %d (%s)\n", rc, err[0] ? err : "no message");
    CHECK(rc != 0, "inject: a failed command buffer passed as a successful evaluation");
    ds4_session_free(c);

    /* The same failure during a prefill. */
    ds4_session *p = NULL;
    CHECK(ds4_session_create(&p, e, 8192) == 0, "inject: session create");
    ds4_gpu_mtl4_inject_failure();
    err[0] = '\0';
    const int prc = ds4_session_sync(p, &prompt, err, sizeof(err));
    printf("injected   prefill failure: sync returned %d (%s)\n", prc, err[0] ? err : "no message");
    CHECK(prc != 0, "inject: a failed prefill passed");
    ds4_session_free(p);

    /* A fresh session after both failures gives the reference answer. */
    ds4_session *d = NULL;
    CHECK(ds4_session_create(&d, e, 8192) == 0, "after: session create");
    CHECK(ds4_session_sync(d, &prompt, err, sizeof(err)) == 0, "after: prefill: %s", err);
    int agree = 0;
    for (int i = 0; i < nref; i++) {
        const int tok = ds4_session_argmax(d);
        if (tok != ref[i]) break;
        agree++;
        if (ds4_session_eval(d, tok, err, sizeof(err)) != 0) break;
    }
    printf("after      a fresh session matches the reference for %d of %d tokens\n",
           agree, nref);
    CHECK(agree == nref, "after: fresh session differs from the reference");
    ds4_session_free(d);
    const unsigned long long fb = ds4_gpu_backend_m3_fallback() - fb0;
    printf("fallback   Metal 3 work noted under the Metal 4 backend: %llu\n", fb);
    CHECK(fb == 0, "Metal 3 fallback under Metal 4");
    ds4_tokens_free(&prompt);
}

/* An engine opened after another was closed in the same process: it must
 * choose its backend again and give the same answer, with nothing carried
 * over from the engine before it. */
static void reopened(const ds4_engine_options *opt, const char *text, const int *ref, int nref) {
    ds4_engine *e = NULL;
    CHECK(ds4_engine_open(&e, opt) == 0, "reopen: engine open");
    if (!e) return;
    ds4_tokens prompt = {0};
    chat_prompt(e, text, &prompt);
    char err[256];
    ds4_session *s = NULL;
    CHECK(ds4_session_create(&s, e, 8192) == 0, "reopen: session create");
    CHECK(ds4_session_sync(s, &prompt, err, sizeof(err)) == 0, "reopen: prefill: %s", err);
    int agree = 0;
    for (int i = 0; i < nref; i++) {
        const int tok = ds4_session_argmax(s);
        if (tok != ref[i]) break;
        agree++;
        if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) break;
    }
    printf("reopen     an engine opened after the first closed matches the reference for "
           "%d of %d tokens\n", agree, nref);
    CHECK(agree == nref, "reopen: differs from the reference");
    unsigned long long h[8];
    ds4_gpu_mtl4_health(h);
    CHECK(h[0] == h[1], "reopen: commits and feedback received differ");
    CHECK(ds4_gpu_backend_m3_fallback() == 0, "reopen: Metal 3 work");
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    ds4_gpu_mtl4_health(h);
    printf("closed     after the second close: residency held %llu entries\n", h[3]);
    CHECK(h[3] == 0, "residency held after close");
}

static char *read_prefix(const char *path, size_t bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *buf = calloc(1, bytes + 1);
    const size_t n = fread(buf, 1, bytes, f);
    fclose(f);
    buf[n] = '\0';
    return buf;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [story.txt]\n", argv[0]);
        return 2;
    }
    const int steps = getenv("STEPS") ? atoi(getenv("STEPS")) : 2048;
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = 10268,
        .context_size = 8192, .placement_ctx_hint = 8192,
        .placement_session_count_hint = 2, .power_percent = 100,
    };
    if (ds4_engine_open(&engine, &opt) != 0) return 2;
    const int vocab = ds4_engine_vocab_size(engine);
    /* The model-view warmup runs only with full residency, never in this
     * streaming configuration; driven here, it has to go through Metal 4. */
    unsigned long long m3a[4], m3b[4], c0 = 0, c1 = 0, st0[16], st1[16];
    ds4_gpu_backend_m3_after_selection(m3a);
    ds4_gpu_mtl4_counts(&c0, NULL, NULL);
    ds4_gpu_mtl4_structure(st0);
    const int warmed = ds4_gpu_test_warm_model_views();
    ds4_gpu_backend_m3_after_selection(m3b);
    ds4_gpu_mtl4_counts(&c1, NULL, NULL);
    ds4_gpu_mtl4_structure(st1);
    const unsigned long long enc = (st1[0] + st1[1]) - (st0[0] + st0[1]);
    printf("warmup     model views touched: %s; Metal 4 commits +%llu, encoders +%llu; "
           "Metal 3 commands +%llu\n", warmed ? "ok" : "FAILED", c1 - c0, enc, m3b[0] - m3a[0]);
    CHECK(warmed && c1 > c0 && enc > 0 && m3b[0] == m3a[0], "warmup did not run on Metal 4 only");
    /* A Metal 3-only path under Metal 4 is refused, counted once, and runs
     * nothing. */
    const int warmq = ds4_gpu_warm_command_queue();
    unsigned long long m3c[4];
    ds4_gpu_backend_m3_after_selection(m3c);
    printf("refusal    a Metal 3-only call returned %d; refusals +%llu, Metal 3 commands +%llu\n",
           warmq, m3c[1] - m3b[1], m3c[0] - m3b[0]);
    CHECK(warmq == 0 && m3c[1] - m3b[1] == 1 && m3c[0] == m3b[0], "refusal not counted exactly once");
    int *ref = calloc((size_t)steps, sizeof(int));
    uint64_t hits = 0, misses = 0;

    paired(engine, "short", "Explain how a modern CPU executes an instruction, from fetch "
           "to retire, in as much detail as you can.", steps, vocab, ref);
    paired(engine, "haiku", "俳句を三つ、季語を変えて詠んでください。", 256, vocab, NULL);
    if (argc > 2) {
        char *story = read_prefix(argv[2], 22000);
        if (story) {
            char *text = malloc(strlen(story) + 64);
            sprintf(text, "%s\n\nSummarize the story so far.", story);
            paired(engine, "long", text, 256, vocab, NULL);
            free(text);
            free(story);
        }
    }
    injected(engine, "Explain how a modern CPU executes an instruction, from fetch "
             "to retire, in as much detail as you can.", ref, steps < 128 ? steps : 128);
    unsigned long long h[8];
    ds4_gpu_mtl4_health(h);
    printf("health     Metal 4: %llu commits, %llu feedback, %llu errors (the injected ones), "
           "residency held %llu entries %.1f MiB, peak %.1f MiB, %llu adds, %llu releases\n",
           h[0], h[1], h[2], h[3], (double)h[4] / 1048576.0, (double)h[7] / 1048576.0,
           h[5], h[6]);
    CHECK(h[0] == h[1], "commits and feedback received differ");
    CHECK(ds4_gpu_backend_m3_fallback() == 0, "Metal 3 work over the whole run");
    ds4_gpu_backend_report("test", 0);
    /* Read while the first engine's cache still exists: closing it clears
     * the cache and may clear what it counted. */
    ds4_gpu_stream_expert_cache_counters(&hits, &misses, NULL, NULL, NULL);
    ds4_engine_close(engine);
    reopened(&opt, "Explain how a modern CPU executes an instruction, from fetch "
             "to retire, in as much detail as you can.", ref, steps);
    free(ref);
    unsigned long long m3f[4];
    ds4_gpu_backend_m3_after_selection(m3f);
    printf("m3api      Metal 3 command execution after Metal 4 selection: %llu; refused %llu "
           "(the deliberate one); Metal 3 queues "
           "created before selection %llu, after %llu\n",
           m3f[0], m3f[1], m3f[2], m3f[3]);
    /* No Metal 3 queue at all: the queue is made only when the Metal 3
     * backend first needs one, and V4.1 selects Metal 4 before that. */
    /* The full expert address table runs on Metal 4 like everything else;
     * the single-row decode never consults it, so it is used only when the
     * environment enables it and a path that asks for it runs. The one
     * refusal is the deliberate Metal 3-only call above. */
    const unsigned long long full_uses = ds4_gpu_full_addr_table_uses();
    printf("fulltable  full expert address table uses %llu; expert cache hits %llu, misses %llu\n",
           full_uses, (unsigned long long)hits, (unsigned long long)misses);
    CHECK(hits + misses > 0, "the ordinary expert-cache path did not engage");
    CHECK(m3f[0] == 0 && m3f[1] == 1 && m3f[2] == 0 && m3f[3] == 0,
          "Metal 3 after selection, or queue counts not as expected");
    printf("%s\n", g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}
