/* What a V4.1 build computes, reduced to hashes that two builds can compare.
 *
 *   test_deepseek41_state <model.gguf> [story.txt [parts]]
 *
 * With `parts`, every Metal 4 context is encoded in that many command buffers
 * (where its graph has that many barriers to cut at) and submitted together:
 * the hashes must not move.
 *
 * Run with the production environment. For each case it prints the hash of the
 * token ids, the hash of every logit of every evaluation (the prefill's and
 * each decode step's, all of the vocabulary, as bits), and the hash and size
 * of the session's saved payload at the end - the KV cache and every piece of
 * state a later evaluation carries forward. Nothing is compared here: the same
 * source is built in the baseline tree and in the candidate, and the lines are
 * diffed.
 *
 * Cases: three prompts decoded greedily (one to 2048 tokens, one long prompt
 * whose prefill crosses the prefill chunk); two sessions stepped alternately;
 * and a conversation continued over three consecutive requests on one
 * session, each request syncing the whole transcript so far. */
#include "../ds4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

typedef struct {
    uint64_t ids, logits;
    int evals, tokens;
} trace;

static void trace_init(trace *t) {
    t->ids = t->logits = 1469598103934665603ull;
    t->evals = t->tokens = 0;
}

static int note_logits(ds4_session *s, trace *t, float *buf, int vocab) {
    if (ds4_session_copy_logits(s, buf, vocab) != vocab) return 0;
    t->logits = fnv(t->logits, buf, (size_t)vocab * sizeof(float));
    t->evals++;
    return 1;
}

static int step(ds4_session *s, trace *t, float *buf, int vocab, int *out) {
    char err[256];
    const int tok = ds4_session_argmax(s);
    t->ids = fnv(t->ids, &tok, sizeof(tok));
    t->tokens++;
    if (out) *out = tok;
    if (ds4_session_eval(s, tok, err, sizeof(err)) != 0) {
        fprintf(stderr, "eval failed: %s\n", err);
        return 0;
    }
    return note_logits(s, t, buf, vocab);
}

static int payload_hash(ds4_session *s, uint64_t *hash, uint64_t *bytes) {
    char err[256];
    FILE *fp = tmpfile();
    if (!fp) return 0;
    if (ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) {
        fprintf(stderr, "payload: %s\n", err);
        fclose(fp);
        return 0;
    }
    *bytes = (uint64_t)ftell(fp);
    rewind(fp);
    uint64_t h = 1469598103934665603ull;
    unsigned char chunk[1 << 16];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0) h = fnv(h, chunk, n);
    fclose(fp);
    *hash = h;
    return 1;
}

static void report(const char *name, const trace *t, ds4_session *s) {
    uint64_t ph = 0, pb = 0;
    const int ok = payload_hash(s, &ph, &pb);
    printf("%-12s tokens %5d evals %5d ids %016llx logits %016llx state %016llx (%llu bytes)%s\n",
           name, t->tokens, t->evals, (unsigned long long)t->ids,
           (unsigned long long)t->logits, (unsigned long long)ph, (unsigned long long)pb,
           ok ? "" : " STATE-FAILED");
}

static void chat_prompt(ds4_engine *e, const char *text, ds4_tokens *out) {
    ds4_chat_begin(e, out);
    ds4_chat_append_message(e, out, "user", text);
    ds4_chat_append_assistant_prefix(e, out, DS4_THINK_NONE);
}

static int single(ds4_engine *e, const char *name, const char *text, int steps,
                  float *buf, int vocab) {
    ds4_tokens prompt = {0};
    chat_prompt(e, text, &prompt);
    ds4_session *s = NULL;
    char err[256];
    trace t;
    trace_init(&t);
    int ok = ds4_session_create(&s, e, 8192) == 0 &&
             ds4_session_sync(s, &prompt, err, sizeof(err)) == 0 &&
             note_logits(s, &t, buf, vocab);
    for (int i = 0; ok && i < steps; i++) ok = step(s, &t, buf, vocab, NULL);
    if (s) report(name, &t, s);
    if (!ok) printf("%-12s FAILED\n", name);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    return ok;
}

/* Two sessions, one evaluation each in turn. */
static int alternating(ds4_engine *e, float *buf, int vocab) {
    ds4_tokens pa = {0}, pb = {0};
    chat_prompt(e, "Explain how a B-tree insert works, including node splitting.", &pa);
    chat_prompt(e, "Prove that the square root of two is irrational, carefully.", &pb);
    ds4_session *a = NULL, *b = NULL;
    char err[256];
    trace ta, tb;
    trace_init(&ta);
    trace_init(&tb);
    int ok = ds4_session_create(&a, e, 8192) == 0 && ds4_session_create(&b, e, 8192) == 0 &&
             ds4_session_sync(a, &pa, err, sizeof(err)) == 0 && note_logits(a, &ta, buf, vocab) &&
             ds4_session_sync(b, &pb, err, sizeof(err)) == 0 && note_logits(b, &tb, buf, vocab);
    for (int i = 0; ok && i < 384; i++)
        ok = step(a, &ta, buf, vocab, NULL) && step(b, &tb, buf, vocab, NULL);
    if (a) report("pair-a", &ta, a);
    if (b) report("pair-b", &tb, b);
    if (!ok) printf("pair         FAILED\n");
    ds4_session_free(a);
    ds4_session_free(b);
    ds4_tokens_free(&pa);
    ds4_tokens_free(&pb);
    return ok;
}

/* A conversation over consecutive requests on one session. Each request syncs
 * the whole transcript: the user turns plus what the model answered. */
static int conversation(ds4_engine *e, float *buf, int vocab) {
    static const char *turns[3] = {
        "Explain the TCP three-way handshake and why it needs three steps.",
        "Now explain what happens when the final ACK is lost.",
        "Summarize both answers in three sentences.",
    };
    ds4_session *s = NULL;
    char err[256];
    if (ds4_session_create(&s, e, 8192) != 0) return 0;
    ds4_tokens transcript = {0};
    ds4_chat_begin(e, &transcript);
    int ok = 1;
    for (int r = 0; ok && r < 3; r++) {
        trace t;
        trace_init(&t);
        ds4_chat_append_message(e, &transcript, "user", turns[r]);
        ds4_chat_append_assistant_prefix(e, &transcript, DS4_THINK_NONE);
        ok = ds4_session_sync(s, &transcript, err, sizeof(err)) == 0 && note_logits(s, &t, buf, vocab);
        for (int i = 0; ok && i < 160; i++) {
            int tok = 0;
            ok = step(s, &t, buf, vocab, &tok);
            if (ok) ds4_tokens_push(&transcript, tok);
        }
        char name[32];
        snprintf(name, sizeof(name), "turn-%d", r + 1);
        report(name, &t, s);
        if (ok) ds4_tokens_push(&transcript, ds4_token_eos(e));
    }
    if (!ok) printf("conversation FAILED: %s\n", err);
    ds4_tokens_free(&transcript);
    ds4_session_free(s);
    return ok;
}

static char *read_prefix(const char *path, size_t max) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char *b = malloc(max + 1);
    const size_t n = fread(b, 1, max, f);
    fclose(f);
    b[n] = 0;
    return b;
}

void ds4_gpu_test_mtl4_parts(unsigned parts) __attribute__((weak));

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.gguf [story.txt [parts]]\n", argv[0]); return 2; }
    if (argc > 3) {
        if (!ds4_gpu_test_mtl4_parts) { fprintf(stderr, "this build cannot split contexts\n"); return 2; }
        ds4_gpu_test_mtl4_parts((unsigned)atoi(argv[3]));
        printf("parts %s\n", argv[3]);
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = 10268,
        .context_size = 8192, .placement_ctx_hint = 8192,
        .placement_session_count_hint = 2, .power_percent = 100,
    };
    if (ds4_engine_open(&engine, &opt) != 0) return 2;
    const int vocab = ds4_engine_vocab_size(engine);
    float *buf = malloc((size_t)vocab * sizeof(float));
    int ok = 1;
    ok &= single(engine, "short", "Explain how a modern CPU executes an instruction, from "
                 "fetch to retire, in as much detail as you can.", 2048, buf, vocab);
    ok &= single(engine, "haiku", "俳句を三つ、季語を変えて詠んでください。", 256, buf, vocab);
    if (argc > 2) {
        char *story = read_prefix(argv[2], 22000);
        if (story) {
            char *text = malloc(strlen(story) + 64);
            sprintf(text, "%s\n\nSummarize the story so far.", story);
            ok &= single(engine, "long", text, 256, buf, vocab);
            free(text);
            free(story);
        }
    }
    ok &= alternating(engine, buf, vocab);
    ok &= conversation(engine, buf, vocab);
    free(buf);
    ds4_engine_close(engine);
    printf("%s\n", ok ? "DONE" : "FAILED");
    return ok ? 0 : 1;
}
