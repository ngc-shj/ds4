/* One long prompt at one expert-cache size, every logit written out.
 *
 *   test_deepseek41_capacity <model.gguf> <story.txt> <cache_experts> <steps> <logits.out> [bytes [token-major] [parts=N]]
 *
 * The expert cache is a performance state: what it holds, and how often it
 * misses, must not change a single logit. Run this once per cache size, each
 * in its own process through bench/gpurun.sh, and compare the files byte for
 * byte; the first differing evaluation is where to look. Uses the public API
 * only, so the same source links against the frozen baseline as well.
 *
 * The prompt is the backend test's long one: the story's first 22000 bytes
 * (or `bytes`) and a request to summarize it. A shorter one lets a cache that
 * fits the guard hold every expert the prompt touches, so the same binary can
 * be compared with and without evictions. A story argument of "text:..." is
 * the user message itself, for the backend test's short prompts. The file
 * holds the logits after prefill and after each greedy step, vocab floats
 * each; the line printed at the end also hashes the session's saved payload,
 * its KV cache and carried state, as test_deepseek41_state does. */
#include "../ds4.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void ds4_test_v41_token_major_tail(int on) __attribute__((weak));
void ds4_gpu_test_mtl4_parts(unsigned parts) __attribute__((weak));
void ds4_gpu_backend_report(const char *tag, unsigned long long tokens) __attribute__((weak));

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

static int payload_hash(ds4_session *s, uint64_t *hash, uint64_t *bytes) {
    char err[256];
    FILE *fp = tmpfile();
    if (!fp) return 0;
    if (ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) { fclose(fp); return 0; }
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

static volatile sig_atomic_t g_stop;
static void on_term(int sig) { (void)sig; g_stop = 1; }

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s model.gguf story.txt cache_experts steps logits.out\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGTERM, on_term);
    const unsigned cache = (unsigned)atoi(argv[3]);
    const int steps = atoi(argv[4]);
    const size_t want = argc > 6 ? (size_t)atol(argv[6]) : 22000;
    char *story = NULL;
    if (strncmp(argv[2], "text:", 5) == 0) {
        story = strdup(argv[2] + 5);
    } else {
        FILE *f = fopen(argv[2], "rb");
        if (!f) return 2;
        story = calloc(1, want + 64);
        const size_t n = fread(story, 1, want, f);
        fclose(f);
        strcpy(story + n, "\n\nSummarize the story so far.");
    }
    FILE *out = fopen(argv[5], "wb");
    if (!out) return 2;

    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = cache,
        .context_size = 8192, .placement_ctx_hint = 8192,
        .placement_session_count_hint = 2, .power_percent = 100,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 2;
    const int vocab = ds4_engine_vocab_size(engine);
    float *buf = malloc((size_t)vocab * sizeof(float));
    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", story);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    ds4_session *s = NULL;
    char err[256] = "";
    /* A seventh argument "token-major" runs a streaming append's tail as the
     * engine did before it was batched, for a cold time-to-first-token pair. */
    /* "parts=N" encodes every Metal 4 context in N parts where its graph
     * can be cut that often - N=1 is the single-part encoding. */
    for (int a = 7; a < argc; a++) {
        if (strcmp(argv[a], "token-major") == 0) {
            if (!ds4_test_v41_token_major_tail) return 2;
            ds4_test_v41_token_major_tail(1);
        } else if (strncmp(argv[a], "parts=", 6) == 0) {
            if (!ds4_gpu_test_mtl4_parts) return 2;
            ds4_gpu_test_mtl4_parts((unsigned)atoi(argv[a] + 6));
        } else return 2;
    }
    struct timespec t0, t1;
    int ok = ds4_session_create(&s, engine, 8192) == 0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    ok = ok && ds4_session_sync(s, &prompt, err, sizeof(err)) == 0;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (ok) printf("prefill  %d tokens in %.3f s (cold, this process's first)\n", prompt.len,
                   (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) * 1e-9);
    uint64_t ids = 1469598103934665603ull, lh = ids;
    int i = 0;
    for (; ok && !g_stop; i++) {
        ok = ds4_session_copy_logits(s, buf, vocab) == vocab &&
             fwrite(buf, sizeof(float), (size_t)vocab, out) == (size_t)vocab;
        if (!ok) break;
        lh = fnv(lh, buf, (size_t)vocab * sizeof(float));
        if (i == steps) break;
        const int t = ds4_session_argmax(s);
        ids = fnv(ids, &t, sizeof(t));
        ok = ds4_session_eval(s, t, err, sizeof(err)) == 0;
    }
    fclose(out);
    uint64_t sh = 0, sb = 0;
    if (ok) ok = payload_hash(s, &sh, &sb);
    printf("capacity %u experts: prompt %d tokens, %d evaluations written, ids %016llx, "
           "logits %016llx, state %016llx (%llu bytes)%s%s\n", cache, prompt.len, i + (ok ? 1 : 0),
           (unsigned long long)ids, (unsigned long long)lh, (unsigned long long)sh,
           (unsigned long long)sb, ok ? "" : " FAILED: ", ok ? "" : err);
    if (ds4_gpu_backend_report) ds4_gpu_backend_report("capacity", (unsigned long long)i);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    free(buf);
    free(story);
    ds4_engine_close(engine);
    return ok && !g_stop ? 0 : 1;
}
