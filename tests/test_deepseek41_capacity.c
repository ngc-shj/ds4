/* One long prompt at one expert-cache size, every logit written out.
 *
 *   test_deepseek41_capacity <model.gguf> <story.txt> <cache_experts> <steps> <logits.out> [bytes]
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
 * each. */
#include "../ds4.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
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
    int ok = ds4_session_create(&s, engine, 8192) == 0 &&
             ds4_session_sync(s, &prompt, err, sizeof(err)) == 0;
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
    printf("capacity %u experts: prompt %d tokens, %d evaluations written, ids %016llx, "
           "logits %016llx%s%s\n", cache, prompt.len, i + (ok ? 1 : 0),
           (unsigned long long)ids, (unsigned long long)lh, ok ? "" : " FAILED: ", ok ? "" : err);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    free(buf);
    free(story);
    ds4_engine_close(engine);
    return ok && !g_stop ? 0 : 1;
}
