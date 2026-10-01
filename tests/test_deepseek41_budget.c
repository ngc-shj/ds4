/* The memory budget with more sessions than the engine planned for.
 *
 *   test_deepseek41_budget <model.gguf> [sessions] [ctx] [steps] [cache_experts]
 *
 * Run through bench/gpurun.sh under the production environment. The engine is
 * opened for one session with the expert cache at the size asked (default
 * 10268, which the budget fits), prewarmed, so the cache already holds what the
 * budget allows. Then sessions are created and prefilled one at a time: each
 * must be paid for by the cache giving slabs back, or be refused, never by the
 * device going past the budget. All of them are then decoded together through
 * the native batch. Prints, at every point, what the device has allocated
 * against the budget, and at the end each session's token-id hash - the same
 * run at a cache size the budget never touches must give the same ids.
 *
 * SIGTERM asks it to stop between steps; it closes the engine and exits 4. */
#include "../ds4.h"

#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void ds4_gpu_test_lifecycle(unsigned long long out[14]);
void ds4_gpu_memory_budget_counts(unsigned long long out[6]);

static volatile sig_atomic_t g_stop;
static void on_term(int sig) { (void)sig; g_stop = 1; }

static int point(const char *when) {
    unsigned long long l[14], b[6];
    ds4_gpu_test_lifecycle(l);
    ds4_gpu_memory_budget_counts(b);
    const double G = 1073741824.0;
    const int over = b[0] != 0 && l[0] > b[0];
    printf("%-10s allocated %.2f GiB of a %.2f GiB budget (%.2f GiB reserve) | peak %.2f GiB | "
           "cache %llu entries in %llu slots, %.2f GiB | slabs given back %llu, allocations refused "
           "%llu, cache growth stopped %llu%s\n",
           when, l[0] / G, b[0] / G, b[1] / G, b[2] / G, l[8], l[9], l[11] / G, b[3], b[4], b[5],
           over ? " | OVER BUDGET" : "");
    return !over;
}

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [sessions] [ctx] [steps] [cache_experts]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    signal(SIGTERM, on_term);
    enum { MAX_SESSIONS = 16 };
    int n = argc > 2 ? atoi(argv[2]) : 4;
    const int ctx = argc > 3 ? atoi(argv[3]) : 8192;
    const int steps = argc > 4 ? atoi(argv[4]) : 32;
    const unsigned cache = argc > 5 ? (unsigned)atoi(argv[5]) : 10268;
    if (n < 1) n = 1;
    if (n > MAX_SESSIONS) n = MAX_SESSIONS;
    static const char *prompts[] = {
        "Explain how a B-tree insert works.",
        "Prove that the square root of two is irrational.",
        "Write a quicksort in C and explain the partition step.",
        "Describe how TCP congestion control reacts to loss.",
    };
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = cache,
        .context_size = ctx, .placement_ctx_hint = ctx,
        .placement_session_count_hint = 1, .power_percent = 100,
    };
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 2;
    int ok = point("opened");
    ds4_session *s[MAX_SESSIONS] = {0};
    ds4_tokens p[MAX_SESSIONS] = {{0}};
    uint64_t h[MAX_SESSIONS];
    char err[256] = "";
    int made = 0;
    for (int i = 0; i < n && !g_stop; i++) {
        ds4_chat_begin(engine, &p[i]);
        ds4_chat_append_message(engine, &p[i], "user", prompts[i % 4]);
        ds4_chat_append_assistant_prefix(engine, &p[i], DS4_THINK_NONE);
        h[i] = 1469598103934665603ull;
        char when[32];
        snprintf(when, sizeof(when), "session %d", i + 1);
        if (ds4_session_create(&s[i], engine, ctx) != 0) {
            printf("%-10s refused at create\n", when);
            point(when);
            break;
        }
        made = i + 1;
        if (ds4_session_sync(s[i], &p[i], err, sizeof(err)) != 0) {
            printf("%-10s refused at prefill: %s\n", when, err);
            ok &= point(when);
            made = i;
            ds4_session_free(s[i]);
            s[i] = NULL;
            break;
        }
        ok &= point(when);
    }
    int step = 0;
    for (; made > 0 && step < steps && !g_stop; step++) {
        ds4_decode_item items[MAX_SESSIONS];
        for (int i = 0; i < made; i++) {
            items[i].session = s[i];
            items[i].token = ds4_session_argmax(s[i]);
            h[i] = fnv(h[i], &items[i].token, sizeof(int));
        }
        if (ds4_sessions_eval_batch(items, made, err, sizeof(err)) != 0) {
            printf("failed     step %d: %s\n", step, err);
            ok = 0;
            break;
        }
    }
    ok &= point("decoded");
    for (int i = 0; i < made; i++)
        printf("ids        session %d, %d steps: %016llx\n", i + 1, step, (unsigned long long)h[i]);
    for (int i = 0; i < n; i++) {
        if (s[i]) ds4_session_free(s[i]);
        ds4_tokens_free(&p[i]);
    }
    ds4_engine_close(engine);
    point("closed");
    if (g_stop) return 4;
    return ok ? 0 : 1;
}
