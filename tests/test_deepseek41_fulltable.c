/* The full expert address table on Metal 4, brought up in stages.
 *
 *   test_deepseek41_fulltable <model.gguf> plan|run [experts] [cache_experts] [steps] [parts]
 *
 * Run with the production environment plus
 * DS4_METAL_ENABLE_V41_STREAMING_SESSION_BATCH=1,
 * DS4_METAL_ENABLE_STREAMING_FULL_EXPERT_ADDR_TABLE=1 and
 * DS4_METAL_FULL_ADDR_LAYERS=<n>.
 *
 *   plan   open the engine and print the working set, what is allocated
 *          against it, and how many full-table layers would fit; run nothing
 *   run    two sessions decoded together through the native batch for `steps`
 *          steps, giving only the first `experts` experts of each admitted
 *          layer buffers of their own (0: all of them), then the table's
 *          residency ledger, the Metal 4 feedback errors and the token ids
 *
 * The expert cache is sized by `cache_experts` (default 4096) so the table has
 * room. With `experts` below the layer's count the answer is not the model's:
 * the stage exists to load the residency, not to be compared. */
#include "../ds4.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mach/mach.h>

/* Weak, so the same source builds against the frozen Metal 3 tree, which has
 * none of these: there the ledger lines are skipped and the table runs as that
 * build runs it. */
void ds4_gpu_test_full_addr_admit(void) __attribute__((weak));
void ds4_gpu_test_full_addr_expert_limit(uint32_t limit) __attribute__((weak));
void ds4_gpu_test_working_set(unsigned long long *recommended, unsigned long long *allocated) __attribute__((weak));
void ds4_gpu_full_addr_counts(unsigned long long out[8]) __attribute__((weak));
void ds4_gpu_mtl4_health(unsigned long long out[8]) __attribute__((weak));
void ds4_gpu_test_mtl4_parts(unsigned parts) __attribute__((weak));

/* The Objective-C runtime's pool, from C: with a tenth argument each round
 * runs inside one, which tells a leak through autorelease apart from one
 * through a reference somebody still holds. */
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *pool);

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p;
    for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

/* What the process holds in physical memory, which every build can report. */
static double footprint_gib(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) return -1.0;
    return (double)info.phys_footprint / 1073741824.0;
}

static void ledger(const char *when) {
    printf("%-8s footprint %.2f GiB\n", when, footprint_gib());
    if (!ds4_gpu_full_addr_counts || !ds4_gpu_mtl4_health || !ds4_gpu_test_working_set) return;
    unsigned long long f[8], h[8], ws = 0, al = 0;
    ds4_gpu_full_addr_counts(f);
    ds4_gpu_mtl4_health(h);
    ds4_gpu_test_working_set(&ws, &al);
    printf("%-8s full table: %llu allocations held (%.2f GiB), set reports %llu; adds %llu, "
           "removes %llu, duplicate adds %llu; layers admitted %llu, refused %llu | Metal 4 "
           "feedback errors %llu of %llu | allocated %.2f of %.2f GiB\n",
           when, f[0], f[1] / 1073741824.0, f[7], f[2], f[3], f[4], f[5], f[6],
           h[2], h[1], al / 1073741824.0, ws / 1073741824.0);
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

/* Two sessions prefilled, then decoded together through the native batch:
 * token ids, every logit of both after every step, and both saved states. */
static int one_round(ds4_engine *engine, int steps, int round) {
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
    char err[256];
    int ok = ds4_session_create(&a, engine, 4096) == 0 && ds4_session_create(&b, engine, 4096) == 0 &&
             ds4_session_sync(a, &pa, err, sizeof(err)) == 0 &&
             ds4_session_sync(b, &pb, err, sizeof(err)) == 0;
    ledger("prefill");
    uint64_t ha = 1469598103934665603ull, hb = ha, la = ha, lb = ha;
    int n = 0;
    for (; ok && n < steps; n++) {
        ds4_decode_item items[2] = {
            { a, ds4_session_argmax(a) }, { b, ds4_session_argmax(b) },
        };
        ha = fnv(ha, &items[0].token, sizeof(int));
        hb = fnv(hb, &items[1].token, sizeof(int));
        ok = ds4_sessions_eval_batch(items, 2, err, sizeof(err)) == 0;
        if (ok && ds4_session_copy_logits(a, buf, vocab) == vocab)
            la = fnv(la, buf, (size_t)vocab * sizeof(float));
        if (ok && ds4_session_copy_logits(b, buf, vocab) == vocab)
            lb = fnv(lb, buf, (size_t)vocab * sizeof(float));
    }
    if (!ok) printf("failed   round %d step %d: %s\n", round, n, err);
    ledger("decoded");
    uint64_t sa = 0, sb = 0, za = 0, zb = 0;
    const int pa_ok = a && payload_hash(a, &sa, &za), pb_ok = b && payload_hash(b, &sb, &zb);
    printf("ids      round %d, %d steps: a %016llx b %016llx | logits a %016llx b %016llx | "
           "state a %016llx (%llu B) b %016llx (%llu B)%s\n", round, n,
           (unsigned long long)ha, (unsigned long long)hb, (unsigned long long)la,
           (unsigned long long)lb, (unsigned long long)sa, (unsigned long long)za,
           (unsigned long long)sb, (unsigned long long)zb, pa_ok && pb_ok ? "" : " STATE-FAILED");
    ds4_session_free(a);
    ds4_session_free(b);
    ds4_tokens_free(&pa);
    ds4_tokens_free(&pb);
    free(buf);
    return ok;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf plan|run [experts] [cache_experts] [steps] [parts] [rounds]\n",
                argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    const int run = strcmp(argv[2], "run") == 0;
    const unsigned experts = argc > 3 ? (unsigned)atoi(argv[3]) : 0;
    const unsigned cache = argc > 4 ? (unsigned)atoi(argv[4]) : 4096;
    const int steps = argc > 5 ? atoi(argv[5]) : 16;
    /* Every Metal 4 context encoded in this many command buffers; 0 leaves
     * the build's own rule. */
    if (argc > 6 && atoi(argv[6]) > 0 && ds4_gpu_test_mtl4_parts)
        ds4_gpu_test_mtl4_parts((unsigned)atoi(argv[6]));
    /* Engines opened, run and closed one after another in this process. */
    const int rounds = argc > 7 ? atoi(argv[7]) : 1;
    if (ds4_gpu_test_full_addr_admit) ds4_gpu_test_full_addr_admit();
    if (ds4_gpu_test_full_addr_expert_limit) ds4_gpu_test_full_addr_expert_limit(experts);
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = cache,
        .context_size = 4096, .placement_ctx_hint = 4096,
        .placement_session_count_hint = 2, .power_percent = 100,
    };
    int ok = 1;
    const int pooled = argc > 9 && atoi(argv[9]) != 0;
    for (int r = 0; r < (run ? rounds : 1); r++) {
        void *pool = pooled ? objc_autoreleasePoolPush() : NULL;
        ds4_engine *engine = NULL;
        if (ds4_engine_open(&engine, &opt) != 0) return 2;
        ledger("opened");
        const char *layers = getenv("DS4_METAL_FULL_ADDR_LAYERS");
        unsigned long long ws = 0, al = 0;
        if (ds4_gpu_test_working_set) ds4_gpu_test_working_set(&ws, &al);
        /* A V4.1 layer's experts: 384 of them, gate, up and down, 9.49 MiB each. */
        const double layer_gib = 384.0 * 9.49 / 1024.0;
        const double room = (double)ws / 1073741824.0 - (double)al / 1073741824.0 - 8.0;
        printf("plan     %s layers asked; a layer is %.2f GiB; room below the working set less "
               "8 GiB: %.2f GiB, %d layers\n", layers ? layers : "all", layer_gib, room,
               room > 0 ? (int)(room / layer_gib) : 0);
        if (run) ok &= one_round(engine, steps, r);
        ds4_engine_close(engine);
        if (pool) objc_autoreleasePoolPop(pool);
        ledger("closed");
        /* With a ninth argument, hold still after each close so the process
         * can be inspected from outside (vmmap, footprint). */
        if (argc > 8) { printf("paused   %d s after round %d\n", atoi(argv[8]), r); sleep((unsigned)atoi(argv[8])); }
    }
    return ok ? 0 : 1;
}
