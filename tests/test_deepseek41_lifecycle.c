/* One engine opened and closed, and what the process holds before and after.
 *
 *   test_deepseek41_lifecycle <model.gguf> [cache_experts] [steps] [pause_s] [pool] [rounds]
 *
 * Run through bench/gpurun.sh, never beside another GPU process. Prints, at
 * each point - before the engine opens, after it has decoded, after it has
 * closed, and after the autorelease pool the whole lifetime ran in has been
 * drained - the process's physical footprint and what the GPU side still
 * holds (ds4_gpu_test_lifecycle). With pause_s the process waits that long at
 * the end so its memory can be read from outside.
 *
 * pool 1 (the default) wraps each round in an autorelease pool the test drains;
 * pool 0 calls the engine as a plain C caller with none, so whatever the
 * engine leaves to a caller's pool shows up. rounds opens and closes that
 * many engines one after another; each round reports after its close, and
 * again five seconds later, when the driver has had time to unmap.
 *
 * By default only one engine is ever opened. Whether the close returned the memory is
 * judged from these lines, before anything opens a second engine.
 *
 * SIGTERM asks it to stop: the request is honoured as soon as control is back
 * in this program - after the engine has opened, or between decode steps -
 * by closing the engine and draining as usual, so a stopped run still reports
 * its close. It exits 4 then. */
#include "../ds4.h"

#include <mach/mach.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void ds4_gpu_test_lifecycle(unsigned long long out[14]);
void ds4_gpu_mtl4_health(unsigned long long out[8]);
void *objc_autoreleasePoolPush(void);
void objc_autoreleasePoolPop(void *pool);
void _objc_autoreleasePoolPrint(void);

static double footprint_gib(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        return -1.0;
    return (double)info.phys_footprint / 1073741824.0;
}

static volatile sig_atomic_t g_stop;
static void on_term(int sig) { (void)sig; g_stop = 1; }

static void point(const char *when) {
    unsigned long long l[14], h[8];
    ds4_gpu_test_lifecycle(l);
    ds4_gpu_mtl4_health(h);
    const double G = 1073741824.0;
    printf("%-9s footprint %.2f GiB | device allocated %.2f GiB | tensors %llu, model views %llu | "
           "slabs %llu, slots %llu (%llu free), entries %llu, cache %.2f GiB | expert set %llu "
           "(%.2f GiB), model set %llu (%.2f GiB), queue sets %llu, contexts %llu | context "
           "residency adds %llu releases %llu, held %llu (%.2f GiB) | commits %llu, feedback %llu, "
           "errors %llu\n",
           when, footprint_gib(), l[0] / G, l[1], l[2], l[3], l[9], l[10], l[8], l[11] / G,
           l[4], l[12] / G, l[5], l[13] / G, l[6], l[7], h[5], h[6], h[3], h[4] / G, h[0], h[1], h[2]);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [cache_experts] [steps] [pause_s]\n", argv[0]);
        return 2;
    }
    setvbuf(stdout, NULL, _IOLBF, 0);
    const unsigned cache = argc > 2 ? (unsigned)atoi(argv[2]) : 4096;
    const int steps = argc > 3 ? atoi(argv[3]) : 16;
    const int pause_s = argc > 4 ? atoi(argv[4]) : 0;
    const int use_pool = argc > 5 ? atoi(argv[5]) : 1;
    const int rounds = argc > 6 ? atoi(argv[6]) : 1;
    /* With a seventh argument, the thread's autorelease pools are printed to
     * stderr before each close: what the engine has left for a caller's pool
     * to release, object by object. */
    const int print_pool = argc > 7 && atoi(argv[7]) != 0;
    signal(SIGTERM, on_term);
    point("before");
    int ok = 1;
    for (int r = 0; r < rounds && !g_stop; r++) {
        void *pool = use_pool ? objc_autoreleasePoolPush() : NULL;
        ds4_engine_options opt = {
            .model_path = argv[1], .backend = DS4_BACKEND_METAL,
            .ssd_streaming = true, .ssd_streaming_cache_experts = cache,
            .context_size = 4096, .placement_ctx_hint = 4096,
            .placement_session_count_hint = 1, .power_percent = 100,
        };
        ds4_engine *engine = NULL;
        if (ds4_engine_open(&engine, &opt) != 0) return 2;
        point("opened");
        ds4_tokens prompt = {0};
        ds4_chat_begin(engine, &prompt);
        ds4_chat_append_message(engine, &prompt, "user", "Explain how a B-tree insert works.");
        ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
        ds4_session *s = NULL;
        char err[256];
        ok = !g_stop && ds4_session_create(&s, engine, 4096) == 0 &&
             ds4_session_sync(s, &prompt, err, sizeof(err)) == 0;
        for (int i = 0; ok && !g_stop && i < steps; i++)
            ok = ds4_session_eval(s, ds4_session_argmax(s), err, sizeof(err)) == 0;
        if (!ok) printf("failed    %s\n", err);
        if (g_stop) printf("stopping  on request, closing the engine\n");
        point("decoded");
        if (print_pool) _objc_autoreleasePoolPrint();
        ds4_session_free(s);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        point("closed");
        if (pool) { objc_autoreleasePoolPop(pool); point("drained"); }
        sleep(5);
        point("later");
        printf("round     %d done\n", r);
    }
    if (g_stop) return 4;
    if (pause_s > 0) {
        printf("paused    %d s\n", pause_s);
        sleep((unsigned)pause_s);
    }
    return ok ? 0 : 1;
}
