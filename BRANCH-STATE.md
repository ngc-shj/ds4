# perf/batched-decode-poc — Current State (Day-12)

This file is a single-page snapshot of "the conclusion". The original
exploration branch (code + commits + NOTES.md + memory) co-evolved
across 11 days and became hard to read for the questions "where are
we, what landed, what was tried and reverted, what's next." For
details follow the links to NOTES.md sections, memory files, and git
log.

**HEAD**: `b1f4dc7` (section 33h, working tree clean as of 2026-05-17)
**Branch-off from main**: 50 commits ahead

---

## 1. Baseline and current state

| Phase | Generation t/s | Source |
|---|---|---|
| End of Day-9 (CLI single, ctx=2048 gen=30) | **17.85** | NOTES.md section 28, memory `project-perf-branch-state` |
| End of Day-11 (same conditions, 5-run median) | **24.45** | section 33h, git log b1f4dc7 |
| **Cumulative gain** | **+37.0%** | |

### Measurement command (for reproduction)

```bash
make cuda-spark
for i in 1 2 3 4 5; do
  DS4_CUDA_Q4_DECODE=1 ./ds4 --cuda \
    -p "ad essa, abbiam voluto che" -n 30 --ctx 2048 --nothink 2>&1 \
    | grep "prefill:"
done
```

---

## 2. Landed essential commits (since Day-9 baseline)

Only the substantive fixes after the Day-9 baseline is established.
Docs-only commits and the reverts themselves are excluded.

### Day-9: stabilization (structural root fix for ctx>8K / pool=8 corruption)

| Commit | Change | NOTES section |
|---|---|---|
| `b176c7f` | lazy alloc: layer_attn_comp_cache | 27 |
| `8a863a5` | lazy alloc: raw_cache / state_kv / state_score + drop ctx-adaptive guard | 28 |
| `d02a270` | lazy alloc: layer_index_* (phase 3) + init-fill sync fence | 28 |
| `305bb53` | apply visibility-lag sync fence symmetrically to phase-2 ensure() helpers | 28 |
| `a22e6d2` | scatter-completion sync fixes N>=2 Q4 batch NaN race (D8-2 hypothesis corrected) | 24 |

→ Day-9 baseline of **17.85 t/s** established; pool=8 becomes the
default at all ctx.

### Day-10 / Day-11: single-decode speedup (+37% cumulative)

| # | Commit | Change | gen t/s | Section |
|---|---|---|---|---|
| 1 | `8f635ff` | server auto fast-path (cooldown-based gather budget) | (within variance) | 29 |
| 2 | `88e95fa` | skip shared FFN gate/up pair kernel at long ctx (D8-3 root fix) | clean | 30 |
| 3 | `f6704d9`+`3390b68`+`993e9cb` | B.4 multi-session attention kernel (env-gated, default OFF) | ~ | 32 |
| 4 | `bed1d1f` | B.4 third "fix" re-audit → no-op (defensive code retained) | — | 32 |
| 5 | `93bd5b8` | section 33c: auto-fast-path cold-start race fix. N=8 burst wall 18.60s → 3.27s = **5.7x** | — | 33c |
| 6 | `e765342` | section 33d: Q4 decode cache preload + dual Q8_0 GEMV | **21.40 (+19.9%)** | (memory) |
| 7 | `45fb079` | section 33e: wire Q4 dispatch into single-decode attn_output_a | **23.81 (+11.3%)** | (memory) |
| 8 | `5f4cc91` | section 33f: fused shared_gate_up_swiglu Q4 kernel (2-pass) | **23.97 (+0.9%)** | (memory) |
| 9 | `278c99f` | section 33g: F16→Q4 preload (358 tensors) + fill_f32 stream fix + Graphs scaffolding | **24.38 (+1.7%)** | (memory) |
| 10 | `b1f4dc7` | section 33h: kv_fp8_store_raw_one + grouped_q4_0_a_fused_quant fused kernels | **24.45 (+0.3%)** | (memory) |

NOTES.md sections 33d-33h live only in memory
`project-perf-branch-state` (not appended to NOTES.md).

---

## 3. Production recipe (single-user peak, 24.4 t/s)

```bash
DS4_CUDA_Q4_DECODE=1 \
DS4_SERVER_BATCH_MAX_RUNTIME=1 \
DS4_CUDA_BATCHED_QKV=1 \
DS4_CUDA_BATCHED_Q_B=1 \
DS4_CUDA_BATCHED_ATTN_OUTPUT=1 \
DS4_CUDA_BATCHED_SHARED_FFN=1 \
DS4_CUDA_BATCHED_ROUTED_MOE=1 \
  ./ds4-server --cuda --warm-weights --port 8000 --ctx 65536 \
    --kv-disk-dir /tmp/ds4-kv
```

Caveats:
- Mixed traffic up to N=2 sustains 24+ t/s even with `BATCH_MAX_RUNTIME=2`.
- pool=8 default falls into a degraded regime at session count >=4,
  pulling single-user throughput down to 9-15 t/s. (Section 29's
  "fast-path = pool=1" claim no longer holds in the current code base.)
- `DS4_CUDA_BATCHED_DECODE_ATTENTION=1` (B.4) is **default OFF**.
  Correctness is confirmed at N>=2 burst, but perf delta is not yet
  established (section 33b needs re-measurement).
- **Effective ctx limit on this Q4 quant is 65,536 (= native training
  context, no YARN scaling).**  The model's YARN-scaled theoretical
  max is 1,048,576 (65,536 × 16x), and `--ctx 1048576` boots and
  routes the KV cache through cudaMallocManaged ("CUDA using managed
  KV cache" log line), but YARN-extended positions show precision
  degradation on this Q4 quant: under sustained use at `--ctx 131072`
  or above, decode output drifts off-language (English → Chinese →
  Polish-like byte sequences) and the rendered bytes are not valid
  UTF-8 when concatenated, which causes streaming SDKs to throw
  `'utf-8' codec can't decode bytes ... invalid continuation byte` and
  abort.  Native `--ctx 65536` runs are stable under the same
  workload.  Use `--ctx 65536` for production; `--ctx 131072` and
  above are currently for experimentation only.

---

## 4. Dead-ends (tried and reverted; require new theory to revisit)

Details in [memory `project-dead-ends`](../../../home/noguchi/.claude/projects/-home-noguchi-ghq-github-com-antirez-ds4/memory/project_dead_ends.md).

| # | Attempt | Reason | Branch artifact |
|---|---|---|---|
| 1 | D8-5 batched shared_down | unfusing for batching → -40% regression | reverted in `bb27ebc` |
| 2 | B.4 scatter-fence sync | helper-level sync is counter-productive under the GB10 degraded regime | recorded in `66f0c79` |
| 3 | B.4 "third fix" raw_kv fallback | no-op due to wrapper short-circuit; initial toggle was GPU variance | corrected in `bed1d1f` |
| 4 | nvcc -O2 as default | variance effect 2.6x is dwarfed by allocator state 44x | section 33 |
| 5 | sm_89 PTX fallback as default | statistically indistinguishable from sm_121 | section 33a (`ds4*.sm89` retained) |
| 6 | B.4 preliminary perf claim | both binaries suspected of being measured in singleton dispatch; needs re-run after 33c fix | section 33b |
| 7 | getenv hot-path caching | Linux getenv is empirically negligible; lazy-init branch-predictor miss made it worse | (reverted) |
| 8 | single `__constant__ g_step_args` struct (33h precursor) | per-layer scalar clobber produced garbage output | (reverted) |
| 9 | **Day-12 33i**: `moe_down_sum6` midq → shared-mem collaborative load | midq=14 KiB is 100% L2-resident (24 MiB); barrier overhead cost -0.7% | reverted today, not committed |
| -- | Older (sections 1-10): templated weight-shared / cuBLAS f16 / L2 persist / full CUDA Graph / MTP at temp=0 / Q3_0 / Q3_K_S / INT8 TC | reverted after days of investigation | — |

---

## 5. Next priority (pick one)

**Candidate A (recommended): gateup kernel optimization**
- Surfaced by Day-12 33i: gateup 0.165 ms/call > down 0.106 ms/call
  (gateup is now the top stage).
- Target: `moe_gate_up_mid_decode_lut_qwarp32_kernel` (ds4_cuda.cu:10948).
- Env A/B (`MOE_TILE4`, `NO_DECODE_LUT_GATE`, `NO_DIRECT_DOWN_SUM6`)
  is exhausted → audit kernel-internal occupancy / warp scheduling next.
- Expected +5% decode, single-day effort.

**Candidate B: finish CUDA Graphs (continuation of 33g)**
- Refactor 27 kernels onto a per-layer `g_step_args[DS4_N_LAYER]` array.
- Expected +50% (encode 27 ms → ~1 ms); multi-day with high
  roll-back cost on failure.
- WIP reference: `5ae22be` on `perf/cuda-graph-wip`.

**Candidate C: re-measure B.4 perf (post 33c fix)**
- Run `speed-bench/b4-perf-ab.sh` plain on cold boot.
- Expected: the true effect becomes visible (33b was unfair because
  both sides were singleton-dispatched).

---

## 6. Open follow-ups (low priority / large effort)

- **B.3** fused per-session pre-attention (collapse
  head_rms + rope + kv_store + compressor_qkv into one launch).
- **C** multi-stream pipeline (lift the `g_kernel_stream = 0` constraint).
- **E** batched fused shared_down + HC expand in a single kernel
  (carry over the section-23 lesson).
- **F** per-session 2.3 GiB × 8 = 18 GiB memory-footprint reduction.
- **G** extend visibility-lag protection to other allocations
  (generalize the section-28 pattern).

---

## 7. Known sm_121 / GB10 platform issues

| Issue | URL |
|---|---|
| nvcc -O3 sm_121 miscompilation | https://github.com/ggml-org/llama.cpp/issues/18331 |
| DGX Spark `cudaMemGetInfo "Not Supported"` | https://docs.nvidia.com/dgx/dgx-spark/known-issues.html |
| DGX Spark "zombie" (unified-memory pressure) | https://forums.developer.nvidia.com/t/.../353752 |
| Memory creep on DGX Spark | https://forums.developer.nvidia.com/t/.../364886 |
| sm_121 + CUDA Graphs illegal memory access | https://github.com/sgl-project/sglang/issues/19799 |

---

## 8. Variance hierarchy (settled at Day-11)

| Rank | Factor | Magnitude |
|---|---|---|
| 1 (dominant) | GPU/allocator state continuity (clean post-idle vs degraded post-burst) | **~44x** |
| 2 (secondary) | nvcc opt-level (-O3 vs -O2) | ~2.6x |
| 3 (negligible) | Architecture target (sm_121 vs sm_89) | ~1.0x |

→ Tweaking build flags as a workaround is futile. The operational
workaround is to insert a 2-3 minute GPU idle between runs.
