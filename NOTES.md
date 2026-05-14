# Batched-decode notes (perf/batched-decode-poc)

Hardware: NVIDIA GB10 / DGX Spark (sm_121, Blackwell).
Model: gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf

## Where we are after Phase 1c

Aggregate throughput, `ds4-bench --ctx-start 2048 --ctx-max 2048 --gen-tokens
50 --batch N`, with `DS4_CUDA_Q4_DECODE=1`:

| N  | t/s aggregate | × N=1 |
|----|---------------|-------|
| 1  | 18.12         | 1.00  |
| 2  | 18.75         | 1.03  |
| 4  | 18.97         | 1.05  |
| 8  | 19.36         | 1.07  |
| 16 | 16.14         | 0.89  |

The +5–7 % at N ≤ 8 comes from two things: the single-sync amortization in
Phase 1a, and implicit L2 reuse inside the naive Q4 batched warp kernel
used by the batched output head (grid layout `(out_dim/8, n_tok)` lets
later toks for the same row hit L2 while earlier toks are still in
flight).  Past N ≈ 8 the L2 working-set spills and things fall off.

The per-token wall clock is essentially the same as a serial-of-N tokens.
Phase 1c is structural / API; it does NOT add real kernel-level
parallelism for the dense layer matmuls, which is where the bulk of the
work is.

## Where we are after sections 11-13 (current head)

**Caveat (2026-05-13):** the percentages in this section are
`gen_tps_aggregate` from the bench tool -- warm-decode tokens per
second only.  They are NOT user-visible wall-clock speedups in
`ds4-server` until the request scheduler exercises the batched-decode
code path.  Until commit a608e4d the scheduler was FIFO serial, so N=4
parallel HTTP requests took the same wall-clock as N=4 sequential
requests (~36 s each at ctx=2048+50tok).  Section 15 documents the
discovery and the Day-1 2-slot continuous-batching MVP added to
`ds4-server`; section 16 documents the Day-2 follow-ups (N=4 depth +
per-slot SSE streaming + `g_kernel_stream` consistency + per-session
KV cache reuse, commits 228b62a..435e966) that close the gap between
the microbench numbers and what real chat workloads see.  N=2 parallel
HTTP chats with KV cache hits now take ~6 s wall-clock on the same
2466p+30c smoke (vs the original 36 s pre-a608e4d).  Sections 11-13
numbers below remain the **upper-bound microbench targets** for the
decode region; section 16 documents the per-feature wall-clock land
points.

Sections 11, 12, and 13 below add three engine-level **batched
substitutes** for the per-layer interleaved decode encoder:

  - `DS4_CUDA_BATCHED_Q_B`         replaces PRE_B  (attn_q_b matmul)
  - `DS4_CUDA_BATCHED_QKV`         replaces PRE_A2 (attn_q_a + attn_kv + qkv_rms_norm)
  - `DS4_CUDA_BATCHED_ATTN_OUTPUT` replaces PRE_C2 (attn_output_a + output_b)

Each stages N sessions' inputs into engine-level scratch, runs the
underlying kernel(s) once with `n_tok = N`, and scatters rows back to
per-session tensors.  Stacking all three:

| N | baseline median | ALL substitutes median | Δ |
|---|----------------:|----------------------:|--:|
| 4 | 19.62 t/s       | **25.48 t/s**         | **+29.9 %** (range 25.46-25.50, variance 0.04) |
| 8 | 15.62 t/s       | **24.60 t/s**         | **+57.5 %** (5-run median, fresh-boot, see section 13) |

This is the first time the per-layer dense matmuls actually share weight
reads across sessions (instead of N back-to-back `n_tok=1` calls), and
it lifts the N=4 aggregate by ~6 t/s on top of the Phase 1a-1c +5-7 %
shape ceiling.

The +26.5 % win was preserved through an upstream rebase and a
post-refactor simplification pass (encoder pass1/pass2/pass3 helpers,
generalized `ensure_engine_scratch`, `DS4_CUDA_BATCH_MAX` macro +
static_assert).  See sections 11-13 below for per-substitute analysis
and the most recent simplify commit for the encoder cleanup.

## What does NOT help (B-4 experiments tried, reverted)

### 1. Templated weight-sharing matmul kernel

Tried a `matmul_q4_0_preq_nbatch_warp8_kernel<N_TOK>` template that
collapses `grid.y` (= `n_tok`) into an inner unrolled loop over toks
within one warp, so the row's Q4 nibbles are loaded once and dotted
against all N tok activations.  In theory: 1/N weight HBM.  In practice:

| N | naive  | templated |
|---|--------|-----------|
| 2 | 18.26  | 18.28     |
| 4 | 18.73  | 14.58     |
| 8 | 18.79  | 11.47     |

Regression at every N ≥ 4.  Hypothesis: occupancy is the dominant
factor on GB10's Blackwell SMs; collapsing `grid.y` cuts the in-flight
warp count by N and the resulting longer-per-warp work doesn't pipeline
through the warp scheduler as well as more, thinner warps did.  The
implicit L2 reuse in the naive kernel was already paying most of the
weight-amortization that the explicit kernel was supposed to deliver.

### 2. cuBLAS f16 GEMM for the batched output head

Routed `metal_graph_encode_output_head_batched` through
`ds4_gpu_matmul_q8_0_tensor` (which hits cuBLAS f16 at `n_tok > 1`)
instead of `ds4_gpu_matmul_q4_0_batch_warp_tensor`.

| N  | naive Q4 | cuBLAS f16 |
|----|----------|------------|
| 2  | 18.75    | 17.86      |
| 4  | 18.97    | 17.07      |
| 8  | 19.36    | 13.38      |
| 16 | 16.14    | 12.93      |

Regression at every N ≥ 2.  f16 reads 2× the weight HBM that Q4 does;
cuBLAS's compute efficiency does not recover that on a memory-bound
decode-shape GEMM (one or a few rows on the activation side).  Q4 +
implicit L2 wins on this shape.

### 3. Tiled weight-shared kernel (TILE=2 per warp, grid.y preserved)

The fix for the first attempt: keep `grid.y = ceil(n_tok / TILE)` so
occupancy only drops by TILE (not by N), and manually PRMT-unpack the
row's nibbles once per block iteration before the inner unrolled
tok loop (don't rely on the compiler hoisting `q4_block_dot`'s unpack
out of an unrolled call sequence).

Two-run averages on `ds4-bench --ctx-start 2048 --gen-tokens 50`:

| N  | naive | TILE=2 | Δ |
|----|------:|-------:|---:|
| 2  | 18.66 | 18.96  | +1.6 % |
| 4  | 18.95 | 19.13  | +1.0 % |
| 8  | 19.16 | 17.42  | -9 %   (high run-to-run variance) |
| 16 | 9.86  | 9.94   | unstable both ways; a fresh single run gave naive 10.92, TILE 6.34 |

Small win at N=2/4, **clear regression at N=16** (the kernel is
correct, but apparently activation locality / cache pressure flips
against the tiled layout once n_tok is large).  Output head is only
~1.5 % of total decode time anyway, so the upper bound for this kernel
in isolation is ~+1.5 % aggregate — exactly what N=4 measured.

The TILE approach itself is sound and the unpack-once-per-block
pattern is what a later layer-matmul batched kernel should use.  It
just isn't worth shipping for the output head in isolation, because
the win is below noise and the N=16 regression is real.  Reverted.

### Combined finding

The batched output head matmul is not the bottleneck and is already
close to optimal for this shape under `DS4_CUDA_Q4_DECODE`.  Three
distinct kernel-level experiments (templated all-tok, cuBLAS f16,
TILE=2) failed to materially beat the naive `grid.y = n_tok` baseline,
which gets most of its amortization for free from GB10's L2.  Further
optimization of the output head alone has no measurable headroom; the
remaining headroom lives in the layer-internal dense matmuls (Q/K/V,
FFN gate/up/down, attn_output) where the existing per-session encoder
runs N matmul calls at `n_tok = 1` and the naive L2 reuse does not
compound across sessions.

## What DOES help: per-layer interleaved batched-decode encoder

Splitting `metal_graph_encode_decode_layer` into PRE / SHARED / POST
phases and running the outer loop as "layer 0 (all N sessions) → layer
1 (all N sessions) → ..." instead of "session 0 (all 43 layers) →
session 1 (all 43 layers) → ..." turns out to give a real win at low N
purely from implicit L2 reuse: when all N sessions touch layer L's
weights back-to-back, the second through Nth session hit L2 instead of
re-reading from HBM.

`ds4-bench --ctx-start 2048 --gen-tokens 50` aggregate throughput:

| N | B-3 baseline | per-layer interleaved | Δ |
|---|-------------:|----------------------:|--:|
| 1 | 18.12 | 18.05 | noise |
| 2 | 18.75 | 19.25 | +2.7 % |
| 4 | 18.97 | 19.82 | +4.5 % |
| 8 | 19.36 | 14–21 (high variance) | up to +9 % when stable |

At N > 8 the interleaved path becomes unstable on this hardware (high
run-to-run variance, occasional `Metal batched per-layer forward
failed` crashes -- likely working-set / `cuda_tmp_alloc` contention
at high session count).  The shipped code caps the interleaved path at
N ≤ DS4_BATCH_INTERLEAVED_MAX (= 8) and falls back to the older
sync-amortized per-session encoder loop for higher N, which itself is
still better than the original per-call sync.

### L2 persistence policy (tried, reverted)

`cudaDeviceGetAttribute(cudaDevAttrL2CacheSize)` reports GB10's L2 at
**24 MiB**, with a 18 MiB max-persisting cap (75 %).  Hypothesis: the
per-layer interleaved L2-reuse win could be amplified by explicitly
arming `cudaStreamAttributeAccessPolicyWindow` on the layer's hot
weight range (FFN shared expert ~22 MiB).

Result: clear regression.

| N | L2 persist OFF (default) | L2 persist ON | Δ |
|---|-------------------------:|--------------:|---|
| 2 | 19.31 | 13.55 | -30 % |
| 4 | 14.06 | 13.94 | flat |
| 8 | 17.30 | (crash / no output) | bad |

Diagnosis: 18 MiB pinned for FFN weights leaves only ~6 MiB of L2 for
*everything else* in the decode step -- per-session scratch
(g->cur_hc, g->attn_norm, g->qr, g->q, etc.), other dense weight
matrices (attn_q_a/b, attn_output_a/b, ~5-10 MiB), `cuda_tmp_alloc`
prequant buffers, and routing tables.  The cache becomes hostile to
*everything that isn't the FFN shared expert*, so net throughput
falls.

GB10's L2 is genuinely too small for any "pin one matrix, let
everything else fight over scraps" strategy.  The natural L2 reuse
the per-layer interleaved encoder gets *for free* is already close to
optimal for this cache footprint.

### Layer-internal FFN batching (still not enough on its own)

The infrastructure also ships an opt-in (`DS4_CUDA_BATCHED_SHARED_FFN=1`)
batched FFN shared expert: stage N sessions' `g->ffn_norm` into engine
scratch, two Q4 batched matmuls for gate and up, one batched swiglu,
scatter back to per-session `g->shared_mid`.  At N = 2/4 this is
within noise of the per-session fused `shared_gate_up_swiglu` pair
matmul; at N = 8 it slightly regresses.  Why: the FFN `out_dim =
DS4_N_FF_EXP = 2048` is small enough that the per-session fused pair
matmul saturates the GPU on a single call, while the batched path
costs an extra stage + scatter + uses the non-fused 2-matmul-plus-
swiglu shape.

The path off this plateau is a custom Q4 batched-pair kernel
(`matmul_q4_0_pair_preq_batch_warp8_kernel`-style) that loads both
gate and up rows once across N toks.  Without it, batched FFN does
not pay; with it, all the staging/scatter wiring is already in place.

## Where the actual time goes

`DS4_METAL_DECODE_STAGE_PROFILE=1` (one decode step, layer-by-layer; the
profile is over-counted because each stage adds a `cudaEventRecord`
sync, but relative ratios are still meaningful):

| Stage              | per layer (ms) | dominant op                              |
|--------------------|----------------|------------------------------------------|
| q\_path            | ~1.5           | attn\_q\_a Q8/Q4 matmul + LoRA + RoPE    |
| attn\_output       | ~1.3           | attn\_output\_a/b grouped Q8 matmuls     |
| shared\_gate\_up   | ~0.75          | FFN shared expert gate/up Q8 matmul      |
| shared\_down       | ~0.36          | FFN shared expert down Q8 matmul         |
| routed\_moe        | ~0.30          | top-K expert dispatch                    |
| router             | ~0.18          | MoE routing decision                     |
| compressor\_indexer| 0.0–0.7        | ratio-4 layers only                      |
| attention          | 0.04–0.7       | indexed attention kernel                 |
| (all others)       | ~0.05 each     | norms, HC pre/post, etc.                 |

Dense matmuls (q\_path + attn\_output + shared\_\*) are ~76 % of layer
time.  43 layers × ~4 dense matmuls per layer = ~172 matmul call sites
all running serially-per-session today.

## Where the wall clock actually goes (post-`DS4_METAL_DECODE_STAGE_PROFILE`)

The stage profile inflates totals because each boundary inserts a
`cudaEventRecord`/sync.  `DS4_METAL_GRAPH_TOKEN_PROFILE=1` on a steady
single-session decode gives the un-instrumented split:

```
ds4: metal graph token pos=15 encode=21.9 ms execute=24.6 ms read=0.0 ms total=46.6 ms
ds4: metal graph token pos=16 encode=19.3 ms execute=25.7 ms read=0.0 ms total=45.1 ms
ds4: metal graph token pos=17 encode=19.3 ms execute=25.8 ms read=0.0 ms total=45.1 ms
```

- `encode` = host-side time spent inside
  `metal_graph_encode_token_raw_swa`.  Default-flushed path takes a
  cudaDeviceSynchronize after layer 4 (`allow_split_flush=true`), so the
  ~19 ms includes the GPU stall waiting for layers 0–3.  In the batched
  API path we already force `allow_split_flush=false`, so this stall
  isn't in our hot path.  Pure host-side encode (kernel-launch dispatch
  for all 43 layers + the output head) is much smaller than 19 ms.
- `execute` = final `cudaDeviceSynchronize` wait, i.e. essentially all
  GPU compute time per token.  ~25 ms.
- `total` ≈ 45 ms per token = ~22 t/s, consistent with the
  ds4-bench-measured 18 t/s once you account for the bench loop's
  per-iteration argmax + slot fill on the CPU.

`nsys profile` over a 200-decode-token run confirms the same picture
at the GPU-utilization level:

  | Profile | N=1 | N=4 |
  |---|---:|---:|
  | GPU busy / wall | 91.8 % | 93.6 % |
  | Total kernels | 408,645 | 1,630,548 (4× N=1) |

The GPU is already 90%+ utilized at N = 1, so CUDA-streams parallelism
across sessions can claim at most the remaining ~8 % of gap time.  The
batched-decode work has to share weight reads across sessions to gain
anything beyond that; pure overlap is not going to deliver the missing
3× to 5× implied by Phase 0's chunked-prefill ceiling.

Implication: GPU compute is ~25 ms/token at N = 1.  At N = 4 we observe
~210 ms / 4 tokens = 52 ms/token effective.  That's 4× the GPU compute
of N = 1 — i.e. the batched path is essentially fully serial on the GPU
side.  Whatever implicit L2 reuse we get inside the naive Q4 batched
output head kernel does NOT compound across the layer matmuls because
those still each run from `cuda_matmul_q8_0_tensor_labeled` at
`n_tok = 1` per session.

## The real lever for N batching

To get past +7 % at N = 4–8, the per-layer dense matmuls have to actually
share weight reads across sessions.  The current per-session encoder
calls `ds4_gpu_matmul_q8_0_tensor(..., n_tok = 1)` once per session per
matmul, which routes to the single-row warp kernels.  Even when N
sessions run back-to-back on the default stream, each session's matmul
re-reads the same weights from HBM with negligible L2 reuse across
sessions (the working set per matmul is several hundred MB, while L2 is
< 100 MB on GB10).

Two refactor shapes that could deliver the gain:

1. **Stage-batch-scatter per layer**: between each layer step, stage all
   N sessions' activations into a single batched scratch, call the
   existing `n_tok > 1` matmul path (cuBLAS or batched warp), scatter
   results.  Requires interleaving the per-session encoder calls so all
   N reach the same step before staging.  ~172 stage / batch / scatter
   cycles per token.

2. **Rewrite the layer encoder to be N-aware natively**: each per-layer
   scratch (`g->qr`, `g->q`, `g->attn_out`, `g->shared_*`, etc.) becomes
   N-row wide; kernels take an extra `n_active` dimension; KV-store /
   attention / RoPE need per-row `pos` and per-row KV pointers (see
   `__constant__ ds4_batch_step_args g_batch_args` already declared in
   `ds4_gpu.h`).  Mirrors how prefill already works for "1 session, N
   consecutive tokens".

Either approach is multi-day.  (1) is more incremental but interleaves
control flow.  (2) is more disruptive but matches DS4's existing prefill
shape and the `g_batch_args` infrastructure we already landed.

### Concrete next-session checklist (shape 1, simplest non-trivial slice)

Target the FFN shared expert at every layer because it's the biggest
single weight read per layer and has no per-row dispatch (KV / attention
stay per-session).

1. Add engine-level scratch in `struct ds4_engine`:
   - `batched_ffn_norm` — `DS4_BATCH_MAX × DS4_N_EMBD × f32`
   - `batched_shared_gate`, `_up`, `_mid` — `DS4_BATCH_MAX × shared_dim × f32`
   Lazy-allocate in `ensure_engine_batched_scratch` (already there for the
   batched output head's scratch).

2. Split `metal_graph_encode_decode_layer` in `ds4.c` at the
   `shared_gate_up` boundary (around line 9703).  Pre half: everything
   through `g->ffn_norm`.  Post half: `keep_ffn_out` / fused shared down
   HC expand onwards.

3. Add `metal_graph_encode_shared_ffn_batched(engine, sessions[], n,
   layer)` that stages N `g->ffn_norm`, calls
   `ds4_gpu_matmul_q8_0_pair_tensor(batched_gate, batched_up, ...,
   n_tok = n_active)` — note: at n_tok > 1 this currently routes through
   cuBLAS f16, which regressed for the output head.  May regress here
   too; if so we need a custom Q4 batched pair kernel
   (`matmul_q4_0_pair_preq_batch_warp8_kernel`) — analogous to the
   existing pair single-row kernel but with the row-major
   weight-shared shape from our failed templated experiment, except
   keeping grid.y = ceil(n_tok / SMALL) for occupancy.  Worth testing
   N = 2 first since occupancy hit there is mild.

4. Loop in `ds4_session_eval_batched_decode`: for each layer, per
   session run "pre", then batched shared FFN, then per session run
   "post".  This means the layer loop runs OUTSIDE the per-session
   encoder.

5. Measurement target: if step 4 shaves anything off N=4 throughput
   (current 18.97 t/s → 21+ t/s would be a clean win), iterate to do
   attn_q_a, attn_q_b, attn_output, FFN shared down in the same shape.
   If it regresses, the shared scratch + cuBLAS f16 path isn't the
   answer and shape 2 with a custom batched kernel is the only way.

### 5. CUDA Graph capture on the per-layer interleaved batched encoder (shelved)

Hypothesis: `perf/cuda-graph-wip` (5ae22be) showed CUDA Graph capture is
healthy on GB10 but regresses end-to-end because `cudaGraphExecUpdate`
over the ~1000-node single-session graph costs ~70 ms / token, one
parameter rewrite per kernel node.  cuda-graph-wip's "what's left"
checklist is to lift every per-token scalar (pos / raw_row / n_raw /
n_comp / top_k / token / …) out of kernel signatures into
`__constant__` memory; once the captured graph is structurally
identical across tokens, `cudaGraphExecUpdate` becomes a no-op and the
~0.2 ms pure-launch path is reachable.

The batched-decode path's per-layer interleaved structure (stable
sequence of N session × 43 layers × phased kernels) is even more
graph-friendly: capture once at `n_active = N`, replay forever as long
as the active session set is stable.

What killed it before opening Phase B:

  - GPU is already 91.8 % busy at N=1, 93.6 % at N=4 (this file,
    "Where the wall clock actually goes").  The "~19 ms host encode"
    figure in the single-session profile is GPU-sync-inflated
    (`allow_split_flush = true` after layer 4) — pure host
    kernel-launch dispatch is "much smaller than 19 ms" (same
    section).  Batched API forces `allow_split_flush = false` so the
    inflated stall isn't in the hot path either.
  - Capture's upper bound is therefore the ~8 % gap-time slice, not
    the 30-40 % the naive single-session encode number suggested.
  - The kernel scalar-lift refactor is ~25-30 hot kernels, 2-3 days.
    8 % at that cost is bad ROI.

Phase A — routing all 143 kernel launches through `g_kernel_stream`
and binding cuBLAS to the same stream — landed as 1d7402e because
it's behavior-preserving infrastructure that any future capture
experiment will need.  Phase B (kernel scalar lift) and Phase C
(capture/replay wrapper) are not started.

Re-open when: GPU utilization at the relevant N drops below ~80 %
(e.g. after Option G / MTP integration accepts ≥50 % of drafts and
the wall clock per kept-token rises into the host-overhead-visible
regime), OR a faster `cudaGraphExecUpdate` lands in a CUDA toolkit
release and the lift refactor can be skipped.

### 6. MTP speculative decode (Option G) — empirically neutral on GB10

Hypothesis: DS4 already has an end-to-end MTP draft-and-verify path
(`ds4_session_eval_speculative_argmax`, `metal_graph_verify_decode2_exact`,
the MTP support model `gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf`).
Premise was 1.3-1.5x effective throughput at typical accept rates,
which could be carried over to the batched path.

Ground-truth measurement on single-session `ds4 --temp 0 --mtp ...
--mtp-draft 2` (the only mode that engages the speculative path in
`ds4_cli.c`; default `temperature = 1.0` skips it entirely), mean of
three runs, `-n 80`:

| Config | gen t/s |
|---|---|
| MTP off, temp=0 | 21.07 |
| MTP on, depth=2, temp=0 (default verifier path) | 20.19 (-4.2 %) |
| MTP on, depth=2, temp=0, DS4_MTP_STRICT (decode2_exact) | 20.09 (-4.7 %) |

The cost model explains the ceiling: target decode ≈ 25 ms, MTP draft
(1 MTP layer + output head) ≈ 5 ms, `decode2_exact` verifier (43-layer
target with `n_tok = 2`) ≈ 30 ms — at *100 % accept* the math gives
3 tokens / 60 ms ≈ 50 t/s vs baseline 40 t/s = 1.25x, and the
observed neutral-to-slightly-negative number means accept rate is
not in the regime where the ceiling pays.  The verifier is essentially
"another target decode" because dense matmuls are weight-HBM-bound
regardless of `n_tok ∈ {1, 2}`.

Implication for the batched plan: integrating MTP into
`ds4_session_eval_batched_decode` is 2-3 days of work (per-row draft
state, batched `n_tok = 2` verifier across N sessions, per-row
accept/reject branching, KV-cache divergence handling).  A 2-3 day
build for a path that is empirically zero-gain at the per-session
level is bad ROI.  Re-open if a faster MTP-draft kernel lands or if
the target verifier can be shrunk to fewer layers (e.g. a learned
short-cut head).

### 7. Q3_0 dense matmul (custom 32-value bit-plane) — dead-end

Hypothesis (Option A' minimal slice): mirror the existing Q4_0 lazy-
convert + dp4a matmul path but drop one bit per weight.  Format used:
32-value block, split-row layout, 2 bytes fp16 scale + 12 bytes bit-
plane packing (3 int32 words = LSB plane / mid plane / sign plane).
14 bytes / block vs Q4's 18 = 22 % HBM reduction.  Decode in the warp8
matmul kernel walks 8 dp4a passes per block, each unpacking 4
consecutive 3-bit values from the bit-planes into a dp4a-ready int32
of 4 sign-extended int8 bytes.

PoC enables Q3 in `ds4_gpu_matmul_q8_0_tensor` (n_tok = 1 path) ahead
of the existing Q4 dispatch when `DS4_CUDA_Q3_DECODE` is set.

Result: failed on both axes.

- **Quality**: greedy "The capital of France is" produces garbage
  tokens (`asarangang Ginhadi насељени насељени …`).  A single fp16
  scale per 32-value block can't carry the dynamic range Q8 source
  weights need at 3 bits — the quantization grid is too coarse.
- **Throughput**: N=1 bench `ctx=2048 gen=30` lands at 13.58 t/s vs
  Q4 baseline 17.42 t/s = -22 %.  The bit-plane decode (8 scalar
  bit extractions × 8 dp4a passes per block) is compute-bound at
  this footprint, so the 22 % HBM savings don't translate.

So the simple "Q4 → Q3 by analogy" loses both the precision AND the
speed it was supposed to buy.  A real Q3 path on this hardware would
need:

  - Sub-block scales (Q3_K-style: 256-value super-block, 16-value
    sub-blocks with 6-bit shared scales) to recover quality; this is
    substantially more code than the PoC.
  - A faster decode than the bit-plane scalar fan-out — likely a
    table-lookup or a __byte_perm-based shuffle that exploits the
    sub-block structure.

That's a multi-day rewrite, not a "drop one bit and reuse Q4's path"
extension.  Code landed in this commit as a documented dead-end so
the Q3 PoC harness (cache, convert kernel, matmul kernel, dispatch
gate, env var) is available if a future Q3_K-style attempt wants to
start from somewhere closer than scratch.

Re-open when: a Q3_K-style scheme is designed for GB10 specifically
(fast decode that doesn't fight the 24 MiB L2), OR if a routed-expert
2-bit scheme (IQ2_XXS-style, already used by DS4 routed weights)
demonstrates a working decode kernel + fast dequant pipeline that the
dense matmuls can borrow.

### 8. Q3_K_S (256-value super-block, 16-value sub-block) — dead-end

Hypothesis: section 7's quality collapse was the single-scale-per-32-
values granularity.  Section 7's speed loss was the bit-plane decode's
scalar fan-out.  Q3_K_S addresses the first by adding sub-block scales
(simplified-Q3_K: 256-value super-block × 16-value sub-block × uint8
sub_scale, single fp16 super-scale; 116 bytes per super-block when
padded to 4-aligned).  The matmul kernel reads the low 2 bits from a
single qs byte (4 values per byte) and the high bit from 4 hmask bytes
(one per value at the appropriate layer position) per dp4a pass; the
4 raw values are byte-wise sign-fixed via `__vsub4(raw, 0x04040404)`
straight into the dp4a-ready int32 form.  Each warp lane handles half
a super-block (128 values, 8 sub-blocks, 32 dp4a passes).

Result: failed on both axes again.

- **Quality**: greedy "The capital of France is" produces
  `freelance<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>…` — better
  than Q3_0's pure noise, but still wrong from token 1 (special-token
  emission means the output head's argmax landed nowhere near the
  intended token).  Likely root cause: the simple
  `sub_scales[s] = round(sub_max[s] / super_max × 255)` derivation
  zeros sub-blocks whose magnitude is below ~1/510 of the super-block
  max.  In DS4 weights with outliers, this happens often enough to
  corrupt the output head.  ggml's real Q3_K uses iterative refinement
  of sub-scales (minimize per-block MSE rather than just cover the
  range) — that closes the quality gap but adds non-trivial convert-
  time work.
- **Throughput**: median of three runs at `ctx=2048 gen=50`:
  - N=1: Q3_K_S 14.43 vs Q4 17.92 = **-19.5 %**.
  - N=4: Q3_K_S 14.98 vs Q4 19.7 = **-24 %**.
  Decode is compute-bound just like Q3_0 was, despite different
  packing.  The structural problem is that 3-bit decode on GB10 needs
  per-value bit reassembly with no `__byte_perm` shortcut analogous to
  Q4's nibble pattern, so the kernel chews ~10 ops per dp4a vs Q4's
  ~5, and the 19 % HBM savings don't cross the gap.

So both Q3 attempts in this session reach the same dead-end via
different routes: HBM savings can't translate when 3-bit decode is
the bottleneck on this GPU's warp-throughput.  A working Q3 path
would need at least one of:

  - Tensor-core MMA-INT4 (Option H) that performs the dot product on
    a packed integer format directly, eliminating the per-dp4a decode
    overhead.
  - A kernel structure that decodes the whole row's weights to int8
    once into shared memory (one big decode, then dense int8 matmul);
    this trades L2 pressure for compute-vs-memory balance and may
    backfire given GB10's 24 MiB L2.
  - A different decode algorithm exploiting `bfind`/`brev` or warp-
    shuffle bit fan-out tricks that Q3_0's bit-plane and Q3_K_S's
    qs/hmask layouts both failed to exploit.

Q3_K_S code lands as section-8 documented dead-end alongside section
7's Q3_0; both share a `cuda_q3_range` / `cuda_q3k_range` cache and
parallel `DS4_CUDA_Q3_DECODE` / `DS4_CUDA_Q3K_DECODE` env gates so
either harness can be picked up by a future attempt without rewriting
the lazy-convert plumbing.

Re-open when: Option H (Blackwell INT4 MMA) is built first and shows
a real dense-matmul speedup at decode shape — that becomes the
substrate any new low-bit packing has to plug into.

### 9. INT8 tensor-core GEMV (Option H, decode-shape) — dead-end

Paper analysis flagged this as risky going in: the matmul shape at
n_tok = 1 is `[out_dim] = W[out_dim, in_dim] × x[in_dim]`, a pure
GEMV.  WMMA `m16n16k16 INT8` requires the result to be `16 × N`, with
N = 16; for a single activation column only 1 of the 16 N slots is
useful so 15/16 of every tensor-core call is wasted.  Even at the
best-case ratio TC INT8 ≈ 4× dp4a on Blackwell, the effective rate
becomes `1/16 × 4 = 0.25× dp4a`, and the kernel additionally loses
Q4_0's HBM advantage because WMMA INT8 reads full Q8_0 weights
(2× the HBM of the Q4 path).

Empirical PoC (`matmul_q8_0_wmma_int8_kernel` gated on
`DS4_CUDA_TC_DECODE=1`): one warp per 16 output rows, weights staged
into 16-byte-aligned shared memory (the on-disk Q8_0 layout has 2-byte
fp16 scale + 32 int8 values per 34-byte block, so the int8 data is
only 2-byte aligned and can't feed WMMA directly), activation
replicated across N=16, two WMMA passes per Q8 block, per-row scale
applied on column-0 of the staged INT32 c_frag.

Bench (ctx=2048 gen=50 N=1, three runs):

  - TC INT8: 7.72 / 9.19 / 11.11 t/s  (median 9.19)
  - Q4 dp4a baseline: 17.92 t/s
  - Regression: **-49 %**

Two stacked penalties show up: (a) the predicted 1/16 utilization
penalty on the TC compute path itself; (b) the byte-by-byte
shared-memory staging of unaligned Q8_0 weight bytes (8 sequential
uint8 loads per lane) doesn't coalesce as well as the Q4 path's
`__byte_perm` decode does.  Quality is preserved (greedy "Paris" emits
correctly) because TC INT8 keeps the full Q8 precision — only the
compute path is different.

A non-decode-shape TC path is gated by upstream work:

  - Increase the n_tok dimension actually fed to the matmul.  At
    n_tok ≥ 8 the TC utilization approaches 50 % and dp4a no longer
    wins on compute.  In the batched API path the natural lever is
    cross-session batching, but the current encoder calls each layer
    matmul with `n_tok = 1` per session and the GPU's L2 (24 MiB) is
    too small for the L2-reuse trick to compound past N = 8.  The
    "real lever" section above already calls out the layer-encoder
    refactor needed; both that refactor AND a TC kernel are required
    before any 3-bit / INT4 packing can land usefully.
  - Or use INT4 wmma `m8n8k32`: 1/8 instead of 1/16 utilization plus
    nominally 2× INT8 TC throughput buys back to roughly break-even
    with dp4a, still no win.  Mixed-precision INT4×INT8 wmma isn't
    available via the C++ API on sm_121 (CUDA 13.0) — it requires
    inline PTX `mma.sync` and a custom INT4 weight format.  Wouldn't
    pay before the cross-session batching is fixed.

Re-open when: (a) the layer encoder is refactored so per-layer
matmuls run at n_tok ≥ 4 per call (cross-session aggregation), AND
(b) the batched path is stable past N = 8.  Until both hold, no
tensor-core variant at this matmul shape can compete with the Q4
dp4a baseline.

### 10. N > 8 batched stability — diagnosis update (no fix)

Earlier comment on `DS4_BATCH_INTERLEAVED_MAX = 8` attributed the
instability past N = 8 to "cuda_tmp_alloc contention at high session
count" (see ds4.c around the cap definition prior to this commit).
That diagnosis was untested.

This session traced cuda_tmp_alloc with `DS4_CUDA_TMP_TRACE=1` and
ran the per-layer interleaved encoder at various N.  Findings:

- The scratch allocator does NOT thrash during decode.  During a
  three-token N = 16 run, the trace records exactly **two** realloc
  events, both at prefill time (`f16 gemm activations` 64 MiB,
  `attention output a cublas` 192 MiB); decode itself produces zero.
- N = 16 does NOT deterministically crash.  Five back-to-back runs
  completed without error, with high variance (5.86 – 11.21 t/s).
- A fresh-GPU sweep, 5 runs per N with 8 s cooldown between runs
  (ctx=2048, gen=50, GPU starting from idle 49-53 °C):

  | N  | min    | median | max   | range  |
  |----|--------|--------|-------|--------|
  |  8 | 13.90  | 16.73  | 20.23 | 6.33   |
  |  9 | 15.98  | 19.43  | 20.12 | 4.14   |
  | 10 |  8.99  | 14.59  | 18.66 | 9.67   |
  | 11 | 11.66  | 14.76  | 17.49 | 5.83   |
  | 12 | 14.76  | 18.60  | 19.48 | 4.72   |
  | 16 |  5.86  | (n=5 too noisy to median) | 15.12 | 9.26 |
  | 24 |  4.28  |  ─     |  ─    |        |
  | 32 | hang on every run                  |        |

  The per-N range is 4-9 t/s wide -- bigger than the inter-N gaps.
  The single-shot sweep earlier in this session that suggested "N=8
  is the clean peak" did not survive 5x repetition: N=9 and N=12
  medians both edge past N=8, but only inside the noise band, and the
  ordering across N is not monotonic.  Likely culprits for variance
  include thermal modulation (prefill_tps fluctuates 185-345 between
  back-to-back runs) and KV-cache fill state on the first decode
  after model load.

  Aggregate clearly DOES collapse past N = 16 (worst-case run never
  exceeds 15 t/s; best run is below N = 8's median).  The N = 8 → 12
  region looks noise-flat at this measurement budget; raising the
  cap would trade the stability of the documented N = 8 path for a
  noise-dominated upside.  Cap stays at 8.

- Per-stage profile at N = 16 (`DS4_METAL_DECODE_STAGE_PROFILE = 1`,
  decode steps only): dense matmuls (`attn_output 39 %`, `q_path
  29 %`, `shared_gate_up 11 %`, `shared_down 6 %`) account for 86 %
  of decode wall time; attention is just 1 %.  Within a layer,
  session 0 of each stage takes ~150 ms while sessions 1-15 take
  ~0.5 ms each — the "session 0 cold-fetches, sessions 1-N hit L2"
  amortization is still working at N = 16 (for THE LAYERS THAT FIT),
  but the per-session scratch + per-session active expert weights
  push the working set past GB10's 24 MiB L2 around N = 12-14, after
  which the cold-miss pattern repeats more often and per-session
  throughput collapses.

So the cap stays at 8 — the right cap, the wrong reason in the old
comment.  The path to lift it isn't an allocator fix; it's the
"real lever" layer-encoder refactor that shares per-layer scratch
across sessions (so per-session working set stops scaling with N).
The hang at N = 32 is uninvestigated; reproducing it long enough to
catch in a debugger would be its own session.

Re-open when: someone designs a per-layer scratch layout that shares
across N sessions (analogous to how attention's KV cache is shared
within a session but private across sessions).  The substrate also
unblocks the Option H upstream described in section 9.

### 11. Batched attn_q_b (DS4_CUDA_BATCHED_Q_B) — first measurable layer-encoder N-aware win

The per-layer interleaved encoder was calling `attn_q_b` (the layer's
single biggest dense matmul, q_rank=1024 → q_dim=32768) once per
session at n_tok = 1, so each session re-read the same weight matrix
from HBM with no L2 reuse compounding across sessions on this 24 MiB
cache.  This commit splits the phased decode-layer encoder's PRE
phase into PRE_A / PRE_B / PRE_C sub-phases (a behaviour-preserving
refactor) and adds an opt-in batched substitute for PRE_B:

- New env var `DS4_CUDA_BATCHED_Q_B` enables `metal_graph_encode_q_b_batched`,
  which stages N sessions' `qr_norm` into engine-level
  `batched_qr_norm`, runs one `ds4_gpu_matmul_q4_0_batch_warp_tensor`
  call with `n_tok = N` to write `batched_q`, and scatters rows back
  to per-session `g->q`.
- The batched encoder runs PRE_A per session, then `q_b_batched`,
  then PRE_C per session (head_rms_norm + rope + kv_store + attention
  + attn_output + HC expand + FFN HC pre + router + MoE).
- If the batched call fails (Q4 lazy convert unavailable, etc.), the
  PRE_C-per-session loop falls back to PRE_B|PRE_C so g->q gets a
  valid per-session fill before head_rms_norm reads it.

Measured (`ds4-bench --ctx-start 2048 --gen-tokens 50 --batch N`,
median of 5 runs with 5 s cooldown):

| N | baseline median | BATCHED_Q_B median | Δ |
|---|----------------:|-------------------:|--:|
| 4 | 19.78  (range 19.10 – 19.85) | **20.49** (range 20.46 – 20.63) | **+3.6 %** |
| 8 | 20.03  (range 12.12 – 20.12; 2 thermal outliers) | **21.10** (range 21.08 – 21.11) | **+5.3 %** |

The throughput delta is meaningful but the more striking change is
the *variance collapse* at N = 8: baseline runs swing 12-20 t/s with
intermittent low outliers (consistent with HBM contention as N
sessions all race for the same `attn_q_b` weights through L2);
batched runs hold a 0.03 t/s spread across 5 measurements.

This is the first successful "real lever for N batching" PoC -- the
shape that NOTES.md section "The real lever for N batching" Option 2
named earlier in this file.  attn_q_b is the easiest of the dense
matmuls to batch (single matmul with a clean qr_norm → q boundary;
no per-row pos baked into the inputs).  The same pattern can extend
to attn_q_a + attn_kv (which feed qkv_rms_norm, so they batch jointly
with the norm), the attn_output_a/b pair, and a fused gate/up shared
FFN.  Each one is now a small follow-up commit instead of a
multi-day project.

Re-open when: extending to the next dense matmul.  The infrastructure
piece (PRE_A/B/C split + engine batched scratch + `metal_graph_encode_*_batched`
pattern) is reusable.

### 12. Batched attn_q_a + attn_kv + qkv_rms_norm (DS4_CUDA_BATCHED_QKV)

The Q_B win (section 11) replaced the single largest dense matmul per
layer with one engine-level call.  This commit applies the same shape
to the smaller-but-still-substantial pair: attn_q_a (DS4_N_EMBD ->
q_rank) + attn_kv (DS4_N_EMBD -> DS4_N_HEAD_DIM), which both consume
g->attn_norm and feed dsv4_qkv_rms_norm_rows.  PRE_A is sub-split
into PRE_A1 (HC pre + attn_norm) and PRE_A2 (q_a + kv + qkv_norm).
DS4_CUDA_BATCHED_QKV substitutes a single
metal_graph_encode_qkv_batched for PRE_A2: stage N sessions' attn_norm
into batched_attn_norm, two Q4 batched matmuls (q_a -> batched_qr, kv
-> batched_kv_raw) at n_tok = N, one batched dsv4_qkv_rms_norm_rows
that writes batched_qr_norm and batched_kv across N rows, scatter
batched_kv to per-session g->kv (PRE_C still consumes it per-session
for rope + kv_store), scatter batched_qr_norm to per-session
g->qr_norm so PRE_B can read it.

When BATCHED_QKV and BATCHED_Q_B are both on, the q_b path skips its
qr_norm staging (the engine buffer is already filled by the qkv
path), and the second per-session pass is just PRE_C; the only
per-session work in PRE then becomes HC pre + attn_norm + head_norm
+ rope + kv_store + attention + attn_output + HC expand + FFN HC pre
+ router + MoE.

Measured at N = 4 (median of 3 runs, GPU cooled to <55 °C between
runs, ctx = 2048, gen = 50):

| Config       | N=4 median | Δ vs baseline |
|--------------|-----------:|--------------:|
| baseline     | 19.64      | —             |
| Q_B only     | 20.40      | +3.9 %        |
| QKV only     | 20.17      | +2.7 %        |
| QKV + Q_B    | 20.65      | +5.1 %        |

QKV alone gives a smaller win than Q_B alone because the two matmuls
it batches (q_a and kv) are individually smaller than q_b; together
they don't quite match q_b's HBM amortization potential.  Stacking
both adds an extra +1.2 % over Q_B alone, which is consistent with
"the leftover savings on q_a + kv after Q_B already shaved q_b."

N = 8 measurements proved unreliable during this session because the
GPU was running hot from the multi-config interleaved sweeps -- the
prefill_tps column drifted between 200 and 350 t/s across nominally-
equivalent runs, and the per-condition variance swallowed the
inter-condition delta.  N = 4 measurements with strict 55 °C
cooldowns were stable.  A clean N = 8 re-measurement on a fresh-boot
GPU is left as a follow-up.

Re-open when: extending to the next dense matmul -- attn_output_a +
attn_output_b is the natural next target (39 % of decode time
versus q_b's 29 % and qkv's ~10 %); shared FFN gate/up has the same
shape but its dimensions are too small for the batched warp kernel
to saturate (see "Layer-internal FFN batching" earlier in this file).

### 13. Batched attn_output (DS4_CUDA_BATCHED_ATTN_OUTPUT) — biggest single win

attn_output is the layer's largest stage at 39 % of decode time
(NOTES "Where the actual time goes").  It comprises grouped output_a
(reduces N_HEAD attention heads into N_OUT_GROUP groups × LoRA rank)
plus output_b (LoRA rank * n_groups -> N_EMBD), and in production it
runs as one fused single-token kernel
(`ds4_gpu_attention_output_low_q8_tensor` + Q8 hc_expand fusion).
That fusion forecloses any batched-N path.

This commit:
- Splits PRE_C into PRE_C1 (head_norm + rope + kv_store + compressor +
  attention -> writes g->heads), PRE_C2 (attn_output -> writes
  g->attn_out and, when fused, g->after_attn_hc), and PRE_C3
  (steering + hc_expand + FFN HC pre + router + MoE).  Behaviour-
  preserving (PRE_C = PRE_C1|PRE_C2|PRE_C3, default callers see no
  change).
- Adds Q4 dispatch inside `ds4_gpu_attention_output_q8_batch_tensor`
  at n_tokens in (1, 32]: output_a uses the existing-but-previously-
  gated-only-at-n_tok=1 `grouped_q4_0_a_preq_warp8_kernel`; output_b
  uses `ds4_gpu_matmul_q4_0_batch_warp_tensor` directly instead of
  routing through the cuBLAS f16 fallback.  Prefill (n_tokens ~ 2048)
  still goes through cuBLAS f16 since the TC-friendly path beats the
  Q4 warp kernel at that shape.
- Adds engine-level batched_heads / batched_attn_low / batched_attn_out
  scratch, and `metal_graph_encode_attn_output_batched` (stage N
  sessions' heads, one n_tokens = N call to the batched output
  tensor, scatter rows back to per-session g->attn_out).
- PRE_C3 now runs hc_expand whenever PRE_C2 was substituted (not just
  when `fuse_attn_out_hc` was off), so the batched path's
  non-fused-equivalent semantics flow through correctly.

DS4_CUDA_BATCHED_ATTN_OUTPUT=1 selects the batched substitute for
PRE_C2.  Stacking with DS4_CUDA_BATCHED_QKV and DS4_CUDA_BATCHED_Q_B
splits the per-session work into three loop passes around the three
batched calls (qkv between PRE_A1 and PRE_B; q_b between qkv and
PRE_C1; attn_output between PRE_C1 and PRE_C3).  Any batched call
that fails at runtime falls back to per-session phased() with the
corresponding sub-phase, so the graph result stays equivalent.

Measured at N = 4 (median of 3 runs, GPU cooled to <55 °C between
runs, ctx = 2048, gen = 50):

| Config                    | N=4 median | Δ vs baseline |
|---------------------------|-----------:|--------------:|
| baseline                  | 19.67      | —             |
| ATTN_OUTPUT only          | 23.30      | +18.5 %       |
| QKV + Q_B + ATTN_OUTPUT   | **24.97**  | **+26.9 %**   |

This is the largest single batched-substitute win so far; ATTN_OUTPUT
alone moves N=4 throughput from ~19.7 to ~23.3 t/s.  Combining all
three batched substitutes brings N=4 to 25 t/s, a 27 % aggregate
gain vs the per-session baseline.

Fresh-boot N=8 re-measurement (5 runs each, cooldown to <55 °C between
runs, ctx=2048, gen=50):

| Config   | runs                                  | median  | mean  | range          | CoV   |
|----------|---------------------------------------|--------:|------:|----------------|------:|
| baseline | 17.31 / 15.62 / 11.16 / 14.19 / 18.82 | 15.62   | 15.42 | 11.16 – 18.82  | ~17 % |
| ALL ON   | 24.60 / 22.42 / 19.03 / 25.00 / 24.77 | 24.60   | 23.16 | 19.03 – 25.00  | ~10 % |

Δ median +57.5 %, Δ mean +50.2 % — substantially above the earlier
thermally-noisy +12 % single-shot.  Baseline variance stays high
because per-session serial kernel-launch latency on N=8 sits in the
unfortunate zone where small scheduling jitter dominates throughput;
ALL ON tightens up because batched substitutes amortize most of that
serial overhead.  ALL ON median 24.60 is within noise of the N=4
median (24.97), so the batched substitutes effectively flatten the
N=4 → N=8 throughput cliff.

Re-open when: the remaining dense matmul candidates are the
shared-FFN gate+up pair (small dims, NOTES section "Layer-internal
FFN batching" suggests this needs a custom Q4 pair kernel to actually
pay) and the routed-MoE experts (per-session expert selection makes
batching non-trivial).  Output head is already engine-batched in
production via metal_graph_encode_output_head_batched.  So the
section-12 + section-13 pair has effectively captured the bulk of the
layer-encoder N-aware refactor named earlier in this file under "The
real lever for N batching".

### 14. Shared-FFN v2 — day-1 recon (existing flag is broken; measurement floor is the real blocker)

Scope: re-evaluate the **existing** `DS4_CUDA_BATCHED_SHARED_FFN` flag
stacked on top of the sections 11-13 wins, then decide whether a v2
custom Q4 pair kernel is worth designing.

**Handoff doc fact correction.** Shared FFN does *not* live inside
PRE_C3.  It lives in an independent `SHARED` phase (bitmask `64u`,
ds4.c:9079-9093), invoked between PRE_C3 and POST per session via
`metal_graph_encode_decode_layer_phased`.  The existing batched
function `metal_graph_encode_shared_ffn_batched` (ds4.c:15824-15864)
already uses `ds4_gpu_matmul_q4_0_batch_warp_tensor` — i.e. it does
go through the Q4 batch path, contrary to "Q4 dispatch 経路通らない"
in the handoff prompt.

**Existing flag is broken.** Stacking
`DS4_CUDA_BATCHED_SHARED_FFN=1` on ALL ON (Q4_DECODE + QKV + Q_B +
ATTN_OUTPUT) at N=8 produced:
- 2 fatal failures out of 15 total runs (5 + 10 reproducer pass).
  First failure printed the clean `Metal batched per-layer forward
  failed` string; the second printed *garbage bytes* (`(4J\xff…`)
  where that string should have been — a memory-corruption smoking
  gun, not just an OOM / driver fault.
- Throughput in the successful runs sat inside the variance band of
  ALL ON without the flag (no clean +N% signal at all).

ALL ON itself had **0 fatal failures** across 15 runs in the same
session, so the corruption is SHARED_FFN-specific, not GPU- or
thermal-driven.

**Stream-consistency hypothesis (leading, unverified).**  The
existing `metal_graph_encode_shared_ffn_batched` flow is:
1. default stream (0): `ds4_gpu_tensor_copy_async` gather of N
   sessions' `ffn_norm` into `batched_ffn_norm` (ds4_cuda.cu:1948
   issues `cudaMemcpyAsync(..., 0)` — default stream).
2. `g_kernel_stream`: matmul gate → matmul up → swiglu (three
   back-to-back kernels).
3. default stream (0): scatter copy back into per-session
   `shared_mid`.

The same default-stream copy_async pattern is used by
`metal_graph_encode_attn_output_batched` (section 13) without
visible failures, but attn_output is **one** big kernel between
gather and scatter; shared_ffn has **three** consecutive kernels,
widening the race window between the gather write on stream 0 and
the matmul read on `g_kernel_stream`.  CUDA legacy-default-stream
synchronization is only guaranteed when nothing else has overridden
the program's default-stream behavior; with cuBLAS in the picture
this can break silently.  The follow-up named in NOTES top-of-file
"4. ds4_gpu_tensor_copy_async stream consistency" is exactly this.

**Measurement environment regression.**  The same ALL ON config
measured 24.60 t/s median at the start of this session, 21.30 t/s
an hour later, and 11.94 t/s an hour after that — same N=8, same
cooldown<55 °C protocol, same binary.  Cumulative thermal envelope
from the prior bench rounds plus an idle `ollama serve` that briefly
took GPU during one round are the two known contaminants.  Net: the
+1 % detection threshold required to ship Shared-FFN v2 is **not
achievable in this measurement environment**.  Cold-boot + zero
ambient GPU process is a prerequisite, not a nicety.

**Next session prerequisites** before reopening Shared-FFN v2:
1. Cold-boot bench protocol (full host reboot, kill ollama / any
   GPU daemon, single bench-only session).
2. Decide on `ds4_gpu_tensor_copy_async` stream change — either move
   to `g_kernel_stream` (and re-verify sections 11-13 wins survive)
   or insert explicit `cudaStreamWaitEvent` synchronization at the
   boundary.  Without this, even a correct v2 kernel will inherit
   the same race window.
3. Only then design the custom Q4 pair kernel (gate + up fused into
   one launch, sharing the dequantized weight tiles for the same
   input row).  Until 1 + 2 are done, a perfect v2 kernel cannot
   be measured.

### 15. gen_tps vs wall-clock — bench numbers do not reach the user yet, plus Day-1 server scheduler MVP

Scope: this section reframes what sections 11-13 actually deliver,
documents the `ds4-server` scheduler observation that motivated the
reframing, and records the Day-1 2-slot continuous-batching MVP that
landed on top.

**The gen_tps / wall-clock discrepancy.**  Bench tool reports
`gen_tps_aggregate = cfg.gen_tokens * batch_n / gen_sec` where
`gen_sec` is the pure decode interval (post-prefill, post-snapshot)
-- see ds4_bench.c:396-444.  It excludes everything else in the
process wall-clock.  Measured directly:

| Config                          | gen_tps   | wall-clock | breakdown                                 |
|---------------------------------|----------:|-----------:|-------------------------------------------|
| N=1, no flags, batch=1          | 12.48     | 16.14 s    | prefill 5.23s + gen 4.01s + overhead 6.90s|
| N=1, ALL ON, batch=1            | 18.54     | 17.36 s    | prefill 5.27s + gen 2.70s + overhead 9.39s|
| N=4, ALL ON, batch=4            | 25.49 agg | 35.37 s    | prefill 20.04s + gen 7.85s + overhead 7.48s|

The N=1 ALL ON row is the killer: `gen_tps` claims +48 % over no-flags
single, but wall-clock is **+1.22 s WORSE** because the +2.49 s
initialization overhead (most likely `DS4_CUDA_Q4_DECODE=1` lazy Q4
conversion of 80 GB of weights -- `ds4-server --warm-weights` logs
"warmed tensor pages in 2.04s" which matches) eats the gen speedup.
At 50-token generations the warm-decode win is invisible to a single
user; only at longer generations does the gen region dominate
wall-clock.

**The server scheduler discovery.**  Started `ds4-server --cuda
--warm-weights` (Q4 lazy convert paid once at startup), then measured
HTTP requests with the same 2466-token prompt + 50-token gen:

- 1 request:                            9.19 s wall-clock
- 4 parallel HTTP requests:            36.29 s wall-clock total
- 4 sequential HTTP requests:          36.31 s wall-clock total

The 0.02 s difference between "4 parallel" and "4 sequential" is the
smoking gun: the request scheduler is FIFO with one worker thread
processing one global session at a time
(ds4_server.c:4719+ 8106 + 7510-7522 pre-a608e4d).
`ds4_session_eval_batched_decode` -- the entry point bench tool uses
and sections 11-13 measure against -- was **never called from the
server**.  The +29.9 % / +57.5 % numbers above are therefore upper
bounds reachable by a future scheduler, not deliverables in their
current shape.

**Day-1 MVP (commit a608e4d, 2026-05-13).**  Added a 2-slot
continuous-batching scheduler to `ds4-server`:

- Server gains `batched_sessions[2]` alongside the existing
  `s->session` (legacy path is unchanged).
- New helpers `dequeue_pair_wait(50 ms)`, `signal_done`,
  `request_is_batchable`, `generate_batched` in ds4_server.c.
- `worker_main` now tries to pair the freshly-dequeued job with a
  second batchable request that arrives within 50 ms.  If paired,
  `generate_batched` runs both prompts through per-session prefill
  (sequential) then decodes both in lockstep via
  `ds4_session_eval_batched_decode(slots, 2, ...)`.

Predicate `request_is_batchable` is deliberately narrow for Day-1:
OpenAI API only, non-stream, no tools, no thinking, no stops, no KV
disk cache, temperature > 0.  Everything else falls through to the
legacy `generate_job()` path.  This keeps the change shape low-risk:
DSML tool tracking, MTP speculative decode, Anthropic/OpenAI live
streams, structured streams, KV disk-cache reuse, and stop-string
scanning are all untouched.

Smoke (post-commit, GB10 --cuda --warm-weights --ctx 4096 ALL ON):
- Single non-stream chat:          9.66 s  (legacy path, unchanged)
- 2 parallel non-stream chats:    17.44 s (both `ctx=batched` markers
  in server log, batched_decode hit confirmed for both responses)
- N=2 serial via legacy (control): 18.11 s

Wall-clock saving on this smoke is small (~3.7 %) **because prefill
is sequential and prefill dominates the wall-clock at this
short-decode geometry**: 6.9 s prefill #1 + 6.9 s prefill #2 + 3.6 s
lockstep decode = 17.4 s, vs serial 2 × 9.0 s = 18.1 s.  Inside the
3.6 s lockstep decode region the per-session rate is 13.8 t/s
(aggregate 27.55 t/s), versus 18.54 t/s single-session decode -- so
the +49 % decode-region throughput is the real bench-equivalent win.
Long-decode workloads (chat with 500+ token responses) will see a
much larger end-to-end win because the decode region grows linearly
while prefill stays fixed per request.

**Day-2 levers, in order of expected impact:**
1. Parallel prefill (issue both sessions' `ds4_session_sync` calls so
   their kernel launches overlap on `g_kernel_stream`; today they
   serialize because they share the engine).  Brings the
   17.4 s smoke closer to 12 s.
2. N=2 → N=4/8 scheduler depth (queue allows it, batched_decode caps
   at 8 per ds4.c:18341).
3. Streaming SSE per slot (currently disabled in
   `request_is_batchable`; main work is plumbing the
   `sse_chunk(jobs[k]->fd, ...)` call inside the lockstep loop
   per-slot).
4. KV cache reuse for batched sessions (today `request_is_batchable`
   bails on `s->kv.enabled`; need per-session KV cache state).
5. `ds4_gpu_tensor_copy_async` stream consistency
   (NOTES top open-followups item 4) -- still applicable, no longer
   the limiting factor for batched server throughput at this scope.

### 16. Day-2 outcomes: continuous batching + KV cache + stream consistency landed

Scope: implement levers 1-5 from section 15 above (or revert + record
the reason).  All commits land on perf/batched-decode-poc and stack
cleanly on top of the Day-1 MVP (a608e4d).

**D2-1 (parallel prefill) -- REVERTED.**  Tried spawning a per-session
pthread that called `ds4_session_sync` concurrently for both batched
slots, hoping host-side encoder loop overlap would shrink the
sequential prefill cost (the dominant wall-clock fraction in Day-1).
CUDA returned "operation not supported on global/shared address space"
mid-prefill, with corrupted stderr garbage on subsequent error paths
-- the engine's `__constant__ g_batch_args`, scratch buffers, and
`g_kernel_stream` are shared state that the bench tool drives from a
single thread.  Two threads issuing prefill on different sessions of
the same engine race on every step.  Reverted cleanly (no commit
made).  The proper fix is **engine-level batched prefill** (a true
multi-row prefill API mirroring `ds4_session_eval_batched_decode`) --
that is a Day-3+ scope and not a server-only change.

**D2-2 (N=2 → N=4 depth, commit 228b62a).**  Bumped the pool to
`DS4_SERVER_BATCH_MAX = 4` (engine caps at 8, section 14 documented
N>8 instability so 4 is the conservative ceiling that still hits the
+29.9 % N=4 win).  Replaced the original "50 ms then non-blocking"
gather pattern with a 50 ms total time budget shared across N-1
gather attempts -- without this, near-simultaneous 4-curl bursts paired
as 2×N=2 instead of one N=4 because the second pair-mate beat the
non-blocking re-check by ~5-20 ms.  Verified: 4 parallel curls produce
`ctx=batched(n=4)` × 4 in the log.  Decode-region aggregate 29.9 t/s
(per-session 7.5 t/s) matches bench section 13's N=4 +29.9 % within
noise.

**D2-3 (per-slot SSE streaming, commit 06bfd85).**  Lifted the
`r->stream` exclusion from `request_is_batchable` and threaded plain
SSE writes through `generate_batched`.  Per-slot state arrays
(`stream_send / stream_headers_ok / stream_failed / plain_stream_pos`)
keep each slot's SSE flow independent -- a write failure on one slot
marks that slot done with finish=error but the lockstep loop keeps
running for the rest.  Scope: plain SSE only (sse_chunk delta +
sse_done).  The OpenAI live-stream structured flow (role chunks
beyond the initial one, tool-call deltas, openai_live finish) is NOT
exercised in batched mode -- has_tools and think_mode are already
excluded, so the remaining stream payload is a sequence of plain
content deltas which sse_chunk() handles correctly.  Smoke: 2 parallel
streaming chats land 66-line SSE outputs (role + 30 deltas + finish +
[DONE]) with `ctx=batched(n=2) ... stream=1` markers.

**D2-4 (`ds4_gpu_tensor_copy_async` -> `g_kernel_stream`, commit
f0e1248).**  Routed the device-to-device async copy through
`g_kernel_stream` instead of the default stream (0).  The default
stream worked through CUDA's legacy default-stream serialization but
inserted a synchronization gap that section 14 identified as the
leading cause of intermittent Shared-FFN_batched fatal corruption and
likely contributed to the wider bench variance under thermal load.
Smoke: N=1/2/4 all produce coherent Italian responses with no
corruption.  Sections 11-13 wins should be re-measured under a
cold-boot bench protocol; expected outcome is no regression and
possibly small improvement.

**D2-5 (per-session KV cache, commit 435e966).**  Lifted
`s->kv.enabled` from `request_is_batchable` and threaded
`ds4_session *` through the full kv_cache_* API so each batched
session maintains its own continued-store frontier.  Implementation:
- `kv_disk_cache.continued_last_store_tokens` changed from `int` to
  `int[1 + DS4_SERVER_BATCH_MAX]`, index 0 = legacy session, indices
  1..N = batched_sessions[0..N-1].
- `kv_cache_continued_store_target / kv_cache_note_store` signatures
  now take the counter explicitly, no longer reading from kc.
- `kv_cache_store_live_prefix / kv_cache_store_current /
  kv_cache_maybe_store_continued / kv_cache_try_load_text /
  kv_cache_try_load` all take an explicit `ds4_session *sess`.
- generate_batched calls kv_cache_try_load before sync (preserves
  cache hits) and kv_cache_store_current after decode (preserves
  state for next-turn hits).

Smoke: N=1 cold-stored 2048 tokens, then N=2 parallel chats both hit
the cache with `kv_cached=2048` per slot, wall-clock dropped to
**5.97 s** (vs Day-1 MVP's 17.4 s on the same geometry without KV
reuse, **-66 % time**).  Per-session KV state correctness confirmed
via .kv files in /tmp/ds4-kv/ -- cold-from-N=1, batched-continued
from N=2, batched-continued from N=4 all distinct files, no cross-
slot corruption.

**Where the wall-clock now goes (post-Day-2, 2466p + 30c smoke):**
- N=1 cold (legacy + KV cold store):       8.74 s
- N=2 batched + KV hit (both slots):       5.97 s  (-66 % vs Day-1)
- N=4 batched + KV hit (all four slots):  11.34 s  (-66 % vs Day-1)
- Single batched_decode step at N=2:       ~70 ms (decode-bound)
- Sequential prefill remains the largest residual cost when KV misses

**Day-3 lever shifts after Day-2:**
1. Engine-level batched prefill API (the real cure for the
   sequential-prefill cost; D2-1's host-thread attempt failed because
   the engine's `__constant__` / scratch / `g_kernel_stream` are
   single-driver-thread shared state).  Mirrors
   `ds4_session_eval_batched_decode`'s lockstep structure.
2. N=4 → N=8 cap when L2 working-set lets it (bench section 13
   showed +57.5 % at N=8, but section 14 flagged instability that
   needs re-measurement under D2-4's tighter stream discipline).
3. OpenAI live-stream structured flow in batched mode (role chunks,
   tool-call deltas) -- needed before batched can handle tool-using
   agents.
4. Per-session KV cache continued-store DURING batched decode (not
   only post-decode); requires per-batched-session progress callbacks.

### 17. Day-3 outcomes: N=8 stable, engine batched prefill scoped as Day-4+

**D3-1 (DS4_SERVER_BATCH_MAX 4 -> 8, commit 8542d34).**  Section 14's
flagged Shared-FFN fatal failures at N>=4 were caused by the
default-stream race that D2-4 (f0e1248) closed.  Under the tighter
stream discipline N=8 is stable end-to-end:

- 8 parallel curls collapse to one batched(n=8) lockstep
- `ctx=batched(n=8) ... kv_cached=2048 ... 22.97s` x 8 in server log
- All 8 responses coherent, 0 errors
- Aggregate throughput at N=8 ~10.4 t/s on this 2466p + 30c smoke,
  matching N=2 (10.1) and N=4 (10.6) -- L2 working-set saturates
  around N=2-4 (sharing-factor peak ~1.6x).  N=8's value at this
  geometry is queue-depth headroom for high request rates, not
  per-request latency reduction.

| N | wall-clock | aggregate t/s | sharing factor vs N=1 |
|---|----------:|--------------:|----------------------:|
| 1 | 6.70 s    | 4.5           | 1.00x baseline        |
| 2 | 5.97 s    | 10.1          | 2.25x                 |
| 4 | 11.34 s   | 10.6          | 2.36x                 |
| 8 | 22.97 s   | 10.4          | 2.32x                 |

The N=2 row showing the lowest wall-clock means single users sending
a request to a non-loaded server see the same response time as 2-user
parallel.  Above N=2, aggregate is the only metric that improves, and
even that flattens by N=4.

**D3-2 (engine-level batched prefill) -- NOT FEASIBLE in this scope.**
Reconnaissance found two architectural options:

- **Option A** (token-wise lockstep using existing batched-decode
  encoder): wrap the per-session prefill loop so all N sessions
  advance one prefill token per step in the same lockstep encoder
  that already powers batched decode.  Cheap to write (~150 LoC) but
  a **hard wall-clock regression**: the layer-major prefill encoder
  (`metal_graph_prefill_layer_major` at ds4.c:13164) achieves
  ~400 t/s per session by amortizing layer setup across a chunk of
  2048 tokens, while the decode encoder processes 1 token per layer
  per step at ~18.5 t/s single-session.  Token-wise prefill at N=4
  would take 4 x 2466 / 30 t/s = 328 s vs ~24 s sequential prefill --
  unusable.
- **Option B** (new multi-session prefill encoder): a real batched
  layer-major prefill that processes N sessions x n_tok tokens per
  layer in one kernel.  Mirrors `metal_graph_prefill_layer_major` but
  with `sessions[]` plumbing and per-session cur/next tensors.  Needs
  new kernel dispatch patterns (no existing batched prefill kernels
  on the GPU side).  Estimate 200-300+ LoC and multi-week scope.

Sequential prefill therefore stays the largest residual wall-clock
cost for cache-miss batched requests on this branch.  Day-4 work.

**D3-3 (OpenAI live-stream structured flow in batched mode) -- low
marginal value at this scope.**  The "live" flow's main differentiator
over plain SSE is tool-call streaming and reasoning_content
structured deltas.  request_is_batchable already excludes has_tools
and ds4_think_mode_enabled, so the only payload remaining is plain
content deltas -- which the simple sse_chunk path covers.  Real
OpenAI chat clients work with the current Day-2 D2-3 streaming.
Defer until tool-using agents are admitted into the batched path.

**D3-4 (KV continued-store during batched decode) -- not relevant at
typical generation lengths.**  `kv_cache_continued_store_target`
fires at multiples of `continued_interval_tokens` (default 10240).
Typical decode is 50-500 tokens; the session never crosses the
threshold during decode.  Defer until very long generations (>10k
tokens) become a workload.

**Day-4 starting points (in order of expected impact):**
1. Option B engine batched prefill -- the real cure for sequential
   prefill cost; biggest remaining wall-clock lever for cache-miss
   N>1 workloads.  ~2-week exploration.
2. Re-measure sections 11-13 bench numbers under D2-4's tighter
   stream discipline (cold-boot + 5-run median protocol) to confirm
   the +29.9 % / +57.5 % wins hold or improve.
3. OpenAI live-stream + tool-using requests in batched mode (D3-3
   gated on this).
4. Output head and routed-MoE batched substitutes (sections 11-13
   already covered most of attention; FFN side has the shared-FFN
   experiment in section 14 that needs the stream consistency fix
   to be re-evaluated; routed MoE is its own scope because each
   session selects its own experts).

### 18. Day-4 D4-1: SHARED_FFN re-evaluation post-D2-4 — kernel broken, not stream race

**REVISION (Day-6, post-engine-scratch-zero-init).**  The "kernel
broken" diagnosis in this section is wrong.  Day-6 retested
`DS4_CUDA_BATCHED_SHARED_FFN=1` on top of commit `2974421`
(zero-init for engine batched scratch on alloc) and found the
substitute produces fully coherent N=2/N=4 KV-hit output.  The
BOS-attractor failure documented below was caused by uninitialized
tile-padding rows in `e->batched_ffn_norm` / `batched_shared_gate`
/ `batched_shared_up` (cudaMalloc returned recycled GPU memory and
the Q4 batched-warp kernel folded those garbage values into its
warp-tile reads), not by a fundamentally broken Q4 batched matmul
at `N_EMBD -> N_FF_EXP = 2048`.

Day-6 D6-1/D6-2 therefore did not need to build a "new fused
gate+up Q4 pair kernel just to fix correctness" -- it builds one
for the activation-side traffic reduction it offers vs two
back-to-back batch-warp calls.  See section 20.

The original Day-4 D4-1 record follows for history:


Section 14 hypothesized that `DS4_CUDA_BATCHED_SHARED_FFN`'s
intermittent fatal failures (2/15 runs with corrupted error strings)
were caused by the `ds4_gpu_tensor_copy_async` default-stream race
that Day-2 D2-4 (f0e1248) closed.  Day-4 D4-1 tested this hypothesis
directly: start the post-D2-4 server with `DS4_CUDA_BATCHED_SHARED_FFN=1`
in env, run the same N=2/4/8 smoke that produced coherent Italian
under D3-1, and check for both crashes and content quality.

Result (warm-up + N=2 smoke, 2466p + 30c):
- **No crashes** — the fatal-failure path that section 14 saw is
  closed.  Server stayed up; both responses came back with valid
  HTTP+JSON envelopes and `completion_tokens=30`.
- **Silent numerical corruption** — the generated content was
  `<｜begin▁of▁sentence｜><｜begin▁of▁sentence｜>...` repeating.  This
  is the canonical symptom of logits collapsing to ~zero across the
  vocabulary, leaving argmax stuck on the BOS token.

D2-4 did help with the **stream consistency** part of section 14's
hypothesis (no more error-string corruption / crashes), but the
**numerical correctness** part is independent: the batched Q4 matmul
itself produces wrong output at the SHARED FFN geometry
(`DS4_N_EMBD -> DS4_N_FF_EXP = 2048`).  Section 14's original verdict
("naive Q4 batch matmul doesn't beat fused per-session pair matmul
at this layer's small out_dim") stands as definitive AND more severe
than originally diagnosed: the naive batch path is not just slow at
this geometry, it is **broken** (silent corruption).

`DS4_CUDA_BATCHED_SHARED_FFN` therefore remains NOT shippable.
Shipping a real SHARED_FFN substitute requires a **new fused
gate+up Q4 pair kernel**, not just a refactor of the call site --
this is the multi-day shared-FFN v2 work originally scoped in
section 14 and now formally re-confirmed.

### 19. Day-4 D4-2: engine batched prefill scaffolding (API in place, sequential body for now)

Scope: add the `ds4_sessions_sync_batched` public API to the engine so
future commits can replace the body with a true per-layer interleaved
multi-session prefill, without disturbing callers.  This is Day-4
close-out scaffolding; the real wall-clock win comes Day-5+ when the
body is replaced.

**What changed.**
- `ds4.h`: added `ds4_sessions_sync_batched(sessions[], prompts[], n,
  err, errlen)`.
- `ds4.c`: added the implementation -- a sequential loop calling
  `ds4_session_sync` per slot.  Equivalent wall-clock to the existing
  per-session loop; the engine's per-layer interleaved batched encoder
  is NOT exercised here yet.
- No caller change: `ds4-server`'s generate_batched still drives
  `ds4_session_sync` directly per slot.  Day-5+ switches it to the new
  API once the batched body is in place (so the caller-side error
  handling can be adjusted in lockstep with the new semantics).

**Why the body is sequential today, not batched.**  The prefill
encoder `metal_graph_encode_layer_attention_batch` (ds4.c:11119) is a
monolithic ~250-line function that walks all attention sub-ops
(rms_norm -> hc_attn_fn -> attn_norm -> attn_q_a -> attn_kv -> qkv_rms
-> attn_q_b -> head_rms -> rope -> KV store -> attention compute ->
attn_output) in one shot for one session.  Sections 11-13's
batched-decode substitutes work by splitting the decode encoder into
phases (DS4_DECODE_LAYER_PRE_A1/A2/B/C bitmask) and inserting
multi-session batched substitutes between phases; prefill has no such
split.  Building one is the Day-5 prerequisite for any substitute
that wants to pool N sessions' rows.

**Day-5 design (split into committable units).**

D5-1: **Phase-split metal_graph_encode_layer_attention_batch** into a
`_phased(g, ..., phases)` variant taking the same bitmask shape as the
decode side.  The existing `metal_graph_encode_layer_attention_batch`
becomes a thin wrapper that calls the phased version with ALL phases.
Pure refactor, no behavior change.  Suggested phase boundaries:
- PREFILL_HC_PRE      : rms_norm_plain_rows + hc_attn_fn + hc_split/weighted_sum
- PREFILL_ATTN_NORM   : rms_norm_weight_rows (attn_norm)
- PREFILL_Q_A         : attn_q_a matmul
- PREFILL_KV          : attn_kv matmul
- PREFILL_QKV_NORM    : dsv4_qkv_rms_norm_rows (fused or split)
- PREFILL_Q_B         : attn_q_b matmul + head_rms
- PREFILL_ROPE_KV     : rope_tail (Q+KV) + KV store + fp8 quantize
- PREFILL_ATTN_COMP   : compressor path (if ratio != 0)
- PREFILL_ATTENTION   : attention_prefill_raw_heads
- PREFILL_ATTN_OUTPUT : attn_output_a + attn_output_b matmuls
Estimated 200-300 LoC of variable hoisting and gate wrapping.

D5-2: **Outer batched prefill loop** -- a new
`metal_graph_prefill_layer_major_batched_n_sessions(sessions[], n_active,
prompts[], ...)` that iterates layers and (for now) within each layer
calls per-session phased(ALL) sequentially across sessions.  Sets the
stage for substitutes to slot in between phases.  Estimated 100-150
LoC.  Wires this into `ds4_sessions_sync_batched`'s body, replacing
the sequential per-session loop.

D5-3: **First batched substitute** -- `metal_graph_encode_attn_q_a_batched`
(per the recon agent's MVP estimate).  Gathers N sessions' batch_attn_norm
into engine-level scratch, runs one larger `ds4_gpu_matmul_q8_0_tensor`
(n_rows = n_active x n_tokens), scatters back to each session's
batch_qr.  Gated on `DS4_CUDA_BATCHED_PREFILL_Q_A` env var initially.
Estimated 120-150 LoC.

D5-4: **Server hookup** -- ds4_server.c generate_batched switches its
sequential prefill loop to `ds4_sessions_sync_batched`.  Error
handling adapts: on batched failure, fall back to per-session sync to
identify which slot broke (so other slots still finish).  Estimated
30-50 LoC.

D5-5+: **Stack more substitutes** (qkv, attn_output for prefill) the
same way sections 12-13 stacked on top of section 11's first
substitute.  Each subsequent substitute is a separate commit.

**Expected wall-clock impact.**  Day-3 D3-2 recon estimated 2-5 % per
substitute on the 2466p+30c geometry where prefill already dominates
(the per-session matmul is at n_tok=2048, so within-session weight
amortization is mostly there already).  Stacking 3-4 substitutes is
the realistic ceiling without Option B's full multi-session prefill
encoder rewrite.  For cache-miss N=4 workloads (sequential prefill
~24 s today), a 5-10 % win = 1-2 s saved.  Small but measurable; the
larger contribution is unblocking the same pattern for Day-6+ work
on the FFN side once the SHARED_FFN naive-Q4 kernel is replaced
(section 14 / 18).

### 19.1. Day-5 D5-1 land: phase split scaffolding (no behavior change)

The phase split landed exactly as designed above: a new
`metal_graph_encode_layer_attention_batch_phased(g, ..., phases)`
wraps the original ~1300-line attention-half encoder body with six
`if (phases & DS4_PREFILL_LAYER_PRE_<X>) { ... }` gates (PRE_A1/A2/B/
C1/C2/C3), and the existing
`metal_graph_encode_layer_attention_batch` becomes a thin wrapper that
calls the phased variant with `DS4_PREFILL_LAYER_ALL`.  No phase
boundaries reorder or skip any work today; existing callers (prefill_
layer_major and the split_profile path) see byte-identical behavior.

**Pre-existing bug found while bringing up D5-1.**  The bring-up
initially mis-attributed a regression to D5-1 because the smoke
relied on a single N=1 cold-prefill run and the output collapsed to
a BOS attractor.  After ruling out D5-1 (HEAD reproduces the same
failure), the bug was traced to uninitialized GPU memory: the
engine-level batched scratch (`engine->batched_attn_norm`,
`batched_qr`, `batched_kv*`, etc.) is sized for `DS4_BATCH_MAX` rows
and the Q4 warp-tiled matmul reads tile-padded rows beyond the
caller-supplied `n_active`; those rows must read as zero.  Plain
`cudaMalloc` does not zero, and when the driver hands back recycled
GPU memory the substitutes produce nonsense logits.  The fix
(landed separately as a sibling commit on this branch) adds
`ds4_gpu_tensor_clear` and calls it from `ensure_engine_scratch`
with `cudaMemsetAsync` ordered on `g_kernel_stream`.

**Why the failure looked like FP non-determinism.**  cudaMalloc
returns whatever GPU memory it has, so the garbage value was
*per-session deterministic* (same across calls within one server
process) but *per-process non-deterministic* (different across
server restarts).  A handful of "the bug is non-deterministic"
observations were just two server processes seeing two different
garbage values; the section-15 cuBLAS TC reduction-order story does
not apply.

Verification protocol for D5-2+ on this fixed baseline:
  1. Start server, run a warm-up curl to populate
     `/tmp/ds4-kv-fresh`.  The warm-up itself can be Q4-only-no-
     batched if a clean coherent KV file is desired, but with the
     scratch-clear fix the full batched env-var combo also produces
     coherent output on empty KV.
  2. Run N=1/2/4 parallel curls and verify each slot returns
     grammatically-correct Italian text discussing I Promessi
     Sposi.  Minor FP-noise tokens (occasional foreign-language
     fragments) are within expected variance; full BOS collapse
     indicates a real bug.

**Follow-up resolved.**  The remaining single-session uninit-read
was pinned down with `compute-sanitizer --tool=initcheck`: every
report on the no-batched single-session path traced back to
`metal_graph_warmup_prefill_kernels` -> `ds4_gpu_matmul_f16_tensor`
-> `f32_to_f16_kernel`, all reading `g->batch_flat_hc` straight off
a fresh `cudaMalloc`.  The warmup matmul's *output* gets overwritten
by real prefill (which is why the comment originally claimed "the
output is overwritten by the real graph"), but the per-process
deterministic garbage f32 input was perturbing cuBLAS state enough
to bias the test's sampling trajectory consistently in the same
direction across runs.  Disabling the warmup via
`DS4_METAL_NO_PREFILL_KERNEL_WARMUP=1` produced an initcheck-clean
run end-to-end, confirming this was the only remaining source.
Commit on this branch zero-inits `batch_flat_hc` before the warmup
matmul reads it; after the fix `tests/ds4_test.c` `--long-context`
no longer fails with the same assertion pattern three runs in a
row.  Residual flakiness on `--long-context` past this point is
ordinary cuBLAS TC reduction-order FP variance interacting with a
narrow sampling budget, not a stuck attractor.

### 19.2. Day-5 D5-2 + D5-4 land: outer batched loop + server wire-up

D5-2 adds `metal_graph_prefill_layer_major_batched_n_sessions`
(ds4.c) which drives N sessions through each layer in lockstep, then
runs the output head per session.  The per-session work is the same
`metal_graph_encode_layer_batch` the single-session path uses; the
only difference is that all sessions process layer `il` before any
session moves to layer `il+1`, so the layer weight takes one L2
sweep across the batch instead of N.  No batched substitutes
slot in yet -- that is D5-3+'s job, and the phase split from D5-1
is in place to support it.

D5-4 wires `ds4_sessions_sync_batched` into `ds4_server.c`'s
`generate_batched` path by splitting the prefill loop into three
phases:
  1. per-slot KV-disk-cache try-load + prompt collection
  2. one `ds4_sessions_sync_batched()` call across all slots
     (falls back to a per-slot `ds4_session_sync` loop on failure
     so the broken slot is identified without poisoning the others)
  3. per-slot post-sync bookkeeping (prompt_tokens, max_tokens,
     request id, rng seed)

**Eligibility today (silent per-session fallback).**  The batched
fast path bails on the first check that fails:
  - any session on a CPU backend
  - any session bound to a different engine
  - any prompt longer than that session's prefill_cap
  - any session with a live checkpoint (i.e. KV-disk-cache hit
    landed during phase 1 -- which is the common bench case)
The last check is the most consequential: in the handoff bench
workflow (warm-up curl + N=2/4/8 parallel) every parallel slot has
a KV-hit checkpoint, so the fast path falls back and the slots are
synced sequentially via `ds4_session_sync` (= the pre-D5-4
behavior).  The fast path does fire on the warm-up's empty-KV
N=1 case, where it is operationally equivalent to the single-
session prefill it replaces.  Lifting the checkpoint-resume
restriction is on the D5-5+ list -- it requires a batched analog
to `metal_graph_prefill_chunked_range`.

**Verification.**  Handoff bench workflow against
promessi_sposi.txt (warm-up + N=2 parallel + N=4 parallel): all
six KV-hit slots return grammatically-correct Italian Manzoni
discussion (occasional FP-noise tokens within Section-15
variance).  N=1 empty-KV warm-up still sometimes collapses to BOS
-- that is the section-19.1 known follow-up bug, not new
regression from D5-2/D5-4.

### 19.3. Day-5 D5-3 land: prefill PRE_A2 batched substitute (scaffolding only)

Adds `metal_graph_encode_prefill_qkv_batched`, the prefill-side analog
of decode's `metal_graph_encode_qkv_batched` (ds4.c:16060).  Gathers
per-session `g->batch_attn_norm` rows into engine-level
`batched_prefill_attn_norm`, runs one pair of Q8 matmuls (attn_q_a +
attn_kv) + one fused `dsv4_qkv_rms_norm_rows` across `n_active * n_tok`
rows, then scatters back to per-session `batch_qr_norm` / `batch_kv`.

A new `ensure_engine_scratch_resizable` helper allocates and clears
each prefill scratch tensor on demand (sized for the current
`n_active * n_tok`, resized when a bigger workload arrives so memory
matches the actual peak rather than reserving the worst case).

The substitute is opt-in behind `DS4_CUDA_BATCHED_PREFILL_QKV=1` and
fires from `metal_graph_prefill_layer_major_batched_n_sessions` only
when (a) the env var is set, (b) `n_active >= 2`, and (c) all
sessions have the same `n_tok`.

**Empirical wall-clock impact: net negative on cold N=2 today.**  In
practice the substitute slows prefill: per-session prefill already
runs at `n_tok ~= 944-2048` rows per matmul, large enough that one
batched matmul of `N * n_tok` rows is no faster than N back-to-back
matmuls (each weight is L2-hot across the back-to-back calls; the
GPU is already saturated).  The substitute's gather + scatter add
~30 MiB of memory copies per layer per request that the per-session
path does not pay, and the larger batched matmul's reduction order
differs enough to widen the FP-variance band into the BOS-attractor
zone at temperature=0 (caller-visible as "first N tokens coherent
then BOS" in the substitute-on smoke).

In short: the section-19 recon estimate of 2-5% per substitute does
not generalize to substitutes that replace already-amortized
per-session matmuls.  Decode-side wins came from collapsing
*degenerate* (n_rows=1) matmuls into batched (n_rows=N) ones;
prefill's per-session matmul is not degenerate, so collapsing N of
them does not pay.

**Why land it anyway.**  The code is the pattern D5-5+ substitutes
on the FFN side will copy (shared FFN gate+up, routed MoE) -- those
*are* per-session amortization-limited (smaller per-row matmuls or
expert-sized) and have a real shot at wall-clock wins.  Landing the
scaffolding (engine struct fields, the resizable scratch helper,
the two-pass phased call inside the outer loop, the env-var-gated
substitute fork) means D5-5+ work can re-use it directly instead
of rebuilding it.  The substitute itself stays opt-in and OFF by
default; no user is affected.

**Known follow-up.**  The substitute also exhibits the
section-19.1 single-session BOS-attractor more often than the
per-session path.  Whether that is purely the reduction-order FP
variance or a missed clear/sync in the gather-matmul-scatter chain
is unresolved; a bit-exact comparison against per-session output
on a small fixed input is the next step.  Until then, keep the env
var off for any run whose output is consumed (warm-up curls that
only populate KV are fine to flip it on for, since the bad output
is discarded).

### 20. Day-6 D6-1 + D6-2: SHARED_FFN Q4 pair-batch kernel + wire-up

**Background revision: section 18 was wrong about why SHARED_FFN was
broken.**  Section 18 (Day-4 D4-1) concluded that
`DS4_CUDA_BATCHED_SHARED_FFN=1` was broken because the Q4 batched
matmul kernel itself produced silent corruption at
`N_EMBD -> N_FF_EXP = 2048`.  That diagnosis pre-dates the
engine-batched-scratch zero-init fix in commit `2974421` (sibling
fix landed during Day-5 bring-up): cudaMalloc returns whatever GPU
memory the driver has, so the gather-target tile
(`e->batched_ffn_norm` etc.) contained garbage in its tile-padding
rows, the Q4 batch-warp kernel folded those rows into reductions,
and the resulting bad activations cascaded to all-BOS logits.
After `2974421` lands, the existing
`metal_graph_encode_shared_ffn_batched` (ds4.c line ~15962) is
correctness-clean; section 18's revision note records this.

**D6-1: new Q4 pair-batch kernel.**  Adds
`matmul_q4_0_pair_preq_batch_warp_kernel` to ds4_cuda.cu plus the
`ds4_gpu_matmul_q4_0_pair_batch_warp_tensor` wrapper.  Generalizes
the existing single-token `matmul_q4_0_pair_preq_warp8_kernel`
(two Q4 weight matrices, one shared activation, two outputs --
used by per-session shared_gate_up at n_tok=1) over n_tok rows.
Reads the shared activation once per (tok, block) and emits both
out0 and out1 from one warp; this halves the activation-side HBM
traffic vs two back-to-back `matmul_q4_0_batch_warp` calls (the
gate and up matrices are still read independently because they
sit at different model-file offsets).

**D6-2: wire-up.**  `metal_graph_encode_shared_ffn_batched`
(ds4.c) prefers the new pair-batch helper; if it rejects the
geometry (e.g. dp4a unsupported, oversized weights), it falls
back to the original two-call pattern so the path is no worse
than today.  The two ds4_gpu_matmul_q4_0_batch_warp_tensor calls
stay in the file for the fallback.

**Verified correctness.**  Day-6 D6-2 smoke (warm-up via Q4-only
to populate KV, then N=4 parallel KV-hit with full env-var combo
+ `DS4_CUDA_BATCHED_SHARED_FFN=1`): all four slots return
grammatically-correct Italian Manzoni text with the
section-15 FP-noise band.  No BOS-attractor regression vs the
zero-init-restored naive double-matmul path.

**Performance: needs cold-boot measurement.**  Same-session
back-to-back tests showed wall-clock numbers that drift upward
across runs (12.3s -> 15.9s -> 21.3s for N=4 KV-hit across three
configs in the same session), matching section-14's "thermal
envelope" caveat -- the GB10's bench-cumulative thermal floor is
the dominant variance here, not the kernel choice.  The pair
kernel's design saving (half the activation-side reads) is real
on paper but cannot be cleanly quantified in this session.
Reopen with a cold-boot single-session measurement.

**What this unlocks.**  The pair-batch kernel and its wrapper are
the building blocks the routed-MoE batched substitutes will also
want (gate + up per expert at the same offset pattern).  Day-7+
work can re-use this directly.

### Appendix A. Reproducible server smoke recipe (consolidated)

This section consolidates the build + run + verify recipe that
sections 1-20 have been using ad-hoc.  Future sessions can reach
this in one place instead of stitching together commands from
sections 11-20.

**1. Build for GB10.**

```
make cuda-spark
```

That target expands to `make ds4 ds4-server ds4-bench CUDA_ARCH=`,
which compiles ds4.o with the default C flags and `ds4_cuda.o` with
nvcc's `--use_fast_math` (CUDA_ARCH= lets nvcc auto-detect sm_121
for GB10).  Optional `make cuda-regression` runs the long-context
correctness test; not required for a perf smoke.

**2. Start the server with the bench env-var combo.**

The handoff bench workflow uses all four opt-in batched-decode
substitutes plus the Q4 lazy-convert cache.  Empty KV-disk-cache
dir on the first run is fine; subsequent runs benefit from the
on-disk KV that the warm-up populates.

```
mkdir -p /tmp/ds4-kv
DS4_CUDA_Q4_DECODE=1 \
DS4_CUDA_BATCHED_QKV=1 \
DS4_CUDA_BATCHED_Q_B=1 \
DS4_CUDA_BATCHED_ATTN_OUTPUT=1 \
  ./ds4-server --cuda --warm-weights --port 8000 --ctx 4096 \
               --kv-disk-dir /tmp/ds4-kv \
               > /tmp/ds4-server.log 2>&1 &
disown
until ss -tlnp 2>/dev/null | grep -q 8000; do sleep 2; done
```

Env-var reference (full table in `docs/GB10_JOURNEY.md` lines
321-340; per-substitute land in sections 11-13):

| Env var | Phase replaced | Section |
|---|---|---|
| `DS4_CUDA_Q4_DECODE=1` | Q4 lazy weight cache + batched output head | baseline gate |
| `DS4_CUDA_BATCHED_Q_B=1` | decode PRE_B (attn_q_b) | 11 |
| `DS4_CUDA_BATCHED_QKV=1` | decode PRE_A2 (attn_q_a + attn_kv + qkv_rms_norm) | 12 |
| `DS4_CUDA_BATCHED_ATTN_OUTPUT=1` | decode PRE_C2 (attn_output) | 13 |
| `DS4_CUDA_BATCHED_PREFILL_QKV=1` | prefill PRE_A2 (D5-3 scaffolding; **leave off by default**) | 19.3 |
| `DS4_CUDA_BATCHED_SHARED_FFN=1` | broken Q4 batched matmul at FFN gate/up (**do not set**) | 14, 18 |

**3. Verification curls.**

```
PROMPT=$(head -c 8000 speed-bench/promessi_sposi.txt)
BODY=$(jq -nc --arg p "$PROMPT" \
  '{messages:[{role:"user",content:$p}],max_tokens:30,thinking:{type:"disabled"}}')

# Warm-up populates /tmp/ds4-kv with a coherent KV file for the prompt.
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" -d "$BODY" -o /tmp/r-warm.json

# Parallel KV-hit smokes (each slot should return Italian Manzoni text).
for N in 2 4 8; do
  time (for i in $(seq 1 $N); do
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" -d "$BODY" \
      -o /tmp/r-$N-$i.json &
  done; wait)
  for i in $(seq 1 $N); do
    jq -r '.choices[0].message.content[0:100]' /tmp/r-$N-$i.json
  done
done
```

**4. Expected output shape.**

Each KV-hit slot returns a sensible Italian opener referencing
*I Promessi Sposi* / *Promessi sposi* / Alessandro Manzoni /
*Introduzione*, often paraphrasing the prompt's first paragraph.
Acceptable FP-noise tokens (section 15): occasional Arabic /
Cebuano / Filipino / fictional fragments mixed into otherwise
correct Italian -- the cuBLAS f16 TC reduction order varies enough
to flip the argmax on near-tied tokens.  **Full BOS-attractor
collapse** (`<｜begin▁of▁sentence｜>` repeating after a few
coherent tokens) is **not** the expected variance; it indicates
a real bug.

**5. Q4 vocab head BOS-bias caveat (section 19.1 follow-up note).**

Post-`6ca93c3` initcheck reports zero uninit reads on the single-
session decode path, so the historical "warm-up sometimes BOS"
explanation (uninit GPU memory) no longer applies.  What is still
true: `DS4_CUDA_Q4_DECODE=1` routes the vocab projection through the
batched Q4 warp matmul (`metal_graph_encode_output_head_batched` ->
`ds4_gpu_matmul_q4_0_batch_warp_tensor`) instead of the Q8 path, and
the Q4 logits are biased enough toward `<|begin_of_sentence|>` that
greedy decode locks into BOS attractor.  Reproduction on `2026-05-14`
(both ds4flash.gguf models):

  - Q4_DECODE + temperature 0 (greedy) -- BOS attractor 100% (both
    imatrix and legacy gguf)
  - Q4_DECODE + temperature 1 (default) -- coherent Italian output
  - Q4_DECODE + temperature 1 + the full batched env-var combo
    (BATCHED_QKV + BATCHED_Q_B + BATCHED_ATTN_OUTPUT) -- BOS
    attractor returns at N=4 parallel (batched substitutes fire,
    interact with the Q4 logit bias)

Workaround: leave temperature at the default (1.0+) when
`DS4_CUDA_Q4_DECODE` is on, or unset it for greedy correctness
testing.  Root cause is the Q4 vocab matmul kernel, not unitialized
memory; fix likely needs per-row scale-factor calibration in the
lazy Q8 -> Q4 conversion at `cuda_q4_from_q8_ptr` so the BOS row's
quantization error doesn't elevate its dot-product.

**6. Server log sanity checks.**

```
grep "ctx=batched(n=" /tmp/ds4-server.log
```

Each parallel run should emit one `ctx=batched(n=N)` line per slot
with `gen=<max_tokens> finish=length kv_cached=<>`.  The
`kv_cached=2048` field after warm-up confirms that the
KV-disk-cache hit is fully consumed and only the tail
(prompt.len - 2048) tokens go through fresh prefill.

### What NOT to retry without a new theory

Already shown not to help on this hardware / model:
- Templated weight-shared kernel that collapses grid.y (occupancy loss)
- cuBLAS f16 GEMM at n_tok > 1 (2× weight HBM vs Q4)
- Batching only the output head (it's already near-optimal)
- L2 persistence policy with 18 MiB pin (24 MiB L2 too small)
- Full CUDA Graph capture on this encoder *without* the kernel scalar
  lift (regression matches cuda-graph-wip; ceiling is ~8 % anyway)
- MTP speculative decode at default temperature (CLI gates on
  `temperature == 0`; even with temp = 0 it's empirically neutral on
  GB10 — see section 6)
- Q3_0 with a single fp16 scale per 32-value block (quality collapses
  to garbage tokens AND decode is compute-bound below Q4's HBM-bound
  rate — see section 7)
- Q3_K_S with sub-block scales but simple (non-refined) sub-scale
  derivation (quality still collapses to broken tokens AND decode is
  still compute-bound, -19.5 % at N=1 — see section 8)
- INT8 tensor-core GEMV at decode shape (1/16 N utilization × Q8 HBM
  regression = -49 % vs Q4 dp4a baseline; pre-gated on cross-session
  batching + N ≥ 8 stability — see section 9)

## What is committed and ready for next session

Phase 1a-1c (the original PoC scaffolding):
- 60664bc Phase 1a — sync amortization (+2-3 %)
- 1ea0307 Phase 1b — `__constant__` batch args + uploader
- 635629d Phase 1c-1 — `matmul_q4_0_preq_batch_warp8_kernel` + wrapper
- 503c725 Phase 1c-2 — batched output head wiring
- c44c0e6 per-layer interleaved batched decode encoder (+3-5 % at N=2/4)

Layer-encoder N-aware refactor (the "real lever" — sections 11-13):
- 88f64e3 batched attn_q_b (PRE_B substitute, +3.9 % N=4)
- 8175c3a batched attn_q_a + attn_kv + qkv_rms_norm (PRE_A2 substitute, +5.1 % stacked)
- b2be06c batched attn_output (PRE_C2 substitute, +26.9 % combined)
- 319cac8 DS4_BENCH_PRINT_TOKENS correctness hook
- efa9390 simplify pass: encoder per-session helper, ensure_engine_scratch
  generalization, DS4_CUDA_BATCH_MAX + static_assert (-59 LoC, win preserved)

Server scheduler (the path from microbench to user, section 15):
- a608e4d 2-slot continuous-batching scheduler MVP in `ds4-server`,
  exercising `ds4_session_eval_batched_decode` for the first time
  from a non-bench binary (~3.7 % wall-clock win on 2x parallel curl
  smoke; decode-region throughput +49 % per server log)

Day-2 follow-ups on top of the Day-1 MVP (section 16):
- 228b62a N=2 -> N=4 batched scheduler depth + time-budget gather
- 06bfd85 per-slot SSE streaming inside generate_batched
- f0e1248 ds4_gpu_tensor_copy_async routed through g_kernel_stream
- 435e966 per-session KV cache in generate_batched (N=2 wall-clock
  with KV hit dropped from 17.4 s to 5.97 s, -66 %)

Day-3 land (section 17):
- 8542d34 DS4_SERVER_BATCH_MAX 4 -> 8 (N=8 stable under D2-4 stream
  discipline; 8 parallel curls collapse to one batched(n=8) lockstep,
  22.97 s wall-clock for 8x cache-hit requests, aggregate flat ~10 t/s)

Day-4 (sections 18, 19):
- D4-1 (no commit): SHARED_FFN re-evaluation post-D2-4 confirmed silent
  numerical corruption -- D2-4 closed the crash path but the naive Q4
  batch matmul is fundamentally broken at the SHARED FFN's
  DS4_N_FF_EXP=2048 geometry.  Section 14's verdict re-confirmed.
- D4-2: ds4_sessions_sync_batched API added to ds4.h + ds4.c (engine
  scaffolding for Day-5+ batched prefill; today the body is sequential
  per-session, equivalent to the existing pattern).

Phase A infrastructure (kernel stream plumbing, sections 1-10):
- 1d7402e all kernel launches threaded through g_kernel_stream
- 6412fad, 86e5144 dead-end shelving notes (Option I' / G / N>8 cliff)
- 2c5da98, 45a5f93, 666b031 Q3_0 / Q3_K_S / INT8 WMMA documented dead-ends

The batched continuous-decode API works correctly at N = 1 – 16, falls
back cleanly to serial outside that range, and the batched substitutes
preserve output coherence (verified via DS4_BENCH_PRINT_TOKENS: all
five env-var combinations produce grammatically-correct Italian text
on the promessi_sposi.txt prompt; argmax tokens differ in low-confidence
positions due to FP-accumulation order, not broken kernels).

Open follow-ups in priority order:
1. Fresh-boot N = 8 re-measurement (current N = 8 is thermally noisy).
2. Extend the stage→batched→scatter pattern to shared FFN gate+up (NOTES
   "Layer-internal FFN batching" said this needs a custom Q4 pair kernel
   to actually pay; revisit with the now-validated infrastructure).
3. Bench-gate `ds4_gpu_tensor_copy_async` switch to `g_kernel_stream`
   for stream-consistency with Phase A (section 9 of the local-LLM
   review, deferred because the +26.5 % win relies on the current
   serialization order).
4. Push DS4_BATCH_INTERLEAVED_MAX from 8 to a verified higher value
   once shared-FFN batching lands and the per-session L2 working set
   stops scaling with N (section 10 re-open condition).

### Upstream cherry-pick: a464840 "cuda: reduce long-context prefill slope"

antirez landed a much larger CUDA selector / top-k / indexed-
attention rewrite on `upstream/main` (+633 lines `ds4_cuda.cu`).
Cherry-picked onto this branch as `898d42c`.  Two conflict zones:

- `indexer_scores` selector: upstream replaced the single
  `indexer_scores_wmma_kernel` dispatch with a wmma128 → wmma64 →
  wmma32 → wmma cascade.  Took upstream's cascade and added
  `, 0, g_kernel_stream` to all four launches to preserve the
  Phase-A stream invariant.
- `attention_indexed_mixed_batch_heads_tensor` selector: upstream
  rewrote `attention_indexed_mixed_heads8_online_kernel` to a
  templated `<8, 16>` variant with a 16-head grid and 512 threads,
  and added a topk-index sort prepass (`indexed_topk_sort_512_asc_
  kernel`).  Took upstream's selector + new online kernel; our
  earlier `abdf382` env-var flip (`DS4_CUDA_INDEXED_ONLINE`) is
  silently superseded because the line it modified now reads from
  the upstream `DS4_CUDA_INDEXED_TWOPASS` selector.  `abdf382`
  stays in history for traceability; it is a no-op after the
  cherry-pick.

Four additional new kernel launches added wholesale by upstream
(`indexer_topk_8192_cub_kernel` x2, `indexer_topk_pow2_u16_kernel`,
`indexed_topk_sort_512_asc_kernel`) were also routed through
`g_kernel_stream` so the Phase-A invariant holds across the entire
new code path.

Measured on this branch with `ds4-bench --cuda --ctx-start 8192
--ctx-max 32768 --step-mul 2 --gen-tokens 32`:

| ctx   | Before cherry-pick | After   | Δ      |
|-------|-------------------:|--------:|-------:|
| 8K    | 328.19 t/s         | 387.10  | +18.0% |
| 16K   | 286.98             | 374.60  | +30.5% |
| 32K   | 251.72             | 356.35  | +41.6% |

Generation tps unchanged (within noise).  Regression suite
(`--server --metal-kernels --tool-call-quality --logprob-vectors`)
green on the first run.
