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

### What NOT to retry without a new theory

Already shown not to help on this hardware / model:
- Templated weight-shared kernel that collapses grid.y (occupancy loss)
- cuBLAS f16 GEMM at n_tok > 1 (2× weight HBM vs Q4)
- Batching only the output head (it's already near-optimal)

## What is committed and ready for next session

- 60664bc Phase 1a — sync amortization (+2-3 %)
- 1ea0307 Phase 1b — `__constant__` batch args + uploader
- 635629d Phase 1c-1 — `matmul_q4_0_preq_batch_warp8_kernel` + wrapper
- 503c725 Phase 1c-2 — batched output head wiring

The batched continuous-decode API works correctly at N = 1 – 16, falls
back cleanly to serial outside that range, and produces bit-equivalent
output to the per-session path.  Layer-level batching (the real win) is
the next session's work.
