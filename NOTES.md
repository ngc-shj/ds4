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

### Combined finding

The batched output head matmul is not the bottleneck and is already
close to optimal for this shape under DS4_CUDA_Q4_DECODE.  Further
optimization of the output head alone has no measurable headroom.

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

## What is committed and ready for next session

- 60664bc Phase 1a — sync amortization (+2-3 %)
- 1ea0307 Phase 1b — `__constant__` batch args + uploader
- 635629d Phase 1c-1 — `matmul_q4_0_preq_batch_warp8_kernel` + wrapper
- 503c725 Phase 1c-2 — batched output head wiring

The batched continuous-decode API works correctly at N = 1 – 16, falls
back cleanly to serial outside that range, and produces bit-equivalent
output to the per-session path.  Layer-level batching (the real win) is
the next session's work.
