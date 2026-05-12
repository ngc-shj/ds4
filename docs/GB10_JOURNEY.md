# DS4 on NVIDIA GB10 — Fork Journey

This fork of [`antirez/ds4`](https://github.com/antirez/ds4) is a sustained
effort to make DeepSeek V4 Flash run well on NVIDIA Grace+Blackwell
hardware (GB10 / DGX Spark, sm_121), and to keep around the small,
hardware-specific performance work that does not belong upstream.  The
narrative below tracks what was tried, what worked, and — perhaps more
usefully — what didn't.

## At a glance

| Chapter | Branch | Theme | Outcome |
| --- | --- | --- | --- |
| 1 | `cuda-gb10-backend` | Original Metal → CUDA port (73 entry points, 5 rounds of numerical fixes, mmap pinning, end-to-end inference). | Functionally superseded once upstream merged its own CUDA backend, but the post-mortem in `PORT_CUDA.md` remains the canonical bisection guide. |
| 2 | `perf/cuda-hostregister-fallback` | GB10 needs `cudaHostRegisterDefault` (not `ReadOnly`) for the GGUF mmap, and indexed-attention is faster register-blocked on Blackwell. | Two surgical fixes.  Base for every later perf branch. |
| 3 | `perf/q4-only` | Lazy `Q8 / f16 → Q4_0` weight cache + `dp4a` matmul variants for decode, then faster nibble unpack via `__byte_perm` + int32-batched loads. | Decode **12.33 → 18.80 t/s (+52.5 %)** on the chat-v2 GGUF.  Merge candidate. |
| 4 | `perf/cuda-graph-wip` | CUDA Graph capture for the per-token decode step, with `__constant__ g_step_args` carrying the per-token mutable state. | **Net negative on GB10** (capture / launch overhead dominates).  Dormant; kept as a reference implementation of the `g_step_args` pattern. |
| 5 | `perf/batched-decode-poc` | True N-session continuous batched decode (the prerequisite for `ds4-server` continuous batching). | Phase 1a-1c shipped: API + scaffolding + batched output head.  Layer-internal matmul batching is multi-day work, deferred; current ceiling **~+7 % aggregate at N = 8** without that surgery.  See the branch's `NOTES.md`. |

All four `perf/*` branches stack linearly on top of upstream `main`:

```
upstream main (CUDA support already merged)
└─ perf/cuda-hostregister-fallback   (2 commits — GB10 mmap + attention)
   └─ perf/q4-only                   (+2 commits — Q4 lazy cache + faster unpack)
      ├─ perf/cuda-graph-wip         (+1 commit  — CUDA Graph experiment, dormant)
      └─ perf/batched-decode-poc     (+7 commits — batched-decode PoC, current)
```

## Chapter 1 — `cuda-gb10-backend`: porting the Metal engine to CUDA

When this fork started, `ds4` was a Metal-only engine for DeepSeek V4
Flash on Apple Silicon.  The `cuda-gb10-backend` branch is the original
attempt to make the same engine run on NVIDIA Grace+Blackwell, with the
unified-memory assumption that the Metal design implicitly relies on.

The port went in roughly this order:

1. **Build + dispatch** (`b281956`, `2825863`, `8f96bc6`).  Introduce
   `DS4_BACKEND=cuda`, implement all 73 `ds4_metal_*` entry points
   (tensor lifecycle, RMSNorm, RoPE-tail YaRN, KV store / FP8, Q8/F16
   matmul, FlashAttention variants, compressors, indexer, router, MoE,
   hyper-connection, embeddings), wire `--cuda` to the CLI.
2. **First end-to-end run**.  Inference completes on the 86.7 GiB Q2
   GGUF; numbers are bad but the pipeline is alive.

   | Configuration | Prefill | Generation |
   | --- | ---: | ---: |
   | Initial (no mmap pinning) | 1.34 t/s | 3.51 t/s |
   | After `cudaHostRegister` + `cudaMemAdvise` | **7.92 t/s** | **7.22 t/s** |
   | After compressor_update full pipeline | 8.16 t/s | 7.28 t/s |

3. **Five rounds of numerical accuracy fixes**, all bisected against
   the CPU reference with `--metal-graph-test` and per-tensor `dump_diff`:

   | Round | What broke | Effect on `logits` diff |
   | --- | --- | ---: |
   | 1 | `sqrt(softplus())` router probs / F32 attention sinks / HC-expand split | 40.6 → 4.5 |
   | 2 | Hash-mode router used uniform 1/6 weights | 4.5 → 0.18 |
   | 3 | `cudaHostRegisterDefault` over `ReadOnly` for the mmap (perf, not accuracy) | unchanged |
   | 4 | Compressor prefill missing post-emit RMSNorm + RoPE pass; FP8 KV missing per-64-element power-of-2 scaling | story prompts ~30 tokens stable |
   | 5 | `prefill_static_mixed` comp visibility — match Metal's `n_visible = (q+1) / ratio` | story prompts ~50 tokens stable |

   Diagnostic tools introduced along the way (`a3e7f47`,
   `b8e1173`, `2408546`) — per-layer CPU/CUDA diff, prefill+decode
   bisection harness, Metal-vs-CUDA binary tensor dump — outlived the
   branch and are still the fastest way to localise a kernel-level
   correctness regression.

4. **The compressor_prefill grid-y bug** (`9c136e0`).  `hc_expand_*`
   kernels dispatched with `grid.y = nh = 4`, which silently dropped
   most batched prefill rows; classic example of a Metal idiom that
   needed re-thinking in CUDA's launch geometry.

What this chapter cost was buying every layer of the engine, line by
line, on a different GPU vendor.  What it bought was the working
end-to-end run that made the upstream port easier when antirez later
merged official CUDA support (`0ac5df3` + `48beef8`).  The branch is no
longer in the merge path, but `PORT_CUDA.md` on the branch is the
canonical reference for anyone hitting a CUDA bring-up bug on similar
hardware.

## Chapter 2 — `perf/cuda-hostregister-fallback`: small GB10-specific fixes

Once upstream picked up its own CUDA backend, the fork shifted to
patching things that only matter on actual GB10 hardware.

- **`7a0f8c0` cuda: fall back to cudaHostRegisterDefault when ReadOnly is
  declined.** The linux kernel + file-backed mmap combination on GB10
  refuses `cudaHostRegisterReadOnly`.  Fall back to plain `Default`,
  then add `cudaMemAdvise(SetReadMostly, SetPreferredLocation)` to tell
  unified memory to cache weight rows for matmul reuse.  This alone
  takes prefill from 1.3 → 7.9 t/s (see Chapter 1 table).
- **`0d1bc24` cuda: switch indexed-attention batched kernel to
  register-blocked by default.**  The cooperative shared-mem variant
  underperforms the register-blocked variant on Blackwell's larger
  register file and warp-scheduler shape.

Both fixes are upstream-friendly but small enough to keep here while the
rest of the perf work is still in flight.

## Chapter 3 — `perf/q4-only`: the biggest single decode win (+52 %)

Built directly on top of Chapter 2.  Two commits:

1. **`aaee405` cuda: lazy Q8/f16 → Q4_0 cache + Q4 dp4a matmul variants
   for decode.**  At decode (`n_tok = 1`), the bottleneck is weight HBM
   bandwidth, not FLOPS.  Convert the Q8/f16 weight rows to the split-row
   Q4_0 layout (18 bytes / block: scales then packed nibbles) lazily on
   first decode reference, cache the result, and add `matmul_q4_0_preq_*`
   dp4a variants (single, pair, hc_expand) so the decode path stays on Q4
   instead of falling back to Q8.  Roughly 50 % of the per-token weight
   read disappears.
2. **`5d9f2bb` cuda: faster Q4_0 nibble unpack via `__byte_perm` +
   int32-batched loads.**  Inside `q4_block_dot`, replace the lookup-table
   nibble unpack with PRMT (`__byte_perm`) and read 16 packed bytes as
   four `uint32_t` words.  Tightens the inner loop noticeably.

End result on chat-v2 GGUF (one ds4-bench frontier at ctx=2048,
gen=50):

| Config | Decode (t/s) |
| --- | ---: |
| Upstream CUDA baseline (Q8) | 12.33 |
| `perf/q4-only` (Q4 lazy cache + faster unpack) | **18.80 (+52.5 %)** |

This is the merge-candidate branch.  Prefill is unaffected because the
`n_tok > 1` path stays on cuBLAS f16 (where FLOPS dominate over HBM).

## Chapter 4 — `perf/cuda-graph-wip`: the CUDA Graph experiment

CUDA Graphs amortise per-kernel launch overhead by recording an entire
sequence of kernels and replaying it with one `cudaGraphLaunch`.  For
decode (which issues ~2500 kernels per token), this looks like an
obvious win.

The branch added:

- A `__constant__ ds4_step_args` symbol holding the per-token mutable
  state (`pos`, `raw_row`, `n_raw`, `n_comp`, `comp_row`, `emit`,
  `token`, indexer `top_k`, etc.), uploaded once per token via
  `cudaMemcpyToSymbolAsync`.  The previously per-launch scalar kernel
  arguments became reads from this `__constant__` so the captured graph
  is structurally identical across tokens.
- Capture-and-replay scaffolding gated by an env var.

What the measurements showed on GB10:

- Capture overhead is real and the captured graph isn't measurably
  faster than the live launch sequence on this hardware.
- The mid-token `allow_split_flush` sync used by the live path interacts
  badly with capture.
- Net effect: small but consistent regression.

The branch is dormant rather than discarded, because:

1. The `g_step_args` design is still the right shape for the layer-level
   batching the `perf/batched-decode-poc` work eventually needs.  The
   `__constant__` batch-args struct introduced in
   `perf/batched-decode-poc`'s Phase 1b commit `1ea0307` is a deliberate
   adaptation of this design (per-row arrays of the same fields).
2. The `SKIP_UPDATE` graph-launch path, the node-diff infrastructure,
   and the single-value `g_step_args` refactor remain a useful reference
   if CUDA Graphs are revisited on later hardware where the trade-off
   tips the other way.

## Chapter 5 — `perf/batched-decode-poc`: continuous batched decode

The motivation is `ds4-server`: serving multiple chat sessions
concurrently on one GB10 requires a forward pass that processes N
independent sessions per token, not N sessions serially.  A
2-process test on the same GPU dropped aggregate throughput from 18.8
→ 11.6 t/s, so the GPU does not naturally batch independent processes;
software batching is required.

### Phase 0 (chunked-prefill proxy)

Forcing `DS4_METAL_RESUME_PREFILL_MIN=1` makes the existing chunked
prefill encoder process `N` consecutive tokens of one session
together, sharing weight loads.  This is *not* the target workload (it
assumes a single growing context), but it bounds the achievable per-token
speedup:

| N | per-token (ms) | ×N=1 |
| ---: | ---: | ---: |
| 1 | 46.5 | 1.00 |
| 4 | 33   | 1.4 |
| 8 | 21   | 2.5 |
| 16 | 15  | 3.5 |
| 32 | 11  | 4.8 |

So the upper bound for ideal weight-shared decode is ~2.5× at N=8,
~4.8× at N=32.  Our target for *N independent sessions* sits below
this because attention can't share KV across sessions.

### Phases 1a-1c (what shipped)

Seven commits land on the branch:

| Commit | Phase | Effect |
| --- | --- | --- |
| `0f6a33d` | 1 PoC | `ds4_session_eval_batched_decode` API + `ds4-bench --batch N` + serial-fallback body. |
| `60664bc` | 1a | Submit-amortized serial — encode all N forwards back-to-back on the default stream, sync once, read each session's logits.  **+2-3 % aggregate at N=2-4** from eliminating N-1 `cudaDeviceSynchronize` + N-1 blocking D2H copies. |
| `1ea0307` | 1b | `__constant__ ds4_batch_step_args g_batch_args` + uploader (mirror of the `perf/cuda-graph-wip` design, per-row arrays).  No kernel reads it yet — pure scaffolding. |
| `635629d` | 1c-1 | `matmul_q4_0_preq_batch_warp8_kernel` + extern "C" `ds4_gpu_matmul_q4_0_batch_warp_tensor` (Q4 batched warp variant for the n_tok > 1 path). |
| `503c725` | 1c-2 | Split `metal_graph_encode_output_head` into prefix + vocab.  Add `metal_graph_encode_output_head_batched`: per-session prefix, stage N `output_norm` into engine-level scratch, one batched Q4 vocab matmul, scatter logits. |
| `e965c1b` | (notes) | `NOTES.md` capturing the B-4 dead-ends below. |
| `33208f6` | (notes) | `NOTES.md` host/GPU split + concrete shape-1 next-session checklist. |

Aggregate throughput on `ds4-bench --ctx-start 2048 --gen-tokens 50`:

| N  | t/s aggregate | × N=1 |
| ---: | ---: | ---: |
| 1 | 18.12 | 1.00 |
| 2 | 18.75 | 1.03 |
| 4 | 18.97 | 1.05 |
| 8 | 19.36 | 1.07 |
| 16 | 16.14 | 0.89 |

The +5-7 % at N ≤ 8 is real but small.  Most of it comes from implicit
L2 reuse inside the naive Q4 batched warp kernel (grid layout
`(out_dim/8, n_tok)` lets later toks for the same row hit L2 while
earlier toks are still in flight), plus the Phase 1a sync amortization.

### What did NOT help (the two B-4 dead-ends)

1. **Templated weight-shared Q4 matmul kernel.**  Collapse `grid.y` (= `n_tok`)
   into an inner unrolled tok loop inside one warp, so the row's nibbles
   are read once across N rows of activation instead of N times.

   | N | naive | templated |
   | ---: | ---: | ---: |
   | 2 | 18.26 | 18.28 |
   | 4 | 18.73 | 14.58 |
   | 8 | 18.79 | 11.47 |

   Regression at every N ≥ 4.  Hypothesis: dropping `grid.y` cuts the
   in-flight warp count by N, and Blackwell's warp scheduler does not
   pipeline the resulting longer-per-warp work as well as it did the
   thinner naive warps.  The implicit L2 reuse in the naive kernel was
   already paying most of the weight-amortization that the explicit
   kernel was supposed to deliver.
2. **cuBLAS f16 GEMM for the batched output head.**  Route through
   `ds4_gpu_matmul_q8_0_tensor` (which hits cuBLAS f16 at `n_tok > 1`)
   instead of the Q4 batched warp kernel.

   | N | naive Q4 | cuBLAS f16 |
   | ---: | ---: | ---: |
   | 2 | 18.75 | 17.86 |
   | 4 | 18.97 | 17.07 |
   | 8 | 19.36 | 13.38 |

   Regression everywhere.  f16 reads 2× the weight HBM that Q4 does; on
   a decode-shape GEMM (one or a few rows on the activation side) the
   workload is bandwidth-bound, and cuBLAS's FLOPS efficiency cannot
   recover that.

Both negative results are documented in the branch's `NOTES.md`.

### Where the wall-clock time actually goes

`DS4_METAL_GRAPH_TOKEN_PROFILE=1` on a steady single-session decode:

```
metal graph token pos=16 encode=19.3 ms execute=25.7 ms read=0.0 ms total=45.1 ms
metal graph token pos=17 encode=19.3 ms execute=25.8 ms read=0.0 ms total=45.1 ms
```

≈ 25 ms GPU compute + 19 ms host-side (the host side is inflated by the
`allow_split_flush=true` mid-token sync; the batched API path forces it
off, so the real host portion is smaller).

At N=4 the bench measures ≈ 52 ms GPU compute per generated token, i.e.
the batched path is **essentially fully serial on the GPU** for the
layer matmuls.  This is consistent with the observation that the only
thing currently amortizing across N is the batched output head — every
layer's Q/K/V projection, FFN gate/up/down, and attention still
re-runs per session.

The next real lever — and the next session's work — is to share weight
reads across sessions for the **per-layer dense matmuls**, where the
profile puts ~76 % of layer time.  The branch's `NOTES.md` carries a
concrete 5-step shape-1 checklist (engine-level batched scratch, split
point in `metal_graph_encode_decode_layer`, kernel-level fallbacks if
cuBLAS f16 regresses again).

## Lessons that generalise

A few patterns recur across the branches:

- **Implicit L2 reuse from `grid.y` is hard to beat.**  Both the
  templated weight-sharing kernel and the CUDA Graph experiment looked
  like obvious wins on paper, and both lost to the naive `grid.y =
  n_tok` baseline.  GB10's L2 + Blackwell's warp scheduler do enough
  implicit amortization that an explicit kernel restructure has to be
  *significantly* better than break-even to actually win.
- **Decode is bandwidth-bound; prefill is FLOPS-bound.**  Q4 lazy cache
  wins decode by 52 %.  The same Q4 path is barely faster than cuBLAS
  f16 for prefill because at `n_tok > 1` cuBLAS can amortize weights
  the FLOPS way.  Pick the right path per `n_tok`.
- **Numerical bisection beats reading the code.**  Every accuracy bug
  in Chapter 1 was found by `--metal-graph-test` + `dump_diff.py`, not
  by inspection.  The infrastructure stays useful long after the bugs
  it found are forgotten.
- **`__constant__` for per-step state is the right shape.**  Both the
  CUDA Graph experiment (`g_step_args`) and the batched-decode
  scaffold (`g_batch_args`) use this pattern.  It keeps the captured /
  batched launch structurally identical across tokens, which is what
  enables either CUDA-Graph replay or per-row dispatch in batched
  kernels.

## Pointers

- Bring-up / numerical bisection: `PORT_CUDA.md` on
  `cuda-gb10-backend`.
- Live perf branch: `perf/q4-only` (decode 18.8 t/s).
- Batched decode work-in-progress: `perf/batched-decode-poc`, with the
  full ledger in `NOTES.md` on that branch.
- Dormant reference: `perf/cuda-graph-wip` (CUDA Graph + `g_step_args`
  pattern).
