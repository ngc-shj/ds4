# CUDA Backend Port (target: NVIDIA GB10 / DGX Spark)

`ds4.c` started life as a Metal-only engine for DeepSeek V4 Flash on Apple
Silicon.  This document tracks the in-progress port of the GPU path to CUDA so
the same engine can run on NVIDIA Grace + Blackwell hardware (GB10, sm_121),
which has the unified-memory architecture that the Metal design implicitly
relies on.

## Current status

| Layer | Status |
| --- | --- |
| Linux ARM64 build of CPU reference path (`make`) | works |
| `make DS4_BACKEND=cuda` build (binaries link) | works |
| CUDA device init / stream / cudaMallocManaged tensors | works |
| `./ds4 --cuda-smoke-test` (6 kernel-level tests) | **6/6 PASS** |
| `ds4-server` build under `DS4_BACKEND=cuda` | works |
| Q2 GGUF mmap loading on GB10 (86.7 GB) | works |
| **End-to-end inference on a GGUF** | **WORKS** — prefill 1.92 t/s, decode 6.05 t/s on GB10 |
| Output text quality | **degraded** — numerical refinements still needed (see below) |

### Kernel status snapshot

| Kernel family | Functions | Status |
| --- | --- | --- |
| Tensor / lifecycle / model map | 12 | implemented |
| Elementwise (add, swiglu) | 2 | implemented |
| RMSNorm (plain, weight, rows, QKV-fused, head) | 6 | implemented |
| RoPE tail (YaRN) | 1 | implemented |
| KV store / FP8 round trip | 4 | implemented |
| F16 / F32 dense matmul (custom + cuBLAS for F32) | 3 | implemented |
| Q8_0 matmul + fused shared GLU + Q8_0 attn-output LoRA | 4 | implemented |
| FlashAttention (raw + 4 mixed variants) | 6 | implemented |
| Compressors (prefill / update / store / ratio4 / state) | 5 | implemented |
| Indexer (score / topk / mask) | 4 | implemented |
| Router (top-K softmax + hash mode, single + batch) | 2 | implemented |
| Routed MoE (IQ2_XXS gate/up + Q2_K down) | 2 | implemented |
| Hyper-connection (10 variants incl. sinkhorn, expand, fused Q8_0) | 10 | implemented |
| Embeddings (single / batched) | 2 | implemented |
| **Total** | **73** | **73 implemented, 0 stub** |

### Real-world driver test (Q2 model)

Running `./ds4 -p "Hello" --metal --ctx 1024 -n 5 --nothink` against the
86.7 GiB `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-...gguf` GGUF on a GB10 system now
**runs end to end**:

```
ds4: context buffers 111.32 MiB (ctx=1024, backend=cuda, ...)
ds4-cuda: using NVIDIA GB10 (sm_121, 121.6 GiB global mem)
ds4: Metal backend initialized for graph diagnostics
...generated text...
ds4: prefill: 1.92 t/s, generation: 6.05 t/s
```

All 73 ds4_metal_* entry points have a CUDA implementation.  The complete
pipeline (embedding → norm → Q/KV projection → KV store → RoPE → attention →
output projection → HC mixers → router → IQ2_XXS+Q2_K MoE → compressor →
indexer → mixed attention → output norm → unembedding) is functional.

### Numerical accuracy work (in progress)

Use `./ds4 -p "Hello" --metal --ctx 256 --metal-graph-test` to see
per-layer max-abs diff between the CPU reference and the active GPU
backend.  This is the primary tool for bisecting CUDA bugs against CPU.

#### Round 1 fixes (committed)

| Bug | Layer it broke | Diff before | Diff after |
| --- | --- | --- | --- |
| Router computed softmax instead of `sqrt(softplus(logit))` plus 1.5x | `router_w` weights | (selection wrong) | 0.955 |
| Attention sinks read as `__half`, but model stores F32 | many downstream | (sinks corrupt) | 0 |
| HC expand split read `pre[hc]` as `post`, dropped the combine matrix entirely | after_attn_hc | 4.69 | 0.003 |

After these three fixes, logits diff shrank from 40.6 to 4.5.

#### Round 2 fix (committed)

| Bug | Diff before | Diff after |
| --- | --- | --- |
| Hash-mode router used uniform 1/6 weights, but CPU computes `probs[selected]/sum * 1.5` | `router_w` 0.955 → 0.000288; `routed` 18.5 → 0.171; `logits` 4.52 → 0.18 | — |

DS4 V4 Flash early layers use hash routing, and my hash-mode path was
just stuffing uniform weights instead of running the `sqrt(softplus())`
probability through the same renormalisation as the top-K path.  Single
fix in `cuda/router.cuh`; everything else stays the same.

After Round 2, the pipeline produces multilingual semi-coherent text on
"Hello" (was BOS-repetition before any fixes).

#### Remaining diffs at layer 0 (CPU reference vs CUDA)

```
embed_hc      = 0
hc_pre        = 0
attn_norm     = 1.5e-08
q_rope        = 0.032
kv_rope       = 0.053
raw_cache     = 0.063
attn_out      = 0.068
after_attn_hc = 0.003
ffn_cur       = 0.003
ffn_norm      = 0.006
shared        = 0.004
router_w      = 0.000288       (Round 2)
routed        = 0.171          (Round 2)
ffn_out       = 0.171
after_ffn_hc  = 0.205
logits        = 0.18           (Round 2; was 4.52, was 40.64)
```

The remaining ~0.03-0.07 diffs in q_rope/kv_rope/raw_cache/attn_out are
upstream Q8_0 / F16 matmul precision compounding.  The CPU reference
quantises x to q8_K before the Q-LoRA matmul (block_q8_K input), while
the CUDA path keeps x as F32 (matching Metal).  These are different
algorithms and their results legitimately diverge by ~3-7%; bit-exact
agreement with CPU would require quantising x to q8_K in CUDA too.

### Performance (GB10, 86.7 GiB Q2 GGUF)

| Configuration | Prefill | Generation |
| --- | ---: | ---: |
| Initial (before mmap pinning) | 1.34 t/s | 3.51 t/s |
| After cudaHostRegister + cudaMemAdvise | **7.92 t/s** | **7.22 t/s** |
| After compressor_update full pipeline | 8.16 t/s | 7.28 t/s |

For comparison: Metal on M3 Max for the same Q2 model achieves ~25-30
t/s.  Remaining 3-4x gap is from kernel-launch overhead (~2500 launches
per token) and the naive Q8_0 / IQ2_XXS / Q2_K matmul kernels that don't
use shared memory or tensor cores.

The biggest single win was getting cudaHostRegister to accept the
GGUF mmap.  `cudaHostRegisterReadOnly` was declined on this kernel +
file-backed mapping combo; falling back to `cudaHostRegisterDefault`
worked, and adding `cudaMemAdvise(SetReadMostly, SetPreferredLocation)`
on top tells the unified-memory hardware to cache weight rows for
matmul reuse.

Further perf headroom (not yet exploited):
- Tiled / shared-memory GEMM in place of the naive block-per-row kernels.
- Fused gate+up MoE matmul + SwiGLU writing directly to the down input
  (avoids a `mid` round-trip through global memory).
- cuBLAS-LT for the F16 / Q8_0 dense projections, if it accepts the
  host-pinned pointers now.

### Empirical inference quality (after compressor + FP8 fixes)

Sampled and greedy decoding both produce coherent English / code now,
with the coherent run extending further after each numerical-accuracy
round:

| Round | --temp 0 coherent run length |
| --- | --- |
| Initial | 0 tokens (BOS/image repetition) |
| Round 1+2 (router, sinks, HC) | ~10 tokens |
| Compressor RMSNorm+RoPE fix | ~30 tokens (4-line license header) |
| FP8 per-chunk scaling fix | ~60 tokens (7-line license header — but actually license-header gibberish was a different bug) |
| Round 3 (HC expand grid) | "Hello" → normal assistant reply |
| Round 4 (compressor prefill RMSNorm/RoPE/FP8) | story prompts ~30 tokens |
| Round 5 (prefill_static_mixed comp visibility) | story prompts ~50 tokens |

Greedy `-p "Hello"` after Round 3+:
> _"Hello! How can I assist you today? If you have any questions or
> need help with something, feel free to ask."_

Greedy chess story after Round 5:
> _"Jasper was a sleek, gray tabby who spent his days napping in
> sunbeams and watching his owner, Arthur, with quiet, feline disdain.
> Arthur was a chess enthusiast"_ (then degrades to mojibake)

The output still degrades at the tail of long generations.  Likely
remaining sources: cumulative Q8_0 / F16 GEMM precision drift over 43
layers × ~30+ decode steps, and the still-unimplemented ratio=128
indexer compressor emit.

### Key insight: Metal also reads F32 x for Q8_0 matvec

Confirmed by reading `metal/dense.metal::kernel_mul_mv_q8_0_f32_impl`:

```c
device const float * y = (device const float *) (src1 + offset1);
...
for (i = 0; i < NQ; ++i) sumq += qs[i] * yl[i];  // int8 × F32
sumf[row] += sumq * ax[row][ib].d;
```

So **Metal does the same F32 dot as our CUDA**.  The CPU reference is the
outlier: it quantises x to q8_K before the GEMM, which introduces ~0.4%
error per element and produces the 0.03-0.07 diffs we see against CPU.

This means Metal's actual numerical behaviour should be very close to
our CUDA, and the inference output quality should also be similar.
The fact that ds4-cuda still produces somewhat degraded text suggests
there is an undiscovered structural bug, not just precision drift.

### Decode-at diagnostic (added)

`--metal-graph-decode-test [N]` prefills N tokens on both CPU and CUDA,
then runs a single decode step at `pos=N` on `prompt[N]` and reports
per-layer hidden-state diffs plus final logits diff.  This exercises
the decode-only kernels (`compressor_update`, `indexer_score_one`,
`decode_mixed_batch`) that the prefill-only graph test cannot reach.

Two env-var modes for finer bisection:

- `DS4_METAL_DECODE_TRACE_CACHE=1` — before the decode step, also report
  CPU-vs-GPU diffs of the post-prefill `raw_kv` / `attn_comp_kv` /
  `index_comp_kv` caches per layer.  Separates prefill drift from decode
  drift.
- `DS4_METAL_DECODE_TEACHER_FORCE=1` — at the start of each decode
  layer, overwrite the GPU `cur_hc` with the CPU's, so each layer's
  reported diff measures only that single layer's compute drift, not
  the accumulated drift from earlier layers.
- `DS4_METAL_DECODE_TRACE_HC_STAGES=1` — at each layer print per-HC-stage
  CPU vs CUDA diffs for `attn_cur` (HC attn pre out), `attn_norm` (RMS
  norm of attn_cur), `after_attn_hc` (HC attn post out), `ffn_cur`
  (HC ffn pre out), `ffn_norm`.  CPU intermediates come straight from
  `ds4_cpu_decode_scratch` so no recomputation is needed.

  **Finding**: with teacher-force the `attn_cur` and `attn_norm` diffs
  drop to **1e-9..1e-7 at every layer**, i.e. the fused
  `hc_split_weighted_sum_norm_hc4` kernel is bit-exact against CPU's
  `hc_pre_from_state_one_scratch` + `layer_attn_norm_one`.  The HC
  kernels themselves are not a drift source; remaining per-layer drift
  comes from Q/K/V projections, attention, output projection, and FFN
  paths (Q8_0 / F16 matmul precision), and only flows through HC as
  inherited residual_hc.

#### First-run findings (N=16, 19-token prompt)

The diagnostic immediately bisected the dominant drift source:

```
prefill-cache layer  0 raw_kv  max=0.125   rms=0.007    (essentially correct)
prefill-cache layer  1 raw_kv  max=11.539  rms=0.750    (BLAST)
prefill-cache layer  2 raw_kv  max=3.822   rms=0.535
prefill-cache layer  3 raw_kv  max=7.129   rms=0.646
...                    raw_kv  max~5-9 steady state for all subsequent layers
```

Every subsequent layer's post-prefill KV cache differs from CPU by a
similar 4-9 max-abs.  The **single** large step is between layer 0 and
layer 1; once corrupted, the streams stay corrupted.

This is an actionable lead because:

1. `metal_graph_first_token_full_test` (single-token decode at pos=0)
   already shows per-layer hc match between CPU and CUDA across all
   43 layers — so per-token decode kernels at layer 0→1 are correct.
2. `metal_graph_decode_test` (Round 1+2) reports layer-0 prefill diff of
   logits=0.18 — layer 0 prefill itself is correct.
3. The new test runs **batched** prefill of N=16 tokens, and exposes
   that the layer 0 → layer 1 transition diverges *only in batch mode*.

The likely culprit is a CUDA **batched** prefill kernel that does not
match its per-token CPU equivalent: HC post weighted-sum, batched
SwiGLU, batched HC mix, or one of the HC-stream split/expand kernels
called from `metal_graph_encode_layer_batch` at layer 0.

With teacher forcing (each decode-layer input forced to CPU), top-1
greedy token matches (cpu=9035 = gpu=9035) and logits diff drops from
15.4 to 5.7.  Without teacher forcing the top-1 mismatches.  Layers
28, 36, 42 also stand out as having larger per-layer compute drift
(11.7, 13.9, 25.0) even with teacher-forced inputs — these are all
ratio=4 layers (compressed + indexer), suggesting a secondary bug in
the `decode_mixed_batch` / indexer path that compounds the prefill
divergence.

#### Round 3 fix (committed)

The decode-at diagnostic with per-row trace
(`DS4_METAL_DECODE_TRACE_CACHE_PER_ROW=1`) immediately localised the
bug to a 4-token boundary:

```
layer 1 raw_kv  row 0..3  max=0.125          (clean F16 noise)
layer 1 raw_kv  row 4..7  max=10-11          (BLAST)
```

The first four token rows pass through layer 0 correctly; rows 4+ do
not.  Searching for hard-coded "4" in the batched prefill HC kernels
found:

```c
int ds4_metal_hc_expand_add_split_tensor(...) {
    dim3 grid(ds4_cuda_ceil_div((int)ne, BLK), nh);   // nh = DS4_N_HC = 4
    ds4_cuda_kernel_hc_expand_add_split_f32<<<grid, ...>>>(...);
}
```

The kernel uses `blockIdx.y` as the **token** index (`t`), but the
dispatcher set `grid.y = nh = 4`.  For batched prefill with
`n_tokens > 4`, only the first 4 token rows of post-FFN HC mixing ran;
rows 4+ kept stale buffer contents, silently corrupting all
subsequent layers' inputs.

`ds4_metal_hc_expand_split_tensor` had the same bug.  Fix: derive
`n_tok` from output buffer size (mirroring `ds4_metal_hc_expand_tensor`
which already did this) and use it as `grid.y`.

After the fix, layer 1 raw_kv max drops from **11.539 → 0.125** and
all layers' raw_kv caches match CPU within F16 quantization noise.
Greedy `-p "Hello"` now produces:

> _"Hello! How can I assist you today? If you have any questions or
> need help with something, feel free to ask."_

(was: garbage license-header-style mojibake within ~60 tokens).
Tail degradation in long generations still occurs — the indexer /
compressor caches at ratio=4 layers still differ by ~3-5 max-abs from
CPU, which is the next bug to attack.

#### Round 4 fix (committed)

`ds4_metal_compressor_prefill_tensor` was running ONLY the
softmax-pool kernel and silently ignored its `no` (norm offset),
`nr` (n_rotation), `qfp8`, RoPE, and `eps` parameters via `(void)`
casts.  The decode-time `compressor_update_tensor` correctly chains
**pool → RMSNorm(weight) → RoPE → FP8(if attn-class)**; the prefill
path was missing the last three steps.

CPU reference (`compressor_decode_one` in ds4.c):
```c
compressor_pool_decode_state(pooled, ...);
// RMSNorm(weight=norm, eps=DS4_RMS_EPS)
// RoPE at comp_pos = pos + 1 - ratio
if (head_dim == DS4_N_HEAD_DIM) dsv4_fp8_kv_quantize_row(out_comp, ...);
```

Without the finishing pipeline, every prefilled comp row entered the
attention path as a raw pre-norm, pre-rotated value, biasing each
ratio=4 layer's mixed attention.

After the fix, layer 2 `attn_comp` max diff drops from **3.857 →
0.215** (94%) and `idx_comp` from **4.958 → 0.042** (99%).  Per-layer
decode hc diff also drops broadly: layer 1 0.051→0.006, layer 30
15.66→4.86, final logits 15.4→7.5.  `Hello` still produces a normal
assistant reply; longer story prompts get a longer coherent prefix
(~30 tokens) but eventually still degrade.

#### Round 5 fix (committed)

`ds4_cuda_kernel_attn_prefill_static_mixed_f32` was using the wrong
formula to gate compressed-row visibility per token:

```c
const uint32_t comp_visible = (jraw_lo >= (int32_t)ratio)
    ? (uint32_t)jraw_lo / ratio : 0u;
```

This zeroed `comp_visible` whenever the SWA window still reached
position 0 — i.e. essentially always for prefill batches shorter than
`DS4_N_SWA = 128`.  Tokens 4+ in our N=16 test should have been
attending to compressor rows 1..n, but the kernel showed them none.

Metal's `ds4_metal_fill_static_mixed_prefill_mask` and CPU's
`layer_attention_mixed_one` both use:

```c
const uint32_t n_visible = (q + 1u) / ratio;
```

Token q sees comp rows [0, (q+1)/ratio) **in addition to** the raw
SWA window — both are visible even where their position ranges
overlap.  Fix: replace the CUDA formula with `(t + 1u) / ratio`.

Bisection: this bug created the row 0..3 vs row 4+ pattern in layer 3
raw_kv (and every subsequent ratio=4 / ratio=128 layer's KV).  The
4-token boundary corresponded exactly to the moment compressor row 0
becomes visible to attention (token 3 in CPU/Metal but never in
CUDA's old formula).

After the fix:
- Layer 3 raw_kv per-row: all 16 rows max=0.125 (clean F16 noise)
- decode-at worst hc_max: **24.46 → 8.99** (~63% reduction)
- decode-at logits_max: **7.5 → 3.8** (~50% reduction)
- Greedy story prompt coherent prefix: ~30 tokens → ~50 tokens

### Metal vs CUDA tensor-dump comparison (Method 2)

The decode-at test compares CUDA against the CPU reference, but CPU
runs different algorithms in places (q8_K activation quantize for
Q-LoRA, two-pass softmax pool, etc.) so its diffs are not a clean
ground truth for "is CUDA matching the production Metal path".  The
project ships a per-tensor binary dump hook that works on **both**
backends — set `DS4_METAL_GRAPH_DUMP_PREFIX=<prefix>` and any of
`DS4_METAL_GRAPH_DUMP_NAME=<substring>`,
`DS4_METAL_GRAPH_DUMP_LAYER=<n|all>`,
`DS4_METAL_GRAPH_DUMP_POS=<pos>`.

54 dump points cover the full pipeline: HC pre/post mixers, attn
norm/Q/KV at every stage, attention output, MoE logits/probs/topk and
each per-expert intermediate, FFN norm, compressor cache and rolling
state, indexer scores and topk, and the output head.  Files are raw
F32 (`.bin`) or int32 (`.i32`) — no header.

`tools/dump_diff.py` pairs files between two prefixes by their
`<name>-<layer>_pos<pos>` suffix and reports per-tensor max-abs,
RMS, relative-to-peak, and the top mismatched indices.  Filters
support `name=<substr>`, `layer=<n|a..b>`, `pos=<n|a..b>`.

Workflow once a Mac is available for a Metal capture:

```sh
# On Mac (Metal):
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/metal_run \
DS4_METAL_GRAPH_DUMP_LAYER=all DS4_METAL_GRAPH_DUMP_POS=15 \
  ./ds4 -m model.gguf --metal -p "Hello" --metal-graph-decode-test 16

# Sync /tmp/metal_run_*.bin to the CUDA box.

# On Linux (CUDA, identical command):
DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/cuda_run \
DS4_METAL_GRAPH_DUMP_LAYER=all DS4_METAL_GRAPH_DUMP_POS=15 \
  ./ds4 -m model.gguf --metal -p "Hello" --metal-graph-decode-test 16

tools/dump_diff.py /tmp/metal_run /tmp/cuda_run
```

The same command pair, run with both prefixes pointed at the CUDA
build, validates the tool itself: every file should diff to exactly
zero (CUDA self-determinism check).

### Next investigations (priority order)

1. **Run Method 2 against a Metal capture.**  This is the only way
   to verify whether the remaining drift after Rounds 3-5 is real
   algorithmic divergence or precision noise inherited from CPU's
   different quantization path.  Until we have a Metal-side dump,
   chasing further per-layer diffs blind is unproductive.
2. **Remaining accumulated drift.**  After Round 3+4+5 fixes,
   decode-time per-layer hc diff still grows from ~0.006 at layer 1
   to ~9 at layer 42.  The growth is now smoother and broadly
   distributed.  Suspects: precision drift across 43 layers of
   Q8_0/F16 matmul, the `compressor_state_init_ratio4` path that
   leaves state[4..7]=0/-INF where CPU per-token leaves them as a
   duplicate of state[0..3].
2. **Try cuBLAS-LT for F16 GEMM**.  CUDA 13.0 may have lifted the
   `CUBLAS_STATUS_NOT_SUPPORTED` we saw for host-mmap pointers + small-m
   shapes.  cublasLtMatmul has a wider supported config space.
3. **Audit decode-time numeric paths**.  The metal-graph-test only
   covers layer-0 prefill.  decode_mixed_batch / decode_heads /
   compressor_update are exercised only during text generation, and
   subtle bugs there would only show in actual inference.
4. **FP8 KV quantization (deferred)**.  A Round 2 attempt at adding
   per-64-element power-of-2 scaling (matching Metal's
   `kernel_dsv4_fp8_kv_quantize_f32`) made the metal-graph-test diff
   bigger and inference output worse.  The CPU reference appears to
   skip FP8 quantization, so faithfully reproducing Metal's FP8 drifts
   us further from CPU.  Need a Metal-vs-CUDA per-layer dump (item 1)
   before retrying this, because the CPU diff metric is misleading here.

### cuBLAS GemmEx note

cuBLAS GemmEx F16 with host-mmap pointers + small-m shapes returns
`CUBLAS_STATUS_NOT_SUPPORTED` on the GB10 driver stack we tested with.  We
fell back to a custom block-per-row CUDA kernel for the F16 path; the F32
path stays on cuBLAS GemmEx (TF32 tensor ops) because that combination
accepts the GGUF pointer.  Tensor-core / cuBLASLt replacement is filed
under tier 1 follow-up.

## Repo layout for the CUDA backend

```
ds4_metal.h           -- shared API surface (Metal AND CUDA backends).
                         Wrapped in `extern "C"` so nvcc-compiled units can
                         #include it.
ds4_cuda.cu           -- CUDA implementation of the 73 ds4_metal_* entry points
                         (mirror of ds4_metal.m on Apple).
cuda/common.cuh       -- shared device helpers, stream/global state, GGUF
                         quant-tag enum, error-check macros.
cuda/elementwise.cuh  -- add, swiglu, plain rms_norm (warp-reduced).
cuda/norm.cuh         -- RMSNorm with learned weight (rows + fused QKV +
                         3D head-wise).
cuda/rope.cuh         -- DS4 partial RoPE with YaRN scaling.
cuda/kv.cuh           -- F16 KV round trip + FP8 e4m3 KV quantise.
cuda/gemm.cuh         -- cuBLAS handle setup, F16/F32 matmul wrappers.
cuda/q8_0.cuh         -- Q8_0 dequant + matmul + fused gate-up SwiGLU.
Makefile              -- DS4_BACKEND=cpu (default) | DS4_BACKEND=cuda on Linux;
                         Darwin always builds the Metal backend.
```

`ds4.c` is **unchanged at the call sites**: the same `ds4_metal_*` symbols are
used regardless of backend.  Only `ds4_backend_name()` was taught to report
`"cuda"` instead of `"metal"` when `DS4_CUDA` is defined.

## Architectural design choices

1. **Unified memory via `cudaMallocManaged`.**  Metal's
   `MTLResourceStorageModeShared` lets host and GPU share one buffer.  On GB10
   the equivalent is managed memory: the same virtual address is reachable
   from both sides over NVLink-C2C with zero copies.  This is why
   `ds4_metal_tensor_contents()` can keep returning a host-usable pointer.
   On discrete (non-Grace) NVIDIA boxes the same code works but with
   demand-paging; performance is worse but correctness is preserved.
2. **GGUF mmap pinning.**  `ds4_metal_set_model_map()` calls
   `cudaHostRegister(.., cudaHostRegisterReadOnly)`.  The driver may decline
   (older kernels, /proc-backed maps); we log and continue, leaning on UVA so
   the device can still dereference the pointer, just less efficiently.
3. **Stream-based command lifecycle.**  Metal's `begin/flush/end` map to
   "open a logical batch / leave kernels in-flight / synchronise the stream".
   A future optimisation is to capture into a `cudaGraph` for prefill steps
   that repeat identical layer sequences.
4. **Backend enum stays as `DS4_BACKEND_METAL`.**  Splitting the enum would
   force changes across the engine, server, CLI, tests, and disk KV cache
   header.  Instead the CUDA backend reuses the Metal slot, and the public
   `ds4_backend_name()` reports the actual hardware.  The `--cuda` CLI flag is
   an alias of `--metal`.

## Kernels left to port

This is the hot-path list.  Every entry is a one-shot `fprintf` stub in
`ds4_cuda.cu` today; the engine will report the missing kernel cleanly the
first time it tries to use it.  Items are roughly in dependency order: each
group makes the layer above it possible.

### Tier 0 - DONE

- `ds4_metal_add_tensor`
- `ds4_metal_swiglu_tensor`
- `ds4_metal_rms_norm_plain_tensor`
- `ds4_metal_rms_norm_plain_rows_tensor`

### Tier 1 - dense and quantized matmul (PARTIALLY DONE)

Implemented:
- `ds4_metal_matmul_f16_tensor`, `_pair_tensor` (cuBLAS GemmEx, F16 weights × F32 acts)
- `ds4_metal_matmul_f32_tensor` (cuBLAS GemmEx, TF32 tensor cores enabled)
- `ds4_metal_matmul_q8_0_tensor` (custom dequant+MAC, one block per (row, token))
- `ds4_metal_shared_gate_up_swiglu_q8_0_tensor` (fused Q8_0 gate+up dot + SwiGLU)

Remaining:
- `ds4_metal_matmul_q8_0_hc_expand_tensor` (fuse Q8_0 GEMM with HC expand)
- `ds4_metal_shared_down_hc_expand_q8_0_tensor` (Q8_0 down + HC expand fused)

Note on perf: the Q8_0 kernel here is the simple "block-per-row + warp
reduce" version.  For decode with out_dim ~7168 that schedules 7168 blocks
of 128 threads, which fills the GPU but doesn't yet exploit tensor cores.
A second-pass optimization should switch to a CUTLASS-style tiled GEMM that
dequantises into shared memory and accumulates with mma.sync.

### Tier 2 - RMSNorm with learned weight + QKV fused norm (DONE)

- `ds4_metal_rms_norm_weight_tensor`, `_rows_tensor` (one block per row, warp-reduced)
- `ds4_metal_dsv4_qkv_rms_norm_rows_tensor` (fused Q+KV pass with `grid.y` task split)
- `ds4_metal_head_rms_norm_tensor` (3D [n_tok, n_head, head_dim] head-wise norm)

### Tier 3 - RoPE + KV state (DONE)

- `ds4_metal_rope_tail_tensor` (partial RoPE with YaRN frequency correction)
- `ds4_metal_kv_fp8_store_raw_tensor` (fused FP8/F16 round trip + raw cache write)
- `ds4_metal_store_raw_kv_tensor`, `_batch_tensor` (F16 round trip)
- `ds4_metal_dsv4_fp8_kv_quantize_tensor` (in-place FP8 round trip across nope, F16 across tail)

FP8 uses `__nv_fp8_e4m3` which matches DS4's e4m3 layout on Hopper/Blackwell.

### Tier 4 - FlashAttention variants

This is the biggest single item.  Metal has six attention variants in
`metal/flash_attn.metal` plus `metal/dsv4_hc.metal` (1426 + 861 lines):

- decode raw / mixed / indexed
- prefill raw / static mixed / masked mixed

Options, roughly cheap-to-expensive:
1. **CUTLASS-based custom kernel** mirroring the Metal layout.  Best perf,
   most engineering.
2. **FlashAttention-2 / -3 from open source.**  Existing implementations have
   forward kernels for many variants but not the DS4 raw/compressed/indexed
   mix.  Useful as starting point for the raw window-only decode path.
3. **Naive split-K attention.**  Slow but a usable correctness baseline.  A
   single kernel of `Q @ K^T -> softmax -> @ V` with online softmax would
   unblock prefill correctness checks while the optimised version is built.

Recommendation: start with option 3 to get end-to-end inference correctness
first (so the rest of the engine can be validated), then replace it.

### Tier 5 - Compressors and indexer (the only remaining end-to-end blocker)

DS4's defining feature: ratio-4 compressed KV state and a small "indexer" that
picks visible compressed rows.  These power four FlashAttention variants
(prefill_static_mixed, prefill_masked_mixed, decode_mixed_batch,
attention_indexed_mixed_batch) which all read from compressed KV in addition
to the raw window.

Compressor functions (stubs):
- `ds4_metal_compressor_update_tensor` — single-token rolling update
- `ds4_metal_compressor_store_batch_tensor` — batched roll
- `ds4_metal_compressor_prefill_tensor` — initial fill from prefill chunk
- `ds4_metal_compressor_prefill_ratio4_replay_tensor` — replay for KV resume
- `ds4_metal_compressor_prefill_state_ratio4_tensor` — state-only path

Indexer functions (stubs):
- `ds4_metal_indexer_score_one_tensor` — per-comp-row score, single token
- `ds4_metal_indexer_scores_prefill_tensor`, `_decode_batch_tensor`
- `ds4_metal_indexer_topk_tensor` — pick top-K compressed rows
- `ds4_metal_dsv4_topk_mask_tensor` — boolean mask from top-K indices

Reference Metal kernels:
- `metal/dsv4_hc.metal::kernel_dsv4_softmax_pool` (the core ratio-pool op)
- `metal/dsv4_misc.metal::kernel_dsv4_compressor_store_one`
- `metal/dsv4_misc.metal::kernel_dsv4_indexer_*`
- `metal/argsort.metal` for the top-K sort
- CUB's block-wise radix select is a drop-in replacement for the top-K once
  the indexer scores are computed.

The math is:
1. **Compress**: every `ratio` (=4) raw rows are pooled into one compressed
   row using softmax weights derived from per-row scores plus an absolute
   position embedding (APE).  APE comes from the model map.
2. **Index**: per query, score each compressed row against an indexer Q
   projection and keep only the top-K most relevant rows.
3. **Attention**: read raw window + selected compressed rows together.

Once those six families are implemented, the Q2 / Q4 inference path runs end
to end with no remaining stubs.

### Tier 6 - MoE routing and routed experts (DONE)

Implemented:
- `ds4_metal_router_select_tensor`, `_batch_tensor` (softmax + top-K with
  optional bias / hash-mode lookup).
- `ds4_metal_routed_moe_one_tensor`, `_batch_tensor` (IQ2_XXS gate+up and
  Q2_K down dequant, sequential over the top-K selected experts).

The implementation in `cuda/moe.cuh` carries the two GGML lookup tables
(`iq2xxs_grid`, `ksigns_iq2xs`) in `__constant__` memory and emits one block
per (output_row, token) just like the Q8_0 matmul.  Per-expert scheduling is
naive: for each of the K=6 selected experts it issues gate, up, swiglu+route,
down, accumulate.  Three obvious optimisations are filed for later:
1. Fuse gate+up into one kernel that shares the per-block dequant.
2. Use a grouped GEMM that runs all K experts in one launch.
3. Add tensor-core paths (warp-MMA) once IQ2_XXS / Q2_K are vectorised
   enough to feed MMA tiles.

### Tier 7 - Hyper-connection mixers

- `ds4_metal_hc_split_sinkhorn_tensor`
- `ds4_metal_hc_weighted_sum_tensor`, `_split_tensor`
- `ds4_metal_hc_split_weighted_sum_tensor`, `_norm_tensor`
- `ds4_metal_output_hc_weights_tensor`
- `ds4_metal_hc_expand_tensor`, `_split_tensor`, `_add_split_tensor`

Small fused reductions across DS4's four residual streams.  Each is short
(~30 lines of Metal) and dominated by global-memory IO, so the CUDA port is
trivial; only the Sinkhorn iteration needs a real reduction loop.

### Tier 8 - Embeddings

- `ds4_metal_embed_token_hc_tensor`
- `ds4_metal_embed_tokens_hc_tensor`

Single-token and batched token embedding lookup with HC seeding.  These
launch infrequently and are simple gather operations.

## Validation checklist

Once each tier is in place, run the project's own gate:

```
make DS4_BACKEND=cuda
./ds4-server --trace /tmp/ds4-trace.txt -m gguf/<file>.gguf
./ds4 --dump-logprobs /tmp/cuda.json --temp 0 -p "..."
./ds4_test --logprob-vectors          # token-by-token comparison vs
                                      # tests/test-vectors captured from the
                                      # official API.
```

The logprob vectors will catch silent numerical drift well before generation
quality degrades visibly.

## Performance notes for GB10 specifically

- GB10 is sm_121.  `NVCCFLAGS` already targets `compute_121`/`sm_121` plus
  `sm_120` and `sm_90` as fallbacks.
- 128 GiB unified LPDDR5X is shared across CPU and GPU.  Holding the GGUF
  mmapped and `cudaHostRegister`-pinned is genuinely zero-copy.
- The Blackwell SM has 192 KiB of shared memory configurable per block.
  Plan attention tile sizes around 64 KiB shared per block to stay well within
  the per-SM budget (≥3 blocks resident).
- FP8 tensor cores are native; consider `__nv_fp8_e4m3` for KV store as on
  Apple Silicon, and explore FP8 GEMM via cuBLASLt for the Q-LoRA paths.

## Out of scope for the port

- MTP speculative path: defer until tier 4 (attention) is stable.
- Disk KV cache format: unchanged.  The on-disk payload is portable between
  backends because it stores raw tensor bytes that share the same
  little-endian F32/F16/FP8 layouts on both platforms.
- Server endpoints: no changes needed.  `ds4-server` already builds against
  the CUDA backend and exposes the same HTTP API; it just won't serve real
  inference until tiers 1-4 are implemented.
