// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint tg_ [[threadgroup_position_in_grid]], uint tid_ [[thread_position_in_threadgroup]], uint ntg_ [[threads_per_threadgroup]]) {
    const uint gid = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

/* RoPE at a position a node carries, not one the row index implies.
 *
 * kernel_dsv41_rope takes start + row * stride, which is every position a chain
 * can be at and no position a tree needs: siblings share a depth and a branch
 * skips none, so the position has to come from the node's own descriptor. The
 * arithmetic is the same to the bit - the same frequency a lane, precise cos and
 * sin, the same bf16 rounding on store - so a chain fed positions start + row
 * produces what the other kernel does. */
kernel void kernel_dsv41_rope_nodes(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        device const uint *positions,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(positions[group.y]) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong i = ((ulong)group.y * args.heads + group.x) * args.width +
                    args.width - 64u + 2u * lane;
    const float re = x[i], im = x[i + 1u];
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

/* ---- the verify executor's node-addressed forms --------------------------
 * Each of these is an existing kernel's arithmetic with one change: where a
 * row is read from. A candidate tree's rows are not consecutive positions, so
 * "the row before" becomes "the parent", and "compressed row c" becomes either
 * the committed cache or the node on the reader's own path that produced it. */

/* pool2 over explicit pairs: a is the even partner's row (-1 for the committed
 * carry), b the odd node, which is also the output row. The expression is
 * kernel_dsv41_pool2's, term for term. */
struct ds4_metal_args_dsv41_pool_pairs {
    uint width;
    uint pairs;     /* pairs in this dispatch */
    uint first;     /* index of the first of them in the pair lists */
    uint row0;      /* node id of this dispatch's row 0 */
};

kernel void kernel_dsv41_pool2_pairs(
        constant ds4_metal_args_dsv41_pool_pairs &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        device const int *pair_a,
        device const int *pair_b,
        device const float *carry_kv,
        device const float *carry_scores,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (index.x >= args.width || index.y >= args.pairs) return;
    /* a names a node, and its projection is read from the node-indexed carry
     * rows rather than from this dispatch's rows: the partner may have run in
     * another pass of the same block. */
    const uint k = args.first + index.y;
    const long a = pair_a[k];
    const uint node = (uint)pair_b[k];
    /* The odd node's own projection is a row of this dispatch; its output row is
     * its node row of the full buffer. */
    const ulong b = (ulong)(node - args.row0) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : carry_kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : carry_scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)node * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

/* Rows gathered by a permutation: dst[r] = src[perm[r]]. The source is a copy
 * of the rows taken before, so a permutation in place needs no ordering. */
kernel void kernel_dsv41_permute_rows(
        constant uint &row_floats,
        device const float *src,
        device const uint *perm,
        device float *dst,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (index.x >= row_floats) return;
    dst[(ulong)index.y * row_floats + index.x] =
        src[(ulong)perm[index.y] * row_floats + index.x];
}

/* Where compressed row c of a node's view lives: below `committed` it is the
 * shared cache; above it, the node on the reader's path at depth
 * (c + 1) * ratio - 1 - base produced it, into its own slot of the delta. */
struct ds4_metal_args_dsv41_comp_map {
    uint base;
    uint ratio;
    uint committed;
    uint rows;          /* selected rows (gather) or visible rows (score) */
    uint path_offset;
    uint width;
    float scale;
};

static inline ulong dsv41_comp_row(constant ds4_metal_args_dsv41_comp_map &m,
                                   device const uint *path, uint c, thread bool &delta) {
    if (c < m.committed) { delta = false; return c; }
    delta = true;
    return path[m.path_offset + (c + 1u) * m.ratio - 1u - m.base];
}

/* ds4_gpu_dsv41_gather_kv, one node: a row copy either way. */
kernel void kernel_dsv41_gather_comp_dag(
        constant ds4_metal_args_dsv41_comp_map &args,
        device const int *ids,
        device const float *cache,
        device const float *delta,
        device const uint *path,
        device float *out,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (index.x >= args.width || index.y >= args.rows) return;
    bool from_delta;
    const ulong r = dsv41_comp_row(args, path, (uint)ids[index.y], from_delta);
    out[(ulong)index.y * args.width + index.x] =
        (from_delta ? delta : cache)[r * args.width + index.x];
}

/* kernel_glm_indexer_score_one_direct for one node, keys through the map. The
 * body is that kernel's line for line: 32 heads of 128, four simdgroups, the
 * same load, the same dot, the same simd_sum and the same accumulation order. */
kernel void kernel_dsv41_indexer_score_dag(
        constant ds4_metal_args_dsv41_comp_map &args,
        device const float *q,
        device const float *weights,
        device const float *cache,
        device const float *delta,
        device const uint *path,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.rows) return;
    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;
    bool from_delta;
    const ulong kr = dsv41_comp_row(args, path, row, from_delta);
    device const float *keys = from_delta ? delta : cache;
    if (tid < 128u) ktg[tid] = keys[kr * 128u + tid];
    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q + (ulong)head * 128u);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;
        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) psum[sg] = max(s * args.scale, 0.0f) * weights[head];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scores[row] = acc;
}

/* The chosen path's compressed and index rows into the shared caches, at the
 * absolute rows a session that ran only that path would have written. The
 * depth and the path come from the tree accept's own output. */
struct ds4_metal_args_dsv41_path_commit {
    uint base;
    uint ratio;
};

kernel void kernel_dsv41_path_commit_comp(
        constant ds4_metal_args_dsv41_path_commit &args,
        device const uint *tree,
        device const float *comp_delta,
        device const float *index_delta,
        device float *compressed,
        device float *index_cache,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    const uint i = index.y, x = index.x;
    if (i >= tree[0] || x >= 512u) return;
    const uint p = args.base + i;
    if ((p + 1u) % args.ratio != 0u) return;
    const ulong c = (p + 1u) / args.ratio - 1u;
    const ulong node = tree[3u + i];
    compressed[c * 512u + x] = comp_delta[node * 512u + x];
    if (x < 128u) index_cache[c * 128u + x] = index_delta[node * 128u + x];
}

/* The pooling carry a session that ran only the chosen path would hold: the
 * projection at the last even position of that path, or the committed one when
 * the path has no even position of its own. */
kernel void kernel_dsv41_path_commit_carry(
        constant ds4_metal_args_dsv41_path_commit &args,
        device const uint *tree,
        device const float *carry_kv,
        device const float *carry_score,
        device float *previous_kv,
        device float *previous_score,
        uint tg_ [[threadgroup_position_in_grid]], uint tid_ [[thread_position_in_threadgroup]], uint ntg_ [[threads_per_threadgroup]]) {
    const uint x = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (x >= 512u) return;
    const uint d = tree[0];
    if (d == 0u) return;
    const uint last = args.base + d - 1u;
    long src = -1;
    if ((last & 1u) == 0u) src = tree[3u + d - 1u];
    else if (d >= 2u) src = tree[3u + d - 2u];
    if (src < 0) return;
    previous_kv[x] = carry_kv[(ulong)src * 512u + x];
    previous_score[x] = carry_score[(ulong)src * 512u + x];
}

/* One node's logits row into the session's own logits, chosen on the device. */
kernel void kernel_dsv41_path_logits(
        constant uint &row_floats,
        device const uint *tree,
        device const float *logits,
        device float *out,
        uint tg_ [[threadgroup_position_in_grid]], uint tid_ [[thread_position_in_threadgroup]], uint ntg_ [[threads_per_threadgroup]]) {
    const uint x = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (x >= row_floats) return;
    out[x] = logits[(ulong)tree[1] * row_floats + x];
}

/* The accept decision, on the GPU.
 *
 * A verifier does not need the vocabulary on the host. The performance path
 * wants the accepted length and the one token that replaces the first wrong
 * candidate, which is tens of bytes; reading K x 129280 floats back and taking
 * the argmax there costs 16.5 MiB and a queue drain at K = 32, and the full
 * logits are only needed by the correctness pass that compares them.
 *
 * The tie-break is sample_argmax's: a strict greater-than keeps the earliest
 * index, so ties resolve to the lowest one. NaN is skipped here and cannot be
 * selected there either, because every comparison against it is false. */
struct ds4_metal_args_dsv41_block_argmax {
    uint32_t n_vocab;
    uint32_t rows;
    uint32_t row_floats;   /* stride between rows, in floats */
};

/* Two passes, because one threadgroup a row leaves the machine idle.
 *
 * The first version put 256 threads on each of K rows - eight threadgroups for a
 * million floats - and measured 5.78 ms a token worse than reading the logits
 * back and taking the argmax on the host, which costs 1.5 ms for all K rows. The
 * bytes were never the problem on unified memory; the parallelism was. So the
 * grid is chunk x row here, 32 chunks a row at 4096 values, and a second pass
 * folds the partials. */
kernel void kernel_dsv41_block_argmax(
        constant ds4_metal_args_dsv41_block_argmax &args,
        device const float *logits,
        device uint2 *partials,          /* [rows][chunks] as (bits, index) */
        uint2  gid  [[threadgroup_position_in_grid]],
        ushort tid  [[thread_index_in_threadgroup]]) {
    /* The dispatch below uses 256; taking it from threads_per_threadgroup would
     * have to match the grid position's dimensionality, which is not worth an
     * attribute here. */
    const ushort nthr = 256;
    const uint row = gid.y;
    if (row >= args.rows) return;
    const uint chunks = (args.n_vocab + 4095u) / 4096u;
    if (gid.x >= chunks) return;
    device const float *y = logits + (ulong)row * args.row_floats;
    const uint begin = gid.x * 4096u;
    const uint end = min(begin + 4096u, args.n_vocab);
    float best = -INFINITY;
    uint index = 0xffffffffu;
    for (uint i = begin + tid; i < end; i += nthr) {
        const float v = y[i];
        if (v != v) continue;
        if (index == 0xffffffffu || v > best) { best = v; index = i; }
    }
    const float top = simd_max(best);
    const uint winner = simd_min(best == top ? index : 0xffffffffu);
    threadgroup float scores[8];    /* 256 threads is eight simdgroups */
    threadgroup uint indices[8];
    const ushort lane = tid % 32u;
    const ushort sg = tid / 32u;
    if (lane == 0) { scores[sg] = top; indices[sg] = winner; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const ushort groups = (nthr + 31u) / 32u;
        float b = scores[0];
        uint ix = indices[0];
        for (ushort g = 1; g < groups; g++)
            if (indices[g] != 0xffffffffu &&
                (ix == 0xffffffffu || scores[g] > b || (scores[g] == b && indices[g] < ix))) {
                b = scores[g]; ix = indices[g];
            }
        partials[(ulong)row * chunks + gid.x] = uint2(as_type<uint>(b), ix);
    }
}

/* Fold one row's partials, keeping the earliest index on a tie so the answer is
 * sample_argmax's. One threadgroup a row, 32 partials at K = 32 vocabularies. */
kernel void kernel_dsv41_block_argmax_fold(
        constant ds4_metal_args_dsv41_block_argmax &args,
        device const uint2 *partials,
        device uint32_t *out_index,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.rows || tid != 0) return;
    const uint chunks = (args.n_vocab + 4095u) / 4096u;
    float best = -INFINITY;
    uint index = 0xffffffffu;
    for (uint c = 0; c < chunks; c++) {
        const uint2 p = partials[(ulong)row * chunks + c];
        if (p.y == 0xffffffffu) continue;
        const float v = as_type<float>(p.x);
        if (index == 0xffffffffu || v > best || (v == best && p.y < index)) { best = v; index = p.y; }
    }
    out_index[row] = index == 0xffffffffu ? 0u : index;
}

/* Row i answers the position candidate i sits on, so it predicts candidate
 * i + 1. The accepted length is one plus the number of leading rows whose argmax
 * matches, capped by max_commit, and the replacement token is the argmax of the
 * row that stopped it - the same walk the host loop did. */
struct ds4_metal_args_dsv41_block_accept {
    uint32_t rows;
    uint32_t max_commit;
};

/* Publishing the accepted prefix of one layer's KV delta into the committed
 * ring. The count is read from the accept kernel's own output, so the host does
 * not have to learn it before the rows can land; and one dispatch a layer
 * replaces one blit a row a layer - 320 of them a block at forty layers and
 * eight rows.
 *
 * slots is the path: slots[i] is the delta row that path index i holds. A linear
 * block passes none and the identity is used. */
struct ds4_metal_args_dsv41_block_publish {
    uint32_t depth;
    uint32_t row_floats;
    uint32_t ring_cap;
    uint32_t delta_start;
    uint32_t cap;          /* an upper bound on the accepted count */
    uint32_t has_slots;
};

kernel void kernel_dsv41_block_publish(
        constant ds4_metal_args_dsv41_block_publish &args,
        device const uint32_t *accepted,
        device const uint32_t *slots,
        device const float *delta,
        device float *window,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 gid = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    const uint i = gid.y;
    const uint col = gid.x;
    if (i >= args.depth || col >= args.row_floats) return;
    const uint n = min(accepted[0], args.cap);
    if (i >= n) return;
    const uint src = args.has_slots ? slots[i] : i;
    if (src >= args.depth) return;
    const uint ring = (args.delta_start + i) % args.ring_cap;
    window[(ulong)ring * args.row_floats + col] =
        delta[(ulong)src * args.row_floats + col];
}

/* The accept for a candidate tree. Node 0 is the pending token - the one the
 * session's last logits already chose - so it is accepted by construction, and
 * every other node is accepted when its parent is and its token is the one the
 * parent's logits pick. Siblings carry distinct tokens, so at most one child of
 * an accepted node can match and the accepted set is a single root path: the
 * deepest one the verifier's own argmaxes agree with.
 *
 * out: [0] accepted depth, [1] deepest accepted node, [2] token after it,
 *      [3 + d] the node at depth d of the chosen path. */
struct ds4_metal_args_dsv41_tree_accept {
    uint32_t node_count;
    uint32_t max_depth;
};

kernel void kernel_dsv41_tree_accept(
        constant ds4_metal_args_dsv41_tree_accept &args,
        device const uint32_t *argmax,
        device const int32_t  *token,
        device const uint32_t *parent,
        device uint32_t *out,
        uint tg_ [[threadgroup_position_in_grid]], uint tid_ [[thread_position_in_threadgroup]], uint ntg_ [[threads_per_threadgroup]]) {
    const uint tid = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (tid != 0 || args.node_count == 0u) return;
    uint cur = 0u;
    uint depth = 1u;
    out[3] = 0u;
    while (depth < args.max_depth) {
        const uint want = argmax[cur];
        uint next = 0xffffffffu;
        for (uint c = 1u; c < args.node_count; c++)
            if (parent[c] == cur && (uint)token[c] == want) { next = c; break; }
        if (next == 0xffffffffu) break;
        out[3u + depth] = next;
        cur = next;
        depth++;
    }
    out[0] = depth;
    out[1] = cur;
    out[2] = argmax[cur];
}

kernel void kernel_dsv41_block_accept(
        constant ds4_metal_args_dsv41_block_accept &args,
        device const uint32_t *argmax,
        device const int32_t *candidates,
        device uint32_t *out,          /* [0] accepted, [1] token after the prefix */
        uint tg_ [[threadgroup_position_in_grid]], uint tid_ [[thread_position_in_threadgroup]], uint ntg_ [[threads_per_threadgroup]]) {
    const uint tid = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (tid != 0) return;
    uint n = args.max_commit == 0u ? 0u : 1u;
    for (uint i = 0; n > 0u && n < args.max_commit && i + 1u < args.rows; i++) {
        if (argmax[i] != (uint32_t)candidates[i + 1u]) break;
        n++;
    }
    out[0] = n;
    out[1] = n == 0u ? argmax[0] : argmax[n - 1u];
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    /* Row strides of the output and of the block mask, in floats. Zero keeps
     * the packed layout these kernels always assumed - a row is exactly as wide
     * as the visible window - which is only true when the caller allocated for
     * this width. A block's per-row buffers are allocated for the context, so
     * batching the rows needs the real stride rather than a derived one. */
    uint out_stride;
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    const uint out_stride = args.out_stride ? args.out_stride : count;
    blocks[(ulong)index.y * out_stride + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 tg_ [[threadgroup_position_in_grid]], uint2 tid_ [[thread_position_in_threadgroup]], uint2 ntg_ [[threads_per_threadgroup]]) {
    const uint2 index = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint mask_stride = args.mask_stride ? args.mask_stride : blocks;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

/* ---- Abort gate -----------------------------------------------------------
 * A layer's routed experts can only run if every selected expert has an
 * address in this layer's table. Deciding that on the host costs a command
 * buffer boundary and a readback a layer, and an event cannot be skipped
 * because it is encoded before anyone has looked.
 *
 * So the GPU decides. If a lane is unreachable this records which layer, which
 * lanes and which ids, and then zeroes the indirect dispatch arguments of
 * every dispatch encoded after it, so the rest of the segment issues commands
 * that launch no threadgroups and touch nothing. The command buffer completes
 * normally, and the host repairs from a record it can trust because completion
 * is an ordering point.
 *
 * miss[0] = aborted layer + 1 (0 when the segment ran clean), [1] = lane mask,
 * [2] = expert count, [3..] = the selected ids as they were.
 */
kernel void kernel_dsv41_gate_validate(
        device const int   *selected  [[buffer(0)]],
        device const ulong *gate_addr [[buffer(1)]],
        device const ulong *up_addr   [[buffer(2)]],
        device const ulong *down_addr [[buffer(3)]],
        device uint        *grid      [[buffer(4)]],
        device atomic_uint *miss      [[buffer(5)]],
        constant uint      &layer     [[buffer(6)]],
        constant uint      &n_sel     [[buffer(7)]],
        constant uint      &gate_from [[buffer(8)]],
        constant uint      &n_slots   [[buffer(9)]],
        constant uint      &n_total   [[buffer(10)]],
        device uint        *route     [[buffer(11)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tsz [[threads_per_threadgroup]]) {
    /* The route this layer took, written whether it hits or misses. The host
     * reads it when the command buffer completes - no new synchronisation -
     * and uses it to do the cache accounting the readback path does inline.
     * Without it the 71% of layers whose ids never reach the host are
     * invisible to the cache's own policy. */
    if (tid == 0) {
        device uint *slot = route + 8u * layer;
        slot[1] = layer;
        for (uint k = 0; k < n_sel && k < 6u; ++k) slot[2u + k] = (uint)selected[k];
        slot[0] = 1u;
    }
    threadgroup atomic_uint tg_mask;
    if (tid == 0) atomic_store_explicit(&tg_mask, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n_sel) {
        const int id = selected[tid];
        const bool ok = id >= 0 && (uint)id < n_total &&
                        gate_addr[id] != 0 && up_addr[id] != 0 && down_addr[id] != 0;
        if (!ok) atomic_fetch_or_explicit(&tg_mask, 1u << tid, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint mask = atomic_load_explicit(&tg_mask, memory_order_relaxed);
    if (mask == 0u) return;
    if (tid == 0) {
        /* The first layer to abort owns the record; later layers in the same
         * segment are already gated off and must not overwrite it. */
        uint expected = 0u;
        if (atomic_compare_exchange_weak_explicit(&miss[0], &expected, layer + 1u,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&miss[1], mask, memory_order_relaxed);
            atomic_store_explicit(&miss[2], n_sel, memory_order_relaxed);
            for (uint k = 0; k < n_sel; ++k)
                atomic_store_explicit(&miss[3 + k], (uint)selected[k], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = gate_from + tid; i < n_slots; i += tsz) {
        grid[3u * i + 0u] = 0u;
        grid[3u * i + 1u] = 0u;
        grid[3u * i + 2u] = 0u;
    }
}

/* The router's whole finalize for one token of V4.1 - 384 experts, 6 used - in
 * one dispatch instead of eight: softplus and sqrt, the bias add, the top-6
 * selection, the gather of the six probabilities, their sum, the clamp, the
 * division and the scale. Each step is the expression, and the order, of the
 * kernel it replaces:
 *   - softplus and sqrt: kernel_dsv4_softplus_sqrt_f32_4, over the same float4
 *     groups;
 *   - the selection: kernel_argsort_f32_i32 (descending, non-causal) as one
 *     block of 512, the same compare-exchange network and padding rule;
 *   - the sum: kernel_sum_rows_impl dispatched with six threads - two simd_sum
 *     passes over six active lanes;
 *   - clamp(x, min, +inf), then weight / sum, then * scale.
 * So the selected ids, their order and the weights come out bit for bit. */
struct ds4_metal_args_dsv41_router_finalize {
    uint32_t n_expert;       /* 384 */
    float    scale;          /* 1.5 */
    float    clamp_min;      /* 6.103515625e-5 */
    float    clamp_max;      /* +inf */
};

kernel void kernel_dsv41_router_finalize(
        constant ds4_metal_args_dsv41_router_finalize &args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        ushort col   [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    const int width = (int)args.n_expert;
    threadgroup float *shmem_f32 = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *probs_tg = shmem_f32 + ntg;

    if (col < width / 4) {
        device const float4 *s = (device const float4 *)logits;
        device       float4 *d = (device       float4 *)probs;
        const float4 x = s[col];
        const float4 ex = exp(x);
        const float4 em = min(ex, 0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float4 sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float4 p = sqrt(sp);
        d[col] = p;
        probs_tg[4*col + 0] = p.x;
        probs_tg[4*col + 1] = p.y;
        probs_tg[4*col + 2] = p.z;
        probs_tg[4*col + 3] = p.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    shmem_i32[col] = col;
    if (col < width) shmem_f32[col] = probs_tg[col] + bias[col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                        (shmem_i32[ixj] < width &&
                         shmem_f32[shmem_i32[col]] < shmem_f32[shmem_i32[ixj]])) {
                        const int t = shmem_i32[col]; shmem_i32[col] = shmem_i32[ixj]; shmem_i32[ixj] = t;
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                        (shmem_i32[col] < width &&
                         shmem_f32[shmem_i32[col]] > shmem_f32[shmem_i32[ixj]])) {
                        const int t = shmem_i32[col]; shmem_i32[col] = shmem_i32[ixj]; shmem_i32[ixj] = t;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (col < 6) {
        const int id = shmem_i32[col];
        selected[col] = id;
        const float w = probs_tg[id];
        /* kernel_sum_rows_impl with a six-thread group: the row sum, then the
         * second pass over lane 0's sum and five zeros. */
        const float s1 = simd_sum(w);
        const float s2 = simd_sum(tiisg == 0 ? s1 : 0.0f);
        /* Each of these was its own kernel - clamp, divide, scale - so each
         * result is stored and read back through a volatile slot: under fast
         * math the compiler is otherwise free to fold (w / c) * scale into a
         * form that rounds differently, which it did for 39% of the weights. */
        volatile threadgroup float *slot = (volatile threadgroup float *)(probs_tg + 384) + col * 2;
        slot[0] = clamp(s2, args.clamp_min, args.clamp_max);
        slot[1] = w / slot[0];
        weights[col] = slot[1] * args.scale;
    }
}

/* The fused finalize against the generic chain on the same logits: bitwise
 * mismatches in the six ids, the 384 probabilities and the six weights, added
 * to a counter the host reads at the end of a run. */
kernel void kernel_dsv41_router_finalize_compare(
        device const float   *probs_a,
        device const float   *probs_b,
        device const int32_t *sel_a,
        device const int32_t *sel_b,
        device const float   *w_a,
        device const float   *w_b,
        device atomic_uint   *mismatch,        /* [0] probs, [1] ids, [2] weights, [3] calls */
        ushort tid [[thread_position_in_threadgroup]]) {
    uint bad = 0;
    for (uint i = tid; i < 384u; i += 256u)
        if (as_type<uint>(probs_a[i]) != as_type<uint>(probs_b[i])) bad++;
    if (bad) atomic_fetch_add_explicit(&mismatch[0], bad, memory_order_relaxed);
    if (tid < 6) {
        if (sel_a[tid] != sel_b[tid])
            atomic_fetch_add_explicit(&mismatch[1], 1u, memory_order_relaxed);
        if (as_type<uint>(w_a[tid]) != as_type<uint>(w_b[tid]))
            atomic_fetch_add_explicit(&mismatch[2], 1u, memory_order_relaxed);
    }
    if (tid == 0) atomic_fetch_add_explicit(&mismatch[3], 1u, memory_order_relaxed);
}

/* route_and_schedule: the router finalize above and the abort gate's
 * validation in one dispatch. The finalize part is kernel_dsv41_router_finalize
 * line for line; the validation part is kernel_dsv41_gate_validate reading the
 * six ids from threadgroup memory instead of back from the buffer the first
 * part just wrote. What it writes - the route log, the miss record, the zeroed
 * grid slots - is what the separate validate writes. */
struct ds4_metal_args_dsv41_route_validate {
    uint32_t n_expert;
    float    scale;
    float    clamp_min;
    float    clamp_max;
    uint32_t layer;
    uint32_t n_sel;
    uint32_t gate_from;
    uint32_t n_slots;
};

/* The router's finalize and the gate's validation for one token, over
 * threadgroup storage the kernel owns: kernel_dsv41_route_and_validate, and
 * the last threadgroup of kernel_dsv41_route_schedule. */
static inline void dsv41_route_validate_body(
        constant ds4_metal_args_dsv41_route_validate &args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        device const ulong *gate_addr,
        device const ulong *up_addr,
        device const ulong *down_addr,
        device        uint *grid,
        device atomic_uint *miss,
        device        uint *route,
        threadgroup int32_t *shmem_i32,
        threadgroup float *shmem_f32,
        threadgroup float *probs_tg,
        threadgroup atomic_uint *tg_mask,
        ushort col, ushort tiisg, ushort ntg, bool validate) {
    const int width = (int)args.n_expert;
    if (col < width / 4) {
        device       float4 *d = (device       float4 *)probs;
        /* volatile: in the fused form the rows were written by other
         * threadgroups of the same dispatch, published by a device fence. */
        const float4 x = *((device volatile const float4 *)(logits + 4 * col));
        const float4 ex = exp(x);
        const float4 em = min(ex, 0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float4 sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float4 p = sqrt(sp);
        d[col] = p;
        probs_tg[4*col + 0] = p.x;
        probs_tg[4*col + 1] = p.y;
        probs_tg[4*col + 2] = p.z;
        probs_tg[4*col + 3] = p.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    shmem_i32[col] = col;
    if (col < width) shmem_f32[col] = probs_tg[col] + bias[col];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (shmem_i32[col] >= width ||
                        (shmem_i32[ixj] < width &&
                         shmem_f32[shmem_i32[col]] < shmem_f32[shmem_i32[ixj]])) {
                        const int t = shmem_i32[col]; shmem_i32[col] = shmem_i32[ixj]; shmem_i32[ixj] = t;
                    }
                } else {
                    if (shmem_i32[ixj] >= width ||
                        (shmem_i32[col] < width &&
                         shmem_f32[shmem_i32[col]] > shmem_f32[shmem_i32[ixj]])) {
                        const int t = shmem_i32[col]; shmem_i32[col] = shmem_i32[ixj]; shmem_i32[ixj] = t;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (col < 6) {
        const int id = shmem_i32[col];
        selected[col] = id;
        const float w = probs_tg[id];
        const float s1 = simd_sum(w);
        const float s2 = simd_sum(tiisg == 0 ? s1 : 0.0f);
        volatile threadgroup float *slot = (volatile threadgroup float *)(probs_tg + 384) + col * 2;
        slot[0] = clamp(s2, args.clamp_min, args.clamp_max);
        slot[1] = w / slot[0];
        weights[col] = slot[1] * args.scale;
    }

    /* The validation, as kernel_dsv41_gate_validate. */
    if (!validate) return;
    if (col == 0) {
        device uint *rs = route + 8u * args.layer;
        rs[1] = args.layer;
        for (uint k = 0; k < args.n_sel && k < 6u; ++k) rs[2u + k] = (uint)shmem_i32[k];
        rs[0] = 1u;
        atomic_store_explicit(tg_mask, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (col < args.n_sel) {
        const int id = shmem_i32[col];
        const bool ok = id >= 0 && id < width &&
                        gate_addr[id] != 0 && up_addr[id] != 0 && down_addr[id] != 0;
        if (!ok) atomic_fetch_or_explicit(tg_mask, 1u << col, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint mask = atomic_load_explicit(tg_mask, memory_order_relaxed);
    if (mask == 0u) return;
    if (col == 0) {
        uint expected = 0u;
        if (atomic_compare_exchange_weak_explicit(&miss[0], &expected, args.layer + 1u,
                                                  memory_order_relaxed, memory_order_relaxed)) {
            atomic_store_explicit(&miss[1], mask, memory_order_relaxed);
            atomic_store_explicit(&miss[2], args.n_sel, memory_order_relaxed);
            for (uint k = 0; k < args.n_sel; ++k)
                atomic_store_explicit(&miss[3 + k], (uint)shmem_i32[k], memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = args.gate_from + col; i < args.n_slots; i += ntg) {
        grid[3u * i + 0u] = 0u;
        grid[3u * i + 1u] = 0u;
        grid[3u * i + 2u] = 0u;
    }
}

kernel void kernel_dsv41_route_and_validate(
        constant ds4_metal_args_dsv41_route_validate &args,
        device const float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        device const ulong *gate_addr,
        device const ulong *up_addr,
        device const ulong *down_addr,
        device        uint *grid,
        device atomic_uint *miss,
        device        uint *route,
        ushort col   [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    /* Static threadgroup storage, not a length set on the encoder: the gate is
     * armed right after this dispatch, and the driver faulted encoding the next
     * indirect dispatch behind a dynamic threadgroup length. The separate
     * validate this replaces never set one. */
    threadgroup int32_t shmem_i32[512];
    threadgroup float shmem_f32[512];
    threadgroup float probs_tg[384 + 12];
    threadgroup atomic_uint tg_mask;
    dsv41_route_validate_body(args, logits, bias, probs, selected, weights, gate_addr,
                              up_addr, down_addr, grid, miss, route, shmem_i32, shmem_f32,
                              probs_tg, &tg_mask, col, tiisg, ntg, true);
}

// ---- V4.1 layer transitions, fused -------------------------------------
// Each of these is dispatches the graph already runs, put in one: the
// arithmetic is the separate kernels' own, statement for statement, so the
// bits stored are theirs. DS4_METAL_V41_FUSE_BF16=3 runs both forms and counts
// every word that differs.

struct ds4_metal_args_dsv41_hc_tail {
    int32_t  n_embd;
    int32_t  sinkhorn_iters;
    float    hc_eps;
    float    norm_eps;
    int64_t  n_hc;
};

// The tail of a hyper-connection pre step, in the one threadgroup
// kernel_rms_norm_mul_f32_4 (bf16) would run with: kernel_dsv4_hc_split_sinkhorn
// (HC=4) on thread 0, and on every thread kernel_dsv4_hc_weighted_sum (bf16)
// for the elements of the float4s that thread then normalizes.
// The weighted sum reads the previous split's weights, never the one written
// here, so the two do not wait on each other.
kernel void kernel_dsv41_hc_tail(
        constant ds4_metal_args_dsv41_hc_tail & args,
        device const float  * mix,
        device const float  * scale,
        device const float  * base,
        device       float  * out,
        device const float  * x,
        device const float  * weights,
        device const float4 * norm_w,
        device       float  * sum_out,
        device       float4 * norm_out,
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    threadgroup float shmem_f32[32];

    /* sinkhorn_iters 0: the split is computed elsewhere, off the critical
     * path, and this is the weighted sum and the norm only. */
    if (tid == 0 && args.sinkhorn_iters > 0) {
        const float epsv       = args.hc_eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) out) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    // The weighted sum for every element this thread's float4s cover...
    const int ne00_t = args.n_embd / 4;
    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        for (short k = 0; k < 4; ++k) {
            const int64_t d = (int64_t)i00 * 4 + k;
            float acc = 0.0f;
            for (int64_t h = 0; h < args.n_hc; ++h) {
                const float xv = x[d + h * args.n_embd];
                const float wv = weights[h];
                acc += xv * wv;
            }
            uint bits = as_type<uint>(acc);
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            acc = as_type<float>(bits & 0xffff0000u);
            sum_out[d] = acc;
        }
    }

    // ...then kernel_rms_norm_fuse_impl<float4, 2> as written, reading back
    // what this same thread stored: summing registers instead let the
    // compiler contract the running sum into the dot, which moved the scale.
    device const float4 * xs = (device const float4 *) sum_out;
    float sumf = 0.0f;

    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        sumf += dot(xs[i00], xs[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.n_embd;
    const float scl   = 1.0f/sqrt(mean + args.norm_eps);

    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        float4 v = (xs[i00]*scl);
        v = v*norm_w[i00];
        norm_out[i00] = ds4_round_bf16(v);
    }
}

// kernel_dsv4_hc_expand4 with the after-MoE copy of the split's pre weights
// folded in: the copy was a blit, and a blit ends the compute encoder.
kernel void kernel_dsv41_hc_expand4_pre(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        device  const float * split,
        device        float * pre,
        uint tg_ [[threadgroup_position_in_grid]],
        uint tid_ [[thread_position_in_threadgroup]],
        uint ntg_ [[threads_per_threadgroup]]) {
    /* Indexed through the threadgroup - see kernel_swiglu_flat_f32. */
    const uint gid = tg_ * ntg_ + tid_;
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }
    if (gid < 4) {
        pre[gid] = split[gid];
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) =
            args.bf16 ? ds4_round_bf16(acc) : acc;
    }
}

// Words of b that differ from a, added to counts[slot].
kernel void kernel_dsv41_words_compare(
        device const uint * a,
        device const uint * b,
        constant uint     & n,
        constant uint     & slot,
        device atomic_uint * counts,
        uint tg_ [[threadgroup_position_in_grid]], uint tid_ [[thread_position_in_threadgroup]], uint ntg_ [[threads_per_threadgroup]]) {
    const uint gid = tg_ * ntg_ + tid_; /* through the threadgroup: see kernel_swiglu_flat_f32 */
    if (gid < n && a[gid] != b[gid])
        atomic_fetch_add_explicit(&counts[slot], 1u, memory_order_relaxed);
}

// kernel_dsv41_quantize that also stores the result into `store`: the KV row
// the attention window keeps was a blit of the quantized row, and a blit ends
// the compute encoder.
kernel void kernel_dsv41_quantize_store(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        device float *store,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) {
        const float stored = dsv41_bf16(result);
        x[index] = stored;
        store[index] = stored;
    }
}

// kernel_dsv4_attn_out_low_q8_0_f32 rounding its store to BF16, which the
// separate pass after it did to the same values.
kernel void kernel_dsv41_attn_out_low_q8_0_f32_round(
        constant ds4_metal_args_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    device const char * src0_cur = src0s + idx*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;
    device       char * dst_cur  = dst   + (idx*args.ne0 + i12*args.ne1*args.ne0)*sizeof(float);

    ds4_metal_args_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02,
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12,
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
        /*.round_bf16 =*/ 1,
    };

    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, thread ds4_metal_args_mul_mv &>(
        args0,
        src0_cur,
        src1_cur,
        dst_cur,
        shmem,
        tgpig,
        tiisg,
        sgitg);
}

// The HC expand after attention and the FFN's weighted sum and norm in one
// threadgroup of kernel_rms_norm_mul_f32_4's width: each thread expands the
// elements of the float4s it then sums (kernel_dsv4_hc_expand4, statement for
// statement), and the sum and the norm read them back as kernel_dsv41_hc_tail
// does. `split` is the attention's: post at +4, comb at +8, pre weights at 0.
struct ds4_metal_args_dsv41_hc_expand_tail {
    int32_t  n_embd;
    float    norm_eps;
    int64_t  n_hc;
};

kernel void kernel_dsv41_hc_expand_tail(
        constant ds4_metal_args_dsv41_hc_expand_tail & args,
        device const float  * block,
        device const float  * residual,
        device const float  * split,
        device       float  * after,
        device const float4 * norm_w,
        device       float  * sum_out,
        device       float4 * norm_out,
        ushort tid   [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    threadgroup float shmem_f32[32];
    const int ne00_t = args.n_embd / 4;
    device const float * post = split + 4;
    device const float * comb = split + 8;

    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        for (short k = 0; k < 4; ++k) {
            const int64_t d = (int64_t)i00 * 4 + k;
            const float block_v = block[d];
            const float r0 = residual[d + 0 * args.n_embd];
            const float r1 = residual[d + 1 * args.n_embd];
            const float r2 = residual[d + 2 * args.n_embd];
            const float r3 = residual[d + 3 * args.n_embd];
            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * post[dst_hc];

                acc += comb[dst_hc + 0 * 4] * r0;
                acc += comb[dst_hc + 1 * 4] * r1;
                acc += comb[dst_hc + 2 * 4] * r2;
                acc += comb[dst_hc + 3 * 4] * r3;

                after[d + dst_hc * args.n_embd] = ds4_round_bf16(acc);
            }
        }
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        for (short k = 0; k < 4; ++k) {
            const int64_t d = (int64_t)i00 * 4 + k;
            float acc = 0.0f;
            for (int64_t h = 0; h < args.n_hc; ++h) {
                const float xv = after[d + h * args.n_embd];
                const float wv = split[h];
                acc += xv * wv;
            }
            uint bits = as_type<uint>(acc);
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            acc = as_type<float>(bits & 0xffff0000u);
            sum_out[d] = acc;
        }
    }

    device const float4 * xs = (device const float4 *) sum_out;
    float sumf = 0.0f;

    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        sumf += dot(xs[i00], xs[i00]);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf/args.n_embd;
    const float scl   = 1.0f/sqrt(mean + args.norm_eps);

    for (int i00 = tid; i00 < ne00_t; i00 += ntg) {
        float4 v = (xs[i00]*scl);
        v = v*norm_w[i00];
        norm_out[i00] = ds4_round_bf16(v);
    }
}

// route_and_schedule with the router's projection in it: kernel_mul_mv_f32_f32_4
// (nsg 8, one row a group, as the single-row M5 dispatch runs it) for two rows
// in each 512-thread group - two clusters of eight simdgroups, each a
// statement-for-statement copy of kernel_mul_mv_t_t_4_impl<.., NR0 = 1> for its
// row - then the last group to publish runs the finalize and the validation
// (dsv41_route_validate_body) over the rows all groups wrote. validate 0 is the
// comparison form: finalize only, nothing armed or recorded.
struct ds4_metal_args_dsv41_route_schedule {
    ds4_metal_args_dsv41_route_validate rv;
    int32_t  ne00;       // router input width
    int32_t  ne01;       // experts
    uint64_t nb01;       // bytes a weight row
    uint32_t n_groups;   // threadgroups in the grid
    uint32_t validate;
};

kernel void kernel_dsv41_route_schedule(
        constant ds4_metal_args_dsv41_route_schedule &args,
        device const char  *w,
        device const char  *x,
        device       float *logits,
        device const float *bias,
        device       float *probs,
        device     int32_t *selected,
        device       float *weights,
        device const ulong *gate_addr,
        device const ulong *up_addr,
        device const ulong *down_addr,
        device        uint *grid,
        device atomic_uint *miss,
        device        uint *route,
        device atomic_uint *completion,
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort col   [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort ntg   [[threads_per_threadgroup]]) {
    threadgroup int32_t shmem_i32[512];
    threadgroup float shmem_f32[512];
    threadgroup float probs_tg[384 + 12];
    threadgroup atomic_uint tg_mask;
    threadgroup float mv_shmem[2 * 32];
    threadgroup uint is_last;

    {
        constexpr short NSG = 8;
        constexpr short NW  = N_SIMDWIDTH;
        constexpr short NB  = 32;
        constexpr short NF  = 16;
        constexpr short NF4 = NF/4;
        const short cluster = sgitg / NSG;
        const short sg = sgitg % NSG;

        const int nb = args.ne00/NB;
        const int r0 = (int)tgpig*2 + cluster;

        device const float  * y  = (device const float  *) x;
        device const float4 * y4 = (device const float4 *) x;
        device const float  * ax  = (device const float  *) (w + (uint64_t)r0*args.nb01);
        device const float4 * ax4 = (device const float4 *) (w + (uint64_t)r0*args.nb01);

        float sumf[1] = { 0.f };

        const short ix = tiisg/(NW/NF);
        const short il = tiisg%(NW/NF);

        const int ib0 = sg*NF + ix;

        float4 yl4[NF4];

        device const float4 * yb4 = y4 + (ib0*NB + il*NF)/4;

        for (int ib = ib0; ib < nb; ib += NSG*NF) {
            for (short i = 0; i < NF4; ++i) {
                yl4[i] = yb4[i];
            }

            for (short row = 0; row < 1; row++) {
                device const float4 * xb4 = ax4 + (ib*NB + il*NF)/4;

                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NF4; ++i) {
                    sumq += dot(float4(xb4[i]), float4(yl4[i]));
                }

                sumf[row] += sumq;
            }

            yb4 += NSG*NF*NW/4;
        }

        for (int i = nb*NB + sg*NW + tiisg; i < args.ne00; i += NW*NSG) {
            for (short row = 0; row < 1; row++) {
                sumf[row] += ax[i] * y[i];
            }
        }

        helper_mv_reduce_and_write<1>(logits, sumf, r0, args.ne01, tiisg, sg,
                                      (threadgroup char *)(mv_shmem + 32*cluster), false);
    }

    // Publish this group's rows; the last group to arrive finalizes.
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    if (col == 0) {
        const uint old = atomic_fetch_add_explicit(completion, 1u, memory_order_relaxed);
        is_last = old + 1u == args.n_groups ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);

    dsv41_route_validate_body(args.rv, logits, bias, probs, selected, weights, gate_addr,
                              up_addr, down_addr, grid, miss, route, shmem_i32, shmem_f32,
                              probs_tg, &tg_mask, col, tiisg, ntg, args.validate != 0u);
    if (col == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        atomic_store_explicit(completion, 0u, memory_order_relaxed);
    }
}
