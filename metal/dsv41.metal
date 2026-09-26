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
        uint gid [[thread_position_in_grid]]) {
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
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
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

kernel void kernel_dsv41_block_accept(
        constant ds4_metal_args_dsv41_block_accept &args,
        device const uint32_t *argmax,
        device const int32_t *candidates,
        device uint32_t *out,          /* [0] accepted, [1] token after the prefix */
        uint tid [[thread_position_in_grid]]) {
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
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint blocks = (args.width + 7u) / 8u;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * blocks + index.x / 8u] == 0.0f
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
