#!/usr/bin/env python3
"""Compare per-tensor binary dumps from the Metal and CUDA backends.

Usage:
    tools/dump_diff.py <metal_prefix> <cuda_prefix> [filter ...]

Each backend writes files named <prefix>_<name>-<layer>_pos<pos>.bin (raw F32)
or .i32 (int32) when DS4_METAL_GRAPH_DUMP_PREFIX is set.  This tool pairs
files by their (name, layer, pos) suffix and reports per-file divergence
statistics.

Filter args (optional) limit the comparison:
    name=attn_norm           only the named tensor
    layer=2                  only that layer
    pos=10                   only that position
    name=hc_,layer=0..2      multiple comma- or repeat-separated filters

Example workflow:

    # On Mac (Metal):
    DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/metal_run \
    DS4_METAL_GRAPH_DUMP_LAYER=all \
    DS4_METAL_GRAPH_DUMP_POS=15 \
        ./ds4 -m model.gguf --metal -p "Hello" --metal-graph-decode-test 16

    # Sync /tmp/metal_run_*.bin to Linux box

    # On Linux (CUDA, with same prompt):
    DS4_METAL_GRAPH_DUMP_PREFIX=/tmp/cuda_run \
    DS4_METAL_GRAPH_DUMP_LAYER=all \
    DS4_METAL_GRAPH_DUMP_POS=15 \
        ./ds4 -m model.gguf --metal -p "Hello" --metal-graph-decode-test 16

    # Compare:
    tools/dump_diff.py /tmp/metal_run /tmp/cuda_run

The comparison reports per-file:
    - n_elements
    - max|a-b|, rms|a-b|
    - relative max diff (max|a-b| / max|a|)
    - top-3 mismatched index/values

A single-line summary per file plus a final per-tensor-name aggregate
makes it easy to spot the layer/stage where Metal and CUDA first diverge.
"""

import os
import re
import struct
import sys
from collections import defaultdict


SUFFIX_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_]+)-(?P<layer>\d+)_pos(?P<pos>\d+)\.(?P<ext>bin|i32)$"
)


class Filter:
    def __init__(self, args):
        self.names = []
        self.layers = []
        self.positions = []
        for a in args:
            for kv in a.split(","):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k == "name":
                    self.names.append(v)
                elif k == "layer":
                    self.layers.append(_parse_int_range(v))
                elif k in ("pos", "position"):
                    self.positions.append(_parse_int_range(v))

    def matches(self, name, layer, pos):
        if self.names and not any(n in name for n in self.names):
            return False
        if self.layers and not any(lo <= layer <= hi for lo, hi in self.layers):
            return False
        if self.positions and not any(lo <= pos <= hi for lo, hi in self.positions):
            return False
        return True


def _parse_int_range(s):
    if ".." in s:
        a, b = s.split("..", 1)
        return (int(a), int(b))
    v = int(s)
    return (v, v)


def _index_dir(prefix):
    """Return {(name,layer,pos,ext): full_path} for files matching <prefix>_*.

    Strips the literal `<prefix>_` from the front, then matches the rest
    against SUFFIX_RE.  This avoids greedy matching pitfalls when the
    tensor name itself contains underscores (e.g. `hc_attn_pre_mixes`).
    """
    direc, base = os.path.split(prefix)
    if not direc:
        direc = "."
    head = base + "_"
    out = {}
    for fn in os.listdir(direc):
        if not fn.startswith(head):
            continue
        suffix = fn[len(head):]
        m = SUFFIX_RE.match(suffix)
        if not m:
            continue
        key = (m.group("name"), int(m.group("layer")), int(m.group("pos")), m.group("ext"))
        out[key] = os.path.join(direc, fn)
    return out


def _load_f32(path):
    with open(path, "rb") as fp:
        data = fp.read()
    n = len(data) // 4
    return struct.unpack(f"<{n}f", data)


def _load_i32(path):
    with open(path, "rb") as fp:
        data = fp.read()
    n = len(data) // 4
    return struct.unpack(f"<{n}i", data)


def _stats_f32(a, b):
    if len(a) != len(b):
        return None
    n = len(a)
    max_abs = 0.0
    max_idx = -1
    sumsq = 0.0
    sumabs_a = 0.0
    max_a = 0.0
    for i in range(n):
        d = a[i] - b[i]
        ad = d if d >= 0 else -d
        if ad > max_abs:
            max_abs = ad
            max_idx = i
        sumsq += d * d
        aa = a[i] if a[i] >= 0 else -a[i]
        sumabs_a += aa
        if aa > max_a:
            max_a = aa
    rms = (sumsq / n) ** 0.5
    rel = max_abs / max_a if max_a > 0 else float("nan")
    # Top mismatches (small heap, but n_top<=3; just sort partial)
    top = sorted(
        ((abs(a[i] - b[i]), i, a[i], b[i]) for i in range(n)),
        reverse=True,
    )[:3]
    return {
        "n": n,
        "max_abs": max_abs,
        "max_idx": max_idx,
        "rms": rms,
        "rel": rel,
        "max_a": max_a,
        "top": top,
    }


def main():
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    a_prefix = sys.argv[1]
    b_prefix = sys.argv[2]
    flt = Filter(sys.argv[3:])

    a_idx = _index_dir(a_prefix)
    b_idx = _index_dir(b_prefix)

    common = sorted(set(a_idx.keys()) & set(b_idx.keys()))
    only_a = sorted(set(a_idx.keys()) - set(b_idx.keys()))
    only_b = sorted(set(b_idx.keys()) - set(a_idx.keys()))

    if not common:
        print(f"no overlapping dumps under {a_prefix} and {b_prefix}", file=sys.stderr)
        if only_a:
            print(f"  {len(only_a)} files only under A", file=sys.stderr)
        if only_b:
            print(f"  {len(only_b)} files only under B", file=sys.stderr)
        sys.exit(1)

    per_name_max = defaultdict(float)
    per_name_count = defaultdict(int)

    print(
        f"{'name':30s} {'layer':>5s} {'pos':>5s} "
        f"{'n':>8s} {'max_abs':>11s} {'rms':>11s} {'rel':>8s}  top"
    )
    for name, layer, pos, ext in common:
        if not flt.matches(name, layer, pos):
            continue
        a_path = a_idx[(name, layer, pos, ext)]
        b_path = b_idx[(name, layer, pos, ext)]
        if ext == "bin":
            a = _load_f32(a_path)
            b = _load_f32(b_path)
            st = _stats_f32(a, b)
            if st is None:
                print(
                    f"{name:30s} {layer:5d} {pos:5d} "
                    f"size mismatch: {len(a)} vs {len(b)}"
                )
                continue
            top_str = ", ".join(
                f"[{i}] {av:.4g}/{bv:.4g}" for _, i, av, bv in st["top"]
            )
            print(
                f"{name:30s} {layer:5d} {pos:5d} "
                f"{st['n']:8d} {st['max_abs']:11.4g} {st['rms']:11.4g} "
                f"{st['rel']:8.2%}  {top_str}"
            )
            per_name_max[name] = max(per_name_max[name], st["max_abs"])
            per_name_count[name] += 1
        else:  # i32
            ai = _load_i32(a_path)
            bi = _load_i32(b_path)
            if len(ai) != len(bi):
                print(
                    f"{name:30s} {layer:5d} {pos:5d} "
                    f"size mismatch i32: {len(ai)} vs {len(bi)}"
                )
                continue
            mismatches = sum(1 for x, y in zip(ai, bi) if x != y)
            print(
                f"{name:30s} {layer:5d} {pos:5d} "
                f"{len(ai):8d} i32 mismatches={mismatches}/{len(ai)}"
            )

    if per_name_count:
        print("\n--- per-tensor max-abs across compared files ---")
        for name in sorted(per_name_max, key=lambda n: -per_name_max[n]):
            print(
                f"{name:30s} max_abs={per_name_max[name]:11.4g}  "
                f"files={per_name_count[name]}"
            )

    if only_a or only_b:
        print(
            f"\n--- unpaired files: A_only={len(only_a)} "
            f"B_only={len(only_b)} ---"
        )
        for k in only_a[:5]:
            print(f"  A only: {k}")
        for k in only_b[:5]:
            print(f"  B only: {k}")


if __name__ == "__main__":
    main()
