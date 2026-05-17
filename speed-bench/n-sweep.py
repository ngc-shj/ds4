#!/usr/bin/env python3
"""N-sweep batched-decode throughput bench.

Fires N parallel chat-completion requests against a running ds4-server
for each N in --ns (default 1,2,4,8), runs --runs trials each, and
reports per-N median total wall time + total tokens generated.

Prompt is the first 8000 bytes of speed-bench/promessi_sposi.txt by
default so each session does a realistic prefill before decoding.

Server must be started externally with the full batched-decode recipe:

  DS4_CUDA_BATCHED_DECODE_ATTENTION=1 DS4_CUDA_Q4_DECODE=1 \\
  DS4_CUDA_BATCHED_QKV=1 DS4_CUDA_BATCHED_Q_B=1 \\
  DS4_CUDA_BATCHED_ATTN_OUTPUT=1 DS4_CUDA_BATCHED_SHARED_FFN=1 \\
  DS4_CUDA_BATCHED_ROUTED_MOE=1 \\
      ./ds4-server --cuda --warm-weights --port 18001 --ctx 8192

A/B between -O2 and -O3 binaries: run once per binary on a cold GPU
(>=5 min cooldown between binary swaps) and compare medians.
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                  headers={"content-type": "application/json"},
                                  method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return json.loads(body), time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, str(e)


def run_burst(url, prompt, n, max_tokens):
    payload = {"messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "thinking": {"type": "disabled"}}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(lambda _: post(url, payload), range(n)))
    wall = time.time() - t0
    tokens = 0
    errs = 0
    for data, _dt, err in results:
        if err is not None:
            errs += 1
            continue
        usage = (data or {}).get("usage") or {}
        # ds4-server returns completion_tokens in OpenAI-compatible usage
        tokens += int(usage.get("completion_tokens") or 0)
    return wall, tokens, errs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url",
                    default="http://127.0.0.1:18001/v1/chat/completions")
    ap.add_argument("--ns", default="1,2,4,8")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--prompt-file",
                    default="speed-bench/promessi_sposi.txt")
    ap.add_argument("--prompt-bytes", type=int, default=8000)
    ap.add_argument("--warmup", action="store_true",
                    help="Fire one warmup request before measurement")
    ap.add_argument("--label", default="",
                    help="Tag prepended to CSV rows (e.g. 'O2' / 'O3')")
    ap.add_argument("--csv", default="",
                    help="Append rows to this CSV file")
    args = ap.parse_args()

    prompt = (Path(args.prompt_file).read_bytes()[:args.prompt_bytes]
              .decode("utf-8", errors="replace"))
    ns = [int(x) for x in args.ns.split(",")]

    if args.warmup:
        print("warmup ...", flush=True)
        run_burst(args.url, prompt, 1, args.max_tokens)

    print(f"{'N':>3} {'run':>4} {'wall(s)':>9} {'tok':>5} {'errs':>5} "
          f"{'tok/s':>7}", flush=True)
    rows = []
    for n in ns:
        walls = []
        for r in range(args.runs):
            wall, tok, errs = run_burst(args.url, prompt, n, args.max_tokens)
            tps = tok / wall if wall > 0 else 0.0
            walls.append((wall, tok, errs))
            print(f"{n:>3} {r+1:>4} {wall:>9.2f} {tok:>5} {errs:>5} "
                  f"{tps:>7.2f}", flush=True)
            rows.append((args.label, n, r + 1, wall, tok, errs, tps))
        med_wall = statistics.median(w for w, _, _ in walls)
        med_tok = statistics.median(t for _, t, _ in walls)
        med_tps = med_tok / med_wall if med_wall > 0 else 0.0
        print(f"{n:>3} {'med':>4} {med_wall:>9.2f} {int(med_tok):>5} "
              f"{'':>5} {med_tps:>7.2f}", flush=True)

    if args.csv:
        path = Path(args.csv)
        # Write header if file is missing or empty (harnesses often
        # truncate via `: > $CSV` which leaves an empty file behind).
        need_header = (not path.exists()) or path.stat().st_size == 0
        with path.open("a") as f:
            if need_header:
                f.write("label,N,run,wall_s,tokens,errs,tok_per_s\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
