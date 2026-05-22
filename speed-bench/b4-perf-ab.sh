#!/usr/bin/env bash
# B.4 multi-session attention perf A/B.
#
# Measures N=1/2/4/8 batched-decode throughput on the current
# ds4-server with DS4_CUDA_BATCHED_DECODE_ATTENTION on vs off,
# using n-sweep.py and a long idle window before each binary
# to keep the GPU/allocator in a fresh regime (section 33
# concluded that allocator state dominates ~44 x over opt and
# arch knobs, so a perf A/B that does not control for it is
# noise).
#
# Section 33b runbook (the run conditions that produced
# preliminary B.4-is-a-regression numbers had three caveats;
# this harness's defaults are now tuned to neutralise them):
#  * AUTO_FAST_PATH=0     so N>=2 traffic actually batches
#                         (section 29's auto fast-path otherwise
#                         intercepts and forces the N=1 path)
#  * N_TOKENS=128         long enough decode that B.4's per-step
#                         kernel-launch amortisation has room
#                         to win over per-session attention
#  * cold-boot GPU        ideal but external to this script:
#                         run after `systemctl reboot` or a
#                         GPU reset, with at least 5 min of
#                         post-boot idle.  The PRE_IDLE=300
#                         default protects against the
#                         degraded-regime OOM seen in section 33b.
#
# Env knobs:
#   PORT=18001
#   PRE_IDLE=300        seconds to idle before each server start
#   COOLDOWN=240        seconds to idle between binary swaps
#   N_RUNS=3            runs per N (median is reported)
#   N_TOKENS=128        max_tokens per request
#   NS=1,2,4,8          comma-separated N values
#   ORDER=off,on        comma-separated order of B.4 toggle
#   AUTO_FAST_PATH=0    propagated to DS4_SERVER_AUTO_FAST_PATH;
#                       set to 1 only to study fast-path effects
#   LOG_DIR=/tmp/...
#
# Output:
#   $LOG_DIR/b4-perf-ab.csv   appended per-run rows from n-sweep.py
#   $LOG_DIR/server-{off,on}.log
#   $LOG_DIR/n-sweep-{off,on}.log
#   stdout summary at the end (median tok/s per N x label).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PORT="${PORT:-18001}"
PRE_IDLE="${PRE_IDLE:-300}"
COOLDOWN="${COOLDOWN:-240}"
N_RUNS="${N_RUNS:-3}"
N_TOKENS="${N_TOKENS:-128}"
NS="${NS:-1,2,4,8}"
ORDER="${ORDER:-off,on}"
AUTO_FAST_PATH="${AUTO_FAST_PATH:-0}"
LOG_DIR="${LOG_DIR:-/tmp/ds4-b4-perf-ab}"
mkdir -p "$LOG_DIR"
CSV="$LOG_DIR/b4-perf-ab.csv"
: > "$CSV"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT INT TERM

start_server() {
  local tag="$1"
  local b4_env=""
  if [[ "$tag" == "on" ]]; then
    b4_env="DS4_CUDA_BATCHED_DECODE_ATTENTION=1"
  fi
  echo "=== B.4=${tag} ===" >&2
  echo "Pre-idle ${PRE_IDLE}s..." >&2
  sleep "$PRE_IDLE"

  local log="$LOG_DIR/server-${tag}.log"
  : > "$log"

  env $b4_env \
    DS4_SERVER_AUTO_FAST_PATH="$AUTO_FAST_PATH" \
    DS4_CUDA_Q4_DECODE=1 \
    DS4_CUDA_BATCHED_QKV=1 DS4_CUDA_BATCHED_Q_B=1 \
    DS4_CUDA_BATCHED_ATTN_OUTPUT=1 DS4_CUDA_BATCHED_SHARED_FFN=1 \
    DS4_CUDA_BATCHED_ROUTED_MOE=1 \
    ./ds4-server --cuda --warm-weights --port "$PORT" --ctx 8192 \
      --kv-disk-dir "/tmp/ds4-kv-b4perf" \
      > "$log" 2>&1 &
  SERVER_PID=$!

  for _ in $(seq 1 180); do
    if grep -q "listening on" "$log"; then return 0; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "server-${tag} exited early; see $log" >&2
      SERVER_PID=""
      return 1
    fi
    sleep 1
  done
  echo "server-${tag} did not become ready in 180s; see $log" >&2
  cleanup
  return 1
}

run_sweep() {
  local tag="$1"
  python3 speed-bench/n-sweep.py \
    --url "http://127.0.0.1:${PORT}/v1/chat/completions" \
    --ns "$NS" \
    --runs "$N_RUNS" \
    --max-tokens "$N_TOKENS" \
    --warmup \
    --label "$tag" \
    --csv "$CSV" \
    2>&1 | tee "$LOG_DIR/n-sweep-${tag}.log"
}

IFS=',' read -ra order_arr <<< "$ORDER"
for tag in "${order_arr[@]}"; do
  if ! start_server "$tag"; then
    echo "skipping sweep for ${tag}" >&2
    continue
  fi
  run_sweep "$tag"
  cleanup
  echo "Cooldown ${COOLDOWN}s..." >&2
  sleep "$COOLDOWN"
done

echo
echo "=== SUMMARY (median tok/s per N, B.4 off vs on) ==="
python3 - "$CSV" <<'PYEOF'
import csv, statistics, sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists() or path.stat().st_size == 0:
    print("(no CSV)")
    sys.exit(0)
rows = list(csv.DictReader(open(path)))
by = {}
for r in rows:
    tag = r["label"]
    n = int(r["N"])
    tps = float(r["tok_per_s"])
    by.setdefault((tag, n), []).append(tps)

ns = sorted({n for _, n in by})
tags = sorted({tag for tag, _ in by})

print(f"{'N':>4}" + "".join(f"{t+' tok/s':>14}" for t in tags) + f"{'on/off':>8}")
for n in ns:
    cells = []
    for t in tags:
        tpss = by.get((t, n), [])
        med = statistics.median(tpss) if tpss else 0.0
        cells.append(med)
    ratio = ""
    if "off" in tags and "on" in tags:
        off = by.get(("off", n), [])
        on  = by.get(("on",  n), [])
        if off and on:
            r = statistics.median(on) / statistics.median(off)
            ratio = f"{r:.3f}x"
    print(f"{n:>4}" + "".join(f"{c:>14.2f}" for c in cells) + f"{ratio:>8}")
PYEOF
