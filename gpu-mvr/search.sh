#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$script_dir/build"

threads="${GPU_MVR_THREADS:-24}"
core_set="${GPU_MVR_CORE_SET:-0-23}"
warmup="${GPU_MVR_WARMUP:-5}"
timestamp="$(date +%Y%m%d_%H%M%S)"
logs_dir="${GPU_MVR_LOG_DIR:-$build_dir/search_bench_logs/$timestamp}"

mkdir -p "$logs_dir"

common_args=(
  --index /data/juelin/gpu-mvr/dataset/lotte/gpu_mvr_index
  --doclens /data/juelin/gpu-mvr/dataset/lotte/doclens.bin
  --query /data/juelin/gpu-mvr/dataset/lotte/query.bin
  --gt /data/juelin/gpu-mvr/dataset/lotte/gt.tsv
  --warmup "$warmup"
)

if (($# > 0)); then
  common_args+=("$@")
fi

effective_warmup="$warmup"
for ((i = 1; i <= $#; ++i)); do
  if [[ "${!i}" == "--warmup" && $((i + 1)) -le $# ]]; then
    next_index=$((i + 1))
    effective_warmup="${!next_index}"
  fi
done

benchmarks=(
  "v0:$build_dir/gpu_search_v0"
  "v1:$build_dir/gpu_search_v1"
  "v2:$build_dir/gpu_search_v2"
  "v3:$build_dir/gpu_search_v3"
)

if [[ ! -x "$build_dir/gpu_search_v3" && -x "$build_dir/gpu_search" ]]; then
  benchmarks+=("v3_compat:$build_dir/gpu_search")
fi

extract_runtime_seconds() {
  local file="$1"
  sed -n 's/^\[\([0-9.][0-9.]*\) s\].*/\1/p' "$file" | tail -n 1
}

extract_query_count() {
  local file="$1"
  sed -n 's/.*time for \([0-9][0-9]*\) queries\..*/\1/p' "$file" | tail -n 1
}

extract_recall() {
  local file="$1"
  sed -n 's/^Recall@[0-9][0-9]*: \([0-9.][0-9.]*\).*/\1/p' "$file" | tail -n 1
}

summary_csv="$logs_dir/summary.csv"
printf "method,binary,queries,time_seconds,qps,speedup_vs_baseline,improvement_vs_baseline_pct,speedup_vs_prev,improvement_vs_prev_pct,recall\n" > "$summary_csv"

baseline_time=""
previous_time=""
ran_count=0

echo "Pinned benchmark configuration:"
echo "  cores   : $core_set"
echo "  threads : $threads"
echo "  warmup  : $effective_warmup"
echo "  logs    : $logs_dir"
if (($# > 0)); then
  echo "  extra args     : $*"
fi
echo

for entry in "${benchmarks[@]}"; do
  IFS=":" read -r label bin <<< "$entry"
  if [[ ! -x "$bin" ]]; then
    echo "Skipping $label ($bin not found)"
    continue
  fi

  log_file="$logs_dir/${label}.log"
  echo "=== Running $label ==="
  echo "Binary: $bin"

  env \
    OMP_NUM_THREADS="$threads" \
    OMP_DYNAMIC=false \
    OMP_PROC_BIND=spread \
    OMP_PLACES=cores \
    taskset -c "$core_set" "$bin" "${common_args[@]}" \
    | tee "$log_file"

  runtime_s="$(extract_runtime_seconds "$log_file")"
  query_count="$(extract_query_count "$log_file")"
  recall="$(extract_recall "$log_file")"

  if [[ -z "$runtime_s" || -z "$query_count" ]]; then
    echo "Failed to parse runtime metrics for $label from $log_file" >&2
    exit 1
  fi

  if [[ -z "$baseline_time" && "$label" == "v0" ]]; then
    baseline_time="$runtime_s"
  fi
  if [[ -z "$baseline_time" ]]; then
    baseline_time="$runtime_s"
  fi

  qps="$(awk -v q="$query_count" -v t="$runtime_s" 'BEGIN { printf "%.2f", q / t }')"
  speedup_vs_baseline="$(awk -v base="$baseline_time" -v t="$runtime_s" 'BEGIN { printf "%.3fx", base / t }')"
  improvement_vs_baseline_pct="$(awk -v base="$baseline_time" -v t="$runtime_s" 'BEGIN { printf "%.2f", ((base - t) / base) * 100.0 }')"

  if [[ -z "$previous_time" ]]; then
    speedup_vs_prev="1.000x"
    improvement_vs_prev_pct="0.00"
  else
    speedup_vs_prev="$(awk -v prev="$previous_time" -v t="$runtime_s" 'BEGIN { printf "%.3fx", prev / t }')"
    improvement_vs_prev_pct="$(awk -v prev="$previous_time" -v t="$runtime_s" 'BEGIN { printf "%.2f", ((prev - t) / prev) * 100.0 }')"
  fi

  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "$label" \
    "$(basename "$bin")" \
    "$query_count" \
    "$runtime_s" \
    "$qps" \
    "$speedup_vs_baseline" \
    "$improvement_vs_baseline_pct" \
    "$speedup_vs_prev" \
    "$improvement_vs_prev_pct" \
    "${recall:-}" \
    >> "$summary_csv"

  previous_time="$runtime_s"
  echo
  ran_count=$((ran_count + 1))
done

if ((ran_count == 0)); then
  echo "No benchmark binaries were found in $build_dir" >&2
  exit 1
fi

echo "Summary:"
if command -v column >/dev/null 2>&1; then
  column -s, -t "$summary_csv"
else
  cat "$summary_csv"
fi
