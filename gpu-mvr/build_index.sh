#!/bin/bash

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profiling_dir="$repo_dir/profiling"
mkdir -p "$profiling_dir"

timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="$profiling_dir/build_index_${timestamp}.log"
timing_file="$profiling_dir/build_index_${timestamp}.timings.txt"

cmd=(
  ./build/gpu_build
  --index_dir /data/juelin/gpu-mvr/dataset/lotte/gpu_mvr_index
  --doclens /data/juelin/gpu-mvr/dataset/lotte/doclens.bin
  --data /data/juelin/gpu-mvr/dataset/lotte/data.bin
  --n_clusters 2097152
)

(
  cd "$repo_dir"
  "${cmd[@]}"
) 2>&1 | tee "$log_file"
status=${PIPESTATUS[0]}

{
  printf "timestamp=%s\n" "$timestamp"
  printf "log_file=%s\n" "$log_file"
  printf "command=%s\n" "${cmd[*]}"
  grep -E '^\[build_index\] Step [0-9]+ done|^\[build_index\]\[profile\]' "$log_file" || true
} > "$timing_file"

cp -f "$log_file" "$profiling_dir/build_index_latest.log"
cp -f "$timing_file" "$profiling_dir/build_index_latest.timings.txt"

exit "$status"
