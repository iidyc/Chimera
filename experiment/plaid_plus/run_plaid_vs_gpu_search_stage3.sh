#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiment/plaid_plus/run_plaid_vs_gpu_search_stage3.sh [options]

Description:
  Prepare Stage-3 focused experiments for gpu_search.

Modes:
  chunk-sweep
    Vary overlap_chunks and benchmark gpu_search.

  fullbit-efficiency
    Print the intended command / output layout for the future full-bit
    efficiency experiment. Source support is still missing.

The script defaults to dry-run mode. Pass --execute to run the supported
chunk-sweep commands.

Options:
  --mode <chunk-sweep|fullbit-efficiency>
                           Default: chunk-sweep
  --dataset <name>         Dataset. Repeatable. Default: lotte hotpot msmarco
  --version <gpu_search|gpu_search_nolut|gpu_search_nosum|gpu_search_nolut_nosum>
                           Repeatable. Default: gpu_search
  --k <top_k>              Retrieval depth. Default: 100
  --label <config_label>   Base gpu_search config label. Repeatable.
                           Default: d1 f1 l1
  --chunks <csv>           Comma-separated overlap_chunks values.
                           Default: 1,2,4,8,16
  --gpu-config <path>      gpu_search config CSV. Default: profiling/gpu_search_config.csv
  --build-dir <path>       gpu_search build dir. Default: Chimera/build
  --output-root <path>     Output profiling root.
                           Default: profiling/experiments/plaid_vs_gpu_search
  --log-root <path>        Output log root.
                           Default: log/bench/experiments/plaid_vs_gpu_search
  --warmup <count>         Warmup queries. Default: 5
  --nq <count>             Timed queries. Default: -1
  --execute                Execute supported commands
  --dry-run                Force dry-run mode
  -h, --help               Show help
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

normalize_dataset_name() {
    case "$1" in
        hotspot) printf '%s' "hotpot" ;;
        *) printf '%s' "$1" ;;
    esac
}

append_unique() {
    local value="$1"
    shift
    local existing=""
    for existing in "$@"; do
        [[ "$existing" == "$value" ]] && return 0
    done
    return 1
}

run_cmd() {
    printf '[cmd] %q ' "$@"
    printf '\n'
    if [[ $execute -eq 1 ]]; then
        "$@"
    fi
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

bench_gpu="${repo_root}/experiment/chimera/bench_gpu_search.sh"
parse_logs="${repo_root}/experiment/common/parse_bench_logs.py"

[[ -x "$bench_gpu" ]] || die "missing executable helper script: $bench_gpu"
[[ -f "$parse_logs" ]] || die "missing parser: $parse_logs"

mode="chunk-sweep"
datasets=(lotte hotpot msmarco)
versions=(gpu_search)
labels=(d1 f1 l1)
chunk_values_csv="1,2,4,8,16"
k=100
gpu_config="${repo_root}/profiling/gpu_search_config.csv"
build_dir="${repo_root}/Chimera/build"
output_root="${repo_root}/profiling/experiments/plaid_vs_gpu_search"
log_root="${repo_root}/log/bench/experiments/plaid_vs_gpu_search"
warmup=5
nq=-1
execute=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || die "missing value for --mode"
            mode="$2"
            shift 2
            ;;
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            value="$(normalize_dataset_name "$2")"
            if ! append_unique "$value" "${datasets[@]}"; then
                datasets+=("$value")
            fi
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || die "missing value for --version"
            case "$2" in
                gpu_search|gpu_search_nolut|gpu_search_nosum|gpu_search_nolut_nosum) ;;
                *) die "--version must be one of: gpu_search, gpu_search_nolut, gpu_search_nosum, gpu_search_nolut_nosum" ;;
            esac
            if ! append_unique "$2" "${versions[@]}"; then
                versions+=("$2")
            fi
            shift 2
            ;;
        --k)
            [[ $# -ge 2 ]] || die "missing value for --k"
            k="$2"
            shift 2
            ;;
        --label)
            [[ $# -ge 2 ]] || die "missing value for --label"
            if [[ ${#labels[@]} -eq 3 && "${labels[0]}" == "d1" && "${labels[1]}" == "f1" && "${labels[2]}" == "l1" ]]; then
                labels=()
            fi
            if ! append_unique "$2" "${labels[@]}"; then
                labels+=("$2")
            fi
            shift 2
            ;;
        --chunks)
            [[ $# -ge 2 ]] || die "missing value for --chunks"
            chunk_values_csv="$2"
            shift 2
            ;;
        --gpu-config)
            [[ $# -ge 2 ]] || die "missing value for --gpu-config"
            gpu_config="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "missing value for --build-dir"
            build_dir="$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || die "missing value for --output-root"
            output_root="$2"
            shift 2
            ;;
        --log-root)
            [[ $# -ge 2 ]] || die "missing value for --log-root"
            log_root="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "missing value for --warmup"
            warmup="$2"
            shift 2
            ;;
        --nq)
            [[ $# -ge 2 ]] || die "missing value for --nq"
            nq="$2"
            shift 2
            ;;
        --execute)
            execute=1
            shift
            ;;
        --dry-run)
            execute=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

case "$mode" in
    chunk-sweep)
        ;;
    fullbit-efficiency)
        cat <<EOF
[todo] Full-bit efficiency is not exported by the current gpu_search source.
[todo] Needed source additions:
       - actual Stage-3 doc count
       - actual full-bit computation count
       - optional per-chunk Stage-3 work counters
[todo] Intended future output root:
       ${output_root}/stage3_fullbit
[todo] Intended future command shape:
       gpu_search ... --stage3-efficiency-csv <path>
EOF
        exit 0
        ;;
    *)
        die "--mode must be one of: chunk-sweep, fullbit-efficiency"
        ;;
esac

tmp_config="$(mktemp "${TMPDIR:-/tmp}/gpu_search_stage3_chunks.XXXXXX.csv")"
cleanup() {
    rm -f "$tmp_config"
}
trap cleanup EXIT

python - "$gpu_config" "$tmp_config" "$chunk_values_csv" "${labels[@]}" <<'PY'
import csv
import sys

src, dst, chunks_csv, *labels = sys.argv[1:]
chunk_values = [value.strip() for value in chunks_csv.split(",") if value.strip()]
label_set = set(labels)

with open(src, newline="") as handle:
    rows = list(csv.DictReader(handle))

selected = [row for row in rows if row["label"] in label_set]
if len(selected) != len(label_set):
    found = {row["label"] for row in selected}
    missing = sorted(label_set - found)
    raise SystemExit(f"missing base labels in {src}: {', '.join(missing)}")

with open(dst, "w", newline="") as handle:
    fieldnames = ["label", "itopk_size", "nprobe", "k_rank_cluster", "k_rank_all_tokens", "overlap_chunks"]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in selected:
        for chunk in chunk_values:
            out = dict(row)
            out["label"] = f"{row['label']}_c{chunk}"
            out["overlap_chunks"] = chunk
            writer.writerow({key: out[key] for key in fieldnames})
PY

for version in "${versions[@]}"; do
    profile_root="${output_root}/stage3_chunks/${version}"
    profile_log_root="${log_root}/stage3_chunks/${version}"
    cmd=(
        "$bench_gpu"
        --version "$version"
        --build-dir "$build_dir"
        --config-file "$tmp_config"
        --output-dir "$profile_root"
        --log-dir "$profile_log_root"
        --k "$k"
        --nq "$nq"
        --warmup "$warmup"
    )
    for dataset in "${datasets[@]}"; do
        cmd+=(--dataset "$dataset")
    done
    run_cmd "${cmd[@]}"
    run_cmd python "$parse_logs" --mode chimera "${profile_log_root}/${version}"
done
