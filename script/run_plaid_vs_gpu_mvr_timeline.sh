#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/run_plaid_vs_gpu_mvr_timeline.sh [options]

Description:
  Prepare `nsys` timeline runs for:
    - gpu_search_v6
    - gpu_search_v8
    - PLAID cpu-hosted
    - PLAID gpu-resident

  The script defaults to dry-run mode. Pass --execute to actually run the
  `nsys` commands later.

Options:
  --dataset <name>           Dataset. Default: lotte
  --k <top_k>                Retrieval depth. Default: 100
  --gpu-label <config_label> GPU-MVR config label. Default: l1
  --gpu-config <path>        GPU-MVR config CSV. Default: profiling/gpu_mvr_config.csv
  --colbert-pair <n,m>       ColBERT pair. Default: 4,8000
  --colbert-index-name <x>   ColBERT index name. Default: autodetect
  --build-dir <path>         GPU-MVR build dir. Default: gpu-mvr/build
  --output-root <path>       Output profiling root.
                             Default: profiling/experiments/plaid_vs_gpu_mvr
  --warmup <count>           Warmup queries. Default: 1
  --execute                  Execute commands instead of printing them
  --dry-run                  Force dry-run mode
  -h, --help                 Show help
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

run_cmd() {
    printf '[cmd] %q ' "$@"
    printf '\n'
    if [[ $execute -eq 1 ]]; then
        "$@"
    fi
}

resolve_gpu_index_dir() {
    local dataset_dir="$1"
    local candidate=""
    for candidate in \
        "${dataset_dir}/gpu_mvr_2m" \
        "${dataset_dir}/gpu_mvr_1m" \
        "${dataset_dir}/gpu_mvr" \
        "${dataset_dir}/gpu_mvr_index"; do
        if [[ -d "$candidate" && -f "${candidate}/doclens.bin" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

resolve_colbert_index_name() {
    local indexes_dir="$1"
    local names=()
    local name=""
    while IFS= read -r name; do
        [[ -n "$name" ]] && names+=("$name")
    done < <(find "$indexes_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    if [[ ${#names[@]} -eq 1 ]]; then
        printf '%s' "${names[0]}"
        return 0
    fi
    return 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

dataset="lotte"
k=100
gpu_label="l1"
gpu_config="${repo_root}/profiling/gpu_mvr_config.csv"
colbert_pair="4,8000"
colbert_index_name=""
build_dir="${repo_root}/gpu-mvr/build"
output_root="${repo_root}/profiling/experiments/plaid_vs_gpu_mvr"
warmup=1
execute=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            dataset="$2"
            shift 2
            ;;
        --k)
            [[ $# -ge 2 ]] || die "missing value for --k"
            k="$2"
            shift 2
            ;;
        --gpu-label)
            [[ $# -ge 2 ]] || die "missing value for --gpu-label"
            gpu_label="$2"
            shift 2
            ;;
        --gpu-config)
            [[ $# -ge 2 ]] || die "missing value for --gpu-config"
            gpu_config="$2"
            shift 2
            ;;
        --colbert-pair)
            [[ $# -ge 2 ]] || die "missing value for --colbert-pair"
            colbert_pair="$2"
            shift 2
            ;;
        --colbert-index-name)
            [[ $# -ge 2 ]] || die "missing value for --colbert-index-name"
            colbert_index_name="$2"
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
        --warmup)
            [[ $# -ge 2 ]] || die "missing value for --warmup"
            warmup="$2"
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

command -v nsys >/dev/null 2>&1 || die "nsys is required for timeline capture"
[[ -f "$gpu_config" ]] || die "missing GPU config CSV: $gpu_config"

dataset_dir="${repo_root}/dataset/${dataset}"
raw_dir="${dataset_dir}/raw"
gpu_index_dir="$(resolve_gpu_index_dir "$dataset_dir")" || die "failed to resolve GPU index dir for ${dataset}"
query_file="${raw_dir}/query.bin"
gt_file="${raw_dir}/gt.tsv"
doclens_file="${gpu_index_dir}/doclens.bin"
gpu_index_file="${gpu_index_dir}/ivf.bin"

[[ -f "$query_file" ]] || die "missing query file: $query_file"
[[ -f "$gt_file" ]] || die "missing gt file: $gt_file"
[[ -f "$doclens_file" ]] || die "missing doclens file: $doclens_file"
[[ -f "$gpu_index_file" ]] || die "missing index file: $gpu_index_file"

readarray -t gpu_cfg < <(
    python - "$gpu_config" "$gpu_label" <<'PY'
import csv
import sys
path, label = sys.argv[1:]
with open(path, newline="") as handle:
    for row in csv.DictReader(handle):
        if row["label"] == label:
            print(row["nprobe"])
            print(row["k_rank_cluster"])
            print(row["k_rank_all_tokens"])
            print(row["itopk_size"])
            print(row["overlap_chunks"])
            break
    else:
        raise SystemExit(f"missing label {label} in {path}")
PY
)
nprobe="${gpu_cfg[0]}"
k_rank_cluster="${gpu_cfg[1]}"
k_rank_all_tokens="${gpu_cfg[2]}"
itopk_size="${gpu_cfg[3]}"
overlap_chunks="${gpu_cfg[4]}"

colbert_experiment_dir="${dataset_dir}/colbert"
colbert_indexes_dir="${colbert_experiment_dir}/indexes"
if [[ -z "$colbert_index_name" ]]; then
    colbert_index_name="$(resolve_colbert_index_name "$colbert_indexes_dir")" || \
        die "unable to autodetect ColBERT index under ${colbert_indexes_dir}; pass --colbert-index-name"
fi

timeline_root="${output_root}/timeline/${dataset}"

echo "[note] GPU-MVR timeline readability will improve if GPU_MVR_TIMELINE / NVTX ranges are enabled."
echo "[note] ColBERT timeline currently lacks NVTX ranges around search stages."

run_cmd nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
    --output "${timeline_root}/gpu_search_v6_${gpu_label}_k${k}" \
    "${build_dir}/gpu_search_v6" \
    --query "$query_file" \
    --doclens "$doclens_file" \
    --gt "$gt_file" \
    --index "$gpu_index_file" \
    --k "$k" \
    --nq 1 \
    --warmup "$warmup" \
    --nprobe "$nprobe" \
    --k-rank-cluster "$k_rank_cluster" \
    --k-rank-all-tokens "$k_rank_all_tokens" \
    --itopk-size "$itopk_size" \
    --overlap-chunks "$overlap_chunks"

run_cmd nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
    --output "${timeline_root}/gpu_search_v8_${gpu_label}_k${k}" \
    "${build_dir}/gpu_search_v8" \
    --query "$query_file" \
    --doclens "$doclens_file" \
    --gt "$gt_file" \
    --index "$gpu_index_file" \
    --k "$k" \
    --nq 1 \
    --warmup "$warmup" \
    --nprobe "$nprobe" \
    --k-rank-cluster "$k_rank_cluster" \
    --k-rank-all-tokens "$k_rank_all_tokens" \
    --itopk-size "$itopk_size" \
    --overlap-chunks "$overlap_chunks"

run_cmd nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
    --output "${timeline_root}/plaid_cpu_hosted_${colbert_pair//,/x}_k${k}" \
    python "${repo_root}/ColBERT/experiment/search.py" \
    --root-path "${dataset_dir}" \
    --experiment-name colbert \
    --index-name "${colbert_index_name}" \
    --query-path "$query_file" \
    --ground-truth-path "$gt_file" \
    --pairs "$colbert_pair" \
    --k "$k" \
    --warmup "$warmup" \
    --dataset-name "$dataset" \
    --implementation-label plaid_cpu_hosted

run_cmd nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
    --output "${timeline_root}/plaid_gpu_resident_${colbert_pair//,/x}_k${k}" \
    python "${repo_root}/ColBERT/experiment/search.py" \
    --root-path "${dataset_dir}" \
    --experiment-name colbert \
    --index-name "${colbert_index_name}" \
    --query-path "$query_file" \
    --ground-truth-path "$gt_file" \
    --pairs "$colbert_pair" \
    --k "$k" \
    --warmup "$warmup" \
    --compressed-embeddings-storage gpu \
    --gpu-index-resident \
    --dataset-name "$dataset" \
    --implementation-label plaid_gpu_resident
