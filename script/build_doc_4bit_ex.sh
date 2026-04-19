#!/usr/bin/env bash
set -euo pipefail

die() {
    echo "build_doc_4bit_ex.sh: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: script/build_doc_4bit_ex.sh --dataset <name> [options]

Options:
  --dataset <name>     Dataset under dataset/<name>. Required.
  --index-dir <path>   Existing split index dir. Default: dataset/<name>/gpu_mvr_2m, then gpu_mvr_1m.
  --data <path>        Raw embeddings.bin path. Default: dataset/<name>/raw/data.bin
  --output <path>      Output sidecar path. Default: <index-dir>/doc_4bit_ex.bin
  --ex-bits <count>    Residual bits to encode. Default: 4
  --batch-size <n>     Quantization batch size. Default: 16384
  --threads <n>        OpenMP threads for CPU quantization.
  --help               Show this help.
EOF
}

dataset=""
index_dir=""
data_path=""
output_path=""
ex_bits=4
batch_size=16384
threads=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            dataset="$2"
            shift 2
            ;;
        --index-dir)
            [[ $# -ge 2 ]] || die "missing value for --index-dir"
            index_dir="$2"
            shift 2
            ;;
        --data)
            [[ $# -ge 2 ]] || die "missing value for --data"
            data_path="$2"
            shift 2
            ;;
        --output)
            [[ $# -ge 2 ]] || die "missing value for --output"
            output_path="$2"
            shift 2
            ;;
        --ex-bits)
            [[ $# -ge 2 ]] || die "missing value for --ex-bits"
            ex_bits="$2"
            shift 2
            ;;
        --batch-size)
            [[ $# -ge 2 ]] || die "missing value for --batch-size"
            batch_size="$2"
            shift 2
            ;;
        --threads)
            [[ $# -ge 2 ]] || die "missing value for --threads"
            threads="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$dataset" ]] || {
    usage
    die "--dataset is required"
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/gpu-mvr/build"
binary="${build_dir}/gpu_build_doc_4bit_ex"

[[ -x "$binary" ]] || die "missing binary: $binary"

if [[ -z "$index_dir" ]]; then
    if [[ -f "${repo_root}/dataset/${dataset}/gpu_mvr_2m/doc_1bit.bin" ]]; then
        index_dir="${repo_root}/dataset/${dataset}/gpu_mvr_2m"
    elif [[ -f "${repo_root}/dataset/${dataset}/gpu_mvr_1m/doc_1bit.bin" ]]; then
        index_dir="${repo_root}/dataset/${dataset}/gpu_mvr_1m"
    else
        die "could not resolve index dir for dataset ${dataset}"
    fi
fi

if [[ -z "$data_path" ]]; then
    data_path="${repo_root}/dataset/${dataset}/raw/data.bin"
fi

cmd=(
    "$binary"
    --index_dir "$index_dir"
    --data "$data_path"
    --ex_bits "$ex_bits"
    --batch_size "$batch_size"
)

if [[ -n "$output_path" ]]; then
    cmd+=(--output "$output_path")
fi
if [[ -n "$threads" ]]; then
    cmd+=(--threads "$threads")
fi

echo "[build_doc_4bit_ex] index_dir=${index_dir}"
echo "[build_doc_4bit_ex] data=${data_path}"
echo "[build_doc_4bit_ex] ex_bits=${ex_bits}"
"${cmd[@]}"
