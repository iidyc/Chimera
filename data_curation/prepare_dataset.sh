#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  data_curation/prepare_dataset.sh --dataset <name|all> [options]

End-to-end dataset preparation:
  1. download normalized raw text into dataset/<name>/text/
  2. encode text into dataset/<name>/raw/{data.bin,doclens.bin,query.bin}
  3. generate dataset/<name>/raw/gt.tsv
  4. build Chimera index under dataset/<name>/gpu_search_*/

Options:
  --dataset <name|all>      Dataset to process. Repeatable. Default: all.
  --output-root <path>      Root containing dataset directories. Default: ./dataset
  --checkpoint <name/path>  ColBERT checkpoint. Default: colbert-ir/colbertv2.0
  --gpus <list|auto>        CUDA devices for document encoding. Default: auto.
  --chunk-size <count>      Collection rows encoded per chunk. Default: 25000
  --n-clusters <count>      Chimera index clusters. Default: 2000000
  --topk <count>            Ground-truth depth. Default: 1000
  --doc-batch-size <count>  Docs per brute-force top-k batch. Default: 50000
  --skip-download           Do not download text files.
  --skip-encode             Do not create data.bin/query.bin/gt.tsv.
  --skip-gt                 Encode embeddings but do not generate gt.tsv.
  --skip-index              Do not build Chimera index.
  --skip-existing           Reuse existing outputs where possible.
  --overwrite               Replace existing text/raw outputs.
  --dry-run                 Print commands without running them.
  -h, --help                Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
output_root="${repo_root}/dataset"
checkpoint="colbert-ir/colbertv2.0"
gpus="auto"
chunk_size="25000"
n_clusters="2000000"
topk="1000"
doc_batch_size="50000"
skip_download=0
skip_encode=0
skip_gt=0
skip_index=0
skip_existing=0
overwrite=0
dry_run=0
datasets=()

append_dataset() {
    local name="$1"
    case "$name" in
        all)
            datasets=(lotte hotpot msmarco)
            ;;
        lotte|hotpot|msmarco)
            datasets+=("$name")
            ;;
        *)
            die "unsupported dataset: $name"
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            append_dataset "$2"
            shift 2
            ;;
        --output-root)
            [[ $# -ge 2 ]] || die "missing value for --output-root"
            output_root="$2"
            shift 2
            ;;
        --checkpoint)
            [[ $# -ge 2 ]] || die "missing value for --checkpoint"
            checkpoint="$2"
            shift 2
            ;;
        --gpus)
            [[ $# -ge 2 ]] || die "missing value for --gpus"
            gpus="$2"
            shift 2
            ;;
        --chunk-size)
            [[ $# -ge 2 ]] || die "missing value for --chunk-size"
            chunk_size="$2"
            shift 2
            ;;
        --n-clusters)
            [[ $# -ge 2 ]] || die "missing value for --n-clusters"
            n_clusters="$2"
            shift 2
            ;;
        --topk)
            [[ $# -ge 2 ]] || die "missing value for --topk"
            topk="$2"
            shift 2
            ;;
        --doc-batch-size)
            [[ $# -ge 2 ]] || die "missing value for --doc-batch-size"
            doc_batch_size="$2"
            shift 2
            ;;
        --skip-download)
            skip_download=1
            shift
            ;;
        --skip-encode)
            skip_encode=1
            shift
            ;;
        --skip-gt)
            skip_gt=1
            shift
            ;;
        --skip-index)
            skip_index=1
            shift
            ;;
        --skip-existing)
            skip_existing=1
            shift
            ;;
        --overwrite)
            overwrite=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ ${#datasets[@]} -gt 0 ]] || append_dataset all
if (( overwrite && skip_existing )); then
    die "--overwrite and --skip-existing are mutually exclusive"
fi

run_cmd() {
    if (( dry_run )); then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

for dataset in "${datasets[@]}"; do
    if (( skip_download )); then
        echo "[${dataset}] skipping text download"
    else
        download_cmd=(
            python "$repo_root/data_curation/download/download_raw_text.py"
            --dataset "$dataset"
            --output-root "$output_root"
        )
        if (( overwrite )); then
            download_cmd+=(--overwrite)
        fi
        run_cmd "${download_cmd[@]}"
    fi

    if (( skip_encode )); then
        echo "[${dataset}] skipping raw binary creation"
    else
        encode_cmd=(
            "$repo_root/data_curation/create_raw_binaries.sh"
            --dataset "$dataset"
            --output-root "$output_root"
            --checkpoint "$checkpoint"
            --gpus "$gpus"
            --chunk-size "$chunk_size"
            --topk "$topk"
            --doc-batch-size "$doc_batch_size"
        )
        if (( overwrite )); then
            encode_cmd+=(--overwrite)
        fi
        if (( skip_existing )); then
            encode_cmd+=(--skip-existing)
        fi
        if (( skip_gt )); then
            encode_cmd+=(--skip-gt)
        fi
        if (( dry_run )); then
            encode_cmd+=(--dry-run)
        fi
        run_cmd "${encode_cmd[@]}"
    fi

    if (( skip_index )); then
        echo "[${dataset}] skipping Chimera index build"
    else
        index_cmd=(
            "$repo_root/data_curation/chimera/build_index.sh"
            --dataset "$dataset"
            --dataset-root "$output_root"
            --n-clusters "$n_clusters"
        )
        if (( dry_run )); then
            index_cmd+=(--dry-run)
        fi
        run_cmd "${index_cmd[@]}"
    fi
done
