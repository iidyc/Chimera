#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  data_curation/create_raw_binaries.sh --dataset <name|all> [options]

Convert normalized raw text files into the binary files consumed by experiments:
  dataset/<name>/raw/data.bin
  dataset/<name>/raw/doclens.bin
  dataset/<name>/raw/query.bin
  dataset/<name>/raw/gt.tsv

Inputs:
  dataset/<name>/text/collection.tsv
  dataset/<name>/text/queries.tsv

Options:
  --dataset <name|all>      Dataset to process. Repeatable. Default: all.
  --output-root <path>      Root containing dataset directories. Default: ./dataset
  --checkpoint <name/path>  ColBERT checkpoint. Default: colbert-ir/colbertv2.0
  --gpus <list|auto>        CUDA devices for document encoding. Default: auto.
                            Example: --gpus 0,1,2,3
  --chunk-size <count>      Collection rows encoded per chunk. Default: 25000
  --topk <count>            Ground-truth depth. Default: 1000
  --doc-batch-size <count>  Docs per brute-force top-k batch. Default: 50000
  --skip-gt                 Do not generate gt.tsv.
  --skip-existing           Skip outputs that already exist.
  --overwrite               Replace existing raw outputs.
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
topk="1000"
doc_batch_size="50000"
skip_gt=0
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
        --skip-gt)
            skip_gt=1
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
[[ "$topk" =~ ^[0-9]+$ && "$topk" -gt 0 ]] || die "--topk must be a positive integer"
[[ "$chunk_size" =~ ^[0-9]+$ && "$chunk_size" -gt 0 ]] || die "--chunk-size must be a positive integer"
[[ "$doc_batch_size" =~ ^[0-9]+$ && "$doc_batch_size" -gt 0 ]] || die "--doc-batch-size must be a positive integer"
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

maybe_remove() {
    local path="$1"
    if [[ -e "$path" && $overwrite -eq 1 ]]; then
        if (( dry_run )); then
            printf '[dry-run] rm -f %q\n' "$path"
        else
            rm -f "$path"
        fi
    fi
}

resolve_gpu_list() {
    local requested="$1"
    if [[ "$requested" != "auto" ]]; then
        tr ',' '\n' <<<"$requested" | awk 'NF {print $1}'
        return
    fi

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        tr ',' '\n' <<<"$CUDA_VISIBLE_DEVICES" | awk 'NF {print $1}'
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local detected
        detected="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)"
        if [[ -n "$detected" ]]; then
            tr ' ' '\n' <<<"$detected" | awk 'NF {print $1}'
            return
        fi
    fi

    printf '%s\n' "0"
}

encode_documents() {
    local dataset="$1"
    local collection="$2"
    local data_bin="$3"
    local doclens_bin="$4"
    local raw_dir="$5"

    mapfile -t gpu_list < <(resolve_gpu_list "$gpus")
    if [[ ${#gpu_list[@]} -le 1 ]]; then
        run_cmd python "$repo_root/data_curation/plaid/encode_dataset.py" \
            "$collection" "$data_bin" "$doclens_bin" \
            --checkpoint "$checkpoint" \
            --chunk-size "$chunk_size" \
            --cuda-device "${gpu_list[0]}"
        return
    fi

    local total_docs
    total_docs="$(wc -l < "$collection")"
    [[ "$total_docs" =~ ^[0-9]+$ && "$total_docs" -gt 0 ]] || die "empty collection: $collection"

    local shard_dir="${raw_dir}/embedding_shards"
    run_cmd mkdir -p "$shard_dir"

    local data_shards=()
    local doclens_shards=()
    local pids=()
    local shard_id=0
    local gpu start end data_shard doclens_shard
    for gpu in "${gpu_list[@]}"; do
        start=$(( total_docs * shard_id / ${#gpu_list[@]} ))
        end=$(( total_docs * (shard_id + 1) / ${#gpu_list[@]} ))
        if (( end <= start )); then
            shard_id=$(( shard_id + 1 ))
            continue
        fi

        data_shard="${shard_dir}/data.shard${shard_id}.bin"
        doclens_shard="${shard_dir}/doclens.shard${shard_id}.bin"
        data_shards+=("$data_shard")
        doclens_shards+=("$doclens_shard")

        if (( dry_run )); then
            printf '[dry-run] CUDA_VISIBLE_DEVICES=%q python %q %q %q %q --checkpoint %q --chunk-size %q --start-doc %q --end-doc %q\n' \
                "$gpu" "$repo_root/data_curation/plaid/encode_dataset.py" \
                "$collection" "$data_shard" "$doclens_shard" \
                "$checkpoint" "$chunk_size" "$start" "$end"
        else
            (
                export CUDA_VISIBLE_DEVICES="$gpu"
                python "$repo_root/data_curation/plaid/encode_dataset.py" \
                    "$collection" "$data_shard" "$doclens_shard" \
                    --checkpoint "$checkpoint" \
                    --chunk-size "$chunk_size" \
                    --start-doc "$start" \
                    --end-doc "$end"
            ) &
            pids+=("$!")
        fi
        shard_id=$(( shard_id + 1 ))
    done

    if (( ! dry_run )); then
        local pid
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
    fi

    run_cmd python "$repo_root/data_curation/plaid/merge_embedding_shards.py" \
        --data-shards "${data_shards[@]}" \
        --doclens-shards "${doclens_shards[@]}" \
        --output-data "$data_bin" \
        --output-doclens "$doclens_bin"
}

ensure_not_blocked() {
    local path="$1"
    if [[ -e "$path" && $overwrite -eq 0 && $skip_existing -eq 0 ]]; then
        die "output exists: $path (pass --overwrite or --skip-existing)"
    fi
}

for dataset in "${datasets[@]}"; do
    dataset_dir="${output_root}/${dataset}"
    text_dir="${dataset_dir}/text"
    raw_dir="${dataset_dir}/raw"
    collection="${text_dir}/collection.tsv"
    queries="${text_dir}/queries.tsv"
    data_bin="${raw_dir}/data.bin"
    doclens_bin="${raw_dir}/doclens.bin"
    query_bin="${raw_dir}/query.bin"
    gt_tsv="${raw_dir}/gt.tsv"

    [[ -f "$collection" ]] || die "missing collection text: $collection"
    [[ -f "$queries" ]] || die "missing query text: $queries"

    run_cmd mkdir -p "$raw_dir"

    if [[ -f "$data_bin" && -f "$doclens_bin" && $skip_existing -eq 1 ]]; then
        echo "[${dataset}] skipping existing data.bin/doclens.bin"
    else
        ensure_not_blocked "$data_bin"
        ensure_not_blocked "$doclens_bin"
        maybe_remove "$data_bin"
        maybe_remove "$doclens_bin"
        encode_documents "$dataset" "$collection" "$data_bin" "$doclens_bin" "$raw_dir"
    fi

    if [[ -f "$query_bin" && $skip_existing -eq 1 ]]; then
        echo "[${dataset}] skipping existing query.bin"
    else
        ensure_not_blocked "$query_bin"
        maybe_remove "$query_bin"
        run_cmd python "$repo_root/data_curation/plaid/encode_queries.py" \
            "$queries" "$query_bin" \
            --checkpoint "$checkpoint"
    fi

    if (( skip_gt )); then
        echo "[${dataset}] skipping gt.tsv generation"
    elif [[ -f "$gt_tsv" && $skip_existing -eq 1 ]]; then
        echo "[${dataset}] skipping existing gt.tsv"
    else
        ensure_not_blocked "$gt_tsv"
        maybe_remove "$gt_tsv"
        run_cmd python "$repo_root/data_curation/plaid/compute_topk.py" \
            "$data_bin" "$doclens_bin" "$query_bin" "$gt_tsv" \
            --topk "$topk" \
            --doc_batch_size "$doc_batch_size"
    fi
done
