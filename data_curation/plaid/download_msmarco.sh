#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  data_curation/plaid/download_msmarco.sh [options]

Download MS MARCO passage collection/query files from the official Microsoft
archive and normalize them into:
  dataset/msmarco/text/collection.tsv
  dataset/msmarco/text/queries.tsv
  dataset/msmarco/text/qrels.tsv
  dataset/msmarco/text/query_id_map.tsv
  dataset/msmarco/text/metadata.json

Options:
  --output-root <path>   Root that contains dataset directories. Default: ./dataset
  --download-dir <path>  Directory for downloaded/extracted archives. Default: <output-root>/_downloads/msmarco
  --overwrite            Replace an existing dataset/msmarco/text directory.
  --limit <count>        Debug option: only write this many collection/query rows.
  -h, --help             Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
output_root="${repo_root}/dataset"
download_dir=""
overwrite=0
limit=""
url="https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            [[ $# -ge 2 ]] || die "missing value for --output-root"
            output_root="$2"
            shift 2
            ;;
        --download-dir)
            [[ $# -ge 2 ]] || die "missing value for --download-dir"
            download_dir="$2"
            shift 2
            ;;
        --overwrite)
            overwrite=1
            shift
            ;;
        --limit)
            [[ $# -ge 2 ]] || die "missing value for --limit"
            limit="$2"
            shift 2
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

[[ -z "$limit" || "$limit" =~ ^[0-9]+$ ]] || die "--limit must be an integer"

download_dir="${download_dir:-${output_root}/_downloads/msmarco}"
archive="${download_dir}/collectionandqueries.tar.gz"
text_dir="${output_root}/msmarco/text"

if [[ ! -f "${download_dir}/collection.tsv" || ! -f "${download_dir}/queries.dev.small.tsv" ]]; then
    mkdir -p "$download_dir"
    if [[ ! -f "$archive" ]]; then
        echo "[msmarco] downloading ${url}"
        curl -L -f -o "$archive" "$url"
    fi
    echo "[msmarco] extracting ${archive}"
    tar -xzf "$archive" -C "$download_dir"
fi

cmd=(
    python "$repo_root/data_curation/download/normalize_official_text.py" msmarco
    --input-dir "$download_dir"
    --output-dir "$text_dir"
)

if (( overwrite )); then
    cmd+=(--overwrite)
fi
if [[ -n "$limit" ]]; then
    cmd+=(--limit "$limit")
fi

echo "[msmarco] normalizing into ${text_dir}"
"${cmd[@]}"
