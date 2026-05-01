#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  data_curation/plaid/download_hotpot.sh [options]

Download HotpotQA from the official BEIR archive and normalize it into:
  dataset/hotpot/text/collection.tsv
  dataset/hotpot/text/queries.tsv
  dataset/hotpot/text/qrels.tsv
  dataset/hotpot/text/query_id_map.tsv
  dataset/hotpot/text/metadata.json

Options:
  --output-root <path>   Root that contains dataset directories. Default: ./dataset
  --download-dir <path>  Directory for downloaded/extracted archives. Default: <output-root>/_downloads/hotpot
  --qrels-split <name>   BEIR qrels split to normalize. Default: test
  --max-queries <count>  Query subset size. Default: 1000
  --overwrite            Replace an existing dataset/hotpot/text directory.
  --limit <count>        Debug option: only write this many collection rows.
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
qrels_split="test"
max_queries="1000"
overwrite=0
limit=""
url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip"

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
        --qrels-split)
            [[ $# -ge 2 ]] || die "missing value for --qrels-split"
            qrels_split="$2"
            shift 2
            ;;
        --max-queries)
            [[ $# -ge 2 ]] || die "missing value for --max-queries"
            max_queries="$2"
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

[[ "$max_queries" =~ ^[0-9]+$ ]] || die "--max-queries must be an integer"
[[ -z "$limit" || "$limit" =~ ^[0-9]+$ ]] || die "--limit must be an integer"

download_dir="${download_dir:-${output_root}/_downloads/hotpot}"
archive="${download_dir}/hotpotqa.zip"
extract_dir="${download_dir}/hotpotqa"
text_dir="${output_root}/hotpot/text"

if [[ ! -f "${extract_dir}/corpus.jsonl" ]]; then
    mkdir -p "$download_dir"
    if [[ ! -f "$archive" ]]; then
        echo "[hotpot] downloading ${url}"
        curl -L -f -o "$archive" "$url"
    fi
    echo "[hotpot] extracting ${archive}"
    rm -rf "$extract_dir"
    unzip -q "$archive" -d "$download_dir"
fi

cmd=(
    python "$repo_root/data_curation/download/normalize_official_text.py" hotpot
    --input-dir "$extract_dir"
    --output-dir "$text_dir"
    --qrels-split "$qrels_split"
    --max-queries "$max_queries"
)

if (( overwrite )); then
    cmd+=(--overwrite)
fi
if [[ -n "$limit" ]]; then
    cmd+=(--limit "$limit")
fi

echo "[hotpot] normalizing into ${text_dir}"
"${cmd[@]}"
