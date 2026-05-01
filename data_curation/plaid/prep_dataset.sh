#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  data_curation/plaid/prep_dataset.sh [options]

Download official raw text inputs for PLAID/Chimera experiments.

Options:
  --output-root <path>  Root that contains dataset directories. Default: ./dataset
  --overwrite           Replace existing text outputs.
  --limit <count>       Debug option passed to download scripts.
  -h, --help            Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
output_root="${repo_root}/dataset"
overwrite=0
limit=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-root)
            [[ $# -ge 2 ]] || die "missing value for --output-root"
            output_root="$2"
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

common_args=(--output-root "$output_root")
if (( overwrite )); then
    common_args+=(--overwrite)
fi
if [[ -n "$limit" ]]; then
    common_args+=(--limit "$limit")
fi

"${script_dir}/download_msmarco.sh" "${common_args[@]}"
"${script_dir}/download_hotpot.sh" "${common_args[@]}"
