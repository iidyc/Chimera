#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
colbert_dir="${repo_root}/ColBERT"

usage() {
    cat <<'EOF'
Usage: setup/build_plaid.sh

Install the local ColBERT/PLAID checkout in editable mode. Run
setup/setup_plaid_env.sh first when creating a fresh environment.
EOF
}

if [[ "${1-}" == "--help" || "${1-}" == "-h" ]]; then
    usage
    exit 0
fi

python -m pip install -e "${colbert_dir}" --no-deps
