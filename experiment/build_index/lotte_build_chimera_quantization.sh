#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/build_chimera_quantization.sh" --dataset lotte "$@"
