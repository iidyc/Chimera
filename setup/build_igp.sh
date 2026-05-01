#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
source_dir="${repo_root}/IGP"
build_dir="${source_dir}/build"
target=""
cmake_args=()

usage() {
    cat <<'EOF'
Usage: setup/build_igp.sh [--build-dir <path>] [--target <name>] [-- <extra-cmake-args...>]

Configure and build IGP from the root-level IGP source tree.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 ]] || { echo "missing value for --build-dir" >&2; exit 1; }
            build_dir="$2"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || { echo "missing value for --target" >&2; exit 1; }
            target="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            cmake_args+=("$@")
            break
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release "${cmake_args[@]}"

build_args=(--build "${build_dir}" -j "$(nproc)")
if [[ -n "${target}" ]]; then
    build_args+=(--target "${target}")
fi
cmake "${build_args[@]}"
