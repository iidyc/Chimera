#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
source_dir="${repo_root}/gpu-mvr"
env_name="gpu-mvr"
cpu_only=0
target=""
build_dir=""
cmake_args=()

usage() {
    cat <<'EOF'
Usage: script/cmake.sh [--cpu-only] [--target <name>] [--build-dir <path>] [-- <extra-cmake-configure-args...>]

Options:
  --cpu-only          Configure with -DGPU_MVR_ENABLE_GPU=OFF and use build-cpu-only by default.
  --target <name>     Build only the specified target.
  --build-dir <path>  Override the build directory.
  --help, -h          Show this help.

Examples:
  script/cmake.sh
  script/cmake.sh --cpu-only
  script/cmake.sh --cpu-only --target cpu_search_v3
  script/cmake.sh -- -DCMAKE_CXX_COMPILER=clang++
EOF
}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available in PATH" >&2
    exit 1
fi

if [[ ! -f "${source_dir}/CMakeLists.txt" ]]; then
    echo "Could not find CMakeLists.txt under ${source_dir}" >&2
    exit 1
fi

conda_activate() {
    local hook=""
    if ! hook="$(conda shell.bash hook 2>/dev/null)"; then
        echo "failed to initialize conda shell integration" >&2
        exit 1
    fi
    eval "${hook}"
    conda activate "${env_name}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only)
            cpu_only=1
            shift
            ;;
        --target)
            [[ $# -ge 2 ]] || { echo "missing value for --target" >&2; exit 1; }
            target="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || { echo "missing value for --build-dir" >&2; exit 1; }
            build_dir="$2"
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

if [[ -z "${build_dir}" ]]; then
    if [[ "${cpu_only}" -eq 1 ]]; then
        build_dir="${source_dir}/build-cpu-only"
    else
        build_dir="${source_dir}/build"
    fi
fi

cmake_configure_args=(
    -S "${source_dir}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=Release
)

if [[ "${cpu_only}" -eq 1 ]]; then
    cmake_configure_args+=(-DGPU_MVR_ENABLE_GPU=OFF)
fi

cmake_configure_args+=("${cmake_args[@]}")

mkdir -p "${build_dir}"

echo "[cmake] source_dir=${source_dir}"
echo "[cmake] build_dir=${build_dir}"
echo "[cmake] conda_env=${env_name}"
echo "[cmake] build_type=Release"
if [[ "${cpu_only}" -eq 1 ]]; then
    echo "[cmake] gpu_support=OFF"
else
    echo "[cmake] gpu_support=ON"
fi
if [[ -n "${target}" ]]; then
    echo "[cmake] target=${target}"
fi

# conda_activate

cmake "${cmake_configure_args[@]}"

cmake_build_args=(
    --build "${build_dir}"
    -j "$(nproc)"
)

if [[ -n "${target}" ]]; then
    cmake_build_args+=(--target "${target}")
fi

cmake "${cmake_build_args[@]}"
