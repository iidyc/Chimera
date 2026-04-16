#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/setup_colbert_env.sh [--env-name <name>] [--python <version>] [--verify] [--dry-run]

Description:
  Create or refresh an isolated GPU-enabled ColBERT environment for this artifact.

  This script intentionally does NOT use ColBERT/conda_env.yml directly.
  Instead it installs the exact stack validated on this machine:

  - Python 3.10
  - CUDA toolkit 11.7 inside the env
  - gcc_linux-64=11 and gxx_linux-64=11
  - numpy<2
  - setuptools<81
  - torch==1.13.1
  - faiss-gpu==1.7.2
  - transformers==4.35.2

  The local ColBERT checkout is installed in editable mode with --no-deps so
  those pins are not overridden by newer transitive dependencies.

  The script also writes conda activation hooks so activating the env sets:
  - CUDA_HOME=$CONDA_PREFIX
  - LD_LIBRARY_PATH includes $CONDA_PREFIX/lib and $CONDA_PREFIX/lib64

  The script prefers micromamba, then mamba, then conda.

Options:
  --env-name <name>   Environment name. Default: colbert
  --python <version>  Python version for the new env. Default: 3.10
  --verify            Run post-install import checks.
  --dry-run           Print commands without executing them.
  -h, --help          Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

run_cmd() {
    echo "+ $(printf '%q ' "$@")"
    if [[ $dry_run -eq 0 ]]; then
        "$@"
    fi
}

find_env_manager() {
    if command -v micromamba >/dev/null 2>&1; then
        echo "micromamba"
        return
    fi
    if command -v mamba >/dev/null 2>&1; then
        echo "mamba"
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        echo "conda"
        return
    fi
    echo ""
}

env_exists() {
    if [[ $dry_run -eq 1 ]]; then
        return 1
    fi

    [[ -n "$(resolve_env_prefix_from_list)" ]]
}

resolve_env_prefix_from_list() {
    "$env_manager" env list 2>/dev/null | awk -v env="${env_name}" '
        $1 == env || $2 == env { print $NF }
    ' | tail -n 1
}

resolve_env_prefix() {
    local env_prefix
    env_prefix="$(resolve_env_prefix_from_list)"
    [[ -n "$env_prefix" ]] || die "failed to resolve prefix for env ${env_name}"
    echo "$env_prefix"
}

resolve_env_binary() {
    local env_prefix="$1"
    local binary_name="$2"
    local binary_path="${env_prefix}/bin/${binary_name}"

    if [[ $dry_run -eq 0 && ! -x "$binary_path" ]]; then
        die "missing executable ${binary_path}"
    fi

    echo "$binary_path"
}

run_in_env() {
    local env_prefix="$1"
    local binary_name="$2"
    shift 2

    local binary_path
    binary_path="$(resolve_env_binary "$env_prefix" "$binary_name")"

    run_cmd env \
        "PATH=${env_prefix}/bin:${PATH}" \
        "$binary_path" \
        "$@"
}

run_in_env_with_cuda() {
    local env_prefix="$1"
    local binary_name="$2"
    shift 2

    local binary_path
    local ld_library_path="${env_prefix}/lib:${env_prefix}/lib64"
    if [[ -n "${LD_LIBRARY_PATH-}" ]]; then
        ld_library_path="${ld_library_path}:${LD_LIBRARY_PATH}"
    fi
    binary_path="$(resolve_env_binary "$env_prefix" "$binary_name")"

    run_cmd env \
        "CUDA_HOME=${env_prefix}" \
        "PATH=${env_prefix}/bin:${PATH}" \
        "LD_LIBRARY_PATH=${ld_library_path}" \
        "$binary_path" \
        "$@"
}

install_cuda_activation_hooks() {
    local env_prefix="$1"
    local activate_dir="${env_prefix}/etc/conda/activate.d"
    local deactivate_dir="${env_prefix}/etc/conda/deactivate.d"
    local activate_hook="${activate_dir}/colbert_cuda.sh"
    local deactivate_hook="${deactivate_dir}/colbert_cuda.sh"

    echo "+ install CUDA activation hooks in ${env_prefix}/etc/conda"
    if [[ $dry_run -eq 1 ]]; then
        return
    fi

    mkdir -p "$activate_dir" "$deactivate_dir"

    cat > "$activate_hook" <<'EOF'
#!/usr/bin/env bash

export _COLBERT_OLD_CUDA_HOME="${CUDA_HOME-}"
export _COLBERT_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"

export CUDA_HOME="$CONDA_PREFIX"

if [[ -n "${LD_LIBRARY_PATH-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64"
fi
EOF

    cat > "$deactivate_hook" <<'EOF'
#!/usr/bin/env bash

if [[ -n "${_COLBERT_OLD_CUDA_HOME+x}" ]]; then
    if [[ -n "$_COLBERT_OLD_CUDA_HOME" ]]; then
        export CUDA_HOME="$_COLBERT_OLD_CUDA_HOME"
    else
        unset CUDA_HOME
    fi
    unset _COLBERT_OLD_CUDA_HOME
else
    unset CUDA_HOME
fi

if [[ -n "${_COLBERT_OLD_LD_LIBRARY_PATH+x}" ]]; then
    if [[ -n "$_COLBERT_OLD_LD_LIBRARY_PATH" ]]; then
        export LD_LIBRARY_PATH="$_COLBERT_OLD_LD_LIBRARY_PATH"
    else
        unset LD_LIBRARY_PATH
    fi
    unset _COLBERT_OLD_LD_LIBRARY_PATH
fi
EOF

    chmod +x "$activate_hook" "$deactivate_hook"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

env_name="colbert"
python_version="3.10"
verify=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-name)
            [[ $# -ge 2 ]] || die "missing value for --env-name"
            env_name="$2"
            shift 2
            ;;
        --python)
            [[ $# -ge 2 ]] || die "missing value for --python"
            python_version="$2"
            shift 2
            ;;
        --verify)
            verify=1
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
            die "unknown option: $1"
            ;;
    esac
done

env_manager="$(find_env_manager)"
if [[ -z "$env_manager" ]]; then
    if [[ $dry_run -eq 1 ]]; then
        env_manager="micromamba"
    else
        die "none of micromamba, mamba, or conda is available on PATH"
    fi
fi

[[ -f "${repo_root}/ColBERT/setup.py" ]] || die "missing local ColBERT checkout"

conda_toolchain_deps=(
    "cuda-toolkit=11.7"
    "gcc_linux-64=11"
    "gxx_linux-64=11"
    "ninja"
)

pip_deps=(
    "numpy<2"
    "setuptools<81"
    "wheel"
    "torch==1.13.1"
    "faiss-gpu==1.7.2"
    "transformers==4.35.2"
    "bitarray"
    "datasets"
    "gitpython"
    "python-dotenv"
    "scipy"
    "tqdm"
    "ujson"
    "flask"
)

if env_exists; then
    echo "Environment ${env_name} already exists; reusing it."
else
    run_cmd "$env_manager" create -n "$env_name" -y "python=${python_version}" pip
fi

if [[ $dry_run -eq 1 ]]; then
    env_prefix="<resolved-env-prefix>"
else
    env_prefix="$(resolve_env_prefix)"
fi

run_cmd "$env_manager" install -n "$env_name" -y \
    -c "nvidia/label/cuda-11.7.0" \
    -c "conda-forge" \
    "${conda_toolchain_deps[@]}"

run_in_env "$env_prefix" python -m pip install --upgrade "pip<25"
run_in_env "$env_prefix" python -m pip install "${pip_deps[@]}"
run_in_env "$env_prefix" python -m pip install -e "${repo_root}/ColBERT" --no-deps

install_cuda_activation_hooks "$env_prefix"

if [[ $verify -eq 1 ]]; then
    run_in_env_with_cuda "$env_prefix" python --version
    run_in_env_with_cuda "$env_prefix" nvcc --version
    run_in_env_with_cuda "$env_prefix" python -c "import os, pkg_resources, numpy, torch, faiss, transformers, torch.utils.cpp_extension as ce; print('env_prefix', os.environ.get('CUDA_HOME')); print('numpy', numpy.__version__); print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('transformers', transformers.__version__); print('detected_cuda_home', ce.CUDA_HOME); print('faiss imported'); print('pkg_resources imported')"
    run_in_env_with_cuda "$env_prefix" python -c "from colbert import Searcher; print('colbert Searcher import ok')"
fi

echo
echo "Environment setup script finished for ${env_name} using ${env_manager}."
echo "To use it:"
if [[ "$env_manager" == "micromamba" ]]; then
    echo "  micromamba activate ${env_name}"
else
    echo "  conda activate ${env_name}"
fi
echo "After activation, CUDA_HOME will point to the env prefix automatically."
