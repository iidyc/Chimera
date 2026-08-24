#!/usr/bin/env bash

set -euo pipefail

command -v conda >/dev/null 2>&1 || {
    echo "error: conda is not available on PATH" >&2
    exit 1
}

conda create -n chimera -y \
    -c rapidsai \
    -c conda-forge \
    "python=3.11" \
    numpy \
    "libcuvs=26.04" \
    "librmm=26.04" \
    "cuda-version=12.9" \
    "cuda-nvcc-impl=12.9" \
    cxx-compiler \
    cmake \
    ninja \
    "pybind11>=2.12"
