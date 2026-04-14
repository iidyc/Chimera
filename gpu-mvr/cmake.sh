#!/usr/bin/env bash
set -euo pipefail
rm -rf build

PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/targets/x86_64-linux/bin:$PATH"
unset CPATH

CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
CUDACXX="$CONDA_PREFIX/bin/nvcc" \
cmake -G Ninja -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

cmake --build build -j