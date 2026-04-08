#!/usr/bin/env bash
set -euo pipefail

rm -rf CMakeFiles
rm CMakeCache.txt
cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=../cmake/conda-toolchain.cmake ..