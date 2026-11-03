#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?worker allocation required}"
root=/scratch3/workspace/juelinliu_umass_edu-gpu-mvr/GIGP-plus
toolchain=/home/juelinliu_umass_edu/conda/envs/IGP
restart="${SLURM_RESTART_COUNT:-0}"
build="/tmp/gigpp_native_${SLURM_JOB_ID}_r${restart}"
entry="/tmp/gigpp_build_${SLURM_JOB_ID}"
[[ ! -e "$build" ]]
[[ ! -e "$entry" || -L "$entry" ]]
export PATH="$toolchain/bin:$PATH"
export LD_LIBRARY_PATH="$toolchain/lib:${LD_LIBRARY_PATH:-}"
export CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDAHOSTCXX=/usr/bin/g++-13
[[ -x "$CXX" ]]
cmake -S "$root/upstream/benchmark" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_COMPILER="$toolchain/bin/nvcc" \
  -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_ARCHITECTURES="${GIGP_CUDA_ARCH:-native}" \
  -DCMAKE_PREFIX_PATH="$toolchain" -DPython_EXECUTABLE="$toolchain/bin/python" \
  -DGIGP_CPU_ARCH=native -DGIGP_RAFT_TEMPLATE_REDUCE=ON
cmake --build "$build" --target IGPGPUPP raft_reduce_check --parallel 4
if [[ "${GIGP_CUDA_ARCH:-native}" == native ]]; then
  "$build/raft_reduce_check"
fi
metadata="$root/results/build_metadata/job${SLURM_JOB_ID}"
mkdir -p "$metadata"
cp "$build/CMakeCache.txt" "$metadata/CMakeCache.txt"
cp "$build/build.ninja" "$metadata/build.ninja"
mkdir -p "$metadata/restart${restart}"
cp "$build/CMakeCache.txt" "$metadata/restart${restart}/CMakeCache.txt"
ln -sfn "$build" "$entry"
grep -E 'CMAKE_CUDA_ARCHITECTURES:|GIGP_CPU_ARCH:|CMAKE_BUILD_TYPE:' "$build/CMakeCache.txt"
sha256sum "$build/IGPGPUPP"
"$CXX" --version
"$toolchain/bin/nvcc" --version
git -C "$root/upstream" rev-parse HEAD
git -C "$root/upstream" diff --binary | sha256sum
sha256sum "$root/upstream/benchmark/CMakeLists.txt"
