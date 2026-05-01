# Chimera

Chimera is the C++/CUDA source tree for building and searching the Chimera document-token retrieval index. Experiment launchers, dataset pipelines, and environment scripts live at the repository root in `experiment/`, `datasets/`, and `setup/`.

## Layout

- `cpp/app`: supported executable entry points
- `cpp/include`: shared public headers
- `cpp/src`: CPU-side implementation
- `cpp/cuda/include`: CUDA headers
- `cpp/cuda/src`: GPU index/search implementation and kernels
- `cpp/legacy`: retired versioned CPU/GPU implementations kept for reference
- `cpp/dep`: vendored header-only dependencies

Important current files:

- `CMakeLists.txt`
- `cpp/app/gpu_build.cpp`
- `cpp/app/gpu_search.cu`
- `cpp/app/cpu_search.cpp`
- `cpp/cuda/src/build_index.cu`
- `cpp/cuda/src/gpu_index.cu`
- `cpp/cuda/src/gpu_kernels.cu`
- `cpp/src/cpu_kernel.cpp`
- `cpp/include/doc_format.hpp`
- `cpp/include/clustered_format.hpp`

## Supported Targets

Default build targets:

- `gpu_build`: builds Chimera GPU index files.
- `gpu_search`: canonical GPU search binary.
- `gpu_search_nolut`: GPU search without LUT scoring.
- `gpu_search_nosum`: GPU search without sparse doc-score summation.
- `gpu_search_nolut_nosum`: combines no-LUT and no-sum behavior.
- `gpu_search_profile`: profiling build of the canonical GPU search path.
- `cpu_search`: canonical CPU search binary.

Legacy versioned targets such as `gpu_search_v6*`, `gpu_search_v7*`, `gpu_search_v9`, `cpu_search_v1`, and `cpu_search_v2` are gated behind `-DCHIMERA_ENABLE_LEGACY_TARGETS=ON` and source from `cpp/legacy`.

## Build

Use the root setup wrapper for a reproducible build:

```bash
setup/build_chimera.sh
```

Or configure directly:

```bash
cd Chimera

CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
CUDACXX="$CONDA_PREFIX/bin/nvcc" \
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

cmake --build build --target gpu_build gpu_search gpu_search_nolut gpu_search_nosum gpu_search_nolut_nosum cpu_search -j 4
```

CPU-only build:

```bash
cmake -S Chimera -B Chimera/build-cpu -G Ninja \
  -DCHIMERA_ENABLE_GPU=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build Chimera/build-cpu --target cpu_search -j 4
```

## Index Formats

The current index is a directory of separate files, not one monolithic binary.

`doc_format` covers the document-ordered quantized payloads:

- `doc_1bit.bin`
- `doc_4bit.bin`
- `doc_4bit_ex.bin`

`clustered_format` covers the cluster-ordered stage-1 payload:

- `cluster_1bit.bin`

The full GPU index directory also includes:

- `ivf.bin`
- `index_metadata.json`
- `centroids.carga`
- `centroids.hnsw`
- `doclens.bin`

## Build An Index

Use the root dataset pipeline:

```bash
datasets/chimera/build_index.sh --dataset lotte --n-clusters 2000000
```

Direct binary example:

```bash
Chimera/build/gpu_build \
  --index_dir dataset/lotte/gpu_search_2m \
  --doclens dataset/lotte/raw/doclens.bin \
  --data dataset/lotte/raw/embedding.bin \
  --n_clusters 2000000
```

`gpu_build` samples centroids, assigns token embeddings, builds IVF/CAGRA metadata, rotates and quantizes token embeddings, and writes the split index directory.

## Run Search

Direct canonical GPU search:

```bash
Chimera/build/gpu_search \
  --query dataset/lotte/raw/query.bin \
  --gt dataset/lotte/raw/gt.tsv \
  --index dataset/lotte/gpu_search_2m \
  --k 100 \
  --nq -1 \
  --warmup 5 \
  --config-file profiling/gpu_search_config.csv
```

Run the benchmark launcher:

```bash
experiment/chimera/bench_gpu_search.sh \
  --dataset lotte \
  --version gpu_search \
  --config-file profiling/gpu_search_config.csv
```

Run the main experiment and ablations:

```bash
experiment/chimera/run_gpu_search_main_experiment.py --build
experiment/ablation/run_gpu_search_lut_ablation.py
experiment/ablation/run_gpu_search_recall95_ablation.py --build
```

## Notes

- `Q_DOCLEN` in `cpp/cuda/include/gpu_config.cuh` must match the query token length in `query.bin`.
- The canonical GPU path is implemented by `gpu_index.cu` and `gpu_kernels.cu`; older `gpu_index_v*` and `gpu_kernels_v*` files live under `cpp/legacy`.
- The canonical CPU path is implemented by `cpu_kernel.cpp`; older `cpu_kernel_v*` files live under `cpp/legacy`.
- The historical dataset directory name `gpu_mvr_2m` is still accepted by the dataset and benchmark scripts for compatibility with existing generated indexes.
