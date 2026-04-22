# GPU-MVR

`gpu-mvr` is a C++/CUDA project for building and searching a document retrieval index from token embeddings.

At a high level:

1. Token embeddings are grouped back into documents via `doclens.bin`.
2. A coarse IVF partition is built over token embeddings.
3. A CAGRA graph is built over centroids to accelerate cluster selection.
4. Token embeddings are quantized after rotation.
5. Search uses token-level scoring plus document-level aggregation on CPU or GPU.

This README is an initial architecture and operator guide. It starts with the current GPU build path (`gpu_build`) and gives a first-pass description of the other binaries under `gpu-mvr/cpp/app`. Some of the smaller utilities are still experimental or legacy.

## Repository Layout

Key directories:

- `cpp/app`: executable entry points
- `cpp/src`: core CPU-side implementation
- `cpp/cuda/src`: GPU index build/search implementation and CUDA kernels
- `cpp/include`: public/shared headers
- `cpp/cuda/include`: GPU-specific headers

Important files:

- [CMakeLists.txt](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/CMakeLists.txt)
- [cpp/app/gpu_build.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_build.cu)
- [cpp/app/gpu_search_v3.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_search_v3.cu)
- [cpp/cuda/src/build_gpu_index.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/cuda/src/build_gpu_index.cu)
- [cpp/cuda/src/gpu_index.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/cuda/src/gpu_index.cu)
- [cpp/cuda/src/gpu_index_v3.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/cuda/src/gpu_index_v3.cu)
- [cpp/src/ivf_pg.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/src/ivf_pg.cpp)
- [cpp/include/gpu_index_layout.hpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/include/gpu_index_layout.hpp)
- [cpp/include/mvr_index_file_format.hpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/include/mvr_index_file_format.hpp)

## Architecture

The codebase is organized around four layers:

- Data I/O
  Input readers in `cpp/src/io.cpp` load or memory-map embeddings, queries, and document lengths.
- Core index structures
  `cpp/src/index.cpp` and `cpp/src/ivf_pg.cpp` implement the CPU-side quantized index and IVF/CAGRA metadata structures.
- GPU implementations
  `cpp/cuda/src/build_gpu_index.cu`, `gpu_index.cu`, and the CUDA kernels implement the high-throughput build and search paths.
- App wrappers
  `cpp/app` contains small binaries that glue together file loading, index construction, search, and evaluation.

The current production-oriented path is:

`embeddings.bin` + `doclens.bin` -> `gpu_build` -> split index directory -> `gpu_search`

The split index directory contains:

- quantized payload plus embedded rotator
- IVF assignment lists and cluster offsets
- persisted centroid graph

## Data Model

The project works on token embeddings, not pre-aggregated document embeddings.

- `embeddings.bin`: all token embeddings
- `doclens.bin`: one integer length per document
- `query_embeddings.bin`: query token embeddings grouped by query

`doclens.bin` is used to reconstruct which token embeddings belong to which document.

## Input File Formats

Current binary formats used by the code:

- `embeddings.bin`
  - `int32 num_embeddings`
  - `int32 dim`
  - `num_embeddings * dim` floats
- `query_embeddings.bin`
  - `int32 num_queries`
  - `int32 q_doclen`
  - `int32 dim`
  - `num_queries * q_doclen * dim` floats
- `doclens.bin`
  - `int32 num_docs`
  - `num_docs` integers

These formats are implemented in [cpp/src/io.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/src/io.cpp).

## Index Layout

The current GPU-oriented index layout is split into four files inside one index directory:

- `ivf.bin`
- `cpu_index.bin`
- `gpu_index.bin`
- `centroids.carga`

This layout is defined in [cpp/include/gpu_index_layout.hpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/include/gpu_index_layout.hpp).

Notes:

- `cpu_index.bin` contains the index header, the embedded rotator, and the CPU-side quantized payload.
- `gpu_index.bin` contains the cluster-ordered one-bit GPU layout plus the original-token to cluster-position map used by `v3`.
- `ivf.bin` contains the IVF inverted list and cluster offsets.
- `ivf.bin` now stores vector IDs as `uint32_t`.
- `centroids.carga` stores the persisted centroid graph used by IVF-PG.
- The filename is currently spelled `centroids.carga` in code. That is the filename the current build/search path expects.

## Build Pipeline

The current recommended build flow is the GPU build path:

1. Sample `n_clusters` token embeddings as centroids.
2. Assign every token embedding to its nearest centroid with a temporary non-rotated CAGRA index.
3. Rotate centroids and build the persisted centroid CAGRA graph.
4. Build `IVF_PG` from the assignments.
5. Rotate and quantize all token embeddings.
6. Write the split index directory.

The implementation lives in:

- [cpp/cuda/src/build_gpu_index.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/cuda/src/build_gpu_index.cu)
- [cpp/include/build_gpu_index.hpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/include/build_gpu_index.hpp)

Recent behavior worth knowing:

- Step 2 uses double buffering per GPU worker to overlap data loading and device work.
- Step timings are printed as human-readable durations.
- Step 2 progress is reported as `% documents handled`.
- The supported end-to-end entry point is `gpu_build`; it prints per-phase timings directly to stdout.

## Search Pipeline

The main GPU search path is implemented in [cpp/cuda/src/gpu_index.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/cuda/src/gpu_index.cu).

At a high level:

1. Load the split index layout.
2. Read the embedded rotator from `cpu_index.bin`.
3. Rotate query token embeddings.
4. Use the centroid graph to pick clusters.
5. Score token candidates with one-bit or LUT-based approximate distance on GPU.
6. Aggregate token scores into document scores.
7. Re-rank a reduced candidate set.

There is also a `v0` comparison path in [cpp/cuda/src/gpu_index_baseline.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/cuda/src/gpu_index_baseline.cu).

## Building

Create and activate the Conda environment first:

```bash
conda env create -n gpu-mvr -y python=3.11
conda activate gpu-mvr
conda install -c rapidsai -c conda-forge libcuvs=26.04 'cuda-version=12.9'
conda install -c conda-forge cxx-compiler cmake ninja
```

Configure directly with CMake/Ninja. The important part is to point CMake at the Conda toolchain and CUDA toolkit:

```bash
cd gpu-mvr

CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
CUDACXX="$CONDA_PREFIX/bin/nvcc" \
cmake -S . -B build -G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

cmake --build build -j
```

If your shell exports a conflicting `CPATH` from another toolchain, unset it before configuring.

If you only need specific binaries:

```bash
cmake --build build --target gpu_build gpu_search gpu_search_v0 gpu_search_v1 gpu_search_v2 gpu_search_v3 build_cagra build_hnsw_from_cagra -j 4
```

This direct configure command replaces the removed `cmake.sh` wrapper.

## Binaries

Executables are defined in [CMakeLists.txt](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/CMakeLists.txt) and live in `build/` after compilation.

### `gpu_build`

Source: [cpp/app/gpu_build.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_build.cu)

Purpose:

- Build the current split GPU index from token embeddings and document lengths.

CLI:

```bash
./build/gpu_build \
  --index_dir <index_dir> \
  --doclens <doclens.bin> \
  --data <embeddings.bin> \
  --n_clusters <num_centroids>
```

Arguments:

- `--index_dir`: output directory for the built index
- `--doclens`: path to `doclens.bin`
- `--data`: path to `embeddings.bin`
- `--n_clusters`: number of sampled centroids for IVF/CAGRA build

Output:

- `<index_dir>/ivf.bin`
- `<index_dir>/doc_1bit.bin`
- `<index_dir>/doc_4bit.bin`
- `<index_dir>/doc_4bit_ex.bin`
- `<index_dir>/cluster_1bit.bin`
- `<index_dir>/index_metadata.json`
- `<index_dir>/centroids.carga`
- `<index_dir>/centroids.hnsw`

Example:

```bash
./build/gpu_build \
  --index_dir ./my_index \
  --doclens ./doclens.bin \
  --data ./embeddings.bin \
  --n_clusters 2097152
```

Operational notes:

- This is the primary build binary for the current GPU index format.
- It memory-maps the embedding file instead of loading the whole dataset eagerly.
- It currently fixes `ex_bits = 3` in the app wrapper.
- It also writes a separate `doc_4bit_ex.bin` residual sidecar with `ex_bits = 4`.
- The `hnswlib` centroid graph is built directly from the in-memory rotated centroids during the build; it does not need to round-trip through `centroids.carga`.
- Within each cluster, `cluster_1bit.bin` already follows original token order, which is equivalent to doc-id then token-id order for natural document packing.

### `gpu_search`

Source: [cpp/app/gpu_search.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_search.cu)

Purpose:

- Run the main GPU search pipeline and compute recall against a TSV ground-truth file.

CLI:

```bash
./build/gpu_search \
  --query <query_embeddings.bin> \
  --doclens <doclens.bin> \
  --gt <groundtruth.tsv> \
  --index <index_dir_or_cpu_index.bin> \
  [--k <top_k>] \
  [--nprobe <num_probes>] \
  [--k-rank-cluster <count>] \
  [--k-rank-all-tokens <count>] \
  [--itopk-size <count>] \
  [--overlap-chunks <count>] \
  [--nq <num_queries_to_run>] \
  [--warmup <num_warmup_queries>]
```

Notes:

- `Q_DOCLEN` in `gpu_config.cuh` must match the `q_doclen` stored in `query_embeddings.bin`.
- `--nq` and `--warmup` are clamped to the number of queries in the query file.
- `gpu_search` is the default `v3` build target; [cpp/app/gpu_search.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_search.cu) simply includes `gpu_search_v3.cu`.
- The GPU search binaries share one standardized CLI parser, so `gpu_search`, `gpu_search_v0`, `gpu_search_v1`, `gpu_search_v2`, and `gpu_search_v3` accept the same search-tuning flags.
- Default values still differ by binary where the code previously differed, especially `--k-rank-cluster`.
- `--itopk-size` controls CAGRA's internal intermediate top-k during centroid search.
- `--overlap-chunks` is used by the persistent overlap pipeline in `v1`/`v2`/`v3` and is ignored by `v0`.
- For `gpu_search`/`gpu_search_v3`, `--index` must resolve to a split index directory: pass either the directory itself or its `cpu_index.bin`, and keep `gpu_index.bin` beside it. Older directories can still fall back to `clustered_stage1.bin`.

### `gpu_search_v3`

Source: [cpp/app/gpu_search_v3.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_search_v3.cu)

Purpose:

- Run the explicit `v3` GPU search binary. This is the same implementation as `gpu_search`, but exposed as a versioned executable for comparisons and artifact runs.

CLI:

```bash
./build/gpu_search_v3 \
  --query <query_embeddings.bin> \
  --doclens <doclens.bin> \
  --gt <groundtruth.tsv> \
  --index <index_dir_or_cpu_index.bin> \
  [--k <top_k>] \
  [--nprobe <num_probes>] \
  [--k-rank-cluster <count>] \
  [--k-rank-all-tokens <count>] \
  [--itopk-size <count>] \
  [--overlap-chunks <count>] \
  [--nq <num_queries_to_run>] \
  [--warmup <num_warmup_queries>]
```

Notes:

- `gpu_search_v3` uses the same parser and defaults as `gpu_search`.
- The v3 loader requires the split GPU layout because stage 1 and stage 2 both read `gpu_index.bin`.
- Passing `--index <index_dir>` or `--index <index_dir>/cpu_index.bin` is supported. Passing only a monolithic legacy index file is not supported for `v3`.

### `gpu_search_v0`

Source: [cpp/app/gpu_search_baseline.cu](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/gpu_search_baseline.cu)

Purpose:

- Run the baseline GPU search implementation for comparison against the main GPU path.

CLI:

```bash
./build/gpu_search_v0 \
  --query <query_embeddings.bin> \
  --doclens <doclens.bin> \
  --gt <groundtruth.tsv> \
  --index <index_dir_or_cpu_index.bin> \
  [--k <top_k>] \
  [--nprobe <num_probes>] \
  [--k-rank-cluster <count>] \
  [--k-rank-all-tokens <count>] \
  [--itopk-size <count>] \
  [--overlap-chunks <count>] \
  [--nq <num_queries_to_run>] \
  [--warmup <num_warmup_queries>]
```

Notes:

- This is the comparison/baseline GPU path, not the main optimized one.
- Like `gpu_search`, it requires `Q_DOCLEN` to match the query file.

### `search`

Source: [cpp/app/search.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/search.cpp)

Purpose:

- Run the CPU search path over the current split-format index.

CLI:

```bash
./build/search \
  --query <query_embeddings.bin> \
  --doclens <doclens.bin> \
  --gt <groundtruth.tsv> \
  --index <index_dir_or_cpu_index.bin> \
  [--k <top_k>] \
  [--nprobe <num_probes>] \
  [--nq <num_queries_to_run>]
```

Notes:

- This is the CPU search path.
- It accepts the same split index entry point as the GPU loaders.

### `build`

Source: [cpp/app/build.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/build.cpp)

Purpose:

- Legacy CPU build utility.

CLI:

```bash
./build/build \
  --data <embeddings.bin> \
  --output <index_file> \
  --n_clusters <num_centroids> \
  [--ex_bits <ex_bits>]
```

Notes:

- This builds the older monolithic CPU index format.
- It is not the recommended path for the current split GPU index layout.

### `build_cagra`

Source: [cpp/app/build_cagra.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/build_cagra.cpp)

Purpose:

- Small utility to build only a persisted CAGRA graph from `centroids.bin`.

CLI:

```bash
./build/build_cagra \
  --centroids <centroids.bin> \
  --output <graph_prefix>
```

Notes:

- This writes `<graph_prefix>.cagra`.
- It is a helper utility, not the main end-to-end build path.

### `build_hnsw_from_cagra`

Source: [cpp/app/build_hnsw_from_cagra.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/build_hnsw_from_cagra.cpp)

Purpose:

- Deserialize a persisted `centroids.carga` CAGRA graph with cuVS.
- Read the centroid embeddings stored in that index.
- Rebuild an `hnswlib` HNSW index from those centroid embeddings.

CLI:

```bash
./build/build_hnsw_from_cagra \
  --input <index_dir>/centroids.carga \
  --output <index_dir>/centroids.hnsw \
  [--M 16] \
  [--ef-construction 500] \
  [--batch-rows 65536]
```

Notes:

- `gpu_build` now emits `centroids.hnsw` directly as part of the normal build path.
- This helper still exists for rebuilding the HNSW artifact from an existing `centroids.carga`.

### `test_gather`

Source: [cpp/app/test_gather.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/test_gather.cpp)

Purpose:

- Compare candidate gathering strategies and write recall statistics to `gather_recall.csv`.

CLI:

```bash
./build/test_gather \
  --query <query_embeddings.bin> \
  --doclens <doclens.bin> \
  --gt <groundtruth.tsv> \
  --index <index_dir_or_cpu_index.bin> \
  --output <gather_recall.csv> \
  [--k <top_k>] \
  [--nq <num_queries_to_run>] \
  [--nprobe <num_probes>]
```

Notes:

- This is a CPU-side analysis utility.
- It writes one CSV row per tested candidate-gathering setting.

### `test_dist_decomp`

Source: [cpp/app/test_dist_decomp.cpp](/data/juelin/gpu-mvr/gpu-mvr/gpu-mvr/cpp/app/test_dist_decomp.cpp)

Purpose:

- Compare estimated distances against brute-force document scores and write `dist_decomp.csv`.

CLI:

```bash
./build/test_dist_decomp \
  --data <embeddings.bin> \
  --query <query_embeddings.bin> \
  --doclens <doclens.bin> \
  --index <index_dir_or_cpu_index.bin> \
  --output <dist_decomp.csv> \
  [--nq <num_queries_to_run>] \
  [--nprobe <num_probes>] \
  [--top_k_out <num_rows_per_query>]
```

Notes:

- This is a CPU-side analysis utility.
- It compares approximate document scores against brute-force scores over the raw token embeddings.

## Recommended Starting Point

If you are new to the repo, start here:

1. Configure and build with the direct CMake/Ninja commands above.
2. Use `gpu_build` to generate a split-format index directory.
3. Use `gpu_search_v3` for the explicit `v3` path, or `gpu_search` if you want the default alias.

Example:

```bash
cd gpu-mvr

CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" \
CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++" \
CUDACXX="$CONDA_PREFIX/bin/nvcc" \
cmake -S . -B build -G Ninja \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_PREFIX_PATH="$CONDA_PREFIX"

cmake --build build --target gpu_build gpu_search_v3 -j 4

./build/gpu_build \
  --index_dir ./my_index \
  --doclens ./doclens.bin \
  --data ./embeddings.bin \
  --n_clusters 2097152

./build/gpu_search_v3 \
  --query ./query_embeddings.bin \
  --doclens ./doclens.bin \
  --gt ./lotte-groundtruth-top1000--.tsv \
  --index ./my_index
```

## Notes And Follow-Up

This document is intentionally strongest on the current GPU build/search path.

Areas still worth refining later:

- document `gpu_config.cuh` tuning knobs
- describe the stage-1 and stage-2 GPU kernels in more detail
- add a dataset preparation section
