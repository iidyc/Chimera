# GIGP+ comparison on A100 (Table 4)

From the artifact root, run:

```sh
python3 experiment/gigp_plus/reproduce_table4.py
```

This uses the saved A100 frontier and requires only Python 3. It picks the
highest-throughput configuration at or above each Recall@100 target from
verified three-repeat full runs with 4-bit indexes and three queries in flight.
The output gives achieved recall, QPS, mean and percentile latencies in ms, and
speedup. The tested HotpotQA GIGP+ grid has no qualifying point at 0.95.

## Rerun the benchmark

The worker scripts come from the October 25 A100 reproduction snapshot. The
measured setup used an A100 80 GB and Xeon Platinum 8352Y, with 32 allocated
physical cores, three queries in flight, CUDA 12.9, GCC 13, RAFT 26.04 for GIGP+,
and cuVS for Chimera. Use Linux, Slurm, and numactl. The scripts contain the
original cluster paths. Set the workspace and environment paths in
`job.sh`, `build_worker.sh`, `prepare_job.sh`, and `chimera_job.sh` for your
installation. The GIGP+ build dependencies are listed in
`benchmark/CMakeLists.txt`.

1. Copy `benchmark/` into the benchmark workspace. Clone
   [the official GIGP+ repository](https://github.com/DBGroup-SUSTech/multi-vector-retrieval)
   into its `upstream/` subdirectory and check out
   `1899e9316e71b023518f9090fc64fe66dbd0c202`. From that checkout, apply
   `gigp-adapter.patch` with `git apply`, copy `RaftReductionCompat.hpp` into
   `src/include/alg/igp_gpu_pp/`, and copy the packaged `benchmark/` into
   the upstream checkout as well.
2. Decompress `chimera-three-slot-source.tar.gz` into the workspace as
   `paired/source_pinned_slots3.tar` and extract that tar into
   `paired/source/`. This repack of the pinned Chimera source omits generated
   Python bytecode. Link the prepared dataset directory at
   `paired/source/dataset`.
3. Prepare the inputs with the artifact's dataset and index workflows. The
   measured cohorts are LoTTE pooled dev search (2,931 queries), the first
   1,000 HotpotQA train queries in source order, and MS MARCO dev.small
   (6,980 queries). Check query and exact MaxSim ground-truth hashes against
   `input_verification.json` and `cohort_recheck.json`. GPU reruns need the
   original matching embeddings and 4-bit indexes or compatible regenerated
   inputs. These large files are separate from this package.
4. Build the `prepare_ivf` target from the adapted upstream benchmark CMake
   project and put the binary at `bin/prepare_ivf` in the workspace. Run
   `bash benchmark/prepare_job.sh <dataset> 4` for each dataset. This reuses
   the matching 4-bit IGP scalar index and prepares the GIGP+ input layout.
   Chimera's worker expects the `chimera_2m_4bit_chimera_v35` index directory
   for each dataset.
5. In an A100 allocation with 32 physical cores and matching NUMA placement,
   run `bash benchmark/job.sh a100 <dataset> full 4` for GIGP+ and
   `bash benchmark/chimera_job.sh a100 <dataset> full` for Chimera. Use
   `lotte`, `hotpot`, or `msmarco` as the dataset argument. Each worker
   builds natively. GIGP+ preloads query embeddings outside its timer, while
   Chimera includes query transfers.
6. Run `python3 benchmark/collect_results.py --root <workspace> --output <exports>`
   and pass the resulting `verified_frontier.csv` to
   `reproduce_table4.py --frontier`. Keep the worker logs and hardware
   records with the results.

The saved table can be regenerated without access to the original cluster.
Fresh timings depend on the hardware, toolchain, and input indexes.
