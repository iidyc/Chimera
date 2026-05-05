# Chimera Artifact Demo Experiment Summary

This file summarizes the end-to-end reproduction run performed from a fresh checkout of the `artifact` branch.

## Scope

- Repository: `/data/juelin/Chimera`
- Branch: `artifact`
- Commit: `4343f77 Add data curation pipeline scripts`
- Run window: 2026-05-01 22:29 to 2026-05-04 13:24, America/New_York
- Approximate wall-clock duration: 62 hours 55 minutes
- Figures generated in: `figure/`
- Detailed running log: `progress.md`

Notes:

- Chimera benchmark and ablation reruns used `--num-runs 3`.
- PLAID and PLAID+ used one benchmark pass per config, with each script's configured warmup.
- IGP was shortened after the full 56-config sweep proved too slow. The final IGP results use one selected run per paper config, as requested. The LoTTE k=10 IGP point reuses the completed test result from the partial full-grid attempt.
- PLAID+ was limited by the local RTX 3090 24GB GPU. LoTTE and HotpotQA were run on 24GB-compatible grids; MS MARCO GPU-resident PLAID+ was omitted because it did not fit.

## Test Bed

| Component | Value |
|---|---|
| CPU | 2x Intel Xeon Silver 4214R @ 2.40 GHz |
| CPU cores / threads | 24 cores / 48 threads |
| RAM | 376 GiB |
| Swap | 127 GiB |
| GPUs | 4x NVIDIA GeForce RTX 3090 |
| GPU memory | 24,576 MiB per GPU |
| NVIDIA driver | 590.48.01 |
| Storage mount | `/data` |
| Final `/data` usage | 3.1T total, 2.9T used, 34G free |
| OS architecture | x86_64 |

Key software versions observed during setup:

| Environment | Versions |
|---|---|
| Chimera | Python 3.11.15, CMake 4.3.0, nvcc 12.9.86 |
| IGP | Python 3.11.15, CMake 4.3.0, nvcc 12.9.86, faiss 1.14.1, torch 2.11.0+cu128 |
| PLAID / ColBERT | Python 3.10.20, nvcc 11.7.64, numpy 1.26.4, torch 1.13.1+cu117 |

## Timing Summary

### Setup And Build

| Process | Time |
|---|---:|
| Chimera env setup and verification | 1:12.60 |
| IGP env setup and verification | 2:37.56 |
| PLAID env setup and verification | 1:31.89 initial, 0:35.22 rerun after user-site isolation fix |
| Chimera build | 1:26.67 |
| IGP build | 1:04.30 final successful build, after Eigen/include fixes |
| PLAID build | 0:06.87 |

### Data Preparation And Embedding Generation

| Process | Time | Output |
|---|---:|---|
| Raw text download | 5:18.11 | LoTTE, HotpotQA, MS MARCO text datasets |
| Raw binary / embedding generation | partial multi-hour run | LoTTE and HotpotQA raw binaries and ground truth; MS MARCO raw binaries |
| MS MARCO 4-GPU ground truth generation | 3:27:05 | `dataset/msmarco/raw/gt.tsv` with 6,980,000 rows |
| README figure generation from checked-in expected profiling | 0:09.93 | Initial figure sanity check |

The raw binary stage included failed/restarted attempts due to CUDA memory pressure and MS MARCO ground-truth runtime. These are documented in `progress.md`.

## Code And Script Modifications

Yes, this reproduction required minimal code/script modifications. They were made to fix environment isolation, missing dependencies, path defaults, local GPU-memory limits, and one Chimera file-format compatibility issue.

Modified files:

| File | Purpose |
|---|---|
| `setup/setup_plaid_env.sh` | Added `PYTHONNOUSERSITE=1` to avoid accidentally importing user-site packages instead of the Conda environment packages. |
| `setup/setup_igp_env.sh` | Added Eigen to the IGP environment dependencies. |
| `IGP/CMakeLists.txt` | Added `${CONDA_PREFIX}/include/eigen3` so IGP could compile against Conda's Eigen installation. |
| `data_curation/plaid/compute_topk.py` | Moved `torch` import to module scope, reduced default scoring tensor memory pressure for 24GB GPUs, and added `--query-start` / `--query-end` for sharded ground-truth generation. |
| `experiment/plaid/bench_colbert.sh` | Fixed default config path to `config/colbert_config.csv`, fixed default search script path, and used `PYTHONNOUSERSITE=1`. |
| `experiment/plaid/run_plaid_large_vram.sh` | Fixed default config path to `config/colbert_config.csv`. |
| `experiment/plaid/search.py` | Explicitly releases the previous GPU-resident `Searcher` between PLAID+ configs to avoid CUDA OOM. |
| `experiment/plaid_plus/bench_all_colbert_gpu_resident.sh` | Fixed `repo_root` use before it was initialized. |
| `experiment/plaid_plus/run_plaid_vs_gpu_search_main.sh` | Fixed default config path to `config/colbert_config.csv`. |
| `experiment/plaid_plus/run_plaid_vs_gpu_search_breakdown.sh` | Fixed default config path to `config/colbert_config.csv`. |
| `data_curation/igp/build_index.py` | Put `IGP/build` ahead of `IGP/` on `sys.path` so the compiled extension is imported. |
| `experiment/igp/run_igp_paper_cpu_benchmark.py` | Same compiled-extension import-path fix for IGP benchmarking. |
| `Chimera/cpp/include/clustered_format.hpp` | Fixed v2 header-size calculation so existing `cluster_1bit.bin` files pass validation. |
| `plots/plot_system_pareto_frontier.py` | Updated the IGP plot label from 32 threads to 48 threads for this demo run. |
| `config/plaid_plus_lotte_24gb.csv` | Added a 24GB-compatible PLAID+ LoTTE grid. |
| `config/plaid_plus_hotpot_24gb.csv` | Added a 24GB-compatible PLAID+ HotpotQA grid. |
| `config/plaid_plus_config_24gb.csv` | Scratch 24GB PLAID+ config generated during the run; retained for traceability. |
| `progress.md` | Continuous execution log. |
| `experiment.md` | This summary. |

`ColBERT/colbert_ai.egg-info/SOURCES.txt` was also changed by the editable ColBERT install/build process.

## Command Appendix

The commands below are the command patterns used for this run. Run them from the repository root.

### Environment Setup

```bash
setup/setup_chimera_env.sh --verify
setup/setup_igp_env.sh --verify
setup/setup_plaid_env.sh --verify
```

After applying the IGP Eigen/path fixes, build the systems:

```bash
conda run -n chimera setup/build_chimera.sh
conda run -n IGP setup/build_igp.sh
conda run -n colbert env PYTHONNOUSERSITE=1 setup/build_plaid.sh
```

Hardware notes:

- Use the same machine for all benchmark comparisons when possible.
- Use `PYTHONNOUSERSITE=1` for PLAID/ColBERT commands to avoid contamination from user-site packages.
- Use `CUDA_VISIBLE_DEVICES=<gpu_id>` to select a specific GPU for single-GPU stages.

### Data Download

```bash
python data_curation/download/download_raw_text.py --dataset all
```

To download one dataset:

```bash
python data_curation/download/download_raw_text.py --dataset lotte
python data_curation/download/download_raw_text.py --dataset hotpot
python data_curation/download/download_raw_text.py --dataset msmarco
```

### Embedding / Raw Binary Generation

Default all-dataset pipeline:

```bash
conda run -n colbert env PYTHONNOUSERSITE=1 \
  data_curation/create_raw_binaries.sh --dataset all --gpus auto
```

Resume without recomputing completed outputs:

```bash
conda run -n colbert env PYTHONNOUSERSITE=1 \
  data_curation/create_raw_binaries.sh --dataset all --gpus auto --skip-existing
```

MS MARCO ground truth was long on one GPU, so it was sharded across GPUs using the added query-range arguments:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n colbert env PYTHONNOUSERSITE=1 \
  python data_curation/plaid/compute_topk.py \
  --data-path dataset/msmarco/raw/data.bin \
  --doclens-path dataset/msmarco/raw/doclens.bin \
  --query-path dataset/msmarco/raw/query.bin \
  --output-path dataset/msmarco/raw/gt.part0.tsv \
  --topk 1000 --query-start 0 --query-end 1745

CUDA_VISIBLE_DEVICES=1 conda run -n colbert env PYTHONNOUSERSITE=1 \
  python data_curation/plaid/compute_topk.py \
  --data-path dataset/msmarco/raw/data.bin \
  --doclens-path dataset/msmarco/raw/doclens.bin \
  --query-path dataset/msmarco/raw/query.bin \
  --output-path dataset/msmarco/raw/gt.part1.tsv \
  --topk 1000 --query-start 1745 --query-end 3490

CUDA_VISIBLE_DEVICES=2 conda run -n colbert env PYTHONNOUSERSITE=1 \
  python data_curation/plaid/compute_topk.py \
  --data-path dataset/msmarco/raw/data.bin \
  --doclens-path dataset/msmarco/raw/doclens.bin \
  --query-path dataset/msmarco/raw/query.bin \
  --output-path dataset/msmarco/raw/gt.part2.tsv \
  --topk 1000 --query-start 3490 --query-end 5235

CUDA_VISIBLE_DEVICES=3 conda run -n colbert env PYTHONNOUSERSITE=1 \
  python data_curation/plaid/compute_topk.py \
  --data-path dataset/msmarco/raw/data.bin \
  --doclens-path dataset/msmarco/raw/doclens.bin \
  --query-path dataset/msmarco/raw/query.bin \
  --output-path dataset/msmarco/raw/gt.part3.tsv \
  --topk 1000 --query-start 5235 --query-end 6980
```

Hardware notes:

- `--gpus auto` uses available GPUs where the script supports it.
- For fewer GPUs, run fewer shards with wider `--query-start` / `--query-end` ranges.
- For smaller GPU memory, reduce the query batch / document chunk parameters exposed by `compute_topk.py`, or keep the patched lower defaults used in this run.

### Index Construction

Chimera indexes:

```bash
conda run -n chimera env CUDA_VISIBLE_DEVICES=0 CHIMERA_USE_MICROMAMBA=0 \
  data_curation/chimera/build_index.sh --dataset lotte --log-dir log/reproduction/build

conda run -n chimera env CUDA_VISIBLE_DEVICES=0 CHIMERA_USE_MICROMAMBA=0 \
  data_curation/chimera/build_index.sh --dataset hotpot --log-dir log/reproduction/build

conda run -n chimera env CUDA_VISIBLE_DEVICES=0 CHIMERA_USE_MICROMAMBA=0 \
  data_curation/chimera/build_index.sh --dataset msmarco --log-dir log/reproduction/build
```

PLAID indexes:

```bash
conda run -n colbert env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python data_curation/plaid/build_colbert_index.py \
  --data-path dataset/lotte/raw/data.bin \
  --doclens-path dataset/lotte/raw/doclens.bin \
  --output-dir dataset/lotte/colbert/indexes \
  --index-name 4bit --experiment-name colbert

conda run -n colbert env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python data_curation/plaid/build_colbert_index.py \
  --data-path dataset/hotpot/raw/data.bin \
  --doclens-path dataset/hotpot/raw/doclens.bin \
  --output-dir dataset/hotpot/colbert/indexes \
  --index-name 4bit --experiment-name colbert

conda run -n colbert env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 \
  python data_curation/plaid/build_colbert_index.py \
  --data-path dataset/msmarco/raw/data.bin \
  --doclens-path dataset/msmarco/raw/doclens.bin \
  --output-dir dataset/msmarco/colbert/indexes \
  --index-name 4bit --experiment-name colbert
```

IGP indexes:

```bash
conda run -n IGP env CUDA_VISIBLE_DEVICES=0 \
  python data_curation/igp/build_index.py --dataset lotte \
  --local-data-root dataset/lotte/igp --n-bit 4

conda run -n IGP env CUDA_VISIBLE_DEVICES=0 \
  python data_curation/igp/build_index.py --dataset hotpot \
  --local-data-root dataset/hotpot/igp --n-bit 4

conda run -n IGP env CUDA_VISIBLE_DEVICES=0 \
  python data_curation/igp/build_index.py --dataset msmarco \
  --local-data-root dataset/msmarco/igp --n-bit 4
```

Hardware notes:

- Index construction is storage-heavy. This run retained about 943G of durable generated artifacts and used additional transient shard space during embedding generation.
- Chimera and PLAID index builds were run on one selected GPU with `CUDA_VISIBLE_DEVICES=0`.
- IGP index build peak CPU memory was high, up to about 282G RSS for MS MARCO in this run.

### Chimera Benchmarks

Use a unique `RUN_ID` for every benchmark run so old CSVs are not overwritten:

```bash
RUN_ID="repro_$(date -u +%Y%m%dT%H%M%SZ)"
```

Top-k baseline:

```bash
conda run -n chimera env CUDA_VISIBLE_DEVICES=0 CHIMERA_USE_MICROMAMBA=0 \
  python experiment/chimera/run_gpu_search_topk_baseline.py \
  --run-id "$RUN_ID" \
  --datasets lotte hotpot msmarco \
  --k 100 --slots 1 --num-runs 3
```

Variant-target ablation:

```bash
conda run -n chimera env CUDA_VISIBLE_DEVICES=0 CHIMERA_USE_MICROMAMBA=0 \
  python experiment/chimera/run_gpu_search_variant_targets.py \
  --run-id "$RUN_ID" \
  --datasets lotte hotpot msmarco \
  --targets 0.90 0.95 \
  --systems gpu_search gpu_search_nolut gpu_search_nosum gpu_search_nolut_nosum \
  --slots 1 --num-runs 3
```

Chunk-target ablation:

```bash
conda run -n chimera env CUDA_VISIBLE_DEVICES=0 CHIMERA_USE_MICROMAMBA=0 \
  python experiment/chimera/run_gpu_search_chunk_targets.py \
  --run-id "$RUN_ID" \
  --datasets lotte hotpot msmarco \
  --targets 0.90 0.95 \
  --chunks 1 2 3 4 5 6 7 8 \
  --slots 1 --num-runs 3
```

Hardware and `run_id` notes:

- `--run-id` controls the suffix of the generated result and summary CSVs under `profiling/chimera_*`.
- Use a new `--run-id` after any code, config, or hardware change.
- `--num-runs 3` matches the repeated Chimera measurements used here. Increase for tighter confidence intervals; decrease for smoke tests.
- `--slots 1` was used for this run. Change only if you understand the GPU memory and concurrency tradeoff.

### PLAID Benchmarks

PLAID CPU-hosted k=10:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  experiment/plaid/bench_all_colbert.sh \
  --dataset lotte --dataset hotpot --dataset msmarco \
  --implementation-label plaid \
  --output-dir profiling/plaid_runs/k10 \
  --log-dir log/bench/plaid_runs/k10 \
  --k 10
```

PLAID CPU-hosted k=100:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  experiment/plaid/bench_all_colbert.sh \
  --dataset lotte --dataset hotpot --dataset msmarco \
  --implementation-label plaid \
  --output-dir profiling/plaid_runs/k100 \
  --log-dir log/bench/plaid_runs/k100 \
  --k 100
```

Hardware notes:

- Select the GPU with `CUDA_VISIBLE_DEVICES`.
- The default PLAID config is `config/colbert_config.csv`.
- Use smaller `ncells` / `ndocs` grids in the config CSV for faster smoke tests.

### PLAID+ Benchmarks

The full paper grid did not fit in 24GB for all settings, so this run used 24GB-compatible config files.

LoTTE k=10:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  experiment/plaid_plus/bench_all_colbert_gpu_resident.sh \
  --dataset lotte \
  --implementation-label plaid_gpu_resident \
  --config-file config/plaid_plus_lotte_24gb.csv \
  --output-dir profiling/plaid_plus_runs/k10 \
  --log-dir log/bench/plaid_plus_runs/k10_lotte \
  --k 10
```

HotpotQA k=10:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  experiment/plaid_plus/bench_all_colbert_gpu_resident.sh \
  --dataset hotpot \
  --implementation-label plaid_gpu_resident \
  --config-file config/plaid_plus_hotpot_24gb.csv \
  --output-dir profiling/plaid_plus_runs/k10 \
  --log-dir log/bench/plaid_plus_runs/k10_hotpot_safe \
  --k 10
```

LoTTE k=100:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  experiment/plaid_plus/bench_all_colbert_gpu_resident.sh \
  --dataset lotte \
  --implementation-label plaid_gpu_resident \
  --config-file config/plaid_plus_lotte_24gb.csv \
  --output-dir profiling/plaid_plus_runs/k100 \
  --log-dir log/bench/plaid_plus_runs/k100_lotte \
  --k 100
```

HotpotQA k=100:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  experiment/plaid_plus/bench_all_colbert_gpu_resident.sh \
  --dataset hotpot \
  --implementation-label plaid_gpu_resident \
  --config-file config/plaid_plus_hotpot_24gb.csv \
  --output-dir profiling/plaid_plus_runs/k100 \
  --log-dir log/bench/plaid_plus_runs/k100_hotpot \
  --k 100
```

Hardware notes:

- For GPUs larger than 24GB, use the paper/default config grid instead of the `*_24gb.csv` files.
- For GPUs smaller than 24GB, reduce the grid further by lowering `ncells` and `ndocs` in a copied config CSV.
- MS MARCO GPU-resident PLAID+ was omitted here because it did not fit on 24GB GPUs.

### IGP Benchmarks

Full paper-style sweep attempted in this run:

```bash
conda run -n IGP python experiment/igp/run_igp_paper_cpu_benchmark.py \
  --datasets lotte hotpot msmarco \
  --topks 10 100 \
  --threads 48 \
  --output-dir profiling/igp_paper_cpu_t48
```

Selected one-config runs used after shortening:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=48 \
  conda run -n IGP python experiment/igp/run_igp_paper_cpu_benchmark.py \
  --datasets lotte --topks 100 --threads 48 \
  --output-dir profiling/igp_selected_t48 \
  --configs 32:6000 --query-split all

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=48 \
  conda run -n IGP python experiment/igp/run_igp_paper_cpu_benchmark.py \
  --datasets hotpot --topks 10 --threads 48 \
  --output-dir profiling/igp_selected_t48 \
  --configs 64:600 --query-split all

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=48 \
  conda run -n IGP python experiment/igp/run_igp_paper_cpu_benchmark.py \
  --datasets hotpot --topks 100 --threads 48 \
  --output-dir profiling/igp_selected_t48 \
  --configs 64:6000 --query-split all

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=48 \
  conda run -n IGP python experiment/igp/run_igp_paper_cpu_benchmark.py \
  --datasets msmarco --topks 10 --threads 48 \
  --output-dir profiling/igp_selected_t48 \
  --configs 64:600 --query-split all

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=48 \
  conda run -n IGP python experiment/igp/run_igp_paper_cpu_benchmark.py \
  --datasets msmarco --topks 100 --threads 48 \
  --output-dir profiling/igp_selected_t48 \
  --configs 64:6000 --query-split all
```

Hardware notes:

- Set `--threads` to the number of CPU threads you want IGP to use.
- Keep `OMP_NUM_THREADS` aligned with `--threads`.
- `OPENBLAS_NUM_THREADS=1` avoids nested OpenBLAS oversubscription and warning spam while IGP itself uses `--threads`.
- On smaller-memory machines, run one dataset/top-k at a time as shown above.

### Plotting Commands

Prefer non-timestamped defaults where the plotting scripts support them. The Chimera plotting scripts use `latest_required(...)` for timestamped Chimera summaries, so users do not need to hardcode `RUN_ID`-specific paths for those plots.

System Pareto frontier from measured outputs:

```bash
conda run -n chimera python plots/plot_system_pareto_frontier.py \
  --plaid profiling/plaid/summary.csv \
  --plaid-plus profiling/plaid_plus/summary.csv \
  --output-dir figure
```

Notes:

- `--chimera` is intentionally omitted above. The script automatically selects the latest `profiling/chimera_topk_baseline/chimera_topk_baseline_summary_*.csv`.
- `--plaid` and `--plaid-plus` point at stable aggregate files generated by the aggregation step.
- `--igp` is intentionally omitted above. The script defaults to the stable aggregate file `profiling/igp/summary.csv`.

Chimera variant figures:

```bash
conda run -n chimera python plots/plot_chimera_variant_targets.py --output-dir figure
```

This automatically selects the latest `profiling/chimera_variant_targets/chimera_variant_targets_summary_*.csv`.

Chimera chunk figures:

```bash
conda run -n chimera python plots/plot_chimera_chunk_targets.py \
  --output-dir figure --plot stacked --chunks 1 2 3 4 5 6 7 8
```

This automatically selects the latest `profiling/chimera_chunk_targets/chimera_chunk_targets_summary_*.csv`.

LoTTE latency breakdown from checked-in profiling summary:

```bash
conda run -n chimera python plots/plot_lotte_latency_breakdown_targets.py \
  --input expected_profiling/chimera_profile/chimera_profile_summary_20260429T200445Z.csv \
  --output-dir figure
```

The latency breakdown command intentionally uses a checked-in expected profiling summary because this demo did not rerun the lower-level Chimera profiling pass that produces `profiling/chimera_profile/chimera_profile_summary_*.csv`. If that profiling pass is rerun, omit `--input` and the script will select the latest generated profile summary automatically.

If you need to plot a specific historical run rather than the latest one, pass the corresponding `--chimera` or `--input` path explicitly. For normal reruns, omitting those timestamped paths is preferred.

### Index Construction

| System | Dataset | Time | Index size |
|---|---|---:|---:|
| Chimera | LoTTE | 42:39.05 | 50G |
| Chimera | HotpotQA | 47:06.14 | 57G |
| Chimera | MS MARCO | 1:27:01 | 108G |
| PLAID | LoTTE | 13:53.21 | 18G |
| PLAID | HotpotQA | 21:09.14 | 21G |
| PLAID | MS MARCO | 36:46.95 | 40G |
| IGP | LoTTE | 1:33:14 | 17G |
| IGP | HotpotQA | 1:38:16 | 20G |
| IGP | MS MARCO | 3:39:07 | 39G |

### Chimera Benchmarks And Ablations

| Experiment | Time | Notes |
|---|---:|---|
| Top-k baseline, first run | 1:40:02 | Invalidated by `cluster_1bit.bin` header-size issue |
| Variant-target ablation, first run | 31:41.33 | Invalidated by same issue |
| Chunk-target ablation, first run | 1:01:59 | Invalidated by same issue |
| Header fix rebuild and smoke test | about 1:35 | Rebuilt after `clustered_format.hpp` fix |
| Top-k baseline rerun | 2:47:59 | Successful, 3 runs per point |
| Variant-target ablation rerun | 13:49:18 | Successful, 3 runs per point |
| Chunk-target ablation rerun | 1:47:22 | Successful, 3 runs per point |

### PLAID And PLAID+

| Experiment | Time | Notes |
|---|---:|---|
| PLAID k=10 CPU-hosted benchmark | 7:12:40 | All three datasets |
| PLAID k=100 CPU-hosted benchmark | 6:44:34 | All three datasets |
| PLAID summary aggregation | <1m | Wrote `profiling/plaid/summary.csv` |
| PLAID+ k=10 first attempt | 2:08.48 | Failed due stale GPU-resident `Searcher` lifetime |
| PLAID+ k=10 24GB run | 40:02.59 | LoTTE complete; HotpotQA OOM at `8,8000` |
| PLAID+ HotpotQA k=10 safe rerun | 12:29.89 | Completed 24GB-compatible grid |
| PLAID+ LoTTE k=100 24GB run | 27:29.18 | Completed 24GB-compatible grid |
| PLAID+ HotpotQA k=100 24GB run | 12:40.69 | Completed 24GB-compatible grid |
| PLAID+ summary aggregation | <1m | Wrote `profiling/plaid_plus/summary.csv` |

### IGP

| Experiment | Time | Notes |
|---|---:|---|
| Full 48-thread CPU sweep attempt | about 4:48 before stop | Stopped after user approved shortening; produced LoTTE k=10 full-grid/test JSON |
| LoTTE k=100 selected config | 7:39.10 | `phi_pb=32`, `phi_ref=6000`, recall 0.933415 |
| HotpotQA k=10 selected config | 6:24.34 | `phi_pb=64`, `phi_ref=600`, recall 0.862200 |
| HotpotQA k=100 selected config | 6:26.09 | `phi_pb=64`, `phi_ref=6000`, recall 0.896770 |
| MS MARCO k=10 selected config | 13:42.52 | `phi_pb=64`, `phi_ref=600`, recall 0.933195 |
| MS MARCO k=100 selected config | 14:11.92 | `phi_pb=64`, `phi_ref=6000`, recall 0.943828 |
| IGP summary aggregation | <1m | Wrote `profiling/igp/summary.csv`; the original 48-thread run copy is also retained as `profiling/igp/summary_t48.csv` |

### Plotting

| Figure set | Time | Source |
|---|---:|---|
| System Pareto frontier | <1m | Measured PLAID, PLAID+, IGP, and Chimera summaries |
| Chimera variant-target figures | <1m | Measured Chimera variant summary |
| Chimera chunk-target figures | <1m | Measured Chimera chunk summary |
| LoTTE latency breakdown | <1m | Checked-in `expected_profiling/chimera_profile` summary |

Generated files:

- `figure/system_pareto_frontier_topk10_topk100_qps.png`
- `figure/system_pareto_frontier_topk10_topk100_qps.pdf`
- `figure/system_pareto_frontier_topk10_topk100_qps_selected.csv`
- `figure/chimera_variant_targets_qps.png`
- `figure/chimera_variant_targets_qps.pdf`
- `figure/chimera_variant_targets_latency.png`
- `figure/chimera_variant_targets_latency.pdf`
- `figure/chimera_chunk_targets_qps_all_datasets_r090_r095_stacked.png`
- `figure/chimera_chunk_targets_qps_all_datasets_r090_r095_stacked.pdf`
- `figure/lotte_latency_breakdown_recall090_095.png`
- `figure/lotte_latency_breakdown_recall090_095.pdf`

## Storage Summary

Final disk snapshot:

| Path / category | Size |
|---|---:|
| `/data` total | 3.1T |
| `/data` used | 2.9T |
| `/data` available | 34G |
| `dataset/_downloads` | 6.6G |
| `dataset/hotpot/text` | 1.5G |
| `dataset/lotte/text` | 2.2G |
| `dataset/msmarco/text` | 2.9G |
| `dataset/lotte/raw` | 128G |
| `dataset/hotpot/raw` | 147G |
| `dataset/msmarco/raw` | 285G |
| `dataset/lotte/gpu_search_2m` | 50G |
| `dataset/hotpot/gpu_search_2m` | 57G |
| `dataset/msmarco/gpu_search_2m` | 108G |
| `dataset/lotte/colbert` | 18G |
| `dataset/hotpot/colbert` | 21G |
| `dataset/msmarco/colbert` | 40G |
| `dataset/lotte/igp` | 17G |
| `dataset/hotpot/igp` | 20G |
| `dataset/msmarco/igp` | 39G |
| `profiling` | 6.0M |
| `figure` | 2.8M |
| `log/bench` | 1.9M |
| `log/reproduction` | 520K |

Approximate durable project data produced or retained by the end-to-end run:

| Category | Approximate size |
|---|---:|
| Downloads and text | 13.2G |
| Raw embedding binaries and ground truth | 560G |
| Chimera indexes | 215G |
| PLAID indexes | 79G |
| IGP indexes | 76G |
| Profiling, figures, and logs | <20M |
| Total listed artifacts | about 943G |

Additional transient storage was required during embedding generation. Redundant `embedding_shards` directories were removed after merged raw binaries were created, freeing roughly 127G for LoTTE, 147G for HotpotQA, and 285G for MS MARCO.

## Reproducibility Caveats

- The demo machine was storage constrained by the end of the run: only 34G remained free on `/data`.
- PLAID+ full paper settings require more GPU memory than 24GB for some grid points. This run used 24GB-compatible configs.
- IGP was run with 48 CPU threads and shortened to selected one-run configs to avoid the multi-hour full sweep for every dataset/top-k pair.
- Several minimal fixes were needed for this checkout to run end-to-end, including environment isolation, Eigen include paths, PLAID script defaults, IGP import ordering, PLAID+ GPU memory cleanup, and Chimera clustered-format header validation.
