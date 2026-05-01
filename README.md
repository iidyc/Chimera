# Chimera / IGP / ColBERT Research Artifact

This repository separates source code from reproducible experiment, setup, and dataset/index pipelines.

## Directory Structure

- `Chimera/`: Chimera C++/CUDA source and CMake build configuration.
- `IGP/`: IGP source tree and implementation utilities.
- `ColBERT/`: ColBERT/PLAID source tree.
- `experiment/`: benchmark, comparison, profiling, and ablation drivers.
- `setup/`: environment creation and build entrypoints.
- `data_curation/`: raw data preparation, metadata generation, and index construction drivers.
- `dataset/`: local dataset inputs and generated indexes.
- `profiling/`: benchmark CSV outputs.
- `backup/log/`: archived run logs.

## Workflow

1. Create environments:

```bash
setup/setup_chimera_env.sh --verify
setup/setup_igp_env.sh --verify
setup/setup_plaid_env.sh --verify
```

2. Build systems:

```bash
setup/build_chimera.sh
setup/build_igp.sh
setup/build_plaid.sh
```

3. Download raw text datasets:

```bash
python data_curation/download/download_raw_text.py --dataset all
```

This writes `dataset/<name>/text/collection.tsv`, `queries.tsv`, `qrels.tsv`,
and `metadata.json` for `lotte`, `hotpot`, and `msmarco`. Use
`--dataset lotte`, `--dataset hotpot`, or `--dataset msmarco` to download one
dataset at a time.

LoTTE uses the ColBERTv2 release (`colbertv2/lotte_passages`) and downloads the
pooled `dev` search split, which matches the artifact's 2,931-query LoTTE
Pooled setup. HotpotQA and MS MARCO are downloaded through the official-source
scripts in `data_curation/plaid/`: HotpotQA uses the BEIR archive
`hotpotqa.zip`, and MS MARCO uses Microsoft's `collectionandqueries.tar.gz`.
Those scripts also write `query_id_map.tsv` because the benchmark files use
local sequential query ids.

4. Create experiment binary inputs:

```bash
data_curation/create_raw_binaries.sh --dataset all --gpus auto
```

This encodes `collection.tsv` into `raw/data.bin` and `raw/doclens.bin`,
encodes `queries.tsv` into `raw/query.bin`, and generates the top-1000
reference ranking `raw/gt.tsv`. Document encoding streams the collection in
chunks and, when multiple GPUs are visible, encodes contiguous shards in
parallel before streaming them back into one ordered `data.bin`/`doclens.bin`
pair. Generating `gt.tsv` is a brute-force MaxSim pass and is usually the most
expensive data-curation step.

To run the entire data-preparation pipeline for one dataset:

```bash
data_curation/prepare_dataset.sh --dataset lotte --gpus auto
```

Use `--skip-gt` when you only need embeddings for index construction and do not
need benchmark recall yet.

5. Build indexes:

```bash
data_curation/chimera/build_index.sh --dataset lotte
data_curation/chimera/build_all_indices.sh
python data_curation/igp/build_index.py --dataset lotte
python data_curation/plaid/build_colbert_index.py --dataset lotte
```

Default outputs are under `dataset/<name>/gpu_search_2m/`, `dataset/<name>/igp/`, and the ColBERT experiment root configured by the PLAID scripts. Chimera experiment drivers also detect older generated index directories for compatibility.

6. Run benchmarks:

```bash
experiment/chimera/bench_gpu_search.sh --dataset lotte --version gpu_search --binary Chimera/build/gpu_search
experiment/chimera/run_gpu_search_main_experiment.py --build
experiment/plaid/bench_colbert.sh --dataset lotte
experiment/plaid_plus/run_plaid_vs_gpu_search_main.sh --execute
```

7. Run ablations:

```bash
experiment/ablation/run_gpu_search_lut_ablation.py
experiment/ablation/run_gpu_search_recall95_ablation.py --build
```

## Reproducing Paper Figures

The figure scripts read normalized benchmark artifacts and do not rerun
benchmarks. Use `expected_profiling/` to regenerate the paper figures from the
checked-in results, or copy those files into `profiling/` and replace the paths
below with your new run outputs.

```bash
mkdir -p figure profiling
cp -a expected_profiling/igp expected_profiling/plaid expected_profiling/plaid_plus profiling/
```

The plotted values are means over three benchmark runs. The Chimera drivers use
`--num-runs 3` by default and write averaged `*_summary_<RUN_ID>.csv` files.

### Figure 7: End-to-end comparison

Purpose: compare Chimera, PLAID, PLAID+, and IGP throughput/recall Pareto
frontiers on LoTTE, HotpotQA, and MS MARCO. Columns are datasets; the top row
is Recall@10 and the bottom row is Recall@100.

Inputs:

- `profiling/plaid/summary.csv`
- `profiling/plaid_plus/summary.csv`
- `profiling/igp/summary_t32.csv`
- `expected_profiling/chimera_topk_baseline/chimera_topk_baseline_summary_20260429T182345Z.csv`

Generate the figure:

```bash
python plots/plot_system_pareto_frontier.py \
  --plaid profiling/plaid/summary.csv \
  --plaid-plus profiling/plaid_plus/summary.csv \
  --igp profiling/igp/summary_t32.csv \
  --chimera expected_profiling/chimera_topk_baseline/chimera_topk_baseline_summary_20260429T182345Z.csv \
  --output-dir figure
```

Outputs:

- `figure/system_pareto_frontier_topk10_topk100_qps.png`
- `figure/system_pareto_frontier_topk10_topk100_qps.pdf`
- `figure/system_pareto_frontier_topk10_topk100_qps_selected.csv`

To regenerate the Chimera input from benchmarks:

```bash
export RUN_ID=paper_$(date -u +%Y%m%dT%H%M%SZ)
python experiment/chimera/run_gpu_search_topk_baseline.py \
  --build --run-id "$RUN_ID" \
  --datasets lotte hotpot msmarco \
  --k 100 --slots 1 --num-runs 3
```

This requires `Chimera/build/gpu_search`, `dataset/<name>/raw/query.bin`,
`dataset/<name>/raw/gt.tsv`, a Chimera index such as
`dataset/<name>/gpu_search_2m/`, and runtime configs in `config/`.

### Figure 8: Chimera LUT and score aggregation ablation

Purpose: measure the query-latency impact of Chimera's LUT and document-score
aggregation kernel optimizations at target Recall@100 values 0.90 and 0.95.

Input:

- `expected_profiling/chimera_variant_targets/chimera_variant_targets_summary_20260429T164347Z.csv`

Generate the figure:

```bash
python plots/plot_chimera_variant_targets.py \
  --input expected_profiling/chimera_variant_targets/chimera_variant_targets_summary_20260429T164347Z.csv \
  --output-dir figure
```

Outputs:

- `figure/chimera_variant_targets_latency.png`
- `figure/chimera_variant_targets_latency.pdf`
- `figure/chimera_variant_targets_qps.png`
- `figure/chimera_variant_targets_qps.pdf`

To regenerate the input from benchmarks:

```bash
export RUN_ID=paper_$(date -u +%Y%m%dT%H%M%SZ)
python experiment/chimera/run_gpu_search_variant_targets.py \
  --build --run-id "$RUN_ID" \
  --datasets lotte hotpot msmarco \
  --targets 0.90 0.95 \
  --systems gpu_search gpu_search_nolut gpu_search_nosum gpu_search_nolut_nosum \
  --slots 1 --num-runs 3
```

### Figure 9: Overlap chunk count

Purpose: show how Chimera throughput changes with the number of overlapped
chunks. The expected trend is best throughput at a moderate chunk count.

Input:

- `expected_profiling/chimera_chunk_targets/chimera_chunk_targets_summary_20260429T172145Z.csv`

Generate the paper-style stacked figure:

```bash
python plots/plot_chimera_chunk_targets.py \
  --input expected_profiling/chimera_chunk_targets/chimera_chunk_targets_summary_20260429T172145Z.csv \
  --output-dir figure \
  --plot stacked \
  --chunks 1 2 3 4 5 6 7 8
```

Output:

- `figure/chimera_chunk_targets_qps_all_datasets_r090_r095_stacked.png`
- `figure/chimera_chunk_targets_qps_all_datasets_r090_r095_stacked.pdf`

To regenerate the input from benchmarks:

```bash
export RUN_ID=paper_$(date -u +%Y%m%dT%H%M%SZ)
python experiment/chimera/run_gpu_search_chunk_targets.py \
  --build --run-id "$RUN_ID" \
  --datasets lotte hotpot msmarco \
  --targets 0.90 0.95 \
  --chunks 1 2 3 4 5 6 7 8 \
  --slots 1 --num-runs 3
```

### Figure 10: LoTTE latency breakdown

Purpose: break Chimera's LoTTE query latency into candidate selection,
candidate refinement, document scoring, and overhead at Recall@100 targets 0.90
and 0.95.

Input:

- `expected_profiling/chimera_profile/chimera_profile_summary_20260429T200445Z.csv`

Generate the figure:

```bash
python plots/plot_lotte_latency_breakdown_targets.py \
  --input expected_profiling/chimera_profile/chimera_profile_summary_20260429T200445Z.csv \
  --output-dir figure
```

Outputs:

- `figure/lotte_latency_breakdown_recall090_095.png`
- `figure/lotte_latency_breakdown_recall090_095.pdf`

The checked-in profile summary is the canonical input for this figure. To
regenerate it, run `Chimera/build/gpu_search` on LoTTE with
`--profile-eval-all-queries` at the two operating points in
`experiment/common/gpu_search_experiment_lib.py` (`lotte` targets `0.90` and
`0.95`), then normalize the profile log into the columns consumed by
`plots/plot_lotte_latency_breakdown_targets.py`.

### Data and environment assumptions

- Plotting requires Python with `matplotlib`; benchmark regeneration additionally
  requires the Chimera CUDA/cuVS environment.
- Chimera benchmark scripts discover indexes under `dataset/<name>/gpu_search_2m/`,
  `gpu_search_1m/`, and compatible older Chimera index directory names.
- Required raw inputs for regenerated Chimera runs are
  `dataset/<name>/raw/query.bin` and `dataset/<name>/raw/gt.tsv`.
- PLAID, PLAID+, and IGP paper inputs are reused from `expected_profiling/`;
  rerunning those systems requires their setup and index pipelines.

## Outputs

- Raw Chimera logs: `profiling/chimera_*/logs/<RUN_ID>/`.
- Parsed benchmark CSVs: `profiling/**/benchmark_results*.csv`.
- Pareto frontiers: `profiling/**/pareto_frontier*.csv`.
- Experiment summaries: `profiling/<experiment-name>/`.

## Runtime Estimates

Runtime depends heavily on GPU, storage, and dataset size. Typical ranges:

- Dataset encoding: tens of minutes for LoTTE, hours for MS MARCO or HotpotQA.
- Chimera index build: tens of minutes per dataset for 2M clusters.
- IGP index build: tens of minutes to hours per dataset.
- PLAID index build: tens of minutes to hours per dataset.
- Single benchmark sweep: minutes per dataset/config set.
- Full PLAID vs. Chimera comparison or ablation: hours across all datasets.

## Troubleshooting

- If CMake cannot find CUDA/cuVS, activate the Chimera environment or pass `CUDAToolkit_ROOT` through `setup/build_chimera.sh --`.
- If ColBERT cannot import CUDA extensions, rerun `setup/setup_plaid_env.sh --verify` and then `setup/build_plaid.sh`.
- If benchmark scripts cannot find data, verify `dataset/<name>/raw/query.bin`, `dataset/<name>/raw/gt.tsv`, and the relevant index directory exist.
