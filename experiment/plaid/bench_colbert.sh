#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  experiment/plaid/bench_colbert.sh [--dataset <name> ...] [options]
  experiment/plaid/bench_colbert.sh [dataset_name ...] [options]

Description:
  Benchmark ColBERT search on one or more datasets and write per-dataset
  recall-vs-qps measurements plus a Pareto frontier CSV.

  If no dataset is provided, the script benchmarks the datasets that already
  have a ColBERT index at:
    dataset/<name>/colbert/indexes/<index_name>

  Outputs:
    profiling/<dataset>/colbert/benchmark_results.csv
    profiling/<dataset>/colbert/pareto_frontier.csv
    profiling/<dataset>/colbert/search_py_output.csv
    log/bench/colbert/<dataset>/benchmark.log

Options:
  --dataset <name>         Dataset under dataset/. Repeatable.
  --config-file <path>     Config CSV. Default: config/colbert_config.csv
  --env-name <name>        Conda/mamba env name. Default: colbert
  --search-script <path>   Search driver. Default: ColBERT/experiment/search.py
  --experiment-name <name> ColBERT experiment directory. Default: colbert
  --index-name <name>      Index subdirectory under indexes/. Default: autodetect
  --implementation-label <label>
                           Output folder label. Default: colbert
  --output-dir <path>      CSV output directory. Default: profiling
  --log-dir <path>         Log output directory. Default: log/bench
  --k <top_k>              Final retrieval depth / recall depth. Default: 100
  --warmup <count>         Warmup query count before timed runs restart from query 0. Default: 5
  --compressed-embeddings-storage <cpu|gpu>
                           Where to keep ColBERT's packed codes/residuals during search.
                           Default: cpu
  --gpu-index-resident     Preload IVF/doclens plus packed embeddings onto GPU for
                           a GPU-resident ColBERT baseline. Requires
                           --compressed-embeddings-storage gpu.
  --profile-breakdown-csv <path>
                           Optional long-form CSV for aggregated stage and H2D
                           transfer profiling. Runs one extra profiling pass per
                           configuration and reports the average once.
  --dry-run                Print planned commands without executing them.
  -h, --help               Show this help message.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

trim_field() {
    local value="$1"
    value="${value//$'\r'/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

log_line() {
    local line="$1"
    if [[ $dry_run -eq 1 ]]; then
        echo "$line"
    else
        echo "$line" | tee -a "$current_log_file"
    fi
}

require_positive_int() {
    local name="$1"
    local value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "${name} must be a non-negative integer"
}

validate_dataset_name() {
    local dataset_name="$1"
    case "$dataset_name" in
        ""|.|..|*/*)
            die "dataset name must be a simple directory name under dataset/: ${dataset_name}"
            ;;
    esac
}

validate_path_component() {
    local value="$1"
    local name="$2"
    case "$value" in
        ""|.|..|*/*)
            die "${name} must be a simple path component: ${value}"
            ;;
    esac
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

run_in_env_with_cuda() {
    local env_prefix="$1"
    shift

    local ld_library_path="${env_prefix}/lib:${env_prefix}/lib64"
    local host_cc="${env_prefix}/bin/x86_64-conda-linux-gnu-cc"
    local host_cxx="${env_prefix}/bin/x86_64-conda-linux-gnu-c++"
    if [[ -n "${LD_LIBRARY_PATH-}" ]]; then
        ld_library_path="${ld_library_path}:${LD_LIBRARY_PATH}"
    fi

    echo "+ $(printf '%q ' env \
        "CUDA_HOME=${env_prefix}" \
        "PATH=${env_prefix}/bin:${PATH}" \
        "LD_LIBRARY_PATH=${ld_library_path}" \
        "PYTHONNOUSERSITE=1" \
        "PYTHONPATH=${repo_root}/ColBERT" \
        "TORCH_EXTENSIONS_DIR=${torch_extensions_dir}" \
        "CC=${host_cc}" \
        "CXX=${host_cxx}" \
        "CUDAHOSTCXX=${host_cxx}" \
        "$@")"

    if [[ $dry_run -eq 0 ]]; then
        env \
            "CUDA_HOME=${env_prefix}" \
            "PATH=${env_prefix}/bin:${PATH}" \
            "LD_LIBRARY_PATH=${ld_library_path}" \
            "PYTHONNOUSERSITE=1" \
            "PYTHONPATH=${repo_root}/ColBERT" \
            "TORCH_EXTENSIONS_DIR=${torch_extensions_dir}" \
            "CC=${host_cc}" \
            "CXX=${host_cxx}" \
            "CUDAHOSTCXX=${host_cxx}" \
            "$@"
    fi
}

get_cpu_model() {
    local cpu_model=""
    if command -v lscpu >/dev/null 2>&1; then
        cpu_model="$(lscpu | awk -F: '/Model name:/ {gsub(/^[ \t]+/, "", $2); print $2; exit}')"
    fi
    if [[ -z "$cpu_model" && -r /proc/cpuinfo ]]; then
        cpu_model="$(awk -F: '/model name/ {gsub(/^[ \t]+/, "", $2); print $2; exit}' /proc/cpuinfo)"
    fi
    if [[ -z "$cpu_model" ]]; then
        cpu_model="unknown"
    fi
    echo "$cpu_model"
}

get_gpu_count_used() {
    local cuda_visible_devices="${CUDA_VISIBLE_DEVICES-}"
    if [[ -n "$cuda_visible_devices" ]]; then
        local cleaned="${cuda_visible_devices// /}"
        local count=0
        local gpu_id=""
        IFS=',' read -r -a gpu_ids <<< "$cleaned"
        for gpu_id in "${gpu_ids[@]}"; do
            [[ -n "$gpu_id" ]] && ((count += 1))
        done
        echo "$count"
        return
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu_count_output=""
        if gpu_count_output="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
            printf '%s\n' "$gpu_count_output" | sed '/^$/d' | wc -l | tr -d '[:space:]'
            return
        fi
    fi

    echo "unknown"
}

print_hardware_info() {
    log_line "[driver] cpu_model=$(get_cpu_model)"
    log_line "[driver] cpu_threads_available=$(nproc)"
    log_line "[driver] omp_num_threads=${OMP_NUM_THREADS-<unset>}"
    log_line "[driver] cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
    log_line "[driver] gpu_count_used=$(get_gpu_count_used)"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_line "[driver] nvidia_smi=unavailable"
        return
    fi

    local gpu_query_output=""
    local gpu_info_lines=()
    local gpu_line=""
    if ! gpu_query_output="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null)"; then
        log_line "[driver] nvidia_smi_query=failed"
        return
    fi

    mapfile -t gpu_info_lines <<< "$gpu_query_output"
    log_line "[driver] host_gpu_count=${#gpu_info_lines[@]}"
    for gpu_line in "${gpu_info_lines[@]}"; do
        [[ -n "$gpu_line" ]] || continue
        log_line "[driver] host_gpu=${gpu_line}"
    done
}

discover_default_datasets() {
    local matches=()
    local path=""
    for path in "${repo_root}"/dataset/*/"${experiment_name}"/indexes; do
        [[ -d "$path" ]] || continue
        matches+=("$(basename "$(dirname "$(dirname "$path")")")")
    done
    if [[ ${#matches[@]} -eq 0 ]]; then
        die "no datasets with ${experiment_name}/indexes found under ${repo_root}/dataset"
    fi
    printf '%s\n' "${matches[@]}" | sort -u
}

resolve_index_name() {
    local dataset_name="$1"
    local experiment_dir="$2"

    if [[ -n "$index_name_override" ]]; then
        printf '%s' "$index_name_override"
        return
    fi

    local indexes_dir="${experiment_dir}/indexes"
    local candidates=()
    local candidate=""

    [[ -d "$indexes_dir" ]] || die "missing indexes directory: ${indexes_dir}"

    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] && candidates+=("$candidate")
    done < <(find "$indexes_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

    if [[ ${#candidates[@]} -eq 1 ]]; then
        printf '%s' "${candidates[0]}"
        return
    fi

    if [[ ${#candidates[@]} -eq 0 ]]; then
        die "no ColBERT index found under ${indexes_dir}"
    fi

    die "multiple ColBERT indices found for ${dataset_name}; pass --index-name explicitly"
}

write_pareto_frontier() {
    local results_csv="$1"
    local pareto_csv="$2"
    local header=""

    header="$(head -n 1 "$results_csv")"
    {
        printf '%s\n' "$header"
        tail -n +2 "$results_csv" | awk -F, '
            {
                n++
                row[n] = $0
                qps[n] = $15 + 0.0
                recall[n] = $16 + 0.0
            }
            END {
                for (i = 1; i <= n; ++i) {
                    dominated = 0
                    for (j = 1; j <= n; ++j) {
                        if (i == j) {
                            continue
                        }
                        if ((qps[j] >= qps[i] && recall[j] >= recall[i]) &&
                            (qps[j] > qps[i] || recall[j] > recall[i])) {
                            dominated = 1
                            break
                        }
                    }
                    if (!dominated) {
                        print row[i]
                    }
                }
            }
        ' | sort -t, -k15,15g -k16,16g
    } > "$pareto_csv"
}

benchmark_dataset() {
    local dataset_name="$1"
    local dataset_dir="${repo_root}/dataset/${dataset_name}"
    local raw_dir="${dataset_dir}/raw"
    local source_experiment_dir="${dataset_dir}/${experiment_name}"
    local active_experiment_dir=""
    local active_root_path=""
    local active_index_name=""
    local query_file="${raw_dir}/query.bin"
    local gt_file="${raw_dir}/gt.tsv"
    local impl_output_dir="${output_dir}/${dataset_name}/${implementation_label}"
    local impl_log_dir="${log_dir}/${implementation_label}/${dataset_name}"
    local results_csv="${impl_output_dir}/benchmark_results.csv"
    local pareto_csv="${impl_output_dir}/pareto_frontier.csv"
    local search_py_output_csv="${impl_output_dir}/search_py_output.csv"
    local log_file="${impl_log_dir}/benchmark.log"
    local index_source="dataset"
    local pair_args=()
    local labels=()
    local ncells_values=()
    local ndocs_values=()
    local raw_label=""
    local raw_ncells=""
    local raw_ndocs=""

    validate_dataset_name "$dataset_name"

    [[ -d "$dataset_dir" ]] || die "dataset directory not found: ${dataset_dir}"
    [[ -f "$query_file" ]] || die "missing query file: ${query_file}"
    [[ -f "$gt_file" ]] || die "missing ground truth file: ${gt_file}"
    [[ -d "$source_experiment_dir" ]] || die "missing ColBERT experiment directory: ${source_experiment_dir}"
    [[ -d "${source_experiment_dir}/indexes" ]] || die "missing ColBERT indexes directory: ${source_experiment_dir}/indexes"

    current_log_file="$log_file"

    if [[ $dry_run -eq 0 ]]; then
        mkdir -p "$impl_output_dir" "$impl_log_dir" "$torch_extensions_dir"
        : > "$log_file"
    fi

    active_experiment_dir="$source_experiment_dir"
    active_root_path="$(dirname "$active_experiment_dir")"
    active_index_name="$(resolve_index_name "$dataset_name" "$active_experiment_dir")"

    while IFS=, read -r raw_label raw_ncells raw_ndocs || [[ -n "${raw_label}${raw_ncells}${raw_ndocs}" ]]; do
        local label=""
        local ncells=""
        local ndocs=""

        label="$(trim_field "$raw_label")"
        [[ -z "$label" || "$label" == "label" || "$label" == \#* ]] && continue

        ncells="$(trim_field "$raw_ncells")"
        ndocs="$(trim_field "$raw_ndocs")"
        require_positive_int "ncells" "$ncells"
        require_positive_int "ndocs" "$ndocs"
        (( ncells > 0 )) || die "ncells must be > 0"
        (( ndocs > 0 )) || die "ndocs must be > 0"

        labels+=("$label")
        ncells_values+=("$ncells")
        ndocs_values+=("$ndocs")
        pair_args+=("${ncells},${ndocs}")
    done < <(tr -d '\r' < "$config_file")

    [[ ${#pair_args[@]} -gt 0 ]] || die "no ColBERT benchmark pairs loaded from ${config_file}"

    log_line "[driver] repo_root=${repo_root}"
    log_line "[driver] implementation=${implementation_label}"
    log_line "[driver] dataset=${dataset_name}"
    log_line "[driver] env_manager=${env_manager}"
    log_line "[driver] env_name=${env_name}"
    log_line "[driver] env_prefix=${env_prefix}"
    log_line "[driver] search_script=${search_script}"
    log_line "[driver] config_file=${config_file}"
    log_line "[driver] experiment_name=${experiment_name}"
    log_line "[driver] index_name=${active_index_name}"
    log_line "[driver] query_file=${query_file}"
    log_line "[driver] gt_file=${gt_file}"
    log_line "[driver] source_experiment_dir=${source_experiment_dir}"
    log_line "[driver] active_experiment_dir=${active_experiment_dir}"
    log_line "[driver] active_index_dir=${active_experiment_dir}/indexes/${active_index_name}"
    log_line "[driver] active_root_path=${active_root_path}"
    log_line "[driver] index_source=${index_source}"
    log_line "[driver] results_csv=${results_csv}"
    log_line "[driver] pareto_csv=${pareto_csv}"
    log_line "[driver] search_py_output_csv=${search_py_output_csv}"
    log_line "[driver] k=${k}"
    log_line "[driver] warmup=${warmup}"
    log_line "[driver] compressed_embeddings_storage=${compressed_embeddings_storage}"
    log_line "[driver] gpu_index_resident=${gpu_index_resident}"
    log_line "[driver] profile_breakdown_csv=${profile_breakdown_csv:-<unset>}"
    log_line "[driver] torch_extensions_dir=${torch_extensions_dir}"
    print_hardware_info

    if [[ $dry_run -eq 0 ]]; then
        printf '%s\n' \
            "implementation,dataset,label,ncells,ndocs,index_source,index_root,index_path,query_file,gt_file,queries,k,num_runs,avg_time_s,qps,recall,gpu_mem_current_mib,gpu_mem_peak_mib,gpu_mem_total_mib,torch_peak_allocated_mib,torch_peak_reserved_mib" \
            > "$results_csv"
    fi

    local cmd=(
        python
        "$search_script"
        --root-path "$active_root_path"
        --experiment-name "$experiment_name"
        --index-name "$active_index_name"
        --query-path "$query_file"
        --ground-truth-path "$gt_file"
        --output-csv "$search_py_output_csv"
        --pairs "${pair_args[@]}"
        --k "$k"
        --warmup "$warmup"
        --compressed-embeddings-storage "$compressed_embeddings_storage"
        --dataset-name "$dataset_name"
        --implementation-label "$implementation_label"
    )
    if [[ $gpu_index_resident -eq 1 ]]; then
        cmd+=(--gpu-index-resident)
    fi
    if [[ -n "$profile_breakdown_csv" ]]; then
        cmd+=(--profile-breakdown-csv "$profile_breakdown_csv")
    fi

    log_line "[run] command=$(printf '%q ' "${cmd[@]}")"

    if [[ $dry_run -eq 1 ]]; then
        run_in_env_with_cuda "$env_prefix" "${cmd[@]}"
        return
    fi

    local run_output
    run_output="$(mktemp "${TMPDIR:-/tmp}/colbert_benchmark.${dataset_name}.XXXXXX.log")"
    local run_start_utc
    local run_end_utc
    local status=0
    run_start_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    log_line "[run] start_utc=${run_start_utc}"

    set +e
    run_in_env_with_cuda "$env_prefix" "${cmd[@]}" > "$run_output" 2>&1
    status=$?
    set -e

    cat "$run_output" | tee -a "$current_log_file"
    run_end_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    log_line "[run] end_utc=${run_end_utc}"
    log_line "[run] exit_code=${status}"

    if [[ $status -ne 0 ]]; then
        rm -f "$run_output"
        die "ColBERT benchmark failed for dataset=${dataset_name}; see ${current_log_file}"
    fi

    local query_count=""
    query_count="$(sed -nE 's/^Expecting to read ([0-9]+) document embeddings of dimension [0-9]+ from .*$/\1/p' "$run_output" | tail -n 1)"
    [[ -n "$query_count" ]] || die "failed to parse query count from ${run_output}"

    local -A avg_time_by_pair=()
    local -A qps_by_pair=()
    local -A recall_by_pair=()
    local -A gpu_mem_current_by_pair=()
    local -A gpu_mem_peak_by_pair=()
    local -A gpu_mem_total_by_pair=()
    local -A torch_peak_allocated_by_pair=()
    local -A torch_peak_reserved_by_pair=()
    local parsed_ncells=""
    local parsed_ndocs=""
    local avg_time_s=""
    local parsed_qps=""
    local recall=""
    local gpu_mem_current_mib=""
    local gpu_mem_peak_mib=""
    local gpu_mem_total_mib=""
    local torch_peak_allocated_mib=""
    local torch_peak_reserved_mib=""

    while IFS=, read -r parsed_ncells parsed_ndocs avg_time_s parsed_qps recall gpu_mem_current_mib gpu_mem_peak_mib gpu_mem_total_mib torch_peak_allocated_mib torch_peak_reserved_mib; do
        [[ "$parsed_ncells" == "ncells" ]] && continue
        local pair_key="${parsed_ncells},${parsed_ndocs}"
        avg_time_by_pair["$pair_key"]="$avg_time_s"
        qps_by_pair["$pair_key"]="$parsed_qps"
        recall_by_pair["$pair_key"]="$recall"
        gpu_mem_current_by_pair["$pair_key"]="$gpu_mem_current_mib"
        gpu_mem_peak_by_pair["$pair_key"]="$gpu_mem_peak_mib"
        gpu_mem_total_by_pair["$pair_key"]="$gpu_mem_total_mib"
        torch_peak_allocated_by_pair["$pair_key"]="$torch_peak_allocated_mib"
        torch_peak_reserved_by_pair["$pair_key"]="$torch_peak_reserved_mib"
    done < "$search_py_output_csv"

    local idx=0
    local label=""
    local ncells=""
    local ndocs=""
    local pair_key=""
    for idx in "${!labels[@]}"; do
        label="${labels[$idx]}"
        ncells="${ncells_values[$idx]}"
        ndocs="${ndocs_values[$idx]}"
        pair_key="${ncells},${ndocs}"

        [[ -n "${avg_time_by_pair[$pair_key]-}" ]] || die "missing ColBERT output for pair ${pair_key}"

        printf '%s\n' \
            "${implementation_label},${dataset_name},${label},${ncells},${ndocs},${index_source},${active_root_path},${active_experiment_dir}/indexes/${active_index_name},${query_file},${gt_file},${query_count},${k},3,${avg_time_by_pair[$pair_key]},${qps_by_pair[$pair_key]},${recall_by_pair[$pair_key]},${gpu_mem_current_by_pair[$pair_key]},${gpu_mem_peak_by_pair[$pair_key]},${gpu_mem_total_by_pair[$pair_key]},${torch_peak_allocated_by_pair[$pair_key]},${torch_peak_reserved_by_pair[$pair_key]}" \
            >> "$results_csv"
    done

    write_pareto_frontier "$results_csv" "$pareto_csv"
    log_line "[summary] results_csv=${results_csv}"
    log_line "[summary] pareto_csv=${pareto_csv}"
    log_line "[summary] pareto_points=$(tail -n +2 "$pareto_csv" | wc -l | tr -d '[:space:]')"
    rm -f "$run_output"
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"

config_file="${repo_root}/config/colbert_config.csv"
search_script="${repo_root}/experiment/plaid/search.py"
experiment_name="colbert"
index_name_override=""
implementation_label="colbert"
output_dir="${repo_root}/profiling"
log_dir="${repo_root}/log/bench"
env_name="colbert"
k=100
warmup=5
compressed_embeddings_storage="cpu"
gpu_index_resident=0
profile_breakdown_csv=""
dry_run=0
dataset_set=0
current_log_file=""
torch_extensions_dir=""
datasets=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            datasets+=("$2")
            dataset_set=1
            shift 2
            ;;
        --config-file)
            [[ $# -ge 2 ]] || die "missing value for --config-file"
            config_file="$2"
            shift 2
            ;;
        --env-name)
            [[ $# -ge 2 ]] || die "missing value for --env-name"
            env_name="$2"
            shift 2
            ;;
        --search-script)
            [[ $# -ge 2 ]] || die "missing value for --search-script"
            search_script="$2"
            shift 2
            ;;
        --experiment-name)
            [[ $# -ge 2 ]] || die "missing value for --experiment-name"
            experiment_name="$2"
            shift 2
            ;;
        --index-name)
            [[ $# -ge 2 ]] || die "missing value for --index-name"
            index_name_override="$2"
            shift 2
            ;;
        --implementation-label)
            [[ $# -ge 2 ]] || die "missing value for --implementation-label"
            implementation_label="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || die "missing value for --output-dir"
            output_dir="$2"
            shift 2
            ;;
        --log-dir)
            [[ $# -ge 2 ]] || die "missing value for --log-dir"
            log_dir="$2"
            shift 2
            ;;
        --k)
            [[ $# -ge 2 ]] || die "missing value for --k"
            k="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "missing value for --warmup"
            warmup="$2"
            shift 2
            ;;
        --compressed-embeddings-storage)
            [[ $# -ge 2 ]] || die "missing value for --compressed-embeddings-storage"
            compressed_embeddings_storage="$2"
            shift 2
            ;;
        --gpu-index-resident)
            gpu_index_resident=1
            shift
            ;;
        --profile-breakdown-csv)
            [[ $# -ge 2 ]] || die "missing value for --profile-breakdown-csv"
            profile_breakdown_csv="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            datasets+=("$1")
            dataset_set=1
            shift
            ;;
    esac
done

validate_path_component "$experiment_name" "experiment name"
validate_path_component "$implementation_label" "implementation label"
[[ -z "$index_name_override" ]] || validate_path_component "$index_name_override" "index name"
require_positive_int "--k" "$k"
(( k > 0 )) || die "--k must be > 0"
require_positive_int "--warmup" "$warmup"
case "$compressed_embeddings_storage" in
    cpu|gpu)
        ;;
    *)
        die "--compressed-embeddings-storage must be one of: cpu, gpu"
        ;;
esac
if [[ $gpu_index_resident -eq 1 && "$compressed_embeddings_storage" != "gpu" ]]; then
    die "--gpu-index-resident requires --compressed-embeddings-storage gpu"
fi
[[ -f "$config_file" ]] || die "config CSV not found: ${config_file}"
[[ -f "$search_script" ]] || die "search script not found: ${search_script}"

env_manager="$(find_env_manager)"
[[ -n "$env_manager" ]] || die "none of micromamba, mamba, or conda is available on PATH"
env_exists || die "environment ${env_name} is not available through ${env_manager}"
env_prefix="$(resolve_env_prefix)"
torch_extensions_dir="${repo_root}/.cache/torch_extensions/${env_name}"

if [[ $dataset_set -eq 0 ]]; then
    while IFS= read -r dataset_name; do
        [[ -n "$dataset_name" ]] && datasets+=("$dataset_name")
    done < <(discover_default_datasets)
fi

[[ ${#datasets[@]} -gt 0 ]] || die "no datasets selected"

for dataset_name in "${datasets[@]}"; do
    benchmark_dataset "$dataset_name"
done

if [[ $dry_run -eq 1 ]]; then
    echo
    echo "Dry run complete."
else
    echo
    echo "Benchmarking finished."
fi
