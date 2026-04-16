#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  script/bench_gpu_mvr.sh [--dataset <name> ...] [options]
  script/bench_gpu_mvr.sh [dataset_name ...] [options]

Description:
  Benchmark a GPU-MVR gpu_search version on one or more datasets and write per-dataset
  recall-vs-qps measurements plus a Pareto frontier CSV.

  By default, the script benchmarks:
    hotpot
    lotte

  It reads search configurations from profiling/gpu_mvr_config.csv and writes:
    profiling/<dataset>/<implementation>/benchmark_results.csv
    profiling/<dataset>/<implementation>/pareto_frontier.csv
    log/bench/<implementation>/<dataset>/benchmark.log

  Optional /tmp staging can copy dataset/<name>/gpu_mvr_index into local
  storage first and reuse the copy across subsequent runs.

Options:
  --dataset <name>         Dataset under dataset/. Repeatable.
  --version <v0|v1|v2|v3>  GPU-MVR search version. Default: v3
  --implementation-label <label>
                           Output folder label. Default: gpu_search_<version>
  --config-file <path>     Config CSV. Default: profiling/gpu_mvr_config.csv
  --build-dir <path>       Dedicated build directory. Default: gpu-mvr/build
  --binary <path>          Search binary. Overrides --build-dir and --version.
                           Default: <build-dir>/gpu_search_<version>
  --output-dir <path>      CSV output directory. Default: profiling
  --log-dir <path>         Log output directory. Default: log/bench
  --k <top_k>              Final retrieval depth / recall depth. Default: 100
  --nq <count>             Evaluation queries after warmup. Default: -1
  --warmup <count>         Warmup query count. Default: 5
  --copy-index-to-tmp      Copy gpu_mvr_index into /tmp before benchmarking.
  --refresh-tmp-index      Re-copy the /tmp index even if it already exists.
  --tmp-root <path>        Root directory for /tmp index copies.
                           Default: /tmp/$USER/gpu_mvr_benchmark
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

run_cmd() {
    echo "+ $(printf '%q ' "$@")"
    if [[ $dry_run -eq 0 ]]; then
        "$@"
    fi
}

log_line() {
    local line="$1"
    if [[ $dry_run -eq 1 ]]; then
        echo "$line"
    else
        echo "$line" | tee -a "$current_log_file"
    fi
}

log_capture_safe_line() {
    local line="$1"
    if [[ $dry_run -eq 1 ]]; then
        echo "$line" >&2
    else
        echo "$line" | tee -a "$current_log_file" >&2
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
    log_line "[driver] slurm_job_id=${SLURM_JOB_ID-<unset>}"
    log_line "[driver] slurm_job_nodelist=${SLURM_JOB_NODELIST-<unset>}"

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

extract_last_match_or_die() {
    local source_file="$1"
    local sed_expr="$2"
    local description="$3"
    local value=""
    value="$(sed -nE "$sed_expr" "$source_file" | tail -n 1)"
    [[ -n "$value" ]] || die "failed to parse ${description} from ${source_file}"
    printf '%s' "$value"
}

copy_index_to_tmp_if_needed() {
    local dataset_name="$1"
    local source_index_dir="$2"
    local dest_index_dir="${tmp_root}/${dataset_name}/gpu_mvr_index"

    if [[ $copy_index_to_tmp -eq 0 ]]; then
        printf '%s' "$source_index_dir"
        return
    fi

    if [[ -d "$dest_index_dir" && $refresh_tmp_index -eq 0 \
        && -f "$dest_index_dir/cpu_index.bin" \
        && -f "$dest_index_dir/gpu_index.bin" \
        && -f "$dest_index_dir/ivf.bin" \
        && -f "$dest_index_dir/centroids.carga" ]]; then
        log_capture_safe_line "[driver] reusing_tmp_index=${dest_index_dir}"
        printf '%s' "$dest_index_dir"
        return
    fi

    log_capture_safe_line "[driver] tmp_index_source=${source_index_dir}"
    log_capture_safe_line "[driver] tmp_index_target=${dest_index_dir}"

    if [[ $dry_run -eq 1 ]]; then
        if command -v rsync >/dev/null 2>&1; then
            echo "+ rsync -a --delete ${source_index_dir}/ ${dest_index_dir}/" >&2
        else
            echo "+ rm -rf ${dest_index_dir}" >&2
            echo "+ mkdir -p $(dirname "$dest_index_dir") ${dest_index_dir}" >&2
            echo "+ cp -a ${source_index_dir}/. ${dest_index_dir}/" >&2
        fi
        printf '%s' "$dest_index_dir"
        return
    fi

    local parent_dir
    parent_dir="$(dirname "$dest_index_dir")"
    mkdir -p "$parent_dir"

    local source_size_bytes
    local available_bytes
    source_size_bytes="$(du -sb "$source_index_dir" | awk '{print $1}')"
    available_bytes="$(df -B1 "$parent_dir" | awk 'NR==2 {print $4}')"
    if (( available_bytes < source_size_bytes )); then
        die "not enough free space under ${parent_dir} to copy ${source_index_dir}"
    fi

    local copy_start_utc
    local copy_end_utc
    local copy_start_s
    local copy_end_s
    copy_start_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    copy_start_s="$(date +%s)"
    log_capture_safe_line "[driver] tmp_copy_start_utc=${copy_start_utc}"

    if command -v rsync >/dev/null 2>&1; then
        echo "+ $(printf '%q ' rsync -a --delete "${source_index_dir}/" "${dest_index_dir}/")" | tee -a "$current_log_file" >&2
        rsync -a --delete "${source_index_dir}/" "${dest_index_dir}/"
    else
        echo "+ $(printf '%q ' rm -rf "$dest_index_dir")" | tee -a "$current_log_file" >&2
        rm -rf "$dest_index_dir"
        echo "+ $(printf '%q ' mkdir -p "$dest_index_dir")" | tee -a "$current_log_file" >&2
        mkdir -p "$dest_index_dir"
        echo "+ $(printf '%q ' cp -a "${source_index_dir}/." "${dest_index_dir}/")" | tee -a "$current_log_file" >&2
        cp -a "${source_index_dir}/." "${dest_index_dir}/"
    fi

    copy_end_utc="$(date -u +%Y%m%dT%H%M%SZ)"
    copy_end_s="$(date +%s)"
    log_capture_safe_line "[driver] tmp_copy_end_utc=${copy_end_utc}"
    log_capture_safe_line "[driver] tmp_copy_elapsed_seconds=$((copy_end_s - copy_start_s))"
    log_capture_safe_line "[driver] tmp_index_size=$(du -sh "$dest_index_dir" | awk '{print $1}')"

    printf '%s' "$dest_index_dir"
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
                qps[n] = $16 + 0.0
                recall[n] = $17 + 0.0
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
        ' | sort -t, -k16,16g -k17,17g
    } > "$pareto_csv"
}

benchmark_dataset() {
    local dataset_name="$1"

    validate_dataset_name "$dataset_name"

    local dataset_dir="${repo_root}/dataset/${dataset_name}"
    local raw_dir="${dataset_dir}/raw"
    local source_index_dir="${dataset_dir}/gpu_mvr_index"
    local query_file="${raw_dir}/query.bin"
    local doclens_file="${raw_dir}/doclens.bin"
    local gt_file="${raw_dir}/gt.tsv"
    local impl_output_dir="${output_dir}/${dataset_name}/${implementation_label}"
    local impl_log_dir="${log_dir}/${implementation_label}/${dataset_name}"
    local results_csv="${impl_output_dir}/benchmark_results.csv"
    local pareto_csv="${impl_output_dir}/pareto_frontier.csv"
    local log_file="${impl_log_dir}/benchmark.log"
    local index_source="nfs"
    local active_index_dir=""

    [[ -d "$dataset_dir" ]] || die "dataset directory not found: ${dataset_dir}"
    [[ -f "$query_file" ]] || die "missing query file: ${query_file}"
    [[ -f "$doclens_file" ]] || die "missing doclens file: ${doclens_file}"
    [[ -f "$gt_file" ]] || die "missing ground truth file: ${gt_file}"
    [[ -d "$source_index_dir" ]] || die "missing index directory: ${source_index_dir}"

    current_log_file="$log_file"

    if [[ $dry_run -eq 0 ]]; then
        mkdir -p "$impl_output_dir" "$impl_log_dir"
        : > "$log_file"
    fi

    active_index_dir="$(copy_index_to_tmp_if_needed "$dataset_name" "$source_index_dir")"
    if [[ "$active_index_dir" != "$source_index_dir" ]]; then
        index_source="tmp"
    fi

    log_line "[driver] repo_root=${repo_root}"
    log_line "[driver] implementation=${implementation_label}"
    log_line "[driver] dataset=${dataset_name}"
    log_line "[driver] config_file=${config_file}"
    log_line "[driver] build_dir=${build_dir}"
    log_line "[driver] binary=${binary}"
    log_line "[driver] query_file=${query_file}"
    log_line "[driver] doclens_file=${doclens_file}"
    log_line "[driver] gt_file=${gt_file}"
    log_line "[driver] source_index_dir=${source_index_dir}"
    log_line "[driver] active_index_dir=${active_index_dir}"
    log_line "[driver] index_source=${index_source}"
    log_line "[driver] results_csv=${results_csv}"
    log_line "[driver] pareto_csv=${pareto_csv}"
    log_line "[driver] k=${k}"
    log_line "[driver] nq=${nq}"
    log_line "[driver] warmup=${warmup}"
    log_line "[driver] copy_index_to_tmp=${copy_index_to_tmp}"
    log_line "[driver] refresh_tmp_index=${refresh_tmp_index}"
    log_line "[driver] tmp_root=${tmp_root}"
    print_hardware_info

    if [[ $dry_run -eq 0 ]]; then
        printf '%s\n' \
            "implementation,dataset,label,nprobe,k_rank_cluster,k_rank_all_tokens,itopk_size,overlap_chunks,index_source,index_path,queries,k,warmup,search_seconds,end_to_end_ms,qps,recall,avg_latency_ms,p50_ms,p90_ms,p95_ms,p99_ms,max_ms,stddev_ms" \
            > "$results_csv"
    fi

    while IFS=, read -r raw_label raw_nprobe raw_k_rank_cluster raw_k_rank_all_tokens raw_itopk_size raw_overlap_chunks; do
        local label=""
        local nprobe=""
        local k_rank_cluster=""
        local k_rank_all_tokens=""
        local itopk_size=""
        local overlap_chunks=""
        local run_output=""
        local status=0
        local run_start_utc=""
        local run_end_utc=""
        local search_seconds=""
        local queries=""
        local end_to_end_ms=""
        local qps=""
        local recall=""
        local avg_latency_ms=""
        local p50_ms=""
        local p90_ms=""
        local p95_ms=""
        local p99_ms=""
        local max_ms=""
        local stddev_ms=""
        local command=()

        label="$(trim_field "$raw_label")"
        [[ -z "$label" || "$label" == "label" || "$label" == \#* ]] && continue

        nprobe="$(trim_field "$raw_nprobe")"
        k_rank_cluster="$(trim_field "$raw_k_rank_cluster")"
        k_rank_all_tokens="$(trim_field "$raw_k_rank_all_tokens")"
        itopk_size="$(trim_field "$raw_itopk_size")"
        overlap_chunks="$(trim_field "$raw_overlap_chunks")"

        require_positive_int "nprobe" "$nprobe"
        require_positive_int "k_rank_cluster" "$k_rank_cluster"
        require_positive_int "k_rank_all_tokens" "$k_rank_all_tokens"
        require_positive_int "itopk_size" "$itopk_size"
        require_positive_int "overlap_chunks" "$overlap_chunks"
        (( nprobe > 0 )) || die "nprobe must be > 0"
        (( k_rank_cluster > 0 )) || die "k_rank_cluster must be > 0"
        (( k_rank_all_tokens > 0 )) || die "k_rank_all_tokens must be > 0"
        (( itopk_size > 0 )) || die "itopk_size must be > 0"
        (( overlap_chunks > 0 )) || die "overlap_chunks must be > 0"

        command=(
            "$binary"
            --query "$query_file"
            --doclens "$doclens_file"
            --gt "$gt_file"
            --index "$active_index_dir"
            --k "$k"
            --nq "$nq"
            --warmup "$warmup"
            --nprobe "$nprobe"
            --k-rank-cluster "$k_rank_cluster"
            --k-rank-all-tokens "$k_rank_all_tokens"
            --itopk-size "$itopk_size"
            --overlap-chunks "$overlap_chunks"
        )

        log_line "[run] dataset=${dataset_name} label=${label}"
        log_line "[run] command=$(printf '%q ' "${command[@]}")"

        if [[ $dry_run -eq 1 ]]; then
            continue
        fi

        run_output="$(mktemp "${TMPDIR:-/tmp}/gpu_search_${version}.${dataset_name}.${label}.XXXXXX.log")"
        run_start_utc="$(date -u +%Y%m%dT%H%M%SZ)"
        log_line "[run] start_utc=${run_start_utc}"

        set +e
        "${command[@]}" > "$run_output" 2>&1
        status=$?
        set -e

        cat "$run_output" | tee -a "$current_log_file"
        run_end_utc="$(date -u +%Y%m%dT%H%M%SZ)"
        log_line "[run] end_utc=${run_end_utc}"
        log_line "[run] exit_code=${status}"

        if [[ $status -ne 0 ]]; then
            rm -f "$run_output"
            die "gpu_search_${version} failed for dataset=${dataset_name} label=${label}; see ${current_log_file}"
        fi

        search_seconds="$(extract_last_match_or_die "$run_output" 's/^\[([0-9.]+) s\] GPU search time for [0-9]+ queries\.$/\1/p' 'search seconds')"
        queries="$(extract_last_match_or_die "$run_output" 's/^\[[0-9.]+ s\] GPU search time for ([0-9]+) queries\.$/\1/p' 'query count')"
        end_to_end_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] End-to-end measured time: ([0-9.]+) ms$/\1/p' 'end-to-end time')"
        qps="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Throughput: ([0-9.]+) qps$/\1/p' 'throughput')"
        recall="$(extract_last_match_or_die "$run_output" 's/^Recall@[0-9]+: ([0-9.]+)$/\1/p' 'recall')"
        avg_latency_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Average latency per query: ([0-9.]+) ms$/\1/p' 'average latency')"
        p50_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Query latency distribution \(ms\): .*p50=([0-9.]+),.*$/\1/p' 'p50 latency')"
        p90_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Query latency distribution \(ms\): .*p90=([0-9.]+),.*$/\1/p' 'p90 latency')"
        p95_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Query latency distribution \(ms\): .*p95=([0-9.]+),.*$/\1/p' 'p95 latency')"
        p99_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Query latency distribution \(ms\): .*p99=([0-9.]+),.*$/\1/p' 'p99 latency')"
        max_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Query latency distribution \(ms\): .*max=([0-9.]+),.*$/\1/p' 'max latency')"
        stddev_ms="$(extract_last_match_or_die "$run_output" 's/^\[SEARCH\] Query latency distribution \(ms\): .*stddev=([0-9.]+)$/\1/p' 'latency stddev')"

        printf '%s\n' \
            "${implementation_label},${dataset_name},${label},${nprobe},${k_rank_cluster},${k_rank_all_tokens},${itopk_size},${overlap_chunks},${index_source},${active_index_dir},${queries},${k},${warmup},${search_seconds},${end_to_end_ms},${qps},${recall},${avg_latency_ms},${p50_ms},${p90_ms},${p95_ms},${p99_ms},${max_ms},${stddev_ms}" \
            >> "$results_csv"

        log_line "[run] parsed_metrics label=${label} qps=${qps} recall=${recall} avg_latency_ms=${avg_latency_ms}"
        rm -f "$run_output"
    done < <(tr -d '\r' < "$config_file")

    if [[ $dry_run -eq 0 ]]; then
        write_pareto_frontier "$results_csv" "$pareto_csv"
        log_line "[summary] results_csv=${results_csv}"
        log_line "[summary] pareto_csv=${pareto_csv}"
        log_line "[summary] pareto_points=$(tail -n +2 "$pareto_csv" | wc -l | tr -d '[:space:]')"
        log_line "[summary] pareto_frontier_begin"
        tail -n +2 "$pareto_csv" | tee -a "$current_log_file"
        log_line "[summary] pareto_frontier_end"
    fi
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

config_file="${repo_root}/profiling/gpu_mvr_config.csv"
build_dir="${repo_root}/gpu-mvr/build"
binary=""
output_dir="${repo_root}/profiling"
log_dir="${repo_root}/log/bench"
tmp_root="/tmp/${USER:-user}/gpu_mvr_benchmark"
version="v3"
implementation_label=""
k=100
nq=-1
warmup=5
copy_index_to_tmp=0
refresh_tmp_index=0
dry_run=0
dataset_set=0
binary_set=0
implementation_label_set=0
current_log_file=""
datasets=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)
            [[ $# -ge 2 ]] || die "missing value for --dataset"
            datasets+=("$2")
            dataset_set=1
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || die "missing value for --version"
            version="$2"
            shift 2
            ;;
        --implementation-label)
            [[ $# -ge 2 ]] || die "missing value for --implementation-label"
            implementation_label="$2"
            implementation_label_set=1
            shift 2
            ;;
        --config-file)
            [[ $# -ge 2 ]] || die "missing value for --config-file"
            config_file="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "missing value for --build-dir"
            build_dir="$2"
            shift 2
            ;;
        --binary)
            [[ $# -ge 2 ]] || die "missing value for --binary"
            binary="$2"
            binary_set=1
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
        --nq)
            [[ $# -ge 2 ]] || die "missing value for --nq"
            nq="$2"
            shift 2
            ;;
        --warmup)
            [[ $# -ge 2 ]] || die "missing value for --warmup"
            warmup="$2"
            shift 2
            ;;
        --copy-index-to-tmp)
            copy_index_to_tmp=1
            shift
            ;;
        --refresh-tmp-index)
            refresh_tmp_index=1
            shift
            ;;
        --tmp-root)
            [[ $# -ge 2 ]] || die "missing value for --tmp-root"
            tmp_root="$2"
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

if [[ $dataset_set -eq 0 ]]; then
    datasets=(hotpot lotte)
fi

case "$version" in
    v0|v1|v2|v3)
        ;;
    *)
        die "--version must be one of: v0, v1, v2, v3"
        ;;
esac

if [[ $implementation_label_set -eq 0 ]]; then
    implementation_label="gpu_search_${version}"
fi

validate_path_component "$implementation_label" "implementation label"

if [[ $binary_set -eq 0 ]]; then
    binary="${build_dir}/gpu_search_${version}"
fi

require_positive_int "--k" "$k"
[[ "$nq" =~ ^-?[0-9]+$ ]] || die "--nq must be an integer"
require_positive_int "--warmup" "$warmup"
(( k > 0 )) || die "--k must be > 0"
(( warmup >= 0 )) || die "--warmup must be >= 0"
(( nq >= -1 )) || die "--nq must be >= -1"

[[ -f "$config_file" ]] || die "config CSV not found: ${config_file}"
if [[ $dry_run -eq 0 ]]; then
    [[ -x "$binary" ]] || die "search binary not found or not executable: ${binary}. Build gpu_search_${version} in a dedicated build directory such as ${build_dir}"
elif [[ ! -x "$binary" ]]; then
    echo "[dry-run] warning: binary not found or not executable: ${binary}"
fi

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
