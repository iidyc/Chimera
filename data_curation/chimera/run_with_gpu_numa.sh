#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  data_curation/chimera/run_with_gpu_numa.sh [--mode auto|off|<node>] [--gpu <id-or-uuid>] -- <command> [args...]

Description:
  Run a command with CPU and memory placement bound to the NUMA node closest to
  the selected NVIDIA GPU. By default the selected GPU is the first entry in
  CUDA_VISIBLE_DEVICES, or GPU 0 when CUDA_VISIBLE_DEVICES is unset.

Environment:
  CHIMERA_NUMA_GPU              Override the selected GPU index or UUID.
  CHIMERA_NUMA_MEM_POLICY       membind, preferred, or cpubind. Default: membind.
EOF
}

die() {
    echo "error: $*" >&2
    exit 1
}

trim() {
    local value="$1"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    printf '%s' "$value"
}

normalize_pci_bus_id() {
    local bus_id="${1,,}"
    local domain="${bus_id%%:*}"
    local rest="${bus_id#*:}"

    if (( ${#domain} > 4 )); then
        domain="${domain: -4}"
    fi

    printf '%s:%s' "$domain" "$rest"
}

first_visible_gpu() {
    if [[ -n "${CHIMERA_NUMA_GPU-}" ]]; then
        printf '%s' "$CHIMERA_NUMA_GPU"
        return
    fi

    if [[ -v CUDA_VISIBLE_DEVICES ]]; then
        local cleaned="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
        if [[ -z "$cleaned" ]]; then
            printf '%s' ""
            return
        fi
        printf '%s' "${cleaned%%,*}"
        return
    fi

    printf '%s' "0"
}

gpu_pci_bus_id() {
    local selected_gpu="$1"
    local query_output line index pci uuid

    query_output="$(nvidia-smi --query-gpu=index,pci.bus_id,uuid --format=csv,noheader,nounits 2>/dev/null)" || return 1

    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        IFS=',' read -r index pci uuid <<< "$line"
        index="$(trim "$index")"
        pci="$(trim "$pci")"
        uuid="$(trim "$uuid")"

        if [[ "$selected_gpu" == "$index" || "$selected_gpu" == "$uuid" ]]; then
            printf '%s' "$pci"
            return 0
        fi
    done <<< "$query_output"

    return 1
}

sysfs_numa_node_for_pci() {
    local pci="$1"
    local sysfs_pci
    sysfs_pci="$(normalize_pci_bus_id "$pci")"

    if [[ -r "/sys/bus/pci/devices/${sysfs_pci}/numa_node" ]]; then
        cat "/sys/bus/pci/devices/${sysfs_pci}/numa_node"
        return 0
    fi

    return 1
}

topo_numa_node_for_gpu() {
    local selected_gpu="$1"

    [[ "$selected_gpu" =~ ^[0-9]+$ ]] || return 1
    nvidia-smi topo -m 2>/dev/null | awk -v gpu="GPU${selected_gpu}" '
        $1 == gpu {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^[0-9]+$/) {
                    node = $i
                }
            }
            if (node != "") {
                print node
                exit 0
            }
            exit 1
        }
    '
}

run_bound() {
    local node="$1"
    shift

    local policy="${CHIMERA_NUMA_MEM_POLICY:-membind}"
    local numactl_cmd=(numactl "--cpunodebind=${node}")

    case "$policy" in
        membind)
            numactl_cmd+=("--membind=${node}")
            ;;
        preferred)
            numactl_cmd+=("--preferred=${node}")
            ;;
        cpubind)
            ;;
        *)
            die "CHIMERA_NUMA_MEM_POLICY must be membind, preferred, or cpubind"
            ;;
    esac

    echo "[numa] node=${node} policy=${policy} command=$(printf '%q ' "$@")" >&2
    exec "${numactl_cmd[@]}" -- "$@"
}

mode="auto"
selected_gpu=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || die "missing value for --mode"
            mode="$2"
            shift 2
            ;;
        --gpu)
            [[ $# -ge 2 ]] || die "missing value for --gpu"
            selected_gpu="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            die "unknown option before --: $1"
            ;;
    esac
done

[[ $# -gt 0 ]] || die "missing command after --"

if [[ "$mode" == "off" ]]; then
    exec "$@"
fi

if [[ "$mode" != "auto" && ! "$mode" =~ ^[0-9]+$ ]]; then
    die "--mode must be auto, off, or a NUMA node id"
fi

if [[ "$mode" =~ ^[0-9]+$ ]]; then
    command -v numactl >/dev/null 2>&1 || die "numactl is not available on PATH"
    run_bound "$mode" "$@"
fi

command -v nvidia-smi >/dev/null 2>&1 || {
    echo "[numa] nvidia-smi unavailable; running without NUMA binding" >&2
    exec "$@"
}
command -v numactl >/dev/null 2>&1 || {
    echo "[numa] numactl unavailable; running without NUMA binding" >&2
    exec "$@"
}

if [[ -z "$selected_gpu" ]]; then
    selected_gpu="$(first_visible_gpu)"
fi

if [[ -z "$selected_gpu" ]]; then
    echo "[numa] CUDA_VISIBLE_DEVICES disables GPUs; running without NUMA binding" >&2
    exec "$@"
fi

pci_bus_id=""
if ! pci_bus_id="$(gpu_pci_bus_id "$selected_gpu")"; then
    echo "[numa] could not resolve GPU ${selected_gpu} PCI bus; running without NUMA binding" >&2
    exec "$@"
fi

numa_node=""
if numa_node="$(sysfs_numa_node_for_pci "$pci_bus_id")" && [[ "$numa_node" =~ ^[0-9]+$ ]]; then
    :
else
    numa_node="$(topo_numa_node_for_gpu "$selected_gpu" || true)"
fi

if [[ ! "$numa_node" =~ ^[0-9]+$ ]]; then
    echo "[numa] GPU ${selected_gpu} PCI ${pci_bus_id} has no usable NUMA node (${numa_node:-unknown}); running without NUMA binding" >&2
    exec "$@"
fi

echo "[numa] selected_gpu=${selected_gpu} pci_bus_id=${pci_bus_id} numa_node=${numa_node}" >&2
run_bound "$numa_node" "$@"
