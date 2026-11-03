"""Check the CUDA-visible device and its relationship to the process CPU mask."""
import argparse
import csv
import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path


def cpu_list(value):
    result = set()
    for part in value.strip().split(','):
        ends = list(map(int, part.split('-')))
        result.update(range(ends[0], ends[-1] + 1))
    return result


def gpu_numa(pci, uuid):
    device = Path(f'/sys/bus/pci/devices/{pci}').resolve()
    for parent in [device, *device.parents]:
        path = parent / 'numa_node'
        if path.exists() and int(path.read_text()) >= 0:
            node = int(path.read_text())
            print(f'GPU_NUMA_PROOF pci={pci} node={node} sysfs={path}', file=sys.stderr)
            return node
    devices = csv.reader(subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True).splitlines(),
        skipinitialspace=True)
    index, = [int(index) for index, device_uuid in devices if device_uuid == uuid]
    topology = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    print(topology, file=sys.stderr)
    row, = [line.split() for line in topology.splitlines() if line.split()[:1] == [f'GPU{index}']]
    # NVIDIA's trailing columns are CPU Affinity, NUMA Affinity, GPU NUMA ID.
    if len(row) < 4 or not re.fullmatch(r'[0-9]+(?:[-,][0-9]+)*', row[-3]):
        raise RuntimeError(f'Unrecognized NVIDIA topology row: {row}')
    affinity = cpu_list(row[-3])
    nodes = [path for path in Path('/sys/devices/system/node').glob('node[0-9]*')
             if affinity <= cpu_list((path / 'cpulist').read_text())]
    if len(nodes) != 1 or (row[-2] != 'N/A' and row[-2] != nodes[0].name[4:]):
        raise RuntimeError(f'Cannot prove a unique GPU-local NUMA node from {row}')
    node = int(nodes[0].name[4:])
    print(f'GPU_NUMA_PROOF pci={pci} node={node} source=nvidia_topology', file=sys.stderr)
    return node


def probe(platform=None):
    runtime = ctypes.CDLL('libcudart.so')
    runtime.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    runtime.cudaDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError(f'Expected exactly one CUDA-visible GPU; got {count.value}')
    bus = ctypes.create_string_buffer(64)
    if runtime.cudaDeviceGetPCIBusId(bus, len(bus), 0) != 0:
        raise RuntimeError('Cannot resolve CUDA device 0 PCI bus')
    pci = bus.value.decode().lower()
    gpu, memory, uuid = next(csv.reader([subprocess.check_output([
        'nvidia-smi', '-i', pci, '--query-gpu=name,memory.total,uuid',
        '--format=csv,noheader,nounits'], text=True).strip()], skipinitialspace=True))
    numa = gpu_numa(pci, uuid)
    cpus = os.sched_getaffinity(0)
    local = cpu_list(Path(f'/sys/devices/system/node/node{numa}/cpulist').read_text())
    if not cpus or not cpus <= local:
        raise RuntimeError(f'CPU mask {sorted(cpus)} is not GPU-local to NUMA {numa}')
    physical = set()
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple((topology / name).read_text().strip()
                    for name in ['physical_package_id', 'core_id'])
        if key in physical:
            raise RuntimeError('CPU mask contains SMT siblings')
        physical.add(key)
    cpu_model = next(line.split(':', 1)[1].strip() for line in
                     Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    if platform:
        gpu_name, cpu_name, min_memory, cores = {
            'a100': ('A100', '8352Y', 79000, 32),
            'v100': ('V100', '6240', 31000, 18),
        }[platform]
        if not (gpu_name in gpu and cpu_name in cpu_model and float(memory) >= min_memory
                and len(cpus) == cores):
            raise RuntimeError(f'Wrong benchmark platform: {gpu}, {memory} MiB, {cpu_model}, {len(cpus)} cores')
    return [gpu, cpu_model, str(numa), ','.join(map(str, sorted(cpus))), pci, uuid]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--platform', choices=['a100', 'v100'])
    args = parser.parse_args()
    print('\n'.join(probe(args.platform)))
