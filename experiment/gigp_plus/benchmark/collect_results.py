"""Export measured points and coverage; never infer missing benchmark values."""
import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


def pareto(rows):
    return [r for r in rows if not any(
        s['recall_at_100'] >= r['recall_at_100'] and s['qps'] >= r['qps']
        and (s['recall_at_100'] > r['recall_at_100'] or s['qps'] > r['qps'])
        for s in rows)]


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    if not keys:
        keys = ['system', 'gpu', 'dataset']
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def verify_chimera_affinity(log, hardware):
    groups = re.search(r'slot_cpu_groups=(\S+) stage3_threads_per_slot=(\d+)', hardware)
    assert groups, 'Missing slot CPU groups'
    expected = [set(map(int, group.split(','))) for group in groups[1].split(';')]
    threads = int(groups[2])
    assert len(expected) == 3 and threads > 0
    assert all(len(group) >= threads for group in expected)
    assert sum(map(len, expected)) == len(set.union(*expected)), 'Overlapping slot CPU groups'
    observed = [set() for _ in expected]
    for slot, thread, cpus in re.findall(r'\[AFFINITY_TEAM\] slot=(\d+) thread=(\d+) cpus=([0-9,]+)', log):
        slot, thread = int(slot), int(thread)
        assert slot in range(3) and thread not in observed[slot]
        assert set(map(int, cpus.split(','))) == expected[slot], 'Unbound OpenMP team member'
        observed[slot].add(thread)
    assert all(team == set(range(threads)) for team in observed), 'Incomplete OpenMP team affinity proof'


def verify_paper_run(raw, folder, hardware, build_log, manifest, reference):
    assert raw['bits'] == manifest['bits'] == 2
    assert raw['source_commit'] == '1899e9316e71b023518f9090fc64fe66dbd0c202'
    assert raw['query_sha256'] == manifest['query_sha256'] == reference['query_sha256']
    assert manifest['source_gt'] == reference['source_gt'], 'Different ground-truth source'
    assert manifest['centroids_reused'] and manifest['residual_sha256']
    command = json.loads((folder / 'command.json').read_text())
    for flag, value in [('-num-bit', 2), ('-nprobe', raw['nprobe']),
                        ('-probe_topk', raw['probe_topk']), ('-num-centroid', manifest['n_centroid'])]:
        assert command[command.index(flag) + 1] == str(value), f'Command mismatch: {flag}'
    assert raw['binary_sha256'] in hardware, 'Binary does not match hardware/build record'
    assert 'four GPU/CPU reference checks passed' in build_log
    assert '86bc9805062928b5b51f3b7a7e4d381cd68bf2a919f07abf4bd62b339040fc58' in build_log
    mask = re.search(r'allowed_cpus=([0-9,]+)', hardware)
    assert mask, 'Missing allocated CPU mask'
    allowed = set(map(int, mask[1].split(',')))
    affinity = re.findall(r'level 1 thread (\S+) affinity (\d+)\s*$',
                          (folder / 'stdout.log').read_text(), re.MULTILINE)[-3:]
    assert len(affinity) == len({thread for thread, _ in affinity}) == 3
    cpus = {int(cpu) for _, cpu in affinity}
    assert len(cpus) == 3 and cpus <= allowed, 'Invalid query-worker CPU pinning'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gigp-bits', type=int, choices=[2, 4], default=4)
    p.add_argument('--min-recall', type=float, default=.90)
    p.add_argument('--require-paper-summary', action='store_true',
                   help='Only include paper-precision folders with final summary/frontier artifacts')
    a = p.parse_args()
    if not 0 <= a.min_recall <= 1:
        p.error('--min-recall must be between zero and one')
    if a.require_paper_summary and a.gigp_bits != 2:
        p.error('--require-paper-summary requires --gigp-bits 2')
    a.output.mkdir(parents=True, exist_ok=True)
    runs, rejected = [], []
    measurement_root = a.root / ('paper2bit_results' if a.gigp_bits == 2 else 'results')
    manifests = {}
    for path in sorted(measurement_root.rglob('*.json')):
        if path.name != 'result.json' and path.parent.parent.name != 'checkpoints':
            continue
        # Measurement folders are <root>/results/<system_gpu_dataset_bits>/<mode>.
        relative = path.relative_to(measurement_root)
        cohort, mode = relative.parts[:2]
        if a.require_paper_summary:
            folder = measurement_root / cohort / mode
            if not all((folder / name).is_file() for name in ['summary.csv', 'paper_pareto.csv']):
                continue
        try:
            if cohort.startswith('chimera3_'):
                _, gpu, dataset, _ = cohort.split('_')
                saved = json.loads(path.read_text())
                raw, proof = saved['row'], saved['provenance']
                assert raw['returncode'] == '0' and raw['effective_concurrent_queries'] == '3'
                system = 'Chimera'
                config = json.dumps(proof['config'], sort_keys=True)
                query_sha = proof['query_sha256']
                source_sha = proof['source_sha256']
                repeat = int(raw['run_index'])
                job = re.search(r'job(\d+)', raw['log']).group(1)
                cache = path.parents[2] / 'build_metadata' / f'job{job}' / 'CMakeCache.txt'
                cache_text = cache.read_text()
                assert 'CHIMERA_ENABLE_PROFILE:BOOL=OFF' in cache_text
                assert 'CHIMERA_ARCH_FLAGS:STRING=-march=native' in cache_text
                log_text = Path(raw['log']).read_text()
                assert '[AFFINITY]' in log_text
            else:
                gpu, dataset, _ = cohort.split('_')
                raw = json.loads(path.read_text())
                assert raw['concurrent_queries'] == 3 and raw['bits'] == a.gigp_bits
                system = 'GIGP+'
                config = json.dumps({k: raw[k] for k in ['nprobe', 'probe_topk']}, sort_keys=True)
                query_sha = raw['query_sha256']
                source_sha = raw['source_commit']
                repeat = int(raw['repeat'])
                job = raw['slurm_job_id']
                cache = a.root / 'results/build_metadata' / f'job{job}' / 'CMakeCache.txt'
                cache_text = cache.read_text()
                assert 'GIGP_CPU_ARCH:STRING=native' in cache_text
                assert 'affinity' in (path.parent / 'stdout.log').read_text().lower()
            assert re.search(r'CMAKE_CUDA_ARCHITECTURES:[^=\n]+=native(?:\n|$)', cache_text)
            hardware = measurement_root / cohort / mode / f'hardware_{job}.txt'
            hardware_text = hardware.read_text()
            if system == 'Chimera':
                verify_chimera_affinity(log_text, hardware_text)
            elif a.gigp_bits == 2:
                if dataset not in manifests:
                    index, = (a.root / 'data_paper2bit/Index' / dataset).glob('*-n_bit_2')
                    ref, = (a.root / 'data/Index' / dataset).glob('*-n_bit_4')
                    manifests[dataset] = (json.loads((index / 'manifest.json').read_text()),
                                          json.loads((ref / 'manifest.json').read_text()))
                verify_paper_run(raw, path.parent, hardware_text,
                                 (hardware.parent / f'build_{job}.log').read_text(),
                                 *manifests[dataset])
            assert ('A100' in hardware_text and '8352Y' in hardware_text) if gpu == 'a100' else ('V100' in hardware_text and '6240' in hardware_text)
            row = dict(system=system, gpu=gpu, dataset=dataset, bits=a.gigp_bits, concurrent_queries=3,
                       config=config, repeat=repeat, mode=mode, query_sha256=query_sha,
                       source_signature=source_sha, job_id=job, checkpoint=str(path))
            for key in ['recall_at_100', 'qps', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms']:
                row[key] = float(raw[key])
                assert math.isfinite(row[key])
            assert 0 <= row['recall_at_100'] <= 1 and row['qps'] > 0
            runs.append(row)
        except (KeyError, ValueError, AssertionError, AttributeError, OSError) as e:
            rejected.append({'checkpoint': str(path), 'reason': f'{type(e).__name__}: {e}'})
    groups = defaultdict(list)
    for row in runs:
        groups[tuple(row[k] for k in ['system', 'gpu', 'dataset', 'config', 'mode', 'query_sha256', 'source_signature'])].append(row)
    points = []
    for group in groups.values():
        assert len({r['repeat'] for r in group}) == len(group), 'Duplicate repeat in cohort'
        row = {k: group[0][k] for k in ['system', 'gpu', 'dataset', 'bits', 'concurrent_queries', 'config', 'mode', 'query_sha256', 'source_signature']}
        row['runs'] = len(group)
        row['three_runs_verified'] = {r['repeat'] for r in group} == {1, 2, 3}
        for key in ['recall_at_100', 'qps', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms']:
            row[key] = statistics.mean(r[key] for r in group)
            row[f'{key}_std'] = statistics.stdev(r[key] for r in group) if len(group) > 1 else ''
        points.append(row)
    preferred = {}
    for row in points:
        if not row['three_runs_verified'] or row['recall_at_100'] < a.min_recall:
            continue
        key = tuple(row[k] for k in ['system', 'gpu', 'dataset', 'config', 'query_sha256', 'source_signature'])
        # Prefer a completed full sweep to its pilot, independently of measured speed.
        if key not in preferred or (row['mode'] == 'full' and preferred[key]['mode'] != 'full'):
            preferred[key] = row
    verified = list(preferred.values())
    fronts, coverage = [], []
    for gpu in ['a100', 'v100']:
        for dataset in ['lotte', 'hotpot', 'msmarco']:
            for system in (['GIGP+'] if a.gigp_bits == 2 else ['GIGP+', 'Chimera']):
                matching = [r for r in verified if r['gpu'] == gpu and r['dataset'] == dataset and r['system'] == system]
                fronts.extend(pareto(matching))
                coverage.append(dict(gpu=gpu, dataset=dataset, system=system,
                                     verified_points=len(matching),
                                     reaches_090=any(r['recall_at_100'] >= .90 for r in matching),
                                     reaches_095=any(r['recall_at_100'] >= .95 for r in matching)))
    for name, rows in [('runs.csv', runs), ('points.csv', points), ('verified_frontier.csv', fronts),
                       ('coverage.csv', coverage), ('rejected.csv', rejected)]:
        write_csv(a.output / name, rows)
    print(json.dumps(dict(validated_runs=len(runs), measured_points=len(points),
                         verified_frontier_points=len(fronts), rejected=len(rejected))))


if __name__ == '__main__':
    main()
