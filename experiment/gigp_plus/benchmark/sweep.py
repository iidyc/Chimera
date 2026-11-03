"""Run the authors' three-worker scheduler and retain per-query measurements."""
import argparse
import csv
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


def frontier(rows):
    return [r for r in rows if not any(
        s['recall_at_100'] >= r['recall_at_100'] and s['qps'] >= r['qps']
        and (s['recall_at_100'] > r['recall_at_100'] or s['qps'] > r['qps'])
        for s in rows)]


def paper_confirmation_requests(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row['nprobe'], row['probe_topk']), {})[row['repeat']] = row
    means = []
    for (probes, refine), repeats in grouped.items():
        means.append(dict(nprobe=probes, probe_topk=refine,
                          recall_at_100=float(np.mean([r['recall_at_100'] for r in repeats.values()])),
                          qps=float(np.mean([r['qps'] for r in repeats.values()]))))
    return [(r['nprobe'], r['probe_topk'], repeat)
            for r in frontier(means) for repeat in (2, 3)
            if repeat not in grouped[(r['nprobe'], r['probe_topk'])]]


def priority_configs(path, dataset, query_sha256):
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if data['dataset'] != dataset or data['query_sha256'] != query_sha256:
        raise ValueError(f'Priority configuration input mismatch: {path}')
    configs = []
    for row in data['configs']:
        probes, refine = row['nprobe'], row['probe_topk']
        if type(probes) is not int or type(refine) is not int or probes < 1 or refine < 100:
            raise ValueError(f'Invalid priority configuration: {row}')
        if row['recall_at_100'] >= .90 and (probes, refine) not in configs:
            configs.append((probes, refine))
    return configs


def search_grid():
    return list(itertools.product(
        [1, 2, 4, 8, 12, 16, 32, 64, 128],
        [100, 200, 500, 1000, 1200, 1400, 1600, 1800, 2000]))


def pilot_priorities(configs):
    return list(dict.fromkeys(configs[:1] + configs[-1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--bits', type=int, default=4)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--screening', action='store_true', help='Recall discovery only; omit confirmation repeats')
    p.add_argument('--priority-configs', type=Path)
    p.add_argument('--paper-default', action='store_true',
                   help='Run the paper-default 2-bit (12,1000) point first, with three repeats')
    a = p.parse_args()
    if a.paper_default and a.bits != 2:
        p.error('--paper-default requires --bits 2')
    a.output.mkdir(parents=True, exist_ok=True)
    idx, = (a.data / 'Index' / a.dataset).glob(f'IGPGPUPP-*-n_bit_{a.bits}')
    manifest = json.loads((idx / 'manifest.json').read_text())
    gt = {}
    for line in Path(manifest['source_gt']).open():
        qid, docid, rank, *_ = line.split()
        if int(rank) <= 100:
            gt.setdefault(int(qid), []).append(int(docid))
    assert list(gt) == manifest['query_ids']
    assert hashlib.sha256((Path(manifest['source_gt']).parent / 'query.bin').read_bytes()).hexdigest() == manifest['query_sha256']
    # Upstream output names do not include the GPU or run; isolate every process.
    view = a.output / 'input_view'
    view.mkdir(exist_ok=True)
    for name in ['Index', 'Embedding', 'RawData']:
        link = view / name
        if not link.exists():
            link.symlink_to((a.data / name).resolve(), target_is_directory=True)
    (view / 'Result/answer').mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIGP_DATA_ROOT=str(view.resolve()))
    binary_sha256 = hashlib.sha256(a.binary.read_bytes()).hexdigest()
    rows = []

    def run(probes, refine, repeat):
        folder = a.output / f'p{probes}_r{refine}_run{repeat}'
        folder.mkdir(exist_ok=True)
        result = folder / 'result.json'
        if result.exists():
            row = json.loads(result.read_text())
            if row.get('query_sha256') != manifest['query_sha256']:
                raise RuntimeError(f'Query checksum mismatch in checkpoint: {result}')
            if row.get('source_commit') != '1899e9316e71b023518f9090fc64fe66dbd0c202':
                raise RuntimeError(f'Unverified source in checkpoint: {result}')
            rows.append(row)
            return row
        cmd = [str(a.binary), '-username', os.environ['USER'], '-dataset', a.dataset,
               '-method-name', 'IGPGPUPP', '-num-centroid', str(manifest['n_centroid']),
               '-num-bit', str(a.bits), '-topk', '100', '-nprobe', str(probes),
               '-probe_topk', str(refine), '-is_profile', 'false']
        (folder / 'command.json').write_text(json.dumps(cmd))
        with (folder / 'stdout.log').open('w') as log:
            try:
                done = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            except subprocess.TimeoutExpired:
                (folder / 'failure.json').write_text(json.dumps({'reason': 'timeout', 'timeout_seconds': 1800}))
                print(f'CONFIG_TIMEOUT: {folder}', flush=True)
                return
        if done.returncode:
            (folder / 'failure.json').write_text(json.dumps({'returncode': done.returncode}))
            print(f'CONFIG_FAILED: {done.returncode} {folder}', flush=True)
            return
        prefix = f'{a.dataset}-IGPGPUPP'
        suffix = f'top100-n_centroid_{manifest["n_centroid"]}-n_bit_{a.bits}-nprobe_{probes}-probe_topk_{refine}'
        answers = view / 'Result/answer'
        pred = {}
        answer = answers / f'{prefix}-{suffix}.tsv'
        for line in answer.open():
            qid, docid, *_ = line.split()
            pred.setdefault(int(qid), []).append(int(docid))
        assert set(pred) == set(gt) and all(len(v) == 100 for v in pred.values())
        recall = lambda k: float(np.mean([len(set(pred[q][:k]) & set(gt[q][:k])) / k for q in gt]))
        perf = answers / f'{prefix}-performance-{suffix}.tsv'
        lat = np.loadtxt(perf, skiprows=1, ndmin=1) * 1000
        assert len(lat) == len(gt) and np.all(np.isfinite(lat)) and np.all(lat > 0)
        totals = dict(line.strip().split('\t') for line in perf.with_suffix('.txt').open())
        row = dict(dataset=a.dataset, gpu=a.gpu, bits=a.bits, topk=100, nprobe=probes,
                   probe_topk=refine, concurrent_queries=3, warmup=10, repeat=repeat,
                   nq=len(gt), recall_at_10=recall(10), recall_at_100=recall(100),
                   qps=len(gt) / float(totals['batch_query_time(s)']), avg_latency_ms=float(lat.mean()),
                   p50_ms=float(np.percentile(lat, 50)), p95_ms=float(np.percentile(lat, 95)),
                   p99_ms=float(np.percentile(lat, 99)), query_sha256=manifest['query_sha256'],
                   binary_sha256=binary_sha256, slurm_job_id=os.environ.get('SLURM_JOB_ID', ''),
                   source_commit='1899e9316e71b023518f9090fc64fe66dbd0c202')
        shutil.copy2(answer, folder / 'answers.tsv')
        shutil.copy2(perf, folder / 'latencies.tsv')
        shutil.copy2(perf.with_suffix('.txt'), folder / 'totals.txt')
        temporary = result.with_suffix('.tmp')
        temporary.write_text(json.dumps(row, indent=2))
        temporary.replace(result)
        rows.append(row)
        print(json.dumps(row), flush=True)
        return row

    # Paper grid plus high-probe/small-shortlist points near our recall target.
    grid = [(12, 1000)] if a.smoke else search_grid()
    if a.pilot:
        grid = [(32, 4000), (64, 8000), (128, 16000)]
    priorities = []
    if not a.screening and not a.smoke:
        if a.paper_default:
            priorities = [(12, 1000)]
            print('PAPER_DEFAULT_2BIT: nprobe=12 probe_topk=1000', flush=True)
        else:
            path = a.priority_configs or a.data.parent / 'benchmark/priority_configs' / f'{a.dataset}.json'
            priorities = priority_configs(path, a.dataset, manifest['query_sha256'])
        if a.pilot:
            priorities = pilot_priorities(priorities)
        if priorities and not a.paper_default:
            print(f'JUPITER_RECALL_PRIORITY: {path} {priorities}', flush=True)
        for probes, refine in priorities:
            measured = run(probes, refine, 1)
            if measured is not None and (measured['recall_at_100'] >= .90 or a.paper_default):
                for repeat in [2, 3]:
                    run(probes, refine, repeat)
    for probes, refine in grid:
        if (probes, refine) in priorities:
            continue
        run(probes, refine, 1)
    if not a.smoke:
        # Extend beyond the paper grid for our >=0.90 targets.
        if not a.pilot:
            for probes, refine in itertools.product([32, 64, 128], [4000, 8000, 16000]):
                if (probes, refine) in priorities:
                    continue
                run(probes, refine, 1)
        # Repeat every qualifying point: timing noise can change frontier membership.
        chosen = {(r['nprobe'], r['probe_topk']) for r in rows if r['recall_at_100'] >= .90}
        if not a.screening:
            for probes, refine in sorted(chosen):
                if (probes, refine) in priorities:
                    continue
                for repeat in [2, 3]:
                    run(probes, refine, repeat)
        if a.paper_default and not a.screening:
            attempted = set()
            while True:
                pending = set(paper_confirmation_requests(rows)) - attempted
                if not pending:
                    break
                for probes, refine, repeat in sorted(pending):
                    attempted.add((probes, refine, repeat))
                    run(probes, refine, repeat)
            if paper_confirmation_requests(rows):
                raise RuntimeError('Paper-default frontier confirmation failed; inspect failed runs')
    grouped = {}
    for row in rows:
        grouped.setdefault((row['nprobe'], row['probe_topk']), []).append(row)
    summaries = []
    for runs in grouped.values():
        row = dict(runs[0])
        row.pop('repeat')
        row['runs'] = len(runs)
        for key in ['recall_at_10', 'recall_at_100', 'qps', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms']:
            row[key] = float(np.mean([r[key] for r in runs]))
        summaries.append(row)
    verified = [r for r in summaries if r['runs'] == 3 and r['recall_at_100'] >= .90]
    outputs = [('results.csv', rows), ('summary.csv', summaries),
               ('screening_pareto.csv', frontier(summaries)), ('pareto.csv', frontier(verified))]
    if a.paper_default:
        outputs.append(('paper_pareto.csv', frontier([r for r in summaries if r['runs'] == 3])))
    for name, data in outputs:
        if data:
            with (a.output / name).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    if not rows:
        raise RuntimeError('No configurations completed successfully; inspect failure logs')
    if max(r['recall_at_100'] for r in rows) < .90:
        print('TARGET_NOT_REACHED: extend sweep or investigate quantization ceiling', flush=True)


if __name__ == '__main__':
    main()
