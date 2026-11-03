"""Synthetic collector tests; fixtures stay in a temporary directory."""
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from collect_results import pareto, verify_chimera_affinity
from plot_precision_comparison import comparison_points


class CollectionTests(unittest.TestCase):
    def test_team_affinity_proof(self):
        hardware = 'slot_cpu_groups=0,1;2,3;4,5 stage3_threads_per_slot=2'
        log = '\n'.join(f'[AFFINITY_TEAM] slot={slot} thread={thread} cpus={2*slot},{2*slot+1}'
                        for slot in range(3) for thread in range(2))
        verify_chimera_affinity(log, hardware)
        with self.assertRaises(AssertionError):
            verify_chimera_affinity('[AFFINITY] slot=0 cpus=0,1', hardware)
        with self.assertRaises(AssertionError):
            verify_chimera_affinity(log.replace('cpus=2,3', 'cpus=0,1,2,3,4,5'), hardware)

    def test_dominance(self):
        rows = [{'recall_at_100': .90, 'qps': 100},
                {'recall_at_100': .95, 'qps': 80},
                {'recall_at_100': .91, 'qps': 70}]
        self.assertEqual(pareto(rows), rows[:2])

    def test_precision_plot_inputs(self):
        matched = dict(system='Chimera', bits='4', gpu='a100', dataset='lotte',
                       config='config', query_sha256='same-query', three_runs_verified='True',
                       concurrent_queries='3', mode='pilot', recall_at_100='.9', qps='100')
        full = dict(matched, mode='full', qps='50')
        paper = dict(matched, system='GIGP+', bits='2', recall_at_100='.8')
        rows = comparison_points([matched, full], [paper])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['qps'], 50)
        with self.assertRaises(ValueError):
            comparison_points([matched], [dict(paper, query_sha256='other-query')])
        with self.assertRaises(ValueError):
            comparison_points([matched], [dict(paper, bits='4')])

    def test_three_runs_and_build_gate(self):
        with tempfile.TemporaryDirectory(prefix='gigpp_collector_test_') as tmp:
            root = Path(tmp)
            output = root / 'export'
            mode = root / 'results/a100_lotte_4bit/pilot'
            mode.mkdir(parents=True)
            for i in [1, 2, 3]:
                folder = mode / f'p32_r4000_run{i}'
                folder.mkdir()
                row = dict(concurrent_queries=3, bits=4, nprobe=32, probe_topk=4000,
                           query_sha256='synthetic-query-hash', source_commit='synthetic-source',
                           repeat=i, slurm_job_id='synthetic_job', recall_at_100=.92,
                           qps=100 * i, avg_latency_ms=20, p50_ms=19, p95_ms=25, p99_ms=30)
                (folder / 'result.json').write_text(json.dumps(row))
                (folder / 'stdout.log').write_text('synthetic affinity record\n')
            (mode / 'hardware_synthetic_job.txt').write_text('synthetic A100 8352Y\n')
            metadata = root / 'results/build_metadata/jobsynthetic_job'
            metadata.mkdir(parents=True)
            cache = metadata / 'CMakeCache.txt'
            cache.write_text('GIGP_CPU_ARCH:STRING=native\nCMAKE_CUDA_ARCHITECTURES:STRING=native\n')
            cmd = [sys.executable, str(Path(__file__).with_name('collect_results.py')),
                   '--root', str(root), '--output', str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            with (output / 'verified_frontier.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]['qps']), 200)
            self.assertEqual(int(rows[0]['runs']), 3)
            full = mode.with_name('full')
            shutil.copytree(mode, full)
            for result in full.rglob('result.json'):
                row = json.loads(result.read_text())
                row['qps'] = 50
                result.write_text(json.dumps(row))
            subprocess.run(cmd, check=True, capture_output=True)
            with (output / 'verified_frontier.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['mode'], 'full')
            self.assertEqual(float(rows[0]['qps']), 50)
            cache.write_text('GIGP_CPU_ARCH:STRING=native\nCMAKE_CUDA_ARCHITECTURES:STRING=80\n')
            subprocess.run(cmd, check=True, capture_output=True)
            with (output / 'verified_frontier.csv').open() as f:
                self.assertEqual(list(csv.DictReader(f)), [])
            with (output / 'rejected.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 6)

    def test_paper_precision_and_below_target_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mode = root / 'paper2bit_results/a100_lotte_2bit/pilot'
            mode.mkdir(parents=True)
            hardware = mode / 'hardware_123.txt'
            hardware.write_text('A100 8352Y allowed_cpus=0,1,2 binary-sha\n')
            (mode / 'build_123.log').write_text(
                'four GPU/CPU reference checks passed\n'
                '86bc9805062928b5b51f3b7a7e4d381cd68bf2a919f07abf4bd62b339040fc58\n')
            metadata = root / 'results/build_metadata/job123'
            metadata.mkdir(parents=True)
            (metadata / 'CMakeCache.txt').write_text(
                'GIGP_CPU_ARCH:STRING=native\nCMAKE_CUDA_ARCHITECTURES:STRING=native\n')
            for name, bits in [('data_paper2bit', 2), ('data', 4)]:
                index = root / name / f'Index/lotte/IGPGPUPP-n_centroid_128-n_bit_{bits}'
                index.mkdir(parents=True)
                (index / 'manifest.json').write_text(json.dumps(dict(
                    bits=bits, n_centroid=128, query_sha256='query-hash',
                    source_gt='same-ground-truth', centroids_reused=True, residual_sha256='index-sha')))
            for repeat in [1, 2, 3]:
                folder = mode / f'p12_r1000_run{repeat}'
                folder.mkdir()
                row = dict(dataset='lotte', gpu='a100', concurrent_queries=3, bits=2,
                    nprobe=12, probe_topk=1000, query_sha256='query-hash', binary_sha256='binary-sha',
                    source_commit='1899e9316e71b023518f9090fc64fe66dbd0c202',
                    repeat=repeat, slurm_job_id='123', recall_at_100=.81,
                    qps=100 * repeat, avg_latency_ms=20, p50_ms=19, p95_ms=25, p99_ms=30)
                (folder / 'result.json').write_text(json.dumps(row))
                (folder / 'command.json').write_text(json.dumps([
                    'binary', '-num-bit', '2', '-nprobe', '12', '-probe_topk', '1000', '-num-centroid', '128']))
                (folder / 'stdout.log').write_text('\n'.join(
                    f'level 1 thread 0x{i} affinity {i}' for i in range(3)))
            output = root / 'export'
            def read_export(name):
                with (output / name).open() as f:
                    return list(csv.DictReader(f))
            cmd = [sys.executable, str(Path(__file__).with_name('collect_results.py')),
                   '--root', str(root), '--output', str(output)]
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(read_export('runs.csv'), [])
            cmd += ['--gigp-bits', '2', '--min-recall', '0']
            subprocess.run(cmd, check=True, capture_output=True)
            rows = read_export('verified_frontier.csv')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['bits'], '2')
            self.assertEqual(float(rows[0]['recall_at_100']), .81)
            self.assertEqual(float(rows[0]['qps']), 200)
            guarded = cmd + ['--require-paper-summary']
            subprocess.run(guarded, check=True, capture_output=True)
            self.assertEqual(read_export('verified_frontier.csv'), [])
            for filename in ['summary.csv', 'paper_pareto.csv']:
                (mode / filename).write_text('synthetic final-artifact marker\n')
            subprocess.run(guarded, check=True, capture_output=True)
            self.assertEqual(len(read_export('verified_frontier.csv')), 1)
            hardware.write_text('A100 8352Y allowed_cpus=0,1 binary-sha\n')
            subprocess.run(cmd, check=True, capture_output=True)
            self.assertEqual(len(read_export('rejected.csv')), 3)
            self.assertEqual(read_export('verified_frontier.csv'), [])


if __name__ == '__main__':
    unittest.main()
