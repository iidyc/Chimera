#!/usr/bin/env python3
"""Print Table 4 operating points from the saved A100 frontier."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontier', type=Path,
                        default=Path(__file__).with_name('a100_frontier.csv'))
    args = parser.parse_args()
    with args.frontier.open(newline='') as source:
        rows = [
            row for row in csv.DictReader(source)
            if (row['gpu'], row['bits'], row['concurrent_queries'],
                row['mode'], row['three_runs_verified']) == ('a100', '4', '3', 'full', 'True')
        ]

    print('Dataset  Target System    Recall      QPS   Mean    p50    p95    p99  Speedup')
    for dataset in ('lotte', 'hotpot', 'msmarco'):
        for target in (0.90, 0.95):
            selected = {}
            for system in ('Chimera', 'GIGP+'):
                selected[system] = max(
                    (row for row in rows
                     if row['dataset'] == dataset and row['system'] == system
                     and float(row['recall_at_100']) >= target),
                    key=lambda row: float(row['qps']),
                    default=None,
                )
            chimera, gigp = selected.values()
            if chimera and gigp and chimera['query_sha256'] != gigp['query_sha256']:
                raise ValueError(f'Query cohorts differ for {dataset} at {target}')
            for system, row in selected.items():
                label = f'{dataset:8} {target:.2f}   {system:7}'
                if row is None:
                    print(f'{label}  target not reached')
                    continue
                metrics = ' '.join(
                    f'{float(row[key]):6.1f}'
                    for key in ('qps', 'avg_latency_ms', 'p50_ms', 'p95_ms', 'p99_ms')
                )
                speedup = (f'  {float(chimera["qps"]) / float(gigp["qps"]):.2f}x'
                           if system == 'Chimera' and gigp else '')
                print(f'{label}  {float(row["recall_at_100"]):.6f} {metrics}{speedup}')


if __name__ == '__main__':
    main()
