"""Export recall-qualified screening checkpoints, without selecting by GPU speed."""
import argparse
import json
from pathlib import Path


def export(root, output, dataset):
    rows = []
    for path in sorted(root.glob('*/p*_r*_run1/result.json')):
        row = json.loads(path.read_text())
        if row['dataset'] != dataset or row['gpu'] != 'rtx3090' or row['bits'] != 4:
            raise ValueError(f'Unexpected screening input: {path}')
        rows.append(row)
    if not rows:
        raise ValueError('No completed screening configurations')
    hashes = {row['query_sha256'] for row in rows}
    if len(hashes) != 1:
        raise ValueError('Mixed query cohorts')
    qualified = {}
    for row in rows:
        if row['recall_at_100'] >= .90:
            key = row['nprobe'], row['probe_topk']
            qualified[key] = max(row['recall_at_100'], qualified.get(key, 0))
    # Increasing recall samples the target range first; retain every qualifying config.
    configs = [dict(nprobe=p, probe_topk=r, recall_at_100=recall)
               for (p, r), recall in sorted(qualified.items(), key=lambda item: (item[1], item[0]))]
    data = dict(dataset=dataset, query_sha256=hashes.pop(), screening_gpu='rtx3090',
                screening_only=True, configs=configs)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(output)
    print(f'{dataset}: {len(configs)} recall-qualified priority configurations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    export(args.root, args.output, args.dataset)
