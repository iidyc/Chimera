"""Check the shared query cohort, ground truth, and prepared GIGP+ inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    reports = []
    for dataset in ['lotte', 'hotpot', 'msmarco']:
        index, = (a.root / 'data/Index' / dataset).glob('IGPGPUPP-*-n_bit_4')
        manifest = json.loads((index / 'manifest.json').read_text())
        raw = Path(manifest['source_gt']).parent
        with (raw / 'query.bin').open('rb') as f:
            shape = tuple(map(int, np.fromfile(f, np.uint32, 3)))
            queries = np.fromfile(f, np.float32).reshape(shape)
        prepared = np.load(a.root / 'data/Embedding' / dataset / 'query_embedding.npy')
        assert np.array_equal(queries, prepared)
        assert shape[1:] == (32, 128)
        assert np.all(np.isfinite(prepared))
        assert digest(raw / 'query.bin') == manifest['query_sha256']
        with (raw / 'doclens.bin').open('rb') as f:
            nd = int(np.fromfile(f, np.uint32, 1)[0])
            lengths = np.fromfile(f, np.uint32)
        assert len(lengths) == nd and np.all(lengths > 0)
        assert np.array_equal(lengths, np.load(index / 'item_n_vec_l.npy'))
        truth = {}
        for line in (raw / 'gt.tsv').open():
            qid, docid, rank, *_ = line.split()
            qid, docid, rank = int(qid), int(docid), int(rank)
            assert 0 <= docid < nd
            if rank <= 100:
                truth.setdefault(qid, []).append((rank, docid))
        assert list(truth) == manifest['query_ids'] == list(range(shape[0]))
        for items in truth.values():
            assert [rank for rank, _ in items] == list(range(1, 101))
            assert len({doc for _, doc in items}) == 100
        reports.append(dict(dataset=dataset, queries=shape[0], query_vectors=shape[1],
                            dimensions=shape[2], documents=nd, tokens=int(lengths.sum(dtype=np.uint64)),
                            query_sha256=manifest['query_sha256'], gt_sha256=digest(raw / 'gt.tsv'),
                            query_order_matches=True, top100_ground_truth_valid=True,
                            prepared_queries_identical=True, document_lengths_identical=True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
