"""Reuse the existing IGP scalar index and freeze the baseline query order."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--bits', type=int, default=4)
    p.add_argument('--prepare-binary', required=True)
    a = p.parse_args()
    source = a.dataset_root / a.dataset
    matches = list((source / 'igp/Index' / a.dataset).glob(f'IGP-*-n_bit_{a.bits}'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one source index, found {matches}')
    old = matches[0]
    centroids = np.load(old / 'centroid_l.npy', mmap_mode='r')
    nc = len(centroids)
    idx = a.output / 'Index' / a.dataset / f'IGPGPUPP-n_centroid_{nc}-n_bit_{a.bits}'
    idx.mkdir(parents=True, exist_ok=True)
    for name in ['centroid_l.npy', 'vq_code_l.npy', 'weight_l.npy', 'residual_code_l.npy']:
        target = idx / name
        if not target.exists():
            target.symlink_to((old / name).resolve())
    with (source / 'raw/doclens.bin').open('rb') as f:
        nd = int(np.fromfile(f, np.uint32, 1)[0])
        lengths = np.fromfile(f, np.uint32)
    assert len(lengths) == nd
    codes = np.load(idx / 'vq_code_l.npy', mmap_mode='r')
    assert int(lengths.sum(dtype=np.uint64)) == len(codes)
    residual = np.load(idx / 'residual_code_l.npy', mmap_mode='r')
    assert residual.size == len(codes) * centroids.shape[1] * a.bits // 8
    np.save(idx / 'item_n_vec_l.npy', lengths)
    qpath = source / 'raw/query.bin'
    with qpath.open('rb') as f:
        shape = tuple(map(int, np.fromfile(f, np.uint32, 3)))
        queries = np.fromfile(f, np.float32).reshape(shape)
    emb = a.output / 'Embedding' / a.dataset
    emb.mkdir(parents=True, exist_ok=True)
    np.save(emb / 'query_embedding.npy', queries)
    ids = []
    for line in (source / 'raw/gt.tsv').open():
        qid = int(line.split('\t')[0])
        if not ids or ids[-1] != qid:
            ids.append(qid)
    assert len(ids) == shape[0] and len(set(ids)) == len(ids)
    raw = a.output / 'RawData' / a.dataset / 'document'
    raw.mkdir(parents=True, exist_ok=True)
    (raw / 'queries.dev.tsv').write_text(''.join(f'{qid}\tfrozen-query\n' for qid in ids))
    manifest = dict(dataset=a.dataset, bits=a.bits, n_centroid=nc, query_shape=shape,
                    query_ids=ids, query_sha256=hashlib.sha256(qpath.read_bytes()).hexdigest(),
                    source_index=str(old.resolve()), source_gt=str(source / 'raw/gt.tsv'))
    (idx / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if not (idx / 'ivf.index').exists():
        subprocess.run([a.prepare_binary, str(idx / 'vq_code_l.npy'),
                        str(idx / 'item_n_vec_l.npy'), str(nc), str(idx / 'ivf.index')], check=True)


if __name__ == '__main__':
    main()
