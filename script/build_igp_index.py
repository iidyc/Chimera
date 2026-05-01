#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    igp_root = repo_root / "IGP"
    sys.path.insert(0, str(igp_root / "script" / "evaluation"))
    sys.path.insert(1, str(igp_root))

    from script.data import dataset_io, util
    from script.evaluation import eval_igp
    import IGP

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["lotte", "msmarco", "hotpot"])
    parser.add_argument("--local-data-root", required=True)
    parser.add_argument(
        "--n-bit",
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help="Residual scalar quantization bits. Default: 4.",
    )
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("IGP_DATA_MODE", "binary")
    os.environ.setdefault("GPU_MVR_DATASET_ROOT", str(repo_root / "dataset"))
    os.environ["IGP_LOCAL_DATA_ROOT"] = args.local_data_root

    username = "juelin"
    dataset = args.dataset
    module_name = "IGP"

    if args.compile:
        util.compile_file(username=username, module_name=module_name, is_debug=False, move_path="evaluation")

    item_n_vec_l = dataset_io.load_doclens(username=username, dataset=dataset).astype(np.uint32)
    n_item = int(item_n_vec_l.shape[0])
    n_vecs = int(np.sum(item_n_vec_l, dtype=np.uint64))
    vec_dim = dataset_io.embedding_dim(username=username, dataset=dataset)
    n_centroid = util.paper_n_centroid(n_vecs)
    n_bit = args.n_bit

    constructor_insert_item = {
        "item_n_vec_l": item_n_vec_l.tolist(),
        "n_item": n_item,
        "vec_dim": vec_dim,
        "n_centroid": n_centroid,
        "n_bit": n_bit,
    }
    build_index_suffix = f"n_centroid_{n_centroid}-n_bit_{n_bit}"

    print("module_file", IGP.__file__, flush=True)
    print("dataset", dataset, "docs", n_item, "tokens", n_vecs, "vec_dim", vec_dim, flush=True)
    print("n_centroid", n_centroid, "n_bit", n_bit, flush=True)
    print("local_data_root", dataset_io.local_data_root(), flush=True)
    print("index_dir", dataset_io.index_dir(dataset), flush=True)

    eval_igp.approximate_solution_build_index(
        username=username,
        dataset=dataset,
        constructor_insert_item=constructor_insert_item,
        module=IGP,
        module_name=module_name,
        build_index_config={"n_centroid_f": util.paper_n_centroid, "n_bit": n_bit},
        build_index_suffix=build_index_suffix,
        save_index=True,
    )

    print("done", dataset_io.index_dir(dataset) / f"{module_name}-{build_index_suffix}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
