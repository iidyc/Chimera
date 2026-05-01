#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path


DEFAULT_COLBERT_PREFIX = Path("/data/juelin/conda/envs/colbert")


def parse_bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--colbert-conda-prefix", type=Path, default=DEFAULT_COLBERT_PREFIX)
    parser.add_argument("--no-reexec", action="store_true")
    return parser.parse_known_args()[0]


def configure_colbert_env(repo_root: Path, conda_prefix: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = conda_prefix / "bin"
    colbert_src = repo_root / "ColBERT"

    env["PYTHONPATH"] = f"{colbert_src}:{env.get('PYTHONPATH', '')}"
    env["CUDA_HOME"] = str(conda_prefix)
    env["TORCH_EXTENSIONS_DIR"] = str(repo_root / ".cache" / "torch_extensions" / "colbert")
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = ":".join(
        [
            str(conda_prefix / "lib"),
            str(conda_prefix / "lib64"),
            env.get("LD_LIBRARY_PATH", ""),
        ]
    )

    cc = bin_dir / "x86_64-conda-linux-gnu-cc"
    cxx = bin_dir / "x86_64-conda-linux-gnu-c++"
    if cc.exists():
        env["CC"] = str(cc)
    if cxx.exists():
        env["CXX"] = str(cxx)
        env["CUDAHOSTCXX"] = str(cxx)

    return env


def maybe_reexec_with_colbert_python(repo_root: Path) -> None:
    args = parse_bootstrap_args()
    conda_prefix = args.colbert_conda_prefix.resolve()
    colbert_python = conda_prefix / "bin" / "python"

    if args.no_reexec:
        return
    if Path(sys.executable).resolve() == colbert_python.resolve():
        return
    if not colbert_python.exists():
        return

    env = configure_colbert_env(repo_root, conda_prefix)
    os.execve(str(colbert_python), [str(colbert_python), str(Path(__file__).resolve()), *sys.argv[1:]], env)


REPO_ROOT = Path(__file__).resolve().parents[2]
maybe_reexec_with_colbert_python(REPO_ROOT)

BOOTSTRAP_ARGS = parse_bootstrap_args()
ACTIVE_COLBERT_PREFIX = BOOTSTRAP_ARGS.colbert_conda_prefix.resolve()
os.environ.update(configure_colbert_env(REPO_ROOT, ACTIVE_COLBERT_PREFIX))
sys.path.insert(0, str(REPO_ROOT / "ColBERT"))

import numpy as np
import torch

from colbert import EmbeddingIndexer
from colbert.infra import ColBERTConfig, Run, RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PLAID/ColBERT index from precomputed embedding and doclens .bin files."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPO_ROOT / "dataset" / "hotpot" / "raw" / "data.bin",
        help="Embedding .bin file: int32 rows, int32 dim, then float32 embeddings.",
    )
    parser.add_argument(
        "--doclens-path",
        type=Path,
        default=None,
        help="Doclens .bin file: int32 num_docs, then int32 doc lengths. Defaults to doclens.bin next to data-path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "dataset" / "hotpot" / "colbert" / "indexes",
        help="ColBERT index root. The final index path is output-dir/index-name.",
    )
    parser.add_argument("--index-name", default="4bit")
    parser.add_argument("--experiment-name", default="colbert")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--nbits", type=int, default=4)
    parser.add_argument("--colbert-conda-prefix", type=Path, default=DEFAULT_COLBERT_PREFIX)
    parser.add_argument("--no-reexec", action="store_true")
    parser.add_argument(
        "--overwrite",
        choices=["true", "false", "reuse", "resume", "force_silent_overwrite"],
        default="force_silent_overwrite",
    )
    parser.add_argument("--verbose", type=int, default=2)
    return parser.parse_args()


def read_embeddings(path: Path) -> torch.Tensor:
    with path.open("rb") as handle:
        shape = np.fromfile(handle, dtype=np.int32, count=2)
    if len(shape) != 2:
        raise ValueError(f"{path} does not contain a two-int32 shape header")

    rows, dim = map(int, shape)
    header_bytes = 2 * np.dtype(np.int32).itemsize
    data = np.memmap(path, dtype=np.float32, mode="r", offset=header_bytes, shape=(rows, dim))
    embeddings = torch.from_numpy(data)
    print(f"Loaded embeddings: rows={rows} dim={dim} path={path}")
    return embeddings


def read_doclens(path: Path) -> list[int]:
    with path.open("rb") as handle:
        count = np.fromfile(handle, dtype=np.int32, count=1)
        if len(count) != 1:
            raise ValueError(f"{path} does not contain a num_docs header")
        num_docs = int(count[0])
        doclens = np.fromfile(handle, dtype=np.int32, count=num_docs).astype(np.int64).tolist()

    if len(doclens) != num_docs:
        raise ValueError(f"{path} expected {num_docs} doclens, found {len(doclens)}")
    print(f"Loaded doclens: docs={len(doclens)} total_tokens={sum(doclens)} path={path}")
    return doclens


def overwrite_value(value: str):
    if value == "true":
        return True
    if value == "false":
        return False
    return value


def main() -> int:
    args = parse_args()

    data_path = args.data_path.resolve()
    doclens_path = (args.doclens_path or data_path.with_name("doclens.bin")).resolve()
    output_dir = args.output_dir.resolve()
    index_path = output_dir / args.index_name

    if not data_path.exists():
        raise FileNotFoundError(f"missing embeddings file: {data_path}")
    if not doclens_path.exists():
        raise FileNotFoundError(f"missing doclens file: {doclens_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"python={sys.executable}")
    print(f"CUDA_HOME={os.environ.get('CUDA_HOME')}")
    print(f"PYTHONPATH={os.environ.get('PYTHONPATH')}")
    print(f"TORCH_EXTENSIONS_DIR={os.environ.get('TORCH_EXTENSIONS_DIR')}")
    print(f"CC={os.environ.get('CC', '')}")
    print(f"CXX={os.environ.get('CXX', '')}")
    print(f"index_path={index_path}")

    embeddings = read_embeddings(data_path)
    doclens = read_doclens(doclens_path)

    if embeddings.shape[1] != args.dim:
        raise ValueError(f"--dim={args.dim} does not match embeddings dim={embeddings.shape[1]}")
    if sum(doclens) != embeddings.shape[0]:
        raise ValueError(f"sum(doclens)={sum(doclens)} does not match embedding rows={embeddings.shape[0]}")

    run_root = output_dir.parent.parent if output_dir.name == "indexes" else output_dir.parent
    with Run().context(RunConfig(experiment=args.experiment_name, root=str(run_root))):
        config = ColBERTConfig(
            dim=args.dim,
            nbits=args.nbits,
            root=str(run_root),
            experiment=args.experiment_name,
            index_root=str(output_dir),
        )
        built_path = EmbeddingIndexer(config=config, verbose=args.verbose).index(
            args.index_name,
            embeddings=embeddings,
            doclens=doclens,
            overwrite=overwrite_value(args.overwrite),
        )

    print(f"[done] index_path={built_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
