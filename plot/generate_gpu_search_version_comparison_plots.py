#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


DEFAULT_VERSIONS = ["v2", "v3", "v4", "v5"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate small scatter plots comparing multiple GPU-MVR gpu_search "
            "versions using each version's Pareto frontier independently."
        )
    )
    parser.add_argument(
        "--profiling-root",
        type=Path,
        default=Path("profiling"),
        help="Profiling root containing per-dataset GPU-MVR benchmark outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plot"),
        help="Directory to write plots and helper cache directories.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Dataset to plot. Repeatable. Default: lotte, hotpot, msmarco.",
    )
    parser.add_argument(
        "--version",
        action="append",
        dest="versions",
        help=(
            "GPU-MVR gpu_search version to compare, e.g. v2 or gpu_search_v2. "
            "Repeatable. Default: v2, v3, v4, v5."
        ),
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.8,
        help="Filter out plotted points with recall below this threshold. Default: 0.8.",
    )
    return parser.parse_args()


def normalize_version(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("gpu_search_"):
        stripped = stripped.removeprefix("gpu_search_")
    if not stripped.startswith("v") or not stripped[1:].isdigit():
        raise ValueError(f"invalid version {value!r}; expected vN or gpu_search_vN")
    return stripped


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def filter_rows_by_min_recall(rows: list[dict[str, str]], min_recall: float) -> list[dict[str, str]]:
    return [row for row in rows if float(row["recall"]) >= min_recall]


def find_latest_frontier(dataset_dir: Path, version: str) -> Path:
    version_dir = dataset_dir / f"gpu_search_{version}"
    candidates = sorted(version_dir.glob("pareto_frontier_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no pareto_frontier_*.csv found under {version_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, path.name))
    return candidates[-1]


def load_version_rows(
    dataset_dir: Path,
    versions: list[str],
    min_recall: float,
) -> list[tuple[str, list[dict[str, str]], Path]]:
    series: list[tuple[str, list[dict[str, str]], Path]] = []
    for version in versions:
        frontier = find_latest_frontier(dataset_dir, version)
        rows = sorted(read_rows(frontier), key=lambda row: float(row["recall"]))
        if not rows:
            raise RuntimeError(f"no rows found in {frontier}")
        rows = filter_rows_by_min_recall(rows, min_recall)
        if not rows:
            raise RuntimeError(
                f"no rows remain for gpu_search_{version} after applying min recall {min_recall:.3f}"
            )
        series.append((version, rows, frontier))
    return series


def setup_matplotlib_cache(output_dir: Path) -> None:
    mpl_dir = output_dir / ".mplconfig"
    cache_dir = output_dir / ".cache"
    fontconfig_dir = cache_dir / "fontconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    fontconfig_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())
    os.environ["XDG_CACHE_HOME"] = str(cache_dir.resolve())
    os.environ["FONTCONFIG_PATH"] = "/etc/fonts"
    os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
    os.environ["FONTCONFIG_CACHE"] = str(fontconfig_dir.resolve())


def make_output_stem(versions: list[str], dataset: str) -> str:
    joined_versions = "_".join(versions)
    return f"gpu_search_{joined_versions}_{dataset}"


def render_dataset(
    profiling_root: Path,
    output_dir: Path,
    dataset: str,
    versions: list[str],
    min_recall: float,
) -> tuple[Path, Path]:
    setup_matplotlib_cache(output_dir)

    import matplotlib.pyplot as plt

    dataset_dir = profiling_root / dataset
    series = load_version_rows(dataset_dir, versions, min_recall)

    fig, ax = plt.subplots(figsize=(3, 2))

    all_rows: list[dict[str, str]] = []
    for idx, (version, rows, _) in enumerate(series):
        marker = MARKERS[idx % len(MARKERS)]
        ax.plot(
            [float(row["recall"]) for row in rows],
            [float(row["qps"]) for row in rows],
            marker=marker,
            linewidth=1.2,
            markersize=3.5,
            label=version,
        )
        all_rows.extend(rows)

    recall_values = [float(row["recall"]) for row in all_rows]
    qps_values = [float(row["qps"]) for row in all_rows]

    ax.set_xlim(max(0.0, min_recall), min(1.0, max(recall_values) + 0.005))
    ax.set_ylim(max(0.0, min(qps_values) * 0.95), max(qps_values) * 1.05)
    ax.set_xlabel("Recall@100")
    ax.set_ylabel("QPS")
    ax.set_title(dataset.upper())
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc="best", frameon=False)

    fig.tight_layout(pad=0.3)

    stem = make_output_stem(versions, dataset)
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    profiling_root = args.profiling_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or ["lotte", "hotpot", "msmarco"]
    versions = [normalize_version(version) for version in (args.versions or DEFAULT_VERSIONS)]

    deduped_versions: list[str] = []
    for version in versions:
        if version not in deduped_versions:
            deduped_versions.append(version)

    for dataset in datasets:
        png_path, pdf_path = render_dataset(
            profiling_root,
            output_dir,
            dataset,
            deduped_versions,
            args.min_recall,
        )
        print(
            f"dataset={dataset} versions={','.join(deduped_versions)} "
            f"png={png_path.resolve()} pdf={pdf_path.resolve()}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
