#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path


def latest_required(repo_root: Path, relative_dir: str, pattern: str) -> Path:
    directory = repo_root / relative_dir
    paths = sorted(
        directory.glob(pattern),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not paths:
        raise FileNotFoundError(f"no files matching {directory / pattern}")
    return paths[-1]


def latest_required_log(repo_root: Path, relative_glob: str) -> Path:
    paths = sorted(
        repo_root.glob(relative_glob),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )
    if not paths:
        raise FileNotFoundError(f"no files matching {repo_root / relative_glob}")
    return paths[-1]
