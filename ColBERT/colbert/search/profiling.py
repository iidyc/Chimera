from __future__ import annotations

import threading
import time
from collections import defaultdict
from contextlib import contextmanager

import torch


_THREAD_LOCAL = threading.local()


def set_active_profiler(profiler):
    _THREAD_LOCAL.profiler = profiler


def get_active_profiler():
    return getattr(_THREAD_LOCAL, "profiler", None)


@contextmanager
def activated_profiler(profiler):
    previous = get_active_profiler()
    set_active_profiler(profiler)
    try:
        yield profiler
    finally:
        set_active_profiler(previous)


class SearchBreakdownProfiler:
    def __init__(self, use_cuda: bool):
        self.use_cuda = bool(use_cuda)
        self.query_count = 0
        self.stage_time_s = defaultdict(float)
        self.stage_calls = defaultdict(int)
        self.transfer_time_s = defaultdict(float)
        self.transfer_bytes = defaultdict(int)
        self.transfer_calls = defaultdict(int)

    def sync_cuda(self):
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    def mark_query(self):
        self.query_count += 1

    def record_stage(self, name: str, elapsed_s: float):
        self.stage_time_s[name] += float(elapsed_s)
        self.stage_calls[name] += 1

    def record_transfer(self, name: str, elapsed_s: float, num_bytes: int):
        self.transfer_time_s[name] += float(elapsed_s)
        self.transfer_bytes[name] += int(num_bytes)
        self.transfer_calls[name] += 1

    @contextmanager
    def time(self, name: str, cuda: bool = False):
        if cuda:
            self.sync_cuda()
        start = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                self.sync_cuda()
            self.record_stage(name, time.perf_counter() - start)

    def summary_rows(self, metadata: dict[str, object]):
        queries = max(1, self.query_count)
        rows = []

        for name in sorted(self.stage_time_s):
            rows.append({
                **metadata,
                "metric_type": "stage",
                "metric_name": name,
                "avg_ms": f"{(self.stage_time_s[name] * 1000.0) / queries:.6f}",
                "calls_per_query": f"{self.stage_calls[name] / queries:.6f}",
                "avg_mib": "",
            })

        for name in sorted(self.transfer_time_s):
            rows.append({
                **metadata,
                "metric_type": "transfer_h2d",
                "metric_name": name,
                "avg_ms": f"{(self.transfer_time_s[name] * 1000.0) / queries:.6f}",
                "calls_per_query": f"{self.transfer_calls[name] / queries:.6f}",
                "avg_mib": f"{(self.transfer_bytes[name] / queries) / (1024.0 * 1024.0):.6f}",
            })

        return rows


@contextmanager
def profile_section(name: str, cuda: bool = False):
    profiler = get_active_profiler()
    if profiler is None:
        yield
        return

    with profiler.time(name, cuda=cuda):
        yield


def move_to_cuda(tensor: torch.Tensor, transfer_name: str | None = None, dtype=None):
    if not torch.is_tensor(tensor):
        return tensor

    needs_cuda = not tensor.is_cuda
    needs_dtype = dtype is not None and tensor.dtype != dtype
    if not needs_cuda and not needs_dtype:
        return tensor

    profiler = get_active_profiler()
    if profiler is not None and needs_cuda:
        profiler.sync_cuda()
        start = time.perf_counter()
        result = tensor.cuda()
        if needs_dtype:
            result = result.to(dtype=dtype)
        profiler.sync_cuda()
        profiler.record_transfer(
            transfer_name or "unnamed",
            time.perf_counter() - start,
            tensor.numel() * tensor.element_size(),
        )
        return result

    result = tensor.cuda() if needs_cuda else tensor
    if needs_dtype:
        if result.is_cuda and profiler is not None:
            with profiler.time("tensor_dtype_cast", cuda=True):
                result = result.to(dtype=dtype)
        else:
            result = result.to(dtype=dtype)
    return result
