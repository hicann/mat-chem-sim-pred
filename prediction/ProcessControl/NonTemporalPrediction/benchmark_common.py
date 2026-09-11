# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Shared helpers for model-level CSR operator benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class TimingRecord:
    nodes: int
    edges: int
    channels: int
    resident_ms: float
    custom_ms: float
    expected: torch.Tensor
    actual: torch.Tensor
    extra: dict


def timed(function, warmup, repeat):
    output = None
    for _ in range(warmup):
        output = function()
        torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        torch.npu.synchronize()
        start = time.perf_counter()
        output = function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(samples)), output


def hash_file(path: Path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def repeat_layout(layout, copies, nodes):
    if copies == 1:
        return layout
    edges = layout.source.numel()
    fields = {
        "source": torch.cat([layout.source + index * nodes for index in range(copies)]),
        "source_i32": torch.cat(
            [layout.source_i32 + index * nodes for index in range(copies)]
        ),
        "target": torch.cat([layout.target + index * nodes for index in range(copies)]),
        "row_ptr": torch.cat(
            [layout.row_ptr[:-1] + index * edges for index in range(copies)]
            + [layout.row_ptr[-1:] + (copies - 1) * edges]
        ),
    }
    if hasattr(layout, "edge_weight"):
        fields["edge_weight"] = layout.edge_weight.repeat(copies)
    return type(layout)(**fields)


def timing_result(record: TimingRecord):
    result = {
        "nodes": record.nodes,
        "edges": record.edges,
        "channels": record.channels,
        "resident_ms": record.resident_ms,
        "custom_ms": record.custom_ms,
        "speedup": record.resident_ms / record.custom_ms,
        "reduction_pct": (record.resident_ms - record.custom_ms)
        / record.resident_ms
        * 100.0,
        "max_error": float((record.expected - record.actual).abs().max().cpu()),
    }
    result.update(record.extra)
    return result


def relative_metrics(prefix, baseline_ms, custom_ms):
    return {
        f"{prefix}_speedup": baseline_ms / custom_ms,
        f"{prefix}_reduction_pct": (baseline_ms - custom_ms) / baseline_ms * 100.0,
    }


def output_metrics(prefix, expected, actual):
    return {f"{prefix}_max_error": float((expected - actual).abs().max().cpu())}


def setup_runtime(module, build):
    import torch_npu

    _ = torch_npu
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    operator = getattr(module, "CustomOperator", module)
    return device, operator(build.resolve(), device)


def parse_model_arguments(include_bootstrap=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    if include_bootstrap:
        parser.add_argument("--bootstrap-checkpoint", type=Path)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 4, 16, 32, 64])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def appnp_helpers():
    return locals()


def sage_helpers():
    return locals()


def appnp_shape_helpers():
    return TimingRecord, load_module, setup_runtime, timing_result


def sage_shape_helpers():
    return TimingRecord, load_module, setup_runtime, timing_result
