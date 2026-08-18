# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Shared graph, benchmark, training, and profiling helpers for attention operators."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


LOGGER = logging.getLogger("attention_model_e2e")


def configure_logging() -> None:
    """Configure concise command-line logging once."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclass
class AttentionLayout:
    """Target-sorted CSR plus a padded edge view for resident baselines."""

    source: torch.Tensor
    target: torch.Tensor
    row_ptr: torch.Tensor
    edge_ids: torch.Tensor
    mask: torch.Tensor
    max_degree: int
    dropped_edges: int = 0

    def to(self, device: torch.device) -> AttentionLayout:
        return AttentionLayout(
            source=self.source.to(device),
            target=self.target.to(device),
            row_ptr=self.row_ptr.to(device),
            edge_ids=self.edge_ids.to(device),
            mask=self.mask.to(device),
            max_degree=self.max_degree,
            dropped_edges=self.dropped_edges,
        )


def _neighbor_rows(edge_index: torch.Tensor, node_count: int, self_loops: bool):
    neighbors = [set() for _ in range(node_count)]
    for source, target in edge_index.t().tolist():
        if source != target:
            neighbors[target].add(source)
    rows = []
    for target, sources in enumerate(neighbors):
        values = sorted(sources)
        if self_loops:
            values.insert(0, target)
        rows.append(values)
    return rows


def build_layout(
    edge_index: torch.Tensor,
    node_count: int,
    self_loops: bool = False,
    max_degree: int | None = None,
) -> AttentionLayout:
    """Build deterministic CSR and padded layouts, optionally truncating fan-in."""
    rows = _neighbor_rows(edge_index, node_count, self_loops)
    width = max(len(values) for values in rows)
    if max_degree is not None:
        if max_degree < 1:
            raise ValueError("max_degree must be positive")
        width = max_degree
    sources, targets, row_ptr = [], [], [0]
    edge_ids = torch.zeros((node_count, width), dtype=torch.int64)
    mask = torch.zeros((node_count, width), dtype=torch.bool)
    dropped_edges = 0
    for target, values in enumerate(rows):
        dropped_edges += max(0, len(values) - width)
        selected = values[:width]
        begin = len(sources)
        sources.extend(selected)
        targets.extend([target] * len(selected))
        row_ptr.append(len(sources))
        edge_ids[target, : len(selected)] = torch.arange(begin, len(sources))
        mask[target, : len(selected)] = True
    return AttentionLayout(
        source=torch.tensor(sources, dtype=torch.int64),
        target=torch.tensor(targets, dtype=torch.int64),
        row_ptr=torch.tensor(row_ptr, dtype=torch.int32),
        edge_ids=edge_ids,
        mask=mask,
        max_degree=width,
        dropped_edges=dropped_edges,
    )


def repeat_layout(features: torch.Tensor, layout: AttentionLayout, copies: int):
    """Repeat disconnected copies of a graph without changing per-row fan-in."""
    if copies < 1:
        raise ValueError("copies must be positive")
    if copies == 1:
        return features, layout
    node_count = features.size(0)
    edge_count = layout.source.numel()
    sources = [layout.source + index * node_count for index in range(copies)]
    targets = [layout.target + index * node_count for index in range(copies)]
    row_parts = [layout.row_ptr[:-1] + index * edge_count for index in range(copies)]
    row_parts.append(layout.row_ptr[-1:] + (copies - 1) * edge_count)
    edge_parts = [layout.edge_ids + index * edge_count for index in range(copies)]
    repeated = AttentionLayout(
        source=torch.cat(sources),
        target=torch.cat(targets),
        row_ptr=torch.cat(row_parts),
        edge_ids=torch.cat(edge_parts),
        mask=layout.mask.repeat(copies, 1),
        max_degree=layout.max_degree,
        dropped_edges=layout.dropped_edges * copies,
    )
    return features.repeat(copies, 1), repeated


def load_acl_library(build_dir: Path, host_filename: str):
    """Load generated kernels globally, then return the requested host library."""
    library_dirs = (build_dir / "lib", build_dir)
    kernel_paths = set()
    for directory in library_dirs:
        kernel_paths.update(directory.glob("lib*_kernel_lib.so"))
    for path in sorted(kernel_paths):
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    for directory in library_dirs:
        host_path = directory / host_filename
        if host_path.is_file():
            return ctypes.CDLL(str(host_path), mode=ctypes.RTLD_GLOBAL)
    raise FileNotFoundError(f"{host_filename} not found below {build_dir}")


def timed(function: Callable[[], torch.Tensor], warmup_runs: int, repeat: int):
    """Measure synchronized NPU execution and return median, output, and samples."""
    import torch_npu

    result = None
    for _ in range(warmup_runs):
        result = function()
        torch_npu.npu.synchronize()
    samples = []
    for _ in range(repeat):
        torch_npu.npu.synchronize()
        begin = time.perf_counter()
        result = function()
        torch_npu.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1000.0)
    samples.sort()
    middle = len(samples) // 2
    median = samples[middle]
    if len(samples) % 2 == 0:
        median = (samples[middle - 1] + samples[middle]) / 2.0
    return float(median), result, samples


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        block = stream.read(1024 * 1024)
        while block:
            digest.update(block)
            block = stream.read(1024 * 1024)
    return digest.hexdigest()


def prefixed_state(state: dict, prefix: str) -> dict:
    """Select and strip one module prefix without a complex comprehension."""
    selected = {}
    for key, value in state.items():
        if key.startswith(prefix):
            selected[key.removeprefix(prefix)] = value
    return selected


@dataclass
class AttentionBenchmarkCalls:
    official_model: Callable
    resident_model: Callable
    custom_model: Callable
    official_component: Callable
    resident_component: Callable
    custom_component: Callable


def run_attention_benchmark(
    calls: AttentionBenchmarkCalls, warmup_runs: int, repeat: int
):
    """Measure the six model/component paths shared by attention benchmarks."""
    measurements = {}
    for name in (
        "official_model",
        "resident_model",
        "custom_model",
        "official_component",
        "resident_component",
        "custom_component",
    ):
        measurements[name] = timed(getattr(calls, name), warmup_runs, repeat)
    return measurements


def comparison_result(copies: int, nodes: int, edges: int, measurements: dict):
    """Create the common correctness and latency record for one graph size."""
    official_ms, official_output, official_samples = measurements["official_model"]
    resident_ms, resident_output, resident_samples = measurements["resident_model"]
    custom_ms, custom_output, custom_samples = measurements["custom_model"]
    official_component_ms = measurements["official_component"][0]
    resident_component_ms, resident_component_output, _ = measurements[
        "resident_component"
    ]
    custom_component_ms, custom_component_output, _ = measurements["custom_component"]
    strongest_model = min(official_ms, resident_ms)
    strongest_component = min(official_component_ms, resident_component_ms)
    return {
        "copies": copies,
        "nodes": nodes,
        "edges": edges,
        "official_model_ms": official_ms,
        "resident_model_ms": resident_ms,
        "custom_model_ms": custom_ms,
        "e2e_reduction_percent": 100.0
        * (strongest_model - custom_ms)
        / strongest_model,
        "official_component_ms": official_component_ms,
        "resident_component_ms": resident_component_ms,
        "custom_component_ms": custom_component_ms,
        "component_speedup": strongest_component / custom_component_ms,
        "component_error": float(
            (resident_component_output - custom_component_output).abs().max().cpu()
        ),
        "model_error": float(
            torch.maximum(
                (official_output - custom_output).abs().max(),
                (resident_output - custom_output).abs().max(),
            ).cpu()
        ),
        "prediction_agreement": float(
            (official_output.argmax(-1) == custom_output.argmax(-1))
            .float()
            .mean()
            .cpu()
        ),
        "official_samples_ms": official_samples,
        "resident_samples_ms": resident_samples,
        "custom_samples_ms": custom_samples,
    }


def training_arguments(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    return parser.parse_args()


def _accuracy(model, data, mask) -> float:
    prediction = model(data.x, data.edge_index).argmax(-1)
    return float((prediction[mask] == data.y[mask]).float().mean())


def _clone_state(model) -> dict:
    state = {}
    for key, value in model.state_dict().items():
        state[key] = value.detach().cpu().clone()
    return state


def train_cora_model(model, data, epochs: int) -> dict:
    """Train a full-graph Cora classifier and retain the best validation state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best = {"validation_accuracy": -1.0, "test_accuracy": 0.0, "state_dict": None}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(
            logits[data.train_mask], data.y[data.train_mask]
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = _accuracy(model, data, data.val_mask)
            test = _accuracy(model, data, data.test_mask)
        if validation > best["validation_accuracy"]:
            best.update(
                validation_accuracy=validation,
                test_accuracy=test,
                state_dict=_clone_state(model),
            )
    return best


def save_training_result(result: dict, output: Path, epochs: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**result, "epochs": epochs}, output)
    LOGGER.info(
        "validation=%.4f test=%.4f",
        result["validation_accuracy"],
        result["test_accuracy"],
    )


def profile_once(function: Callable, trace_dir: Path) -> None:
    import torch_npu

    experimental_config = getattr(torch_npu.profiler, "_ExperimentalConfig")
    experimental = experimental_config(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        data_simplification=True,
    )
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(trace_dir), analyse_flag=True, async_mode=False
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
        on_trace_ready=handler,
        record_shapes=True,
        with_modules=True,
        experimental_config=experimental,
    ) as profiler:
        with torch.inference_mode():
            function()
            torch_npu.npu.synchronize()
            profiler.step()


def parse_profile(trace_dir: Path, stage_types: set[str]) -> dict:
    paths = sorted(trace_dir.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one kernel CSV, got {paths}")
    durations, counts = defaultdict(float), defaultdict(int)
    with paths[0].open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            kind = row["Type"].strip()
            durations[kind] += float(row["Duration(us)"].strip())
            counts[kind] += 1
    total = sum(durations.values())
    stage = sum(durations[kind] for kind in stage_types)
    by_type = []
    for kind in sorted(durations, key=durations.get, reverse=True):
        by_type.append(
            {
                "type": kind,
                "count": counts[kind],
                "duration_us": durations[kind],
                "full_model_ratio": durations[kind] / total,
            }
        )
    return {
        "kernel_count": sum(counts.values()),
        "full_model_kernel_us": total,
        "conservative_stage_kernel_us": stage,
        "conservative_stage_hotspot_pct": 100.0 * stage / total,
        "stage_kernel_types": sorted(stage_types),
        "by_type": by_type,
    }


def warmup(function: Callable, count: int = 3) -> None:
    import torch_npu

    with torch.inference_mode():
        for _ in range(count):
            function()
        torch_npu.npu.synchronize()


@dataclass(frozen=True)
class ProfileConfig:
    kind: str
    module_name: str
    default_copies: tuple[int, ...]
    stage_types: frozenset[str]


@dataclass
class ResidentModelCall:
    model: torch.nn.Module
    features: torch.Tensor
    layout: AttentionLayout
    aggregate: Callable

    def __call__(self):
        return self.model(self.features, self.layout, self.aggregate)


def _profile_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int)
    return parser.parse_args()


def _load_benchmark_module(script_dir: Path, module_name: str):
    sys.path.insert(0, str(script_dir))
    return importlib.import_module(module_name)


@dataclass
class ProfileSession:
    module: object
    model: torch.nn.Module
    data: object
    base_layout: AttentionLayout
    args: object
    config: ProfileConfig


def _profile_case(session: ProfileSession, copies: int):
    features, layout = session.module.repeat_layout(
        session.data.x, session.base_layout, copies
    )
    features = features.to(torch.device("npu:0"))
    layout = layout.to(torch.device("npu:0"))
    call = ResidentModelCall(
        session.model, features, layout, session.module.resident_aggregate
    )
    warmup(call)
    trace_dir = session.args.trace_root / f"{session.config.kind}_copies_{copies}"
    profile_once(call, trace_dir)
    return {
        "copies": copies,
        "nodes": features.size(0),
        "edges": layout.source.numel(),
        "replaceable_calls": 2,
        **parse_profile(trace_dir, set(session.config.stage_types)),
    }


def profile_main(script_path: Path, config: ProfileConfig) -> None:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    configure_logging()
    args = _profile_arguments()
    module = _load_benchmark_module(script_path.parent, config.module_name)
    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    layout = module.build_layout(data.edge_index, data.num_nodes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = module.Model(dataset.num_features, dataset.num_classes)
    module.load_weights(model, checkpoint["state_dict"])
    model = model.eval().to(torch.device("npu:0"))
    session = ProfileSession(module, model, data, layout, args, config)
    results = []
    for copies in args.copies or config.default_copies:
        result = _profile_case(session, copies)
        results.append(result)
        LOGGER.info("%s", json.dumps(result))
    payload = {
        "candidate": config.kind,
        "model": "complete maintained PyG two-layer model",
        "profiled_baseline": "exact resident padded NPU implementation",
        "dataset": "Cora",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
