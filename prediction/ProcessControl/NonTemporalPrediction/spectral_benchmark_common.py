# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shared runtime helpers for spectral graph operator benchmarks."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import statistics
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasisOperatorSpec:
    library: str
    workspace_symbol: str
    operator_symbol: str
    planes: int


class CustomBasisOperator:
    def __init__(
        self, build: Path, device: torch.device, spec: BasisOperatorSpec
    ) -> None:
        for library in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(str(build / spec.library), mode=ctypes.RTLD_GLOBAL)
        self.workspace_size = getattr(self.library, spec.workspace_symbol)
        self.workspace_size.argtypes = [ctypes.c_int64] * 3
        self.workspace_size.restype = ctypes.c_uint64
        self.operator = getattr(self.library, spec.operator_symbol)
        self.operator.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 3
        self.operator.argtypes += [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.planes = spec.planes
        self.cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(
        self,
        row_ptr: torch.Tensor,
        source_index: torch.Tensor,
        norm: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        dimensions = (features.size(0), source_index.numel(), features.size(1))
        if dimensions not in self.cache:
            size = int(self.workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty(
                    (self.planes, dimensions[0], dimensions[2]),
                    dtype=features.dtype,
                    device=self.device,
                ),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            ctypes.c_void_p(row_ptr.data_ptr()),
            ctypes.c_void_p(source_index.data_ptr()),
            ctypes.c_void_p(norm.data_ptr()),
            ctypes.c_void_p(features.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"custom operator returned {result}")
        return output


def disjoint_copies(
    features: torch.Tensor, edge_index: torch.Tensor, copies: int
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = features.size(0)
    shifted = [edge_index + copy * nodes for copy in range(copies)]
    return features.repeat(copies, 1), torch.cat(shifted, dim=1)


def timed(function, warmup: int, iterations: int) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = function()
    torch.npu.synchronize()
    samples = []
    for _ in range(iterations):
        torch.npu.synchronize()
        start = time.perf_counter()
        output = function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0)
    if output is None:
        raise RuntimeError("benchmark produced no output")
    return statistics.median(samples), output


def accuracy(output: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float((output.argmax(dim=-1)[mask] == labels[mask]).float().mean().cpu())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(partial(stream.read, 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def component_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser


def sorted_csr(normalized_edges, norm, nodes: int):
    source, target = normalized_edges
    order = target.argsort(stable=True)
    source, target, norm = source[order], target[order], norm[order]
    counts = torch.bincount(target, minlength=nodes).to(torch.int32)
    row_ptr = torch.empty(nodes + 1, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = counts.cumsum(0, dtype=torch.int32)
    return row_ptr, source.to(torch.int32), source, target, norm.contiguous()


@dataclass
class ComponentSpec:
    data: object
    custom: CustomBasisOperator
    layout: object
    native: object
    stage: str


@dataclass
class ComponentCliSpec:
    custom_type: object
    layout: object
    native: object
    transform: object
    stage: str


@dataclass
class BasisCase:
    features: torch.Tensor
    row_ptr: torch.Tensor
    source_i32: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    norm: torch.Tensor


def _basis_case(spec: ComponentSpec, copies: int, device: torch.device) -> BasisCase:
    features, edges = disjoint_copies(spec.data.x, spec.data.edge_index, copies)
    row_ptr, source_i32, source, target, norm = spec.layout(edges, features.size(0))
    return BasisCase(
        features.to(device).contiguous(),
        row_ptr.to(device),
        source_i32.to(device),
        source.to(device),
        target.to(device),
        norm.to(device),
    )


def _component_result(args, spec: ComponentSpec, case: BasisCase, copies: int) -> dict:
    native_call = partial(
        spec.native, case.features, case.source, case.target, case.norm
    )
    custom_call = partial(
        spec.custom, case.row_ptr, case.source_i32, case.norm, case.features
    )
    native_ms, expected = timed(native_call, args.warmup, args.iterations)
    custom_ms, actual = timed(custom_call, args.warmup, args.iterations)
    return {
        "copies": copies,
        "nodes": int(case.features.size(0)),
        "edges": int(case.source.numel()),
        "channels": int(case.features.size(1)),
        "native_ms": native_ms,
        "custom_ms": custom_ms,
        "speedup": native_ms / custom_ms,
        "max_abs_error": float((expected - actual).abs().max().cpu()),
    }


def run_component(args, spec: ComponentSpec) -> None:
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    results = []
    with torch.no_grad():
        for copies in args.copies:
            case = _basis_case(spec, copies, device)
            result = _component_result(args, spec, case, copies)
            results.append(result)
            LOGGER.info("%s", json.dumps(result, sort_keys=True))
    write_payload(
        args.output,
        {"model_stage": spec.stage, "dataset": "Cora", "results": results},
    )


def run_component_cli(args, spec: ComponentCliSpec) -> None:
    data = Planetoid(str(args.data_root), name="Cora", transform=spec.transform)[0]
    device = torch.device("npu:0")
    custom = spec.custom_type(args.build.resolve(), device)
    run_component(
        args,
        ComponentSpec(data, custom, spec.layout, spec.native, spec.stage),
    )


def e2e_parser() -> argparse.ArgumentParser:
    parser = component_parser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    return parser


@dataclass
class E2ESpec:
    model_type: object
    custom_type: object
    layout: object
    native: object
    staged_forward: object
    transform: object
    model_label: str


@dataclass
class E2ERuntime:
    data: object
    model: torch.nn.Module
    custom: CustomBasisOperator
    checkpoint: dict
    device: torch.device


@dataclass
class ModelMetrics:
    copies: int
    source: torch.Tensor
    baseline_ms: float
    custom_ms: float
    baseline_output: torch.Tensor
    custom_output: torch.Tensor
    labels: torch.Tensor
    test_mask: torch.Tensor


def _train_checkpoint(model, data, checkpoint: Path, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 25 == 0:
            LOGGER.info("train epoch=%d loss=%.6f", epoch + 1, float(loss))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "epochs": epochs, "seed": 20260802},
        checkpoint,
    )


def _e2e_runtime(args, spec: E2ESpec) -> E2ERuntime:
    torch.manual_seed(20260802)
    dataset = Planetoid(str(args.data_root), name="Cora", transform=spec.transform)
    data = dataset[0]
    cpu_model = spec.model_type(dataset.num_features, 16, dataset.num_classes)
    if not args.checkpoint.exists():
        _train_checkpoint(cpu_model, data, args.checkpoint, args.epochs)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cpu_model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model = cpu_model.eval().to(device)
    custom = spec.custom_type(args.build.resolve(), device)
    return E2ERuntime(data, model, custom, checkpoint, device)


def _e2e_result(args, spec: E2ESpec, runtime: E2ERuntime, copies: int) -> dict:
    features, edges = disjoint_copies(runtime.data.x, runtime.data.edge_index, copies)
    row_ptr, source_i32, source, target, norm = spec.layout(edges, features.size(0))
    features = features.to(runtime.device).contiguous()
    labels = runtime.data.y.repeat(copies).to(runtime.device)
    test_mask = runtime.data.test_mask.repeat(copies).to(runtime.device)
    native_call = partial(
        spec.native,
        source=source.to(runtime.device),
        target=target.to(runtime.device),
        norm=norm.to(runtime.device),
    )
    custom_call = partial(
        runtime.custom,
        row_ptr.to(runtime.device),
        source_i32.to(runtime.device),
        norm.to(runtime.device),
    )
    baseline = partial(spec.staged_forward, runtime.model, features, native_call)
    fused = partial(spec.staged_forward, runtime.model, features, custom_call)
    baseline_ms, baseline_output = timed(baseline, args.warmup, args.iterations)
    custom_ms, custom_output = timed(fused, args.warmup, args.iterations)
    return _model_metrics(
        ModelMetrics(
            copies,
            source,
            baseline_ms,
            custom_ms,
            baseline_output,
            custom_output,
            labels,
            test_mask,
        )
    )


def _model_metrics(context: ModelMetrics) -> dict:
    baseline_output = context.baseline_output
    custom_output = context.custom_output
    return {
        "copies": context.copies,
        "nodes": int(baseline_output.size(0)),
        "edges": int(context.source.numel()),
        "baseline_e2e_ms": context.baseline_ms,
        "custom_e2e_ms": context.custom_ms,
        "e2e_speedup": context.baseline_ms / context.custom_ms,
        "e2e_reduction_pct": (
            100.0 * (context.baseline_ms - context.custom_ms) / context.baseline_ms
        ),
        "max_model_abs_error": float(
            (baseline_output - custom_output).abs().max().cpu()
        ),
        "prediction_agreement": float(
            (baseline_output.argmax(-1) == custom_output.argmax(-1))
            .float()
            .mean()
            .cpu()
        ),
        "baseline_test_accuracy": accuracy(
            baseline_output, context.labels, context.test_mask
        ),
        "custom_test_accuracy": accuracy(
            custom_output, context.labels, context.test_mask
        ),
    }


def run_e2e(args, spec: E2ESpec) -> None:
    runtime = _e2e_runtime(args, spec)
    results = []
    with torch.no_grad():
        for copies in args.copies:
            result = _e2e_result(args, spec, runtime, copies)
            results.append(result)
            LOGGER.info("%s", json.dumps(result, sort_keys=True))
    write_payload(
        args.output,
        {
            "model": spec.model_label,
            "dataset": "Cora",
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epochs": int(runtime.checkpoint["epochs"]),
            "results": results,
        },
    )
