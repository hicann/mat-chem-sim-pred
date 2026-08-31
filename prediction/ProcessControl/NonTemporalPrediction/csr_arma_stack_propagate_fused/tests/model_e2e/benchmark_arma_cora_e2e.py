# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Benchmark fused ARMA propagation stages in a complete Cora model."""

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
import torch_geometric.transforms as T
import torch_npu
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot

LOGGER = logging.getLogger(__name__)


def synchronize() -> None:
    torch_npu.npu.synchronize()


def timed_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return statistics.median(samples)


@dataclass
class GraphLayout:
    row_ptr: torch.Tensor
    source_i32: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    edge_weight: torch.Tensor

    def to(self, device: torch.device) -> GraphLayout:
        return GraphLayout(
            self.row_ptr.to(device),
            self.source_i32.to(device),
            self.source.to(device),
            self.target.to(device),
            self.edge_weight.to(device),
        )


@dataclass
class StageInputs:
    projected: torch.Tensor
    root: torch.Tensor
    bias: torch.Tensor
    relu: bool


class CustomArmaStage:
    def __init__(self, build: Path, device: torch.device) -> None:
        for library in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(
            str(build / "libcsr_arma_stack_propagate_fused_host.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.workspace_size = (
            self.library.aclnnCsrArmaStackPropagateFusedGetWorkspaceSize
        )
        self.workspace_size.argtypes = [ctypes.c_int64] * 4
        self.workspace_size.restype = ctypes.c_uint64
        self.operator = self.library.aclnnCsrArmaStackPropagateFused
        self.operator.argtypes = [ctypes.c_void_p] * 7
        self.operator.argtypes += [ctypes.c_int64] * 5
        self.operator.argtypes += [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.cache = {}

    def __call__(self, graph: GraphLayout, stage: StageInputs) -> torch.Tensor:
        dimensions = (
            stage.projected.size(1),
            graph.source_i32.numel(),
            stage.projected.size(0),
            stage.projected.size(2),
        )
        if dimensions not in self.cache:
            size = int(self.workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty_like(stage.projected),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            ctypes.c_void_p(graph.row_ptr.data_ptr()),
            ctypes.c_void_p(graph.source_i32.data_ptr()),
            ctypes.c_void_p(graph.edge_weight.data_ptr()),
            ctypes.c_void_p(stage.projected.data_ptr()),
            ctypes.c_void_p(stage.root.data_ptr()),
            ctypes.c_void_p(stage.bias.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            int(stage.relu),
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"custom operator returned {result}")
        return output


def disjoint_graph(edge_index, edge_weight, nodes: int, copies: int) -> GraphLayout:
    source, target = edge_index
    offsets = torch.arange(copies).view(-1, 1) * nodes
    source = (source.view(1, -1) + offsets).reshape(-1)
    target = (target.view(1, -1) + offsets).reshape(-1)
    weight = edge_weight.repeat(copies)
    order = torch.argsort(target * (nodes * copies) + source)
    source, target, weight = source[order], target[order], weight[order]
    counts = torch.bincount(target, minlength=nodes * copies).to(torch.int32)
    row_ptr = torch.empty(nodes * copies + 1, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = counts.cumsum(0, dtype=torch.int32)
    return GraphLayout(row_ptr, source.to(torch.int32), source, target, weight)


def native_stage(graph: GraphLayout, stage: StageInputs) -> torch.Tensor:
    messages = stage.projected[:, graph.source, :] * graph.edge_weight.view(1, -1, 1)
    output = torch.zeros_like(stage.projected)
    output.index_add_(1, graph.target, messages)
    output += stage.root + stage.bias.view(stage.bias.size(0), 1, -1)
    return F.relu(output) if stage.relu else output


class ArmaLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, relu: bool):
        super().__init__()
        self.stacks = 3
        self.relu = relu
        self.dropout = 0.25
        self.init_weight = torch.nn.Parameter(
            torch.empty(self.stacks, in_channels, out_channels)
        )
        self.weight = torch.nn.Parameter(
            torch.empty(self.stacks, out_channels, out_channels)
        )
        self.root_weight = torch.nn.Parameter(
            torch.empty(self.stacks, in_channels, out_channels)
        )
        self.bias = torch.nn.Parameter(torch.zeros(self.stacks, out_channels))
        for parameter in (self.init_weight, self.weight, self.root_weight):
            glorot(parameter)

    def forward(self, features, graph, stage_function, capture=None):
        projected = torch.matmul(features.unsqueeze(0), self.init_weight)
        output = self._stage(features, projected, graph, stage_function, capture)
        projected = torch.matmul(output, self.weight)
        return self._stage(
            features,
            projected,
            graph,
            stage_function,
            capture,
        ).mean(dim=0)

    def _stage(self, features, projected, graph, stage_function, capture):
        root = torch.matmul(
            F.dropout(features, self.dropout, self.training).unsqueeze(0),
            self.root_weight,
        )
        stage = StageInputs(projected, root, self.bias, self.relu)
        if capture is not None:
            capture.append(stage)
        return stage_function(graph, stage)


class ArmaNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = ArmaLayer(in_channels, 16, True)
        self.conv2 = ArmaLayer(16, out_channels, False)

    def forward(self, features, graph, stage_function, capture=None):
        features = F.dropout(features, training=self.training)
        features = self.conv1(features, graph, stage_function, capture)
        features = F.relu(features)
        features = F.dropout(features, training=self.training)
        features = self.conv2(features, graph, stage_function, capture)
        return F.log_softmax(features, dim=1)


def masked_accuracy(output, labels, mask) -> float:
    return float((output.argmax(dim=1)[mask] == labels[mask]).float().mean().cpu())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(partial(stream.read, 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class Runtime:
    data: object
    normalized_index: torch.Tensor
    normalized_weight: torch.Tensor
    model: ArmaNet
    custom: CustomArmaStage
    checkpoint: dict
    device: torch.device


@dataclass
class BenchmarkCase:
    copies: int
    graph: GraphLayout
    features: torch.Tensor
    capture: list[StageInputs]
    baseline_output: torch.Tensor
    labels: torch.Tensor
    test_mask: torch.Tensor


@dataclass
class Measurements:
    model_ms: float
    stages_ms: float
    native_ms: float
    custom_ms: float
    custom_model_ms: float
    expected: torch.Tensor
    actual: torch.Tensor
    custom_output: torch.Tensor


def _load_runtime(args) -> Runtime:
    torch.manual_seed(20260802)
    dataset = Planetoid(args.data_root, "Cora", transform=T.NormalizeFeatures())
    data = dataset[0]
    normalized_index, normalized_weight = gcn_norm(
        data.edge_index,
        None,
        data.num_nodes,
        add_self_loops=False,
        dtype=torch.float32,
    )
    device = torch.device("npu:0")
    model = ArmaNet(dataset.num_features, dataset.num_classes).to(device)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    custom = CustomArmaStage(args.build.resolve(), device)
    return Runtime(
        data, normalized_index, normalized_weight, model, custom, checkpoint, device
    )


def _prepare_case(runtime: Runtime, copies: int) -> BenchmarkCase:
    graph = disjoint_graph(
        runtime.normalized_index,
        runtime.normalized_weight,
        runtime.data.num_nodes,
        copies,
    ).to(runtime.device)
    features = runtime.data.x.repeat(copies, 1).to(runtime.device)
    capture = []
    output = runtime.model(features, graph, native_stage, capture)
    synchronize()
    return BenchmarkCase(
        copies,
        graph,
        features,
        capture,
        output,
        runtime.data.y.repeat(copies).to(runtime.device),
        runtime.data.test_mask.repeat(copies).to(runtime.device),
    )


def _custom_stage(custom: CustomArmaStage, graph, stage) -> torch.Tensor:
    contiguous = StageInputs(
        stage.projected.contiguous(),
        stage.root.contiguous(),
        stage.bias.contiguous(),
        stage.relu,
    )
    return custom(graph, contiguous)


def _all_native(graph: GraphLayout, stages: list[StageInputs]):
    return [native_stage(graph, stage) for stage in stages]


def _measure(args, runtime: Runtime, case: BenchmarkCase) -> Measurements:
    custom_stage = partial(_custom_stage, runtime.custom)
    baseline = partial(runtime.model, case.features, case.graph, native_stage)
    custom_model = partial(runtime.model, case.features, case.graph, custom_stage)
    all_stages = partial(_all_native, case.graph, case.capture)
    native_component = partial(native_stage, case.graph, case.capture[0])
    custom_component = partial(custom_stage, case.graph, case.capture[0])
    model_ms = timed_ms(baseline, args.warmup, args.iterations)
    stages_ms = timed_ms(all_stages, args.warmup, args.iterations)
    native_ms = timed_ms(native_component, args.warmup, args.iterations)
    custom_ms = timed_ms(custom_component, args.warmup, args.iterations)
    expected, actual = native_component().clone(), custom_component().clone()
    custom_model_ms = timed_ms(custom_model, args.warmup, args.iterations)
    custom_output = custom_model()
    synchronize()
    return Measurements(
        model_ms,
        stages_ms,
        native_ms,
        custom_ms,
        custom_model_ms,
        expected,
        actual,
        custom_output,
    )


def _result(case: BenchmarkCase, values: Measurements) -> dict:
    output, custom_output = case.baseline_output, values.custom_output
    return {
        "copies": case.copies,
        "nodes": int(case.features.size(0)),
        "edges": int(case.graph.source.numel()),
        "replaceable_calls": len(case.capture),
        "model_ms": values.model_ms,
        "replaceable_stage_ms": values.stages_ms,
        "timed_stage_share_pct": values.stages_ms / values.model_ms * 100.0,
        "native_component_ms": values.native_ms,
        "custom_component_ms": values.custom_ms,
        "component_speedup": values.native_ms / values.custom_ms,
        "component_max_abs_error": float(
            (values.expected - values.actual).abs().max().cpu()
        ),
        "custom_model_ms": values.custom_model_ms,
        "model_reduction_pct": (
            (values.model_ms - values.custom_model_ms) / values.model_ms * 100.0
        ),
        "model_max_abs_error": float((output - custom_output).abs().max().cpu()),
        "prediction_agreement": float(
            (output.argmax(-1) == custom_output.argmax(-1)).float().mean().cpu()
        ),
        "baseline_accuracy": masked_accuracy(output, case.labels, case.test_mask),
        "custom_accuracy": masked_accuracy(custom_output, case.labels, case.test_mask),
        "finite_output": bool(torch.isfinite(output).all().item()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    return parser


def main() -> None:
    args = _parser().parse_args()
    runtime = _load_runtime(args)
    results = []
    with torch.no_grad():
        for copies in args.copies:
            case = _prepare_case(runtime, copies)
            result = _result(case, _measure(args, runtime, case))
            results.append(result)
            LOGGER.info("%s", json.dumps(result, sort_keys=True))
    payload = {
        "source_model": "PyG examples/arma.py",
        "model": "two ARMAConv layers, K=3, T=2, shared weights",
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epochs": runtime.checkpoint["epochs"],
        "checkpoint_test_accuracy": runtime.checkpoint["test_accuracy"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
