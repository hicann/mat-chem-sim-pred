#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Benchmark residual propagation in maintained PyG GCNII on Cora."""

from __future__ import annotations

import ctypes
import json
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm

sys.path.insert(0, str(Path(__file__).parents[3]))
from spectral_benchmark_common import (
    accuracy,
    component_parser,
    disjoint_copies,
    file_sha256,
    timed,
    write_payload,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ResidualGraph:
    row_ptr: torch.Tensor
    source_i32: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    edge_weight: torch.Tensor

    def to(self, device: torch.device) -> ResidualGraph:
        return ResidualGraph(
            self.row_ptr.to(device),
            self.source_i32.to(device),
            self.source.to(device),
            self.target.to(device),
            self.edge_weight.to(device),
        )


class CustomResidualPropagation:
    def __init__(self, build: Path, device: torch.device) -> None:
        for library in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(
            str(build / "libcsr_gcn2_residual_propagate_fused_host.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.workspace_size = (
            self.library.aclnnCsrGcn2ResidualPropagateFusedGetWorkspaceSize
        )
        self.workspace_size.argtypes = [ctypes.c_int64] * 3
        self.workspace_size.restype = ctypes.c_uint64
        self.operator = self.library.aclnnCsrGcn2ResidualPropagateFused
        self.operator.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int64] * 3
        self.operator.argtypes += [
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __call__(
        self,
        graph: ResidualGraph,
        current: torch.Tensor,
        initial: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        dimensions = (current.size(0), graph.source_i32.numel(), current.size(1))
        if dimensions not in self.cache:
            size = int(self.workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty_like(current),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            ctypes.c_void_p(graph.row_ptr.data_ptr()),
            ctypes.c_void_p(graph.source_i32.data_ptr()),
            ctypes.c_void_p(graph.edge_weight.data_ptr()),
            ctypes.c_void_p(current.data_ptr()),
            ctypes.c_void_p(initial.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            alpha,
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"custom operator returned {result}")
        return output


class Gcn2Net(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden: int,
        out_channels: int,
        layers: int = 64,
        alpha: float = 0.1,
        theta: float = 0.5,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        self.input = torch.nn.Linear(in_channels, hidden)
        self.output = torch.nn.Linear(hidden, out_channels)
        self.convs = torch.nn.ModuleList(
            [
                GCN2Conv(
                    hidden,
                    alpha,
                    theta,
                    layer + 1,
                    shared_weights=True,
                    normalize=False,
                )
                for layer in range(layers)
            ]
        )
        self.dropout = dropout

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        features = F.dropout(features, self.dropout, training=self.training)
        features = initial = self.input(features).relu()
        for conv in self.convs:
            features = F.dropout(features, self.dropout, training=self.training)
            features = conv(features, initial, edge_index, edge_weight).relu()
        features = F.dropout(features, self.dropout, training=self.training)
        return F.log_softmax(self.output(features), dim=-1)


def normalized_layout(edge_index: torch.Tensor, nodes: int) -> ResidualGraph:
    normalized_index, edge_weight = gcn_norm(
        edge_index,
        num_nodes=nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=torch.float32,
    )
    source, target = normalized_index
    order = target.argsort(stable=True)
    source, target, edge_weight = source[order], target[order], edge_weight[order]
    counts = torch.bincount(target, minlength=nodes).to(torch.int32)
    row_ptr = torch.empty(nodes + 1, dtype=torch.int32)
    row_ptr[0] = 0
    row_ptr[1:] = counts.cumsum(0, dtype=torch.int32)
    return ResidualGraph(row_ptr, source.to(torch.int32), source, target, edge_weight)


def native_residual(
    current: torch.Tensor,
    initial: torch.Tensor,
    graph: ResidualGraph,
    alpha: float,
) -> torch.Tensor:
    propagated = torch.zeros_like(current)
    messages = current[graph.source] * graph.edge_weight.view(-1, 1)
    propagated.index_add_(0, graph.target, messages)
    return (1.0 - alpha) * propagated + alpha * initial


def staged_forward(
    model: Gcn2Net, features: torch.Tensor, residual_function
) -> torch.Tensor:
    features = initial = model.input(features).relu()
    for conv in model.convs:
        residual = residual_function(features, initial)
        features = torch.addmm(
            residual,
            residual,
            conv.weight1,
            beta=1.0 - conv.beta,
            alpha=conv.beta,
        ).relu()
    return F.log_softmax(model.output(features), dim=-1)


@dataclass
class TrainContext:
    model: Gcn2Net
    data: object
    graph: ResidualGraph
    path: Path
    epochs: int
    device: torch.device


def train_checkpoint(context: TrainContext) -> dict:
    model = context.model.to(context.device)
    features = context.data.x.to(context.device)
    labels = context.data.y.to(context.device)
    train_mask = context.data.train_mask.to(context.device)
    graph = context.graph.to(context.device)
    edge_index = torch.stack([graph.source, graph.target])
    optimizer = torch.optim.Adam(
        [
            {"params": model.convs.parameters(), "weight_decay": 0.01},
            {
                "params": list(model.input.parameters())
                + list(model.output.parameters()),
                "weight_decay": 5e-4,
            },
        ],
        lr=0.01,
    )
    for epoch in range(context.epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(features, edge_index, graph.edge_weight)
        loss = F.nll_loss(prediction[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            LOGGER.info("train epoch=%d loss=%.6f", epoch + 1, float(loss))
    state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload = {"state_dict": state, "epochs": context.epochs, "seed": 20260802}
    context.path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, context.path)
    return payload


@dataclass
class Runtime:
    data: object
    model: Gcn2Net
    custom: CustomResidualPropagation
    checkpoint: dict
    device: torch.device
    alpha: float


@dataclass
class BenchmarkCase:
    copies: int
    features: torch.Tensor
    graph: ResidualGraph
    initial: torch.Tensor
    labels: torch.Tensor
    test_mask: torch.Tensor


def _load_runtime(args) -> Runtime:
    torch.manual_seed(20260802)
    dataset = Planetoid(
        str(args.data_root), name="Cora", transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    graph = normalized_layout(data.edge_index, data.num_nodes)
    cpu_model = Gcn2Net(
        dataset.num_features, 64, dataset.num_classes, layers=args.layers
    )
    device = torch.device("npu:0")
    if args.checkpoint.exists():
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    else:
        torch.npu.set_device(device)
        context = TrainContext(
            cpu_model, data, graph, args.checkpoint, args.epochs, device
        )
        checkpoint = train_checkpoint(context)
        cpu_model = cpu_model.cpu()
    cpu_model.load_state_dict(checkpoint["state_dict"])
    torch.npu.set_device(device)
    model = cpu_model.eval().to(device)
    custom = CustomResidualPropagation(args.build.resolve(), device)
    return Runtime(data, model, custom, checkpoint, device, 0.1)


def _benchmark_case(runtime: Runtime, copies: int) -> BenchmarkCase:
    features, edge_index = disjoint_copies(
        runtime.data.x, runtime.data.edge_index, copies
    )
    features = features.to(runtime.device)
    graph = normalized_layout(edge_index, features.size(0)).to(runtime.device)
    return BenchmarkCase(
        copies,
        features,
        graph,
        runtime.model.input(features).relu(),
        runtime.data.y.repeat(copies).to(runtime.device),
        runtime.data.test_mask.repeat(copies).to(runtime.device),
    )


def _component_result(args, runtime: Runtime, case: BenchmarkCase, native, fused):
    native_call = partial(native, case.initial, case.initial)
    fused_call = partial(fused, case.initial, case.initial)
    native_ms, expected = timed(native_call, args.warmup, args.iterations)
    custom_ms, actual = timed(fused_call, args.warmup, args.iterations)
    return {
        "copies": case.copies,
        "nodes": int(case.features.size(0)),
        "edges": int(case.graph.source.numel()),
        "channels": int(case.initial.size(1)),
        "native_ms": native_ms,
        "custom_ms": custom_ms,
        "speedup": native_ms / custom_ms,
        "max_abs_error": float((expected - actual).abs().max().cpu()),
    }


def _e2e_result(args, runtime: Runtime, case: BenchmarkCase, native, fused):
    baseline = partial(staged_forward, runtime.model, case.features, native)
    custom = partial(staged_forward, runtime.model, case.features, fused)
    baseline_ms, baseline_output = timed(baseline, args.warmup, args.iterations)
    custom_ms, custom_output = timed(custom, args.warmup, args.iterations)
    return {
        "copies": case.copies,
        "layers": len(runtime.model.convs),
        "baseline_ms": baseline_ms,
        "custom_ms": custom_ms,
        "speedup": baseline_ms / custom_ms,
        "reduction_pct": 100.0 * (baseline_ms - custom_ms) / baseline_ms,
        "max_abs_error": float((baseline_output - custom_output).abs().max().cpu()),
        "prediction_agreement": float(
            (baseline_output.argmax(-1) == custom_output.argmax(-1))
            .float()
            .mean()
            .cpu()
        ),
        "baseline_accuracy": accuracy(baseline_output, case.labels, case.test_mask),
        "custom_accuracy": accuracy(custom_output, case.labels, case.test_mask),
    }


def _run_cases(args, runtime: Runtime) -> tuple[list[dict], list[dict]]:
    component_results, e2e_results = [], []
    with torch.inference_mode():
        for copies in args.copies:
            case = _benchmark_case(runtime, copies)
            native = partial(native_residual, graph=case.graph, alpha=runtime.alpha)
            fused = partial(runtime.custom, case.graph, alpha=runtime.alpha)
            component = _component_result(args, runtime, case, native, fused)
            e2e = _e2e_result(args, runtime, case, native, fused)
            component_results.append(component)
            e2e_results.append(e2e)
            LOGGER.info("component=%s", json.dumps(component, sort_keys=True))
            LOGGER.info("e2e=%s", json.dumps(e2e, sort_keys=True))
    return component_results, e2e_results


def _parser():
    parser = component_parser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.set_defaults(warmup=5, iterations=30)
    return parser


def main() -> None:
    args = _parser().parse_args()
    runtime = _load_runtime(args)
    component, e2e = _run_cases(args, runtime)
    write_payload(
        args.output,
        {
            "model": "maintained PyG examples/gcn2_cora.py, 64-layer GCNII",
            "dataset": "Cora",
            "alpha": runtime.alpha,
            "theta": 0.5,
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epochs": runtime.checkpoint["epochs"],
            "component": component,
            "e2e": e2e,
        },
    )


if __name__ == "__main__":
    main()
