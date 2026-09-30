#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Benchmark the maintained PyG AGNN example with the custom CSR stage."""

from __future__ import annotations

import argparse
from collections import namedtuple
import ctypes
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu


@dataclass
class Layout:
    source: torch.Tensor
    target: torch.Tensor
    row_ptr: torch.Tensor
    padded_edge_ids: torch.Tensor
    padded_mask: torch.Tensor
    max_degree: int

    def to(self, device):
        return Layout(
            self.source.to(device),
            self.target.to(device),
            self.row_ptr.to(device),
            self.padded_edge_ids.to(device),
            self.padded_mask.to(device),
            self.max_degree,
        )


def build_layout(edge_index: torch.Tensor, nodes: int) -> Layout:
    neighbors = [set() for _ in range(nodes)]
    for source, target in edge_index.t().tolist():
        if source != target:
            neighbors[target].add(source)
    rows = [[target, *sorted(values)] for target, values in enumerate(neighbors)]
    max_degree = max(map(len, rows))
    source = []
    target = []
    row_ptr = [0]
    edge_ids = torch.zeros((nodes, max_degree), dtype=torch.int64)
    mask = torch.zeros((nodes, max_degree), dtype=torch.bool)
    for index, values in enumerate(rows):
        begin = len(source)
        source.extend(values)
        target.extend([index] * len(values))
        row_ptr.append(len(source))
        edge_ids[index, : len(values)] = torch.arange(begin, len(source))
        mask[index, : len(values)] = True
    return Layout(
        torch.tensor(source),
        torch.tensor(target),
        torch.tensor(row_ptr, dtype=torch.int32),
        edge_ids,
        mask,
        max_degree,
    )


def repeat_layout(features, layout: Layout, copies: int):
    if copies == 1:
        return features, layout
    nodes = features.size(0)
    edges = layout.source.numel()
    source = torch.cat([layout.source + index * nodes for index in range(copies)])
    target = torch.cat([layout.target + index * nodes for index in range(copies)])
    row_ptr = torch.cat(
        [layout.row_ptr[:-1] + index * edges for index in range(copies)]
        + [layout.row_ptr[-1:] + (copies - 1) * edges]
    )
    edge_ids = torch.cat(
        [layout.padded_edge_ids + index * edges for index in range(copies)]
    )
    return features.repeat(copies, 1), Layout(
        source,
        target,
        row_ptr,
        edge_ids,
        layout.padded_mask.repeat(copies, 1),
        layout.max_degree,
    )


def resident_stage(features: torch.Tensor, beta: torch.Tensor, layout: Layout):
    normalized = F.normalize(features, p=2.0, dim=-1)
    logits = beta * (normalized[layout.target] * normalized[layout.source]).sum(-1)
    padded_logits = logits[layout.padded_edge_ids].masked_fill(
        ~layout.padded_mask, -torch.inf
    )
    weights = torch.softmax(padded_logits, dim=1).masked_fill(~layout.padded_mask, 0.0)
    return (weights[..., None] * features[layout.source][layout.padded_edge_ids]).sum(1)


class Operator:
    def __init__(self, build: Path, device):
        for directory in (build / "lib", build):
            for path in directory.glob("lib*_kernel_lib.so"):
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        name = "libcsr_agnn_cosine_attention_aggregate_fused_host.so"
        host = build / "lib" / name
        if not host.exists():
            host = build / name
        if not host.exists():
            raise FileNotFoundError(f"AGNN host library not found in {build}")
        self.library = ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        self.device = device
        self.cache = {}
        workspace = (
            self.library.aclnnCsrAgnnCosineAttentionAggregateFusedGetWorkspaceSize
        )
        workspace.argtypes = [ctypes.c_int64] * 4
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrAgnnCosineAttentionAggregateFused
        operation.argtypes = (
            [ctypes.c_void_p] * 5
            + [ctypes.c_int64] * 4
            + [ctypes.c_float, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        operation.restype = ctypes.c_int32

    def __call__(self, features, normalized, beta, layout: Layout):
        key = (
            features.size(0),
            layout.source.numel(),
            features.size(1),
            layout.max_degree,
        )
        if key not in self.cache:
            size = int(
                self.library.aclnnCsrAgnnCosineAttentionAggregateFusedGetWorkspaceSize(
                    *key
                )
            )
            self.cache[key] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty_like(features),
                torch.empty_like(features),
                layout.source.to(torch.int32),
            )
        workspace, output0, output1, source = self.cache[key]
        output = output0
        if output.data_ptr() in (features.data_ptr(), normalized.data_ptr()):
            output = output1
        result = self.library.aclnnCsrAgnnCosineAttentionAggregateFused(
            layout.row_ptr.data_ptr(),
            source.data_ptr(),
            features.data_ptr(),
            normalized.data_ptr(),
            output.data_ptr(),
            *key,
            float(beta.detach().cpu()),
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"custom AGNN operator returned {result}")
        return output


class Model(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_channels, 16)
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.lin2 = torch.nn.Linear(16, output_channels)

    def forward(self, features, layout, aggregate):
        features = F.relu(self.lin1(features))
        features = aggregate(features, torch.ones_like(self.beta), layout)
        features = aggregate(features, self.beta, layout)
        return self.lin2(features)


def timed(function, warmup, repeat):
    result = None
    for _ in range(warmup):
        result = function()
        torch_npu.npu.synchronize()
    samples = []
    for _ in range(repeat):
        torch_npu.npu.synchronize()
        begin = time.perf_counter()
        result = function()
        torch_npu.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1000.0)
    return float(np.median(samples)), result, samples


def digest(path):
    value = hashlib.sha256()
    value.update(path.read_bytes())
    return value.hexdigest()


ModelCallbacks = namedtuple(
    "ModelCallbacks",
    "official_model padded_model custom_model "
    "official_component resident_component custom_component",
)


def model_callbacks(context, features, layout):
    model, operator, official1, official2, _device = context
    edge_index = torch.stack((layout.source, layout.target))

    def official_model(features=features, edge_index=edge_index):
        hidden = F.relu(model.lin1(features))
        hidden = official1(hidden, edge_index)
        hidden = official2(hidden, edge_index)
        return model.lin2(hidden)

    def padded_model(features=features, layout=layout):
        return model(features, layout, resident_stage)

    def custom_stage(values, beta, graph):
        normalized = F.normalize(values, p=2.0, dim=-1)
        return operator(values, normalized, beta, graph)

    def custom_model(features=features, layout=layout):
        return model(features, layout, custom_stage)

    hidden = F.relu(model.lin1(features)).contiguous()

    def official_component(hidden=hidden, edge_index=edge_index):
        return official1(hidden, edge_index)

    def resident_component(hidden=hidden, layout=layout):
        return resident_stage(hidden, torch.ones_like(model.beta), layout)

    def custom_component(hidden=hidden, layout=layout):
        return custom_stage(hidden, torch.ones_like(model.beta), layout)

    return ModelCallbacks(
        official_model,
        padded_model,
        custom_model,
        official_component,
        resident_component,
        custom_component,
    )


def summarize_copy(model_runs, component_runs, copies, features, layout):
    (
        (official_ms, official_out, official_samples),
        (resident_ms, resident_out, resident_samples),
        (custom_ms, custom_out, custom_samples),
    ) = model_runs
    (
        (official_component_ms, official_component_out, _),
        (resident_component_ms, resident_component_out, _),
        (custom_component_ms, custom_component_out, _),
    ) = component_runs
    strongest_model = min(official_ms, resident_ms)
    strongest_component = min(official_component_ms, resident_component_ms)
    return {
        "copies": copies,
        "nodes": features.size(0),
        "edges": layout.source.numel(),
        "official_model_ms": official_ms,
        "resident_model_ms": resident_ms,
        "custom_model_ms": custom_ms,
        "strongest_model_ms": strongest_model,
        "e2e_reduction_percent": (strongest_model - custom_ms)
        / strongest_model
        * 100.0,
        "official_component_ms": official_component_ms,
        "resident_component_ms": resident_component_ms,
        "custom_component_ms": custom_component_ms,
        "component_speedup": strongest_component / custom_component_ms,
        "component_error": float(
            torch.maximum(
                (official_component_out - custom_component_out).abs().max(),
                (resident_component_out - custom_component_out).abs().max(),
            ).cpu()
        ),
        "official_model_error": float((official_out - custom_out).abs().max().cpu()),
        "resident_model_error": float((resident_out - custom_out).abs().max().cpu()),
        "prediction_agreement": float(
            (official_out.argmax(-1) == custom_out.argmax(-1)).float().mean().cpu()
        ),
        "official_samples_ms": official_samples,
        "resident_samples_ms": resident_samples,
        "custom_samples_ms": custom_samples,
    }


def benchmark_copy(args, data, base_layout, copies, context):
    device = context[-1]
    features, layout = repeat_layout(data.x, base_layout, copies)
    features, layout = features.to(device), layout.to(device)
    callbacks = model_callbacks(context, features, layout)
    runs = [timed(callback, args.warmup, args.repeat) for callback in callbacks]
    return summarize_copy(runs[:3], runs[3:], copies, features, layout)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import AGNNConv
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    base_layout = build_layout(data.edge_index, data.num_nodes)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Model(data.num_features, dataset.num_classes)
    state = payload["state_dict"]
    model.lin1.load_state_dict(
        {"weight": state["lin1.weight"], "bias": state["lin1.bias"]}
    )
    model.lin2.load_state_dict(
        {"weight": state["lin2.weight"], "bias": state["lin2.bias"]}
    )
    model.beta.data.copy_(state["prop2.beta"])

    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    model = model.to(device).eval()
    operator = Operator(args.build, device)
    official1 = AGNNConv(requires_grad=False, add_self_loops=False).to(device).eval()
    official2 = AGNNConv(requires_grad=True, add_self_loops=False).to(device).eval()
    official2.beta.data.copy_(model.beta.data)

    context = model, operator, official1, official2, device
    results = []
    for copies in args.copies:
        results.append(benchmark_copy(args, data, base_layout, copies, context))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "operator": "CsrAgnnCosineAttentionAggregateFused",
                "model": "maintained PyG examples/agnn.py",
                "dataset": "Planetoid Cora",
                "checkpoint_sha256": digest(args.checkpoint),
                "checkpoint_validation_accuracy": payload["validation_accuracy"],
                "checkpoint_test_accuracy": payload["test_accuracy"],
                "max_degree": base_layout.max_degree,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logging.getLogger(__name__).info("%s", json.dumps(results, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
