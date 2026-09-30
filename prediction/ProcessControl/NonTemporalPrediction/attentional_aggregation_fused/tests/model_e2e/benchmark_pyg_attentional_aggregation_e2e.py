#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark fused PyG AttentionalAggregation with a resident NPU baseline."""

from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# Attention benchmark dependencies.
import numpy as np
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)
BENCHMARK_VARIANT = "attentional_aggregation"


@dataclass
class GcnLayout:
    """Normalized graph layout used by the attention baseline."""

    source: torch.Tensor
    target: torch.Tensor
    weight: torch.Tensor

    def to(self, device):
        return GcnLayout(*(value.to(device) for value in self.__dict__.values()))


@dataclass
class TimingContext:
    model: object
    batch: object
    custom: object
    layout: object
    warmup: int
    repeat: int


def build_gcn_layout(batch):
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    LOGGER.debug("building attention graph layout")
    edge_index, weight = gcn_norm(
        batch.edge_index,
        edge_weight=None,
        num_nodes=batch.num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=batch.x.dtype,
    )
    return GcnLayout(edge_index[0], edge_index[1], weight)


class AttentionPoolClassifier(torch.nn.Module):
    def __init__(self, inputs, classes, hidden=32, layers=3):
        super().__init__()
        from torch_geometric.nn import GCNConv
        from torch_geometric.nn.aggr import AttentionalAggregation

        self.convs = torch.nn.ModuleList([GCNConv(inputs, hidden)])
        self.convs.extend(GCNConv(hidden, hidden) for _ in range(layers - 1))
        self.pool = AttentionalAggregation(torch.nn.Linear(hidden, 1))
        self.classifier = torch.nn.Linear(hidden, classes)

    def node_features(self, data):
        features = data.x
        for conv in self.convs:
            features = F.relu(conv(features, data.edge_index))
        return features

    def resident_node_features(self, data, layout):
        features = data.x
        for conv in self.convs:
            projected = conv.lin(features)
            messages = projected[layout.source] * layout.weight[:, None]
            aggregated = torch.zeros_like(projected)
            aggregated.index_add_(0, layout.target, messages)
            features = F.relu(aggregated + conv.bias)
        return features

    def forward(self, data):
        return self.classifier(self.pool(self.node_features(data), data.batch))


def dense_pool(features, gates, batch):
    from torch_geometric.utils import to_dense_batch

    dense_features, mask = to_dense_batch(features, batch)
    dense_gates, _ = to_dense_batch(gates, batch)
    dense_gates = dense_gates.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return (torch.softmax(dense_gates, dim=1) * dense_features).sum(dim=1)


def accuracy(logits, labels):
    return float((logits.argmax(-1) == labels).float().mean())


def make_split(dataset):
    generator = torch.Generator().manual_seed(20260801)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    split = int(0.8 * len(indices))
    return indices[:split], indices[split:]


def build_batch(dataset, indices, graph_count):
    from torch_geometric.data import Batch

    return Batch.from_data_list(
        [dataset[indices[index % len(indices)]] for index in range(graph_count)]
    )


def train_or_load(model, dataset, checkpoint, epochs):
    logging.getLogger(__name__).debug("training attention benchmark")
    train_indices, test_indices = make_split(dataset)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state"])
        return payload, test_indices
    train_batch = build_batch(dataset, train_indices, len(train_indices))
    test_batch = build_batch(dataset, test_indices, len(test_indices))
    LOGGER.debug("training attention baseline")
    model.eval()
    with torch.no_grad():
        train_features = model.pool(model.node_features(train_batch), train_batch.batch)
        test_features = model.pool(model.node_features(test_batch), test_batch.batch)
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=0.01, weight_decay=1.0e-4
    )
    model.classifier.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(model.classifier(train_features), train_batch.y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_accuracy = accuracy(model.classifier(test_features), test_batch.y)
    payload = {
        "benchmark_variant": "attentional_aggregation",
        "state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "metrics": {"test_accuracy": test_accuracy},
        "split": {"train_graphs": len(train_indices), "test_graphs": len(test_indices)},
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    return payload, test_indices


class CustomOperator:
    def __init__(self, build, device):
        for path in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(
            str(build / "libattentional_aggregation_fused_host.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.workspace_size = (
            self.library.aclnnAttentionalAggregationFusedGetWorkspaceSize
        )
        self.workspace_size.argtypes = [ctypes.c_int64] * 3
        self.workspace_size.restype = ctypes.c_uint64
        self.operator = self.library.aclnnAttentionalAggregationFused
        self.operator.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
        self.operator.argtypes += [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.cache = {}
        self.keepalive = collections.deque(maxlen=256)

    def __call__(self, features, gates, graph_ptr):
        dimensions = (features.size(0), graph_ptr.numel() - 1, features.size(1))
        if dimensions not in self.cache:
            workspace_size = int(self.workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty((workspace_size,), dtype=torch.uint8, device=self.device),
                torch.empty(
                    (dimensions[1], dimensions[2]),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            ctypes.c_void_p(graph_ptr.data_ptr()),
            ctypes.c_void_p(features.data_ptr()),
            ctypes.c_void_p(gates.reshape(-1).data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"aclnnAttentionalAggregationFused returned {result}")
        self.keepalive.append((features, gates, graph_ptr, workspace, output))
        return output


def timed(function, warmup, repeat):
    LOGGER.debug("timing attention benchmark")
    logging.getLogger(__name__).debug("timing %s", BENCHMARK_VARIANT)
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


def hash_file(path):
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _timed_paths(context):
    model, batch, custom = context.model, context.batch, context.custom
    layout, warmup, repeat = context.layout, context.warmup, context.repeat
    graph_ptr = batch.ptr.to(torch.int32)
    with torch.no_grad():
        official_features = model.node_features(batch)
        resident_features = model.resident_node_features(batch, layout)
        resident_gates = model.pool.gate_nn(resident_features)
        pyg_pool = model.pool(resident_features, batch.batch)
        dense_pyg_pool = dense_pool(resident_features, resident_gates, batch.batch)

        def dense_stage():
            return dense_pool(resident_features, resident_gates, batch.batch)

        def custom_stage():
            return custom(resident_features, resident_gates, graph_ptr)

        def official_model():
            return model(batch)

        def dense_model():
            features = model.resident_node_features(batch, layout)
            gates = model.pool.gate_nn(features)
            return model.classifier(dense_pool(features, gates, batch.batch))

        def custom_model():
            features = model.resident_node_features(batch, layout)
            gates = model.pool.gate_nn(features)
            return model.classifier(custom(features, gates, graph_ptr))

        dense_stage_ms, expected_stage = timed(dense_stage, warmup, repeat)
        custom_stage_ms, actual_stage = timed(custom_stage, warmup, repeat)
        official_ms, _ = timed(official_model, warmup, repeat)
        dense_ms, expected_model = timed(dense_model, warmup, repeat)
        custom_ms, actual_model = timed(custom_model, warmup, repeat)
        official_regression = model.classifier(
            model.pool(official_features, batch.batch)
        )
    return locals()


def benchmark_case(model, batch, custom, warmup, repeat):
    layout = build_gcn_layout(batch.clone().cpu()).to(batch.x.device)
    values = _timed_paths(TimingContext(model, batch, custom, layout, warmup, repeat))
    resident_features = values["resident_features"]
    dense_stage_ms = values["dense_stage_ms"]
    custom_stage_ms = values["custom_stage_ms"]
    expected_stage = values["expected_stage"]
    actual_stage = values["actual_stage"]
    official_ms = values["official_ms"]
    dense_ms = values["dense_ms"]
    expected_model = values["expected_model"]
    custom_ms = values["custom_ms"]
    actual_model = values["actual_model"]
    official_regression = values["official_regression"]
    pyg_pool = values["pyg_pool"]
    dense_pyg_pool = values["dense_pyg_pool"]
    strongest_ms = dense_ms
    return {
        "graphs": batch.num_graphs,
        "nodes": batch.num_nodes,
        "edges": batch.num_edges,
        "channels": resident_features.size(1),
        "resident_stage_ms": dense_stage_ms,
        "custom_stage_ms": custom_stage_ms,
        "stage_speedup": dense_stage_ms / custom_stage_ms,
        "stage_reduction_pct": (dense_stage_ms - custom_stage_ms)
        / dense_stage_ms
        * 100.0,
        "official_pyg_e2e_ms": official_ms,
        "dense_resident_e2e_ms": dense_ms,
        "strongest_e2e_ms": strongest_ms,
        "custom_e2e_ms": custom_ms,
        "e2e_speedup": strongest_ms / custom_ms,
        "e2e_reduction_pct": (strongest_ms - custom_ms) / strongest_ms * 100.0,
        "stage_hotspot_pct": dense_stage_ms / dense_ms * 100.0,
        "stage_max_error": float((expected_stage - actual_stage).abs().max().cpu()),
        "pyg_pool_semantic_max_error": float(
            (pyg_pool - dense_pyg_pool).abs().max().cpu()
        ),
        "official_to_resident_model_max_error": float(
            (official_regression - expected_model).abs().max().cpu()
        ),
        "model_max_error": float((expected_model - actual_model).abs().max().cpu()),
        "prediction_agreement": float(
            (expected_model.argmax(-1) == actual_model.argmax(-1)).float().mean().cpu()
        ),
        "baseline_accuracy": accuracy(expected_model, batch.y),
        "custom_accuracy": accuracy(actual_model, batch.y),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    LOGGER.debug("parsing attention benchmark arguments")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--graphs", type=int, nargs="+", default=[16, 64, 188, 376])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def run_benchmark(arguments):
    LOGGER.debug("running attention benchmark")
    from torch_geometric.datasets import TUDataset

    torch.manual_seed(20260801)
    dataset = TUDataset(str(arguments.dataset_root), "MUTAG")
    model = AttentionPoolClassifier(dataset.num_features, dataset.num_classes)
    checkpoint, test_indices = train_or_load(
        model, dataset, arguments.checkpoint, arguments.epochs
    )
    import torch_npu

    _ = torch_npu
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model.to(device).eval()
    custom = CustomOperator(arguments.build.resolve(), device)
    cases = []
    for graph_count in arguments.graphs:
        cases.append(
            benchmark_case(
                model,
                build_batch(dataset, test_indices, graph_count).to(device),
                custom,
                arguments.warmup,
                arguments.repeat,
            )
        )
    return {
        "operator": "attentional_aggregation",
        "model": "PyG GCN + AttentionalAggregation graph classifier",
        "model_contract": "3x GCNConv(32) -> AttentionalAggregation(Linear(32,1)) -> Linear(32,2)",
        "source_call": "torch_geometric.nn.aggr.AttentionalAggregation.forward",
        "dataset": "PyG TUDataset MUTAG: 188 molecular graphs, 7 node features, 2 classes",
        "checkpoint": {
            "model_kind": "attention",
            "path": str(arguments.checkpoint),
            "sha256": hash_file(arguments.checkpoint),
            "metrics": checkpoint["metrics"],
            "split": checkpoint["split"],
        },
        "training": "fixed GCN representation with a trained graph-classification head",
        "baseline_policy": (
            "equivalent all-NPU dense-mask stable-softmax path; direct PyG is "
            "a semantic reference only because scatter_reduce falls back to CPU"
        ),
        "timing": "resident NPU, synchronized wall-clock median; H2D and D2H excluded",
        "results": cases,
    }


def main():
    LOGGER.debug("writing attention benchmark output")
    arguments = parse_args()
    result = run_benchmark(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    logging.getLogger(__name__).info(
        "attention benchmark result: %s", json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
