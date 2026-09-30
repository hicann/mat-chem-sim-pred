#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark fused SortAggregation in a PyG DGCNN graph classifier."""

from __future__ import annotations

import argparse as _argparse
import collections as _collections
import ctypes as _ctypes
from dataclasses import dataclass
import hashlib as _hashlib
import json as _json
import logging as _logging
from pathlib import Path as _Path
import time as _time

import numpy as np
import torch
import torch.nn.functional as F

LOGGER = _logging.getLogger("sort_aggregation.dgcnn_benchmark")
BENCHMARK_VARIANT = "sort_aggregation"


class DgcnnGraphTopology:
    """Normalized graph layout for DGCNN GCN evaluation."""

    def __init__(self, src: torch.Tensor, tgt: torch.Tensor, wt: torch.Tensor):
        self.source = src
        self.target = tgt
        self.weight = wt

    def to(self, device):
        return DgcnnGraphTopology(self.source.to(device), self.target.to(device), self.weight.to(device))


class DgcnnTimingBundle:
    """Evaluation bundle used by sort timing benchmark."""

    def __init__(self, model, batch, custom, layout, warmup: int, repeat: int):
        self.model = model
        self.batch = batch
        self.custom = custom
        self.layout = layout
        self.warmup = warmup
        self.repeat = repeat


class DgcnnClassifier(torch.nn.Module):
    def __init__(self, inputs, hidden, classes, layers=3, top_k=30):
        super().__init__()
        from torch_geometric.nn import GCNConv
        from torch_geometric.nn.aggr import SortAggregation

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(inputs, hidden))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.pool = SortAggregation(top_k)
        self.top_k = top_k
        channels = hidden * layers
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(top_k * channels, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, classes),
        )

    def node_features(self, data):
        outputs = []
        features = data.x
        for conv in self.convs:
            features = torch.tanh(conv(features, data.edge_index))
            outputs.append(features)
        return torch.cat(outputs, dim=-1)

    def resident_node_features(self, data, layout):
        outputs = []
        features = data.x
        for conv in self.convs:
            projected = conv.lin(features)
            aggregated = torch.zeros_like(projected)
            messages = projected[layout.source] * layout.weight[:, None]
            aggregated.index_add_(0, layout.target, messages)
            features = torch.tanh(aggregated + conv.bias)
            outputs.append(features)
        return torch.cat(outputs, dim=-1)

    def forward(self, data):
        pooled = self.pool(self.node_features(data), data.batch)
        return self.classifier(pooled)


def accuracy(logits, labels):
    return float((logits.argmax(-1) == labels).float().mean())


def make_split(dataset):
    generator = torch.Generator().manual_seed(20260801)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    split = int(0.8 * len(order))
    return order[:split], order[split:]


def _fit_classifier_head(clf, train_pair, test_pair, epoch_count):
    train_x, train_y = train_pair
    test_x, test_y = test_pair
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-4)
    clf.train()
    step = 0
    while step < epoch_count:
        opt.zero_grad()
        loss = F.cross_entropy(clf(train_x), train_y)
        loss.backward()
        opt.step()
        step += 1
    clf.eval()
    with torch.no_grad():
        acc = accuracy(clf(test_x), test_y)
    return acc


def train_or_load(model, dataset, checkpoint, epochs):
    from torch_geometric.data import Batch

    tr_idx, te_idx = make_split(dataset)
    if checkpoint.exists():
        ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt_data["state"])
        return ckpt_data, tr_idx, te_idx

    tr_data = Batch.from_data_list([dataset[i] for i in tr_idx])
    te_data = Batch.from_data_list([dataset[i] for i in te_idx])
    model.eval()
    with torch.no_grad():
        tr_reps = model.pool(model.node_features(tr_data), tr_data.batch)
        te_reps = model.pool(model.node_features(te_data), te_data.batch)

    test_acc = _fit_classifier_head(
        model.classifier, (tr_reps, tr_data.y), (te_reps, te_data.y), epochs
    )
    ckpt_payload = {
        "benchmark_variant": "sort_aggregation",
        "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metrics": {"test_accuracy": test_acc},
        "split": {"train_graphs": len(tr_idx), "test_graphs": len(te_idx)},
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt_payload, checkpoint)
    return ckpt_payload, tr_idx, te_idx


class CustomOperator:
    def __init__(self, build, device):
        for path in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            _ctypes.CDLL(str(path), mode=_ctypes.RTLD_GLOBAL)
        self.library = _ctypes.CDLL(
            str(build / "libsort_aggregation_topk_fused_host.so"),
            mode=_ctypes.RTLD_GLOBAL,
        )
        self.get_workspace_size = (
            self.library.aclnnSortAggregationTopkFusedGetWorkspaceSize
        )
        self.get_workspace_size.argtypes = [_ctypes.c_int64] * 4
        self.get_workspace_size.restype = _ctypes.c_uint64
        self.operator = self.library.aclnnSortAggregationTopkFused
        self.operator.argtypes = [_ctypes.c_void_p] * 3 + [_ctypes.c_int64] * 4
        self.operator.argtypes += [_ctypes.c_void_p, _ctypes.c_uint64, _ctypes.c_void_p]
        self.operator.restype = _ctypes.c_int32
        self.device = device
        self.cache = {}
        self.keepalive = _collections.deque(maxlen=256)

    def __call__(self, features, graph_ptr, top_k):
        dimensions = (features.size(0), graph_ptr.numel() - 1, features.size(1), top_k)
        if dimensions not in self.cache:
            workspace_size = int(self.get_workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty((workspace_size,), dtype=torch.uint8, device=self.device),
                torch.empty(
                    (dimensions[1], top_k * dimensions[2]),
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            _ctypes.c_void_p(graph_ptr.data_ptr()),
            _ctypes.c_void_p(features.data_ptr()),
            _ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            _ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            _ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"aclnnSortAggregationTopkFused returned {result}")
        self.keepalive.append((features, graph_ptr, workspace, output))
        return output


def benchmark_timer_ms(action, num_warmups: int, num_runs: int):
    res = None
    w = 0
    while w < num_warmups:
        res = action()
        torch.npu.synchronize()
        w += 1
    durations = []
    for _ in range(num_runs):
        torch.npu.synchronize()
        t0 = _time.perf_counter()
        res = action()
        torch.npu.synchronize()
        durations.append((_time.perf_counter() - t0) * 1000.0)
    return float(np.median(durations)), res


def compute_file_sha256(filepath):
    hasher = _hashlib.sha256()
    with open(filepath, "rb") as stream:
        hasher.update(stream.read())
    return hasher.hexdigest()


def build_batch(dataset, indices, graph_count):
    from torch_geometric.data import Batch

    graphs = [dataset[indices[index % len(indices)]] for index in range(graph_count)]
    return Batch.from_data_list(graphs)


def build_gcn_layout(batch):
    from torch_geometric.nn.conv.gcn_conv import gcn_norm

    LOGGER.debug("building sort graph layout")
    edge_index, weight = gcn_norm(
        batch.edge_index,
        edge_weight=None,
        num_nodes=batch.num_nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=batch.x.dtype,
    )
    edge_index = edge_index.contiguous()
    return DgcnnGraphTopology(edge_index[0], edge_index[1], weight)


def _timed_paths(context):
    model, batch, custom = context.model, context.batch, context.custom
    layout, warmup, repeat = context.layout, context.warmup, context.repeat
    layout = build_gcn_layout(batch.clone().cpu()).to(batch.x.device)
    with torch.no_grad():
        official_node_features = model.node_features(batch)
        node_features = model.resident_node_features(batch, layout)
        graph_ptr = batch.ptr.to(torch.int32)

        def official_stage():
            return model.pool(node_features, batch.batch)

        def custom_stage():
            return custom(node_features, graph_ptr, model.top_k)

        official_stage_ms, official_stage_output = benchmark_timer_ms(
            official_stage, warmup, repeat
        )
        custom_stage_ms, custom_stage_output = benchmark_timer_ms(
            custom_stage, warmup, repeat
        )

        def official_e2e():
            return model(batch)

        def resident_e2e():
            features = model.resident_node_features(batch, layout)
            return model.classifier(model.pool(features, batch.batch))

        def custom_e2e():
            features = model.resident_node_features(batch, layout)
            return model.classifier(custom(features, graph_ptr, model.top_k))

        official_e2e_ms, _official_output = benchmark_timer_ms(
            official_e2e, warmup, repeat
        )
        resident_e2e_ms, resident_output = benchmark_timer_ms(
            resident_e2e, warmup, repeat
        )
        custom_e2e_ms, custom_output = benchmark_timer_ms(
            custom_e2e, warmup, repeat
        )
        official_regression_output = model.classifier(
            model.pool(official_node_features, batch.batch)
        )

    return locals()


def benchmark_case(model, batch, custom, warmup, repeat):
    layout = build_gcn_layout(batch.clone().cpu()).to(batch.x.device)
    values = _timed_paths(
        DgcnnTimingBundle(model, batch, custom, layout, warmup, repeat)
    )
    return _case_metrics(values, batch, model)


def _case_metrics(values, batch, model):
    official_stage_ms = values["official_stage_ms"]
    custom_stage_ms = values["custom_stage_ms"]
    official_e2e_ms = values["official_e2e_ms"]
    resident_e2e_ms = values["resident_e2e_ms"]
    custom_e2e_ms = values["custom_e2e_ms"]
    node_features = values["node_features"]
    strongest_e2e_ms = min(official_e2e_ms, resident_e2e_ms)
    metrics = {
        "graphs": batch.num_graphs,
        "nodes": batch.num_nodes,
        "edges": batch.num_edges,
        "channels": node_features.size(1),
        "top_k": model.top_k,
        "official_stage_ms": official_stage_ms,
        "custom_stage_ms": custom_stage_ms,
        "stage_speedup": official_stage_ms / custom_stage_ms,
        "stage_reduction_pct": (official_stage_ms - custom_stage_ms)
        / official_stage_ms
        * 100.0,
        "official_e2e_ms": official_e2e_ms,
        "resident_equivalent_e2e_ms": resident_e2e_ms,
        "strongest_e2e_ms": strongest_e2e_ms,
        "custom_e2e_ms": custom_e2e_ms,
        "e2e_speedup": strongest_e2e_ms / custom_e2e_ms,
        "e2e_reduction_pct": (strongest_e2e_ms - custom_e2e_ms)
        / strongest_e2e_ms
        * 100.0,
        "stage_hotspot_pct": official_stage_ms / resident_e2e_ms * 100.0,
    }
    metrics.update(_quality_metrics(values, batch))
    return metrics


def _quality_metrics(values, batch):
    official_stage_output = values["official_stage_output"]
    custom_stage_output = values["custom_stage_output"]
    official_node_features = values["official_node_features"]
    node_features = values["node_features"]
    official_regression_output = values["official_regression_output"]
    resident_output = values["resident_output"]
    custom_output = values["custom_output"]
    return {
        "stage_max_error": float(
            (official_stage_output - custom_stage_output).abs().max().cpu()
        ),
        "official_to_resident_node_max_error": float(
            (official_node_features - node_features).abs().max().cpu()
        ),
        "official_to_resident_model_max_error": float(
            (official_regression_output - resident_output).abs().max().cpu()
        ),
        "model_max_error": float((resident_output - custom_output).abs().max().cpu()),
        "prediction_agreement": float(
            (resident_output.argmax(-1) == custom_output.argmax(-1))
            .float()
            .mean()
            .cpu()
        ),
        "baseline_accuracy": accuracy(resident_output, batch.y),
        "custom_accuracy": accuracy(custom_output, batch.y),
    }


def parse_args():
    cli_parser = _argparse.ArgumentParser(description="SortAggregation DGCNN Benchmark")
    spec_table = [
        ("--dataset-root", _Path, True, None),
        ("--checkpoint", _Path, True, None),
        ("--build", _Path, True, None),
        ("--output", _Path, True, None),
        ("--epochs", int, False, 300),
        ("--warmup", int, False, 10),
        ("--repeat", int, False, 50),
    ]
    for opt_name, opt_type, is_req, default_val in spec_table:
        if is_req:
            cli_parser.add_argument(opt_name, type=opt_type, required=True)
        else:
            cli_parser.add_argument(opt_name, type=opt_type, default=default_val)
    cli_parser.add_argument("--graphs", type=int, nargs="+", default=[16, 64, 188, 376])
    return cli_parser.parse_args()


def run_benchmark(arguments):
    import torch_npu
    from torch_geometric.datasets import TUDataset

    _ = torch_npu
    torch.manual_seed(20260801)
    dataset = TUDataset(str(arguments.dataset_root), "MUTAG")
    target_device = torch.device("npu:0")
    torch.npu.set_device(target_device)

    model = DgcnnClassifier(dataset.num_features, 32, dataset.num_classes)
    checkpoint, _, test_indices = train_or_load(
        model, dataset, arguments.checkpoint, arguments.epochs
    )
    model.to(target_device).eval()
    custom_op = CustomOperator(arguments.build.resolve(), target_device)

    test_batches = (
        build_batch(dataset, test_indices, g_cnt).to(target_device)
        for g_cnt in arguments.graphs
    )
    cases = [
        benchmark_case(model, b, custom_op, arguments.warmup, arguments.repeat)
        for b in test_batches
    ]
    return {
        "operator": "sort_aggregation_topk",
        "model": "PyG DGCNN graph classifier with SortAggregation",
        "model_contract": "3x GCNConv(32), concatenate 96 channels, SortAggregation(k=30), MLP",
        "source_call": "torch_geometric.nn.aggr.SortAggregation.forward",
        "dataset": "PyG TUDataset MUTAG: 188 molecular graphs, 7 node features, 2 classes",
        "checkpoint": {
            "model_kind": "sort",
            "path": str(arguments.checkpoint),
            "sha256": compute_file_sha256(arguments.checkpoint),
            "metrics": checkpoint["metrics"],
            "split": checkpoint["split"],
        },
        "training": "fixed three-layer GCN representation with a trained graph-classification MLP",
        "baseline_policy": (
            "faster of official PyG and equivalent resident torch_npu GCN path; "
            "output regression uses the deterministic equivalent path"
        ),
        "timing": "resident NPU, synchronized wall-clock median; H2D and D2H excluded",
        "results": cases,
    }


def main():
    arguments = parse_args()
    result = run_benchmark(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(_json.dumps(result, indent=2) + "\n")
    LOGGER.info(
        "sort benchmark result: %s", _json.dumps(result, indent=2)
    )


if __name__ == "__main__":
    main()
