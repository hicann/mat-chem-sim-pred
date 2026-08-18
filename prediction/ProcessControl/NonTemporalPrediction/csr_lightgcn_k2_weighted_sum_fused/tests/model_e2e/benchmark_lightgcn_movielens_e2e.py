#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Train and benchmark PyG LightGCN on the MovieLens 100K u1 split."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
load_host_library, log_json, sha256, timed = (
    _common.load_host_library,
    _common.log_json,
    _common.sha256,
    _common.timed,
)


@dataclass
class Graph:
    source: torch.Tensor
    target: torch.Tensor
    row_ptr: torch.Tensor
    norm: torch.Tensor
    nodes: int

    def to(self, device):
        return Graph(
            self.source.to(device),
            self.target.to(device),
            self.row_ptr.to(device),
            self.norm.to(device),
            self.nodes,
        )


@dataclass
class TrainingData:
    train_users: np.ndarray
    train_items: np.ndarray
    test_data: tuple
    dimensions: int
    epochs: int


@dataclass
class BenchmarkContext:
    graph: Graph
    embedding: torch.Tensor
    base_pairs: torch.Tensor
    operator: object
    model_type: type
    warmup: int
    repeat: int


@dataclass
class Measurements:
    official: object
    resident: object
    custom: object
    official_component: object
    resident_component: object
    custom_component: object


@dataclass
class LoadedData:
    training: TrainingData
    graph: Graph
    embedding: torch.Tensor
    base_pairs: torch.Tensor
    recall20: float
    users: int
    items: int


def load_ratings(path: Path):
    values = np.loadtxt(path, delimiter="\t", dtype=np.int64)
    return values[:, 0] - 1, values[:, 1] - 1, values[:, 2]


def build_graph(users, items, user_count, item_count):
    item_nodes = items + user_count
    source = np.concatenate((users, item_nodes))
    target = np.concatenate((item_nodes, users))
    order = np.lexsort((source, target))
    source, target = source[order], target[order]
    nodes = user_count + item_count
    degree = np.bincount(target, minlength=nodes).astype(np.float32)
    inverse = np.power(np.maximum(degree, 1.0), -0.5)
    norm = inverse[source] * inverse[target]
    counts = np.bincount(target, minlength=nodes)
    row_ptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    return Graph(
        torch.from_numpy(source).to(torch.int64),
        torch.from_numpy(target).to(torch.int64),
        torch.from_numpy(row_ptr),
        torch.from_numpy(norm),
        nodes,
    )


def repeat_graph(graph: Graph, copies: int):
    if copies == 1:
        return graph
    edges = graph.source.numel()
    source = torch.cat([graph.source + index * graph.nodes for index in range(copies)])
    target = torch.cat([graph.target + index * graph.nodes for index in range(copies)])
    rows = [graph.row_ptr[:-1] + index * edges for index in range(copies)]
    rows.append(graph.row_ptr[-1:] + (copies - 1) * edges)
    return Graph(
        source, target, torch.cat(rows), graph.norm.repeat(copies), graph.nodes * copies
    )


def sparse_propagate(adjacency, features):
    return torch.sparse.mm(adjacency, features)


def _make_adjacency(graph):
    indices = torch.stack((graph.target, graph.source))
    return torch.sparse_coo_tensor(
        indices, graph.norm, (graph.nodes, graph.nodes)
    ).coalesce()


def _train_embeddings(embeddings, adjacency, graph, training):
    optimizer = torch.optim.Adam([embeddings], lr=0.03)
    generator = torch.Generator().manual_seed(20260802)
    users = torch.from_numpy(training.train_users)
    positives = (
        torch.from_numpy(training.train_items) + int(training.train_users.max()) + 1
    )
    for _ in range(training.epochs):
        negatives = torch.randint(
            int(training.train_users.max()) + 1,
            graph.nodes,
            positives.shape,
            generator=generator,
        )
        layer1 = sparse_propagate(adjacency, embeddings)
        layer2 = sparse_propagate(adjacency, layer1)
        output = (embeddings + layer1 + layer2) / 3.0
        positive_score = (output[users] * output[positives]).sum(-1)
        negative_score = (output[users] * output[negatives]).sum(-1)
        loss = (
            -F.logsigmoid(positive_score - negative_score).mean()
            + 1.0e-5 * embeddings.square().mean()
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _recall_at_20(embeddings, adjacency, training):
    layer1 = sparse_propagate(adjacency, embeddings)
    output = (embeddings + layer1 + sparse_propagate(adjacency, layer1)) / 3.0
    user_count = int(training.train_users.max()) + 1
    score = output[:user_count] @ output[user_count:].t()
    score[
        torch.from_numpy(training.train_users), torch.from_numpy(training.train_items)
    ] = -torch.inf
    top = score.topk(20, dim=1).indices
    test_users, test_items, _ = training.test_data
    positives = [set() for _ in range(user_count)]
    for user, item in zip(test_users, test_items):
        positives[int(user)].add(int(item))
    recalls = []
    for user, truth in enumerate(positives):
        if truth:
            recalls.append(len(truth.intersection(top[user].tolist())) / len(truth))
    return float(np.mean(recalls))


def train_or_load(checkpoint, graph, training):
    torch.manual_seed(20260802)
    embeddings = torch.nn.Parameter(torch.empty(graph.nodes, training.dimensions))
    torch.nn.init.xavier_uniform_(embeddings)
    adjacency = _make_adjacency(graph)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        embeddings.data.copy_(payload["embedding"])
    else:
        _train_embeddings(embeddings, adjacency, graph, training)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"embedding": embeddings.detach(), "epochs": training.epochs}, checkpoint
        )
    with torch.no_grad():
        recall20 = _recall_at_20(embeddings, adjacency, training)
    return embeddings.detach(), recall20


def resident_embedding(features, graph: Graph):
    def propagate(values):
        output = torch.zeros_like(values)
        messages = values[graph.source] * graph.norm[:, None]
        return output.index_add_(0, graph.target, messages)

    layer1 = propagate(features)
    layer2 = propagate(layer1)
    return (features + layer1 + layer2) / 3.0


class Operator:
    def __init__(self, build: Path, device):
        self.library = load_host_library(build, "csr_lightgcn_k2_weighted_sum_fused")
        self.device = device
        self.cache = {}
        workspace = self.library.aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize
        workspace.argtypes = [ctypes.c_int64] * 3
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrLightgcnK2WeightedSumFused
        operation.argtypes = (
            [ctypes.c_void_p] * 5
            + [ctypes.c_int64] * 3
            + [ctypes.c_float] * 3
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        operation.restype = ctypes.c_int32

    def __call__(self, features, graph: Graph):
        key = (graph.nodes, graph.source.numel(), features.size(1))
        if key not in self.cache:
            size = int(
                self.library.aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize(*key)
            )
            self.cache[key] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty_like(features),
                graph.source.to(torch.int32),
            )
        workspace, output, source = self.cache[key]
        result = self.library.aclnnCsrLightgcnK2WeightedSumFused(
            graph.row_ptr.data_ptr(),
            source.data_ptr(),
            graph.norm.data_ptr(),
            features.data_ptr(),
            output.data_ptr(),
            *key,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"custom LightGCN operator returned {result}")
        return output


def _score(values, pairs):
    return (values[pairs[0]] * values[pairs[1]]).sum(-1)


def _official_model(model, edge_index, pairs):
    return model(edge_index, pairs)


def _resident_model(embedding, graph, pairs):
    return _score(resident_embedding(embedding, graph), pairs)


def _custom_model(operator, embedding, graph, pairs):
    return _score(operator(embedding, graph), pairs)


def _measure(context, graph, embedding, edge_index, pairs):
    official = context.model_type(graph.nodes, 64, 2).to("npu:0").eval()
    official.embedding.weight.data.copy_(embedding)
    values = (
        timed(
            partial(_official_model, official, edge_index, pairs),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(_resident_model, embedding, graph, pairs),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(_custom_model, context.operator, embedding, graph, pairs),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(official.get_embedding, edge_index), context.warmup, context.repeat
        ),
        timed(
            partial(resident_embedding, embedding, graph),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(context.operator, embedding, graph), context.warmup, context.repeat
        ),
    )
    return Measurements(*values)


def _result_metrics(measured):
    strongest_model = min(measured.official.median_ms, measured.resident.median_ms)
    strongest_component = min(
        measured.official_component.median_ms, measured.resident_component.median_ms
    )
    return {
        "official_model_ms": measured.official.median_ms,
        "resident_model_ms": measured.resident.median_ms,
        "custom_model_ms": measured.custom.median_ms,
        "e2e_reduction_percent": (strongest_model - measured.custom.median_ms)
        / strongest_model
        * 100.0,
        "official_component_ms": measured.official_component.median_ms,
        "resident_component_ms": measured.resident_component.median_ms,
        "custom_component_ms": measured.custom_component.median_ms,
        "component_speedup": strongest_component / measured.custom_component.median_ms,
        "component_error": float(
            torch.maximum(
                (measured.official_component.output - measured.custom_component.output)
                .abs()
                .max(),
                (measured.resident_component.output - measured.custom_component.output)
                .abs()
                .max(),
            ).cpu()
        ),
        "score_error": float(
            torch.maximum(
                (measured.official.output - measured.custom.output).abs().max(),
                (measured.resident.output - measured.custom.output).abs().max(),
            ).cpu()
        ),
        "binary_prediction_agreement": float(
            ((measured.official.output > 0) == (measured.custom.output > 0))
            .float()
            .mean()
            .cpu()
        ),
        "official_samples_ms": measured.official.samples_ms,
        "resident_samples_ms": measured.resident.samples_ms,
        "custom_samples_ms": measured.custom.samples_ms,
    }


def run_case(context, copies):
    graph = repeat_graph(context.graph, copies).to("npu:0")
    embedding = context.embedding.repeat(copies, 1).to("npu:0")
    pairs = torch.cat(
        [context.base_pairs + index * context.graph.nodes for index in range(copies)],
        dim=1,
    ).to("npu:0")
    edge_index = torch.stack((graph.source, graph.target))
    result = {
        "copies": copies,
        "nodes": graph.nodes,
        "edges": graph.source.numel(),
        "pairs": pairs.size(1),
    }
    result.update(
        _result_metrics(_measure(context, graph, embedding, edge_index, pairs))
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def _load_data(args):
    train_users, train_items, _ = load_ratings(args.dataset_dir / "u1.base")
    test_data = load_ratings(args.dataset_dir / "u1.test")
    user_count = int(max(train_users.max(), test_data[0].max()) + 1)
    item_count = int(max(train_items.max(), test_data[1].max()) + 1)
    graph = build_graph(train_users, train_items, user_count, item_count)
    training = TrainingData(train_users, train_items, test_data, 64, args.epochs)
    embedding, recall20 = train_or_load(args.checkpoint, graph, training)
    test_users, test_items, _ = test_data
    negative_items = (test_items * 131 + test_users * 17 + 29) % item_count
    base_pairs = torch.stack(
        (
            torch.from_numpy(np.concatenate((test_users, test_users))),
            torch.from_numpy(np.concatenate((test_items, negative_items)) + user_count),
        )
    )
    return LoadedData(
        training, graph, embedding, base_pairs, recall20, user_count, item_count
    )


def main():
    args = parse_args()
    from torch_geometric.nn import LightGCN

    loaded = _load_data(args)
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    context = BenchmarkContext(
        loaded.graph,
        loaded.embedding,
        loaded.base_pairs,
        Operator(args.build, device),
        LightGCN,
        args.warmup,
        args.repeat,
    )
    results = [run_case(context, copies) for copies in args.copies]
    payload = {
        "operator": "CsrLightgcnK2WeightedSumFused",
        "model": "maintained PyG examples/lightgcn.py topology",
        "dataset": "MovieLens 100K official u1 split",
        "users": loaded.users,
        "items": loaded.items,
        "train_edges": len(loaded.training.train_users),
        "test_edges": len(loaded.training.test_data[0]),
        "recall_at_20": loaded.recall20,
        "checkpoint_sha256": sha256(args.checkpoint),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_json({"recall_at_20": loaded.recall20, "results": results}, indent=2)


if __name__ == "__main__":
    main()
