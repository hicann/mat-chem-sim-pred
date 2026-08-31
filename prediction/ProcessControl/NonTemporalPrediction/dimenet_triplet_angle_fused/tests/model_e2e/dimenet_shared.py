#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shared, deliberately small helpers for the two DimeNet audits."""

from __future__ import annotations

import importlib.util
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.nn import DimeNetPlusPlus
from torch_geometric.utils import scatter


@dataclass
class ForwardCase:
    model: object
    z: torch.Tensor
    position: torch.Tensor
    batch: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    triplet_provider: object
    angle_function: object


@dataclass
class GraphContext:
    dataset: object
    graph_count: int
    args: object
    device: torch.device
    angle_binding: object
    triplet_binding: object


@dataclass
class MeasureContext:
    graph_count: int
    batch: object
    source: torch.Tensor
    target: torch.Tensor
    model: object
    resident_provider: object
    custom_provider: object
    angle_binding: object = None
    args: object = None


@dataclass
class ResultContext:
    graph_count: int
    max_degree: int
    batch: object
    source: torch.Tensor
    expected: object
    mismatch: int
    resident_stage_ms: float
    custom_stage_ms: float
    resident_e2e_ms: float
    custom_e2e_ms: float
    resident_output: object
    custom_output: object


@dataclass
class ProviderContext:
    shared: object
    row_ptr: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    binding: object
    capacity: int


def add_graph_args(parser):
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--graphs", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)


def add_dimenet_args(parser, with_angle_paths):
    parser.add_argument("--operator-dir", type=Path, required=True)
    if with_angle_paths:
        parser.add_argument("--angle-build-dir", type=Path, required=True)
    parser.add_argument("--triplet-dir", type=Path, required=True)
    parser.add_argument("--triplet-build-dir", type=Path, required=True)
    add_graph_args(parser)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    return statistics.median(samples), output


def fixed_radius_graph(position, batch, cutoff, max_neighbors):
    edge_parts = []
    for graph in range(int(batch.max()) + 1):
        node_ids = (batch == graph).nonzero(as_tuple=False).view(-1)
        distances = torch.cdist(position[node_ids], position[node_ids])
        valid = (distances < cutoff) & (distances > 0)
        for target_local in range(node_ids.numel()):
            source_local = valid[:, target_local].nonzero(as_tuple=False).view(-1)
            if source_local.numel() > max_neighbors:
                nearest = distances[source_local, target_local].argsort()[
                    :max_neighbors
                ]
                source_local = source_local[nearest]
            targets = torch.full_like(source_local, target_local)
            edge_parts.append(
                torch.stack([node_ids[source_local], node_ids[targets]], dim=0)
            )
    return torch.cat(edge_parts, dim=1)


def prepare_graph(batch, cutoff=5.0, max_neighbors=32):
    edge_index = fixed_radius_graph(batch.pos, batch.batch, cutoff, max_neighbors)
    source, target = edge_index
    order = torch.argsort(target * batch.num_nodes + source, stable=True)
    source, target = source[order], target[order]
    degree = torch.bincount(target, minlength=batch.num_nodes)
    row_ptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), degree.cumsum(0).to(torch.int32)]
    )
    return row_ptr, source.to(torch.int32), target.to(torch.int32), int(degree.max())


def resident_triplets(row_ptr, source_index, target_index):
    degree = row_ptr[1:].long() - row_ptr[:-1].long()
    incoming_count = degree.index_select(0, source_index.long())
    edge_ji = torch.repeat_interleave(
        torch.arange(source_index.numel(), device=row_ptr.device), incoming_count
    )
    group_begin = torch.cumsum(incoming_count, dim=0) - incoming_count
    relative = torch.arange(edge_ji.numel(), device=row_ptr.device)
    relative -= torch.repeat_interleave(group_begin, incoming_count)
    middle = source_index.long().index_select(0, edge_ji)
    edge_kj = row_ptr.long().index_select(0, middle) + relative
    idx_i = target_index.long().index_select(0, edge_ji)
    idx_j = middle
    idx_k = source_index.long().index_select(0, edge_kj)
    mask = idx_i != idx_k
    return tuple(value[mask] for value in (idx_i, idx_j, idx_k, edge_kj, edge_ji))


def custom_triplets(binding, row_ptr, source, capacity):
    outputs = binding.dimenet_triplet_enumerate_fused(row_ptr, source, capacity)
    count, overflow = outputs[-1].cpu().tolist()
    if overflow:
        raise RuntimeError("triplet output capacity overflow")
    return tuple(value[:count].contiguous() for value in outputs[:5])


def native_angle(position, idx_i, idx_j, idx_k):
    pos_jk = position.index_select(0, idx_j.long()) - position.index_select(
        0, idx_k.long()
    )
    pos_ij = position.index_select(0, idx_i.long()) - position.index_select(
        0, idx_j.long()
    )
    return torch.atan2(
        torch.cross(pos_ij, pos_jk, dim=1).norm(dim=-1), (pos_ij * pos_jk).sum(dim=-1)
    )


def model_forward(case):
    triplets = case.triplet_provider()
    idx_i, idx_j, idx_k, idx_kj, idx_ji = (value.long() for value in triplets)
    i, j = case.target.long(), case.source.long()
    dist = (case.position[i] - case.position[j]).pow(2).sum(dim=-1).sqrt()
    angle = case.angle_function(case.position, idx_i, idx_j, idx_k)
    rbf = case.model.rbf(dist)
    sbf = case.model.sbf(dist, angle, idx_kj)
    features = case.model.emb(case.z, rbf, i, j)
    prediction = case.model.output_blocks[0](
        features, rbf, i, num_nodes=case.position.size(0)
    )
    for interaction, output in zip(
        case.model.interaction_blocks, case.model.output_blocks[1:]
    ):
        features = interaction(features, rbf, sbf, idx_kj, idx_ji)
        prediction = prediction + output(
            features, rbf, i, num_nodes=case.position.size(0)
        )
    return scatter(prediction, case.batch, dim=0, reduce="sum")


def build_model(device):
    return (
        DimeNetPlusPlus(
            hidden_channels=64,
            out_channels=1,
            num_blocks=3,
            int_emb_size=32,
            basis_emb_size=8,
            out_emb_channels=64,
            num_spherical=4,
            num_radial=6,
            cutoff=5.0,
            max_num_neighbors=32,
            num_before_skip=1,
            num_after_skip=2,
            num_output_layers=3,
            output_initializer="glorot_orthogonal",
        )
        .to(device)
        .eval()
    )
