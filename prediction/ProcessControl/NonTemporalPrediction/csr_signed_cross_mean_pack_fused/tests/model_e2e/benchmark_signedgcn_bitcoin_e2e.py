#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark SignedGCN with fused second-layer cross-sign means."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
load_host_library, log_json, sha256, timed_pair = (
    _common.load_host_library,
    _common.log_json,
    _common.sha256,
    _common.timed_pair,
)


@dataclass
class Csr:
    source: torch.Tensor
    target: torch.Tensor
    row_ptr: torch.Tensor
    inverse_degree: torch.Tensor
    edge_ids: torch.Tensor
    mask: torch.Tensor
    max_degree: int

    def to(self, device):
        return Csr(
            self.source.to(device),
            self.target.to(device),
            self.row_ptr.to(device),
            self.inverse_degree.to(device),
            self.edge_ids.to(device),
            self.mask.to(device),
            self.max_degree,
        )


@dataclass
class BenchmarkContext:
    model: object
    operator: object
    warmup: int
    repeat: int


@dataclass
class Case:
    copies: int
    features: torch.Tensor
    positive: Csr
    negative: Csr
    positive_edge: torch.Tensor
    negative_edge: torch.Tensor
    test_edges: torch.Tensor


@dataclass
class Measurements:
    models: object
    components: object
    official_repeat: torch.Tensor
    official_embedding: torch.Tensor
    custom_embedding: torch.Tensor


@dataclass
class SignedSetup:
    payload: dict
    base_values: tuple
    model: object


def build_csr(edge_index, nodes):
    order = sorted(
        range(edge_index.size(1)),
        key=lambda item: (int(edge_index[1, item]), int(edge_index[0, item])),
    )
    order = torch.tensor(order)
    source, target = edge_index[0, order], edge_index[1, order]
    counts = torch.bincount(target, minlength=nodes)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0))).to(
        torch.int32
    )
    inverse = torch.zeros(nodes, dtype=torch.float32)
    valid = counts > 0
    inverse[valid] = counts[valid].float().reciprocal()
    maximum = int(counts.max())
    edge_ids = torch.zeros((nodes, maximum), dtype=torch.int64)
    mask = torch.zeros((nodes, maximum), dtype=torch.bool)
    for node in range(nodes):
        begin, end = int(row_ptr[node]), int(row_ptr[node + 1])
        columns = slice(0, end - begin)
        edge_ids[node, columns] = torch.arange(begin, end)
        mask[node, columns] = True
    return Csr(source, target, row_ptr, inverse, edge_ids, mask, maximum)


def repeat_csr(csr, nodes, copies):
    if copies == 1:
        return csr
    edges = csr.source.numel()
    source = torch.cat([csr.source + index * nodes for index in range(copies)])
    target = torch.cat([csr.target + index * nodes for index in range(copies)])
    rows = [csr.row_ptr[:-1] + index * edges for index in range(copies)]
    rows.append(csr.row_ptr[-1:] + (copies - 1) * edges)
    edge_ids = torch.cat([csr.edge_ids + index * edges for index in range(copies)])
    return Csr(
        source,
        target,
        torch.cat(rows),
        csr.inverse_degree.repeat(copies),
        edge_ids,
        csr.mask.repeat(copies, 1),
        csr.max_degree,
    )


def repeat_edges(edge_index, nodes, copies):
    return torch.cat([edge_index + index * nodes for index in range(copies)], dim=1)


def padded_mean(values, csr):
    gathered = values[csr.source][csr.edge_ids]
    return (
        gathered.masked_fill(~csr.mask[..., None], 0.0).sum(1)
        * csr.inverse_degree[:, None]
    )


def resident_pack(features, positive, negative):
    channels = features.size(1) // 2
    positive_values = features[:, slice(0, channels)]
    negative_values = features[:, slice(channels, None)]
    return torch.cat(
        (
            padded_mean(positive_values, positive),
            padded_mean(negative_values, negative),
            padded_mean(negative_values, positive),
            padded_mean(positive_values, negative),
        ),
        dim=-1,
    )


def native_mean(values, csr):
    output = torch.zeros_like(values)
    output.index_add_(0, csr.target, values[csr.source])
    return output * csr.inverse_degree[:, None]


def native_pack(features, positive, negative):
    channels = features.size(1) // 2
    positive_values = features[:, slice(0, channels)]
    negative_values = features[:, slice(channels, None)]
    return torch.cat(
        (
            native_mean(positive_values, positive),
            native_mean(negative_values, negative),
            native_mean(negative_values, positive),
            native_mean(positive_values, negative),
        ),
        dim=-1,
    )


class Operator:
    def __init__(self, build, device):
        self.library = load_host_library(build, "csr_signed_cross_mean_pack_fused")
        self.device, self.cache, self.positions = device, {}, {}
        workspace = self.library.aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize
        workspace.argtypes = [ctypes.c_int64] * 5
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrSignedCrossMeanPackFused
        operation.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int64] * 5
            + [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_void_p,
            ]
        )
        operation.restype = ctypes.c_int32

    def __call__(self, features, positive, negative):
        channels = features.size(1) // 2
        shape = (
            features.size(0),
            positive.source.numel(),
            negative.source.numel(),
            channels,
            max(positive.max_degree, negative.max_degree),
        )
        if shape not in self.cache:
            size = int(
                self.library.aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize(*shape)
            )
            outputs = [
                torch.empty(
                    (features.size(0), 4 * channels),
                    dtype=features.dtype,
                    device=self.device,
                )
                for _ in range(4)
            ]
            self.cache[shape] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                outputs,
                positive.source.to(torch.int32),
                negative.source.to(torch.int32),
            )
            self.positions[shape] = 0
        workspace, outputs, positive_source, negative_source = self.cache[shape]
        position = self.positions[shape]
        output = outputs[position]
        self.positions[shape] = (position + 1) % len(outputs)
        result = self.library.aclnnCsrSignedCrossMeanPackFused(
            positive.row_ptr.data_ptr(),
            positive_source.data_ptr(),
            negative.row_ptr.data_ptr(),
            negative_source.data_ptr(),
            features.data_ptr(),
            positive.inverse_degree.data_ptr(),
            negative.inverse_degree.data_ptr(),
            output.data_ptr(),
            *shape,
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"custom SignedGCN operator returned {result}")
        return output

    @staticmethod
    def supports(features, positive, negative):
        channels = features.size(1) // 2 if features.ndim == 2 else 0
        return (
            features.dtype == torch.float32
            and features.is_contiguous()
            and features.ndim == 2
            and features.size(1) == 2 * channels
            and 0 < channels <= 64
            and channels % 8 == 0
            and positive.source.numel() > 0
            and negative.source.numel() > 0
            and 0 < max(positive.max_degree, negative.max_degree) <= 1024
        )


def second_layer(model, features, positive, negative, aggregate):
    channels = model.convs[0].in_channels
    packed = aggregate(features, positive, negative)
    positive_values = packed[:, slice(0, 2 * channels)]
    negative_values = packed[:, slice(2 * channels, None)]
    out_pos = model.convs[0].lin_pos_l(positive_values) + model.convs[0].lin_pos_r(
        features[:, slice(0, channels)]
    )
    out_neg = model.convs[0].lin_neg_l(negative_values) + model.convs[0].lin_neg_r(
        features[:, slice(channels, None)]
    )
    return F.relu(torch.cat((out_pos, out_neg), dim=-1))


def replacement_pack(features, positive, negative, operator):
    if operator.supports(features, positive, negative):
        return operator(features, positive, negative)
    return native_pack(features, positive, negative)


def _official_model(model, case):
    embedding = model(case.features, case.positive_edge, case.negative_edge)
    return model.discriminate(embedding, case.test_edges)


def _custom_model(context, case):
    hidden = F.relu(
        context.model.conv1(case.features, case.positive_edge, case.negative_edge)
    )
    aggregate = partial(replacement_pack, operator=context.operator)
    embedding = second_layer(
        context.model, hidden, case.positive, case.negative, aggregate
    )
    return context.model.discriminate(embedding, case.test_edges)


def _component(aggregate, hidden, positive, negative):
    return aggregate(hidden, positive, negative)


def _measure(context, case):
    official = partial(_official_model, context.model, case)
    models = timed_pair(
        official, partial(_custom_model, context, case), context.warmup, context.repeat
    )
    official_repeat = official()
    torch_npu.npu.synchronize()
    with torch.no_grad():
        hidden = F.relu(
            context.model.conv1(case.features, case.positive_edge, case.negative_edge)
        )
        official_embedding = F.relu(
            context.model.convs[0](hidden, case.positive_edge, case.negative_edge)
        )
        custom_embedding = second_layer(
            context.model, hidden, case.positive, case.negative, context.operator
        )
    components = timed_pair(
        partial(_component, native_pack, hidden, case.positive, case.negative),
        partial(_component, context.operator, hidden, case.positive, case.negative),
        context.warmup,
        context.repeat,
    )
    return Measurements(
        models, components, official_repeat, official_embedding, custom_embedding
    )


def _timing_fields(measured):
    return {
        "official_model_ms": measured.models.first.median_ms,
        "custom_model_ms": measured.models.second.median_ms,
        "e2e_reduction_percent": (
            measured.models.first.median_ms - measured.models.second.median_ms
        )
        / measured.models.first.median_ms
        * 100.0,
        "native_component_ms": measured.components.first.median_ms,
        "custom_component_ms": measured.components.second.median_ms,
        "component_speedup": measured.components.first.median_ms
        / measured.components.second.median_ms,
        "official_samples_ms": measured.models.first.samples_ms,
        "custom_samples_ms": measured.models.second.samples_ms,
    }


def _accuracy_fields(measured):
    official = measured.models.first.output
    custom = measured.models.second.output
    official_repeat = measured.official_repeat
    return {
        "component_error": float(
            (measured.components.first.output - measured.components.second.output)
            .abs()
            .max()
            .cpu()
        ),
        "official_model_error": float((official - custom).abs().max().cpu()),
        "official_self_error": float((official - official_repeat).abs().max().cpu()),
        "embedding_error": float(
            (measured.official_embedding - measured.custom_embedding).abs().max().cpu()
        ),
        "prediction_agreement": float(
            (official.argmax(-1) == custom.argmax(-1)).float().mean().cpu()
        ),
        "official_self_prediction_agreement": float(
            (official.argmax(-1) == official_repeat.argmax(-1)).float().mean().cpu()
        ),
        "controlled_prediction_agreement": float(
            (
                measured.official_embedding.argmax(-1)
                == measured.custom_embedding.argmax(-1)
            )
            .float()
            .mean()
            .cpu()
        ),
    }


def prepare_case(payload, base_values, copies):
    base_features, base_positive, base_negative = base_values
    nodes = base_features.size(0)
    features = base_features.repeat(copies, 1).to("npu:0")
    positive = repeat_csr(base_positive, nodes, copies).to("npu:0")
    negative = repeat_csr(base_negative, nodes, copies).to("npu:0")
    positive_edge = torch.stack((positive.source, positive.target))
    negative_edge = torch.stack((negative.source, negative.target))
    test_edges = torch.cat(
        (
            repeat_edges(payload["test_positive"], nodes, copies),
            repeat_edges(payload["test_negative"], nodes, copies),
        ),
        dim=1,
    ).to("npu:0")
    return Case(
        copies, features, positive, negative, positive_edge, negative_edge, test_edges
    )


def run_case(context, payload, base_values, copies):
    case = prepare_case(payload, base_values, copies)
    measured = _measure(context, case)
    result = {
        "copies": copies,
        "nodes": case.features.size(0),
        "positive_edges": case.positive.source.numel(),
        "negative_edges": case.negative.source.numel(),
        "test_pairs": case.test_edges.size(1),
    }
    result.update(_timing_fields(measured))
    result.update(_accuracy_fields(measured))
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=100)
    return parser.parse_args()


def load_signed_setup(checkpoint):
    from torch_geometric.nn import SignedGCN

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    features = payload["features"]
    nodes = features.size(0)
    base_values = (
        features,
        build_csr(payload["train_positive"], nodes),
        build_csr(payload["train_negative"], nodes),
    )
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    model = SignedGCN(64, 64, num_layers=2, lamb=5).to(device).eval()
    model.load_state_dict(payload["state_dict"])
    return SignedSetup(payload, base_values, model)


def main():
    args = parse_args()
    setup = load_signed_setup(args.checkpoint)
    context = BenchmarkContext(
        setup.model,
        Operator(args.build, torch.device("npu:0")),
        args.warmup,
        args.repeat,
    )
    results = [
        run_case(context, setup.payload, setup.base_values, copies)
        for copies in args.copies
    ]
    output = {
        "operator": "CsrSignedCrossMeanPackFused",
        "model": "maintained PyG SignedGCN on Bitcoin-OTC",
        "dataset": "Bitcoin-OTC, 6005 real nodes",
        "source_commit": "pytorch_geometric@003c3cd8a10520567ceaeda619f0315e30ec2f66",
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_auc": setup.payload["auc"],
        "checkpoint_f1": setup.payload["f1"],
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log_json(results, indent=2)


if __name__ == "__main__":
    main()
