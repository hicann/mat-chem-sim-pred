#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark a Cora GATv2 model with dynamic attention fusion."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


ATTENTION_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ATTENTION_ROOT))
COMMON = importlib.import_module("attention_model_e2e_common")
LOGGER = logging.getLogger("benchmark_gatv2_cora")
Layout = COMMON.AttentionLayout


def build_layout(edge_index, nodes):
    return COMMON.build_layout(edge_index, nodes, self_loops=True)


def repeat_layout(features, layout, copies):
    return COMMON.repeat_layout(features, layout, copies)


def resident_aggregate(source_projected, target_projected, attention, layout):
    edge_values = source_projected[layout.source]
    dynamic = F.leaky_relu(
        edge_values + target_projected[layout.target], negative_slope=0.2
    )
    logits = (dynamic * attention).sum(-1)
    padded = logits[layout.edge_ids].masked_fill(~layout.mask[..., None], -torch.inf)
    weights = torch.softmax(padded, dim=1).masked_fill(~layout.mask[..., None], 0.0)
    return (weights[..., None] * edge_values[layout.edge_ids]).sum(1)


class Operator:
    def __init__(self, build, device):
        self.library = COMMON.load_acl_library(
            build, "libcsr_gatv2_dynamic_attention_aggregate_fused_host.so"
        )
        self.device = device
        self.cache = {}
        workspace = (
            self.library.aclnnCsrGatv2DynamicAttentionAggregateFusedGetWorkspaceSize
        )
        workspace.argtypes = [ctypes.c_int64] * 5
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrGatv2DynamicAttentionAggregateFused
        pointer_types = [ctypes.c_void_p] * 6
        shape_types = [ctypes.c_int64] * 5
        tail_types = [ctypes.c_float, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        operation.argtypes = pointer_types + shape_types + tail_types
        operation.restype = ctypes.c_int32

    def __call__(self, source_projected, target_projected, attention, layout):
        nodes, heads, channels = source_projected.shape
        shape = (nodes, layout.source.numel(), heads, channels, layout.max_degree)
        cache_key = (*shape, layout.source.data_ptr())
        if cache_key not in self.cache:
            size = int(
                self.library.aclnnCsrGatv2DynamicAttentionAggregateFusedGetWorkspaceSize(
                    *shape
                )
            )
            self.cache[cache_key] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty_like(source_projected),
                layout.source.to(torch.int32),
            )
        workspace, output, source = self.cache[cache_key]
        result = self.library.aclnnCsrGatv2DynamicAttentionAggregateFused(
            layout.row_ptr.data_ptr(),
            source.data_ptr(),
            source_projected.data_ptr(),
            target_projected.data_ptr(),
            attention.data_ptr(),
            output.data_ptr(),
            *shape,
            0.2,
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"custom GATv2 operator returned {result}")
        return output


class Layer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, heads, concat):
        super().__init__()
        self.heads, self.channels, self.concat = heads, output_channels, concat
        self.lin_l = torch.nn.Linear(input_channels, heads * output_channels)
        self.lin_r = torch.nn.Linear(input_channels, heads * output_channels)
        self.attention = torch.nn.Parameter(torch.empty(heads, output_channels))
        output_size = heads * output_channels if concat else output_channels
        self.bias = torch.nn.Parameter(torch.zeros(output_size))

    def forward(self, features, layout, aggregate):
        left = self.lin_l(features).view(-1, self.heads, self.channels)
        right = self.lin_r(features).view(-1, self.heads, self.channels)
        output = aggregate(
            left.contiguous(), right.contiguous(), self.attention.contiguous(), layout
        )
        output = output.reshape(output.size(0), -1) if self.concat else output.mean(1)
        return output + self.bias


class Model(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer1 = Layer(input_channels, 8, 8, True)
        self.layer2 = Layer(64, output_channels, 1, False)

    def forward(self, features, layout, aggregate):
        features = F.elu(self.layer1(features, layout, aggregate))
        return self.layer2(features, layout, aggregate)


def load_weights(model, state):
    for target, prefix in ((model.layer1, "conv1"), (model.layer2, "conv2")):
        target.lin_l.weight.data.copy_(state[f"{prefix}.lin_l.weight"])
        target.lin_l.bias.data.copy_(state[f"{prefix}.lin_l.bias"])
        target.lin_r.weight.data.copy_(state[f"{prefix}.lin_r.weight"])
        target.lin_r.bias.data.copy_(state[f"{prefix}.lin_r.bias"])
        target.attention.data.copy_(state[f"{prefix}.att"].squeeze(0))
        target.bias.data.copy_(state[f"{prefix}.bias"])


def _official_model(first, second, features, edge_index):
    return second(F.elu(first(features, edge_index)), edge_index)


def _resident_model(model, features, layout):
    return model(features, layout, resident_aggregate)


def _custom_model(model, features, layout, operator):
    return model(features, layout, operator)


def _component(aggregate, left, right, attention, layout):
    return aggregate(left, right, attention, layout)


@dataclass
class BenchmarkContext:
    dataset: object
    data: object
    base_layout: Layout
    payload: dict
    model: Model
    operator: Operator
    official1: torch.nn.Module
    official2: torch.nn.Module
    device: torch.device


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


def _prepare(args) -> BenchmarkContext:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Model(data.num_features, dataset.num_classes)
    load_weights(model, payload["state_dict"])
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    model = model.to(device).eval()
    official1 = GATv2Conv(data.num_features, 8, heads=8, add_self_loops=False).to(
        device
    )
    official2 = GATv2Conv(
        64, dataset.num_classes, heads=1, concat=False, add_self_loops=False
    ).to(device)
    official1.load_state_dict(COMMON.prefixed_state(payload["state_dict"], "conv1."))
    official2.load_state_dict(COMMON.prefixed_state(payload["state_dict"], "conv2."))
    official1.eval()
    official2.eval()
    return BenchmarkContext(
        dataset=dataset,
        data=data,
        base_layout=build_layout(data.edge_index, data.num_nodes),
        payload=payload,
        model=model,
        operator=Operator(args.build, device),
        official1=official1,
        official2=official2,
        device=device,
    )


def _benchmark_case(context: BenchmarkContext, copies: int, warmup: int, repeat: int):
    features, layout = repeat_layout(context.data.x, context.base_layout, copies)
    features, layout = features.to(context.device), layout.to(context.device)
    edge_index = torch.stack((layout.source, layout.target))
    left = context.model.layer1.lin_l(features).view(-1, 8, 8).contiguous()
    right = context.model.layer1.lin_r(features).view(-1, 8, 8).contiguous()
    attention = context.model.layer1.attention
    calls = COMMON.AttentionBenchmarkCalls(
        official_model=partial(
            _official_model, context.official1, context.official2, features, edge_index
        ),
        resident_model=partial(_resident_model, context.model, features, layout),
        custom_model=partial(
            _custom_model, context.model, features, layout, context.operator
        ),
        official_component=partial(context.official1, features, edge_index),
        resident_component=partial(
            _component, resident_aggregate, left, right, attention, layout
        ),
        custom_component=partial(
            _component, context.operator, left, right, attention, layout
        ),
    )
    measurements = COMMON.run_attention_benchmark(calls, warmup, repeat)
    return COMMON.comparison_result(
        copies, features.size(0), layout.source.numel(), measurements
    )


def _write_report(args, context: BenchmarkContext, results: list[dict]) -> None:
    report = {
        "operator": "CsrGatv2DynamicAttentionAggregateFused",
        "model": "two-layer maintained PyG GATv2Conv on Cora",
        "checkpoint_sha256": COMMON.sha256(args.checkpoint),
        "checkpoint_test_accuracy": context.payload["test_accuracy"],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("%s", json.dumps(results, indent=2))


def main() -> None:
    COMMON.configure_logging()
    args = _arguments()
    context = _prepare(args)
    results = []
    for copies in args.copies:
        results.append(_benchmark_case(context, copies, args.warmup, args.repeat))
    _write_report(args, context, results)


if __name__ == "__main__":
    main()
