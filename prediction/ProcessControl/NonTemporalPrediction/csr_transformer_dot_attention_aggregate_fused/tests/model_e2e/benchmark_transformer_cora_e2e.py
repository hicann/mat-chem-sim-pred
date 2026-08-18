#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark Cora TransformerConv with fused dot-attention aggregation."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


ATTENTION_MODULE_PATH = Path(__file__).resolve().parents[3]
if str(ATTENTION_MODULE_PATH) not in sys.path:
    sys.path.append(str(ATTENTION_MODULE_PATH))
COMMON = importlib.import_module("attention_model_e2e_common")
LOGGER = logging.getLogger("benchmark_transformer_cora")
Layout = COMMON.AttentionLayout


def build_layout(edge_index, nodes):
    return COMMON.build_layout(edge_index, nodes)


def repeat_layout(features, layout, copies):
    return COMMON.repeat_layout(features, layout, copies)


def resident_aggregate(query, key, value, layout):
    logits = (query[layout.target] * key[layout.source]).sum(-1)
    logits = logits / math.sqrt(query.size(-1))
    padded = logits[layout.edge_ids].masked_fill(~layout.mask[..., None], -torch.inf)
    probabilities = torch.softmax(padded, dim=1)
    probabilities = probabilities.masked_fill(~layout.mask[..., None], 0.0)
    gathered_values = value[layout.source][layout.edge_ids]
    return (probabilities[..., None] * gathered_values).sum(1)


class Operator:
    def __init__(self, build, device):
        self.device = device
        self.cache = {}
        self.library = COMMON.load_acl_library(
            build, "libcsr_transformer_dot_attention_aggregate_fused_host.so"
        )
        self._configure_api()

    def __call__(self, query, key, value, layout):
        nodes, heads, channels = query.shape
        shape = (nodes, layout.source.numel(), heads, channels, layout.max_degree)
        workspace, output, source = self._buffers(shape, query, layout)
        status = self.library.aclnnCsrTransformerDotAttentionAggregateFused(
            layout.row_ptr.data_ptr(),
            source.data_ptr(),
            query.data_ptr(),
            key.data_ptr(),
            value.data_ptr(),
            output.data_ptr(),
            *shape,
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if status != 0:
            raise RuntimeError(f"custom TransformerConv operator returned {status}")
        return output

    def _configure_api(self):
        workspace = (
            self.library.aclnnCsrTransformerDotAttentionAggregateFusedGetWorkspaceSize
        )
        workspace.argtypes = [ctypes.c_int64] * 5
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrTransformerDotAttentionAggregateFused
        operation.argtypes = [ctypes.c_void_p] * 6
        operation.argtypes += [ctypes.c_int64] * 5
        operation.argtypes += [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        operation.restype = ctypes.c_int32

    def _buffers(self, shape, query, layout):
        cache_key = (*shape, query.data_ptr(), layout.source.data_ptr())
        if cache_key not in self.cache:
            workspace_size = int(
                self.library.aclnnCsrTransformerDotAttentionAggregateFusedGetWorkspaceSize(
                    *shape
                )
            )
            workspace = torch.empty(
                workspace_size, dtype=torch.uint8, device=self.device
            )
            self.cache[cache_key] = (
                workspace,
                torch.empty_like(query),
                layout.source.to(torch.int32),
            )
        return self.cache[cache_key]


class Layer(torch.nn.Module):
    def __init__(self, input_channels, output_channels, heads, concat):
        super().__init__()
        self.heads, self.channels, self.concat = heads, output_channels, concat
        total = heads * output_channels
        self.lin_query = torch.nn.Linear(input_channels, total)
        self.lin_key = torch.nn.Linear(input_channels, total)
        self.lin_value = torch.nn.Linear(input_channels, total)
        output_size = total if concat else output_channels
        self.lin_skip = torch.nn.Linear(input_channels, output_size)

    def forward(self, features, layout, aggregate):
        query = self.lin_query(features).view(-1, self.heads, self.channels)
        key = self.lin_key(features).view(-1, self.heads, self.channels)
        value = self.lin_value(features).view(-1, self.heads, self.channels)
        output = aggregate(
            query.contiguous(), key.contiguous(), value.contiguous(), layout
        )
        output = output.reshape(output.size(0), -1) if self.concat else output.mean(1)
        return output + self.lin_skip(features)


class Model(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layer1 = Layer(input_channels, 8, 8, True)
        self.layer2 = Layer(64, output_channels, 1, False)

    def forward(self, features, layout, aggregate):
        hidden = F.elu(self.layer1(features, layout, aggregate))
        return self.layer2(hidden, layout, aggregate)


def load_weights(model, state):
    names = ("lin_query", "lin_key", "lin_value", "lin_skip")
    layers = ((model.layer1, "conv1"), (model.layer2, "conv2"))
    for target, prefix in layers:
        for name in names:
            target_layer = getattr(target, name)
            target_layer.weight.data.copy_(state[f"{prefix}.{name}.weight"])
            target_layer.bias.data.copy_(state[f"{prefix}.{name}.bias"])


@dataclass
class OfficialStack:
    first: torch.nn.Module
    second: torch.nn.Module
    features: torch.Tensor
    edge_index: torch.Tensor

    def __call__(self):
        hidden = F.elu(self.first(self.features, self.edge_index))
        return self.second(hidden, self.edge_index)


@dataclass
class OfficialComponent:
    layer: torch.nn.Module
    features: torch.Tensor
    edge_index: torch.Tensor

    def __call__(self):
        return self.layer(self.features, self.edge_index)


@dataclass
class MaintainedStack:
    model: Model
    features: torch.Tensor
    layout: Layout
    aggregate: object

    def __call__(self):
        return self.model(self.features, self.layout, self.aggregate)


@dataclass
class DotAttentionCall:
    aggregate: object
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    layout: Layout

    def __call__(self):
        return self.aggregate(self.query, self.key, self.value, self.layout)


def _parse_cli():
    parser = argparse.ArgumentParser()
    required_paths = ("dataset-root", "checkpoint", "build", "output")
    for name in required_paths:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    return parser.parse_args()


class BenchmarkSession:
    def __init__(self, args):
        from torch_geometric.datasets import Planetoid
        from torch_geometric.nn import TransformerConv
        from torch_geometric.transforms import NormalizeFeatures

        self.args = args
        self.dataset = Planetoid(
            str(args.dataset_root), "Cora", transform=NormalizeFeatures()
        )
        self.data = self.dataset[0]
        self.payload = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        self.device = torch.device("npu:0")
        torch_npu.npu.set_device(self.device)
        self.model = Model(self.data.num_features, self.dataset.num_classes)
        load_weights(self.model, self.payload["state_dict"])
        self.model = self.model.to(self.device).eval()
        self.operator = Operator(args.build, self.device)
        self.official = self._official_layers(TransformerConv)
        self.base_layout = build_layout(self.data.edge_index, self.data.num_nodes)

    def benchmark(self, copies: int) -> dict:
        features, layout = repeat_layout(self.data.x, self.base_layout, copies)
        features, layout = features.to(self.device), layout.to(self.device)
        edge_index = torch.stack((layout.source, layout.target))
        query = self.model.layer1.lin_query(features).view(-1, 8, 8).contiguous()
        key = self.model.layer1.lin_key(features).view(-1, 8, 8).contiguous()
        value = self.model.layer1.lin_value(features).view(-1, 8, 8).contiguous()
        calls = COMMON.AttentionBenchmarkCalls(
            OfficialStack(*self.official, features, edge_index),
            MaintainedStack(self.model, features, layout, resident_aggregate),
            MaintainedStack(self.model, features, layout, self.operator),
            OfficialComponent(self.official[0], features, edge_index),
            DotAttentionCall(resident_aggregate, query, key, value, layout),
            DotAttentionCall(self.operator, query, key, value, layout),
        )
        measurements = COMMON.run_attention_benchmark(
            calls, self.args.warmup, self.args.repeat
        )
        return COMMON.comparison_result(
            copies, features.size(0), layout.source.numel(), measurements
        )

    def report(self, results: list[dict]) -> None:
        content = {
            "operator": "CsrTransformerDotAttentionAggregateFused",
            "model": "two-layer maintained PyG TransformerConv on Cora",
            "checkpoint_sha256": COMMON.sha256(self.args.checkpoint),
            "checkpoint_test_accuracy": self.payload["test_accuracy"],
            "results": results,
        }
        self.args.output.parent.mkdir(parents=True, exist_ok=True)
        self.args.output.write_text(json.dumps(content, indent=2), encoding="utf-8")
        LOGGER.info("%s", json.dumps(results, indent=2))

    def _official_layers(self, transformer_conv):
        first = transformer_conv(self.data.num_features, 8, heads=8).to(self.device)
        second = transformer_conv(
            64, self.dataset.num_classes, heads=1, concat=False
        ).to(self.device)
        state = self.payload["state_dict"]
        first.load_state_dict(COMMON.prefixed_state(state, "conv1."))
        second.load_state_dict(COMMON.prefixed_state(state, "conv2."))
        return first.eval(), second.eval()


def main() -> None:
    COMMON.configure_logging()
    session = BenchmarkSession(_parse_cli())
    results = []
    for copies in session.args.copies:
        results.append(session.benchmark(copies))
    session.report(results)


if __name__ == "__main__":
    main()
