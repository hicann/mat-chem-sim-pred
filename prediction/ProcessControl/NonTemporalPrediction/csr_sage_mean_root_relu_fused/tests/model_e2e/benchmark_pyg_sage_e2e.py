#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark the fused operator in the maintained PyG SAGEConv model path."""

from __future__ import annotations

import collections
import ctypes
import json
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

try:
    from prediction.ProcessControl.NonTemporalPrediction.benchmark_common import (
        sage_helpers,
    )
except ModuleNotFoundError:
    from benchmark_common import sage_helpers

(
    hash_file,
    output_metrics,
    parse_model_arguments,
    relative_metrics,
    repeat_layout,
    setup_runtime,
    timed,
) = tuple(sage_helpers().values())

LOGGER = logging.getLogger(__name__)


@dataclass
class CsrLayout:
    source: torch.Tensor
    source_i32: torch.Tensor
    target: torch.Tensor
    row_ptr: torch.Tensor

    def to(self, device):
        return CsrLayout(*(value.to(device) for value in self.__dict__.values()))


@dataclass
class SageBenchmarkCase:
    model: object
    features: torch.Tensor
    labels: torch.Tensor
    test_mask: torch.Tensor
    layout: CsrLayout
    custom: object
    warmup: int
    repeat: int


@dataclass
class SageTimingValues:
    resident_stage_ms: float
    custom_stage_ms: float
    official_ms: float
    transformed_ms: float
    custom_ms: float
    resident_stage_output: torch.Tensor
    custom_stage_output: torch.Tensor
    official_output: torch.Tensor
    transformed_output: torch.Tensor
    custom_output: torch.Tensor


def build_csr(edge_index, nodes):
    source, target = edge_index[0], edge_index[1]
    order = torch.argsort(target * nodes + source)
    source, target = source[order], target[order]
    counts = torch.bincount(target, minlength=nodes)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return CsrLayout(source, source.to(torch.int32), target, row_ptr.to(torch.int32))


def edge_index_from_layout(layout):
    return torch.stack((layout.source, layout.target), dim=0)


def neighbor_sum(features, layout):
    output = torch.zeros_like(features)
    output.index_add_(0, layout.target, features[layout.source])
    return output


def sage_mean_root(neighbor_features, root_features, layout):
    output = neighbor_sum(neighbor_features, layout)
    counts = (layout.row_ptr[1:] - layout.row_ptr[:-1]).to(output.dtype).clamp_min_(1.0)
    return output / counts[:, None] + root_features


class PygSageClassifier(torch.nn.Module):
    def __init__(self, inputs, hidden, classes):
        super().__init__()
        self.conv = SAGEConv(
            inputs, hidden, aggr="mean", normalize=False, root_weight=True, bias=True
        )
        self.output = torch.nn.Linear(hidden, classes)

    def neighbor_projection(self, features):
        return F.linear(features, self.conv.lin_l.weight, None)

    def root_projection(self, features):
        output = self.conv.lin_r(features)
        if self.conv.lin_l.bias is not None:
            output = output + self.conv.lin_l.bias
        return output

    def forward(self, features, edge_index):
        return self.output(F.relu(self.conv(features, edge_index)))


def accuracy(logits, labels, mask):
    return float((logits[mask].argmax(-1) == labels[mask]).float().mean())


def train_or_load(model, data, checkpoint, epochs):
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state"])
        return payload
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5.0e-4)
    best_validation = -1.0
    best_state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = accuracy(model(data.x, data.edge_index), data.y, data.val_mask)
        if validation > best_validation:
            best_validation = validation
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_accuracy = accuracy(model(data.x, data.edge_index), data.y, data.test_mask)
    payload = {
        "state": best_state,
        "metrics": {
            "validation_accuracy": best_validation,
            "test_accuracy": test_accuracy,
        },
    }
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint)
    return payload


class CustomOperator:
    def __init__(self, build, device):
        for path in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(
            str(build / "libcsr_sage_mean_root_relu_fused_host.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.get_workspace_size = (
            self.library.aclnnCsrSageMeanRootReluFusedGetWorkspaceSize
        )
        self.get_workspace_size.argtypes = [ctypes.c_int64] * 3
        self.get_workspace_size.restype = ctypes.c_uint64
        self.operator = self.library.aclnnCsrSageMeanRootReluFused
        self.operator.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 3
        self.operator.argtypes += [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.cache = {}
        self.keepalive = collections.deque(maxlen=256)

    def __call__(self, neighbor_features, root_features, layout):
        dimensions = (
            neighbor_features.size(0),
            layout.source.numel(),
            neighbor_features.size(1),
        )
        if dimensions not in self.cache:
            workspace_size = int(self.get_workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty((workspace_size,), dtype=torch.uint8, device=self.device),
                torch.empty_like(neighbor_features),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            ctypes.c_void_p(layout.row_ptr.data_ptr()),
            ctypes.c_void_p(layout.source_i32.data_ptr()),
            ctypes.c_void_p(neighbor_features.data_ptr()),
            ctypes.c_void_p(root_features.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"aclnnCsrSageMeanRootReluFused returned {result}")
        self.keepalive.append(
            (neighbor_features, root_features, layout, workspace, output)
        )
        return output


def _case_result(case, values):
    strongest_e2e_ms = min(values.official_ms, values.transformed_ms)
    expected = (
        values.official_output
        if values.official_ms <= values.transformed_ms
        else values.transformed_output
    )
    result = {
        "copies": case.features.size(0) // 2708,
        "nodes": case.features.size(0),
        "edges": case.layout.source.numel(),
        "resident_stage_ms": values.resident_stage_ms,
        "custom_stage_ms": values.custom_stage_ms,
        "official_pyg_e2e_ms": values.official_ms,
        "transformed_resident_e2e_ms": values.transformed_ms,
        "strongest_e2e_ms": strongest_e2e_ms,
        "custom_e2e_ms": values.custom_ms,
        "stage_hotspot_pct": values.resident_stage_ms / values.transformed_ms * 100.0,
        "prediction_agreement": float(
            (expected.argmax(-1) == values.custom_output.argmax(-1))
            .float()
            .mean()
            .cpu()
        ),
        "baseline_test_accuracy": accuracy(expected, case.labels, case.test_mask),
        "custom_test_accuracy": accuracy(
            values.custom_output, case.labels, case.test_mask
        ),
    }
    result.update(
        relative_metrics("stage", values.resident_stage_ms, values.custom_stage_ms)
    )
    result.update(relative_metrics("e2e", strongest_e2e_ms, values.custom_ms))
    result.update(
        output_metrics(
            "stage", values.resident_stage_output, values.custom_stage_output
        )
    )
    result.update(
        output_metrics(
            "official_to_transformed", values.official_output, values.transformed_output
        )
    )
    result.update(output_metrics("model", expected, values.custom_output))
    return result


def benchmark_case(case):
    model = case.model
    features = case.features
    layout = case.layout
    custom = case.custom
    warmup = case.warmup
    repeat = case.repeat
    edge_index = edge_index_from_layout(layout)
    with torch.no_grad():
        neighbor_projected = model.neighbor_projection(features)
        root_projected = model.root_projection(features)

        def resident_stage():
            return F.relu(sage_mean_root(neighbor_projected, root_projected, layout))

        def custom_stage():
            return custom(neighbor_projected, root_projected, layout)

        resident_stage_ms, resident_stage_output = timed(resident_stage, warmup, repeat)
        custom_stage_ms, custom_stage_output = timed(custom_stage, warmup, repeat)

        def official_e2e():
            return model(features, edge_index)

        def transformed_e2e():
            neighbor = model.neighbor_projection(features)
            root = model.root_projection(features)
            return model.output(F.relu(sage_mean_root(neighbor, root, layout)))

        def custom_e2e():
            neighbor = model.neighbor_projection(features)
            root = model.root_projection(features)
            return model.output(custom(neighbor, root, layout))

        official_ms, official_output = timed(official_e2e, warmup, repeat)
        transformed_ms, transformed_output = timed(transformed_e2e, warmup, repeat)
        custom_ms, custom_output = timed(custom_e2e, warmup, repeat)

    return _case_result(
        case,
        SageTimingValues(
            resident_stage_ms,
            custom_stage_ms,
            official_ms,
            transformed_ms,
            custom_ms,
            resident_stage_output,
            custom_stage_output,
            official_output,
            transformed_output,
            custom_output,
        ),
    )


def prepare_inputs(arguments):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    torch.manual_seed(20260801)
    dataset = Planetoid(
        str(arguments.dataset_root), "Cora", transform=NormalizeFeatures()
    )
    data = dataset[0]
    model = PygSageClassifier(data.num_features, 32, dataset.num_classes)
    checkpoint = train_or_load(model, data, arguments.checkpoint, arguments.epochs)
    layout = build_csr(data.edge_index, data.num_nodes)

    return data, model, checkpoint, layout


def benchmark_cases(arguments, data, model, layout):
    device, custom = setup_runtime(CustomOperator, arguments.build)
    model.to(device).eval()
    cases = []
    for copies in arguments.copies:
        repeated_layout = repeat_layout(layout, copies, data.num_nodes).to(device)
        cases.append(
            benchmark_case(
                SageBenchmarkCase(
                    model=model,
                    features=data.x.repeat(copies, 1).to(device),
                    labels=data.y.repeat(copies).to(device),
                    test_mask=data.test_mask.repeat(copies).to(device),
                    layout=repeated_layout,
                    custom=custom,
                    warmup=arguments.warmup,
                    repeat=arguments.repeat,
                )
            )
        )
    return cases


def build_result(arguments, checkpoint, cases):
    return {
        "model": "torch_geometric.nn.SAGEConv",
        "model_contract": "SAGEConv(1433,32,aggr=mean) -> ReLU -> Linear(32,7)",
        "dataset": "PyG Planetoid Cora: 2708 nodes, 10556 directed edges, 1433 features, 7 classes",
        "checkpoint": {
            "path": str(arguments.checkpoint),
            "sha256": hash_file(arguments.checkpoint),
            "metrics": checkpoint["metrics"],
        },
        "rewrite": (
            "neighbor Linear is applied before mean; its bias is added to the root "
            "projection, preserving empty-row behavior"
        ),
        "baseline_policy": (
            "faster of the official PyG SAGEConv call and the equivalent resident "
            "torch_npu index_add rewrite"
        ),
        "timing": "resident NPU, synchronized wall-clock median; H2D and D2H excluded",
        "results": cases,
    }


def main():
    arguments = parse_model_arguments()
    data, model, checkpoint, layout = prepare_inputs(arguments)
    cases = benchmark_cases(arguments, data, model, layout)
    result = build_result(arguments, checkpoint, cases)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("SAGE benchmark result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
