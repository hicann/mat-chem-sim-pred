#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark the full propagation operator in maintained PyG APPNP."""

from __future__ import annotations

import collections
import ctypes
import json
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm

try:
    from prediction.ProcessControl.NonTemporalPrediction.benchmark_common import (
        appnp_helpers,
    )
except ModuleNotFoundError:
    from benchmark_common import appnp_helpers

(
    hash_file,
    output_metrics,
    parse_model_arguments,
    relative_metrics,
    repeat_layout,
    setup_runtime,
    timed,
) = tuple(appnp_helpers().values())

LOGGER = logging.getLogger(__name__)


@dataclass
class CsrLayout:
    source: torch.Tensor
    source_i32: torch.Tensor
    target: torch.Tensor
    row_ptr: torch.Tensor
    edge_weight: torch.Tensor

    def to(self, device):
        return CsrLayout(*(value.to(device) for value in self.__dict__.values()))


@dataclass
class AppnpBenchmarkCase:
    model: object
    features: torch.Tensor
    labels: torch.Tensor
    test_mask: torch.Tensor
    raw_edge_index: torch.Tensor
    layout: CsrLayout
    custom: object
    warmup: int
    repeat: int


def normalized_csr(edge_index, nodes):
    normalized_index, edge_weight = gcn_norm(
        edge_index,
        None,
        num_nodes=nodes,
        improved=False,
        add_self_loops=True,
        flow="source_to_target",
        dtype=torch.float32,
    )
    source, target = normalized_index[0], normalized_index[1]
    order = torch.argsort(target * nodes + source)
    source, target, edge_weight = source[order], target[order], edge_weight[order]
    counts = torch.bincount(target, minlength=nodes)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return CsrLayout(
        source, source.to(torch.int32), target, row_ptr.to(torch.int32), edge_weight
    )


def repeat_edge_index(edge_index, copies, nodes):
    if copies == 1:
        return edge_index
    return torch.cat([edge_index + index * nodes for index in range(copies)], dim=1)


def resident_propagate(initial, layout, iterations, alpha):
    current = initial
    for _ in range(iterations):
        output = torch.zeros_like(current)
        output.index_add_(
            0,
            layout.target,
            current[layout.source] * layout.edge_weight[:, None],
        )
        current = (1.0 - alpha) * output + alpha * initial
    return current


class PygAppnpClassifier(torch.nn.Module):
    def __init__(self, inputs, hidden, classes, iterations, alpha):
        super().__init__()
        from torch_geometric.nn import APPNP

        self.hidden = torch.nn.Linear(inputs, hidden)
        self.output = torch.nn.Linear(hidden, classes)
        self.propagation = APPNP(
            K=iterations,
            alpha=alpha,
            dropout=0.0,
            cached=False,
            add_self_loops=True,
            normalize=True,
        )

    def project(self, features):
        return self.output(F.relu(self.hidden(features)))

    def forward(self, features, edge_index):
        return self.propagation(self.project(features), edge_index)


def accuracy(logits, labels, mask):
    return float((logits[mask].argmax(-1) == labels[mask]).float().mean())


def _load_checkpoint(model, checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state"])
    return payload["state"]


def _load_bootstrap(model, bootstrap):
    source = torch.load(bootstrap, map_location="cpu", weights_only=True)
    old = source["states"]["appnp"]
    state = {
        "hidden.weight": old["a.weight"],
        "hidden.bias": old["a.bias"],
        "output.weight": old["b.weight"],
        "output.bias": old["b.bias"],
    }
    model.load_state_dict(state)
    return state


def _train_model(model, data, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5.0e-4)
    best_validation = -1.0
    state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        F.cross_entropy(logits[data.train_mask], data.y[data.train_mask]).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = accuracy(model(data.x, data.edge_index), data.y, data.val_mask)
        if validation > best_validation:
            best_validation = validation
            state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    model.load_state_dict(state)
    return state


def train_or_load(model, data, checkpoint, bootstrap, epochs):
    if checkpoint.exists():
        state = _load_checkpoint(model, checkpoint)
    elif bootstrap is not None and bootstrap.exists():
        state = _load_bootstrap(model, bootstrap)
    else:
        state = _train_model(model, data, epochs)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        metrics = {
            "validation_accuracy": accuracy(logits, data.y, data.val_mask),
            "test_accuracy": accuracy(logits, data.y, data.test_mask),
        }
    payload = {"state": state, "metrics": metrics}
    if (
        not checkpoint.exists()
        or torch.load(checkpoint, map_location="cpu", weights_only=True).get("metrics")
        != metrics
    ):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint)
    return payload


class CustomOperator:
    def __init__(self, build, device):
        for path in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(
            str(build / "libcsr_appnp_propagate_fused_host.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.get_workspace_size = (
            self.library.aclnnCsrAppnpPropagateFusedGetWorkspaceSize
        )
        self.get_workspace_size.argtypes = [ctypes.c_int64] * 4
        self.get_workspace_size.restype = ctypes.c_uint64
        self.operator = self.library.aclnnCsrAppnpPropagateFused
        self.operator.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int64] * 4
        self.operator.argtypes += [
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.cache = {}
        self.keepalive = collections.deque(maxlen=256)

    def __call__(self, initial, layout, iterations, alpha):
        dimensions = (
            initial.size(0),
            layout.source.numel(),
            initial.size(1),
            iterations,
        )
        if dimensions not in self.cache:
            workspace_size = int(self.get_workspace_size(*dimensions))
            self.cache[dimensions] = (
                torch.empty((workspace_size,), dtype=torch.uint8, device=self.device),
                torch.empty_like(initial),
            )
        workspace, output = self.cache[dimensions]
        result = self.operator(
            ctypes.c_void_p(layout.row_ptr.data_ptr()),
            ctypes.c_void_p(layout.source_i32.data_ptr()),
            ctypes.c_void_p(layout.edge_weight.data_ptr()),
            ctypes.c_void_p(initial.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            alpha,
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"aclnnCsrAppnpPropagateFused returned {result}")
        self.keepalive.append((initial, layout, workspace, output))
        return output


def _stage_timings(case):
    model = case.model
    iterations = model.propagation.K
    alpha = model.propagation.alpha
    with torch.no_grad():
        initial = model.project(case.features)

        def official_stage():
            return model.propagation(initial, case.raw_edge_index)

        def resident_stage():
            return resident_propagate(initial, case.layout, iterations, alpha)

        def custom_stage():
            return case.custom(initial, case.layout, iterations, alpha)

        stage = (
            timed(official_stage, case.warmup, case.repeat),
            timed(resident_stage, case.warmup, case.repeat),
            timed(custom_stage, case.warmup, case.repeat),
        )

        def official_e2e():
            return model(case.features, case.raw_edge_index)

        def resident_e2e():
            return resident_propagate(
                model.project(case.features), case.layout, iterations, alpha
            )

        def custom_e2e():
            return case.custom(
                model.project(case.features), case.layout, iterations, alpha
            )

        e2e = (
            timed(official_e2e, case.warmup, case.repeat),
            timed(resident_e2e, case.warmup, case.repeat),
            timed(custom_e2e, case.warmup, case.repeat),
        )
        return stage, e2e


def _case_result(case, stage, e2e):
    (
        (official_stage_ms, official_stage_output),
        (resident_stage_ms, resident_stage_output),
        (custom_stage_ms, custom_stage_output),
    ) = stage
    (
        (official_e2e_ms, official_output),
        (resident_e2e_ms, resident_output),
        (custom_e2e_ms, custom_output),
    ) = e2e
    strongest_stage_ms = min(official_stage_ms, resident_stage_ms)
    strongest_e2e_ms = min(official_e2e_ms, resident_e2e_ms)
    expected = (
        official_output if official_e2e_ms <= resident_e2e_ms else resident_output
    )
    result = {
        "copies": case.features.size(0) // 2708,
        "nodes": case.features.size(0),
        "edges": case.layout.source.numel(),
        "official_pyg_stage_ms": official_stage_ms,
        "resident_stage_ms": resident_stage_ms,
        "strongest_stage_ms": strongest_stage_ms,
        "custom_stage_ms": custom_stage_ms,
        "official_pyg_e2e_ms": official_e2e_ms,
        "resident_e2e_ms": resident_e2e_ms,
        "strongest_e2e_ms": strongest_e2e_ms,
        "custom_e2e_ms": custom_e2e_ms,
        "stage_hotspot_pct": strongest_stage_ms / strongest_e2e_ms * 100.0,
        "prediction_agreement": float(
            (expected.argmax(-1) == custom_output.argmax(-1)).float().mean().cpu()
        ),
        "baseline_test_accuracy": accuracy(expected, case.labels, case.test_mask),
        "custom_test_accuracy": accuracy(custom_output, case.labels, case.test_mask),
    }
    result.update(relative_metrics("stage", strongest_stage_ms, custom_stage_ms))
    result.update(relative_metrics("e2e", strongest_e2e_ms, custom_e2e_ms))
    result.update(
        output_metrics(
            "official_to_resident", official_stage_output, resident_stage_output
        )
    )
    result.update(output_metrics("stage", resident_stage_output, custom_stage_output))
    result.update(output_metrics("model", expected, custom_output))
    return result


def benchmark_case(case):
    with torch.no_grad():
        stage, e2e = _stage_timings(case)
    return _case_result(case, stage, e2e)


def prepare_inputs(arguments):
    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures

    torch.manual_seed(20260801)
    dataset = Planetoid(
        str(arguments.dataset_root), "Cora", transform=NormalizeFeatures()
    )
    data = dataset[0]
    model = PygAppnpClassifier(data.num_features, 32, dataset.num_classes, 10, 0.1)
    checkpoint = train_or_load(
        model,
        data,
        arguments.checkpoint,
        arguments.bootstrap_checkpoint,
        arguments.epochs,
    )
    layout = normalized_csr(data.edge_index, data.num_nodes)
    model.eval()
    with torch.no_grad():
        cpu_initial = model.project(data.x)
        cpu_official = model.propagation(cpu_initial, data.edge_index)
        cpu_resident = resident_propagate(cpu_initial, layout, 10, 0.1)
        cpu_rewrite_max_error = float((cpu_official - cpu_resident).abs().max())

    return data, model, checkpoint, layout, cpu_rewrite_max_error


def benchmark_cases(arguments, data, model, layout):
    device, custom = setup_runtime(CustomOperator, arguments.build)
    model.to(device).eval()
    cases = []
    for copies in arguments.copies:
        repeated_layout = repeat_layout(layout, copies, data.num_nodes).to(device)
        repeated_edge_index = repeat_edge_index(
            data.edge_index, copies, data.num_nodes
        ).to(device)
        cases.append(
            benchmark_case(
                AppnpBenchmarkCase(
                    model=model,
                    features=data.x.repeat(copies, 1).to(device),
                    labels=data.y.repeat(copies).to(device),
                    test_mask=data.test_mask.repeat(copies).to(device),
                    raw_edge_index=repeated_edge_index,
                    layout=repeated_layout,
                    custom=custom,
                    warmup=arguments.warmup,
                    repeat=arguments.repeat,
                )
            )
        )
    return cases


def build_result(arguments, checkpoint, cpu_rewrite_max_error, cases):
    return {
        "model": "torch_geometric.nn.APPNP",
        "model_contract": "Linear(1433,32) -> ReLU -> Linear(32,7) -> APPNP(K=10,alpha=0.1)",
        "dataset": (
            "PyG Planetoid Cora: 2708 nodes, 10556 directed input edges, "
            "13264 normalized edges, 1433 features, 7 classes"
        ),
        "checkpoint": {
            "path": str(arguments.checkpoint),
            "sha256": hash_file(arguments.checkpoint),
            "metrics": checkpoint["metrics"],
        },
        "cpu_rewrite_max_error": cpu_rewrite_max_error,
        "baseline_policy": "faster of maintained PyG APPNP and the equivalent resident torch_npu index_add propagation",
        "timing": "resident NPU, synchronized wall-clock median; H2D and D2H excluded",
        "results": cases,
    }


def main():
    arguments = parse_model_arguments(include_bootstrap=True)
    data, model, checkpoint, layout, cpu_rewrite_max_error = prepare_inputs(arguments)
    cases = benchmark_cases(arguments, data, model, layout)
    result = build_result(arguments, checkpoint, cpu_rewrite_max_error, cases)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("APPNP benchmark result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
