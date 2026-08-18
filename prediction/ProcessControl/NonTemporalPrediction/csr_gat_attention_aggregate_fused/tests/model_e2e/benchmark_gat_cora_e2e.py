#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Train and benchmark a two-layer GAT on Cora with the CSR attention op."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import logging
import statistics
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


SHARED_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SHARED_ROOT))
COMMON = importlib.import_module("attention_model_e2e_common")
LOGGER = logging.getLogger("benchmark_gat_cora")
GraphLayout = COMMON.AttentionLayout


def build_layout(
    edge_index: torch.Tensor, node_count: int, max_degree: int
) -> GraphLayout:
    """Create a deterministic target-sorted CSR graph with bounded fan-in."""
    return COMMON.build_layout(
        edge_index, node_count, self_loops=True, max_degree=max_degree
    )


def repeat_graph(
    features: torch.Tensor, labels: torch.Tensor, layout: GraphLayout, copies: int
):
    repeated_features, repeated_layout = COMMON.repeat_layout(features, layout, copies)
    return repeated_features, labels.repeat(copies), repeated_layout


def scatter_aggregate(logits: torch.Tensor, values: torch.Tensor, layout: GraphLayout):
    """Differentiable CPU path with segment-softmax semantics."""
    segment_count, head_count = layout.edge_ids.shape[0], logits.size(1)
    index = layout.target[:, None].expand(-1, head_count)
    maxima = torch.full(
        (segment_count, head_count),
        -torch.inf,
        dtype=logits.dtype,
        device=logits.device,
    )
    maxima.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
    exponentials = torch.exp(logits - maxima[layout.target])
    denominator = torch.zeros_like(maxima).scatter_add_(0, index, exponentials)
    weights = exponentials / denominator[layout.target].clamp_min(1.0e-12)
    output = torch.zeros(
        (segment_count, head_count, values.size(-1)),
        dtype=values.dtype,
        device=values.device,
    )
    return output.index_add_(0, layout.target, weights[..., None] * values)


def padded_aggregate(logits: torch.Tensor, values: torch.Tensor, layout: GraphLayout):
    """Resident-NPU baseline without per-segment Python launches."""
    padded_logits = logits[layout.edge_ids].masked_fill(
        ~layout.mask[..., None], -torch.inf
    )
    weights = torch.softmax(padded_logits, dim=1)
    weights = weights.masked_fill(~layout.mask[..., None], 0.0)
    return (weights[..., None] * values[layout.edge_ids]).sum(dim=1)


def node_attention_inputs(projected, attention_source, attention_target, layout):
    values = projected[layout.source]
    logits = F.leaky_relu(
        (values * attention_source).sum(-1)
        + (projected[layout.target] * attention_target).sum(-1),
        negative_slope=0.2,
    ).contiguous()
    return logits, values


def scatter_node_aggregate(projected, attention_source, attention_target, layout):
    logits, values = node_attention_inputs(
        projected, attention_source, attention_target, layout
    )
    return scatter_aggregate(logits, values, layout)


def padded_node_aggregate(projected, attention_source, attention_target, layout):
    logits, values = node_attention_inputs(
        projected, attention_source, attention_target, layout
    )
    return padded_aggregate(logits, values, layout)


class CsrGatAttentionOperator:
    def __init__(self, build_dir: Path, device: torch.device):
        self.library = COMMON.load_acl_library(
            build_dir, "libcsr_gat_attention_aggregate_fused_host.so"
        )
        self.device = device
        self.cache = {}
        workspace = self.library.aclnnCsrGatAttentionAggregateFusedGetWorkspaceSize
        workspace.argtypes = [ctypes.c_int64] * 5
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrGatAttentionAggregateFused
        operation.argtypes = [ctypes.c_void_p] * 6
        operation.argtypes += [ctypes.c_int64] * 5
        operation.argtypes += [
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        operation.restype = ctypes.c_int32

    def __call__(
        self,
        projected: torch.Tensor,
        attention_source: torch.Tensor,
        attention_target: torch.Tensor,
        layout: GraphLayout,
    ):
        node_count, head_count, channels = projected.shape
        if node_count != layout.row_ptr.numel() - 1:
            raise ValueError("projected nodes do not match the CSR row count")
        shape = (
            node_count,
            layout.source.numel(),
            head_count,
            channels,
            layout.max_degree,
        )
        cache_key = (*shape, layout.source.data_ptr())
        if cache_key not in self.cache:
            size = int(
                self.library.aclnnCsrGatAttentionAggregateFusedGetWorkspaceSize(*shape)
            )
            self.cache[cache_key] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty_like(projected),
                layout.source.to(torch.int32),
            )
        workspace, output, source_i32 = self.cache[cache_key]
        status = self.library.aclnnCsrGatAttentionAggregateFused(
            layout.row_ptr.data_ptr(),
            source_i32.data_ptr(),
            projected.data_ptr(),
            attention_source.data_ptr(),
            attention_target.data_ptr(),
            output.data_ptr(),
            *shape,
            0.2,
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if status != 0:
            raise RuntimeError(f"aclnnCsrGatAttentionAggregateFused returned {status}")
        return output


class GATLayer(torch.nn.Module):
    def __init__(
        self, input_channels: int, output_channels: int, heads: int, concat: bool
    ):
        super().__init__()
        self.output_channels = output_channels
        self.heads = heads
        self.concat = concat
        self.projection = torch.nn.Linear(
            input_channels, heads * output_channels, bias=False
        )
        self.attention_source = torch.nn.Parameter(torch.empty(heads, output_channels))
        self.attention_target = torch.nn.Parameter(torch.empty(heads, output_channels))
        bias_channels = heads * output_channels if concat else output_channels
        self.bias = torch.nn.Parameter(torch.zeros(bias_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.projection.weight)
        torch.nn.init.xavier_uniform_(self.attention_source)
        torch.nn.init.xavier_uniform_(self.attention_target)
        torch.nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor, layout: GraphLayout, aggregate):
        projected = self.projection(features).reshape(
            -1, self.heads, self.output_channels
        )
        output = aggregate(
            projected.contiguous(),
            self.attention_source.contiguous(),
            self.attention_target.contiguous(),
            layout,
        )
        output = output.reshape(output.size(0), -1) if self.concat else output.mean(1)
        return output + self.bias


class GATClassifier(torch.nn.Module):
    def __init__(self, input_channels: int, class_count: int):
        super().__init__()
        self.layer1 = GATLayer(input_channels, 8, heads=8, concat=True)
        self.layer2 = GATLayer(64, class_count, heads=1, concat=False)

    def forward(
        self, features: torch.Tensor, layout: GraphLayout, aggregate, training=False
    ):
        if training:
            features = F.dropout(features, p=0.6, training=True)
        features = F.elu(self.layer1(features, layout, aggregate))
        if training:
            features = F.dropout(features, p=0.6, training=True)
        return self.layer2(features, layout, aggregate)


def build_pyg_classifier(gat_conv, trained: GATClassifier):
    """Create an official PyG GATConv path with the trained weights."""

    class PyGClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = gat_conv(
                trained.layer1.projection.in_features,
                8,
                heads=8,
                concat=True,
                dropout=0.0,
                add_self_loops=False,
            )
            self.layer2 = gat_conv(
                64,
                trained.layer2.output_channels,
                heads=1,
                concat=False,
                dropout=0.0,
                add_self_loops=False,
            )

        def forward(self, features, edge_index):
            return self.layer2(F.elu(self.layer1(features, edge_index)), edge_index)

    model = PyGClassifier()
    with torch.no_grad():
        model.layer1.lin.weight.copy_(trained.layer1.projection.weight)
        model.layer1.att_src.copy_(trained.layer1.attention_source[None])
        model.layer1.att_dst.copy_(trained.layer1.attention_target[None])
        model.layer1.bias.copy_(trained.layer1.bias)
        model.layer2.lin.weight.copy_(trained.layer2.projection.weight)
        model.layer2.att_src.copy_(trained.layer2.attention_source[None])
        model.layer2.att_dst.copy_(trained.layer2.attention_target[None])
        model.layer2.bias.copy_(trained.layer2.bias)
    return model


def _model_call(model, features, layout, aggregate):
    return model(features, layout, aggregate)


def _pyg_call(model, features, edge_index):
    return model(features, edge_index)


def _component_call(aggregate, projected, source_attention, target_attention, layout):
    return aggregate(projected, source_attention, target_attention, layout)


@dataclass
class BenchmarkContext:
    args: object
    dataset: object
    data: object
    layout: GraphLayout
    payload: dict
    model: GATClassifier
    pyg_model: torch.nn.Module
    custom_operator: CsrGatAttentionOperator
    device: torch.device
    test_accuracy: float


def _arguments():
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-build-dir", type=Path, default=root / "build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--max-degree", type=int, default=169)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2)
    return parser.parse_args()


def _prepare(args) -> BenchmarkContext:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GATConv
    from torch_geometric.transforms import NormalizeFeatures

    torch.manual_seed(20260731)
    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    layout = build_layout(data.edge_index, data.num_nodes, args.max_degree)
    if layout.dropped_edges:
        raise RuntimeError(f"max degree truncates {layout.dropped_edges} Cora edges")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GATClassifier(data.num_node_features, dataset.num_classes)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = model(data.x, layout, scatter_node_aggregate).argmax(-1)
        accuracy = float(
            (prediction[data.test_mask] == data.y[data.test_mask]).float().mean()
        )
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    pyg_model = build_pyg_classifier(GATConv, model).to(device).eval()
    model = model.to(device).eval()
    return BenchmarkContext(
        args,
        dataset,
        data,
        layout,
        payload,
        model,
        pyg_model,
        CsrGatAttentionOperator(args.operator_build_dir, device),
        device,
        accuracy,
    )


def _measure_calls(calls: dict, reverse: bool, warmup: int, repeat: int):
    model_order = ["pyg", "padded", "custom"]
    if reverse:
        model_order.reverse()
    measured = {}
    for name in model_order:
        measured[name] = COMMON.timed(calls[name], warmup, repeat)
    for name in ("scatter_component", "padded_component", "custom_component"):
        measured[name] = COMMON.timed(calls[name], warmup, repeat)
    return measured


def _case_calls(context: BenchmarkContext, copies: int):
    features, labels, layout = repeat_graph(
        context.data.x, context.data.y, context.layout, copies
    )
    features, labels = features.to(context.device), labels.to(context.device)
    layout = layout.to(context.device)
    edge_index = torch.stack((layout.source, layout.target))
    with torch.no_grad():
        projected = context.model.layer1.projection(features).reshape(-1, 8, 8)
        projected = projected.contiguous()
    source_attention = context.model.layer1.attention_source
    target_attention = context.model.layer1.attention_target
    calls = {
        "pyg": partial(_pyg_call, context.pyg_model, features, edge_index),
        "padded": partial(
            _model_call, context.model, features, layout, padded_node_aggregate
        ),
        "custom": partial(
            _model_call, context.model, features, layout, context.custom_operator
        ),
        "scatter_component": partial(
            _component_call,
            scatter_node_aggregate,
            projected,
            source_attention,
            target_attention,
            layout,
        ),
        "padded_component": partial(
            _component_call,
            padded_node_aggregate,
            projected,
            source_attention,
            target_attention,
            layout,
        ),
        "custom_component": partial(
            _component_call,
            context.custom_operator,
            projected,
            source_attention,
            target_attention,
            layout,
        ),
    }
    return features, labels, layout, calls


def _case_result(context, copies: int, trial_index: int) -> dict:
    features, _labels, layout, calls = _case_calls(context, copies)
    measured = _measure_calls(
        calls,
        reverse=trial_index % 2 == 1,
        warmup=context.args.warmup,
        repeat=context.args.repeat,
    )
    pyg_ms, pyg_output, pyg_samples = measured["pyg"]
    padded_ms, padded_output, padded_samples = measured["padded"]
    custom_ms, actual, custom_samples = measured["custom"]
    scatter_ms, scatter_output, _ = measured["scatter_component"]
    padded_component_ms, padded_component_output, _ = measured["padded_component"]
    custom_component_ms, actual_component, _ = measured["custom_component"]
    strongest_component = min(scatter_ms, padded_component_ms)
    component_error = torch.maximum(
        (scatter_output - actual_component).abs().max(),
        (padded_component_output - actual_component).abs().max(),
    )
    return {
        "copies": copies,
        "nodes": int(features.size(0)),
        "edges": int(layout.source.numel()),
        "features": int(features.size(1)),
        "classes": context.dataset.num_classes,
        "dropped_edges": layout.dropped_edges,
        "scatter_component_ms": scatter_ms,
        "padded_component_ms": padded_component_ms,
        "strongest_component_ms": strongest_component,
        "custom_component_ms": custom_component_ms,
        "component_speedup": strongest_component / custom_component_ms,
        "component_max_error": float(component_error.cpu()),
        "pyg_gatconv_model_e2e_ms": pyg_ms,
        "resident_padded_model_e2e_ms": padded_ms,
        "custom_model_e2e_ms": custom_ms,
        "max_logit_error": float((pyg_output - actual).abs().max().cpu()),
        "pyg_vs_padded_max_logit_error": float(
            (pyg_output - padded_output).abs().max().cpu()
        ),
        "prediction_agreement": float(
            (pyg_output.argmax(-1) == actual.argmax(-1)).float().mean().cpu()
        ),
        "pyg_gatconv_samples_ms": pyg_samples,
        "resident_padded_samples_ms": padded_samples,
        "custom_samples_ms": custom_samples,
    }


def _run_trials(context: BenchmarkContext):
    trials = []
    for trial_index in range(context.args.trials):
        trial = []
        for copies in context.args.copies:
            trial.append(_case_result(context, copies, trial_index))
        trials.append(trial)
    return trials


def _base_summary(case: dict) -> dict:
    excluded = {
        "max_logit_error",
        "pyg_vs_padded_max_logit_error",
        "prediction_agreement",
        "component_speedup",
        "component_max_error",
    }
    summary = {}
    for key, value in case.items():
        if not key.endswith("_samples_ms") and key not in excluded:
            summary[key] = value
    return summary


def _summarize_case(cases: list[dict]) -> dict:
    latency_keys = (
        "pyg_gatconv_model_e2e_ms",
        "resident_padded_model_e2e_ms",
        "custom_model_e2e_ms",
        "scatter_component_ms",
        "padded_component_ms",
        "strongest_component_ms",
        "custom_component_ms",
    )
    summary = _base_summary(cases[0])
    for key in latency_keys:
        summary[key] = statistics.fmean(case[key] for case in cases)
    strongest = min(
        summary["pyg_gatconv_model_e2e_ms"],
        summary["resident_padded_model_e2e_ms"],
    )
    summary["strongest_framework_model_e2e_ms"] = strongest
    summary["model_e2e_speedup"] = strongest / summary["custom_model_e2e_ms"]
    summary["model_e2e_reduction_pct"] = (
        100.0 * (strongest - summary["custom_model_e2e_ms"]) / strongest
    )
    summary["component_speedup"] = (
        summary["strongest_component_ms"] / summary["custom_component_ms"]
    )
    for key in (
        "component_max_error",
        "max_logit_error",
        "pyg_vs_padded_max_logit_error",
    ):
        summary[key] = max(case[key] for case in cases)
    summary["prediction_agreement"] = min(
        case["prediction_agreement"] for case in cases
    )
    summary["trials"] = cases
    return summary


def _summarize(trials: list[list[dict]], case_count: int):
    results = []
    for case_index in range(case_count):
        cases = [trial[case_index] for trial in trials]
        results.append(_summarize_case(cases))
    return results


def _write_report(context: BenchmarkContext, results: list[dict]) -> None:
    args, data, layout = context.args, context.data, context.layout
    report = {
        "date": "2026-08-02",
        "device": torch_npu.npu.get_device_name(context.device),
        "framework": "two-layer GAT on PyG Cora",
        "baselines": [
            "official PyG GATConv",
            "resident vectorized padded-segment PyTorch NPU implementation",
        ],
        "baseline_policy": "use the faster median of the two framework paths",
        "dataset": "Cora",
        "dataset_nodes": data.num_nodes,
        "dataset_edges_original": data.num_edges,
        "dataset_edges_used": int(layout.source.numel()),
        "max_degree": args.max_degree,
        "dropped_edges": layout.dropped_edges,
        "checkpoint": args.checkpoint.name,
        "checkpoint_sha256": COMMON.sha256(args.checkpoint),
        "checkpoint_validation_accuracy": context.payload["validation_accuracy"],
        "checkpoint_test_accuracy": context.test_accuracy,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", args.output)


def main() -> None:
    COMMON.configure_logging()
    context = _prepare(_arguments())
    trials = _run_trials(context)
    results = _summarize(trials, len(context.args.copies))
    _write_report(context, results)


if __name__ == "__main__":
    main()
