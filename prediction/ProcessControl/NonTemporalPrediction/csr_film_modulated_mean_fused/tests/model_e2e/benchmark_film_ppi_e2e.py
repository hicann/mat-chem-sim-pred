#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Formal PPI FiLM model E2E benchmark for fused modulated mean aggregation."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch_npu
from sklearn.metrics import f1_score
from torch.nn import BatchNorm1d
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
from torch_geometric.nn import FiLMConv

from film_benchmark_common import CustomOperator, build_layout, native_mean

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
log_json, sha256, timed = _common.log_json, _common.sha256, _common.timed


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [
                FiLMConv(in_channels, 320),
                FiLMConv(320, 320),
                FiLMConv(320, 320),
                FiLMConv(320, out_channels, act=None),
            ]
        )
        self.norms = torch.nn.ModuleList([BatchNorm1d(320) for _ in range(3)])

    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs[:-1], self.norms):
            x = norm(conv(x, edge_index))
        return self.convs[-1](x, edge_index)


@dataclass
class BenchmarkContext:
    model: Net
    data: object
    operator: CustomOperator
    warmup: int
    repeat: int


@dataclass
class Case:
    features: torch.Tensor
    labels: torch.Tensor
    edge_index: torch.Tensor
    layout: object


@dataclass
class Measurements:
    official: object
    resident: object
    custom: object
    resident_component: object
    custom_component: object


def repeat_graph(data, copies):
    nodes = data.num_nodes
    edge_index = torch.cat(
        [data.edge_index + index * nodes for index in range(copies)], dim=1
    )
    return data.x.repeat(copies, 1), data.y.repeat(copies, 1), edge_index


def layer_inputs(features, layer):
    channels = layer.out_channels
    projected = layer.lins[0](features).contiguous()
    beta, gamma = layer.films[0](features).split(channels, dim=-1)
    return projected, beta.contiguous(), gamma.contiguous()


def manual_layer(features, layer, layout, aggregate):
    channels = layer.out_channels
    beta_skip, gamma_skip = layer.film_skip(features).split(channels, dim=-1)
    skip = gamma_skip * layer.lin_skip(features) + beta_skip
    if layer.act is not None:
        skip = torch.relu(skip)
    projected, beta, gamma = layer_inputs(features, layer)
    return skip + aggregate(projected, beta, gamma, layout, layer.act is not None)


def manual_model(model, features, layout, aggregate):
    for layer, norm in zip(model.convs[:-1], model.norms):
        features = norm(manual_layer(features, layer, layout, aggregate))
    return manual_layer(features, model.convs[-1], layout, aggregate)


def _official(model, features, edge_index):
    return model(features, edge_index)


def _component(aggregate, inputs, layout):
    return aggregate(*inputs, layout, True)


def _prepare_case(context, copies):
    features, labels, edge_index = repeat_graph(context.data, copies)
    layout = build_layout(edge_index, features.size(0)).to(torch.device("npu:0"))
    return Case(features.to("npu:0"), labels, edge_index.to("npu:0"), layout)


def _measure(context, case):
    with torch.no_grad():
        inputs = layer_inputs(case.features, context.model.convs[0])
    values = (
        timed(
            partial(_official, context.model, case.features, case.edge_index),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(
                manual_model, context.model, case.features, case.layout, native_mean
            ),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(
                manual_model,
                context.model,
                case.features,
                case.layout,
                context.operator,
            ),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(_component, native_mean, inputs, case.layout),
            context.warmup,
            context.repeat,
        ),
        timed(
            partial(_component, context.operator, inputs, case.layout),
            context.warmup,
            context.repeat,
        ),
    )
    return Measurements(*values)


def _timing_fields(measured):
    strongest = min(measured.official.median_ms, measured.resident.median_ms)
    strongest_name = (
        "official_pyg"
        if measured.official.median_ms <= measured.resident.median_ms
        else "resident_static_degree"
    )
    component_reduction = (
        measured.resident_component.median_ms - measured.custom_component.median_ms
    )
    return {
        "official_pyg_ms": measured.official.median_ms,
        "resident_static_degree_ms": measured.resident.median_ms,
        "strongest_native": strongest_name,
        "strongest_native_ms": strongest,
        "custom_ms": measured.custom.median_ms,
        "e2e_speedup": strongest / measured.custom.median_ms,
        "e2e_reduction_percent": (strongest - measured.custom.median_ms)
        / strongest
        * 100.0,
        "resident_component_ms": measured.resident_component.median_ms,
        "custom_component_ms": measured.custom_component.median_ms,
        "component_speedup": measured.resident_component.median_ms
        / measured.custom_component.median_ms,
        "component_reduction_percent": component_reduction
        / measured.resident_component.median_ms
        * 100.0,
        "official_samples_ms": measured.official.samples_ms,
        "resident_samples_ms": measured.resident.samples_ms,
        "custom_samples_ms": measured.custom.samples_ms,
        "resident_component_samples_ms": measured.resident_component.samples_ms,
        "custom_component_samples_ms": measured.custom_component.samples_ms,
    }


def _accuracy_fields(measured):
    official_error = float(
        (measured.official.output - measured.custom.output).abs().max().cpu()
    )
    official_scale = float(measured.official.output.abs().max().cpu())
    return {
        "component_max_abs_error": float(
            (measured.resident_component.output - measured.custom_component.output)
            .abs()
            .max()
            .cpu()
        ),
        "resident_e2e_max_abs_error": float(
            (measured.resident.output - measured.custom.output).abs().max().cpu()
        ),
        "official_e2e_max_abs_error": official_error,
        "official_output_max_abs": official_scale,
        "official_e2e_max_error_over_output_max": official_error
        / max(official_scale, 1.0e-12),
        "official_binary_prediction_agreement": float(
            ((measured.official.output > 0) == (measured.custom.output > 0))
            .float()
            .mean()
            .cpu()
        ),
        "resident_binary_prediction_agreement": float(
            ((measured.resident.output > 0) == (measured.custom.output > 0))
            .float()
            .mean()
            .cpu()
        ),
    }


def run_case(context, trial_index, copies):
    case = _prepare_case(context, copies)
    measured = _measure(context, case)
    result = {
        "trial": trial_index,
        "copies": copies,
        "graphs": 2 * copies,
        "nodes": context.data.num_nodes * copies,
        "edges": context.data.num_edges * copies,
        "max_degree": case.layout.max_degree,
    }
    result.update(_timing_fields(measured))
    result.update(_accuracy_fields(measured))
    prediction = (measured.official.output > 0).to(torch.float32).cpu().numpy()
    return result, f1_score(case.labels.numpy(), prediction, average="micro")


def _rows_for_copy(trials, copies):
    rows = []
    for trial in trials:
        for row in trial:
            if row["copies"] == copies:
                rows.append(row)
    return rows


def _summarize_rows(rows):
    fields = (
        "official_pyg_ms",
        "resident_static_degree_ms",
        "strongest_native_ms",
        "custom_ms",
        "e2e_speedup",
        "e2e_reduction_percent",
        "resident_component_ms",
        "custom_component_ms",
        "component_speedup",
        "component_reduction_percent",
    )
    result = {}
    for field in fields:
        result[field] = float(np.median([row[field] for row in rows]))
    result["strongest_native"] = min(rows, key=lambda row: row["strongest_native_ms"])[
        "strongest_native"
    ]
    result["max_component_error"] = max(row["component_max_abs_error"] for row in rows)
    result["max_official_e2e_error"] = max(
        row["official_e2e_max_abs_error"] for row in rows
    )
    result["max_error_over_output_max"] = max(
        row["official_e2e_max_error_over_output_max"] for row in rows
    )
    result["min_binary_prediction_agreement"] = min(
        row["official_binary_prediction_agreement"] for row in rows
    )
    return result


def build_summary(trials, copies_values):
    return {
        str(copies): _summarize_rows(_rows_for_copy(trials, copies))
        for copies in copies_values
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2)
    return parser.parse_args()


def prepare(args):
    torch.manual_seed(20260802)
    dataset = PPI(args.dataset_root, split="train")
    data = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Net(dataset.num_features, dataset.num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    context = BenchmarkContext(
        model.eval().to(device),
        data,
        CustomOperator(args.operator_build_dir, device),
        args.warmup,
        args.repeat,
    )
    return dataset, checkpoint, context


def main():
    args = parse_args()
    dataset, checkpoint, context = prepare(args)
    trials = []
    base_f1 = None
    for trial_index in range(args.trials):
        trial = []
        for copies in args.copies:
            row, current_f1 = run_case(context, trial_index, copies)
            trial.append(row)
            if trial_index == 0 and copies == 1:
                base_f1 = current_f1
        trials.append(trial)
    result = {
        "operator": "CsrFilmModulatedMeanFused",
        "model": "maintained PyG examples/film.py topology",
        "dataset": "PPI train split, deterministic first two-graph batch",
        "base_statistics": {
            "graphs": 2,
            "nodes": context.data.num_nodes,
            "edges": context.data.num_edges,
            "features": dataset.num_features,
            "classes": dataset.num_classes,
        },
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epochs": checkpoint["epochs"],
        "checkpoint_validation_f1": checkpoint["validation_f1"],
        "checkpoint_test_f1": checkpoint["test_f1"],
        "base_batch_micro_f1": base_f1,
        "pyg_revision": checkpoint["pyg_revision"],
        "summary": build_summary(trials, args.copies),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    log_json(result, indent=2)


if __name__ == "__main__":
    main()
