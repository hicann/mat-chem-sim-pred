#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Compare the complete AGNN Cora model on CPU and NPU on one host."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu
from benchmark_agnn_cora_e2e import (
    Model,
    Operator,
    build_layout,
    digest,
    repeat_layout,
)


def load_model(payload, input_channels, output_channels):
    model = Model(input_channels, output_channels)
    state = payload["state_dict"]
    model.lin1.load_state_dict(
        {"weight": state["lin1.weight"], "bias": state["lin1.bias"]}
    )
    model.lin2.load_state_dict(
        {"weight": state["lin2.weight"], "bias": state["lin2.bias"]}
    )
    model.beta.data.copy_(state["prop2.beta"])
    return model.eval()


def timed_cpu(call, warmup, repeat):
    result = None
    with torch.no_grad():
        for _ in range(warmup):
            result = call()
        samples = []
        for _ in range(repeat):
            begin = time.perf_counter()
            result = call()
            samples.append((time.perf_counter() - begin) * 1000.0)
    return float(statistics.median(samples)), result, samples


def timed_npu(call, warmup, repeat):
    result = None
    with torch.no_grad():
        for _ in range(warmup):
            result = call()
            torch_npu.npu.synchronize()
        samples = []
        for _ in range(repeat):
            torch_npu.npu.synchronize()
            begin = time.perf_counter()
            result = call()
            torch_npu.npu.synchronize()
            samples.append((time.perf_counter() - begin) * 1000.0)
    return float(statistics.median(samples)), result, samples


def benchmark_cpu(args, context, features_cpu, layout_cpu):
    cpu_model, cpu_prop1, cpu_prop2 = context[:3]
    edge_index_cpu = torch.stack((layout_cpu.source, layout_cpu.target))

    def cpu_forward(features=features_cpu, edge_index=edge_index_cpu):
        hidden = F.relu(cpu_model.lin1(features))
        hidden = cpu_prop1(hidden, edge_index)
        hidden = cpu_prop2(hidden, edge_index)
        return cpu_model.lin2(hidden)

    cpu_sweep = []
    cpu_outputs = {}
    for thread_count in args.cpu_threads:
        torch.set_num_threads(thread_count)
        median_ms, output, samples = timed_cpu(cpu_forward, args.warmup, args.repeat)
        cpu_sweep.append(
            {
                "threads": thread_count,
                "median_ms": median_ms,
                "samples_ms": samples,
            }
        )
        cpu_outputs[thread_count] = output
    best_cpu = min(cpu_sweep, key=lambda item: item["median_ms"])
    cpu_output = cpu_outputs.get(best_cpu["threads"])
    if cpu_output is None:
        raise RuntimeError("CPU sweep did not produce the selected output")

    return cpu_sweep, best_cpu, cpu_output


def benchmark_npu(args, context, features_cpu, layout_cpu):
    npu_model, npu_prop1, npu_prop2, operator, device = context[3:]
    features_npu = features_cpu.to(device)
    layout_npu = layout_cpu.to(device)
    edge_index_npu = torch.stack((layout_npu.source, layout_npu.target))

    def official_npu_forward(features=features_npu, edge_index=edge_index_npu):
        hidden = F.relu(npu_model.lin1(features))
        hidden = npu_prop1(hidden, edge_index)
        hidden = npu_prop2(hidden, edge_index)
        return npu_model.lin2(hidden)

    def custom_stage(values, beta, graph):
        normalized = F.normalize(values, p=2.0, dim=-1)
        return operator(values, normalized, beta, graph)

    def custom_npu_forward(features=features_npu, layout=layout_npu):
        return npu_model(features, layout, custom_stage)

    def custom_host_to_host_forward(features=features_cpu, layout=layout_npu):
        # Model weights and the static Cora topology remain resident.
        host_features = features.to(device)
        return npu_model(host_features, layout, custom_stage).cpu()

    official_ms, official_output, official_samples = timed_npu(
        official_npu_forward, args.warmup, args.repeat
    )
    custom_ms, custom_output, custom_samples = timed_npu(
        custom_npu_forward, args.warmup, args.repeat
    )
    host_to_host_ms, host_to_host_output, host_to_host_samples = timed_npu(
        custom_host_to_host_forward, args.warmup, args.repeat
    )
    return (
        (official_ms, official_output, official_samples),
        (custom_ms, custom_output, custom_samples),
        (host_to_host_ms, host_to_host_output, host_to_host_samples),
    )


def summarize_copy(cpu_run, npu_runs, data, copies, dimensions):
    nodes, edges = dimensions
    cpu_sweep, best_cpu, cpu_output = cpu_run
    (
        (official_ms, official_output, official_samples),
        (custom_ms, custom_output, custom_samples),
        (host_to_host_ms, host_to_host_output, host_to_host_samples),
    ) = npu_runs
    custom_cpu = custom_output.cpu()
    labels = data.y.repeat(copies)
    test_mask = data.test_mask.repeat(copies)
    cpu_predictions = cpu_output.argmax(-1)
    custom_predictions = custom_cpu.argmax(-1)
    strongest_non_custom = min(best_cpu["median_ms"], official_ms)
    return {
        "copies": copies,
        "nodes": nodes,
        "edges_with_self_loops": edges,
        "cpu_thread_sweep": cpu_sweep,
        "best_cpu_threads": best_cpu["threads"],
        "best_cpu_model_ms": best_cpu["median_ms"],
        "official_npu_model_ms": official_ms,
        "custom_npu_resident_model_ms": custom_ms,
        "custom_npu_static_graph_host_to_host_ms": host_to_host_ms,
        "strongest_non_custom_model_ms": strongest_non_custom,
        "resident_reduction_vs_strongest_percent": (
            (strongest_non_custom - custom_ms) / strongest_non_custom * 100.0
        ),
        "host_to_host_speedup_vs_cpu": (best_cpu["median_ms"] / host_to_host_ms),
        "cpu_custom_max_abs_diff": float((cpu_output - custom_cpu).abs().max()),
        "official_custom_max_abs_diff": float(
            (official_output - custom_output).abs().max().cpu()
        ),
        "host_to_host_custom_max_abs_diff": float(
            (host_to_host_output - custom_cpu).abs().max()
        ),
        "cpu_custom_prediction_agreement": float(
            (cpu_predictions == custom_predictions).float().mean()
        ),
        "cpu_test_accuracy": float(
            (cpu_predictions[test_mask] == labels[test_mask]).float().mean()
        ),
        "custom_test_accuracy": float(
            (custom_predictions[test_mask] == labels[test_mask]).float().mean()
        ),
        "official_npu_samples_ms": official_samples,
        "custom_npu_samples_ms": custom_samples,
        "custom_host_to_host_samples_ms": host_to_host_samples,
    }


def benchmark_copy(args, data, base_layout, copies, context):
    features, layout = repeat_layout(data.x, base_layout, copies)
    cpu_run = benchmark_cpu(args, context, features, layout)
    npu_runs = benchmark_npu(args, context, features, layout)
    dimensions = int(features.size(0)), int(layout.source.numel())
    return summarize_copy(cpu_run, npu_runs, data, copies, dimensions)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--cpu-threads", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()


def write_output(args, payload, device, results):
    output = {
        "operator": "CsrAgnnCosineAttentionAggregateFused",
        "scope": "complete two-layer maintained PyG AGNN Cora inference model",
        "dataset": "Planetoid Cora",
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_test_accuracy": payload["test_accuracy"],
        "host": "node202",
        "device": torch_npu.npu.get_device_name(device),
        "timing": "FP32 no-grad host wall clock median with synchronization",
        "host_to_host_boundary": (
            "host node features -> H2D -> resident weights/static topology -> "
            "complete model -> D2H logits"
        ),
        "excluded_from_host_to_host": [
            "dataset loading",
            "checkpoint loading",
            "one-time static graph transfer",
            "one-time model weight transfer",
        ],
        "torchair_fullgraph": "not established for the complete PyG model",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logging.getLogger(__name__).info("%s", json.dumps(output, indent=2))


def main():
    args = parse_args()

    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import AGNNConv
    from torch_geometric.transforms import NormalizeFeatures

    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    base_layout = build_layout(data.edge_index, data.num_nodes)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    cpu_model = load_model(payload, data.num_features, dataset.num_classes)
    cpu_prop1 = AGNNConv(requires_grad=False, add_self_loops=False).eval()
    cpu_prop2 = AGNNConv(requires_grad=True, add_self_loops=False).eval()
    cpu_prop2.beta.data.copy_(cpu_model.beta.data)

    device = torch.device(args.device)
    torch_npu.npu.set_device(device)
    npu_model = load_model(payload, data.num_features, dataset.num_classes).to(device)
    npu_prop1 = AGNNConv(requires_grad=False, add_self_loops=False).to(device).eval()
    npu_prop2 = AGNNConv(requires_grad=True, add_self_loops=False).to(device).eval()
    npu_prop2.beta.data.copy_(npu_model.beta.data)
    operator = Operator(args.build, device)

    context = (
        cpu_model,
        cpu_prop1,
        cpu_prop2,
        npu_model,
        npu_prop1,
        npu_prop2,
        operator,
        device,
    )
    results = []
    for copies in args.copies:
        results.append(benchmark_copy(args, data, base_layout, copies, context))

    write_output(args, payload, device, results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
