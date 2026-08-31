#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Real ModelNet10 PPFConv audit isolating fused PPF geometry generation."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional
from torch_geometric.data import Batch
from torch_geometric.datasets import ModelNet
from torch_geometric.nn.conv.ppf_conv import point_pair_features

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Case:
    row_ptr: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    edge_index: torch.Tensor
    batch: Batch
    normal: torch.Tensor
    model: torch.nn.Module


@dataclass(frozen=True)
class Environment:
    args: argparse.Namespace
    shared: object
    binding: object
    dataset: ModelNet
    device: torch.device


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clouds", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--points", type=int, default=256)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def resident_geometry(position, normals, source_index, target_index):
    return point_pair_features(
        position.index_select(0, target_index.long()),
        position.index_select(0, source_index.long()),
        normals.index_select(0, target_index.long()),
        normals.index_select(0, source_index.long()),
    )


def prepare_case(environment, cloud_count):
    args = environment.args
    cpu_batch = Batch.from_data_list(
        [
            environment.shared.sample_cloud(environment.dataset[index], args.points)
            for index in range(cloud_count)
        ]
    )
    row_ptr, source, target, edge_index = environment.shared.build_knn_csr(
        cpu_batch, args.neighbors
    )
    batch = cpu_batch.to(environment.device)
    model = environment.shared.PpfClassifier(layers=args.layers).to(environment.device)
    return Case(
        row_ptr.to(environment.device),
        source.to(environment.device),
        target.to(environment.device),
        edge_index.to(environment.device),
        batch,
        functional.normalize(batch.pos, dim=-1),
        model.eval(),
    )


def forward(case, shared, geometry, neighbors):
    features = functional.relu(case.model.input(case.batch.pos))
    ppf = geometry(case.batch.pos, case.normal, case.source, case.target)
    for conv in case.model.convs:
        linear = conv.local_nn[0]
        channels = features.size(1)
        neighbor = functional.linear(features, linear.weight[:, :channels])
        weight = linear.weight[:, channels:].t().contiguous()
        aggregated = shared.resident_aggregate(
            case.row_ptr,
            case.source,
            neighbor,
            ppf,
            weight,
            linear.bias,
            neighbors,
        )
        features = conv.global_nn(aggregated)
    return case.model.classifier(shared.global_mean_pool(features, case.batch.batch))


def benchmark_case(environment, cloud_count):
    case = prepare_case(environment, cloud_count)
    args = environment.args
    resident_call = functools.partial(
        forward, case, environment.shared, resident_geometry, args.neighbors
    )
    custom_call = functools.partial(
        forward,
        case,
        environment.shared,
        environment.binding.ppf_point_pair_features_fused,
        args.neighbors,
    )
    native_call = functools.partial(
        case.model.native,
        case.batch.pos,
        case.normal,
        case.batch.batch,
        case.edge_index,
    )
    with torch.no_grad():
        resident_ms, resident_output = timed(resident_call, args.warmup, args.repeat)
        custom_ms, custom_output = timed(custom_call, args.warmup, args.repeat)
        native_ms, native_output = timed(native_call, 1, 3)
    return {
        "clouds": cloud_count,
        "nodes": int(case.batch.num_nodes),
        "edges": int(case.source.numel()),
        "native_pyg_ms": native_ms,
        "resident_ms": resident_ms,
        "custom_ms": custom_ms,
        "reduction_vs_strongest_pct": (resident_ms - custom_ms) / resident_ms * 100.0,
        "native_vs_resident_max_abs_diff": float(
            (native_output - resident_output).abs().max().cpu()
        ),
        "custom_max_abs_diff": float(
            (custom_output - resident_output).abs().max().cpu()
        ),
        "prediction_agreement": float(
            (custom_output.argmax(-1) == resident_output.argmax(-1))
            .float()
            .mean()
            .cpu()
        ),
    }


def main():
    args = parse_args()

    torch.manual_seed(20260817)
    torch.npu.set_device(0)
    device = torch.device("npu")
    operator_dir = args.operator_dir.resolve()
    binding = load_module(
        "ppf_geometry_binding", operator_dir / "integration" / "torch_binding.py"
    )
    binding.configure(args.build_dir)
    shared = load_module(
        "ppf_model_shared",
        operator_dir.parent
        / "csr_ppf_feature_linear_max_aggregate_fused"
        / "tests"
        / "model_e2e"
        / "benchmark_ppfnet_modelnet10_e2e.py",
    )
    dataset = ModelNet(str(args.dataset_root), name="10", train=False)
    environment = Environment(args, shared, binding, dataset, device)
    results = [benchmark_case(environment, count) for count in args.clouds]

    payload = {
        "model": "three-layer PyG PPFConv classifier",
        "dataset": "real ModelNet10 test point clouds",
        "weights": "deterministic initialized model; no checkpoint quality claim",
        "baseline": "fastest correct official/resident path; resident aggregation shared",
        "timing": "synchronized no-grad median; KNN and transfers excluded",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
