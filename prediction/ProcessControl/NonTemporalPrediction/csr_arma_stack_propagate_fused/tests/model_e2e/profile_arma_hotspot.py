#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Profile the replaceable propagation stage in complete PyG ARMA."""

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch_geometric.transforms as T
import torch_npu
from benchmark_arma_cora_e2e import (
    ArmaNet,
    disjoint_graph,
    native_stage,
)
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.conv.gcn_conv import gcn_norm

LOGGER = logging.getLogger(__name__)

STAGE_KERNEL_TYPES = {
    "Add",
    "Index",
    "InplaceIndexAdd",
    "Mul",
    "ZerosLike",
}


def profile_once(function, trace_dir: Path) -> None:
    experimental = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        data_simplification=True,
    )
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(trace_dir), analyse_flag=True, async_mode=False
    )
    with (
        torch.inference_mode(),
        torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=0
            ),
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_modules=True,
            experimental_config=experimental,
        ) as profiler,
    ):
        function()
        torch.npu.synchronize()
        profiler.step()


def parse_profile(trace_dir: Path) -> dict:
    csv_paths = sorted(trace_dir.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(f"expected one kernel CSV, got {csv_paths}")
    durations = defaultdict(float)
    counts = defaultdict(int)
    with csv_paths[0].open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            op_type = row["Type"].strip()
            durations[op_type] += float(row["Duration(us)"].strip())
            counts[op_type] += 1
    full_model_us = sum(durations.values())
    stage_us = sum(durations[name] for name in STAGE_KERNEL_TYPES)
    by_type = [
        {
            "type": name,
            "count": counts[name],
            "duration_us": duration,
            "full_model_ratio": duration / full_model_us,
        }
        for name, duration in durations.items()
    ]
    by_type.sort(key=lambda item: item["duration_us"], reverse=True)
    return {
        "kernel_count": sum(counts.values()),
        "full_model_kernel_us": full_model_us,
        "stage_kernel_us": stage_us,
        "stage_hotspot_pct": 100.0 * stage_us / full_model_us,
        "stage_kernel_types": sorted(STAGE_KERNEL_TYPES),
        "relu_excluded_from_attribution": True,
        "by_type": by_type,
    }


@dataclass
class ProfileContext:
    data: object
    normalized_index: torch.Tensor
    normalized_weight: torch.Tensor
    model: ArmaNet
    device: torch.device


@dataclass
class ProfileCase:
    function: object
    nodes: int
    edges: int


def _load_context(args) -> ProfileContext:
    dataset = Planetoid(
        str(args.data_root), name="Cora", transform=T.NormalizeFeatures()
    )
    data = dataset[0]
    normalized_index, normalized_weight = gcn_norm(
        data.edge_index,
        None,
        data.num_nodes,
        add_self_loops=False,
        dtype=torch.float32,
    )
    model = ArmaNet(dataset.num_features, dataset.num_classes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("npu:0")
    return ProfileContext(
        data, normalized_index, normalized_weight, model.eval().to(device), device
    )


def _profile_case(context: ProfileContext, copies: int) -> ProfileCase:
    graph = disjoint_graph(
        context.normalized_index,
        context.normalized_weight,
        context.data.num_nodes,
        copies,
    ).to(context.device)
    features = context.data.x.repeat(copies, 1).to(context.device)
    function = partial(context.model, features, graph, native_stage)
    return ProfileCase(function, int(features.size(0)), int(graph.source.numel()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    return parser


def main() -> None:
    args = _parser().parse_args()
    context = _load_context(args)
    results = []
    for copies in args.copies:
        case = _profile_case(context, copies)
        with torch.inference_mode():
            for _ in range(3):
                case.function()
            torch.npu.synchronize()
        trace_dir = args.trace_root / f"cora_arma_copies_{copies}"
        profile_once(case.function, trace_dir)
        result = {
            "copies": copies,
            "nodes": case.nodes,
            "edges": case.edges,
            "replaceable_calls": 4,
            **parse_profile(trace_dir),
        }
        results.append(result)
        LOGGER.info("%s", json.dumps(result, sort_keys=True))
    payload = {
        "model": "maintained PyG examples/arma.py",
        "dataset": "Cora",
        "profiler": "torch_npu Level1 complete-model NPU kernel profile",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
