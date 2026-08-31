# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shared torch-npu profiler handling for spectral graph benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch_npu
from spectral_benchmark_common import disjoint_copies, write_payload
from torch_geometric.datasets import Planetoid

LOGGER = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class HotspotSpec:
    label: str
    kernel_types: set[str]


def parse_profile(trace_dir: Path, spec: HotspotSpec) -> dict:
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
    total_us = sum(durations.values())
    hotspot_us = sum(durations[name] for name in spec.kernel_types)
    by_type = [
        {
            "type": name,
            "count": counts[name],
            "duration_us": duration,
            "full_model_ratio": duration / total_us if total_us else 0.0,
        }
        for name, duration in durations.items()
    ]
    by_type.sort(key=lambda item: item["duration_us"], reverse=True)
    return {
        "kernel_count": sum(counts.values()),
        "full_model_kernel_us": total_us,
        f"{spec.label}_kernel_us": hotspot_us,
        f"{spec.label}_hotspot_pct": (
            100.0 * hotspot_us / total_us if total_us else 0.0
        ),
        f"{spec.label}_kernel_types": sorted(spec.kernel_types),
        "by_type": by_type,
    }


def profile_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    return parser


@dataclass
class ProfileCase:
    function: object
    nodes: int
    edges: int
    details: dict


@dataclass
class ProfileSpec:
    case_builder: object
    model_label: str
    trace_prefix: str
    hotspot: HotspotSpec
    warmup: int = 5


def run_profiles(args, spec: ProfileSpec) -> None:
    results = []
    for copies in args.copies:
        case = spec.case_builder(copies)
        with torch.inference_mode():
            for _ in range(spec.warmup):
                case.function()
            torch.npu.synchronize()
        trace_dir = args.trace_root / f"{spec.trace_prefix}_{copies}"
        profile_once(case.function, trace_dir)
        result = {
            "copies": copies,
            "nodes": case.nodes,
            "edges": case.edges,
            **case.details,
            **parse_profile(trace_dir, spec.hotspot),
        }
        results.append(result)
        LOGGER.info("%s", json.dumps(result, sort_keys=True))
    write_payload(
        args.output,
        {
            "model": spec.model_label,
            "dataset": "Cora",
            "profiler": "torch_npu Level1 complete-model NPU kernel profile",
            "results": results,
        },
    )


@dataclass
class BasisProfileContext:
    data: object
    model: torch.nn.Module
    layout: object
    native: object
    staged_forward: object
    device: torch.device


@dataclass
class BasisProfileSetup:
    model_type: object
    hidden: int
    layout: object
    native: object
    staged_forward: object
    transform: object
    model_label: str
    trace_prefix: str
    kernel_types: set[str]


def basis_profile_case(context: BasisProfileContext, copies: int) -> ProfileCase:
    features, edge_index = disjoint_copies(
        context.data.x, context.data.edge_index, copies
    )
    _, _, source, target, norm = context.layout(edge_index, features.size(0))
    features = features.to(context.device)
    source = source.to(context.device)
    target = target.to(context.device)
    norm = norm.to(context.device)
    basis = partial(context.native, source=source, target=target, norm=norm)
    complete_model = partial(context.staged_forward, context.model, features, basis)
    return ProfileCase(complete_model, int(features.size(0)), int(source.numel()), {})


def load_model(model, checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    return model.eval().to(device)


def run_basis_profile(args, setup: BasisProfileSetup) -> None:
    dataset = Planetoid(str(args.data_root), name="Cora", transform=setup.transform)
    device = torch.device("npu:0")
    model = load_model(
        setup.model_type(dataset.num_features, setup.hidden, dataset.num_classes),
        args.checkpoint,
        device,
    )
    context = BasisProfileContext(
        dataset[0],
        model,
        setup.layout,
        setup.native,
        setup.staged_forward,
        device,
    )
    run_profiles(
        args,
        ProfileSpec(
            partial(basis_profile_case, context),
            setup.model_label,
            setup.trace_prefix,
            HotspotSpec("basis", setup.kernel_types),
        ),
    )
