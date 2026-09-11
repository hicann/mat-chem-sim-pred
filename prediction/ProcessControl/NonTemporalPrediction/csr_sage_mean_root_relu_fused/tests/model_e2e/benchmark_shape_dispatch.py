#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Measure the framework fallback boundary on small synthetic CSR graphs."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import randn

try:
    from prediction.ProcessControl.NonTemporalPrediction.benchmark_common import (
        sage_shape_helpers,
    )
except ModuleNotFoundError:
    from benchmark_common import sage_shape_helpers

TimingRecord, load_module, setup_runtime, timing_result = sage_shape_helpers()

LOGGER = logging.getLogger(__name__)


@dataclass
class ShapeCase:
    module: object
    custom: object
    device: torch.device
    nodes: int
    degree: int
    channels: int
    warmup: int
    repeat: int
    seed: int


def make_layout(module, nodes, degree):
    target = torch.arange(nodes, dtype=torch.int64).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1, dtype=torch.int64).repeat(nodes)
    source = (target + offsets) % nodes
    row_ptr = torch.arange(0, nodes * degree + 1, degree, dtype=torch.int32)
    return module.CsrLayout(source, source.to(torch.int32), target, row_ptr)


def measure_case(case):
    module, custom = case.module, case.custom
    device, nodes = case.device, case.nodes
    generator = torch.Generator().manual_seed(case.seed)
    layout = make_layout(module, nodes, case.degree).to(device)
    neighbor_features = randn(
        (nodes, case.channels), generator=generator, dtype=torch.float32
    ).to(device)
    root_features = randn(
        (nodes, case.channels), generator=generator, dtype=torch.float32
    ).to(device)

    def resident_stage():
        return torch.relu(
            module.sage_mean_root(neighbor_features, root_features, layout)
        )

    def fused_stage():
        return custom(neighbor_features, root_features, layout)

    resident_ms, expected = module.timed(resident_stage, case.warmup, case.repeat)
    custom_ms, actual = module.timed(fused_stage, case.warmup, case.repeat)
    return timing_result(
        TimingRecord(
            nodes,
            layout.source.numel(),
            case.channels,
            resident_ms,
            custom_ms,
            expected,
            actual,
            {},
        )
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[64, 256, 1024, 2708])
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    arguments = parser.parse_args()
    module = load_module(
        Path(__file__).with_name("benchmark_pyg_sage_e2e.py"), "pyg_sage_benchmark"
    )

    device, custom = setup_runtime(module, arguments.build)
    results = []
    for index, nodes in enumerate(arguments.nodes):
        case = ShapeCase(
            module,
            custom,
            device,
            nodes,
            arguments.degree,
            arguments.channels,
            arguments.warmup,
            arguments.repeat,
            20260801 + index,
        )
        results.append(measure_case(case))
    output = {
        "device": "Ascend910B3",
        "timing": "resident NPU, synchronized wall-clock median",
        "results": results,
    }
    arguments.output.write_text(json.dumps(output, indent=2) + "\n")
    LOGGER.info("shape dispatch results: %s", json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
