#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Validate direct-operator dispatch shapes on Ascend NPU."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Generator

try:
    from prediction.ProcessControl.NonTemporalPrediction.benchmark_common import (
        appnp_shape_helpers,
    )
except ModuleNotFoundError:
    from benchmark_common import appnp_shape_helpers

TimingRecord, load_module, setup_runtime, timing_result = appnp_shape_helpers()

LOGGER = logging.getLogger(__name__)


@dataclass
class ShapeCase:
    module: object
    custom: object
    device: torch.device
    nodes: int
    degree: int
    channels: int
    iterations: int
    alpha: float
    warmup: int
    repeat: int
    seed: int


def regular_layout(module, nodes, degree):
    target = torch.arange(nodes, dtype=torch.int64).repeat_interleave(degree)
    offsets = torch.arange(degree, dtype=torch.int64).repeat(nodes)
    source = (target + offsets) % nodes
    row_ptr = torch.arange(0, nodes * degree + 1, degree, dtype=torch.int32)
    return module.CsrLayout(
        source,
        source.to(torch.int32),
        target,
        row_ptr,
        torch.full((nodes * degree,), 1.0 / degree, dtype=torch.float32),
    )


def measure_case(case):
    generator = Generator(device="cpu").manual_seed(case.seed)
    initial = torch.randn((case.nodes, case.channels), generator=generator).to(
        case.device
    )
    layout = regular_layout(case.module, case.nodes, case.degree).to(case.device)

    def resident_stage():
        return case.module.resident_propagate(
            initial, layout, case.iterations, case.alpha
        )

    def fused_stage():
        return case.custom(initial, layout, case.iterations, case.alpha)

    resident_ms, expected = case.module.timed(resident_stage, case.warmup, case.repeat)
    custom_ms, actual = case.module.timed(fused_stage, case.warmup, case.repeat)
    return timing_result(
        TimingRecord(
            case.nodes,
            layout.source.numel(),
            case.channels,
            resident_ms,
            custom_ms,
            expected,
            actual,
            {"iterations": case.iterations},
        )
    )


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, nargs="+", default=[64, 256, 1024, 2708])
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    arguments = parser.parse_args()

    module = load_module(
        Path(__file__).with_name("benchmark_pyg_appnp_e2e.py"), "appnp_model_benchmark"
    )
    device, custom = setup_runtime(module, arguments.build)
    results = [
        measure_case(
            ShapeCase(
                module,
                custom,
                device,
                nodes,
                arguments.degree,
                arguments.channels,
                arguments.iterations,
                arguments.alpha,
                arguments.warmup,
                arguments.repeat,
                20260801 + index,
            )
        )
        for index, nodes in enumerate(arguments.nodes)
    ]
    payload = {
        "device": "Ascend910B3",
        "baseline": "resident torch_npu index_add propagation",
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("shape dispatch results: %s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
