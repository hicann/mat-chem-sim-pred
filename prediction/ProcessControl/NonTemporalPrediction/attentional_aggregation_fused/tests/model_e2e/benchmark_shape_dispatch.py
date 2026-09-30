#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Measure the AttentionalAggregation custom-path boundary."""

from __future__ import annotations

# Attention-specific shape dispatch imports.
import argparse
import functools
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)
BENCHMARK_VARIANT = "attentional_aggregation"


@dataclass
class CaseContext:
    arguments: object
    module: object
    custom: object
    device: object
    generator: object
    graphs: int


def load_benchmark():
    path = Path(__file__).with_name("benchmark_pyg_attentional_aggregation_e2e.py")
    spec = importlib.util.spec_from_file_location("attention_pool_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    parser = argparse.ArgumentParser()
    parser.set_defaults(operator="attentional")
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graphs", type=int, nargs="+", default=[4, 16, 64, 188])
    parser.add_argument("--nodes-per-graph", type=int, default=24)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def run_case(context):
    arguments = context.arguments
    nodes = context.graphs * arguments.nodes_per_graph
    features = torch.randn(
        (nodes, arguments.channels), generator=context.generator, dtype=torch.float32
    ).to(context.device)
    gates = torch.randn(
        (nodes, 1), generator=context.generator, dtype=torch.float32
    ).to(context.device)
    batch = (
        torch.arange(context.graphs, dtype=torch.int64)
        .repeat_interleave(arguments.nodes_per_graph)
        .to(context.device)
    )
    graph_ptr = torch.arange(
        0, nodes + 1, arguments.nodes_per_graph, dtype=torch.int32
    ).to(context.device)

    resident = functools.partial(context.module.dense_pool, features, gates, batch)
    fused = functools.partial(context.custom, features, gates, graph_ptr)

    resident_ms, expected = context.module.timed(
        resident, arguments.warmup, arguments.repeat
    )
    custom_ms, actual = context.module.timed(fused, arguments.warmup, arguments.repeat)
    return {
        "graphs": context.graphs,
        "nodes": nodes,
        "channels": arguments.channels,
        "resident_ms": resident_ms,
        "custom_ms": custom_ms,
        "speedup": resident_ms / custom_ms,
        "reduction_pct": (resident_ms - custom_ms) / resident_ms * 100.0,
        "max_error": float((expected - actual).abs().max().cpu()),
    }


def run_cases(arguments, module):
    LOGGER.debug("running %s shape cases", BENCHMARK_VARIANT)
    import torch_npu

    _ = torch_npu
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    custom = module.CustomOperator(arguments.build.resolve(), device)
    generator = torch.Generator().manual_seed(20260801)
    results = [
        run_case(CaseContext(arguments, module, custom, device, generator, graphs))
        for graphs in arguments.graphs
    ]
    return {
        "operator": "attentional",
        "device": "Ascend910B3",
        "timing": "resident NPU, synchronized wall-clock median",
        "results": results,
    }


def main():
    logging.basicConfig(level=logging.INFO)
    arguments = parse_args()
    result = run_cases(arguments, load_benchmark())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    LOGGER.info("benchmark result: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
