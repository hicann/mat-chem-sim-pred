#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Measure SortAggregation fallback boundaries on synthetic graph batches."""

from __future__ import annotations

import argparse as _ap
from dataclasses import dataclass
import functools as _functools
from importlib import util as _import_util
import json as _json
import logging as _logging
from pathlib import Path as _Path
import sys as _sys

import torch

LOGGER = _logging.getLogger("sort_shape_dispatch")
BENCHMARK_VARIANT = "sort_aggregation"


class SortDispatchContext:
    """Evaluation bundle for synthetic sort dispatch experiments."""

    def __init__(self, arguments, module, custom, official, device, generator, graphs: int):
        self.arguments = arguments
        self.module = module
        self.custom = custom
        self.official = official
        self.device = device
        self.generator = generator
        self.graphs = graphs


def load_model_benchmark():
    py_path = _Path(__file__).with_name("benchmark_pyg_dgcnn_e2e.py")
    spec = _import_util.spec_from_file_location("dgcnn_benchmark", py_path)
    benchmark_mod = _import_util.module_from_spec(spec)
    _sys.modules[spec.name] = benchmark_mod
    spec.loader.exec_module(benchmark_mod)
    return benchmark_mod


def parse_args():
    parser = _ap.ArgumentParser(description="Sort shape dispatch benchmark")
    parser.set_defaults(operator="sort")
    parser.add_argument("--build", type=_Path, required=True)
    parser.add_argument("--output", type=_Path, required=True)
    parser.add_argument("--graphs", type=int, nargs="+", default=[4, 16, 64, 188])
    parser.add_argument("--nodes-per-graph", type=int, default=24)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    return parser.parse_args()


def run_case(context: SortDispatchContext):
    args = context.arguments
    tot_nodes = context.graphs * args.nodes_per_graph
    features = torch.randn(
        (tot_nodes, args.channels), generator=context.generator, dtype=torch.float32
    ).to(context.device)
    batch_vec = (
        torch.arange(context.graphs, dtype=torch.int64)
        .repeat_interleave(args.nodes_per_graph)
        .to(context.device)
    )
    graph_ptr = torch.arange(
        0, tot_nodes + 1, args.nodes_per_graph, dtype=torch.int32
    ).to(context.device)

    baseline_fn = _functools.partial(context.official, features, batch_vec)
    fused_fn = _functools.partial(context.custom, features, graph_ptr, args.top_k)

    baseline_ms, expected = context.module.timed(
        baseline_fn, args.warmup, args.repeat
    )
    custom_ms, actual = context.module.timed(fused_fn, args.warmup, args.repeat)
    return {
        "graphs": context.graphs,
        "nodes": tot_nodes,
        "channels": args.channels,
        "top_k": args.top_k,
        "baseline_ms": baseline_ms,
        "custom_ms": custom_ms,
        "speedup": baseline_ms / custom_ms,
        "reduction_pct": (baseline_ms - custom_ms) / baseline_ms * 100.0,
        "max_error": float((expected - actual).abs().max().cpu()),
    }


def run_cases(arguments, module):
    LOGGER.debug("running %s shape cases", BENCHMARK_VARIANT)
    import torch_npu
    from torch_geometric.nn.aggr import SortAggregation

    _ = torch_npu
    dev = torch.device("npu:0")
    torch.npu.set_device(dev)
    custom_op = module.CustomOperator(arguments.build.resolve(), dev)
    official_op = SortAggregation(arguments.top_k).to(dev)
    rng = torch.Generator().manual_seed(20260801)
    case_records = [
        run_case(
            SortDispatchContext(arguments, module, custom_op, official_op, dev, rng, n_g)
        )
        for n_g in arguments.graphs
    ]
    return {
        "operator": "sort",
        "device": "Ascend910B3",
        "timing": "resident NPU, synchronized wall-clock median",
        "results": case_records,
    }


def main():
    _logging.basicConfig(level=_logging.INFO)
    arguments = parse_args()
    result = run_cases(arguments, load_model_benchmark())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(_json.dumps(result, indent=2) + "\n")
    LOGGER.info("benchmark result: %s", _json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
