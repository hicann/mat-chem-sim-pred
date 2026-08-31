#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Incremental DimeNet++ E2E audit for the fused triplet-angle operator."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial

import torch
from dimenet_shared import (
    ForwardCase,
    GraphContext,
    MeasureContext,
    add_dimenet_args,
    build_model,
    custom_triplets,
    load_module,
    model_forward,
    native_angle,
    prepare_graph,
    resident_triplets,
    timed,
)
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9

LOGGER = logging.getLogger(__name__)


def _run_graph(context):
    dataset = context.dataset
    graph_count = context.graph_count
    args = context.args
    device = context.device
    angle_binding = context.angle_binding
    triplet_binding = context.triplet_binding
    cpu_batch = Batch.from_data_list(
        [dataset[2048 + index] for index in range(graph_count)]
    )
    row_ptr, source, target, max_degree = prepare_graph(cpu_batch)
    batch = cpu_batch.to(device)
    row_ptr, source, target = (value.to(device) for value in (row_ptr, source, target))
    resident_provider, custom_provider = _providers(
        row_ptr, source, target, max_degree, triplet_binding
    )
    model = build_model(device)
    return _measure(
        MeasureContext(
            graph_count,
            batch,
            source,
            target,
            model,
            resident_provider,
            custom_provider,
            angle_binding,
            args,
        )
    )


def _providers(row_ptr, source, target, max_degree, triplet_binding):
    capacity = source.numel() * max_degree
    resident = partial(resident_triplets, row_ptr, source, target)
    custom = partial(custom_triplets, triplet_binding, row_ptr, source, capacity)
    return resident, custom


def _measure(context):
    graph_count, batch = context.graph_count, context.batch
    source, target, model = context.source, context.target, context.model
    resident_provider, custom_provider = (
        context.resident_provider,
        context.custom_provider,
    )
    angle_binding, args = context.angle_binding, context.args
    resident_case = ForwardCase(
        model,
        batch.z,
        batch.pos,
        batch.batch,
        source,
        target,
        resident_provider,
        native_angle,
    )
    custom_case = ForwardCase(
        model,
        batch.z,
        batch.pos,
        batch.batch,
        source,
        target,
        custom_provider,
        angle_binding.dimenet_triplet_angle_fused,
    )
    resident_ms, resident_output = timed(
        partial(model_forward, resident_case), args.warmup, args.repeat
    )
    custom_ms, custom_output = timed(
        partial(model_forward, custom_case), args.warmup, args.repeat
    )
    triplets = resident_provider()
    return {
        "graphs": graph_count,
        "triplets": int(triplets[0].numel()),
        "resident_ms": resident_ms,
        "custom_ms": custom_ms,
        "e2e_reduction_pct": (resident_ms - custom_ms) / resident_ms * 100.0,
        "model_max_abs_diff": float(
            (resident_output - custom_output).abs().max().cpu()
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    add_dimenet_args(parser, with_angle_paths=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(20260817)
    torch.npu.set_device(0)
    device = torch.device("npu")
    angle_binding = load_module(
        "dimenet_angle_binding", args.operator_dir / "integration" / "torch_binding.py"
    )
    triplet_binding = load_module(
        "dimenet_triplet_binding", args.triplet_dir / "integration" / "torch_binding.py"
    )
    angle_binding.configure(args.angle_build_dir)
    triplet_binding.configure(args.triplet_build_dir)
    dataset = QM9(str(args.dataset_root))
    results = [
        _run_graph(
            GraphContext(dataset, count, args, device, angle_binding, triplet_binding)
        )
        for count in args.graphs
    ]
    payload = {
        "model": "PyG DimeNetPlusPlus",
        "dataset": "QM9 real molecule batches",
        "weights": "deterministic initialized model; no checkpoint quality claim",
        "baseline": "fastest correct topology plus resident NPU triplet geometry",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
