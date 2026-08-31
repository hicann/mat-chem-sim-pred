#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""DimeNet++ E2E audit for resident versus custom triplet construction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from functools import partial
from pathlib import Path

import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9

SHARED_DIR = (
    Path(__file__).parents[2].parent
    / "dimenet_triplet_angle_fused"
    / "tests"
    / "model_e2e"
)


def _load_shared():
    path = SHARED_DIR / "dimenet_shared.py"
    spec = importlib.util.spec_from_file_location("dimenet_shared", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LOGGER = logging.getLogger(__name__)


def _run_graph(context):
    shared = _load_shared()
    dataset = context.dataset
    graph_count = context.graph_count
    args = context.args
    device = context.device
    binding = context.triplet_binding
    cpu_batch = Batch.from_data_list(
        [dataset[2048 + index] for index in range(graph_count)]
    )
    row_ptr, source, target, max_degree = shared.prepare_graph(cpu_batch)
    batch = cpu_batch.to(device)
    row_ptr, source, target = row_ptr.to(device), source.to(device), target.to(device)
    capacity = source.numel() * max_degree
    model = shared.build_model(device)
    resident_provider, custom_provider = _providers(
        shared.ProviderContext(shared, row_ptr, source, target, binding, capacity)
    )
    return _measure(
        shared.MeasureContext(
            graph_count,
            batch,
            source,
            target,
            model,
            resident_provider,
            custom_provider,
            args=args,
        ),
        max_degree,
        shared,
    )


def _providers(context):
    resident = partial(
        context.shared.resident_triplets,
        context.row_ptr,
        context.source,
        context.target,
    )
    custom = partial(
        context.shared.custom_triplets,
        context.binding,
        context.row_ptr,
        context.source,
        context.capacity,
    )
    return resident, custom


def _forward_case(context, provider, shared):
    return shared.ForwardCase(
        context.model,
        context.batch.z,
        context.batch.pos,
        context.batch.batch,
        context.source,
        context.target,
        provider,
        shared.native_angle,
    )


def _measure(context, max_degree, shared):
    graph_count, batch = context.graph_count, context.batch
    source = context.source
    resident_provider, custom_provider, args = (
        context.resident_provider,
        context.custom_provider,
        context.args,
    )
    expected, mismatch = _topology_check(resident_provider, custom_provider)
    resident_case = _forward_case(context, resident_provider, shared)
    custom_case = _forward_case(context, custom_provider, shared)
    resident_stage_ms, _ = shared.timed(resident_provider, args.warmup, args.repeat)
    custom_stage_ms, _ = shared.timed(custom_provider, args.warmup, args.repeat)
    resident_e2e_ms, resident_output = shared.timed(
        partial(shared.model_forward, resident_case), args.warmup, args.repeat
    )
    custom_e2e_ms, custom_output = shared.timed(
        partial(shared.model_forward, custom_case), args.warmup, args.repeat
    )
    return _result(
        shared.ResultContext(
            graph_count,
            max_degree,
            batch,
            source,
            expected,
            mismatch,
            resident_stage_ms,
            custom_stage_ms,
            resident_e2e_ms,
            custom_e2e_ms,
            resident_output,
            custom_output,
        )
    )


def _topology_check(resident_provider, custom_provider):
    expected = resident_provider()
    actual = custom_provider()
    mismatch = sum(
        int((left != right).sum().cpu()) for left, right in zip(expected, actual)
    )
    return expected, mismatch


def _result(context):
    graph_count, max_degree, batch, source = (
        context.graph_count,
        context.max_degree,
        context.batch,
        context.source,
    )
    expected, mismatch = context.expected, context.mismatch
    resident_stage_ms, custom_stage_ms = (
        context.resident_stage_ms,
        context.custom_stage_ms,
    )
    resident_e2e_ms, custom_e2e_ms = context.resident_e2e_ms, context.custom_e2e_ms
    resident_output, custom_output = context.resident_output, context.custom_output
    return {
        "graphs": graph_count,
        "nodes": int(batch.num_nodes),
        "edges": source.numel(),
        "max_degree": max_degree,
        "triplets": expected[0].numel(),
        "topology_mismatches": mismatch,
        "resident_stage_ms": resident_stage_ms,
        "custom_stage_ms": custom_stage_ms,
        "stage_speedup": resident_stage_ms / custom_stage_ms,
        "resident_e2e_ms": resident_e2e_ms,
        "custom_e2e_ms": custom_e2e_ms,
        "e2e_reduction_pct": (resident_e2e_ms - custom_e2e_ms)
        / resident_e2e_ms
        * 100.0,
        "model_max_abs_diff": float(
            (resident_output - custom_output).abs().max().cpu()
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    shared = _load_shared()
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    shared.add_graph_args(parser)
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(20260817)
    torch.npu.set_device(0)
    device = torch.device("npu")
    binding = _load_shared().load_module(
        "dimenet_triplet_binding",
        args.operator_dir / "integration" / "torch_binding.py",
    )
    binding.configure(args.build_dir)
    dataset = QM9(str(args.dataset_root))
    shared = _load_shared()
    results = [
        _run_graph(shared.GraphContext(dataset, count, args, device, None, binding))
        for count in args.graphs
    ]
    payload = {
        "model": "PyG DimeNetPlusPlus",
        "dataset": "QM9 real molecule batches",
        "weights": "deterministic initialized model; no checkpoint quality claim",
        "baseline": "resident NPU tensor triplet construction plus identical DimeNet++ forward",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
