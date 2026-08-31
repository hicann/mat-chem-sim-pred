#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""NPU correctness, stream, stress, and strong-baseline timing smoke."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import statistics
import time
from pathlib import Path

import torch

LOGGER = logging.getLogger(__name__)


def load_binding(operator_dir):
    path = operator_dir / "integration" / "torch_binding.py"
    spec = importlib.util.spec_from_file_location("dimenet_triplet_binding", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load binding from {path}")
    spec.loader.exec_module(module)
    return module


def resident_triplets(row_ptr, source_index):
    nodes = row_ptr.numel() - 1
    degree = row_ptr[1:].long() - row_ptr[:-1].long()
    target_by_edge = torch.repeat_interleave(
        torch.arange(nodes, device=row_ptr.device), degree
    )
    incoming_count = degree.index_select(0, source_index.long())
    edge_ji = torch.repeat_interleave(
        torch.arange(source_index.numel(), device=row_ptr.device), incoming_count
    )
    group_begin = torch.cumsum(incoming_count, dim=0) - incoming_count
    relative = torch.arange(
        edge_ji.numel(), device=row_ptr.device
    ) - torch.repeat_interleave(group_begin, incoming_count)
    middle = source_index.long().index_select(0, edge_ji)
    edge_kj = row_ptr.long().index_select(0, middle) + relative
    idx_i = target_by_edge.index_select(0, edge_ji)
    idx_j = middle
    idx_k = source_index.long().index_select(0, edge_kj)
    mask = idx_i != idx_k
    return tuple(
        value[mask].to(torch.int32) for value in (idx_i, idx_j, idx_k, edge_kj, edge_ji)
    )


def make_csr(nodes, degree, device):
    row_ptr = torch.arange(
        0, (nodes + 1) * degree, degree, dtype=torch.int32, device=device
    )
    offsets = torch.arange(degree, dtype=torch.int32, device=device)
    target = torch.arange(nodes, dtype=torch.int32, device=device).unsqueeze(1)
    source = ((target + offsets + 1) % nodes).reshape(-1).contiguous()
    return row_ptr, source


def timed(function, warmup=10, repeat=50):
    for _ in range(warmup):
        function()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        torch.npu.synchronize()
        start = time.perf_counter()
        function()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _run_shape(binding, nodes, degree):
    row_ptr, source = make_csr(nodes, degree, "npu")
    capacity = source.numel() * degree
    expected = resident_triplets(row_ptr, source)
    actual = binding.dimenet_triplet_enumerate_fused(row_ptr, source, capacity)
    torch.npu.synchronize()
    counts = actual[-1].cpu()
    count, overflow = int(counts[0]), int(counts[1])
    max_diff = max(
        int((value[:count].cpu() != reference.cpu()).sum())
        for value, reference in zip(actual[:5], expected)
    )
    return (
        {
            "nodes": nodes,
            "edges": source.numel(),
            "degree": degree,
            "triplets": count,
            "overflow": overflow,
            "mismatched_values": max_diff,
            "resident_ms": timed(lambda: resident_triplets(row_ptr, source)),
            "custom_ms": timed(
                lambda: binding.dimenet_triplet_enumerate_fused(
                    row_ptr, source, capacity
                )
            ),
            "independent_outputs": len(
                {value.data_ptr() for value in (*actual, row_ptr, source)}
            )
            == 8,
        },
        row_ptr,
        source,
        capacity,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    binding = load_binding(args.operator_dir)
    binding.configure(args.build_dir)
    results = {"shapes": [], "stream": {}, "stress": {}}
    for nodes, degree in ((64, 4), (512, 8), (2048, 16)):
        shape, row_ptr, source, capacity = _run_shape(binding, nodes, degree)
        results["shapes"].append(shape)

    row_ptr, source = make_csr(257, 8, "npu")
    capacity = source.numel() * 8
    stream_a, stream_b = torch.npu.Stream(), torch.npu.Stream()
    with torch.npu.stream(stream_a):
        output_a = binding.dimenet_triplet_enumerate_fused(row_ptr, source, capacity)
    with torch.npu.stream(stream_b):
        output_b = binding.dimenet_triplet_enumerate_fused(row_ptr, source, capacity)
    stream_a.synchronize()
    stream_b.synchronize()
    results["stream"] = {
        "a_counts": output_a[-1].cpu().tolist(),
        "b_counts": output_b[-1].cpu().tolist(),
    }

    before = torch.npu.memory_allocated()
    checksum = torch.zeros((), dtype=torch.int32, device="npu")
    for _ in range(1000):
        checksum += binding.dimenet_triplet_enumerate_fused(row_ptr, source, capacity)[
            -1
        ][0]
    torch.npu.synchronize()
    after = torch.npu.memory_allocated()
    results["stress"] = {
        "iterations": 1000,
        "checksum": int(checksum.cpu()),
        "allocated_growth_bytes": int(after - before),
    }
    rendered = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    LOGGER.info("%s", rendered)


if __name__ == "__main__":
    main()
