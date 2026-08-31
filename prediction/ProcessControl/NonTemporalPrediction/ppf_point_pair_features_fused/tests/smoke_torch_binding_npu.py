#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Correctness, stream, stress, and stage benchmark for fused PPF geometry."""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import logging
import statistics
import time
from pathlib import Path

import torch
from reference.reference import ppf_point_pair_features_reference

LOGGER = logging.getLogger(__name__)


def load_binding(operator_dir):
    path = operator_dir / "integration" / "torch_binding.py"
    spec = importlib.util.spec_from_file_location("ppf_geometry_binding", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load binding from {path}")
    spec.loader.exec_module(module)
    return module


def reference(position, normal, source, target):
    return ppf_point_pair_features_reference(position, normal, source, target)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def benchmark_sizes(binding):
    results = []
    last_inputs = None
    for nodes, edges in ((1024, 16384), (4096, 65536), (16384, 262144)):
        position = torch.randn(nodes, 3, device="npu")
        normal = torch.nn.functional.normalize(
            torch.randn(nodes, 3, device="npu"), dim=-1
        )
        source = torch.randint(nodes, (edges,), dtype=torch.int32, device="npu")
        target = torch.randint(nodes, (edges,), dtype=torch.int32, device="npu")
        expected = reference(position, normal, source, target)
        actual = binding.ppf_point_pair_features_fused(position, normal, source, target)
        torch.npu.synchronize()
        resident_call = functools.partial(reference, position, normal, source, target)
        custom_call = functools.partial(
            binding.ppf_point_pair_features_fused, position, normal, source, target
        )
        resident_ms = timed(resident_call)
        custom_ms = timed(custom_call)
        results.append(
            {
                "nodes": nodes,
                "edges": edges,
                "resident_ms": resident_ms,
                "custom_ms": custom_ms,
                "speedup": resident_ms / custom_ms,
                "max_abs_diff": float((actual - expected).abs().max().cpu()),
            }
        )
        last_inputs = (position, normal, source, target)
    return results, last_inputs


def validate_streams_and_stress(binding, tensors):
    position, normal, source, target = tensors
    stream_a = torch.npu.Stream()
    stream_b = torch.npu.Stream()
    with torch.npu.stream(stream_a):
        output_a = binding.ppf_point_pair_features_fused(
            position, normal, source, target
        )
    with torch.npu.stream(stream_b):
        output_b = binding.ppf_point_pair_features_fused(
            position, normal, source, target
        )
    stream_a.synchronize()
    stream_b.synchronize()
    stream_diff = float((output_a - output_b).abs().max().cpu())
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    for _ in range(1000):
        binding.ppf_point_pair_features_fused(position, normal, source, target)
    torch.npu.synchronize()
    return stream_diff, int(torch.npu.memory_allocated() - before)


def main():
    args = parse_args()
    torch.manual_seed(20260817)
    torch.npu.set_device(0)
    binding = load_binding(args.operator_dir)
    binding.configure(args.build_dir)
    with torch.no_grad():
        results, last_inputs = benchmark_sizes(binding)
        stream_diff, allocator_growth = validate_streams_and_stress(
            binding, last_inputs
        )
    payload = {
        "device": torch.npu.get_device_name(0),
        "workspace_bytes": 0,
        "dual_stream_max_abs_diff": stream_diff,
        "stress_iterations": 1000,
        "allocator_growth_bytes": allocator_growth,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
