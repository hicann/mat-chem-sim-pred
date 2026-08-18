#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark PointBallQueryFused inside a complete PointNet++ SSG forward."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
from numpy import mean

importlib.import_module("torch_npu")
sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("pointnet2_geometry_model_e2e_common")


@dataclass(frozen=True)
class BenchmarkContext:
    model: object
    utils_module: object
    framework_fps: object
    custom_fps: object
    framework_query: object
    custom_query: object
    warmup: int
    repeat: int


@dataclass
class Measurements:
    framework_stage: object
    custom_stage: object
    original_model: object
    fps_model: object
    optimized_model: object


@dataclass
class BenchmarkSetup:
    context: BenchmarkContext
    checkpoint: dict
    device: torch.device
    make_inputs: object


class PointBallQueryOperator:
    def __init__(self, build_dir, device):
        self.library = _common.load_host_library(build_dir, "point_ball_query_fused")
        self.device = device
        self.cache = {}
        workspace = self.library.aclnnPointBallQueryFusedGetWorkspaceSize
        workspace.argtypes = [ctypes.c_int64] * 4
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnPointBallQueryFused
        operation.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int64] * 4
            + [ctypes.c_float, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        operation.restype = ctypes.c_int32

    def __call__(self, radius, sample_count, points, queries):
        points = points.contiguous()
        queries = queries.contiguous()
        batch, point_count, dimensions = points.shape
        query_batch, query_count, query_dimensions = queries.shape
        if dimensions != 3 or query_dimensions != 3 or query_batch != batch:
            raise ValueError("ball query expects points [B,N,3] and queries [B,S,3]")
        key = batch, point_count, query_count, sample_count
        if key not in self.cache:
            size = int(self.library.aclnnPointBallQueryFusedGetWorkspaceSize(*key))
            self.cache[key] = (
                torch.empty((size,), dtype=torch.uint8, device=self.device),
                torch.empty(
                    (batch, query_count, sample_count),
                    dtype=torch.int32,
                    device=self.device,
                ),
                torch.empty(
                    (batch, query_count), dtype=torch.int32, device=self.device
                ),
            )
        workspace, indices, counts = self.cache[key]
        result = self.library.aclnnPointBallQueryFused(
            points.data_ptr(),
            queries.data_ptr(),
            indices.data_ptr(),
            counts.data_ptr(),
            batch,
            point_count,
            query_count,
            sample_count,
            radius,
            workspace.data_ptr(),
            workspace.numel(),
            torch.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"aclnnPointBallQueryFused returned {result}")
        first = indices[..., :1]
        return torch.where(indices >= 0, indices, first).to(torch.int64)


def load_helpers(benchmark_path):
    helper = _common.load_module(benchmark_path, "pointnet2_fps_benchmark")
    return helper.FarthestPointSamplingOperator, helper.load_model, helper.make_inputs


def _capture_queries(context, model_input):
    captured = []

    def capture(radius, sample_count, points, queries):
        captured.append((radius, sample_count, points.detach(), queries.detach()))
        return context.framework_query(radius, sample_count, points, queries)

    context.utils_module.farthest_point_sample = context.custom_fps
    context.utils_module.query_ball_point = capture
    with torch.no_grad():
        context.model(model_input)
    if len(captured) != 2:
        raise RuntimeError(f"expected two ball-query calls, captured {len(captured)}")
    return captured


def _query_stages(query, captured):
    result = []
    for arguments in captured:
        result.append(query(*arguments))
    return tuple(result)


def _model_forward(context, model_input, fps, query):
    context.utils_module.farthest_point_sample = fps
    context.utils_module.query_ball_point = query
    with torch.no_grad():
        return context.model(model_input)[0]


def _measure(context, model_input, captured, reverse_order):
    calls = {
        "framework_stage": partial(_query_stages, context.framework_query, captured),
        "custom_stage": partial(_query_stages, context.custom_query, captured),
        "original_model": partial(
            _model_forward,
            context,
            model_input,
            context.framework_fps,
            context.framework_query,
        ),
        "fps_model": partial(
            _model_forward,
            context,
            model_input,
            context.custom_fps,
            context.framework_query,
        ),
        "optimized_model": partial(
            _model_forward,
            context,
            model_input,
            context.custom_fps,
            context.custom_query,
        ),
    }
    order = (
        "framework_stage",
        "custom_stage",
        "original_model",
        "fps_model",
        "optimized_model",
    )
    if reverse_order:
        order = (
            "custom_stage",
            "framework_stage",
            "optimized_model",
            "fps_model",
            "original_model",
        )
    measured = {}
    for name in order:
        call = calls.get(name)
        if call is None:
            raise KeyError(f"missing benchmark call: {name}")
        measured[name] = _common.timed(call, context.warmup, context.repeat)
    return Measurements(**measured)


def _call_payload(captured):
    result = []
    for radius, sample_count, points, queries in captured:
        result.append(
            {
                "points": int(points.size(1)),
                "queries": int(queries.size(1)),
                "samples": int(sample_count),
                "radius": float(radius),
            }
        )
    return result


def _mismatches(expected, actual):
    return [int((left != right).sum().cpu()) for left, right in zip(expected, actual)]


def _timing_payload(measured):
    framework = measured.framework_stage.median_ms
    custom = measured.custom_stage.median_ms
    original = measured.original_model.median_ms
    fps = measured.fps_model.median_ms
    optimized = measured.optimized_model.median_ms
    return {
        "framework_ball_query_ms": framework,
        "custom_ball_query_ms": custom,
        "ball_query_reduction_pct": (framework - custom) / framework * 100.0,
        "original_model_e2e_ms": original,
        "fps_optimized_model_e2e_ms": fps,
        "fps_and_ball_query_model_e2e_ms": optimized,
        "incremental_model_reduction_pct": (fps - optimized) / fps * 100.0,
        "reduction_from_original_pct": (original - optimized) / original * 100.0,
        "ball_query_share_after_fps_pct": framework / fps * 100.0,
    }


def _samples_payload(measured):
    return {
        "framework_ball_query": measured.framework_stage.samples_ms,
        "custom_ball_query": measured.custom_stage.samples_ms,
        "original_model": measured.original_model.samples_ms,
        "fps_optimized_model": measured.fps_model.samples_ms,
        "fps_and_ball_query_model": measured.optimized_model.samples_ms,
    }


def run_case(context, model_input, reverse_order):
    captured = _capture_queries(context, model_input)
    measured = _measure(context, model_input, captured, reverse_order)
    result = {
        "batch": int(model_input.size(0)),
        "input_shape": list(model_input.shape),
        "ball_query_calls": _call_payload(captured),
        "parity": {
            "index_mismatches": _mismatches(
                measured.framework_stage.output, measured.custom_stage.output
            ),
            "fps_baseline_vs_custom_ball_query": _common.compare_outputs(
                measured.fps_model.output, measured.optimized_model.output
            ),
            "original_vs_fully_optimized": _common.compare_outputs(
                measured.original_model.output, measured.optimized_model.output
            ),
        },
        "samples_ms": _samples_payload(measured),
    }
    result.update(_timing_payload(measured))
    return result


LATENCY_KEYS = (
    "framework_ball_query_ms",
    "custom_ball_query_ms",
    "original_model_e2e_ms",
    "fps_optimized_model_e2e_ms",
    "fps_and_ball_query_model_e2e_ms",
)
SUMMARY_OMIT = {
    "ball_query_reduction_pct",
    "incremental_model_reduction_pct",
    "reduction_from_original_pct",
    "ball_query_share_after_fps_pct",
    "parity",
    "samples_ms",
}


def _base_average(cases):
    result = {}
    for key, value in cases[0].items():
        if key in LATENCY_KEYS or key in SUMMARY_OMIT:
            continue
        result[key] = value
    for key in LATENCY_KEYS:
        result[key] = float(mean([case[key] for case in cases]))
    return result


def _add_reductions(result):
    result["ball_query_reduction_pct"] = (
        (result["framework_ball_query_ms"] - result["custom_ball_query_ms"])
        / result["framework_ball_query_ms"]
        * 100.0
    )
    result["incremental_model_reduction_pct"] = (
        (
            result["fps_optimized_model_e2e_ms"]
            - result["fps_and_ball_query_model_e2e_ms"]
        )
        / result["fps_optimized_model_e2e_ms"]
        * 100.0
    )
    result["reduction_from_original_pct"] = (
        (result["original_model_e2e_ms"] - result["fps_and_ball_query_model_e2e_ms"])
        / result["original_model_e2e_ms"]
        * 100.0
    )
    result["ball_query_share_after_fps_pct"] = (
        result["framework_ball_query_ms"] / result["fps_optimized_model_e2e_ms"] * 100.0
    )


def _parity_average(cases):
    mismatches = []
    for index in range(2):
        mismatches.append(
            max(case["parity"]["index_mismatches"][index] for case in cases)
        )
    comparisons = [
        case["parity"]["fps_baseline_vs_custom_ball_query"] for case in cases
    ]
    return {
        "index_mismatches": mismatches,
        "max_model_abs_error": max(value["max_abs_error"] for value in comparisons),
        "min_model_top1_agreement": min(
            value["top1_agreement"] for value in comparisons
        ),
    }


def average_trials(all_trials):
    results = []
    for case_index in range(len(all_trials[0])):
        cases = [trial[case_index] for trial in all_trials]
        result = _base_average(cases)
        _add_reductions(result)
        result["parity"] = _parity_average(cases)
        result["trials"] = cases
        results.append(result)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps-benchmark", type=Path, required=True)
    parser.add_argument("--pointnet2-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--point-clouds", type=Path, nargs="+", required=True)
    parser.add_argument("--fps-build-dir", type=Path, required=True)
    parser.add_argument("--operator-build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2)
    return parser.parse_args()


def prepare_benchmark(args):
    fps_class, load_model, make_inputs = load_helpers(args.fps_benchmark)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model, utils_module, checkpoint = load_model(
        args.pointnet2_source, args.checkpoint, device
    )
    context = BenchmarkContext(
        model,
        utils_module,
        utils_module.farthest_point_sample,
        fps_class(args.fps_build_dir, device),
        utils_module.query_ball_point,
        PointBallQueryOperator(args.operator_build_dir, device),
        args.warmup,
        args.repeat,
    )
    return BenchmarkSetup(context, checkpoint, device, make_inputs)


def build_trials(args, setup):
    results = []
    for trial_index in range(args.trials):
        trial = []
        for batch in args.batches:
            model_input = setup.make_inputs(
                batch, 1024, setup.device, args.point_clouds
            )
            trial.append(run_case(setup.context, model_input, trial_index % 2 == 1))
        results.append(trial)
    return results


def build_payload(args, setup, results):
    return {
        "date": "2026-07-31",
        "device": torch.npu.get_device_name(setup.device),
        "model": "PointNet++ SSG classification",
        "checkpoint": setup.checkpoint,
        "input": "real normalized ModelNet40 point-cloud samples",
        "baseline_policy": (
            "measure original PointNet++, then isolate ball query on top of the already "
            "validated FarthestPointSamplingFused path"
        ),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "results": results,
    }


def main():
    args = parse_args()
    setup = prepare_benchmark(args)
    payload = build_payload(args, setup, average_trials(build_trials(args, setup)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
