#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Benchmark FarthestPointSamplingFused inside a complete PointNet++ SSG model."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch

importlib.import_module("torch_npu")
sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("pointnet2_geometry_model_e2e_common")


@dataclass
class BenchmarkConfig:
    model: object
    utils_module: object
    custom_fps: object
    warmup: int
    repeat: int
    device: torch.device
    point_clouds: list[Path] | None
    labels: list[int] | None


@dataclass
class BatchCase:
    batch: int
    model_input: torch.Tensor
    first_stage_points: torch.Tensor
    second_stage_points: torch.Tensor


@dataclass
class Measurements:
    framework_stage: object
    custom_stage: object
    framework_model: object
    custom_model: object


@dataclass
class BenchmarkSetup:
    config: BenchmarkConfig
    checkpoint: dict


class FarthestPointSamplingOperator:
    def __init__(self, build_dir, device):
        self.library = _common.load_host_library(
            build_dir, "farthest_point_sampling_fused"
        )
        self.device = device
        self.cache = {}
        workspace = self.library.aclnnFarthestPointSamplingFusedGetWorkspaceSize
        workspace.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnFarthestPointSamplingFused
        operation.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        operation.restype = ctypes.c_int32

    def __call__(self, points, sample_count):
        points = points.contiguous()
        batch, point_count, dimensions = points.shape
        if dimensions != 3:
            raise ValueError(f"FPS expects [B,N,3], got {tuple(points.shape)}")
        key = batch, point_count, sample_count
        if key not in self.cache:
            size = int(
                self.library.aclnnFarthestPointSamplingFusedGetWorkspaceSize(*key)
            )
            self.cache[key] = (
                torch.empty((size,), dtype=torch.uint8, device=self.device),
                torch.empty(
                    (batch, sample_count), dtype=torch.int32, device=self.device
                ),
            )
        workspace, output = self.cache[key]
        result = self.library.aclnnFarthestPointSamplingFused(
            points.data_ptr(),
            output.data_ptr(),
            batch,
            point_count,
            sample_count,
            workspace.data_ptr(),
            workspace.numel(),
            torch.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"aclnnFarthestPointSamplingFused returned {result}")
        return output.to(torch.int64)


def framework_fps(points, sample_count):
    """Resident Torch NPU FPS matching the operator's deterministic index-zero start."""
    batch, point_count, _ = points.shape
    batch_index = torch.arange(batch, dtype=torch.int64, device=points.device)
    minimum_distance = torch.full(
        (batch, point_count), float("inf"), dtype=points.dtype, device=points.device
    )
    selected = torch.zeros((batch,), dtype=torch.int64, device=points.device)
    output = torch.empty((batch, sample_count), dtype=torch.int64, device=points.device)
    for sample in range(sample_count):
        output[:, sample] = selected
        if sample + 1 == sample_count:
            break
        distance = ((points - points[batch_index, selected, None, :]) ** 2).sum(-1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        selected = minimum_distance.argmax(-1)
    return output


def normalize_points(points):
    points = points - points.mean(0, keepdim=True)
    return points / points.norm(dim=-1).max().clamp_min(1.0e-6)


def make_inputs(batch, point_count, device, point_clouds):
    if point_clouds:
        arrays = []
        for path in point_clouds:
            points = torch.from_numpy(np.load(path).astype(np.float32, copy=False))
            if points.ndim != 2 or points.size(1) < 3 or points.size(0) < point_count:
                raise ValueError(f"Invalid point cloud {path}: {tuple(points.shape)}")
            arrays.append(normalize_points(points[:point_count, :3]))
        values = [arrays[index % len(arrays)] for index in range(batch)]
        return torch.stack(values).permute(0, 2, 1).contiguous().to(device)
    generator = torch.Generator().manual_seed(20260731 + batch)
    points = torch.randn((batch, point_count, 3), generator=generator)
    points = points / points.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    radius = torch.rand((batch, point_count, 1), generator=generator).pow(1.0 / 3.0)
    return (points * radius).permute(0, 2, 1).contiguous().to(device)


def load_model(source_dir, checkpoint, device):
    sys.path.insert(0, str(source_dir))
    model_module = importlib.import_module("pointnet2_cls_ssg")
    utils_module = importlib.import_module("pointnet2_utils")
    torch.manual_seed(20260731)
    model = model_module.get_model(40, normal_channel=False)
    metadata = {"loaded": False, "path": None, "sha256": None}
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload.get("model_state_dict", payload))
        metadata = {
            "loaded": True,
            "path": str(checkpoint),
            "sha256": _common.sha256(checkpoint),
        }
    return model.eval().to(device), utils_module, metadata


def run_model_with_capture(model, utils_module, model_input, fps):
    indices = []

    def capture(points, sample_count):
        result = fps(points, sample_count)
        indices.append(result.detach().clone())
        return result

    utils_module.farthest_point_sample = capture
    with torch.no_grad():
        output = model(model_input)[0]
    torch.npu.synchronize()
    return output, indices


def _prepare_case(config, batch):
    model_input = make_inputs(batch, 1024, config.device, config.point_clouds)
    config.utils_module.farthest_point_sample = framework_fps
    with torch.no_grad():
        first_xyz, _ = config.model.sa1(model_input, None)
    return BatchCase(
        batch,
        model_input,
        model_input.permute(0, 2, 1).contiguous(),
        first_xyz.permute(0, 2, 1).contiguous(),
    )


def _framework_stages(case):
    return framework_fps(case.first_stage_points, 512), framework_fps(
        case.second_stage_points, 128
    )


def _custom_stages(custom_fps, case):
    return custom_fps(case.first_stage_points, 512), custom_fps(
        case.second_stage_points, 128
    )


def _model_forward(config, case, fps):
    config.utils_module.farthest_point_sample = fps
    with torch.no_grad():
        return config.model(case.model_input)[0]


def _measure(config, case, reverse_order):
    calls = {
        "framework_stage": partial(_framework_stages, case),
        "custom_stage": partial(_custom_stages, config.custom_fps, case),
        "framework_model": partial(_model_forward, config, case, framework_fps),
        "custom_model": partial(_model_forward, config, case, config.custom_fps),
    }
    order = ("framework_stage", "custom_stage", "framework_model", "custom_model")
    if reverse_order:
        order = ("custom_stage", "framework_stage", "custom_model", "framework_model")
    measured = {}
    for name in order:
        call = calls.get(name)
        if call is None:
            raise KeyError(f"missing benchmark call: {name}")
        measured[name] = _common.timed(call, config.warmup, config.repeat)
    return Measurements(**measured)


def _mismatches(expected, actual):
    return [int((left != right).sum().cpu()) for left, right in zip(expected, actual)]


def _parity(config, case, measured):
    framework_first, framework_indices = run_model_with_capture(
        config.model, config.utils_module, case.model_input, framework_fps
    )
    framework_second, _ = run_model_with_capture(
        config.model, config.utils_module, case.model_input, framework_fps
    )
    custom_first, custom_indices = run_model_with_capture(
        config.model, config.utils_module, case.model_input, config.custom_fps
    )
    custom_second, _ = run_model_with_capture(
        config.model, config.utils_module, case.model_input, config.custom_fps
    )
    return {
        "fps_index_mismatches": _mismatches(
            measured.framework_stage.output, measured.custom_stage.output
        ),
        "full_model_fps_index_mismatches": _mismatches(
            framework_indices, custom_indices
        ),
        **_common.compare_outputs(
            measured.framework_model.output, measured.custom_model.output
        ),
        "framework_repeatability": _common.compare_outputs(
            framework_first, framework_second
        ),
        "custom_repeatability": _common.compare_outputs(custom_first, custom_second),
        "captured_framework_vs_custom": _common.compare_outputs(
            framework_first, custom_first
        ),
    }


def _timing_payload(measured):
    framework_stage = measured.framework_stage.median_ms
    custom_stage = measured.custom_stage.median_ms
    framework_model = measured.framework_model.median_ms
    custom_model = measured.custom_model.median_ms
    return {
        "framework_fps_ms": framework_stage,
        "custom_fps_ms": custom_stage,
        "fps_reduction_pct": (framework_stage - custom_stage) / framework_stage * 100.0,
        "framework_model_e2e_ms": framework_model,
        "custom_model_e2e_ms": custom_model,
        "model_e2e_reduction_pct": (framework_model - custom_model)
        / framework_model
        * 100.0,
        "framework_fps_share_of_e2e_pct": framework_stage / framework_model * 100.0,
        "framework_fps_samples_ms": measured.framework_stage.samples_ms,
        "custom_fps_samples_ms": measured.custom_stage.samples_ms,
        "framework_model_samples_ms": measured.framework_model.samples_ms,
        "custom_model_samples_ms": measured.custom_model.samples_ms,
    }


def benchmark_batch(config, batch, reverse_order=False):
    case = _prepare_case(config, batch)
    measured = _measure(config, case, reverse_order)
    result = {
        "batch": batch,
        "input_shape": list(case.model_input.shape),
        "fps_calls": [[batch, 1024, 512], [batch, 512, 128]],
        "parity": _parity(config, case, measured),
    }
    result.update(_timing_payload(measured))
    if config.labels:
        expected_labels = torch.tensor(
            [config.labels[index % len(config.labels)] for index in range(batch)],
            device=config.device,
        )
        result["baseline_sample_accuracy"] = float(
            (measured.framework_model.output.argmax(-1) == expected_labels)
            .to(torch.float32)
            .mean()
            .cpu()
        )
        result["custom_sample_accuracy"] = float(
            (measured.custom_model.output.argmax(-1) == expected_labels)
            .to(torch.float32)
            .mean()
            .cpu()
        )
    return result


LATENCY_KEYS = (
    "framework_fps_ms",
    "custom_fps_ms",
    "framework_model_e2e_ms",
    "custom_model_e2e_ms",
)
SUMMARY_OMIT = {
    "parity",
    "fps_reduction_pct",
    "model_e2e_reduction_pct",
    "framework_fps_share_of_e2e_pct",
}


def _base_summary(cases):
    result = {}
    for key, value in cases[0].items():
        if key.endswith("_samples_ms") or key in SUMMARY_OMIT:
            continue
        result[key] = value
    for key in LATENCY_KEYS:
        result[key] = float(np.mean([case[key] for case in cases]))
    return result


def _comparison_summary(cases, comparison):
    values = [case["parity"][comparison] for case in cases]
    return {
        "max_abs_error": max(value["max_abs_error"] for value in values),
        "mean_abs_error": max(value["mean_abs_error"] for value in values),
        "top1_agreement": min(value["top1_agreement"] for value in values),
    }


def _maximum_mismatches(cases, key):
    result = []
    for index in range(2):
        result.append(max(case["parity"][key][index] for case in cases))
    return result


def _parity_summary(cases):
    result = {
        "fps_index_mismatches": _maximum_mismatches(cases, "fps_index_mismatches"),
        "full_model_fps_index_mismatches": _maximum_mismatches(
            cases, "full_model_fps_index_mismatches"
        ),
        "max_abs_error": max(case["parity"]["max_abs_error"] for case in cases),
        "mean_abs_error": max(case["parity"]["mean_abs_error"] for case in cases),
        "top1_agreement": min(case["parity"]["top1_agreement"] for case in cases),
    }
    for comparison in (
        "framework_repeatability",
        "custom_repeatability",
        "captured_framework_vs_custom",
    ):
        result[comparison] = _comparison_summary(cases, comparison)
    return result


def _add_reductions(summary):
    summary["fps_reduction_pct"] = (
        (summary["framework_fps_ms"] - summary["custom_fps_ms"])
        / summary["framework_fps_ms"]
        * 100.0
    )
    summary["model_e2e_reduction_pct"] = (
        (summary["framework_model_e2e_ms"] - summary["custom_model_e2e_ms"])
        / summary["framework_model_e2e_ms"]
        * 100.0
    )
    summary["framework_fps_share_of_e2e_pct"] = (
        summary["framework_fps_ms"] / summary["framework_model_e2e_ms"] * 100.0
    )


def summarize_trials(all_trials):
    results = []
    for case_index in range(len(all_trials[0])):
        cases = [trial[case_index] for trial in all_trials]
        summary = _base_summary(cases)
        _add_reductions(summary)
        summary["parity"] = _parity_summary(cases)
        summary["trials"] = cases
        results.append(summary)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointnet2-source", type=Path, required=True)
    parser.add_argument("--source-commit", default="unknown")
    parser.add_argument("--operator-build-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--point-clouds", type=Path, nargs="*")
    parser.add_argument("--labels", type=int, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()
    if args.labels and len(args.labels) != len(args.point_clouds or []):
        parser.error("--labels must have one entry per --point-clouds file")
    return args


def prepare_benchmark(args):
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    model, utils_module, checkpoint = load_model(
        args.pointnet2_source, args.checkpoint, device
    )
    config = BenchmarkConfig(
        model,
        utils_module,
        FarthestPointSamplingOperator(args.operator_build_dir, device),
        args.warmup,
        args.repeat,
        device,
        args.point_clouds,
        args.labels,
    )
    return BenchmarkSetup(config, checkpoint)


def build_trials(config, batches, trials):
    results = []
    for trial_index in range(trials):
        trial = []
        for batch in batches:
            trial.append(
                benchmark_batch(config, batch, reverse_order=trial_index % 2 == 1)
            )
        results.append(trial)
    return results


def _point_cloud_metadata(paths):
    result = []
    for path in paths or []:
        result.append({"path": str(path), "sha256": _common.sha256(path)})
    return result


def build_payload(args, setup, results):
    return {
        "date": "2026-07-31",
        "device": torch.npu.get_device_name(setup.config.device),
        "model": "PointNet++ SSG classification",
        "model_source": str(args.pointnet2_source),
        "model_source_commit": args.source_commit,
        "checkpoint": setup.checkpoint,
        "input": "normalized point-cloud files"
        if args.point_clouds
        else "deterministic normalized synthetic point clouds",
        "point_clouds": _point_cloud_metadata(args.point_clouds),
        "labels": args.labels,
        "baseline": "resident Torch NPU deterministic FPS inside both Set Abstraction layers",
        "custom": "FarthestPointSamplingFused inside both Set Abstraction layers",
        "scope": "complete PointNet++ SSG forward including grouping, MLPs, pooling, and classifier",
        "warmup": args.warmup,
        "repeat": args.repeat,
        "trials": args.trials,
        "results": results,
    }


def main():
    args = parse_args()
    setup = prepare_benchmark(args)
    trials = build_trials(setup.config, args.batches, args.trials)
    payload = build_payload(args, setup, summarize_trials(trials))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
