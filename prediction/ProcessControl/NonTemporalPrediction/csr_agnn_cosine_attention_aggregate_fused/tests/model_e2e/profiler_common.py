#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shared Level1 torch_npu profiler utilities for candidate qualification."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch
import torch_npu


def profile_once(function, trace_dir: Path) -> None:
    config_type = getattr(torch_npu.profiler, "_ExperimentalConfig", None)
    if config_type is None:
        raise RuntimeError("This torch_npu version does not support Level1 profiling")
    experimental = config_type(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        data_simplification=True,
    )
    handler = torch_npu.profiler.tensorboard_trace_handler(
        str(trace_dir), analyse_flag=True, async_mode=False
    )
    with (
        torch.inference_mode(),
        torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(
                wait=0, warmup=0, active=1, repeat=1, skip_first=0
            ),
            on_trace_ready=handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
            with_modules=True,
            experimental_config=experimental,
        ) as profiler,
    ):
        function()
        torch_npu.npu.synchronize()
        profiler.step()


def parse_profile(trace_dir: Path, stage_types: set[str]) -> dict:
    paths = sorted(trace_dir.glob("*/ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one kernel CSV, got {paths}")
    durations = defaultdict(float)
    counts = defaultdict(int)
    with paths[0].open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            kind = row["Type"].strip()
            durations[kind] += float(row["Duration(us)"].strip())
            counts[kind] += 1
    total = sum(durations.values())
    stage = sum(durations[kind] for kind in stage_types)
    ranked = sorted(durations, key=durations.get, reverse=True)
    return {
        "kernel_count": sum(counts.values()),
        "full_model_kernel_us": total,
        "conservative_stage_kernel_us": stage,
        "conservative_stage_hotspot_pct": 100.0 * stage / total,
        "stage_kernel_types": sorted(stage_types),
        "by_type": [
            {
                "type": kind,
                "count": counts[kind],
                "duration_us": durations[kind],
                "full_model_ratio": durations[kind] / total,
            }
            for kind in ranked
        ],
    }


def warmup(function, count: int = 3) -> None:
    with torch.inference_mode():
        for _ in range(count):
            function()
        torch_npu.npu.synchronize()
