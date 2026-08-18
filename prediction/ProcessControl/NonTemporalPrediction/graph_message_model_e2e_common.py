#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Shared runtime helpers for graph message fusion model benchmarks."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch_npu


LOGGER = logging.getLogger("graph_message_benchmark")


@dataclass
class TimedResult:
    median_ms: float
    output: object
    samples_ms: list[float]


@dataclass
class PairedResult:
    first: TimedResult
    second: TimedResult


def configure_logging() -> None:
    if LOGGER.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def log_json(value, *, indent=None) -> None:
    configure_logging()
    LOGGER.info(json.dumps(value, indent=indent))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_host_library(build: Path, name: str):
    for directory in (build / "lib", build):
        for path in directory.glob("lib*_kernel_lib.so"):
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    candidates = (build / f"lib{name}_host.so", build / "lib" / f"lib{name}_host.so")
    host = next((path for path in candidates if path.exists()), None)
    if host is None:
        raise FileNotFoundError(f"cannot find host library for {name} under {build}")
    return ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)


def timed(function, warmup: int, repeat: int) -> TimedResult:
    result = None
    for _ in range(warmup):
        result = function()
    torch_npu.npu.synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        torch_npu.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0)
    return TimedResult(float(np.median(samples)), result, samples)


def timed_pair(first, second, warmup: int, repeat: int) -> PairedResult:
    samples = ([], [])
    outputs = [None, None]
    with torch.inference_mode():
        for index in range(warmup):
            ordered = (first, second) if index % 2 == 0 else (second, first)
            for function in ordered:
                function()
                torch_npu.npu.synchronize()
        for index in range(repeat):
            order = (0, 1) if index % 2 == 0 else (1, 0)
            for position in order:
                torch_npu.npu.synchronize()
                start = time.perf_counter()
                outputs[position] = (first, second)[position]()
                torch_npu.npu.synchronize()
                samples[position].append((time.perf_counter() - start) * 1_000.0)
    first_result = TimedResult(float(np.median(samples[0])), outputs[0], samples[0])
    second_result = TimedResult(float(np.median(samples[1])), outputs[1], samples[1])
    return PairedResult(first_result, second_result)


def profile_once(function, trace_dir: Path) -> None:
    experimental = torch_npu.profiler._ExperimentalConfig(
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


def _profile_rows(trace_dir: Path, filename: str):
    paths = sorted(trace_dir.glob(f"*/ASCEND_PROFILER_OUTPUT/{filename}"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {filename}, got {paths}")
    with paths[0].open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def parse_profile(trace_dir: Path, stage_types=None) -> dict:
    durations = defaultdict(float)
    counts = defaultdict(int)
    for row in _profile_rows(trace_dir, "kernel_details.csv"):
        kind = row["Type"].strip()
        durations[kind] += float(row["Duration(us)"].strip())
        counts[kind] += 1
    total = sum(durations.values())
    ranked = sorted(durations, key=durations.get, reverse=True)
    result = {
        "kernel_count": sum(counts.values()),
        "full_model_kernel_us": total,
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
    if stage_types is not None:
        stage = sum(durations[kind] for kind in stage_types)
        result.update(
            conservative_stage_kernel_us=stage,
            conservative_stage_hotspot_pct=100.0 * stage / total,
            stage_kernel_types=sorted(stage_types),
        )
    return result


def parse_operator_details(trace_dir: Path) -> list[dict]:
    return _profile_rows(trace_dir, "operator_details.csv")


def warmup(function, count: int = 3) -> None:
    with torch.inference_mode():
        for _ in range(count):
            function()
        torch_npu.npu.synchronize()
