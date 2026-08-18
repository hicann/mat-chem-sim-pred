#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Shared runtime helpers for PointNet++ geometry model benchmarks."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


LOGGER = logging.getLogger("pointnet2_geometry_benchmark")


@dataclass
class TimedResult:
    median_ms: float
    samples_ms: list[float]
    output: object


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


def timed(callable_, warmup: int, repeat: int) -> TimedResult:
    result = None
    for _ in range(warmup):
        result = callable_()
        torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        torch.npu.synchronize()
        begin = time.perf_counter()
        result = callable_()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - begin) * 1000.0)
    return TimedResult(float(np.median(samples)), samples, result)


def compare_outputs(expected, actual) -> dict[str, float]:
    difference = (expected - actual).abs()
    return {
        "max_abs_error": float(difference.max().cpu()),
        "mean_abs_error": float(difference.mean().cpu()),
        "top1_agreement": float(
            (expected.argmax(-1) == actual.argmax(-1)).to(torch.float32).mean().cpu()
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def load_host_library(build_dir: Path, name: str):
    directories = (build_dir / "lib", build_dir)
    kernels = []
    for directory in directories:
        kernels.extend(directory.glob("lib*_kernel_lib.so"))
    for path in sorted(set(kernels)):
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    candidates = [directory / f"lib{name}_host.so" for directory in directories]
    host_path = None
    for path in candidates:
        if path.exists():
            host_path = path
            break
    if host_path is None:
        raise FileNotFoundError(f"host library for {name} not found below {build_dir}")
    return ctypes.CDLL(str(host_path), mode=ctypes.RTLD_GLOBAL)
