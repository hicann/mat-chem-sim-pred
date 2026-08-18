#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Load the standalone PointBallQueryFused libraries and verify an NPU result."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

sys.path.append(str(Path(__file__).resolve().parents[2]))
_common = importlib.import_module("pointnet2_geometry_model_e2e_common")


def load_operator():
    path = Path(__file__).parent / "model_e2e" / "benchmark_pointnet2_ball_query_e2e.py"
    module = _common.load_module(path, "point_ball_query_benchmark")
    return module.PointBallQueryOperator


def main():
    _common.configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("npu:0")
    torch.npu.set_device(device)
    points = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [10.0, 0.0, 0.0]]],
        device=device,
    )
    queries = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
    actual = load_operator()(args.build_dir, device)(1.1, 2, points, queries)
    expected = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.int64, device=device)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    _common.LOGGER.info("PointBallQueryFused standalone NPU smoke: PASS")


if __name__ == "__main__":
    main()
