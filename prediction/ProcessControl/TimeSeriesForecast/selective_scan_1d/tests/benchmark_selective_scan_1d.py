#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
FORECAST_DIR = THIS_DIR.parents[1]

_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "selective_scan_1d_reference",
    FORECAST_DIR / "common" / "selective_scan_1d_reference.py",
)
if _REFERENCE_SPEC is None or _REFERENCE_SPEC.loader is None:
    raise ImportError("failed to load selective_scan_1d_reference.py")
_REFERENCE_MODULE = importlib.util.module_from_spec(_REFERENCE_SPEC)
sys.modules["selective_scan_1d_reference"] = _REFERENCE_MODULE
_REFERENCE_SPEC.loader.exec_module(_REFERENCE_MODULE)
make_selective_scan_1d_case = _REFERENCE_MODULE.make_selective_scan_1d_case
selective_scan_1d = _REFERENCE_MODULE.selective_scan_1d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU reference benchmark for SelectiveScan1D.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--state", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.batch, args.length, args.dim, args.state, args.repeat) <= 0 or args.warmup < 0:
        raise ValueError("batch/length/dim/state/repeat must be positive and warmup must be non-negative")

    case = make_selective_scan_1d_case(args.batch, args.length, args.dim, args.state, args.seed)

    for _ in range(args.warmup):
        selective_scan_1d(case.u, case.delta, case.a, case.b, case.c, case.d)

    timings_ms: list[float] = []
    result = None
    for _ in range(args.repeat):
        start = time.perf_counter()
        result = selective_scan_1d(case.u, case.delta, case.a, case.b, case.c, case.d)
        timings_ms.append((time.perf_counter() - start) * 1000.0)

    assert result is not None
    nan_count = int(np.isnan(result).sum())
    print("name,shape,cpu_ref_mean_ms,cpu_ref_min_ms,cpu_ref_max_ms,nan_count,output_sum")
    print(
        "SelectiveScan1D,"
        f"{args.batch}x{args.length}x{args.dim}xN{args.state},"
        f"{np.mean(timings_ms):.6f},"
        f"{np.min(timings_ms):.6f},"
        f"{np.max(timings_ms):.6f},"
        f"{nan_count},"
        f"{float(np.sum(result, dtype=np.float64)):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
