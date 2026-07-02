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

import sys
import time
from pathlib import Path


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

from pid_windowed_residual_diagnostics_reference import (
    make_windowed_residual_diagnostics_case,
    windowed_residual_diagnostics,
    windowed_residual_diagnostics_cpu_loop,
)


def _time_ms(fn, iters: int) -> float:
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        elapsed = (time.perf_counter() - start) * 1000.0
        best = min(best, elapsed)
    return best


def benchmark_case(batch: int, n: int, window_size: int, stride: int, max_lag: int, iters: int = 5) -> str:
    case = make_windowed_residual_diagnostics_case(batch=batch, n=n)
    num_windows = 1 + (n - window_size) // stride

    vectorized_ms = _time_ms(
        lambda: windowed_residual_diagnostics(
            case.actual,
            case.predicted,
            window_size=window_size,
            stride=stride,
            max_lag=max_lag,
        ),
        iters=iters,
    )
    loop_ms = _time_ms(
        lambda: windowed_residual_diagnostics_cpu_loop(
            case.actual,
            case.predicted,
            window_size=window_size,
            stride=stride,
            max_lag=max_lag,
        ),
        iters=max(1, min(3, iters)),
    )
    speedup = loop_ms / vectorized_ms if vectorized_ms > 0.0 else 0.0
    work_items = batch * num_windows * window_size * max_lag
    return (
        f"B={batch} N={n} W={num_windows} window={window_size} stride={stride} lag={max_lag} "
        f"work_items={work_items} loop_ms={loop_ms:.6f} vectorized_ms={vectorized_ms:.6f} "
        f"vectorized_speedup={speedup:.2f}x"
    )


def main() -> None:
    cases = (
        (64, 2048, 256, 128, 16, 10),
        (128, 4096, 512, 256, 32, 5),
        (256, 4096, 512, 256, 32, 3),
        (512, 8192, 1024, 512, 64, 2),
    )
    for args in cases:
        print(benchmark_case(*args))


if __name__ == "__main__":
    main()
