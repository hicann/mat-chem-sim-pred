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

from pid_residual_diagnostics_reference import (
    make_residual_diagnostics_case,
    residual_diagnostics,
    residual_diagnostics_cpu_loop,
)

def _time_ms(fn, iters: int) -> float:
    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - start) * 1000.0)
    return best


def benchmark_case(batch: int, n: int, max_lag: int, iters: int = 5) -> str:
    case = make_residual_diagnostics_case(batch=batch, n=n)
    vectorized_ms = _time_ms(lambda: residual_diagnostics(case.actual, case.predicted, max_lag=max_lag), iters)
    loop_ms = _time_ms(
        lambda: residual_diagnostics_cpu_loop(case.actual, case.predicted, max_lag=max_lag),
        max(1, min(3, iters)),
    )
    speedup = loop_ms / vectorized_ms if vectorized_ms > 0.0 else 0.0
    work_items = batch * n * max_lag
    return (
        f"B={batch} N={n} lag={max_lag} work_items={work_items} "
        f"loop_ms={loop_ms:.6f} vectorized_ms={vectorized_ms:.6f} vectorized_speedup={speedup:.2f}x"
    )

def main() -> None:
    cases = (
        (64, 1024, 16, 10),
        (128, 4096, 32, 5),
        (256, 4096, 32, 3),
        (512, 8192, 64, 2),
    )
    for args in cases:
        print(benchmark_case(*args))


if __name__ == "__main__":
    main()
