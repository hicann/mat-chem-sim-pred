#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import time

def deeponet_numpy(branch_in, trunk_in, Wb, bb, Wt, bt):
    branch_code = np.tanh(branch_in @ Wb.T + bb)
    trunk_code = np.tanh(trunk_in @ Wt.T + bt)
    return branch_code @ trunk_code.T

def run_benchmark():
    configs = [(1, 10, 100), (8, 50, 500), (32, 100, 1000), (128, 200, 2000)]
    print(f"{'Batch':>6} {'Branch':>8} {'Query':>8} {'Time(ms)':>10}")
    print("-" * 36)

    for batch, branch_dim, query in configs:
        latent_dim = 64
        trunk_dim = 2
        np.random.seed(0)

        branch_in = np.random.randn(batch, branch_dim).astype(np.float32)
        trunk_in = np.random.randn(query, trunk_dim).astype(np.float32)
        Wb = np.random.randn(latent_dim, branch_dim).astype(np.float32)
        bb = np.random.randn(latent_dim).astype(np.float32)
        Wt = np.random.randn(latent_dim, trunk_dim).astype(np.float32)
        bt = np.random.randn(latent_dim).astype(np.float32)

        start = time.perf_counter()
        for _ in range(100):
            _ = deeponet_numpy(branch_in, trunk_in, Wb, bb, Wt, bt)
        elapsed = (time.perf_counter() - start) / 100
        print(f"{batch:>6} {branch_dim:>8} {query:>8} {elapsed*1000:>10.3f}")

if __name__ == "__main__":
    run_benchmark()
