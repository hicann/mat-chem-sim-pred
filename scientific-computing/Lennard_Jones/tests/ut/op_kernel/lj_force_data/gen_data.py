#!/usr/bin/python3
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import os
import numpy as np

def gen_lj_test_data(num_atoms, epsilon=0.0103, sigma=3.4, cutoff=10.0, dtype_str="float32"):
    """生成 LJ 力场测试数据"""
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    # 生成简单立方晶格坐标
    spacing = 4.0
    n_side = int(np.ceil(num_atoms ** (1/3)))
    positions = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(positions) < num_atoms:
                    positions.append([i * spacing, j * spacing, k * spacing])

    positions = np.array(positions, dtype=np_dtype)
    positions.tofile(f"{dtype_str}_positions_lj_force.bin")

    # 保存参数
    params = np.array([epsilon, sigma, cutoff], dtype=np.float32)
    params.tofile(f"{dtype_str}_params_lj_force.bin")

    print(f"Generated test data for {num_atoms} atoms")

def gen_golden_data(num_atoms, epsilon=0.0103, sigma=3.4, cutoff=10.0, dtype_str="float32"):
    """生成 LJ 力场参考结果"""
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    # 读取坐标
    positions = np.fromfile(f"{dtype_str}_positions_lj_force.bin", dtype=np_dtype).reshape(-1, 3)

    # 计算力和能量
    forces = np.zeros_like(positions)
    total_energy = 0.0

    cutoff_sq = cutoff * cutoff
    sigma6 = sigma ** 6
    eps4 = 4.0 * epsilon
    eps24 = 24.0 * epsilon

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                continue
            r_vec = positions[i] - positions[j]
            r2 = np.sum(r_vec ** 2)

            if r2 < cutoff_sq and r2 > 1e-10:
                r2inv = 1.0 / r2
                r6inv = r2inv ** 3
                s6r6 = sigma6 * r6inv
                s12r12 = s6r6 * s6r6

                if i < j:
                    total_energy += eps4 * (s12r12 - s6r6)

                fscalar = eps24 * r2inv * (2.0 * s12r12 - s6r6)
                forces[i] += fscalar * r_vec

    forces.astype(np_dtype).tofile(f"{dtype_str}_forces_golden_lj_force.bin")
    np.array([total_energy], dtype=np.float32).tofile(f"{dtype_str}_energy_golden_lj_force.bin")

    print(f"Generated golden data: total_energy = {total_energy:.6f} eV")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gen_data.py <num_atoms> [epsilon] [sigma] [cutoff] [dtype]")
        sys.exit(1)

    os.system("rm -rf *.bin")

    num_atoms = int(sys.argv[1])
    epsilon = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0103
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 3.4
    cutoff = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0
    dtype_str = sys.argv[5] if len(sys.argv) > 5 else "float32"

    gen_lj_test_data(num_atoms, epsilon, sigma, cutoff, dtype_str)
    gen_golden_data(num_atoms, epsilon, sigma, cutoff, dtype_str)
