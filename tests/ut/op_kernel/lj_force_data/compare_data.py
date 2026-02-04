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
import numpy as np

def compare_results(dtype_str="float32", rtol=1e-3, atol=1e-3):
    """比较算子输出与参考结果"""
    np_dtype = np.float32 if dtype_str == "float32" else np.float16

    # 读取参考结果
    forces_golden = np.fromfile(f"{dtype_str}_forces_golden_lj_force.bin", dtype=np_dtype)
    energy_golden = np.fromfile(f"{dtype_str}_energy_golden_lj_force.bin", dtype=np.float32)

    # 读取算子输出
    forces_output = np.fromfile(f"{dtype_str}_forces_output_lj_force.bin", dtype=np_dtype)
    energy_output = np.fromfile(f"{dtype_str}_energy_output_lj_force.bin", dtype=np.float32)

    # 比较力
    forces_match = np.allclose(forces_output, forces_golden, rtol=rtol, atol=atol)
    if not forces_match:
        max_diff = np.max(np.abs(forces_output - forces_golden))
        print(f"Forces mismatch! Max diff: {max_diff}")
    else:
        print("Forces match!")

    # 比较能量
    energy_match = np.allclose(energy_output, energy_golden, rtol=rtol, atol=atol)
    if not energy_match:
        diff = np.abs(energy_output[0] - energy_golden[0])
        print(f"Energy mismatch! Diff: {diff}")
    else:
        print("Energy match!")

    return forces_match and energy_match

if __name__ == "__main__":
    dtype_str = sys.argv[1] if len(sys.argv) > 1 else "float32"
    rtol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    atol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-3

    success = compare_results(dtype_str, rtol, atol)
    sys.exit(0 if success else 1)
