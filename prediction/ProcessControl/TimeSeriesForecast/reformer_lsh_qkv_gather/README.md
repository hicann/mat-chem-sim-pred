<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshQkvGather 算子

`ReformerLshQkvGather` 为 Reformer LSH Attention 融合执行 query-key 与 value 的成对 gather。两个输出共享同一份 int64 索引和一次索引遍历，减少重复 launch 与地址计算。

## 模型接入点

TSLib 调用链为 `layers/SelfAttention_Family.py::ReformerLayer` -> `reformer_pytorch.reformer_pytorch.LSHAttention.forward`。本算子替换成对的 `batched_index_select(qk, st)` 与 `batched_index_select(v, st)`，输入索引通常来自 `ReformerLshBucketSort`，但两个算子接口独立，可分别提交和使用。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|---|---|---|---|
| `query_key` | float32 | `[R,S,W]` | 合并后的 query-key 数据 |
| `value` | float32 | `[R,S,W]` | value 数据 |
| `indices` | int64 | `[R,I]` | 每行 gather 索引 |
| `sorted_query_key` | float32 | `[R,I,W]` | gather 后的 query-key |
| `sorted_value` | float32 | `[R,I,W]` | gather 后的 value |

支持两个 contiguous、同 shape、非空 rank-3 ND FP32 源 tensor 和非空 rank-2 ND int64 索引；所有维度为正且可由 uint32 表示；`W <= 16384` 且 `W % 8 == 0`。索引值范围 `[0,S)` 由调用方保证，不满足静态合同时 adapter 回退到框架实现。

## 构建与测试

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/reformer_lsh_qkv_gather_msopgen.json -f aclnn \
    -c ai_core-ascend910b -out build/msopgen_reformer_lsh_qkv_gather -lan cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j2
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q integration reference tests
bash tests/run_packaged_gate.sh /tmp/reformer_lsh_qkv_gather_gate 6
```

已在 Node202、CANN 8.1 RC1、Ascend 910B3 上完成 msopgen、独立构建、ACLNN smoke 和精度回读。当前本地 Python 套件共 7 项；官方 SCA、CLA 和仓库 CI 仅能在 MR 阶段执行。

## 性能价值

`B` 表示推理 batch size，`L` 表示输入序列长度。计时均为 NPU resident、同步 wall time，基线为相同语义下两个 framework gather。

| 范围 | 框架基线 | 自定义算子 | 延迟降低 | 加速比 |
|---|---:|---:|---:|---:|
| 算子组件，B32/L336 | 8.1059 ms | 1.7340 ms | 78.61% | 4.675x |
| 单算子替换完整模型，B32/L336 | 55.0994 ms | 48.8108 ms | 11.41% | 1.129x |
| ETTh1 全测试集 E2E，B32 | 1551.12 ms | 1337.96 ms | 13.74% | 1.159x |

E2E 覆盖 ETTh1 官方测试集全部 2,857 个滑窗、7 个特征、输入长度 96、预测长度 24，共 479,976 个预测值。两个 gather 输出和最终预测均逐元素完全一致。详细口径见 [性能说明](docs/benchmark.md) 和 [测试报告](docs/test_report.md)。
