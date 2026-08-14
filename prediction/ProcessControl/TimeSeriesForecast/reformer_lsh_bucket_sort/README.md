<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshBucketSort 算子

`ReformerLshBucketSort` 为 Reformer LSH Attention 提供稳定桶排序。算子将框架侧两次排序及逆置换构造合并为一个 Ascend C kernel，直接输出排序键、原位置索引和逆索引。

## 模型接入点

TSLib 调用链为 `layers/SelfAttention_Family.py::ReformerLayer` -> `reformer_pytorch.reformer_pytorch.LSHAttention.forward`。本算子替换 `sort_key_val(buckets_and_t, ticker)` 及随后逆置换构造，不改变 Reformer 其余计算。

## 输入输出

`keys[R,N]` 中每个键编码为 `bucket_id * sequence_length + position`。

| 名称 | 类型 | Shape | 说明 |
|---|---|---|---|
| `keys` | int64 | `[R,N]` | 编码后的桶键 |
| `sorted_keys` | int64 | `[R,N]` | 稳定排序后的键 |
| `sticker` | int64 | `[R,N]` | 排序位置对应的原位置 |
| `inverse` | int64 | `[R,N]` | 原位置对应的排序位置 |

支持非空 rank-2 ND 输入；行数和 `sequence_length` 必须为正且可由 uint32 表示，单行长度不超过 int32 上限；`1 <= total_buckets <= 4096`。键值范围 `[0, sequence_length * total_buckets)` 由调用方保证，adapter 对不支持的 dtype、layout 或 shape 回退到框架实现。

## 构建与测试

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/reformer_lsh_bucket_sort_msopgen.json -f aclnn \
    -c ai_core-ascend910b -out build/msopgen_reformer_lsh_bucket_sort -lan cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j2
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q integration reference tests
bash tests/run_packaged_gate.sh /tmp/reformer_lsh_bucket_sort_gate 6
```

已在 Node202、CANN 8.1 RC1、Ascend 910B3 上完成 msopgen、独立构建、ACLNN smoke 和精度回读。当前本地 Python 套件共 7 项；官方 SCA、CLA 和仓库 CI 仅能在 MR 阶段执行。

## 性能价值

`B` 表示推理 batch size，`L` 表示输入序列长度。计时均为 NPU resident、同步 wall time，基线为相同语义下最快的框架表达式。

| 范围 | 框架基线 | 自定义算子 | 延迟降低 | 加速比 |
|---|---:|---:|---:|---:|
| 算子组件，B32/L336 | 37.4029 ms | 0.5987 ms | 98.40% | 62.472x |
| 单算子替换完整模型，B32/L336 | 55.1197 ms | 17.5769 ms | 68.11% | 3.136x |
| ETTh1 全测试集 E2E，B32 | 1558.82 ms | 682.56 ms | 56.21% | 2.284x |

E2E 覆盖 ETTh1 官方测试集全部 2,857 个滑窗、7 个特征、输入长度 96、预测长度 24，共 479,976 个预测值。排序结果、索引及最终预测均逐元素完全一致。详细口径见 [性能说明](docs/benchmark.md) 和 [测试报告](docs/test_report.md)。
