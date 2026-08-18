<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshQkvGather 算法说明

## 问题定义

两个源 tensor 共享同一索引：

```text
sorted_query_key[r,i,:] = query_key[r,indices[r,i],:]
sorted_value[r,i,:] = value[r,indices[r,i],:]
```

输入源 shape 为 `[R,S,W]`，索引 shape 为 `[R,I]`，两个输出均为 `[R,I,W]`。

## 模型接入

TSLib `ReformerLayer` 最终调用 `LSHAttention.forward`。框架路径分别对 qk 和 v 执行 `batched_index_select`。本算子融合这两个 producer 相同、索引相同的 gather，但不承担 bucket sort 或后续 attention 计算。

## NPU 实现

kernel 将 `R * I` 个独立 gather job 分配给 vector core。每个 job 只读取一次 int64 索引，计算一次源地址，然后把 qk 和 value 的连续 `W` 个 FP32 元素分别搬运到输出。两个 local buffer 隔离存放数据，输出 storage 相互独立。

Host 用 uint64 计算 job 总数并在选择 block dim 时安全收窄；tiling 中每个维度写入前检查 uint32 范围；kernel 的全局 tensor 地址全部使用 uint64。

## 边界条件

- 两个源 tensor 为 contiguous、同 shape、rank-3、ND、FP32；
- 索引为非空 rank-2、ND、int64，行数等于源 tensor 的 R；
- 所有维度为正且不超过 uint32 上限，`W <= 16384` 且 `W % 8 == 0`；
- 调用方保证每个索引位于 `[0,S)`；
- 不满足静态合同时，由框架 adapter 回退到原实现。
