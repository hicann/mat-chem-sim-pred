<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshBucketSort 算法说明

## 问题定义

输入 `keys[r,n] = bucket_id * sequence_length + position`。对每一行执行稳定升序排序，并同时输出排序置换及其逆置换：

```text
sorted_keys[r,j] = keys[r,sticker[r,j]]
inverse[r,sticker[r,j]] = j
```

稳定性要求相同桶键保持原始先后顺序，这是 Reformer LSH 分桶后正确还原 token 顺序的必要条件。

## 模型接入

TSLib 的 `ReformerLayer` 最终调用 `LSHAttention.forward`。框架路径使用 `sort_key_val` 生成排序结果，再额外构造 inverse。本算子一次遍历同时产生三项输出，替换范围只覆盖该排序阶段。

## NPU 实现

每个逻辑行分配给一个 vector core。kernel 在 Local Memory 中维护不超过 4096 个桶的计数和前缀位置，然后完成：

1. 清零桶计数；
2. 扫描该行键值并统计各桶数量；
3. 计算稳定写入起点；
4. 再次扫描输入，写出 `sorted_keys`、`sticker` 和 `inverse`。

行间互不依赖，无跨核共享状态；所有全局地址偏移使用 uint64 计算。相较两次框架排序及逆索引构造，该方案减少 kernel launch、中间 tensor 和重复遍历。

## 边界条件

- `keys` 为非空 rank-2、ND、int64 tensor；
- 行数和 `sequence_length` 为正且不超过 uint32 上限，单行长度不超过 int32 上限；
- `1 <= total_buckets <= 4096`；
- 调用方保证编码键位于合法范围；
- 不满足合同时，由框架 adapter 回退到原实现。
