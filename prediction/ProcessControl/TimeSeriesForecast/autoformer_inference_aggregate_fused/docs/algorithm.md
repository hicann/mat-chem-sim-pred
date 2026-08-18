<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AutoformerInferenceAggregateFused 算法说明

## 问题定义

对每个 batch，先在 head 和 channel 维上求相关性均值：

```text
score[b,t] = mean(correlation[b,:,:,t])
delay[b,:] = TopK(score[b,:], top_k)
weight[b,:] = Softmax(score[b,delay[b,:]])
output[b,h,c,t] =
    sum_k weight[b,k] * values[b,h,c,(t + delay[b,k]) mod L]
```

延迟和权重在同一 batch 内的所有 head/channel 间共享，与 TSLib `time_delay_agg_inference` 语义一致。

## 模型接入

TSLib Autoformer 先通过 FFT 获得 `correlation[B,H,C,L]`，再进入推理聚合。本算子以相关性和 value 为输入，融合后半段计算。它不接收 Q/K，也不替代 FFT correlation。

## NPU 实现

每个 batch 分配给一个 vector core。kernel 在 Local Memory 内完成：

1. 对 H/C 维相关性做均值归约；
2. 维护不超过 16 个候选的 TopK；
3. 对候选分数执行稳定 softmax；
4. 为每个 value 行生成循环偏移并完成加权累加。

融合避免 mean tensor、TopK 索引、softmax 权重和多次 circular gather 在 Global Memory 中物化。全局 tensor 地址和行数乘积使用 uint64 计算，Host 在写入 uint32 tiling 前验证维度范围。

## 边界与路由

- `values`、`correlation` 为 contiguous、同 shape、rank-4、ND、FP32；
- 所有维度为正且不超过 uint32 上限，`L <= 4096`；
- `L % 8 == 0`，`1 <= top_k <= min(16,L)`；
- Host 对非法合同返回失败；
- adapter 对不支持输入及低收益 B4/L336 路由到框架实现。
