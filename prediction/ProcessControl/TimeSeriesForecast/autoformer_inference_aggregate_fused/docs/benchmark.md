<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AutoformerInferenceAggregateFused 性能说明

## 环境与计时口径

- 设备：Ascend 910B3，研究环境物理 NPU 6；
- 软件：CANN 8.1 RC1，FP32；
- Shape：B4/B16/B32，L96/L336，H4，C16，预测长度 24；
- 算子和单模型计时：3 次预热，9 次 NPU 同步计时取中位数；
- 输入常驻 NPU，不计进程启动及 H2D/D2H；
- 性能基线：相同推理语义下最快的向量化 eager/TorchAir 兼容表达式；
- CPU reference 用于语义正确性，不作为 NPU 加速比分母；
- 自定义路径：msopgen 生成的 `GetWorkspaceSize + aclnn...`，不是研究 launcher。

`B` 是推理 batch size，`L` 是输入序列长度。

## 算子组件

| Shape | 框架基线 | ACLNN | 延迟降低 | 加速比 | 最大误差 |
|---|---:|---:|---:|---:|---:|
| B4/L96 | 0.4868 ms | 0.3147 ms | 35.36% | 1.547x | 2.38e-7 |
| B16/L96 | 0.9276 ms | 0.3101 ms | 66.57% | 2.991x | 4.77e-7 |
| B32/L96 | 1.5345 ms | 0.3074 ms | 79.96% | 4.991x | 3.58e-7 |
| B4/L336 | 0.8978 ms | 0.8597 ms | 4.25% | 1.044x | 2.38e-7 |
| B16/L336 | 2.5777 ms | 0.8644 ms | 66.47% | 2.982x | 4.77e-7 |
| B32/L336 | 4.9690 ms | 0.8660 ms | 82.57% | 5.738x | 3.58e-7 |

路由策略：B4/L336 回退到框架，其余五档启用自定义路径。

## 模型与 checkpoint

| 范围 | 框架基线 | 单算子替换 | 延迟降低 | 加速比 | 最大预测差 |
|---|---:|---:|---:|---:|---:|
| Autoformer B32/L336 单 batch | 14.2116 ms | 8.4345 ms | 40.65% | 1.685x | 5.66e-5 |
| ETTh1 全测试集，B32 | 720.53 ms | 606.95 ms | 15.76% | 1.187x | 1.03e-4 |

checkpoint 闭环使用 ETTh1 官方测试 split：2,857 个重叠窗口、7 个特征、输入 96、预测 24，共 479,976 个预测值和 90 个推理 batch。两条路径严格加载同一 checkpoint，只独立替换本算子，并各自分配输出存储。原始尺度 MSE 从 `12.4020006278` 变为 `12.4020010184`，绝对变化 `3.91e-7`。

机器可读证据位于 `docs/evidence/`，分别记录六档原始样本、单模型消融、checkpoint 全量指标和多 stream 结果。
