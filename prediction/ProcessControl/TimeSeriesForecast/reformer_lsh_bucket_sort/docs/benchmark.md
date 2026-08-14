<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshBucketSort 性能说明

## 环境与计时口径

- 设备：Ascend 910B3，研究环境物理 NPU 6；
- 软件：CANN 8.1 RC1，FP32 模型数据与 int64 索引；
- Shape：B4/B16/B32，L96/L336，预测长度 24；
- 算子和单模型计时：3 次预热，9 次 NPU 同步计时取中位数；
- 输入常驻 NPU，不计进程启动及 H2D/D2H；
- 性能基线：相同语义下最快的 eager/TorchAir 兼容框架表达式；
- CPU reference 用于语义正确性，不作为 NPU 加速比分母；
- 自定义路径：msopgen 生成的 `GetWorkspaceSize + aclnn...`，不是研究 launcher。

`B` 是推理 batch size，`L` 是输入序列长度。

## 算子组件

| Shape | 框架基线 | ACLNN | 延迟降低 | 加速比 | 最大误差 |
|---|---:|---:|---:|---:|---:|
| B4/L96 | 1.4896 ms | 0.1854 ms | 87.55% | 8.034x | 0 |
| B16/L96 | 5.5665 ms | 0.1941 ms | 96.51% | 28.680x | 0 |
| B32/L96 | 10.7926 ms | 0.2741 ms | 97.46% | 39.378x | 0 |
| B4/L336 | 4.7998 ms | 0.2307 ms | 95.19% | 20.802x | 0 |
| B16/L336 | 18.6948 ms | 0.3623 ms | 98.06% | 51.604x | 0 |
| B32/L336 | 37.4029 ms | 0.5987 ms | 98.40% | 62.472x | 0 |

六档实测均启用自定义路径。

## 模型与 checkpoint

| 范围 | 框架基线 | 单算子替换 | 延迟降低 | 加速比 | 最大预测差 |
|---|---:|---:|---:|---:|---:|
| Reformer B32/L336 单 batch | 55.1197 ms | 17.5769 ms | 68.11% | 3.136x | 0 |
| ETTh1 全测试集，B32 | 1558.82 ms | 682.56 ms | 56.21% | 2.284x | 0 |

checkpoint 闭环使用 ETTh1 官方测试 split：2,857 个重叠窗口、7 个特征、输入 96、预测 24，共 479,976 个预测值和 90 个推理 batch。两条路径严格加载同一 checkpoint，只独立替换本算子，并各自分配输出存储。

机器可读证据位于 `docs/evidence/`，分别记录六档原始样本、单模型消融、checkpoint 全量指标和多 stream 结果。
