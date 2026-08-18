<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshQkvGather 性能说明

## 环境与计时口径

- 设备：Ascend 910B3，研究环境物理 NPU 6；
- 软件：CANN 8.1 RC1，FP32 源数据与 int64 索引；
- Shape：B4/B16/B32，L96/L336，预测长度 24；
- 算子和单模型计时：3 次预热，9 次 NPU 同步计时取中位数；
- 输入常驻 NPU，不计进程启动及 H2D/D2H；
- 性能基线：相同语义下最快的两个 eager/TorchAir 兼容 framework gather；
- CPU reference 用于语义正确性，不作为 NPU 加速比分母；
- 自定义路径：msopgen 生成的 `GetWorkspaceSize + aclnn...`，不是研究 launcher。

`B` 是推理 batch size，`L` 是输入序列长度。

## 算子组件

| Shape | 框架基线 | ACLNN | 延迟降低 | 加速比 | 最大误差 |
|---|---:|---:|---:|---:|---:|
| B4/L96 | 0.5603 ms | 0.1855 ms | 66.89% | 3.020x | 0 |
| B16/L96 | 1.5668 ms | 0.3942 ms | 74.84% | 3.975x | 0 |
| B32/L96 | 3.0429 ms | 0.7066 ms | 76.78% | 4.306x | 0 |
| B4/L336 | 1.2297 ms | 0.3244 ms | 73.62% | 3.791x | 0 |
| B16/L336 | 4.2224 ms | 0.9372 ms | 77.80% | 4.505x | 0 |
| B32/L336 | 8.1059 ms | 1.7340 ms | 78.61% | 4.675x | 0 |

六档实测均启用自定义路径。

## 模型与 checkpoint

| 范围 | 框架基线 | 单算子替换 | 延迟降低 | 加速比 | 最大预测差 |
|---|---:|---:|---:|---:|---:|
| Reformer B32/L336 单 batch | 55.0994 ms | 48.8108 ms | 11.41% | 1.129x | 0 |
| ETTh1 全测试集，B32 | 1551.12 ms | 1337.96 ms | 13.74% | 1.159x | 0 |

checkpoint 闭环使用 ETTh1 官方测试 split：2,857 个重叠窗口、7 个特征、输入 96、预测 24，共 479,976 个预测值和 90 个推理 batch。两条路径严格加载同一 checkpoint，只独立替换本算子，并各自分配输出存储。

机器可读证据位于 `docs/evidence/`，分别记录六档原始样本、单模型消融、checkpoint 全量指标和多 stream 结果。
