<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AutoformerInferenceAggregateFused 上游去重检查

检查基线：`cann/mat-chem-sim-pred` 的 upstream/master，commit `fef85c9be08c70dfb65c2254d37a3791edf9c625`，检查日期 2026-07-28。

上游已有 `AutoCorrFusedAggregate` 接收 Q/K/V、自行计算 correlation，并按 head/channel 选择 lag。本算子接收 FFT 已生成的 correlation，在 H/C 均值后为每个 batch 选择一组共享 lag，严格对应 TSLib Autoformer 推理分支，输入合同和算法边界不同。

研发阶段的 `AutoformerLagWeightedAggregate` 是更窄的候选边界，只消费已选择的 delay/weight；它与本算子是同一调用点的替代方案，不能串联，收益也不能相加。本次不提交该窄算子。

由于上游仍会变化，推送前和创建 MR 前均需刷新 upstream/master 并再次检查。
