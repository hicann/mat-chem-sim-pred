<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshBucketSort 上游去重检查

检查基线：`cann/mat-chem-sim-pred` 的 upstream/master，commit `fef85c9be08c70dfb65c2254d37a3791edf9c625`，检查日期 2026-07-28。

结论：当前上游 TimeSeriesForecast 目录没有同时实现 Reformer LSH 稳定桶排序、原位置索引和逆置换构造的算子；与已有算子不存在接口或功能重复。

由于上游仍会变化，推送前和创建 MR 前均需刷新 upstream/master 并再次检查。
