<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AutoformerInferenceAggregateFused API 参考

## 算子信息

| 项目 | 内容 |
|---|---|
| OpDef | `AutoformerInferenceAggregateFused` |
| ACLNN API | `aclnnAutoformerInferenceAggregateFused` |
| 输入 | `values: float32 [B,H,C,L]`，`correlation: float32 [B,H,C,L]` |
| 属性 | `top_k: int64` |
| 输出 | `output: float32 [B,H,C,L]` |
| Format | ND |
| Workspace | 当前 tiling 实现为 0 Byte |
| SoC 注册 | `ascend910b` |

## 参数约束与错误条件

Host tiling 要求两个输入均非空、rank-4、shape 完全一致，各维度可由 uint32 表示，`L <= 4096` 且可被 8 整除，`1 <= top_k <= min(16,L)`。不满足时返回 graph failure，ACLNN 层表现为非零状态码。

integration adapter 额外要求两个 tensor 均为 NPU resident、contiguous、FP32；B4/L336 因实测收益未达到路由阈值而使用 framework fallback。

## ACLNN 工程生成

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/autoformer_inference_aggregate_fused_msopgen.json -f aclnn \
    -c ai_core-ascend910b -out build/msopgen_autoformer_inference_aggregate_fused -lan cpp
bash tests/run_packaged_gate.sh /tmp/autoformer_inference_aggregate_fused_gate 6
```

`run_packaged_gate.sh` 会生成 ACLNN 工程、覆盖本目录的 operator-specific Host/Kernel 源码、构建自定义包并执行 device smoke。
