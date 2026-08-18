<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshQkvGather API 参考

## 算子信息

| 项目 | 内容 |
|---|---|
| OpDef | `ReformerLshQkvGather` |
| ACLNN API | `aclnnReformerLshQkvGather` |
| 输入 | `query_key: float32 [R,S,W]`，`value: float32 [R,S,W]`，`indices: int64 [R,I]` |
| 属性 | 无 |
| 输出 | `sorted_query_key: float32 [R,I,W]`，`sorted_value: float32 [R,I,W]` |
| Format | ND |
| Workspace | 当前 tiling 实现为 0 Byte |
| SoC 注册 | `ascend910b` |

## 参数约束与错误条件

Host tiling 要求两个源 tensor 非空、rank-3、shape 完全一致；索引 tensor 非空、rank-2 且行数一致；所有维度可由 uint32 表示；源宽度不超过 16384 且可被 8 整除。Host 不遍历 device 索引值，`[0,S)` 的动态范围由调用方保证并由 CPU reference 测试覆盖。

integration adapter 额外要求三项输入均为 NPU resident、contiguous，dtype 分别为 FP32、FP32 和 int64；其余情况调用 framework fallback。

## ACLNN 工程生成

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/reformer_lsh_qkv_gather_msopgen.json -f aclnn \
    -c ai_core-ascend910b -out build/msopgen_reformer_lsh_qkv_gather -lan cpp
bash tests/run_packaged_gate.sh /tmp/reformer_lsh_qkv_gather_gate 6
```

`run_packaged_gate.sh` 会生成 ACLNN 工程、覆盖本目录的 operator-specific Host/Kernel 源码、构建自定义包并执行 device smoke。
