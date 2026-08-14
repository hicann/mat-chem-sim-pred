<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshBucketSort API 参考

## 算子信息

| 项目 | 内容 |
|---|---|
| OpDef | `ReformerLshBucketSort` |
| ACLNN API | `aclnnReformerLshBucketSort` |
| 输入 | `keys: int64 [R,N]` |
| 属性 | `sequence_length: int64`，`total_buckets: int64` |
| 输出 | `sorted_keys`、`sticker`、`inverse`，均为 int64 `[R,N]` |
| Format | ND |
| Workspace | 当前 tiling 实现为 0 Byte |
| SoC 注册 | `ascend910b` |

## 参数约束与错误条件

Host tiling 会拒绝空 tensor、非 rank-2 输入、无法由 uint32 表示的行数或 `sequence_length`、超过 int32 上限的单行长度，以及不在 `[1,4096]` 内的 `total_buckets`。编码键范围属于调用方前置条件，CPU reference 覆盖其合法与非法用例。

integration adapter 仅对 NPU resident、contiguous、int64 且满足上述 shape 合同的输入调用 ACLNN；其余情况调用传入的 framework fallback。

## ACLNN 工程生成

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/reformer_lsh_bucket_sort_msopgen.json -f aclnn \
    -c ai_core-ascend910b -out build/msopgen_reformer_lsh_bucket_sort -lan cpp
bash tests/run_packaged_gate.sh /tmp/reformer_lsh_bucket_sort_gate 6
```

`run_packaged_gate.sh` 会生成 ACLNN 工程、覆盖本目录的 operator-specific Host/Kernel 源码、构建自定义包并执行 device smoke。
