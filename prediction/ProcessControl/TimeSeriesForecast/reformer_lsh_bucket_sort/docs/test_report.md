<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ReformerLshBucketSort 测试报告

## 已完成验证

| 检查项 | 结果 |
|---|---|
| Python CPU reference 与 adapter | PASS，7 项测试 |
| msopgen schema 生成 | PASS |
| CANN 8.1 RC1 / Ascend910B 独立构建 | PASS |
| Ascend 910B3 ACLNN smoke 与回读 | PASS |
| 非法合同 Host 拒绝 | PASS，返回非零状态 `561002` |
| 双 stream 各 20 次持续调用 | PASS，独立输出、0 Byte workspace |
| B4/B16/B32 × L96/L336 组件矩阵 | PASS |
| Reformer B32/L336 单算子模型消融 | PASS |
| ETTh1 全部 2,857 个测试窗口 | PASS |

reference 覆盖确定性、随机、最小边界和非法输入；adapter 覆盖 custom、常规 fallback 和 uint32 超界 fallback。组件输出、索引、完整模型预测均逐元素完全一致。

## 可复现命令

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q integration reference tests
bash tests/run_packaged_gate.sh /tmp/reformer_lsh_bucket_sort_gate 6
```

## 外部提交门禁

已基于 2026-07-28 最新 `upstream/master` 完成同名和同功能去重检查。GitCode SCA、CLA 和官方仓库 CI 只能在由 fork 发起 MR 后执行，当前不宣称已通过。
