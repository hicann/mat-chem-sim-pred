<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AutoformerInferenceAggregateFused 算子

`AutoformerInferenceAggregateFused` 融合 Autoformer 推理阶段的相关性均值、TopK 延迟选择、softmax 权重和循环位移聚合，替代原模型中多个小算子与循环组成的路径。

## 模型接入点

TSLib 调用点为 `layers/AutoCorrelation.py::AutoCorrelation.time_delay_agg_inference`。输入相关性已经由上游 FFT 路径生成，本算子只替换 mean-reduce、TopK、softmax、circular gather 和 weighted sum，不重复计算 Q/K correlation。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|---|---|---|---|
| `values` | float32 | `[B,H,C,L]` | 待聚合的 value |
| `correlation` | float32 | `[B,H,C,L]` | FFT 路径生成的相关性 |
| `top_k` | int64 属性 | 标量 | 每个 batch 共享的延迟数量 |
| `output` | float32 | `[B,H,C,L]` | 周期聚合结果 |

支持 contiguous、同 shape、非空 rank-4 ND FP32 输入；所有维度为正且可由 uint32 表示；`L <= 4096`、`L % 8 == 0`，`1 <= top_k <= min(16,L)`。不支持的 dtype、layout 或 shape 由 adapter 回退。B4/L336 虽满足功能合同，但实测收益仅 1.044x，因此也主动回退。

## 构建与测试

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/autoformer_inference_aggregate_fused_msopgen.json -f aclnn \
    -c ai_core-ascend910b -out build/msopgen_autoformer_inference_aggregate_fused -lan cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j2
python -m unittest discover -s tests -p "test_*.py" -v
python -m compileall -q integration reference tests
bash tests/run_packaged_gate.sh /tmp/autoformer_inference_aggregate_fused_gate 6
```

已在 Node202、CANN 8.1 RC1、Ascend 910B3 上完成 msopgen、独立构建、ACLNN smoke 和精度回读。当前本地 Python 套件共 8 项；官方 SCA、CLA 和仓库 CI 仅能在 MR 阶段执行。

## 性能价值

`B` 表示推理 batch size，`L` 表示输入序列长度。计时均为 NPU resident、同步 wall time，基线为等价向量化框架表达式。

| 范围 | 框架基线 | 自定义算子 | 延迟降低 | 加速比 |
|---|---:|---:|---:|---:|
| 算子组件，B32/L336 | 4.9690 ms | 0.8660 ms | 82.57% | 5.738x |
| 单算子替换完整模型，B32/L336 | 14.2116 ms | 8.4345 ms | 40.65% | 1.685x |
| ETTh1 全测试集 E2E，B32 | 720.53 ms | 606.95 ms | 15.76% | 1.187x |

E2E 覆盖 ETTh1 官方测试集全部 2,857 个滑窗、7 个特征、输入长度 96、预测长度 24，共 479,976 个预测值。原始尺度 MSE 变化 `3.91e-7`，最大预测差 `1.03e-4`。详细口径见 [性能说明](docs/benchmark.md) 和 [测试报告](docs/test_report.md)。
