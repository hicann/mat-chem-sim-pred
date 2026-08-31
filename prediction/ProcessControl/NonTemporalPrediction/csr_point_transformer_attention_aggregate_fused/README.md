<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrPointTransformerAttentionAggregateFused

AscendC fusion of the default PyG `PointTransformerConv` message stage. It
forms the positional attention score per channel, applies destination-segment
softmax, adds the positional delta to neighbor values, and aggregates.

On eight held-out ModelNet10 clouds with 256 points and 16 neighbors, the full
same-weight no-grad layer/head path runs in 2.638 ms versus the strongest correct
framework path at 12.236 ms (4.64x). Max absolute output error is `5.96e-08`.
The machine-readable model record is
[`tests/model_e2e/model_e2e_results_formal.json`](tests/model_e2e/model_e2e_results_formal.json).

The optimized forward path supports contiguous FP32, FP16, or BF16 with FP32
accumulation, `C<=128`, int32 CSR, and destination degree at most 512. The
supplied `torch.library` binding supports training through a numerically checked
autograd recomputation, rather than a dedicated AscendC backward kernel.
Non-default `attn_nn`, unsupported shapes/dtypes/layouts, or invalid CSR use
the maintained PyG path.

```bash
cmake -S . -B build -DSOC_VERSION=Ascend910B1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pytest tests -q
```

The TorchAir converter audit was run separately on Ascend910B3 and recorded a
`2.38e-7` maximum FP32 error. The audit harness is intentionally not packaged
in this MR because it contains generated CANN OPP tooling and third-party
utility sources that are not required to build or run this operator. Direct
ACLNN supports FP32/FP16/BF16; the audited GE path was FP32-only.
