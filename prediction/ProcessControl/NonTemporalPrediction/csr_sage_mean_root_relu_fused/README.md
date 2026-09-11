<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->
# CsrSageMeanRootReluFused

`CsrSageMeanRootReluFused` fuses the CSR neighbor mean, GraphSAGE root branch,
and ReLU used by maintained `torch_geometric.nn.SAGEConv`. The integration
keeps both learned Linear projections as native NPU operations. It moves the
neighbor Linear before the mean and folds its bias into the root projection,
which is algebraically exact even for empty neighbor rows.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
python -m pytest tests/test_csr_sage_mean_root_relu_fused.py -q
LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH" \
  ./build/examples/test_aclnn_csr_sage_mean_root_relu_fused 0
```

On Ascend910B3/CANN 8.1.RC1.alpha001, five Cora model shapes show
`1.61x-2.74x` fused-stage speedup and `20.21%-49.60%` full-model latency
reduction against the strongest correct resident baseline. Checkpoint test
accuracy remains `0.7120`; maximum model error is `4.77e-7`.
