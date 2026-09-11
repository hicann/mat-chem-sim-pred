<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->
# CsrAppnpPropagateFused

`CsrAppnpPropagateFused` owns the complete fixed-iteration propagation phase of
maintained `torch_geometric.nn.APPNP`. One public call manages the ping-pong
workspace and launches the ordered propagation steps without exposing Python
loops or intermediate tensors. The MLP before APPNP remains native NPU code.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
python -m pytest tests/test_csr_appnp_propagate_fused.py -q
LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH" \
  ./build/examples/test_aclnn_csr_appnp_propagate_fused 0
```

On Ascend910B3/CANN 8.1.RC1.alpha001, five full Cora model shapes reduce
end-to-end latency by `18.05%-45.59%` against the faster of official PyG APPNP
and an equivalent resident torch_npu rewrite. Test accuracy remains `0.8270`,
prediction agreement is effectively 100%, and maximum model error is `5.72e-6`.
