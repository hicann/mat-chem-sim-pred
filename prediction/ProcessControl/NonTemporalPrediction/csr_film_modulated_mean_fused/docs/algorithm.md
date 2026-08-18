<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For a target node `i` and incoming source node `j`, PyG `FiLMConv` computes:

```text
m_ij = gamma_i * projected_j + beta_i
m_ij = relu(m_ij)                  # optional
y_i = mean_j(m_ij)
```

The operator assigns CSR target rows across AI Vector cores. `beta_i` and
`gamma_i` are loaded into UB once per target. Each source projection is then
loaded, multiplied, shifted, optionally rectified, and accumulated in a local
channel buffer. One final scalar multiplication produces the mean, avoiding
the native `[E, C]` message tensor and `index_add` update.

The FiLM skip path and learned linear projections remain native NPU kernels.
This keeps the custom ABI reusable across input/output channel counts and
limits fusion to the measured hotspot. `apply_relu=false` exactly supports the
last layer from maintained `examples/film.py`.

The PPI benchmark preserves all graph edges. Its maximum fan-in is 286, below
the public 2,048 limit.
