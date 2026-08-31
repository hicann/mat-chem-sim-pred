<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

The operator implements the geometry half of the official GemNet quadruplet
path. From resident edge and interaction topology it computes the two cached
bond-angle arrays `angle_cab` and `angle_abd`, followed by the projected
torsion `theta_cabd` for each quadruplet.

The AIV kernel gathers the required point coordinates through the official
cache indices and evaluates each geometry result directly into its final
array. This avoids the framework's repeated gather, vector, cross-product, and
indexing sequence. Topology enumeration remains a separate contract and is not
silently fused into this operator.
