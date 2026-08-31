<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For every resident DimeNet triplet `k -> j -> i`, the operator forms

`u = position[i] - position[j]` and
`v = position[j] - position[k]`,

then returns `atan2(||u x v||, u dot v)`. Triplet enumeration is intentionally
owned by `DimeNetTripletEnumerateFused`; this operator only replaces the
geometry stage.

One AIV work item owns one triplet. It gathers the three positions, evaluates
the cross product and dot product, and applies the bounded atan2 approximation
without materializing intermediate edge vectors. The approximation error is
covered by the reference and model-level tests.
