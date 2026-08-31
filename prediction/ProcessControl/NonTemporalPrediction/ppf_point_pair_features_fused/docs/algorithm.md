<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For every directed edge from source point `s` to target point `t`, the
operator computes the PyG PPF feature

`[||p_s-p_t||, angle(n_t,p_s-p_t), angle(n_s,p_s-p_t), angle(n_t,n_s)]`.

The kernel assigns edges to AIV workers, loads the two positions and normals,
and evaluates the distance and three clamped cosine angles in one pass. This
removes repeated gathers and the framework reduction used by the maintained
PPFConv geometry path. Graph construction and feature aggregation remain in
the caller so the operator does not change neighborhood semantics.

All geometry arithmetic is FP32. Zero-length vectors use the same epsilon
stabilization as the reference implementation.
