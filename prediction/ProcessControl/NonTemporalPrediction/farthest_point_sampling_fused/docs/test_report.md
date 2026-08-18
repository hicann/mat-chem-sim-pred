<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# Test Report

NumPy tests cover deterministic selection and tie order. The ACL smoke validates
the expected three sampled indices on a one-dimensional point set. Original
three-scale NPU validation produced exact INT32 results.

Final verification on 2026-07-31 used a fresh Release build directory for
`Ascend910B3`; the Python reference test passed and the newly built ACL smoke
reported `farthest_point_sampling_fused PASSED`. The pretrained PointNet++ gate
also passed on two real ModelNet40 test samples: both in-model FPS outputs and
the complete classification output are exactly equal. Full-model reduction is
73.03%/60.74%/53.52% for B1/B4/B8.
