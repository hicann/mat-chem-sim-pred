<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

The ACLNN host API accepts contiguous FP32 `position [N,3]`, contiguous int32
edge/interaction topology, and the official GemNet cache indices. It writes
three independently allocated FP32 arrays for `angle_cab`, `angle_abd`, and
`theta_cabd`.

Workspace is zero bytes. The caller owns all buffers, and execution uses the
supplied ACL stream without retaining pointers. Positive bounded lengths and
all index-domain relationships are validated before launch.

Unsupported dtypes, layouts, empty geometry, invalid topology, or training use
the resident official geometry implementation.
