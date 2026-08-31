<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API reference

`aclnnCsrPointTransformerAttentionAggregateFusedGetWorkspaceSize(N,E,C,max_degree)`
returns workspace bytes.

The launch API accepts `row_ptr`, `source_index`, `alpha_source`, `alpha_target`,
`value`, `delta`, preallocated `output`, the four shape scalars, `dtype`, workspace,
and the caller stream. All feature tensors are same-dtype FP32/FP16/BF16; CSR is int32.
`alpha_*` and `value` are `[N,C]`, `delta` is `[E,C]`, and output is `[N,C]`.
`dtype` is `0` (FP32), `1` (FP16), or `2` (BF16); softmax and message accumulation
use FP32. Limits are `C<=128` and `max_degree<=512`.

The call is asynchronous. Output/workspace cannot alias an input or each other,
and each in-flight call needs its own pair. Host validation failure is the
signal for the integration layer to run the maintained PyG path.
