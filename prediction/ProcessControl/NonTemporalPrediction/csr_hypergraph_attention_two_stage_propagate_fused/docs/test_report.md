<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test report

- NumPy oracle covers two-stage attention reuse and CSR mapping validation;
  framework binding tests cover training autograd.
- Clean Ascend 910B1 kernel/host/op-definition build passed.
- Same-weight Cora hypergraph `C=8` model error: `2.98e-08`.
- Independent two-stream execution passed at `2.98e-08` error.
- Output alias, undersized composite workspace, and segment size 513 rejected.
- 2026-08-14 hardening: ACLNN CTest, FP32/FP16/BF16 direct checks, and
  `torch.library` lifecycle/export/autograd checks passed.

Both internal kernels are enqueued on the supplied stream. The caller owns all
buffers and must provide a distinct output/workspace pair for each in-flight
invocation; the host API performs no allocation and retains no pointers.
