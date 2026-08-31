<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test report

- NumPy oracle tests cover channelwise softmax, single-edge rows, bad shapes,
  and dtype fallback; framework binding tests cover training autograd.
- Clean Ascend 910B1 configure/build: passed.
- Component clean-runtime error: `7.15e-07`.
- Same-weight ModelNet10 model error: `5.96e-08`.
- Two-stream run with separate output/workspace: passed, error `2.38e-07`.
- Output alias, undersized workspace, and degree 513: rejected.
- 2026-08-14 hardening: ACLNN CTest, FP32/FP16/BF16 direct checks, and
  `torch.library` lifecycle/export/autograd checks passed.
- 2026-08-16 custom OPP: FP32 TorchAir `fullgraph=True` passed on Ascend910B3;
  output shape `[32,16]`, 128 edges, max error `2.38e-7`.

The operator does not retain pointers or allocate output memory. Caller buffers
must remain alive through completion of the supplied stream.
