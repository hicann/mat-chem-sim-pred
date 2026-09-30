# Add CsrSuperGatDotAttentionAggregateFused

## What

Add an AscendC/ACLNN fusion for SuperGAT SD scaled-dot attention, row softmax,
and neighbor aggregation, with GE definition, NumPy oracle, guarded dispatcher,
tests, and model evidence.

## Why

PyG eager reaches an NPU-to-CPU `scatter_reduce` fallback. Against the faster
equivalent resident eager path, the no-grad same-weight Cora x4 model is 1.77x
faster (6.001 ms to 3.395 ms), with `7.45e-08` maximum output error. TorchAir
cannot lower the framework scatter-reduce graph, but the supplied custom-op GE
converter and OPP pass `torch.compile(..., fullgraph=True)` with `2.38e-7`
maximum error on Ascend910B3.

## Risk controls

Forward FP32/FP16/BF16 with FP32 accumulation; `H<=8`, `C<=64`, degree `<=512`.
Unsupported shapes, layouts, and dtypes fall back. The `torch.library` binding
keeps the custom forward in the training graph and uses a device-resident
padded-CSR autograd backward; it is not a separate AscendC backward kernel.
Host rejects aliases and undersized workspaces. Clean build, CTest, two-stream,
FP32 central-difference gradient, FP16/BF16 gradient, and 5,000-launch checks
pass. A reproducibly trained local Cora checkpoint (not an official upstream
checkpoint) preserves 74.30% accuracy and labels and completes a bound custom
optimizer step. The packaged GE path is FP32-only; direct ACLNN also covers
FP16/BF16.
