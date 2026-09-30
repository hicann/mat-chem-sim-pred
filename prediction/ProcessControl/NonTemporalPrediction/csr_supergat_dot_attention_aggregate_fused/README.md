# CsrSuperGatDotAttentionAggregateFused

AscendC fusion of the SuperGAT `attention_type="SD"` message stage: scaled dot
score, LeakyReLU, destination-segment softmax, and weighted neighbor aggregate.
This is not the earlier GAT/GATv2 fusion; SuperGAT-SD uses a scaled projected
feature dot product and has no learned left/right attention vector.

The FP32 Cora x4 model benchmark is exact to `7.45e-08` max absolute error and
runs in 3.395 ms versus the strongest correct eager path at 6.001 ms (1.77x).
See [the consolidated evidence](../new_prediction_top5_evidence/README.md).

Supported shapes are `[N,H,C]` with `H<=8`, `C<=64`, non-empty int32 CSR, and
maximum destination degree 512. The ACLNN forward accepts contiguous FP32, FP16,
and BF16 inputs with FP32 softmax/message accumulation. The supplied
`torch.library` binding supports training through a numerically checked autograd
recomputation; it is not yet a dedicated AscendC backward kernel. The caller
owns output/workspace and supplies the stream; distinct in-flight calls require
distinct output/workspace buffers.

```bash
cmake -S . -B build -DSOC_VERSION=Ascend910B1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pytest tests -q
```

The directory contains the AscendC kernel, ACLNN host API, GE op definition,
NumPy oracle, dispatch guards, unit tests, and reproducible model evidence.

`ge_opp/run_torchair_probe.sh` builds and registers the custom OPP, then runs a
real `torch.compile(..., fullgraph=True)` graph through TorchAir. This FP32 GE
path passed on Ascend910B3 with `2.38e-7` maximum error. Direct ACLNN remains
the FP32/FP16/BF16 path; the packaged GE definition is currently FP32-only.
