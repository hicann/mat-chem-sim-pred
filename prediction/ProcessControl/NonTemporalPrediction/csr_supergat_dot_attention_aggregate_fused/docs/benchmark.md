# Benchmark

Recorded FP32 run (build target `Ascend910B1`; runtime attribution corrected in
the consolidated evidence), 5 warmups and 20 synchronized samples, Cora x4 disjoint
batch, `N=10832`, `E=42224`, `H=2`, `C=32`:

| Path | Median |
| --- | ---: |
| PyG SuperGAT eager | 8.079 ms |
| Equivalent resident eager | 6.001 ms |
| Custom model path | 3.395 ms |

Speedup against the faster correct framework path is 1.77x. Timing is under
`torch.no_grad()`. The native TorchAir graph still cannot lower
`aten.scatter_reduce.two`; the packaged custom-op converter and OPP pass FP32
`fullgraph=True` with `2.38e-7` maximum error.
Raw samples are in `../../new_prediction_top5_evidence/model_e2e_results_formal.json`.
