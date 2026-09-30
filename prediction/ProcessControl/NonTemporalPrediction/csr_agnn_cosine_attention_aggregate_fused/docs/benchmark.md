# Benchmark Evidence

The source model is maintained PyG `examples/agnn.py`: `Linear -> ReLU ->
AGNNConv -> AGNNConv -> Linear`, evaluated on full Cora with normalized input
features. The exact CSR includes all 10,556 graph edges plus 2,708 AGNN
self-loops, with no truncation. Environment: Ascend 910B3 device 6, CANN
8.1.RC1.alpha001, PyTorch/torch_npu 2.5, PyG 2.9.0, FP32.

Checkpoint SHA-256 is
`552ccb51da860611cb0f8ed4f42595f8013c276f4493270881cf742187a32927`.
Baseline and custom inference retain 81.80% Cora test accuracy. Each latency is
the median of ten synchronized samples after three warmups; each table uses the
faster correct official PyG or exact resident-NPU baseline.

| Copies | Nodes / edges | Native stage (ms) | Custom (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,708 / 13,264 | 1.645 | 0.754 | 2.18x | 2.38e-7 |
| 4 | 10,832 / 53,056 | 2.868 | 1.503 | 1.91x | 2.38e-7 |
| 16 | 43,328 / 212,224 | 41.897 | 4.376 | 9.58x | 2.68e-7 |

| Copies | Native E2E (ms) | Custom E2E (ms) | Reduction | Prediction agreement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 3.611 | 1.359 | 62.37% | 100% |
| 4 | 5.552 | 2.881 | 48.11% | 99.999994% |
| 16 | 62.580 | 8.861 | 85.84% | 100% |

Complete resident-model Level1 profiler traces conservatively attribute
99.53%/99.86%/99.89% of NPU kernel time to the two AGNN stages. Linear
`MatMul`, ReLU, and framework movement kernels are excluded. Raw results are
`tests/model_e2e/agnn_cora_formal_20260802.json` and
`agnn_cora_hotspot_formal_20260802.json`.

## Same-host CPU/NPU gate

The 2026-09-02 rerun used the same node202 host, the same fixed checkpoint, the
exact 13,264-edge Cora layout including self-loops, and the complete maintained
two-layer model. CPU PyG was swept over 1/4/8/16 threads. NPU measurements are
synchronized FP32 no-grad host wall-clock medians after five warmups and 20
repeats.

| Copies | Best CPU (ms, threads) | Official NPU (ms) | Custom resident (ms) | Custom static-graph H2H (ms) | H2H vs CPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 12.839, 1 | 3.019 | 1.317 | 2.607 | 4.92x |
| 4 | 43.404, 1 | 5.462 | 2.896 | 6.416 | 6.76x |
| 16 | 87.636, 16 | 17.341 | 8.941 | 20.774 | 4.22x |

The strongest correct non-custom path is official PyG NPU at all three scales.
Resident custom reductions are `56.36%/46.99%/48.44%`. The host-to-host path
includes node-feature H2D, the complete model, and logits D2H; weights and the
static Cora topology remain resident, as they would for repeated inference.
Dataset/checkpoint loading and one-time model/topology transfer are excluded.
CPU/custom maximum absolute error is `4.77e-7`, prediction agreement is 100%,
and both paths retain 81.80% test accuracy.

Machine-readable evidence is
`tests/model_e2e/agnn_cora_cpu_npu_e2e_node202_20260902.json`. Complete-model
TorchAir fullgraph remains unestablished; the official NPU path emits a
`scatter_reduce` CPU-fallback warning, so no TorchAir performance claim is made.

The same-host CPU gate above supersedes the earlier resident-only table for
value ranking. The earlier table is retained as a stage/profiler reference;
the submission claim must use the CPU-comparable host-to-host numbers and the
`46.99%–56.36%` resident reduction range.
