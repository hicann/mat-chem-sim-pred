# Benchmark Evidence

## Model and Data

- Model source: PyG `GENConv`, `DeepGCNLayer`, and `SoftmaxAggregation` at
  repository revision `003c3cd8a10520567ceaeda619f0315e30ec2f66`.
- Model: eight DeeperGCN layers, hidden width 64, exact PyG module weights.
- Data: Cora, 2,708 nodes, 1,433 input features, seven classes, and 13,264
  directed edges after adding self-loops.
- Scale cases: one, two, and four disjoint graph copies, up to 10,832 nodes and
  53,056 edges.
- Checkpoint: 80 CPU training epochs, SHA-256
  `d194466af62f8e1666bf8c64d7f28159029b8802d90d7d36f365922445ad4442`.

The E2E benchmark preserves the trained node encoder, every GENConv root path
and MLP, DeepGCN normalization and residual paths, and the classification head.
Only `GENConv.propagate` message construction and feature-wise softmax
aggregation are replaced.

## Environment

- Ascend 910B3, device 6
- CANN 8.1.RC1.alpha001
- PyTorch 2.5 and torch_npu 2.5.1
- PyG 2.9.0
- FP32, model in evaluation mode
- TorchAir was unavailable

The exact PyG edge-index implementation invokes
`aten::scatter_reduce.two_out`, which torch_npu reports as unsupported and
routes through CPU fallback. A second resident NPU baseline implements the same
stable feature-wise softmax by grouping nodes into power-of-two degree buckets.
It pads only to each bucket boundary, uses `1.309x` as many source slots as CSR,
and avoids the CPU fallback. This replaces the historical global
maximum-degree padded baseline, which uses `34.50x` as many source slots on
Cora and is not a strong baseline.

The strongest resident NPU value baseline is selected independently per shape
from official PyG eager, global padding, and power-of-two degree buckets. The
degree-bucketed path won every measured shape. TorchAir was not measured because
the official path contains the unsupported scatter-reduce call; no fullgraph
claim is made.

## Historical Operator Stage

Ten warmups and 50 measured iterations were synchronized around each call;
the median is reported.

| Copies | Nodes | Edges | Native all-NPU (ms) | Custom (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,708 | 13,264 | 17.361 | 0.365 | 47.50x | 3.73e-8 |
| 2 | 5,416 | 26,528 | 40.744 | 0.570 | 71.47x | 3.73e-8 |
| 4 | 10,832 | 53,056 | 70.835 | 0.910 | 77.88x | 3.73e-8 |

The following stage-only table is retained to reproduce the old experiment; it
must not be used as the current value claim because its baseline uses global
maximum-degree padding.

## Same-Host Complete-Model Value Gate

The CPU path calls the official PyG model with CPU inputs and returns CPU
logits. Intra-op thread counts `1/4/8/16` are scanned and the fastest is used.
The resident paths keep model, graph, input, and output on NPU. The host-to-host
paths start with CPU features, include H2D, the complete eight-layer model, and
D2H logits; static topology and model weights remain resident. Each formal
number is the median of ten iterations after three warmups with synchronization
around every NPU forward.

| Copies | Fastest CPU (ms, threads) | Strong NPU bucketed (ms) | Custom resident (ms) | Resident reduction | Custom host-to-host (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 497.352 (1) | 20.593 | 4.570 | 77.81% | 5.924 |
| 2 | 573.072 (16) | 19.695 | 4.565 | 76.82% | 7.095 |
| 4 | 698.806 (16) | 18.452 | 7.378 | 60.02% | 10.595 |

Prediction agreement with the official CPU model is `100%`; both paths retain
`70.10%` Cora test accuracy and the maximum logit error is `3.34e-6`. The CPU
numbers were collected on node202 while several independent research jobs were
active, so they establish a large same-host value margin but should not be used
as standalone CPU microarchitecture results.

The benchmark preallocates one workspace and one caller-owned output per
shape/temperature before timing. This removes allocation from the measured
region and ensures that asynchronous launches do not depend on temporary output
storage.

## Historical Complete Model

Five warmups and 20 measured forward passes were synchronized; medians are
reported.

| Copies | Baseline (ms) | Custom (ms) | Speedup | Reduction | Model error | Test accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 146.093 | 4.606 | 31.72x | 96.85% | 1.91e-6 | 70.10% / 70.10% |
| 2 | 272.475 | 4.525 | 60.22x | 98.34% | 2.38e-6 | 70.10% / 70.10% |
| 4 | 566.428 | 7.273 | 77.88x | 98.72% | 2.38e-6 | 70.10% / 70.10% |

These historical percentages compare against the weak global padded baseline
and are superseded by the same-host table above.

Prediction agreement is 100% for one and two copies and 99.999994% for four.
The accuracy pair is baseline/custom. The exact PyG NPU fallback is
non-deterministic at model level; its one-copy comparison field is preserved in
the raw JSON but is not used as the correctness oracle. Correctness is judged
against the stable all-NPU semantic baseline and the NumPy reference tests.

## Raw Artifacts

- `tests/model_e2e/genconv_softmax_component_formal_20260802.json`
- `tests/model_e2e/genconv_deepergcn_e2e_formal_20260802.json`
- `tests/model_e2e/genconv_softmax_hotspot_20260802.json`
- `tests/model_e2e/genconv_same_host_value_gate_20260902.json`
- `tests/model_e2e/benchmark_component.py`
- `tests/model_e2e/benchmark_deepergcn_cora_e2e.py`
- `tests/model_e2e/benchmark_genconv_value_gate.py`
