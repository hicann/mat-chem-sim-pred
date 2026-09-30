# Model E2E Reproduction

Train the maintained two-layer AGNN Cora checkpoint with
`train_agnn_cora.py`, build this package in Release mode for Ascend910B3, then
run `benchmark_agnn_cora_e2e.py` with copies `1 4 16`, three warmups, and ten
repeats. The benchmark compares official PyG, exact resident NPU, and custom
paths and selects the fastest correct native baseline per scale.

The checked-in JSON files contain checkpoint hash, task accuracy, component
latency/error, E2E latency/error, prediction agreement, and complete-model
profiler attribution. Dataset and checkpoint binaries are intentionally not
stored in the source repository.

`benchmark_agnn_cora_cpu_npu_e2e.py` is the value-gate benchmark. It must run on
the NPU host and records:

- exact complete-model CPU PyG latency at 1/4/8/16 threads;
- official PyG NPU and custom resident NPU latency;
- static-topology host node features through H2D, complete NPU model, and D2H
  logits;
- CPU/official/custom logits, predictions, and Cora test accuracy.

The resulting node202 artifact is
`agnn_cora_cpu_npu_e2e_node202_20260902.json`. The static topology and model
weights are transferred once and remain resident; this boundary is explicit in
the JSON and must not be described as including dataset or checkpoint loading.
