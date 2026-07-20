# SruScanFused -- fused SRU sequence scan (Ascend 910B3, fp32)

Fuses the per-timestep SRU recurrence (Lei & Zhang 2018, Simple Recurrent Units
for Highly Parallelizable Recurrence) into a single NPU kernel launch.

## Op
`SruScanFused(x:[B,L,IN], weight:[3IN,H], bias:[4H]) -> output:[B,L,H]`

Per batch b, timestep s:
```
x_tilde = W x_s                         // input projection
f = sigmoid(Wf x_s + v_f * c + b_f)     // forget gate with peephole on c
r = sigmoid(Wr x_s + v_r * c + b_r)     // reset gate with peephole on c
c = f * c + (1-f) * x_tilde             // cell update
h = r * tanh(c) + (1-r) * x_tilde       // highway output
```

Key features: peephole connections on cell state c, highway output mixing tanh(c)
with projected input. No hidden-to-hidden matmul in gates (input-only + peephole),
making the recurrence extremely launch-bound.

## Three-gate results

| Gate | Value | Threshold | Result |
|---|---|---|---|
| Correctness (probe, 4 shapes) | max_diff 2.24e-8 ~ 3.73e-8 | <=1e-3 | **PASS** |
| Correctness (fused encoder vs fp32 torch) | max_diff 1.04e-7 | <=1e-3 | **PASS** |
| Component (B32 L336 H64) | 128.06ms -> 0.78ms = **163.48x** | >1.5x | **PASS** |
| E2E (3-layer encoder) | 387.41ms -> 5.42ms = **71.46x** | >3x | **PASS** |

**Verdict: STRONG** (highest component speedup in portfolio)

## TorchAir 补充结论

主测试 `B32,L336,IN11,H64` 可完整成图，产生 7,069 个 FX 节点；TorchAir 为 `5.272 ms`，custom 为 `0.884 ms`，custom 仍快 `5.97x`，首次编译和运行耗时 `165.43 s`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
