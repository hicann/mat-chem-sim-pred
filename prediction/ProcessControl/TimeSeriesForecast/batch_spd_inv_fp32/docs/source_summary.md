# BatchSpdInvFp32 — summary

**What:** Batched small SPD matrix inverse (fp32) on Ascend 910B. The missing
primitive behind Koopa's DMD step.

**Why:** Koopa's `KPLayer` fits the Koopman matrix via `torch.linalg.lstsq`, which is
unsupported on the NPU and falls back to CPU. TorchAir cannot compile this
unsupported path into an NPU full graph. The min-norm closed form
`K = x^T (x x^T)^-1 y` has
two trivial matmuls (native bmm, ~0.04 ms) and one tiny m×m Gram inverse — the only
piece that forces the CPU fallback. This op provides that inverse on-device.

**How:** Pure-Vector LDL^T factorization + triangular inverse (no sqrt, no Cube, no
cross-core sync). Batches spread across AIV cores; each batch DMA'd into UB, all
scalar work local, result DMA'd back (required for multi-core cache coherency).

**Results (Ascend 910B3 环境 / 910B3, CANN 8.1.RC1):**
- Correctness: G^-1 vs `torch.linalg.inv` < 1e-7; DMD K vs lstsq < 7e-8; residual < 1e-6.
- Isolated DMD (`B=32,m=7,E=128`): **4341.75 ms CPU fallback -> 0.339 ms custom NPU, 12823.56x**, max diff `6.80e-08`.
- Koopa forecast E2E (`seq=336,pred=96,B=32`): **5300.64 ms CPU fallback -> 8.80 ms custom NPU, 602.19x**, max diff `2.38e-07`.

**Class:** unsupported -> supported. Status: experimental.
