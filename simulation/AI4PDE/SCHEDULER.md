<!--
Copyright (c) 2025 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# AI4PDE — Schedule & Milestones

> Based on [roadmap.md](../../roadmap.md) and [README.md](./README.md).
> Reference implementation patterns: `simulation/AI4MD/Lennard_Jones/`, `simulation/AI4MD/Dissipative_particle_dynamics/`.
> Target hardware: Ascend910B1, CANN >= 7.0.

---

## Phase 0 — Foundation (Weeks 1–4) ✅ DEVELOPED

**Goal**: Scaffold the AI4PDE directory structure, establish CI/build, and deliver a minimal "hello world" Ascend C operator as proof of infrastructure.

| Milestone | Deliverables | Status |
|-----------|-------------|--------|
| **M0.1: Directory scaffold** | `op_host/`, `op_kernel/`, `tests/`, `docs/`, `examples/` per template. Top-level `CMakeLists.txt` + per-component `CMakeLists.txt`. | ✅ DONE |
| **M0.2: Build verified** | `cmake .. -DSOC_VERSION=Ascend910B1` + `make` succeeds. `lib*_kernel.so` / `lib*_host.so` produced. | 🔧 TEST |
| **M0.3: CI pipeline** | Pre-commit hooks pass. `ctest` framework in place with one passing empty test. | 🔧 TEST |
| **M0.4: Docs scaffold** | `docs/algorithm.md`, `docs/api_reference.md` stubs. Update `AGENTS.md` to mark AI4PDE as "in development". | 🔧 INCOMPLETE |

**Built**: `build.sh`, top-level `CMakeLists.txt`, `pde_common/pde_math.h`, `pde_common/pde_types.h`.

---

## Phase 1 — PINN Inference Operator (Weeks 5–10) ✅ DEVELOPED

**Goal**: Full-stack Ascend C operator for fully-connected network forward + AutoDiff gradient computation.

| Milestone | Deliverables | Status |
|-----------|-------------|--------|
| **M1.1: Design & algorithm doc** | `docs/algorithm.md` — PINN math derivation, operator spec, tiling strategy, precision targets. | 🔧 INCOMPLETE |
| **M1.2: FC forward kernel** | Ascend C kernel: `matmul → bias → activation` loop for N hidden layers. Unit test with random weights vs NumPy. | ✅ DONE |
| **M1.3: AutoDiff kernel (1st order)** | Gradient of output w.r.t. input coordinates via manual backward pass. Supports d(u)/d(x), d(p)/d(x), d(T)/d(x). | ✅ DONE |
| **M1.4: Host API + Python test** | `aclnnPinnFC` host API, workspace size calculator. Python integration test comparing NPU output vs NumPy. | ✅ DONE |
| **M1.5: Benchmark & tuning** | `benchmark_pinn.py` — throughput vs problem size. | ✅ DONE |

**Built**: `op_kernel/pinn_kernel.h` (FC forward + backward), `op_kernel/pinn_kernel.cpp` (entry), `op_host/pinn_host.h/.cpp` (host API), `op_host/pinn_def.cpp`, `examples/test_aclnn_pinn.cpp`, `tests/test_pinn.py`, `tests/benchmark_pinn.py`, `tests/ut/op_kernel/test_pinn.cpp`.

---

## Phase 2 — FNO Inference Operator (Weeks 11–16) ✅ DEVELOPED

**Goal**: FFT → frequency-domain linear transform → IFFT operator chain on Ascend C.

| Milestone | Deliverables | Status |
|-----------|-------------|--------|
| **M2.1: Design doc** | `docs/algorithm.md` — FNO math, DFT implementation, tiling. | 🔧 INCOMPLETE |
| **M2.2: 1D DFT kernel** | Ascend C 1D DFT using explicit summation. Unit test vs NumPy rfft. | ✅ DONE |
| **M2.3: Spectral weight kernel** | Complex multiply-accumulate kernel: `R_phi * FFT(v)` with learnable weight tensor. | ✅ DONE |
| **M2.4: IDFT + lift/project** | IDFT back to spatial domain. Input lifting (P) and output projection (Q) linear layers. | ✅ DONE |
| **M2.5: Integration + Python test** | Full FNO forward pass. Compare with `neuraloperator` reference. | ✅ DONE |

**Built**: `op_kernel/fno_kernel.h` (DFT + spectral multiply + IDFT + lift/project), `op_kernel/fno_kernel.cpp`, `op_host/fno_host.h/.cpp`, `op_host/fno_def.cpp`, `examples/test_aclnn_fno.cpp`, `tests/test_fno.py`, `tests/benchmark_fno.py`, `tests/ut/op_kernel/test_fno.cpp`.

---

## Phase 3 — DeepONet & MeshGraphNet (Weeks 17–24) ✅ DEVELOPED

**Goal**: Two operator network operators supporting different PDE paradigms.

### DeepONet

| Milestone | Deliverables | Status |
|-----------|-------------|--------|
| M3.1: DeepONet design | Algorithm doc, Branch net + Trunk net architecture spec. | 🔧 INCOMPLETE |
| M3.2: Branch & Trunk kernel | FC network kernels for both networks. | ✅ DONE |
| M3.3: Inner product kernel | Batched `sum(b_k * t_k)` output. Host API + Python integration test. | ✅ DONE |

**Built**: `op_kernel/deeponet_kernel.h/.cpp`, `op_host/deeponet_host.h/.cpp`, `op_host/deeponet_def.cpp`, `examples/test_aclnn_deeponet.cpp`, `tests/test_deeponet.py`, `tests/benchmark_deeponet.py`, `tests/ut/op_kernel/test_deeponet.cpp`.

### MeshGraphNet

| Milestone | Deliverables | Status |
|-----------|-------------|--------|
| M3.4: MeshGraphNet design | Algorithm doc, graph construction, message-passing loop spec. | 🔧 INCOMPLETE |
| M3.5: Graph construction kernel | Build edge list from mesh connectivity on device. | ✅ DONE |
| M3.6: Message passing kernel | Node→Edge→Node message passing with MLP updates. Host API + Python test. | ✅ DONE |

**Built**: `op_kernel/mesh_graph_net_kernel.h/.cpp`, `op_host/mesh_graph_net_host.h/.cpp`, `op_host/mesh_graph_net_def.cpp`, `examples/test_aclnn_mesh_graph_net.cpp`, `tests/test_mesh_graph_net.py`, `tests/benchmark_mesh_graph_net.py`, `tests/ut/op_kernel/test_mesh_graph_net.cpp`.

---

## Phase 4 — Advanced Operators (Weeks 25–32) ⏳ PLANNED

**Goal**: Differentiable PDE layers + CFD ROM operators.

| Milestone | Deliverables | Deadline |
|-----------|-------------|----------|
| **M4.1: Differentiable FD stencil** | 1D/2D finite difference kernels as differentiable Ascend C ops. | Week 26 |
| **M4.2: Differentiable FV flux** | 1D finite volume flux as differentiable op. | Week 28 |
| **M4.3: POD decomposition kernel** | SVD-based POD on device: snapshot matrix → singular values/modes. | Week 30 |
| **M4.4: POD-RNN / AE-LSTM kernel** | Temporal prediction in latent space. RNN/GRU cell on Ascend C. | Week 32 |

---

## Phase 5 — Integration & Validation (Weeks 33–36) ⏳ PLANNED

**Goal**: End-to-end PDE surrogate benchmarks and documentation.

| Milestone | Deliverables | Deadline |
|-----------|-------------|----------|
| **M5.1: Darcy flow benchmark** | Full FNO/DeepONet comparison on Darcy2D. | Week 33 |
| **M5.2: Reaction-diffusion benchmark** | PINN vs FD on 2D Brusselator. | Week 34 |
| **M5.3: CFD lid-driven cavity** | MeshGraphNet vs OpenFOAM on 2D cavity flow. | Week 35 |
| **M5.4: Docs & publication** | Complete `docs/` for all operators. Update `README.md` and `roadmap.md`. | Week 36 |

---

## Developed Code Summary

```
AI4PDE/
├── SCHEDULER.md              # This file
├── build.sh                  # Build script (all, pinn, fno, deeponet, mesh)
├── CMakeLists.txt            # Top-level build
├── pde_common/
│   ├── pde_math.h            # tanh, sigmoid, relu, exp, cos, sin (no math.h)
│   └── pde_types.h           # Common tiling data structs
├── pinn/                     # Phase 1: PINN
├── fno/                      # Phase 2: FNO
├── deeponet/                 # Phase 3a: DeepONet
└── mesh_graph_net/           # Phase 3b: MeshGraphNet
```

**Total**: 40+ files across 4 complete operator implementations.

---

## Key Metrics

| Metric | Target |
|--------|--------|
| Numerical accuracy (force/field) | rel error < 1e-3 |
| Speedup vs CPU (batch inference) | ≥10× at industrial scale |
| Operators delivered (coded) | 4 of 6 (PINN, FNO, DeepONet, MeshGraphNet) |
| Benchmarks completed | ≥1 per operator (Python) |
| C++ UT coverage | Kernel-level tests per operator |
| Python integration tests | ≥1 per operator |
