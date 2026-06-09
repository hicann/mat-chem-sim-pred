# AGENTS.md — mat-chem-sim-pred

**Monorepo of Ascend C operators** for Huawei NPUs (Atlas A2/A3, Ascend910B1).  
Each operator under `simulation/AI4MD/` builds independently — no top-level `CMakeLists.txt`.

---

## Prerequisites

- **CANN >= 7.0** with `ASCEND_CANN_PACKAGE_PATH` or `ASCEND_TOOLKIT_HOME` set
- **CMake >= 3.14**, Python 3, GTest (for UTs), FFTW3 (for PME), PyTorch (for Python tests)

---

## Build & Test

```bash
cd simulation/AI4MD/<op>
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
make -j$(nproc)
```

| What | Command |
|------|---------|
| Quick build (LJ only) | `./build.sh` / `./build.sh debug` / `./build.sh clean` |
| Run C++ UT | `cd build && ctest --output-on-failure` |
| Single Python test | `pytest tests/test_<op>.py::ClassName::test_name -v` |
| All Python tests | `pytest tests/ -v` |
| Benchmark | `python tests/benchmark_<op>.py` |
| Pre-commit | `pre-commit run --all-files` |

**Only LJ and DPD have test suites.** PME, GAFF2, velocity-verlet, SHAKE have no `tests/` directory.

---

## Operator Architecture

```
<op>/
  op_kernel/   — Device-side kernel (Ascend C, __aicore__, __gm__)
  op_host/     — Host launch API + param defs (aclnn* style)
  tests/       — C++ UT (GTest) + Python tests (pytest, NumPy ref)
  build.sh     — Quick build script (LJ only)
```

Tolerance: relative error < 1e-3 (force), < 1e-4 (energy), < 1e-6 (bond length).

---

## Style

- **C++**: Google-based, **4-space indent**, **120 col limit**, **Allman braces on functions**, `SortIncludes: false`
- **Python**: ruff check + ruff format (via pre-commit)
- **Codespell**: skips `*.py *.cpp *.hpp *.c *.h`; custom word list for CANN terms (see `.pre-commit-config.yaml`)

---

## Repo-specific Notes

- **Build artifacts**: `lib<op>_kernel.so` + `lib<op>_host.so` (or `.a`)
- **Codegen (GAFF2 only)**: `op_host/gen_inc.py` extracts `.ascend.kernel` ELF section into a `.inc` header. Other operators' `.inc` files are pre-generated and checked in.
- **DPD is the most featureful operator**: has `op_proto/`, `python/`, `fusion_result.json`, `test_simple.cpp`
- **Planned (README only, no code)**: `prediction/`, `simulation/MaterialPropertyPrediction/`, `simulation/AI4PDE/`
- **New operators**: follow `template/` (algorithm.md, operator_example.md, test_architecture.md)
- **Docs & PRs in Chinese** — PR template at `.gitcode/PULL_REQUEST_TEMPLATE.md`
- **License**: Apache 2.0; source files use CANN Open Software License v2.0 header
