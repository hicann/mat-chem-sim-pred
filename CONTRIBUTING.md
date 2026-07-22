<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# mat-chem-sim-pred 算子贡献指南

本指南面向向 [mat-chem-sim-pred](https://gitcode.com/cann/mat-chem-sim-pred) 仓库贡献 Ascend C 算子的开发者。内容基于 CANN 社区通用算子贡献规则，结合本仓库已有算子的实践编写。如本指南与 GitCode 页面提示、CLA 机器人指令或维护者要求冲突，以 GitCode 页面和上游维护者要求为准。

---

## 1 仓库定位与已有算子

mat-chem-sim-pred 是基于华为 CANN（Ascend C）计算框架的化工行业专用算子库，聚焦**科学计算**与**预测优化**两大方向，构建"模拟 → 数据 → 预测 → 设计 → 优化"的全链路化工 AI 算子体系。仓库整体采用 [Apache 2.0](https://www.apache.org/licenses/) 许可证；按 CANN 社区惯例，所有源文件版权声明头使用 CANN Open Software License Agreement Version 2.0 格式（详见第 7 节）。

### 1.1 仓库结构

```
mat-chem-sim-pred/
├── simulation/                    # 科学计算方向
│   ├── AI4MD/                     # 机器学习分子动力学（6 个算子）
│   ├── AI4PDE/                     # AI for PDE（3 个算子 + 公共模块）
│   └── MaterialPropertyPrediction/ # 材料性质预测（PyTorch 参考）
├── prediction/                     # 预测优化方向
│   ├── SmallData/                  # 小数据预测优化（PyTorch 参考）
│   └── ProcessControl/             # 工业过程控制
│       └── PIDModelFit/            # PID 模型辨识与整定（12 个算子）
├── template/                       # 算子贡献模板
│   ├── algorithm.md
│   ├── references.md
│   ├── operator_example.md
│   └── test_architecture.md
├── roadmap.md
├── FAQ.md
└── AGENTS.md
```

### 1.2 已有算子清单

贡献新算子前，建议先阅读已有算子代码作为参考。

**simulation/AI4MD — 机器学习分子动力学**（6 个，Ascend C 就绪）

| 算子目录 | 公开接口 | 功能 |
|----------|----------|------|
| `Lennard_Jones/` | LJ 力场 | Lennard-Jones 势能计算 |
| `GAFF2/` | GAFF2 力场 | GAFF2 分子力场 |
| `PME/` | PME 静电 | 粒子网格 Ewald 静电求和 |
| `SHAKE/` | SHAKE 约束 | 键长约束算法 |
| `velocity-verlet/` | Velocity Verlet 积分 | 分子动力学积分器 |
| `Dissipative_particle_dynamics/` | DPD | 耗散粒子动力学 |

**simulation/AI4PDE — AI for PDE**（3 个，Ascend C 就绪）

| 算子目录 | 功能 |
|----------|------|
| `pinn/` | 物理信息神经网络推理算子 |
| `fno/` | 傅里叶神经算子推理算子 |
| `deeponet/` | DeepONet 推理算子 |

**prediction/ProcessControl/PIDModelFit — PID 模型辨识与整定**（12 个，Ascend C 就绪）

| 阶段 | 算子 | 公开接口 |
|------|------|----------|
| 模型辨识 | `pid_fopdt_basis_gemm_fit` | `aclnnPidFopdtBasisGemmFit` |
| 模型辨识 | `pid_ipdt_basis_gemm_fit` | `aclnnPidIpdtBasisGemmFit` |
| 模型辨识 | `pid_sopdt_basis_gemm_fit` | `aclnnPidSopdtBasisGemmFit` |
| 模型诊断 | `pid_residual_diagnostics` | 残差诊断 |
| 模型诊断 | `pid_windowed_residual_diagnostics` | 窗口残差诊断 |
| PID 参数生成 | `pid_tuning_rule_batch` | Ziegler-Nichols / IMC / Cohen-Coon |
| 候选仿真 | `pid_fopdt_batch_rollout_score` | FOPDT 闭环仿真评分 |
| 候选仿真 | `pid_ipdt_batch_rollout_score` | IPDT 闭环仿真评分 |
| 候选仿真 | `pid_sopdt_batch_rollout_score` | SOPDT 闭环仿真评分 |
| 特征提取 | `pid_step_response_features` | 阶跃响应特征提取 |
| 性能评估 | `pid_control_performance_metrics` | 控制性能指标 |
| 性能评估 | `pid_process_capability_metrics` | 过程能力指标 |

---

## 2 提交流程

### 2.1 Fork 与克隆

1. 登录 GitCode，打开 https://gitcode.com/cann/mat-chem-sim-pred ，点击 Fork 到个人账号。
2. 本地克隆 fork 并添加上游远程：

```bash
git clone https://gitcode.com/<your-account>/mat-chem-sim-pred.git
cd mat-chem-sim-pred
git remote add upstream https://gitcode.com/cann/mat-chem-sim-pred.git
git remote -v
```

### 2.2 创建开发分支

每个 PR 单独建分支，基于上游最新 master：

```bash
git fetch upstream
git checkout -b feature/<op-name> upstream/master
```

分支命名约定：

| 前缀 | 用途 |
|------|------|
| `feature/<op-name>` | 新增算子 |
| `fix/<bug-topic>` | 修复缺陷 |
| `docs/<doc-topic>` | 文档补充 |
| `benchmark/<op-name>` | 性能基准 |

### 2.3 配置提交身份

提交前确认本地 Git 身份与 CLA 签署邮箱一致：

```bash
git config user.name "你的姓名"
git config user.email "签署CLA的邮箱"
git log -1 --pretty=fuller
```

CLA 检查以 commit 中的邮箱为准。不要使用 `****@noreply.gitcode.com` 匿名邮箱。如果 PR 包含多个 commit 作者，所有作者都需要完成 CLA 签署。

### 2.4 开发与本地验证

开发完成后，提交前至少运行以下检查：

```bash
git status --short --branch
git diff --check

# Python 语法检查
python -m compileall -q <changed-python-path>

# Python 测试
python -m pytest <op>/tests/test_<op>.py -q
```

有 CANN 环境时追加编译和 smoke 测试：

```bash
cd <op>
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
./build/test_aclnn_<op> <device_id>
```

无法运行的测试项必须在 PR 说明中写明原因。

### 2.5 提交与推送

```bash
git add <files>
git commit -m "feature: add <op> operator"
git push origin feature/<op-name>
```

### 2.6 创建 PR

在 GitCode 个人 fork 页面创建 Pull Request，目标选择上游仓库的默认分支。PR 标题使用类型前缀（见第 6.1 节），描述参见仓库 PR 模板（见第 6.3 节）。

### 2.7 CLA 检查

PR 创建后 GitCode 会自动触发 CLA 检查。如提示未签署：

1. 按 PR 页面或 [CLA签署指南](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md) 完成签署。
2. 个人贡献者选个人 CLA；企业员工按企业法人流程登记。
3. 签署后在 PR 评论区按机器人提示触发重新检查，如 `/check-cla`。
4. 如仍显示 `cann-cla/no`，检查 commit 邮箱：

```bash
git log --pretty=fuller
```

邮箱不一致时用 `git commit --amend --reset-author` 修正后 `push --force-with-lease`。

---

## 3 算子目录结构

### 3.1 标准目录

新建算子使用小写加下划线命名。根据算子所属领域放入对应一级目录：

- 科学计算方向 → `simulation/<子方向>/<算子名>/`
- 预测优化方向 → `prediction/<子方向>/<算子名>/`

标准结构如下（参考 `template/operator_example.md`）：

```
<op>/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── algorithm.md
│   ├── api_reference.md
│   └── benchmark.md
├── op_host/
│   ├── <op>_host.h
│   ├── <op>_host.cpp
│   └── <op>_def.cpp
├── op_kernel/
│   └── <op>_kernel.cpp
├── examples/
│   └── test_aclnn_<op>.cpp
└── tests/
    ├── CMakeLists.txt
    ├── test_<op>.py
    ├── benchmark_<op>_aclnn.cpp
    ├── benchmark_<op>.py
    └── ut/
        ├── CMakeLists.txt
        └── op_kernel/
            ├── CMakeLists.txt
            └── test_<op>.cpp
```

正式 PR 至少包含：`CMakeLists.txt`、`README.md`、`docs/algorithm.md`、`docs/api_reference.md`、`op_host/`、`op_kernel/`、`examples/`、`tests/`。

### 3.2 公共模块复用

如果多个算子共享等价的数学逻辑（如 PIDModelFit 下三个 `*_basis_gemm_fit` 共用 basis-GEMM 底座），可以放在同级 `common/` 目录。但需满足：

- `common/` 只放真正跨算子复用的等价逻辑
- 不得把不同业务语义混成一个带 `family` 参数的大算子
- 每个公开算子仍需独立目录、独立 Host API、独立 kernel 入口、独立测试

PIDModelFit 的 `common/` 目录是可参考的实践案例：

```
PIDModelFit/
├── common/
│   ├── pid_basis_gemm_fit_common.h
│   ├── pid_basis_gemm_fit_host_impl.h
│   ├── pid_basis_gemm_fit_kernel.h
│   ├── pid_basis_gemm_acl_smoke.h
│   └── pid_basis_gemm_reference.py
├── pid_fopdt_basis_gemm_fit/
├── pid_ipdt_basis_gemm_fit/
└── pid_sopdt_basis_gemm_fit/
```

---

## 4 交付件要求

### 4.1 代码交付件

| 交付件 | 说明 |
|--------|------|
| `op_kernel/` | Ascend C device-side kernel，包含 `__aicore__` kernel 入口 |
| `op_host/` | Host launch API、参数检查、workspace 管理、tiling、kernel launcher |
| `CMakeLists.txt` | 独立构建入口，不依赖不可见的本地脚本 |
| `examples/` | 最小 ACL/ACLNN smoke 示例：构造输入 → 调用算子 → 取回结果 → 校验 |
| `tests/` | Python correctness 测试、C++ smoke/UT、benchmark |
| `common/` | 可选，仅放跨算子公共参考实现或公共底座 |

### 4.2 文档交付件

| 文档 | 必须说明的内容 |
|------|----------------|
| `README.md` | 算子功能、应用场景、输入输出、构建命令、测试命令、性能摘要 |
| `docs/algorithm.md` | 数学定义、核心公式、算法流程、边界条件、NPU 实现策略（参考 `template/algorithm.md`） |
| `docs/api_reference.md` | Host API、参数、shape、dtype、workspace、返回值、错误条件 |
| `docs/benchmark.md` | 测试环境、命令、输入规模、CPU baseline、NPU 计时口径、性能表 |
| `docs/test_report.md` | 较大 PR 推荐附上，汇总构建、功能、精度、性能验证结果 |

### 4.3 参考实现

正式 PR 需提供 CPU reference，优先级：

1. NumPy/Python reference — 适合公式和批量计算校验
2. PyTorch reference — 适合已有 PyTorch 生态或 tensor API 对齐
3. C++ reference — 适合 C++ smoke/UT 中逐元素对比

reference 必须与文档中的数学定义一致，不能为通过测试而反向拟合当前实现。

---

## 5 测试规范

### 5.1 功能正确性

至少覆盖以下场景：

- 小规模确定性样例
- 随机输入样例
- 边界条件：空输入、最小维度、极值、全零、重复最优候选
- 非法参数：空指针、shape 不合法、workspace 不足
- 与 CPU reference 的逐元素或逐指标对比

推荐输出指标：`max_abs_err`、`max_rel_err`、`mismatch_count`、`idx_diff_count`、`nan_count`。

### 5.2 精度阈值

不同算子按业务调整阈值，但必须在文档中明确。仓库已有参考阈值：

```
relative error < 1e-3  for force-like outputs
relative error < 1e-4  for energy-like outputs
relative error < 1e-6  for strict geometric/bond constraints
```

模型辨识、统计、评分类算子还需报告：

- 最优下标是否一致（`idx_diff_count=0`）
- 关键业务指标是否一致（`best_sse`、`best_k`、`score`、`metrics`）

### 5.3 构建测试

每个算子应能独立构建：

```bash
cd <op>
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
```

### 5.4 Python 测试

```bash
python -m pytest <op>/tests/test_<op>.py -q
python -m compileall -q <op>
```

有公共 reference 时额外检查：

```bash
python -m compileall -q <parent>/common
```

### 5.5 C++ / ACL smoke

每个正式算子提供一个最小 smoke 程序，至少完成以下步骤：

1. `aclInit` → 2. `aclrtSetDevice` → 3. 创建 stream → 4. 分配 Device memory → 5. H2D → 6. 调用 Host API → 7. stream synchronize → 8. D2H → 9. 与 CPU 预期对比 → 10. 释放资源

### 5.6 性能测试

性能数据必须包含：

- CPU baseline（单线程或多线程，注明线程数）
- NPU kernel / event compute 时间
- NPU end-to-end 时间
- 是否计入 H2D/D2H
- 输入规模说明
- 设备型号、CANN 版本、SOC_VERSION、重复次数

推荐区分以下计时口径：

| 口径 | 说明 |
|------|------|
| `kernel` / `event_compute` | 只统计 kernel 或核心 Device 计算 |
| `resident_e2e` | 输入已在 Device，统计 Device 侧 pipeline 和必要的最终 D2H |
| `cold_e2e` | 统计输入 H2D、计算、最终 D2H |
| `host_build_e2e` | 若 Host 需要构造 basis/索引/元数据，额外报告该成本 |

性能表推荐格式：

| 配置 | CPU baseline | NPU kernel | NPU resident e2e | NPU cold e2e | speedup | 精度 |
|------|--------------|------------|------------------|--------------|---------|------|
| `B=64,N=1024,M=256` | `x ms` | `x ms` | `x ms` | `x ms` | `x.x` | `idx_diff=0` |

禁止只写"性能提升明显"但不给环境、命令和数据。

---

## 6 PR 规范

### 6.1 标题前缀

| 前缀 | 用途 | 示例 |
|------|------|------|
| `feature:` | 新增算子或功能 | `feature: add PID basis GEMM fit operators` |
| `fix:` | 修复缺陷 | `fix: correct PME tiling overflow` |
| `docs:` | 文档补充 | `docs: add GAFF2 benchmark report` |
| `test:` | 测试补充 | `test: add SHAKE reference tests` |
| `benchmark:` | 性能基准 | `benchmark: add FOPDT batch rollout` |

### 6.2 PR 拆分原则

一个 PR 只做一个可评审主题：

- 一个独立算子
- 一个紧密相关的算子组（如 FOPDT/IPDT/SOPDT 三个同构模型辨识算子）
- 一次明确的 bug fix
- 一次明确的文档补充

禁止在同一 PR 中混入：正式算子和探索性算子、性能不成立但"顺便提交"的目录、大规模无关格式化、build 产物 / `__pycache__` / `.pyc` / 临时日志、未经说明的公共接口破坏性变更。

实验性方向满足以下条件后再进入正式 PR：功能正确、精度阈值明确且通过、有可复现性能数据、性能或工程价值说得清楚、交付件齐全。

### 6.3 PR 描述模板

```markdown
## 类型
- [x] 新功能
- [x] 文档内容更新

## 描述
说明算子功能、应用领域、目录、接口、算法逻辑。

## 变更范围
列出新增/修改的目录结构。

## API
列出公开 Host API（如 `aclnn<OpName>`）。

## 算法逻辑
核心公式和输入输出 shape。

## 如何测试
列出 Python 测试、CMake 构建、ACL smoke、benchmark 命令和结果。

## 性能结果
机器、NPU 型号、CANN 版本、CPU baseline 线程数、输入规模、重复次数、计时口径、加速比、精度结果。

## Checklist
- [x] 我的代码遵循这个项目的代码风格
- [x] 我已经自己测试过我的代码
- [x] 我已经更新了相应的文档
- [x] 我已经根据需要更新了对应的变更日志/测试记录
- [x] 我已经在标题中正确使用了类型标签
```

---

## 7 SCA 合规与许可证头

所有提交的源文件必须携带 CANN Open Software License Agreement Version 2.0 版权声明头，并通过 CI 流水线的 SCA 许可证合规扫描；不合规的 PR 将被拦截，无法合入。

### 7.1 版权头要求

所有提交的源文件必须包含 CANN Open Software License Agreement Version 2.0 版权声明头：

| 文件类型 | 版权头风格 |
|---------|-----------|
| `.cpp` / `.h` | C/C++ 块注释 `/* ... */` |
| `.py` | Python 行注释 `# ...` |
| `.sh` | Shell 行注释 `# ...` |
| `.md` | HTML 注释 `<!-- ... -->` |
| `CMakeLists.txt` | 行注释 `# ...` |

### 7.2 SCA 检查

提交 PR 后 CI 流水线自动执行 SCA 扫描，确认 `unconfirmedFileNum` 为 0。如果引入了第三方代码或参考资料，必须在 PR 说明中注明来源和许可证。

### 7.3 禁止提交的文件

以下文件不得进入版本管理：

- `build/` 目录及编译产物
- `__pycache__/`、`*.pyc`、`*.pyo`、`*.pyd`
- 临时日志文件、远端机器临时文件
- `PR-error/`（CI 错误日志目录）

---

## 8 安全编码要求

新算子的 Host 代码必须遵守以下安全编码要求。

### 8.1 64 位安全算术

所有涉及数组大小计算的乘法，先用 `static_cast` 将操作数提升到 `int64_t` 或 `size_t`，避免 `int32_t` 溢出：

```cpp
// 错误：int32 乘法可能溢出
int32_t sz = n_atoms * 3 * sizeof(float);

// 正确：64 位算术
int64_t sz = static_cast<int64_t>(n_atoms) * 3 * sizeof(float);
// 或
size_t sz = static_cast<size_t>(n_atoms) * 3 * sizeof(float);
```

典型溢出场景：`n_atoms * n_atoms * sizeof(int32_t)` 当 `n_atoms > 23170` 时溢出；`mesh_dim^3 * sizeof(float)` 当 `mesh_dim > 2150` 时溢出。

### 8.2 内存分配检查

所有 `malloc` / `aclrtMalloc` / `new` 调用后必须检查返回值，失败时释放已分配资源并返回错误码：

```cpp
// aclrtMalloc 检查
aclError ret = aclrtMalloc(&dev_ptr, sz, ACL_MEM_MALLOC_HUGE_FIRST);
if (ret != ACL_SUCCESS || dev_ptr == nullptr) {
    fprintf(stderr, "aclrtMalloc failed (ret=%d)\n", ret);
    // 释放已分配的内存
    return -1;
}

// malloc 检查
void* host_ptr = malloc(sz);
if (host_ptr == nullptr) {
    fprintf(stderr, "malloc failed\n");
    return -1;
}
```

推荐使用 `new (std::nothrow)` 替代抛异常的 `new[]`，分配失败时优雅返回而非崩溃。需 `#include <new>`。

### 8.3 缓冲区边界验证

所有接受外部 `n_atoms` / `mesh_dim` 等尺寸参数的函数，必须校验其与初始化时记录的值是否一致，防止越界访问：

```cpp
int SetCharges(int n_atoms, const float* charges) {
    if (n_atoms != n_atoms_) {
        fprintf(stderr, "n_atoms mismatch: got %d, expected %d\n", n_atoms, n_atoms_);
        return -1;
    }
    // ...
}
```

输入参数还需校验合法性（正数、不超过安全上限）：

```cpp
if (n_atoms <= 0 || n_atoms > 65535) {
    fprintf(stderr, "invalid n_atoms=%d\n", n_atoms);
    return -1;
}
```

### 8.4 修复注释约定

每处安全修复添加 `CWE-xxx fix:` 注释说明修复原因，便于代码审查和后续追溯。

---

## 9 Commit 规范

### 9.1 消息格式

```
<type>: <简短描述>

<可选的详细说明，每行不超过 72 字符>
```

常用 type：`feature`、`fix`、`docs`、`test`、`benchmark`。

### 9.2 多 commit PR

如果一个 PR 包含多个 commit，每个 commit 应描述一个逻辑独立的变更。合并时维护者可能选择 squash 或保留多 commit，取决于变更结构。

### 9.3 Co-author

使用 AI 辅助工具生成代码时，在 commit message 中添加 Co-authored-by 标注实际人类作者。所有 commit 的 author/committer 邮箱必须通过 CLA 检查。

---

## 10 环境要求

| 项目 | 要求 |
|------|------|
| CANN | ≥ 7.0 |
| 硬件 | Atlas A2 / A3 训练或推理卡 |
| CMake | ≥ 3.16 |
| SOC_VERSION | `Ascend910B1` 或 `Ascend910B3`（视测试环境而定） |
| Python | ≥ 3.8（用于 reference 和集成测试） |
| C++ 标准 | C++11 及以上 |

---

## 11 合入前检查清单

提交 PR 前逐项确认：

- [ ] PR 主题单一，没有混入探索性目录或无关改动
- [ ] 目录结构符合仓库习惯（参见第 3 节）
- [ ] `op_host/` 和 `op_kernel/` 均已提供
- [ ] 已提供 CPU reference
- [ ] 已提供 Python correctness 测试
- [ ] 已提供 ACL smoke 示例
- [ ] 已提供 benchmark
- [ ] 已提供 README、algorithm、api_reference、benchmark 文档
- [ ] 所有源文件包含 CANN OSL v2.0 版权声明头
- [ ] Host 代码遵循安全编码要求（64 位算术、分配检查、边界验证）
- [ ] 已运行 `git diff --check`
- [ ] 已运行 Python 测试或说明未运行原因
- [ ] 已运行 CMake 构建或说明未运行原因
- [ ] 已运行 NPU smoke / benchmark 或说明未运行原因
- [ ] PR 描述列出测试命令和结果
- [ ] 性能结果写清 CPU baseline、NPU 口径和输入规模
- [ ] commit author/committer 姓名、邮箱与 CLA 信息一致
- [ ] PR 中没有 `build/`、`__pycache__/`、`*.pyc`、临时日志

---

## 12 模板与参考资源

### 12.1 仓库内模板

| 模板 | 路径 | 说明 |
|------|------|------|
| 算子示例 | `template/operator_example.md` | 目录结构、Host/Kernel 代码框架、构建配置 |
| 算法说明 | `template/algorithm.md` | 算法原理、NPU 实现、精度与性能分析 |
| 参考文献 | `template/references.md` | 分类整理基础理论、硬件实现、领域应用 |
| 测试架构 | `template/test_architecture.md` | C++ UT、Python 集成测试、性能基准规范 |

### 12.2 CANN 官方参考仓库

学习 Ascend C 算子开发时，可 clone 以下仓库到本地 `Reference/` 目录（已加入 `.gitignore`，不纳入版本管理）：

| 仓库 | 说明 |
|------|------|
| [opbase](https://gitcode.com/cann/opbase) | Ascend C 算子开发基础库 |
| [ops-math](https://gitcode.com/cann/ops-math) | 数学运算算子库 |
| [ops-blas](https://gitcode.com/cann/ops-blas) | BLAS 线性代数算子库 |
| [ops-fft](https://gitcode.com/cann/ops-fft) | 傅里叶变换算子库 |
| [cann-learning-hub](https://gitcode.com/cann/cann-learning-hub) | CANN 学习中心 |
| [oam-tools](https://gitcode.com/cann/oam-tools) | 算子开发调试与性能分析工具 |
| [ops-test-kit](https://gitcode.com/cann/ops-test-kit) | 算子测试套件 |
| [cann-samples](https://gitcode.com/cann/cann-samples) | Ascend C 算子开发完整示例集 |

### 12.3 参考链接

- GitCode PR 帮助文档：https://docs.gitcode.com/docs/help/home/org_project/pullrequests/
- GitCode CLA 帮助文档：https://docs.gitcode.com/v1-docs/docs/pulls/cla/
- CANN community 仓库：https://gitcode.com/cann/community
- 本仓库贡献指南公告：https://gitcode.com/cann/mat-chem-sim-pred/discussions/1

---

## 13 维护团队

### Maintainer

- 黄剑兴 huangjianxing4@huawei.com
- 张玉橙 zhangyucheng23@huawei.com
- 周吉彬 zhoujibin@dicp.ac.cn
- 高菲 gaofei06@petrochina.com.cn

### Committer

- 刘非 liuf23357@gmail.com
- 刘海东 aliutec@163.com
- 高梓博 gaozibo@petrochina.com.cn
- 马博文 iambowen.m@qq.com
- 刘达林 liudalin@huawei.com
- 赵俊 roomdream@qq.com
- 郑柳琪 2557692481@qq.com
- 张强豪 1964035193@qq.com
- 李姝漫 1404537011@qq.com

如有疑问，按场景选择沟通渠道：

- **[Issue](https://gitcode.com/cann/mat-chem-sim-pred/issues)** — 报告 bug、提出功能请求、反馈具体代码问题
- **[Discussions](https://gitcode.com/cann/mat-chem-sim-pred/discussions)** — 提问交流、想法探讨、方案讨论
- **PR 评论区** — 在相关 PR 中 @Maintainer，处理该 PR 范围内的问题
