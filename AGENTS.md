# AGENTS.md — mat-chem-sim-pred 开发指南

本文件面向在本仓库中修改代码的 AI Agent 与开发者，定义可执行的仓库约束、验证方式和交付标准。它是贡献指南的精简工作指引；算子目录、PR、CLA、许可证和交付件的完整规则以 [CONTRIBUTING.md](CONTRIBUTING.md) 为准。若本文件与 GitCode 页面、CI、CLA 机器人或维护者要求冲突，依次以它们为准。

## 1. 仓库与工作边界

- 这是基于 CANN / Ascend C 的材料、化学、科学计算与预测优化算子仓库，主要代码位于 `simulation/` 与 `prediction/`。
- 仓库**没有顶层 CMake 构建入口**；每个算子目录应能够独立配置、编译和测试。
- 修改前先阅读目标目录的 `README.md`、`CMakeLists.txt`、`docs/`、`tests/` 及相邻同类算子；优先复用同级 `common/` 中确有跨算子共性的逻辑。
- 不要擅自修改无关算子、批量格式化全仓库、提交构建产物、临时脚本、日志或测试缓存。
- 不要将探索性原型与正式交付算子混在同一 PR；公开算子保持独立目录、Host API、kernel 入口和测试。

## 2. 目录与交付要求

新增正式算子使用小写下划线命名，并放在合适的业务子目录：

```text
<op>/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── algorithm.md
│   ├── api_reference.md
│   ├── benchmark.md
│   └── test_report.md              # 建议：较大变更必须提供
├── op_host/                         # 参数校验、tiling、workspace、Host launcher
├── op_kernel/                       # Ascend C device kernel 与 __aicore__ 入口
├── examples/                        # 可独立运行的 ACL/ACLNN smoke 示例
├── reference/                       # CPU reference（Python/NumPy/PyTorch/C++）
└── tests/                           # 正确性、C++ UT、smoke、benchmark 或 E2E
```

- 以 `template/` 中的 `operator_example.md`、`algorithm.md`、`test_architecture.md` 为模板。
- 算法、reference、Host API 和 kernel 实现必须语义一致；不得为了通过测试而让 reference 反向拟合当前错误实现。
- `README.md` 至少说明功能、应用场景、输入输出、构建与测试命令、性能摘要。
- `docs/algorithm.md` 说明数学定义、边界条件和 NPU 实现策略；`api_reference.md` 说明 API、shape、dtype、workspace 与错误条件；`benchmark.md` 说明环境、命令、规模、计时口径和性能数据。

## 3. 编码要求

### C++ / Ascend C

- 遵循仓库 `.clang-format`：Google 基础风格、4 空格缩进、120 列、函数使用 Allman 大括号、`SortIncludes: false`。
- 保持 Host 与 kernel 职责清晰：Host 负责参数、shape/dtype、workspace 和 tiling 校验；kernel 只处理已验证的设备端计算。
- 所有尺寸、字节数和索引乘法使用 `int64_t` 或 `size_t` 提升后计算，避免 32 位溢出。
- 检查所有 `malloc`、`new (std::nothrow)`、`aclrtMalloc` 及 ACL/CANN 调用的返回值；失败路径必须释放已取得的资源并返回明确错误。
- 对外部尺寸、空指针、shape/dtype、范围、workspace 和设备缓冲区边界进行校验，禁止依赖未验证的隐式前提。
- 安全修复在修复位置添加 `CWE-xxx fix:` 注释，说明受控的缺陷类型。

### Python 与通用质量

- Python 代码必须通过 `ruff check` 与 `ruff format`；测试使用 `pytest`，避免依赖不可复现的随机状态。
- 遵循现有命名、注释密度、错误处理和文档语言；新增公开接口时同步更新文档和示例。
- 不提交密钥、访问令牌、个人数据、二进制大文件或未经许可的第三方代码。引入第三方内容时，在文档和 PR 中注明来源与许可证。

## 4. 许可证与合规

- 仓库整体采用 Apache License 2.0，详见 [LICENSE](LICENSE)。
- 新增或实质修改的源代码、脚本和构建配置文件必须使用对应格式的 CANN Open Software License Agreement Version 2.0 版权头；具体要求以 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [OAT.xml](OAT.xml) 为准。
- 普通 Markdown 文档默认不强制添加版权头；已有版权和许可证声明必须保留。
- 第三方内容须保留原声明，并在 PR 中注明来源和许可证。
- 提交前运行可用的 License、Copyright 和 SCA 检查；不合规文件可能阻塞合入。
- 禁止纳入：`build/`、编译产物、`__pycache__/`、`*.pyc`、`*.pyo`、`*.pyd`、临时日志、远端环境文件、`PR-error/`。

## 5. 本地验证

先从变更范围最小的检查开始；没有 CANN/NPU 环境时，不要伪造结果，必须在 PR 中说明未运行的项目和原因。

```bash
# 所有变更的基础检查
git diff --check
pre-commit run --all-files

# 变更 Python 文件或 reference
python -m compileall -q <changed-python-path>
python -m pytest <op>/tests/test_<op>.py -q

# 单个 Ascend C 算子独立构建
cmake -S <op> -B <op>/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DSOC_VERSION=Ascend910B1
cmake --build <op>/build -j 2
ctest --test-dir <op>/build --output-on-failure

# 有 NPU 环境时运行示例（名称和参数以目标算子 README 为准）
<op>/build/test_aclnn_<op> <device_id>
```

- 变更公共代码时，运行所有受影响算子的测试；只验证一个调用方不足以证明兼容性。
- 正确性测试至少覆盖：小规模确定性输入、随机输入、边界输入、非法参数及与 CPU reference 的对比。
- 在文档和 PR 中明确精度阈值，报告 `max_abs_err`、`max_rel_err`、`mismatch_count`、`nan_count` 等适用指标；涉及选优时还应验证最优索引和关键业务指标。
- 性能报告必须区分 CPU baseline、NPU kernel/event、resident E2E、cold E2E（如适用），并写明硬件、CANN 版本、SOC_VERSION、输入规模、重复次数和 H2D/D2H 是否计入。

## 6. CI 工作流事实

工作流定义在 `.gitcode/workflows/`，PR/MR 评论触发命令为 `/compile`（或 `compile`）。提交前应确保变更可通过以下检查：

| 工作流 | 实际覆盖内容 |
|---|---|
| `mat-chem-sim-pred_actions.yml` | 社区 PreBuild、CodeCheck、Post 流程，包括 SCA / 社区代码检查 |
| `ci.yml` | `pre-commit run --all-files`；`prediction/SmallData/test_gpr.py` 的 PyTorch reference 测试 |
| `operator-build.yml` | AI4MD（LJ、DPD、GAFF2）、AI4PDE（PINN、FNO、DeepONet）独立构建及部分 C++/Python 测试；材料性质预测 DAO NPU 测试 |
| `nightly.yml` | 每周全量构建 AI4MD 与 AI4PDE，运行可用 C++ UT、GPR/DAO Python 测试，并生成构建报告 |

- CI 对 PR 的 merge commit 执行检查，并启用按 MR 的并发抢占；排查失败时以 merge commit 和流水线日志为准。
- 新增目录或测试后，确认工作流能实际发现并运行它。若尚未接入 CI，在 PR 中明确说明，并补充对应工作流或后续接入计划。

## 7. Git、提交与 PR

- 一个分支和一个 PR 只处理一个可评审主题；基于上游最新 `master` 开发。
- 分支建议：`feature/<op-name>`、`fix/<topic>`、`docs/<topic>`、`test/<topic>`、`benchmark/<op-name>`。
- Commit 与 PR 标题使用：`feature:`、`fix:`、`docs:`、`test:` 或 `benchmark:`，描述简洁且对应实际变更。
- 提交前检查 `git status --short --branch`，仅暂存有意提交的文件。
- PR 描述必须包含：变更范围、公开 API/兼容性影响、算法或实现说明、实际执行的测试命令与结果、未执行测试及原因、性能与精度数据（如适用）。
- 提交者/共同作者邮箱必须满足 CLA 要求；不要使用匿名 `noreply` 邮箱。使用 AI 辅助时遵循仓库 `CONTRIBUTING.md` 的 Co-author 要求。

## 8. Agent 执行清单

- **开始前：**确认需求、阅读同类实现和相关文档、识别 API/兼容性/测试影响。
- **修改时：**保持变更最小且可回滚，同步代码、reference、测试、文档和构建配置。
- **完成前：**运行可用验证，审查 `git diff` 与 `git diff --check`，如实报告修改、测试结果、限制和未验证项。
