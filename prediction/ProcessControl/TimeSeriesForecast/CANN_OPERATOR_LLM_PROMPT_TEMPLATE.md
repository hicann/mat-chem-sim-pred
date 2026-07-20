# CANN 算子大模型协作开发提示词模板与样例

本文用于帮助开发者通过多轮提示词，与大模型协作完成 CANN / Ascend C 自定义算子的需求分析、接口设计、代码实现、编译、NPU 数值验证、性能优化和模型集成。

模板来自本仓库第一批 10 个时序算子的实际开发经验：

- `SelectiveScan1D`
- `S6ScanFused`
- `AutoCorrFusedAggregate`
- `BatchSpdInvFp32`
- `TirexSlstmCell`
- `CfcScanFused`
- `CornnScanFused`
- `SruScanFused`
- `UnicornnScanFused`
- `LtcScanFused`

这些算子的共同经验是：高质量交互不是一句“帮我写一个算子”，而是让大模型围绕可验证的阶段目标持续工作。每一轮都应提供事实、要求落地，并用编译或测试结果进入下一轮。

---

## 1. 最重要的协作原则

### 1.1 先说明算子的价值类型

开发前先判断目标属于哪一类：

| 类型 | 典型场景 | 主要验收证据 |
|---|---|---|
| 性能型 | Python 循环或多个小 `torch_npu` 算子产生大量 kernel launch | 同一 NPU、同一输入下，自定义算子相对框架基线的热启动耗时和端到端收益 |
| 使能型 | 原框架算子不支持 NPU、发生 CPU fallback | 原路径的 fallback 证据、自定义路径全程 NPU、数值正确 |
| 融合型 | 中间 Tensor、重复 GM 读写或多个 aclnn 调用开销高 | 融合前后的 kernel 数、GM 流量、耗时和模型子图收益 |
| 复用型 | 多个上层算子都依赖同一基础能力 | 接口稳定、覆盖多个调用方、正确性和可维护性 |

本仓库第一批算子中，9 个主要属于性能/融合型；`BatchSpdInvFp32` 首先是消除 `torch.linalg.lstsq` CPU fallback 的使能型算子。两类算子不应使用同一套性能口径。

### 1.2 每轮只设置一个可验收目标

推荐顺序：

```text
定位热点 -> 固化数学和接口 -> 写 CPU/PyTorch 真值
-> 生成工程与 Host/tiling -> 实现 Ascend C kernel
-> 编译通过 -> NPU 小 shape 正确 -> 随机/边界正确
-> 性能优化 -> 模型子图集成 -> 文档和提交审查
```

不要在数学定义、接口和目标 shape 尚未确定时，直接要求大模型一次性生成全部代码并宣称完成。

### 1.3 大模型必须以仓库和工具输出为事实来源

在提示词中明确要求：

1. 先读取相邻算子的真实实现和构建脚本，再选择项目已有写法。
2. 不凭记忆编造 Ascend C、CANN、aclnn 或 msopgen API。
3. 接口、tiling 字段、kernel 参数顺序和测试调用必须逐项一致。
4. 修改后必须运行可执行的检查；无法运行时必须明确缺少什么环境。
5. 编译失败时依据完整日志修复，不通过删除功能或绕过测试“解决”。
6. 不得把局部 kernel、单层 encoder 或 block 测试描述为完整模型 E2E。
7. 不得捏造未保存的绝对耗时、精度或加速比。
8. 图模式结果必须区分“原模型表达直接成图”和“因 converter 缺失而人工等价改写后成图”；后者是强框架基线，但不能描述为图编译器自动支持原模型。

当原模型缺少 TorchAir converter 时，还应分别记录两类工程代价：框架方案需要维护额外的 PyTorch 等价表达改写；custom 算子直接实现目标子图语义，模型侧只需在适配层替换为 ACLNN 调用。性能比较可以使用改写后的强图基线，但必须明确该前提。

### 1.4 每次给大模型提供完整反馈

下一轮至少粘贴以下内容：

```text
执行命令：<完整命令>
退出码：<exit code>
关键日志：<从第一条真正错误开始，保留上下文>
当前环境：<芯片/CANN/编译器/torch/torch_npu>
期望结果：<本轮通过标准>
```

只发送最后一行错误，通常不足以定位 Host、kernel、链接、运行时或环境问题。

---

## 2. 开始前的信息表

将下面内容填写后再开始对话。未知项写“待确认”，不要让大模型自行猜测。

```text
【项目】
仓库根目录：<路径>
参考算子目录：<最相近的 1~3 个目录>
构建入口：<脚本或命令>
运行环境：<本机/远端机器>
目标芯片：<例如 Ascend 910B3>
CANN 版本：<版本>
框架版本：<torch / torch_npu 版本>

【目标算子】
算子名：<PascalCase，例如 SelectiveScan1D>
文件/内核名：<snake_case，例如 selective_scan1_d>
所属模型和位置：<模型、block、子图>
价值类型：<性能型/使能型/融合型/复用型>
要替换的原始代码：<文件、函数或代码片段>

【数学与接口】
数学公式或伪代码：<必须给出>
输入：<名称、shape、dtype、layout、含义>
输出：<名称、shape、dtype、layout、含义>
属性：<名称、类型、默认值>
主测试 shape：<真实工作负载>
边界 shape：<最小值、非对齐值、尾块等>
动态 shape 范围：<如不支持，要明确静态约束>
误差阈值：<max_abs_diff / rtol / atol>

【性能与交付】
框架基线：<eager torch_npu / 图模式 / CPU fallback 等>
计时口径：<warmup、repeat、同步位置、统计量>
模型级验收范围：<component/block/layer/encoder/validation loop>
预期交付文件：<Host、tiling、kernel、测试、文档等>
```

---

## 3. 一段式总提示词模板

当需求已经非常清楚，且大模型能够直接访问仓库和终端时，可以使用下面的总提示词。

```text
你是一名资深 CANN / Ascend C 算子工程师。请在当前仓库中完成
【<算子名>】自定义算子的开发和验证，不要只给设计建议。

项目上下文：
- 仓库根目录：<路径>
- 目标硬件：<芯片>
- CANN/框架版本：<版本>
- 构建入口：<命令>
- 最相近的参考算子：<路径 1>、<路径 2>
- 原框架实现：<路径/函数/代码片段>

目标与价值：
- 模型位置：<模型子图>
- 价值类型：<性能型/使能型/融合型/复用型>
- 当前问题：<逐时间步 launch、CPU fallback、中间 Tensor 等>
- 本轮完成标准：<明确且可执行>

算子契约：
- 数学语义：<公式或伪代码>
- 输入：<逐项列出 name/shape/dtype/layout/含义>
- 输出：<逐项列出 name/shape/dtype/layout/含义>
- 属性：<逐项列出>
- 支持范围：<shape/dtype/动态 shape>
- 主 shape：<shape>
- 正确性阈值：<阈值>

请按以下方式工作：
1. 先检查仓库结构、参考算子、构建脚本和现有未提交修改，列出契约歧义和主要风险。
2. 先建立可独立运行的 CPU/PyTorch reference 和确定性测试数据，再实现设备代码。
3. 沿用项目已有的 msopgen、Host infer、tiling、Ascend C kernel、aclnn 和测试模式；不要发明另一套目录结构。
4. 保证 op proto、Host、tiling data、kernel 入口、aclnn 调用的参数顺序和 dtype 完全一致。
5. 设计 tiling 时说明独立并行维度、blockDim、UB 占用、32B 对齐、尾块和越界保护。
6. 对递归维保持正确的数据依赖；不要为了并行破坏数学语义。不同 batch/channel/head 可以并行时再分核。
7. 实际修改文件并运行当前环境可执行的格式、生成、编译和测试命令。遇到错误时根据完整日志继续修复。
8. 正确性至少覆盖：手算小样例、随机样例、主 shape、边界/非对齐 shape、重复运行稳定性。
9. 性能测试必须包含 warmup、设备同步和多次热启动统计；与同一设备上的原框架路径比较，并单列首次编译/冷启动。
10. 不得伪造命令结果、绝对耗时或 E2E 范围；无法在当前环境验证的项目明确标为未验证。

最终报告：
- 修改的文件及作用
- 实际执行的命令和结果
- 数值误差
- 性能数据及计时口径
- 已知限制、未覆盖 shape 和下一步

先开始读取仓库并实施。除非存在会改变接口或数学语义的阻塞问题，否则不要停留在计划阶段等待确认。
```

总提示词适合成熟需求。首次开发或复杂融合算子更推荐下面的分阶段提示词。

---

## 4. 推荐的分阶段提示词

### 阶段 0：仓库摸底和相似实现定位

```text
先不要修改代码。请检查当前 CANN 算子仓库，回答：

1. 与【<目标算子>】最相近的 1~3 个现有算子是什么，为什么；
2. 本项目从算子描述、Host/tiling、Ascend C kernel、aclnn 到 runtime test 的文件链路；
3. 真实构建和运行入口；
4. 可以复用的类型、宏、测试工具和命名约定；
5. 目标需求中仍不明确、会影响接口或正确性的内容；
6. 当前工作区已有修改，哪些必须保留。

结论必须引用具体文件和代码位置，不要根据通用 CANN 经验猜测。
```

验收：大模型找到了正确参考实现和真实构建入口，并指出接口歧义，而不是立即生成孤立代码。

### 阶段 1：热点、价值和融合边界

```text
请分析下面的原框架实现是否值得开发自定义算子：

<粘贴原 PyTorch/torch_npu 函数或给出文件路径>

真实输入 shape：<shape>
目标设备：<芯片>
当前 profile：<耗时、kernel 数、CPU fallback 或 profiler 摘要>

请输出：
1. 当前瓶颈属于计算、launch、GM 搬运、CPU fallback 还是动态 shape；
2. 建议的最小有效融合边界，以及不应该融合的部分；
3. 算子属于性能型、使能型、融合型还是复用型；
4. 独立并行维度和必须顺序执行的维度；
5. 预期收益、主要风险和失败退出条件；
6. component、block、模型验证分别应比较什么。

不要直接承诺加速比。没有 profile 证据的判断标记为假设。
```

验收：融合边界能够减少实际开销，同时没有把过大、变化频繁或难以验证的整段模型盲目塞进一个 kernel。

### 阶段 2：冻结算子契约和 Reference

```text
请为【<算子名>】冻结 v1 契约，并实现独立的 CPU/PyTorch reference 测试。

数学定义：
<公式或伪代码>

候选输入输出：
<name/shape/dtype/layout/含义>

要求：
1. 列出每个维度的含义、合法范围和维度间约束；
2. 明确广播、初始状态、数值稳定处理、输出 layout 和是否原地修改；
3. 明确支持/不支持的 dtype、动态 shape、非连续 Tensor 和空 Tensor；
4. 给出 shape infer 和 dtype infer 规则；
5. 写确定性 reference，不能调用待实现的自定义算子；
6. 设计手算小样例、随机样例、主 shape 和边界样例；
7. 给出 atol/rtol/max_abs_diff 阈值及理由。

如果公式和接口存在冲突，先指出并给出最小修正；本阶段不要写 Ascend C kernel。
```

验收：相同输入能够由 reference 唯一决定输出，接口不再依赖口头解释。

### 阶段 3：工程骨架、Host 和 tiling

```text
基于已经冻结的【<算子名> v1 契约】，在仓库中补齐算子工程骨架、
op proto/msopgen 配置、Host infer、tiling data 和 aclnn/runtime test 骨架。

参考实现：<路径>
构建入口：<命令>

要求：
1. 沿用当前仓库命名和生成方式；
2. 所有层的输入输出顺序、dtype 和 kernel symbol 逐项核对；
3. tiling 字段只保存 kernel 真正需要的 shape/策略数据；
4. Host 校验 rank、维度约束、dtype 和溢出风险；
5. 说明 blockDim 的初始选择依据；
6. 运行 msopgen/静态检查，修复到本阶段可验证项通过；
7. 输出一张“契约字段 -> Host -> tiling -> kernel 参数 -> 测试”的对应表。

本阶段 kernel 可保留最小占位实现，但不得伪装成数值正确。
```

验收：生成工程可用，各层接口完全一致，后续只需要填充真实 kernel 逻辑。

### 阶段 4：Ascend C kernel 实现

```text
现在实现【<算子名>】真实 Ascend C kernel。

已冻结语义：<reference 路径>
Host/tiling：<路径>
相似 kernel：<路径>
主 shape：<shape>

实现要求：
1. 先给出 GM -> UB -> 计算 -> GM 数据流和每核工作范围；
2. 说明顺序维和可并行维，证明不会出现多核写冲突；
3. 估算每个 Queue/Buffer 的 UB 字节数和总量；
4. 处理 32B 对齐、尾块、非对齐长度和越界；
5. 复用数据尽量驻留 UB，中间状态不无意义写回 GM；
6. 优先使用目标 CANN 版本已存在的向量指令；不确定 API 时从仓库或本机头文件查证；
7. 对 exp/log/sigmoid/softplus/三角函数、归约和矩阵求解说明数值风险；
8. 先完成正确的 v1，再基于数据做双缓冲、流水或多核优化；
9. 实际修改代码并运行能执行的检查。

不要用省略号、伪代码或未定义 helper 代替核心计算。
```

验收：kernel 是完整实现，且 tiling、UB、对齐、并行和数值策略都有可核查依据。

### 阶段 5：编译和运行时错误闭环

```text
下面是【<算子名>】的真实失败信息。请先定位第一条根因，再直接修改并复验。

环境：<芯片/CANN/编译器>
命令：<完整命令>
退出码：<退出码>
完整关键日志：
[日志开始]
<日志>
[日志结束]

要求：
1. 区分生成、Host C++、Ascend C、链接、op 注册、aclnn 加载或运行时错误；
2. 检查参数顺序、symbol、tiling struct、blockDim、库路径和生成缓存；
3. 从仓库或目标版本头文件验证 API，不凭印象改函数签名；
4. 做最小根因修复，不删除校验、不跳过测试；
5. 重新执行原命令，并报告新的退出码和关键输出；
6. 若出现下一层错误，继续修复直到本轮目标通过或确认外部阻塞。
```

验收：不是只解释错误，而是原失败命令已经通过，或有可复现的外部阻塞证据。

### 阶段 6：数值正确性和边界测试

```text
请为【<算子名>】建立 NPU 数值验收，reference 为【<路径/函数>】。

测试矩阵至少包括：
- 手算小样例：<shape>
- 随机小样例：<shape/seed>
- 主工作负载：<shape>
- 非 32B/向量长度对齐：<shape>
- 多核分配有尾块：<shape>
- 数值极端值：<范围>
- 重复运行：<次数>

要求：
1. 比较 max/mean abs diff，必要时同时比较 rtol/atol、索引一致率和 NaN/Inf；
2. 对递归算子增加长序列误差累积测试；
3. 对 top-k/排序说明相等元素的 tie-breaking；
4. 对求逆/分解覆盖病态或非正定输入的契约行为；
5. 测试必须调用生成的真实 aclnn 接口和 NPU kernel；
6. 失败时保存 seed、输入摘要、期望值、实际值和首个错误位置；
7. 实际运行并以结果文件或终端输出为证据。

通过阈值：<阈值>。不要只检查程序能启动或输出非空。
```

验收：正确性覆盖真实 NPU 路径、主 shape 和最可能出错的边界，而不只是 UI/接口 smoke。

### 阶段 7：性能分析和优化

```text
【<算子名>】已经通过数值测试。请分析并优化性能，同时保持相同契约。

主 shape：<shape>
当前自定义算子数据：<cold/warm/hot/重复次数>
框架 eager 基线：<数据>
图模式基线：<数据或无法成图原因>
profile 摘要：<AI Core、内存、launch、热点指令等>

要求：
1. 先判断瓶颈是 launch、GM 访问、标量访问、UB、向量利用率、核间负载还是同步；
2. 按预期收益/风险排序提出优化项，每次只实施可单独验证的一项；
3. 关注连续 DataCopy、UB 数据复用、向量化、合理 blockDim、尾块和同步；
4. 每项优化后重跑同一正确性矩阵和同一性能命令；
5. 计时包含 warmup、每次计时前后必要同步、至少多次热启动，报告均值/中位数；
6. cold/首次编译与 hot latency 分开；
7. 只有同设备、同 shape、同输入、同精度和同计时边界的数据才计算加速比；
8. 若优化使主 shape 变快但边界 shape 退化或失败，要显式报告。

最终给出优化前后代码差异、误差、绝对毫秒和加速比，不接受只有理论分析。
```

验收：每个性能结论都有正确性回归和同口径实测支持。

### 阶段 8：模型子图集成和真实收益

```text
请把【<算子名>】集成到【<模型/模块>】的真实推理路径，并评估替换收益。

原实现：<路径/函数>
自定义算子库和接口：<路径/symbol>
真实 shape/数据：<说明>

要求：
1. 只替换目标子图，其余模型路径和权重保持一致；
2. 提供不支持 shape/dtype 时的明确 fallback 或报错策略；
3. 对比原 eager torch_npu、自定义算子；可成图时增加图模式基线；
4. 分别报告 component、block/layer 和实际验证循环耗时；
5. 比较中间输出、最终预测及 MSE/MAE 等任务指标；
6. 明确数据是真实数据、真实预训练权重、确定性随机权重还是合成 Tensor；
7. 明确测试范围是 cell、layer、encoder、block、validation loop 还是完整 E2E；
8. 数据加载、预处理、首次编译是否计时必须单列。

不得把局部范围扩大表述为完整训练或完整模型收益。
```

验收：证明算子解决了原模型的实际问题，而不只是 isolated microbenchmark 更快。

### 阶段 9：提交前独立审查

```text
请以 CANN committer 的标准，对【<算子名>】做提交前审查。先不要修改代码。

重点检查：
1. 数学语义、接口、shape/dtype infer 是否一致；
2. Host tiling data 与 kernel 内解释是否逐字段一致；
3. UB 容量、对齐、尾块、越界、多核竞争和同步；
4. 数值稳定性、NaN/Inf、近零除法和长序列累计误差；
5. aclnn 生命周期、workspace、stream 同步和资源释放；
6. 测试是否真的覆盖 NPU 结果、主 shape、边界和回归；
7. benchmark 是否公平，是否混入冷启动、数据搬运或 CPU fallback；
8. 文档中的数据来源、验证层级和性能表述是否夸大；
9. 是否存在生成文件/手写源不一致或不可复现步骤。

按严重程度列出问题，引用具体文件和行号。没有证据的问题标为疑问，
不要把猜测写成缺陷。审查完成后再根据确认的问题逐项修复和复验。
```

---

## 5. 完整填写样例：SelectiveScan1D

下面以本仓库已经完成的 `SelectiveScan1D` 为例，展示如何把模糊需求变成可执行提示词。

### 5.1 不推荐的提示词

```text
帮我写一个 Mamba 的 CANN selective scan 算子，要比 PyTorch 快。
```

问题：没有接口、shape、精度、目标芯片、参考实现、构建方式和验收范围。“比 PyTorch 快”也没有说明是 CPU、eager `torch_npu`、图模式还是完整模型。

### 5.2 推荐的首次开发提示词

```text
你是一名资深 CANN / Ascend C 算子工程师。请在当前仓库开发
SelectiveScan1D，用于替换 Mamba block 中由 Python 时间循环和多个
torch_npu 小算子组成的 selective scan。目标首先是同一 Ascend 910B3
上相对 eager torch_npu 减少 kernel launch，不与 CPU 做主要性能结论。

仓库与环境：
- 根目录：<mat-chem-sim-pred 仓库根目录>
- 目标芯片：Ascend 910B3
- 参考工程模式：prediction/ProcessControl/TimeSeriesForecast/ 下最相近的已交付算子
- 构建入口：目标算子的 README.md、CMakeLists.txt 或 msopgen 配置
- 交付层次：msopgen/op proto、Host/tiling、Ascend C kernel、aclnn runtime smoke、benchmark、README

数学语义：
state 初始为 0，shape 为 [B,D,N]。对 t=0..L-1：
  decay[b,d,n] = exp(delta[b,t,d] * a[d,n])
  update[b,d,n] = delta[b,t,d] * b[b,t,n] * u[b,t,d]
  state = decay * state + update
  out[b,t,d] = sum_n(state[b,d,n] * c[b,t,n]) + u[b,t,d] * d[d]

接口：
- u:     [B,L,D], float32, ND contiguous
- delta: [B,L,D], float32, ND contiguous
- a:     [D,N],   float32
- b:     [B,L,N], float32
- c:     [B,L,N], float32
- d:     [D],     float32
- out:   [B,L,D], float32
- v1 不做原地修改，不支持空 Tensor，N 必须大于 0。

主 shape：B=1,L=1024,D=1536,N=16。
小样例：B=1,L=2,D=1,N=32；u=[2,3]，delta=1，a=0，b=1，c=1，d=0，
期望 out=[64,160]。
正确性阈值：小样例 max_abs_diff <= 1e-4；随机/主 shape 以 PyTorch
float32 reference 为准，报告 max 和 mean abs diff。

实现约束：
1. L 有递归依赖，必须顺序；将 B*D 作为独立 scan group 分核。
2. 每个 group 的 a[d,:] 和 state[N] 尽量驻留 UB；b/c 按时间步连续搬入 UB。
3. 使用硬件 Exp 和向量 Mul/Add/ReduceSum；不要使用未经误差证明的低阶 exp 近似。
4. 明确 UB 占用、DataCopy 对齐、N 非对齐时的处理和 blockDim。
5. Host tiling 至少传 batch/length/dim/state，并与 kernel struct 完全一致。
6. 先读取仓库实际脚本和相似实现，不凭记忆编造 API。
7. 实际修改代码，运行生成/编译和 NPU smoke；若本机没有 CANN/NPU，完成静态可验证项并给出远端复现命令，明确标记未验证项。

性能测试：
- 对比同一设备上的 eager torch_npu Python scan 和 custom op；
- warmup 后测量至少 10 次，计时边界包含必要的 aclrt/torch.npu 同步；
- 分开报告 cold 和 hot；
- 先报告 scan-only，再在同一个最小 Mamba block 中只替换 scan，报告 block 耗时和输出差异。

最终输出修改文件、命令、编译结果、数值结果、性能结果、已知限制。
现在先检查仓库和接口一致性，然后持续实施到当前环境可达到的完成状态。
```

这个提示词明确了 `L` 必须顺序、`B*D` 可以并行，使大模型不容易错误地沿时间维分核；同时把小样例期望值提前固定，能快速发现参数顺序、状态更新和归约错误。

### 5.3 编译报错反馈样例

```text
SelectiveScan1D 的 msopgen 工程生成成功，但 build.sh 失败。请直接定位、修改并复验。

环境：Ascend 910B3，<填写 CANN 版本>
命令：
cd prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
退出码：1

关键日志：
[日志开始]
<从第一条 error 开始粘贴完整编译上下文>
[日志结束]

请依次核对：
- 生成的 op_kernel 文件名与替换目标；
- kernel symbol `selective_scan1_d`；
- op proto、Host、aclnn 的 6 输入 1 输出顺序；
- `SelectiveScan1DTilingData` 的 batch/length/dim/state 字段顺序和类型；
- 当前 CANN 头文件中 `ReduceSum`/`Exp`/`DataCopy` 的真实签名；
- 旧 build_out 是否造成缓存污染。

不要把真实 kernel 换回生成占位代码。修复后重新运行原命令，并给出退出码、
生成的 `libcust_opapi.so`、aclnn header 和 kernel object 是否存在。
```

注意：样例中的日志占位符必须替换为真实输出。不能让大模型猜一条并不存在的编译错误。

### 5.4 数值错误反馈样例

```text
SelectiveScan1D 已编译并能调用，但随机长序列在 L=1024 时 max_abs_diff 超阈值。

seed：20260716
shape：B=1,L=1024,D=1536,N=16
reference：<文件和函数>
custom：<runtime test 文件>
期望阈值：max_abs_diff <= <阈值>
实际：max_abs_diff=<值>，mean_abs_diff=<值>
首个超阈值位置：[b,t,d]=<位置>
该位置 expected=<值>，actual=<值>
NaN/Inf 数量：<值>

请先构造能复现首个错误的更小 shape，逐步比较 decay、update、state、reduce 和 skip
五个阶段。重点检查：
1. u/delta 的 [B,L,D] offset 与 b/c 的 [B,L,N] offset 是否混用；
2. a 是否按 [D,N] 取行；
3. state 是否只在每个 (B,D) group 开始时清零；
4. ReduceSum 的有效元素数与对齐 padding 是否混淆；
5. 多核是否写入同一 out 区域；
6. exp 输入范围和硬件/近似实现差异。

修复后重跑手算、随机小 shape、主 shape、N 非对齐和重复运行测试，不只重跑失败样例。
```

### 5.5 性能优化提示词样例

```text
SelectiveScan1D 当前已经数值正确，但主 shape 热启动仍慢。请基于 profile 做一次有证据的优化。

shape：B=1,L=1024,D=1536,N=16
当前实现：每个 (B,D) group 顺序扫描 L；a/state 驻留 UB；b/c 每步读取。
当前数据：
- eager torch_npu scan：<ms>
- custom scalar：<ms>
- custom 当前版本：<ms>
- max_abs_diff：<值>
profile：<粘贴摘要>

请先量化以下候选瓶颈：标量 GM GetValue/SetValue、b/c 搬运、Exp、ReduceSum、
blockDim 负载和 Queue 流水。按收益/风险排序，只实现排名第一且能独立验证的一项。

要求：
- 不改变算子接口和 recurrence 顺序；
- 优先把 N 维运算改为连续 DataCopy + LocalTensor 向量计算；
- 给出优化前后 GM 访问次数或字节数、UB 占用和核工作量变化；
- 重跑全部正确性矩阵；
- 用相同 warmup/repeat/sync 口径重跑性能；
- 性能没有改善时保留正确版本并解释数据，不得只报告最好的一次。
```

---

## 6. 另一个典型样例：CPU fallback 型算子

`BatchSpdInvFp32` 与 scan 算子的目标不同。它的提示词应首先证明原路径为什么不能留在 NPU，而不是先承诺高倍加速。

```text
请分析并开发 BatchSpdInvFp32，用于 Koopa KPLayer 的小型 SPD 矩阵求逆。
原路径在目标 torch_npu 环境调用 torch.linalg.lstsq 时发生 CPU fallback。

目标：
1. 保存 profiler/日志证据，证明原算子不支持 NPU或发生 fallback；
2. 将 KPLayer 中 G=x*x^T 后的 batched 小矩阵求逆留在 Ascend 910B3；
3. v1 只支持 float32、对称正定矩阵、m=<范围>，batch=<范围>；
4. 用 CPU float64 或 PyTorch 高精度结果作为 reference；
5. 采用适合小矩阵的 LDLT/Cholesky 类算法前，先写清正定性、pivot、epsilon 和失败行为；
6. batch 可分核，每个小矩阵的分解步骤保持顺序，估算完整矩阵和临时量的 UB 占用；
7. 测试单位阵、对角阵、随机 SPD、接近病态矩阵、非正定输入和主 shape；
8. 比较 residual ||A*A_inv-I||、max_abs_diff、NaN/Inf，不只比较一个输出元素；
9. 性能分别报告 isolated inversion 和实际 Koopa forecast 路径；
10. 结论首先表述为“消除 CPU fallback/使能全 NPU 路径”，只有公平实测后再写加速。

请先读取 Koopa 原实现和仓库相近的小矩阵算子，冻结接口与失败语义，再实施。
```

---

## 7. 每轮结束时要求大模型使用的汇报格式

把下面一段附在每个实施型提示词末尾，可以显著减少“只说做了、没有证据”的情况。

```text
本轮结束请按下面格式汇报：

【结论】PASS / PARTIAL / FAIL
【修改】文件路径 + 每个文件的作用
【命令】实际执行的完整命令
【结果】退出码、产物、误差、耗时等关键数字
【证据】日志或结果文件路径
【未验证】因环境或范围未验证的内容
【风险】仍支持不了的 shape/dtype/边界
【下一轮】唯一的最高优先级目标

没有实际运行的命令不要写成已通过；没有结果文件的数据不要自行补全。
```

---

## 8. 提交前验收清单

| 关卡 | 必须回答的问题 | 最低证据 |
|---|---|---|
| 价值 | 为什么不是继续使用框架算子或图编译 | profile、fallback 或 kernel launch 证据 |
| 契约 | 输入输出、shape、dtype、layout 是否唯一 | 规格表和 infer 规则 |
| 真值 | 谁定义正确结果 | 独立 CPU/PyTorch reference |
| 工程 | 各层接口是否一致 | op/Host/tiling/kernel/aclnn 对应表 |
| 编译 | 目标芯片是否真能构建 | 命令退出码和实际产物 |
| 运行 | 是否调用真实 NPU kernel | aclnn runtime smoke 输出 |
| 正确性 | 主 shape 和边界是否正确 | 误差矩阵、seed、NaN/Inf |
| 稳定性 | 多次运行和长序列是否稳定 | 重复运行与误差累计结果 |
| 性能 | 比较是否公平 | 同设备同 shape 的 cold/hot 数据 |
| 图模式 | 框架可成图时是否有更强基线 | 图模式结果或无法成图证据 |
| 图模式前提 | 是否区分原表达与人工等价改写 | converter 失败点、改写内容和两条路径说明 |
| 集成 | 是否改善真实模型路径 | component + block/layer/validation 数据 |
| 表述 | 是否准确限定证据范围 | 数据/权重性质和验证层级说明 |
| 复现 | 他人能否重跑 | 固化脚本、命令、环境和结果文件 |

只有“编译通过”不是算子开发完成；只有“输出非空”也不是正确性验证；只有 isolated kernel 加速更不能自动推出模型 E2E 加速。

---

## 9. 常见无效提示词及改法

| 无效写法 | 问题 | 改法 |
|---|---|---|
| “写一个最快的算子” | 没有 shape、基线和计时口径 | 给主 shape、框架路径、profile 和验收数据 |
| “照这个公式生成全部代码” | 缺少仓库/版本/API 事实 | 要求先读参考算子和本机头文件 |
| “测试一下有没有问题” | 没有可判定阈值 | 指定 reference、case 矩阵和误差阈值 |
| “编译报错了，修一下” | 缺少命令与完整日志 | 给环境、命令、退出码和第一条错误上下文 |
| “优化到 10 倍” | 容易诱导捏造和过拟合单 shape | 要求 profile 驱动、相同口径和正确性回归 |
| “做完整 E2E” | E2E 边界不清 | 明确 cell/layer/block/encoder/validation/full model |
| “CPU 比 NPU 快多少” | 可能不是用户真实替代路径 | 优先比较 eager/graph `torch_npu` 与 custom NPU |
| “一次性完成，不要解释” | 难以及时冻结错误接口 | 保留持续实施，但设置分阶段验收点 |

好的提示词不是越长越好，而是事实充分、边界明确、验收可执行。大模型最适合承担的是：快速理解相似代码、生成一致的工程改动、根据真实日志迭代、补齐测试矩阵和审查遗漏；开发者仍需对数学契约、目标硬件、真实工作负载和最终性能口径负责。
