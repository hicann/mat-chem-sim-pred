# LJForceFused
## 作者
  - **刘非** (@Magic_LF)
## 学术指导  
  - **黄剑兴** （@huangjianxing）

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- 算子功能：实现 Lennard-Jones 力场的融合计算，一次完成距离计算、势能计算和力向量计算，适用于分子动力学模拟场景。

- 计算公式：

  Lennard-Jones 势能：

  $$
  V_{LJ}(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]
  $$

  Lennard-Jones 力：

  $$
  F_{LJ}(r) = \frac{24\varepsilon}{r} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right] \cdot \frac{\vec{r}}{|r|}
  $$

  其中：
  - $r$ 为两原子间距离
  - $\varepsilon$ 为势阱深度
  - $\sigma$ 为零势能距离
  - $\vec{r}$ 为距离向量

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
|--------|-----------|------|----------|----------|
| positions | 输入 | 原子坐标，shape为[N, 3]，N为原子数 | FLOAT32 | ND |
| epsilon | 属性 | 势阱深度，单位eV | FLOAT32 | - |
| sigma | 属性 | 零势能距离，单位Angstrom | FLOAT32 | - |
| cutoff | 属性 | 截断距离，超过此距离的原子对不计算相互作用 | FLOAT32 | - |
| forces | 输出 | 每个原子受到的力，shape为[N, 3] | FLOAT32 | ND |
| energy | 输出 | 系统总势能 | FLOAT32 | - |

## 约束说明

- 输入坐标必须为 FLOAT32 类型
- 原子数 N 不超过 65535
- cutoff 必须大于 0

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|----------|----------|------|
| aclnn接口 | [test_aclnn_lj_force](examples/test_aclnn_lj_force.cpp) | 通过 aclnnLJForce 接口调用 LJForceFused 算子 |

## 算子特性

| 特性 | 说明 |
|------|------|
| 融合计算 | 距离 + 势能 + 力向量一次完成，减少内存访问 |
| 多核并行 | 原子均匀分配到多个 AI Core 并行计算 |
| 截断优化 | 只计算 r < cutoff 的原子对 |
| 精度保证 | 强制 FP32 精度，满足科学计算需求 |

## 性能数据

| 原子数 | 原子对数 | PyTorch CPU | NPU融合算子 | 加速比 |
|--------|----------|-------------|-------------|--------|
| 64 | 2,016 | 0.54 ms | 0.57 ms | 0.96x |
| 128 | 8,128 | 25.57 ms | 0.75 ms | 34.21x |
| 256 | 32,640 | 174.96 ms | 0.85 ms | 206.23x |
| 512 | 130,816 | 183.00 ms | 1.45 ms | 126.36x |
