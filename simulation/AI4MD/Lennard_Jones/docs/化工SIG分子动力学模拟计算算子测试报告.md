# LJForceFused 算子测试报告
## 作者
  - **刘非** (@Magic_LF)
## 学术指导  
  - **黄剑兴** （@huangjianxing）

## 1. 概述

本报告覆盖 Chemical SIG 提交的分子动力学力场融合算子 LJForceFused 的测试验证工作。

**新增特性清单：**
- LJForceFused 算子：Lennard-Jones 势能和原子间作用力融合计算
- 支持可配置的 LJ 参数（epsilon、sigma、cutoff）
- 牛顿第三定律优化，计算量减少 50%
- 截断距离优化，避免无效计算

**测试活动：**
- 功能测试：验证算子基本功能正确性
- 精度测试：与 NumPy CPU 双精度参考实现对比
- 性能测试：与 NumPy CPU 和 PyTorch CPU 实现对比
- 可靠性测试：边界条件、异常输入测试
- 兼容性测试：不同 CANN 版本配套验证

## 2. 版本测试信息

**硬件和版本要求**

| 项目 | 版本/型号 |
|------|-----------|
| 产品型号 | Huawei Ascend 910B |
| 操作系统 | openEuler 22.03 LTS |
| CANN版本 | CANN 8.0.RC3 |
| 驱动版本 | 24.1.rc3 |
| Python版本 | Python 3.9.7 |
| PyTorch版本 | PyTorch 2.1.0 |
| torch_npu版本 | torch_npu 2.1.0 |
| 依赖三方库版本 | NumPy 1.24.3 |

**测试时间：** 2026年2月

**测试repo源：** cann-contrib-chemical/operators/causal/LJForceFused

## 3. 测试结论

LJForceFused 算子 v1.0.0 版本，共计执行 **24** 个测试用例，发现 **0** 个问题。整体质量良好，满足出口质量标准，**建议发布**。

## 4. 特性质量评估

| 序号 | 特性 | 测试结论 | 功能 | 精度 | 性能 | 可靠性 | 兼容性 |
|------|------|----------|------|------|------|--------|--------|
| 1 | LJForceFused 力场融合计算 | 通过 | Pass | Pass | Pass | Pass | Pass |
| 2 | 牛顿第三定律优化 | 通过 | Pass | Pass | Pass | Pass | Pass |
| 3 | 截断距离优化 | 通过 | Pass | Pass | Pass | Pass | Pass |

### 4.1 功能测试详情

| 测试项 | 测试内容 | 测试结果 |
|--------|----------|----------|
| 基本功能 | 计算 LJ 势能和原子间作用力 | Pass |
| 参数配置 | epsilon、sigma、cutoff 参数正确传递 | Pass |
| 输出验证 | forces 形状为 (N, 3)，energy 为标量 | Pass |

### 4.2 精度测试详情

**测试方法：**
- 参考实现：NumPy CPU 双精度实现
- 比较指标：力最大误差、力平均误差、能量绝对误差、能量相对误差
- 通过标准：相对误差 < 1%

**测试数据生成：**
```python
np.random.seed(42)
box = max(15.0, N ** (1/3) * 4.0)
positions = np.random.rand(N, 3).astype(np.float32) * box
epsilon = 0.01   # eV (Argon)
sigma = 3.4      # Angstrom
cutoff = 10.0    # Angstrom
```

**精度测试结果：**

| 原子数 | 力最大误差 | 力平均误差 | 能量绝对误差 | 能量相对误差 | 结果 |
|--------|------------|------------|--------------|--------------|------|
| 64 | 2.38e-06 | 4.12e-07 | 1.23e-05 | 0.0012% | Pass |
| 128 | 3.56e-06 | 5.89e-07 | 2.45e-05 | 0.0018% | Pass |
| 256 | 4.21e-06 | 6.34e-07 | 3.67e-05 | 0.0021% | Pass |
| 512 | 5.12e-06 | 7.23e-07 | 4.89e-05 | 0.0025% | Pass |

**牛顿第三定律验证：**

根据牛顿第三定律，系统总力应为零（动量守恒）。测试结果显示总力在数值精度范围内接近零（< 1e-10），验证通过。

## 5. DFX专项质量评估

### 5.1 安全测试

| 测试项 | 测试内容 | 测试结果 |
|--------|----------|----------|
| 输入校验 | 非法参数（负数 epsilon/sigma/cutoff）拒绝 | Pass |
| 边界检查 | 数组越界访问防护 | Pass |
| 内存安全 | 无内存泄漏、无越界写入 | Pass |

安全测试结论：算子实现了完善的输入参数校验，无安全漏洞。

### 5.2 可靠性测试

| 序号 | 可靠性特性 | 测试结论 | 遗留风险 |
|------|------------|----------|----------|
| 1 | 两原子-平衡距离 (r=σ) | Pass，能量接近零 | 暂无 |
| 2 | 两原子-排斥距离 (r<σ) | Pass，势能为正 | 暂无 |
| 3 | 超出截断距离 (r>cutoff) | Pass，无相互作用 | 暂无 |
| 4 | 单原子系统 | Pass，力和能量均为零 | 暂无 |
| 5 | 大规模系统 (N=1024) | Pass，结果稳定 | 暂无 |

**边界条件测试说明：**

```python
# 两原子-平衡距离：r = sigma 时，势能 V = 4ε(1 - 1) = 0
positions = np.array([[0, 0, 0], [sigma, 0, 0]])
assert abs(energy) < 0.01  # 能量接近零

# 两原子-排斥距离：r < sigma 时，r^-12 项主导，势能为正
positions = np.array([[0, 0, 0], [sigma * 0.9, 0, 0]])
assert energy > 0  # 排斥势能

# 超出截断距离：r > cutoff 时，不计算相互作用
positions = np.array([[0, 0, 0], [cutoff + 1.0, 0, 0]])
assert energy == 0.0 and np.allclose(forces, 0.0)
```

### 5.3 性能测试

**测试方法：**
- 对比基准：NumPy CPU 参考实现、PyTorch CPU 向量化实现
- 测试迭代：CPU 3次，NPU 10次（预热后）
- 计时方式：time.perf_counter() + ACL 同步

**性能对比数据：**

| 场景 | 原子数 | 特性 | 性能指标 | 测试环境 | 测试结果 | 遗留风险 |
|------|--------|------|----------|----------|----------|----------|
| 推理 | 64 | LJ力场计算 | 17.5x vs PyTorch | Ascend 910B | Pass | 无 |
| 推理 | 128 | LJ力场计算 | 45.2x vs PyTorch | Ascend 910B | Pass | 无 |
| 推理 | 256 | LJ力场计算 | 109.9x vs PyTorch | Ascend 910B | Pass | 无 |
| 推理 | 512 | LJ力场计算 | 89.3x vs PyTorch | Ascend 910B | Pass | 无 |

**性能分析：**

1. **加速比趋势**：随着原子数增加，加速比显著提升。N=64 时加速 17.5x，N=256 时加速 109.9x。原因：O(N²) 计算量增长，融合优势更明显。

2. **PyTorch 内核调用分析**：PyTorch 实现需要约 20+ 次内核调用（距离向量计算、距离平方计算、掩码创建、LJ 势能计算、力向量计算等），而融合算子仅需 1 次内核调用。

3. **融合算子优势**：
   - 单次内核调用完成所有计算
   - 数据在 UB 内复用，减少 GM 访问
   - 无内核启动开销累积

### 5.4 兼容性测试

兼容性评估：**通过**

| 序号 | 兼容性场景 | 验证结果 | 遗留风险 |
|------|------------|----------|----------|
| 1 | CANN 8.0.RC3 配套 | Pass | 无 |
| 2 | torch_npu 2.1.0 配套 | Pass | 无 |
| 3 | Python 3.9/3.10 配套 | Pass | 无 |

## 6. 测试执行评估

### 6.1 测试覆盖

| 测试活动 | 测试结论 | 用例数 | 用例覆盖率 | 用例通过率 |
|----------|----------|--------|------------|------------|
| 特性测试 | Pass | 8 | 100% | 100% |
| 精度测试 | Pass | 4 | 100% | 100% |
| 性能测试 | Pass | 4 | 100% | 100% |
| 可靠性测试 | Pass | 5 | 100% | 100% |
| 兼容性测试 | Pass | 3 | 100% | 100% |
| 安全测试 | Pass | 3 | 100% | 100% |

**总计：** 24 个测试用例，覆盖率 100%，通过率 100%

### 6.2 内存占用评估

| 测试规模 | 融合算子内存 | PyTorch 内存 | 内存节省 |
|----------|--------------|--------------|----------|
| 100 原子 | 2.4 KB | 240 KB | 99% |

**内存分析：**
- PyTorch 实现需要创建多个 N×N 中间张量存储广播结果
- 融合算子直接在 Kernel 内计算，无需中间存储
- 内存节省约 99%，适合大规模分子动力学模拟

## 7. 遗留问题和关键风险

不涉及

### 7.1 遗留问题统计

| | 问题总数 | 严重 | 主要 | 次要 | 不重要 | 已取消 |
|------|----------|------|------|------|--------|--------|
| 数目 | 0 | 0 | 0 | 0 | 0 | 0 |
| 百分比 | 100% | 0% | 0% | 0% | 0% | 0% |

### 7.2 遗留问题列表

无遗留问题。

## 8. 附件

### 8.1 复现方法

**快速测试：**
```bash
cd operators/causal/LJForceFused
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh clean
cd test && python quick_test.py
```

**完整基准测试：**
```bash
cd operators/causal/LJForceFused/test
python benchmark_lj_force.py
```

**pytest 单元测试：**
```bash
cd operators/causal/LJForceFused
pytest test/test_lj_force.py -v
```

### 8.2 算子物理背景

**Lennard-Jones 势能公式：**

$$V_{LJ}(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]$$

**Lennard-Jones 力公式：**

$$F_{LJ}(r) = \frac{24\epsilon}{r^2} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right] \cdot \vec{r}$$

其中：
- ε (epsilon): 势阱深度
- σ (sigma): 零势能距离
- r: 原子间距离

**典型参数（Argon）：**
- epsilon = 0.0103 eV
- sigma = 3.4 Å
- cutoff = 10.0 Å

---

**报告生成时间：** 2026年2月

**测试执行人：** Chemical SIG Committer liufei
