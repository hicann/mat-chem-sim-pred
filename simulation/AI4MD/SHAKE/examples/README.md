# SHAKE 键长约束算子 — 使用示例

## 单元测试

`test_shake.cpp` — **NPU vs CPU 精度对比 + 性能** 单元测试

### 测试项

| # | 测试 | 类型 | 对比基准 |
|---|------|------|----------|
| 1 | 配置 | 功能 | — |
| 2 | 拉伸键修正 NPU vs CPU | 精度 | CPU SHAKE迭代 (Lagrange乘子) |
| 3 | 已满足约束 | 精度 | 解析 r=r0 |
| 4 | 性能基准 (100次) | 性能 | 均值 |

### CPU参考实现

- 标准SHAKE迭代: 每约束计算Lagrange乘子 `g=dr/[(1/mᵢ+1/mⱼ)r]`
- 坐标修正: `xᵢ-=g·dx/mᵢ`, `xⱼ+=g·dx/mⱼ`

### 精度要求

| 指标 | 容差 |
|------|------|
| NPU vs CPU 键长 | < 1e-6 nm |
| 收敛后键长 vs r₀ | < 1e-4 nm |

### 预期结果

```
SHAKE: 6 PASS / 0 FAIL / 6 TOTAL
```

### 性能参考 (Ascend 910B3)

| 规模 | 耗时 |
|------|------|
| 2原子×1约束 | ~10 μs/Apply |

### 编译运行

```bash
cd ../../build && make test_shake && ./npu_ops/shake/test_shake
```
