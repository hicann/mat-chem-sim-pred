# Velocity Verlet 积分器 — 使用示例

## 单元测试

`test_vv.cpp` — **NPU vs CPU解析 精度对比 + 性能** 单元测试

### 测试项

| # | 测试 | 类型 | 对比基准 |
|---|------|------|----------|
| 1 | 初始化 | 功能 | — |
| 2 | NVE 100步 NPU vs CPU | 精度 | CPU VV半步+全步参考 |
| 3 | 性能基准 (100步) | 性能 | 均值 |

### CPU参考实现

- 标准VV半步更新: `v(t+dt/2)=v(t)+½dt·F(t)/m`
- 全步位置: `x(t+dt)=x(t)+dt·v(t+dt/2)`
- 力重新计算: GAFF2 ComputeForces
- 全步速度: `v(t+dt)=v(t+dt/2)+½dt·F(t+dt)/m`

### 精度要求

| 指标 | 容差 |
|------|------|
| NVE ΔE/E | < 0.1% over 100 steps |
| NPU vs CPU 位置 | < 1e-3 nm after 100 steps |

### 预期结果

```
VV: 4 PASS / 0 FAIL / 4 TOTAL
```

### 性能参考 (Ascend 910B3)

| 规模 | 耗时 |
|------|------|
| 2原子 NVE (含GAFF2力) | ~100 μs/step |

### 编译运行

```bash
cd ../../build && make test_vv && ./npu_ops/velocity-verlet/test_vv
```
