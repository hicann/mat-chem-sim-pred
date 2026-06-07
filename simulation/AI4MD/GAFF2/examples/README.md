# GAFF2 力场算子 — 使用示例

## 单元测试

`test_gaff2.cpp` — **NPU vs CPU 精度对比 + 性能** 单元测试

### 测试项

| # | 测试 | 类型 | 对比基准 |
|---|------|------|----------|
| 1 | 初始化/Finalize | 功能 | — |
| 2 | Bond力 NPU vs CPU | 精度 | CPU O(N_bond)谐键参考 |
| 3 | LJ力 NPU vs CPU | 精度 | CPU O(N²) LJ-12-6参考 |
| 4 | Coulomb力 NPU vs CPU | 精度 | CPU O(N²) Coulomb参考 + 解析解 |
| 5 | 性能基准 (64原子) | 性能 | 10次均值 |

### CPU参考实现

- **Bond**: 标准谐键 `E=½k(r-r₀)²`, `F=-2k(r-r₀)`
- **LJ-12-6**: `E=4ε[(σ/r)¹²-(σ/r)⁶]` 含Lorentz-Berthelot混合规则
- **Coulomb**: `E=qᵢqⱼ/(4πε₀r)`, ε₀=138.935485 kJ·nm/mol/e²

### 精度要求

| 算子 | 容差 | 指标 |
|------|------|------|
| Bond | < 1e-4 | 最大相对力误差 |
| LJ | < 1e-3 | 最大相对力误差 |
| Coulomb | < 1e-3 | 最大相对力误差, 能量匹配解析解 |

### 预期结果

```
GAFF2: 9 PASS / 0 FAIL / 9 TOTAL
```

### 性能参考 (Ascend 910B3)

| 规模 | 耗时 |
|------|------|
| 2原子×1键 | ~50 μs/step |
| 64原子×63键+NB | ~200 μs/step |

### 编译运行

```bash
cd ../../build && make test_gaff2 && ./npu_ops/gaff2/test_gaff2
```
