# SHAKE — 键长约束

**类型**: NPU AIV Kernel (迭代) | **依赖**: 无 | **编译**: 可独立编译

## 功能

迭代键长约束算法，固定含氢键长（如C-H, O-H）:
- 对每对约束原子施加位置修正
- 迭代至所有约束满足容差 (默认1×10⁻⁴ nm)
- 与Velocity Verlet配合使用(先积分→后约束)

## 文件结构

```
SHAKE/
├── README.md
├── CMakeLists.txt
├── op_kernel/
│   ├── shake.cpp       # NPU迭代kernel
│   └── shake_kernel.h
├── op_host/
│   ├── shake_host.h/.cpp  # Host端 + 收敛检查
│   └── shake.h            # CPU参考实现 (header-only)
└── docs/algorithm.md
```

## 独立编译

```bash
cd npu_ops/shake && mkdir build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
make -j4
```

## 快速使用

```cpp
#include "shake_host.h"

SHAKEConfig cfg;
cfg.num_constraints = n_bonds;
cfg.max_iter = 100;
cfg.tolerance = 1e-4;

SHAKEHost shake;
shake.Initialize(cfg);
shake.UploadConstraints(constraint_pairs, bond_lengths);
shake.Apply(coords, masses, dt);  // 约束坐标
```

## 参考

算法详情: `docs/algorithm.md`
