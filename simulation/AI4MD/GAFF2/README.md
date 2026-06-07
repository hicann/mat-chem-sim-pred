# GAFF2 — 力场计算

**类型**: NPU AIV Kernel | **依赖**: 无 | **编译**: 可独立编译

## 功能

实现 GAFF2 (General AMBER Force Field 2) 全部5项势能：
- 键伸缩 (Bond Stretch)
- 键角弯曲 (Angle Bend)
- 二面角扭转 (Dihedral Torsion)
- Lennard-Jones 12-6
- 库仑静电 (Coulomb)

## 文件结构

```
GAFF2/
├── README.md            # 本文件
├── CMakeLists.txt       # 可独立编译 (含CANN引导)
├── op_kernel/           # Ascend C NPU kernel
│   ├── gaff2_force.cpp  # 单tile全遍历力计算
│   └── gaff2_force.h
├── op_host/             # Host端封装
│   ├── gaff2_host.h / .cpp   # Kernel启动 + 数据管理
│   ├── gaff2_def.cpp         # 力场参数定义
│   └── gen_inc.py            # kernel .so → .inc 转换
├── docs/algorithm.md    # 算法详解 (公式 + 精度)
└── examples/README.md   # 使用示例
```

## 独立编译

```bash
cd npu_ops/gaff2 && mkdir build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
make -j4
# → libgaff2_force_host.a + libgaff2_force_kernel.so
```

## 快速使用

```cpp
#include "gaff2_host.h"
#include "gaff2_types.h"

// 1. 初始化
GAFF2Config cfg;
cfg.num_atoms    = n_atoms;
cfg.num_bonds    = n_bonds;
cfg.num_angles   = n_angles;
cfg.num_dihedrals = n_dihedrals;
cfg.cutoff        = 1.4;   // nm
cfg.coulomb_cutoff = 1.4;  // nm

GAFF2Host host;
host.Initialize(cfg);

// 2. 上传数据
host.UploadCoords(coords, n_atoms);
host.UploadParams(bonds, angles, dihedrals, type_params);

// 3. 计算力
host.LaunchForce(stream);
// → forces[n_atoms][3], energy
host.DownloadForces(forces, &energy);

// 4. 清理
host.Finalize();
```

## 关键约束

- 力约定: F = -∇E (GROMACS兼容)
- PBC最小镜像内联处理
- 1-4对缩放: LJ×0.5, Coulomb×0.83333
- NPU精度: 数学函数误差 < 1×10⁻⁷

## 参考

算法详情: `docs/algorithm.md`
