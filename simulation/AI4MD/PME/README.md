# PME — 长程静电 (Particle Mesh Ewald)

**类型**: NPU AIV Kernel | **依赖**: FFTW3 | **编译**: 可独立编译

## 功能

Ewald求和 + PME加速的长程静电计算:
- 实空间 (短程Ewald): 直接求和，含erfc衰减
- 倒易空间 (PME): B-spline插值 + 3D FFT
- 自能修正 + 分子内排除

## 文件结构

```
PME/
├── README.md
├── CMakeLists.txt       # 可独立编译
├── op_kernel/
│   ├── pme_kernel.cpp   # PME主kernel (spread + convolution)
│   ├── pme_spread.h     # B-spline电荷涂抹
│   ├── pme_fft3d.h      # 3D FFT (依赖FFTW3)
│   └── ewald_*.h/cpp    # Ewald实空间
├── op_host/
│   ├── pme_host2.h/.cpp # PME Host端
│   └── ewald_host.h/.cpp # Ewald Host端
└── docs/algorithm.md
```

## 独立编译

```bash
cd npu_ops/pme && mkdir build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
make -j4
```

## 快速使用

```cpp
#include "pme_host2.h"
#include "ewald_host.h"

// PME参数
PMEConfig cfg;
cfg.grid_x = 32; cfg.grid_y = 32; cfg.grid_z = 32;
cfg.order = 4;      // B-spline阶数
cfg.ewald_beta = 3.0;  // Ewald分裂参数
cfg.rcoulomb = 1.4;    // 实空间截断
cfg.box = {Lx, Ly, Lz};

// Host端
PMEHost2 pme;
pme.Initialize(cfg);
pme.ComputeForces(charges, coords, forces, &energy);
```

## 参考

算法详情: `docs/algorithm.md`
