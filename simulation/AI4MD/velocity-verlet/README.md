# Velocity Verlet — 时间积分

**类型**: NPU AIV Kernel × 3 | **依赖**: 无 | **编译**: 可独立编译

## 功能

标准3步Velocity Verlet积分器 + NPT扩展:
- `vv_integrate`: 半步速度更新 + 全步位置更新 (v(t+dt/2), x(t+dt))
- `vv_finish`: 力评估后全步速度完成 (v(t+dt))
- `thermo_scale`: NPT恒温/恒压速度+盒子缩放

## 文件结构

```
Velocity_Verlet/
├── README.md
├── CMakeLists.txt
├── op_kernel/
│   ├── vv_integrate.cpp / .h   # 半步+位置
│   ├── vv_finish.cpp / .h      # 速度完成
│   └── thermo_scale.cpp / .h   # 恒温/恒压
├── op_host/
│   ├── vv_host.h/.cpp          # Host端统一接口
│   ├── velocity_verlet.h       # 状态结构体
│   └── gen_vv_inc.py           # kernel→header转换
└── docs/algorithm.md
```

## 独立编译

```bash
cd npu_ops/velocity-verlet && mkdir build && cd build
cmake .. -DASCEND_CANN_PACKAGE_PATH=/path/to/cann
make -j4
# → libvv_host.a + 3个kernel .so
```

## 快速使用

```cpp
#include "vv_host.h"
#include "velocity_verlet.h"

VVConfig cfg;
cfg.num_atoms = n_atoms;
cfg.dt = 0.001;  // ps (1 fs)

VVHost vv;
vv.Initialize(cfg);

// 每步: 积分 → 算力 → 完成
vv.Integrate(coords, velocities, masses, forces, dt, box);
// ... 计算新力 ...
vv.Finish(velocities, masses, forces, dt);

// NPT: 需要额外恒温/恒压缩放
vv.ThermoScale(velocities, box, T_target, P_target, tau_t, tau_p);
```

## 参考

算法详情: `docs/algorithm.md`
