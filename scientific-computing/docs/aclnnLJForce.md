# aclnnLJForce
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

- 接口功能：实现 Lennard-Jones 力场的融合计算，一次完成距离计算、势能计算和力向量计算，适用于分子动力学模拟场景。

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

## 函数原型

```Cpp
int32_t aclnnLJForceDirect(
    void*    positionsAddr,
    void*    forcesAddr,
    void*    energyAddr,
    int32_t  numAtoms,
    float    epsilon,
    float    sigma,
    float    cutoff,
    void*    workspaceAddr,
    uint64_t workspaceSize,
    void*    stream
);
```

```Cpp
uint64_t aclnnLJForceGetWorkspaceSize(int32_t numAtoms);
```

## aclnnLJForceDirect

- **参数说明**：

| 参数名 | 输入/输出 | 描述 | 数据类型 |
|--------|-----------|------|----------|
| positionsAddr | 输入 | 原子坐标数组，shape为[N, 3]，N为原子数。设备内存地址。 | FLOAT32 |
| forcesAddr | 输出 | 每个原子受到的力，shape为[N, 3]。设备内存地址。 | FLOAT32 |
| energyAddr | 输出 | 系统总势能数组，shape为[maxCores * 8]，需要在Host端累加。设备内存地址。 | FLOAT32 |
| numAtoms | 输入 | 原子数量N。 | INT32 |
| epsilon | 输入 | 势阱深度，单位eV。 | FLOAT32 |
| sigma | 输入 | 零势能距离，单位Angstrom。 | FLOAT32 |
| cutoff | 输入 | 截断距离，超过此距离的原子对不计算相互作用，单位Angstrom。 | FLOAT32 |
| workspaceAddr | 输入 | 在Device侧申请的workspace内存地址。 | void* |
| workspaceSize | 输入 | 在Device侧申请的workspace大小，由aclnnLJForceGetWorkspaceSize获取。 | UINT64 |
| stream | 输入 | 指定执行任务的Stream。 | aclrtStream |

- **返回值**：

  返回0表示成功，非0表示失败。

## aclnnLJForceGetWorkspaceSize

- **参数说明**：

| 参数名 | 输入/输出 | 描述 | 数据类型 |
|--------|-----------|------|----------|
| numAtoms | 输入 | 原子数量N。 | INT32 |

- **返回值**：

  返回需要在Device侧申请的workspace大小（字节）。

## 约束说明

- 输入坐标必须为 FLOAT32 类型
- 原子数 N 不超过 65535
- cutoff 必须大于 0
- 强制 FP32 精度计算，满足科学计算需求

## 调用示例

示例代码如下，仅供参考：

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"

extern "C" int32_t aclnnLJForceDirect(
    void* positionsAddr, void* forcesAddr, void* energyAddr,
    int32_t numAtoms, float epsilon, float sigma, float cutoff,
    void* workspaceAddr, uint64_t workspaceSize, void* stream
);
extern "C" uint64_t aclnnLJForceGetWorkspaceSize(int32_t numAtoms);

int main()
{
    // 1. 初始化设备
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 2. 构造输入数据
    const int32_t numAtoms = 64;
    const float epsilon = 0.0103f;  // eV (Argon)
    const float sigma = 3.4f;       // Angstrom
    const float cutoff = 10.0f;     // Angstrom

    std::vector<float> positions(numAtoms * 3);
    // 初始化原子坐标...

    // 3. 分配设备内存
    void *posAddr, *forcesAddr, *energyAddr, *workspaceAddr;
    uint64_t workspaceSize = aclnnLJForceGetWorkspaceSize(numAtoms);

    aclrtMalloc(&posAddr, numAtoms * 3 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&forcesAddr, numAtoms * 3 * sizeof(float) + 32 * 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&energyAddr, 32 * 8 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);

    // 4. 拷贝输入到设备
    aclrtMemcpy(posAddr, numAtoms * 3 * sizeof(float),
                positions.data(), numAtoms * 3 * sizeof(float),
                ACL_MEMCPY_HOST_TO_DEVICE);

    // 5. 调用算子
    aclnnLJForceDirect(posAddr, forcesAddr, energyAddr, numAtoms,
                       epsilon, sigma, cutoff, workspaceAddr, workspaceSize, stream);

    // 6. 同步等待
    aclrtSynchronizeStream(stream);

    // 7. 拷贝结果回主机并累加能量
    std::vector<float> energy(32 * 8, 0.0f);
    aclrtMemcpy(energy.data(), energy.size() * sizeof(float),
                energyAddr, 32 * 8 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    float totalEnergy = 0.0f;
    for (int i = 0; i < 32; i++) {
        totalEnergy += energy[i * 8];
    }

    // 8. 释放资源
    aclrtFree(posAddr);
    aclrtFree(forcesAddr);
    aclrtFree(energyAddr);
    aclrtFree(workspaceAddr);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

## 性能数据

| 原子数 | 原子对数 | PyTorch CPU | NPU融合算子 | 加速比 |
|--------|----------|-------------|-------------|--------|
| 64 | 2,016 | 0.54 ms | 0.57 ms | 0.96x |
| 128 | 8,128 | 25.57 ms | 0.75 ms | 34.21x |
| 256 | 32,640 | 174.96 ms | 0.85 ms | 206.23x |
| 512 | 130,816 | 183.00 ms | 1.45 ms | 126.36x |
