# DPD算子API参考文档

## 概述

本文档提供DPD（耗散粒子动力学）Ascend C算子的完整API参考，包括主机端接口、内核函数、数据结构和使用示例。

## 1. 主机端API

### 1.1 核心类

#### `DpdSimulator` 类

DPD模拟器主类，提供完整的模拟功能。

**头文件**: `dpd_host.h`

**构造函数**:
```cpp
DpdSimulator();
```

**析构函数**:
```cpp
~DpdSimulator();
```

**主要方法**:

##### 初始化
```cpp
bool initialize(const DpdParams& params);
```
初始化模拟器。
- **参数**: `params` - DPD模拟参数
- **返回**: `true` 成功，`false` 失败

##### 从文件初始化
```cpp
bool initialize_from_file(const std::string& config_file);
```
从配置文件初始化。
- **参数**: `config_file` - JSON配置文件路径
- **返回**: `true` 成功，`false` 失败

##### 设置粒子数据
```cpp
void set_particle_data(const ParticleData& data);
```
设置粒子初始状态。
- **参数**: `data` - 粒子数据

##### 设置随机位置
```cpp
void set_random_positions();
```
随机初始化粒子位置（在模拟盒子内）。

##### 设置随机速度
```cpp
void set_random_velocities(float temperature = 1.0f);
```
按麦克斯韦分布初始化粒子速度。
- **参数**: `temperature` - 温度（k_B T单位）

##### 运行完整模拟
```cpp
DpdResult run_simulation();
```
运行完整模拟（参数中指定的步数）。
- **返回**: 模拟结果

##### 运行指定步数
```cpp
DpdResult run_step(int32_t num_steps = 1);
```
运行指定数量的时间步。
- **参数**: `num_steps` - 要运行的步数
- **返回**: 模拟结果

##### 获取当前状态
```cpp
ParticleData get_current_state() const;
```
获取当前粒子状态。
- **返回**: 粒子数据

##### 获取参数
```cpp
DpdParams get_parameters() const;
```
获取当前模拟参数。
- **返回**: DPD参数

##### 检查初始化状态
```cpp
bool is_initialized() const;
```
检查模拟器是否已初始化。
- **返回**: `true` 已初始化

##### 计算动能
```cpp
float compute_kinetic_energy() const;
```
计算系统总动能。
- **返回**: 动能值

##### 计算温度
```cpp
float compute_temperature() const;
```
计算系统温度。
- **返回**: 温度值（k_B T单位）

##### 重建邻居列表
```cpp
void rebuild_neighbor_list();
```
手动重建邻居列表。

##### 获取邻居列表
```cpp
const HostNeighborList& get_neighbor_list() const;
```
获取当前邻居列表。
- **返回**: 邻居列表引用

##### 获取错误信息
```cpp
std::string get_last_error() const;
```
获取最后错误信息。
- **返回**: 错误描述

##### 清空错误
```cpp
void clear_error();
```
清空错误状态。

### 1.2 数据结构

#### `DpdParams` 结构体

DPD模拟参数。

**定义**:
```cpp
struct DpdParams {
    // 时间参数
    float dt;           // 时间步长
    float total_time;   // 总模拟时间
    int32_t num_steps;  // 总步数
    
    // 力场参数
    float rc;           // 截断半径
    float a_ij;         // 保守力系数
    float gamma;        // 耗散系数
    float sigma;        // 随机力系数
    float kBT;          // 热浴能量
    
    // 系统参数
    float box_size[3];  // 模拟盒子尺寸 [Lx, Ly, Lz]
    int32_t num_particles; // 粒子总数
    float particle_mass;   // 粒子质量
    
    // 随机数参数
    int32_t random_seed;   // 随机数种子
    
    // 性能参数
    int32_t num_cores;     // AI Core数量
    int32_t vector_len;    // 向量化长度
    
    // 输出控制
    int32_t output_freq;   // 输出频率
    bool enable_energy;    // 能量计算
    bool enable_pressure;  // 压力计算
    
    // 构造函数
    DpdParams();
};
```

**默认值**:
```cpp
dt = 0.01f
rc = 1.0f
a_ij = 25.0f
gamma = 4.5f
sigma = 3.0f
kBT = 1.0f
box_size = [10.0f, 10.0f, 10.0f]
num_particles = 1000
particle_mass = 1.0f
random_seed = 12345
num_cores = 32
vector_len = 16
output_freq = 10
enable_energy = true
enable_pressure = false
```

#### `ParticleData` 结构体

粒子数据容器。

**定义**:
```cpp
struct ParticleData {
    std::vector<float> positions;  // 位置 [x1,y1,z1, x2,y2,z2, ...]
    std::vector<float> velocities; // 速度 [vx1,vy1,vz1, vx2,vy2,vz2, ...]
    std::vector<float> forces;     // 力 [fx1,fy1,fz1, fx2,fy2,fz2, ...]
    std::vector<int32_t> types;    // 粒子类型
    
    // 构造函数
    ParticleData(int32_t num_particles = 0);
    
    // 方法
    void resize(int32_t num_particles);
    int32_t size() const;
    bool validate() const;
};
```

**方法**:
- `resize()`: 调整数据大小
- `size()`: 获取粒子数量
- `validate()`: 验证数据一致性

#### `DpdResult` 结构体

模拟结果。

**定义**:
```cpp
struct DpdResult {
    // 最终状态
    ParticleData final_state;
    
    // 时间序列
    std::vector<float> time_points;
    std::vector<float> kinetic_energy;
    std::vector<float> potential_energy;
    std::vector<float> total_energy;
    std::vector<float> temperature;
    std::vector<float> pressure;
    
    // 性能统计
    float total_time;      // 总模拟时间（秒）
    float steps_per_second;// 每秒步数
    int32_t num_steps;     // 实际步数
    
    // 状态
    bool success;
    std::string error_msg;
    
    // 构造函数
    DpdResult();
};
```

### 1.3 工具函数

#### 文件IO
```cpp
namespace dpd_utils {
    bool save_particle_data(const ParticleData& data, const std::string& filename);
    bool load_particle_data(ParticleData& data, const std::string& filename);
    
    bool save_simulation_result(const DpdResult& result, const std::string& filename);
    bool load_simulation_result(DpdResult& result, const std::string& filename);
}
```

#### 数据生成
```cpp
namespace dpd_utils {
    ParticleData create_cubic_lattice(int32_t particles_per_side, float spacing);
    ParticleData create_random_system(int32_t num_particles, const float box_size[3]);
}
```

#### 参数验证
```cpp
namespace dpd_utils {
    bool validate_parameters(const DpdParams& params);
    bool validate_particle_data(const ParticleData& data, const DpdParams& params);
}
```

### 1.4 C风格接口

#### 创建和销毁
```cpp
void* dpd_create_simulator();
void dpd_destroy_simulator(void* simulator);
```

#### 初始化
```cpp
int dpd_initialize(void* simulator, const DpdParams* params);
int dpd_initialize_from_file(void* simulator, const char* config_file);
```

#### 数据设置
```cpp
void dpd_set_particle_data(void* simulator, const ParticleData* data);
void dpd_set_random_positions(void* simulator);
void dpd_set_random_velocities(void* simulator, float temperature);
```

#### 运行模拟
```cpp
DpdResult* dpd_run_simulation(void* simulator);
DpdResult* dpd_run_step(void* simulator, int32_t num_steps);
```

#### 状态获取
```cpp
ParticleData* dpd_get_current_state(void* simulator);
DpdParams* dpd_get_parameters(void* simulator);
int dpd_is_initialized(void* simulator);
```

#### 内存管理
```cpp
void dpd_free_particle_data(ParticleData* data);
void dpd_free_result(DpdResult* result);
void dpd_free_params(DpdParams* params);
```

## 2. 内核端API

### 2.1 核函数

#### `dpd_kernel` 核函数

DPD计算主核函数。

**定义**:
```cpp
extern "C" __global__ __aicore__ void dpd_kernel(
    uint8_t* pos_gm,        // 位置数据（输入/输出）
    uint8_t* vel_gm,        // 速度数据（输入/输出）
    uint8_t* force_gm,      // 力数据（输出）
    uint8_t* params_gm,     // 模拟参数
    uint8_t* tiling_gm      // Tiling参数
);
```

**参数说明**:
- `pos_gm`: 全局内存中的位置数据，格式为 `[x1,y1,z1, x2,y2,z2, ...]`
- `vel_gm`: 全局内存中的速度数据，格式同上
- `force_gm`: 全局内存中的力数据，格式同上
- `params_gm`: 模拟参数结构体
- `tiling_gm`: Tiling策略结构体

### 2.2 数据结构

#### `Particle` 结构体

内核端粒子数据结构。

**定义**:
```cpp
struct alignas(32) Particle {
    fp32_t x, y, z;     // 位置
    fp32_t vx, vy, vz;  // 速度
    fp32_t fx, fy, fz;  // 力
    fp32_t type;        // 粒子类型
};
```

#### `SimParams` 结构体

内核端模拟参数。

**定义**:
```cpp
struct SimParams {
    fp32_t dt;          // 时间步长
    fp32_t rc;          // 截断半径
    fp32_t a_ij;        // 保守力系数
    fp32_t gamma;       // 耗散系数
    fp32_t sigma;       // 随机力系数
    fp32_t kBT;         // 热浴能量
    fp32_t box_size[3]; // 模拟盒子尺寸
    int32_t num_particles; // 粒子总数
    int32_t seed;       // 随机数种子
};
```

#### `DpdTiling` 结构体

Tiling策略。

**定义**:
```cpp
struct DpdTiling {
    // 粒子分块
    int32_t total_particles;    // 总粒子数
    int32_t block_size;         // 块大小
    int32_t num_blocks;         // 总块数
    int32_t block_idx;          // 当前块索引
    
    // 邻居列表
    int32_t max_neighbors;      // 最大邻居数
    int32_t neighbor_capacity;  // 邻居容量
    
    // 内存布局
    int32_t ub_particle_capacity;  // UB粒子容量
    int32_t l1_particle_capacity;  // L1粒子容量
    
    // 并行计算
    int32_t core_id;            // AI Core ID
    int32_t num_cores;          // 总AI Core数
    int32_t vector_len;         // 向量化长度
    
    // 性能优化
    int32_t double_buffer_size; // 双缓冲大小
    int32_t prefetch_distance;  // 预取距离
    
    // 验证
    int32_t checksum;           // 校验和
};
```

### 2.3 辅助函数

#### 距离计算
```cpp
__aicore__ inline fp32_t distance_squared(
    fp32_t dx, fp32_t dy, fp32_t dz);
```
计算距离平方。

#### 周期性边界距离
```cpp
__aicore__ inline fp32_t pbc_distance(
    fp32_t dx, fp32_t dy, fp32_t dz,
    const fp32_t box_size[3]);
```
考虑PBC的距离计算。

#### 应用周期性边界
```cpp
__aicore__ inline void apply_pbc(
    fp32_t& x, fp32_t& y, fp32_t& z,
    const fp32_t box_size[3]);
```
应用周期性边界条件。

#### DPD力计算
```cpp
__aicore__ inline void compute_dpd_forces(
    Particle& pi,
    Particle& pj,
    const SimParams& params,
    __ub__ fp32_t* rand_buf);
```
计算两个粒子间的DPD作用力。

#### Velocity-Verlet第一步
```cpp
__aicore__ inline void velocity_verlet_step1(
    Particle& p,
    const SimParams& params);
```
Velocity-Verlet积分第一步。

#### Velocity-Verlet第二步
```cpp
__aicore__ inline void velocity_verlet_step2(
    Particle& p,
    const SimParams& params);
```
Velocity-Verlet积分第二步。

#### 向量化加载
```cpp
template<int VEC_LEN>
__aicore__ inline void vec_load_particles(
    __gm__ uint8_t* gm_addr,
    __ub__ Particle* ub_buf,
    int32_t offset,
    int32_t num_particles);
```
向量化加载粒子数据。

#### 向量化存储
```cpp
template<int VEC_LEN>
__aicore__ inline void vec_store_particles(
    __ub__ Particle* ub_buf,
    __gm__ uint8_t* gm_addr,
    int32_t offset,
    int32_t num_particles);
```
向量化存储粒子数据。

## 3. Python API

### 3.1 安装

```bash
pip install dpd-op
```

### 3.2 基本使用

```python
import dpd_op

# 创建模拟器
simulator = dpd_op.DpdSimulator()

# 设置参数
params = dpd_op.DpdParams()
params.num_particles = 1000
params.box_size = [10.0, 10.0, 10.0]
params.dt = 0.01
params.num_steps = 100

# 初始化
simulator.initialize(params)

# 设置初始条件
simulator.set_random_positions()
simulator.set_random_velocities(temperature=1.0)

# 运行模拟
result = simulator.run_simulation()

if result.success:
    print(f"模拟完成: {result.total_time:.2f}秒")
    print(f"性能: {result.steps_per_second:.1f}步/秒")
    
    # 获取最终状态
    final_state = result.final_state
    positions = np.array(final_state.positions).reshape(-1, 3)
    velocities = np.array(final_state.velocities).reshape(-1, 3)
    
    print(f"最终温度: {simulator.compute_temperature():.3f}")
else:
    print(f"模拟失败: {result.error_msg}")
```

### 3.3 高级功能

#### 逐步模拟
```python
# 逐步运行
for step in range(100):
    result = simulator.run_step(1)
    if not result.success:
        break
    
    if step % 10 == 0:
        ke = simulator.compute_kinetic_energy()
        temp = simulator.compute_temperature()
        print(f"步 {step}: 动能={ke:.3f}, 温度={temp:.3f}")
```

#### 能量监控
```python
# 启用能量输出
simulator.enable_energy_output(True)

# 运行模拟并记录能量
result = simulator.run_simulation()

if result.success and hasattr(result, 'kinetic_energy'):
    import matplotlib.pyplot as plt
    
    plt.plot(result.time_points, result.kinetic_energy)
    plt.xlabel('时间')
    plt.ylabel('动能')
    plt.title('DPD模拟能量演化')
    plt.show()
```

#### 邻居列表访问
```python
# 重建邻居列表
simulator.rebuild_neighbor_list()

# 获取邻居列表
neighbor_list = simulator.get_neighbor_list()

# 访问特定粒子的邻居
particle_idx = 0
neighbors = neighbor_list.indices[particle_idx]
distances = neighbor_list.distances[particle_idx]

print(f"粒子 {particle_idx} 有 {len(neighbors)} 个邻居")
for i, (n_idx, dist) in enumerate(zip(neighbors, distances)):
    print(f"  邻居 {i}: 索引={n_idx}, 距离={dist:.3f}")
```

## 4. 配置参考

### 4.1 JSON配置文件格式

```json
{
  "dpd_params": {
    "dt": 0.01,
    "rc": 1.0,
    "a_ij": 25.0,
    "gamma": 4.5,
    "sigma": 3.0,
    "kBT": 1.0,
    "box_size": [10.0, 10.0, 10.0],
    "num_particles": 1000,
    "particle_mass": 1.0,
    "random_seed": 12345,
    "num_steps": 100,
    "output_freq": 10,
    "enable_energy": true,
    "enable_pressure": false
  },
  "initialization": {
    "random_positions": true,
    "random_velocities": true,
    "initial_temperature": 1.0
  },
  "performance": {
    "num_cores": 32,
    "vector_len": 16,
    "double_buffer": true
  }
}
```

### 4.2 YAML配置文件格式

```yaml
dpd_params:
  dt: 0.01
  rc: 1.0
  a_ij: 25.0
  gamma: 4.5
  sigma: 3.0
  kBT: 1.0
  box_size: [10.0, 10.0, 10.0]
  num_particles: 1000
  particle_mass: 1.0
  random_seed: 12345
  num_steps: 100
  output_freq: 10
  enable_energy: true
  enable_pressure: false

initialization:
  random_positions: true
  random_velocities: true
  initial_temperature: 1.0

performance:
  num_cores: 32
  vector_len: 16
  double_buffer: true
```

## 5. 错误代码

### 5.1 错误类型

#### 初始化错误
- `DPD_ERROR_INIT_FAILED`: 初始化失败
- `DPD_ERROR_INVALID_PARAMS`: 参数无效
- `DPD_ERROR_MEMORY_ALLOC`: 内存分配失败

#### 运行时错误
- `DPD_ERROR_DEVICE_UNAVAILABLE`: 设备不可用
- `DPD_ERROR_KERNEL_LAUNCH`: 核函数启动失败
- `DPD_ERROR_SIMULATION_FAILED`: 模拟失败

#### 数据错误
- `DPD_ERROR_INVALID_DATA`: 数据无效
- `DPD_ERROR_DATA_SIZE_MISMATCH`: 数据大小不匹配
- `DPD_ERROR_FILE_IO`: 文件IO错误

### 5.2 错误处理示例

```cpp
try {
    DpdSimulator simulator;
    DpdParams params;
    
    if (!simulator.initialize(params)) {
        std::string error = simulator.get_last_error();
        std::cerr << "初始化失败: " << error << std::endl;
        return;
    }
    
    DpdResult result = simulator.run_simulation();
    if (!result.success) {
        std::cerr << "模拟失败: " << result.error_msg << std::endl;
        return;
    }
    
    // 处理成功结果
    process_result(result);
    
} catch (const DpdException& e) {
    std::cerr << "DPD异常: " << e.what() << std::endl;
} catch (const std::exception& e) {
    std::cerr << "标准异常: " << e.what() << std::endl;
}
```

## 6. 性能调优API

### 6.1 性能参数设置

```cpp
// 设置AI Core数量
params.num_cores = 64;  // 使用64个AI Core

// 设置向量化长度
params.vector_len = 32;  // 使用32位向量

// 启用双缓冲
simulator.enable_double_buffer(true);

// 设置预取距离
simulator.set_prefetch_distance(2);
```

### 6.2 性能监控

```cpp
// 获取性能统计
float sim_time = simulator.get_simulation_time();
int32_t steps = simulator.get_completed_steps();
float steps_per_sec = simulator.get_steps_per_second();

// 获取详细性能数据
PerformanceStats stats = simulator.get_performance_stats();
std::cout << "计算时间: " << stats.compute_time << "秒" << std::endl;
std::cout << "搬运时间: " << stats.memory_time << "秒" << std::endl;
std::cout << "同步时间: " << stats.sync_time << "秒" << std::endl;
std::cout << "计算占比: " << stats.compute_ratio * 100 << "%" << std::endl;
```

## 7. 扩展API

### 7.1 自定义力场

```cpp
// 自定义力场接口
class CustomForceField {
public:
    virtual void compute_forces(ParticleData& data, 
                               const DpdParams& params) = 0;
    virtual ~CustomForceField() = default;
};

// 注册自定义力场
simulator.register_force_field(std::make_shared<MyForceField>());
```

### 7.2 自定义积分器

```cpp
// 自定义积分器接口
class CustomIntegrator {
public:
    virtual void integrate(ParticleData& data,
                          const DpdParams& params,
                          float dt) = 0;
    virtual ~CustomIntegrator() = default;
};

// 注册自定义积分器
simulator.register_integrator(std::make_shared<MyIntegrator>());
```

### 7.3 回调函数

```cpp
// 步进回调
simulator.set_step_callback([](int step, float time, 
                              const ParticleData& data) {
    std::cout << "步 " << step << ", 时间 " << time << std::endl;
    return true;  // 返回false可停止模拟
});

// 能量回调
simulator.set_energy_callback([](float ke, float pe, float te) {
    std::cout << "能量: KE=" << ke << ", PE=" << pe << ", TE=" << te << std::endl;
});
```

## 8. 版本兼容性

### 8.1 API版本

```cpp
// 获取API版本
std::string version = dpd_op::get_version();
std::cout << "DPD算子版本: " << version << std::endl;

// 检查兼容性
if (dpd_op::check_compatibility("1.0.0")) {
    std::cout << "API兼容" << std::endl;
} else {
    std::cout << "API不兼容，需要升级" << std::endl;
}
```

### 8.2 向后兼容性

- **v1.0.x**: 基础API，稳定版本
- **v1.1.x**: 添加性能监控API
- **v2.0.x**: 重大更新，API有变化

## 9. 最佳实践

### 9.1 内存管理

```cpp
// 使用智能指针管理资源
auto simulator = std::make_unique<DpdSimulator>();
auto params = std::make_unique<DpdParams>();

// 及时释放大内存
{
    ParticleData large_data(1000000);
    // 使用数据
}  // large_data在此自动释放
```

### 9.2 错误处理

```cpp
// 使用RAII包装器
class DpdSimulatorWrapper {
public:
    DpdSimulatorWrapper() : simulator_(std::make_unique<DpdSimulator>()) {}
    
    bool initialize(const DpdParams& params) {
        if (!simulator_->initialize(params)) {
            last_error_ = simulator_->get_last_error();
            return false;
        }
        return true;
    }
    
    const std::string& get_last_error() const { return last_error_; }
    
private:
    std::unique_ptr<DpdSimulator> simulator_;
    std::string last_error_;
};
```

### 9.3 性能优化

```cpp
// 批量处理
const int BATCH_SIZE = 1000;
for (int i = 0; i < total_steps; i += BATCH_SIZE) {
    int steps = std::min(BATCH_SIZE, total_steps - i);
    auto result = simulator.run_step(steps);
    // 处理结果
}

// 重用内存
ParticleData reusable_data;
reusable_data.resize(max_particles);

// 异步执行
auto future = std::async(std::launch::async, [&]() {
    return simulator.run_simulation();
});
// 执行其他任务
auto result = future.get();
```

## 10. 示例代码

### 10.1 完整C++示例

```cpp
#include "dpd_host.h"
#include <iostream>
#include <chrono>

int main() {
    try {
        // 1. 创建模拟器
        DpdSimulator simulator;
        
        // 2. 设置参数
        DpdParams params;
        params.num_particles = 10000;
        params.box_size[0] = params.box_size[1] = params.box_size[2] = 20.0f;
        params.dt = 0.01f;
        params.num_steps = 1000;
        params.output_freq = 100;
        
        // 3. 初始化
        if (!simulator.initialize(params)) {
            std::cerr << "初始化失败: " << simulator.get_last_error() << std::endl;
            return 1;
        }
        
        // 4. 设置初始条件
        simulator.set_random_positions();
        simulator.set_random_velocities(1.0f);
        
        // 5. 运行模拟
        auto start = std::chrono::steady_clock::now();
        DpdResult result = simulator.run_simulation();
        auto end = std::chrono::steady_clock::now();
        
        // 6. 处理结果
        if (result.success) {
            auto duration = std::chrono::duration<double>(end - start);
            
            std::cout << "模拟成功!" << std::endl;
            std::cout << "模拟时间: " << result.total_time << "秒" << std::endl;
            std::cout << "实际耗时: " << duration.count() << "秒" << std::endl;
            std::cout << "性能: " << result.steps_per_second << "步/秒" << std::endl;
            std::cout << "最终温度: " << simulator.compute_temperature() << std::endl;
            
            // 保存结果
            dpd_utils::save_simulation_result(result, "simulation_result.json");
            
        } else {
            std::cerr << "模拟失败: " << result.error_msg << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
```

### 10.2 完整Python示例

```python
import dpd_op
import numpy as np
import matplotlib.pyplot as plt

def run_dpd_simulation():
    """运行DPD模拟并分析结果"""
    
    # 创建模拟器
    simulator = dpd_op.DpdSimulator()
    
    # 设置参数
    params = dpd_op.DpdParams()
    params.num_particles = 5000
    params.box_size = [15.0, 15.0, 15.0]
    params.dt = 0.01
    params.num_steps = 500
    params.output_freq = 10
    
    # 初始化
    if not simulator.initialize(params):
        print(f"初始化失败: {simulator.get_last_error()}")
        return
    
    # 设置初始条件
    simulator.set_random_positions()
    simulator.set_random_velocities(temperature=1.0)
    
    # 运行模拟
    result = simulator.run_simulation()
    
    if result.success:
        print(f"模拟成功!")
        print(f"时间: {result.total_time:.2f}秒")
        print(f"性能: {result.steps_per_second:.1f}步/秒")
        print(f"最终温度: {simulator.compute_temperature():.3f}")
        
        # 绘制能量演化
        if hasattr(result, 'kinetic_energy') and result.kinetic_energy:
            plt.figure(figsize=(10, 6))
            plt.plot(result.time_points, result.kinetic_energy)
            plt.xlabel('时间')
            plt.ylabel('动能')
            plt.title('DPD模拟能量演化')
            plt.grid(True)
            plt.savefig('energy_evolution.png')
            plt.show()
            
        # 保存最终状态
        final_state = result.final_state
        positions = np.array(final_state.positions).reshape(-1, 3)
        np.save('final_positions.npy', positions)
        
    else:
        print(f"模拟失败: {result.error_msg}")

if __name__ == "__main__":
    run_dpd_simulation()
```

---

**文档版本**: 1.0.0  
**最后更新**: 2026-04-06  
**API版本**: v1.0.0  
**兼容性**: CANN 5.0+，PyTorch 1.8+  

**更多信息**:  
- 示例代码: `examples/` 目录  
- 测试用例: `tests/` 目录  
- 问题反馈: GitHub Issues