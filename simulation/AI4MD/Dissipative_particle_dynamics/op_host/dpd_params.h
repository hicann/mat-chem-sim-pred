// dpd_params.h
// DPD算子参数定义头文件
// 定义主机端和内核端共享的数据结构

#ifndef DPD_PARAMS_H
#define DPD_PARAMS_H

#include <cstdint>
#include <vector>

// DPD模拟参数结构体（与内核端对齐）
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
    float kBT;          // 热浴能量（k_B * T）
    
    // 系统参数
    float box_size[3];  // 模拟盒子尺寸 [Lx, Ly, Lz]
    int32_t num_particles; // 粒子总数
    float particle_mass;   // 粒子质量（通常为1.0）
    
    // 随机数参数
    int32_t random_seed;   // 随机数种子
    
    // 性能参数
    int32_t num_cores;     // AI Core数量
    int32_t vector_len;    // 向量化长度
    
    // 输出控制
    int32_t output_freq;   // 输出频率（每隔多少步输出一次）
    bool enable_energy;    // 是否计算能量
    bool enable_pressure;  // 是否计算压力
    
    // 构造函数
    DpdParams() {
        dt = 0.01f;
        total_time = 1.0f;
        num_steps = 100;
        
        rc = 1.0f;
        a_ij = 25.0f;
        gamma = 4.5f;
        sigma = 3.0f;
        kBT = 1.0f;
        
        box_size[0] = 10.0f;
        box_size[1] = 10.0f;
        box_size[2] = 10.0f;
        num_particles = 1000;
        particle_mass = 1.0f;
        
        random_seed = 12345;
        num_cores = 32;
        vector_len = 16;
        
        output_freq = 10;
        enable_energy = true;
        enable_pressure = false;
    }
};

// 粒子数据容器
struct ParticleData {
    std::vector<float> positions;  // 位置数据 [x1,y1,z1, x2,y2,z2, ...]
    std::vector<float> velocities; // 速度数据 [vx1,vy1,vz1, vx2,vy2,vz2, ...]
    std::vector<float> forces;     // 力数据 [fx1,fy1,fz1, fx2,fy2,fz2, ...]
    std::vector<int32_t> types;    // 粒子类型
    
    // 构造函数
    ParticleData(int32_t num_particles = 0) {
        if (num_particles > 0) {
            resize(num_particles);
        }
    }
    
    // 调整大小
    void resize(int32_t num_particles) {
        positions.resize(num_particles * 3);
        velocities.resize(num_particles * 3);
        forces.resize(num_particles * 3);
        types.resize(num_particles);
    }
    
    // 获取粒子数量
    int32_t size() const {
        return static_cast<int32_t>(positions.size() / 3);
    }
    
    // 检查数据一致性
    bool validate() const {
        int32_t n = size();
        return (positions.size() == n * 3) &&
               (velocities.size() == n * 3) &&
               (forces.size() == n * 3) &&
               (types.size() == n);
    }
};

// 模拟结果结构体
struct DpdResult {
    // 最终状态
    ParticleData final_state;
    
    // 时间序列数据
    std::vector<float> time_points;
    std::vector<float> kinetic_energy;
    std::vector<float> potential_energy;
    std::vector<float> total_energy;
    std::vector<float> temperature;
    std::vector<float> pressure;
    
    // 性能统计
    float total_time;      // 总模拟时间（秒）
    float steps_per_second;// 每秒步数
    int32_t num_steps;     // 实际执行的步数
    
    // 错误信息
    bool success;
    std::string error_msg;
    
    // 构造函数
    DpdResult() : total_time(0.0f), steps_per_second(0.0f), num_steps(0), success(true) {}
};

// 邻居列表结构体（主机端）
struct HostNeighborList {
    std::vector<std::vector<int32_t>> indices;  // 每个粒子的邻居索引
    std::vector<std::vector<float>> distances;  // 对应的距离
    
    // 调整大小
    void resize(int32_t num_particles, int32_t max_neighbors = 50) {
        indices.resize(num_particles);
        distances.resize(num_particles);
        
        for (int32_t i = 0; i < num_particles; ++i) {
            indices[i].reserve(max_neighbors);
            distances[i].reserve(max_neighbors);
        }
    }
    
    // 清空所有邻居列表
    void clear() {
        for (auto& list : indices) list.clear();
        for (auto& list : distances) list.clear();
    }
    
    // 获取粒子数量
    int32_t size() const {
        return static_cast<int32_t>(indices.size());
    }
};

// 细胞列表结构体（主机端）
struct HostCellList {
    std::vector<int32_t> grid_size;      // 网格尺寸 [nx, ny, nz]
    std::vector<std::vector<int32_t>> cells;  // 每个细胞中的粒子索引
    std::vector<int32_t> particle_cell;  // 每个粒子所属的细胞索引
    
    // 构建细胞列表
    void build(const std::vector<float>& positions,
               const float box_size[3],
               float cutoff_radius);
    
    // 获取细胞索引
    int32_t get_cell_index(float x, float y, float z) const;
    
    // 获取相邻细胞
    std::vector<int32_t> get_neighbor_cells(int32_t cell_idx) const;
};

// 能量计算函数
float compute_kinetic_energy(const std::vector<float>& velocities, float mass = 1.0f);
float compute_temperature(const std::vector<float>& velocities, int32_t num_particles, float mass = 1.0f);

// 随机数生成器（用于随机力）
class DpdRandom {
private:
    uint32_t seed_;
    
public:
    DpdRandom(uint32_t seed = 12345) : seed_(seed) {}
    
    // 生成[0,1)均匀分布随机数
    float uniform();
    
    // 生成标准正态分布随机数
    float normal();
    
    // 设置种子
    void set_seed(uint32_t seed) { seed_ = seed; }
    
    // 获取当前种子
    uint32_t get_seed() const { return seed_; }
};

#endif // DPD_PARAMS_H