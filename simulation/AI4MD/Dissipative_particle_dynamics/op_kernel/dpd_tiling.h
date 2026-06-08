// dpd_tiling.h
// DPD算子Tiling策略头文件
// 定义UB分块策略和并行计算参数

#ifndef DPD_TILING_H
#define DPD_TILING_H

#include <cstdint>

// UB内存容量配置（单位：字节）
constexpr int32_t UB_CAPACITY = 256 * 1024;  // 256KB UB
constexpr int32_t L1_CAPACITY = 32 * 1024;   // 32KB L1缓存

// 粒子数据结构大小
constexpr int32_t PARTICLE_SIZE = sizeof(float) * 10;  // 3位置 + 3速度 + 3力 + 1类型

// Tiling策略结构体
struct DpdTiling {
    // 粒子分块参数
    int32_t total_particles;    // 总粒子数
    int32_t block_size;         // 每个AI Core处理的粒子块大小
    int32_t num_blocks;         // 总块数
    int32_t block_idx;          // 当前块索引
    
    // 邻居列表参数
    int32_t max_neighbors;      // 最大邻居数
    int32_t neighbor_capacity;  // 邻居列表容量
    
    // 内存布局参数
    int32_t ub_particle_capacity;  // UB能容纳的粒子数
    int32_t l1_particle_capacity;  // L1能容纳的粒子数
    
    // 并行计算参数
    int32_t core_id;            // AI Core ID
    int32_t num_cores;          // 总AI Core数
    int32_t vector_len;         // 向量化长度
    
    // 性能优化参数
    int32_t double_buffer_size; // 双缓冲大小
    int32_t prefetch_distance;  // 预取距离
    
    // 验证参数
    int32_t checksum;           // 校验和
};

// Tiling计算函数
inline DpdTiling calculate_tiling(
    int32_t num_particles,
    int32_t num_cores = 32,
    int32_t vector_len = 16) {
    
    DpdTiling tiling;
    tiling.total_particles = num_particles;
    tiling.num_cores = num_cores;
    tiling.vector_len = vector_len;
    
    // 计算每个AI Core处理的粒子数
    int32_t particles_per_core = (num_particles + num_cores - 1) / num_cores;
    tiling.block_size = particles_per_core;
    tiling.num_blocks = (num_particles + particles_per_core - 1) / particles_per_core;
    
    // 计算UB能容纳的粒子数（预留空间给邻居粒子）
    tiling.ub_particle_capacity = UB_CAPACITY / (PARTICLE_SIZE * 2);
    
    // 计算L1能容纳的粒子数
    tiling.l1_particle_capacity = L1_CAPACITY / PARTICLE_SIZE;
    
    // 设置邻居列表参数（经验值：每个粒子平均30个邻居）
    tiling.max_neighbors = 30;
    tiling.neighbor_capacity = particles_per_core * tiling.max_neighbors;
    
    // 双缓冲配置
    tiling.double_buffer_size = tiling.ub_particle_capacity / 2;
    tiling.prefetch_distance = 2;
    
    return tiling;
}

// 邻居列表结构体
struct NeighborList {
    int32_t* indices;      // 邻居粒子索引
    float* distances;      // 邻居距离
    int32_t count;         // 邻居数量
    int32_t capacity;      // 列表容量
};

// 细胞列表结构体（用于加速邻居搜索）
struct CellList {
    int32_t grid_size[3];   // 网格尺寸
    int32_t* cell_starts;   // 细胞起始索引
    int32_t* cell_ends;     // 细胞结束索引
    int32_t* particle_cells; // 粒子所属细胞
    int32_t* cell_particles; // 细胞中的粒子列表
};

// 边界条件处理函数
inline void apply_periodic_boundary(
    float& x, float& y, float& z,
    const float box_size[3]) {
    
    // 周期性边界条件
    x = x - floor(x / box_size[0]) * box_size[0];
    y = y - floor(y / box_size[1]) * box_size[1];
    z = z - floor(z / box_size[2]) * box_size[2];
}

// 距离计算（考虑PBC）
inline float pbc_distance_squared(
    float x1, float y1, float z1,
    float x2, float y2, float z2,
    const float box_size[3]) {
    
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    
    // 最小镜像约定
    dx = dx - round(dx / box_size[0]) * box_size[0];
    dy = dy - round(dy / box_size[1]) * box_size[1];
    dz = dz - round(dz / box_size[2]) * box_size[2];
    
    return dx * dx + dy * dy + dz * dz;
}

#endif // DPD_TILING_H