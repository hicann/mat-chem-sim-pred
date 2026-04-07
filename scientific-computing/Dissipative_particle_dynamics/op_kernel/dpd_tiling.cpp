// dpd_tiling.cpp
// DPD算子Tiling策略实现
// 实现粒子分块、邻居搜索、内存布局优化

#include "dpd_tiling.h"
#include <cmath>
#include <algorithm>

// 细胞列表构建函数
CellList build_cell_list(
    const float* positions,
    int32_t num_particles,
    const float box_size[3],
    float cutoff_radius) {
    
    CellList cell_list;
    
    // 计算网格尺寸（每个方向至少2个细胞）
    for (int i = 0; i < 3; ++i) {
        cell_list.grid_size[i] = static_cast<int32_t>(box_size[i] / cutoff_radius);
        if (cell_list.grid_size[i] < 2) cell_list.grid_size[i] = 2;
    }
    
    int32_t total_cells = cell_list.grid_size[0] * cell_list.grid_size[1] * cell_list.grid_size[2];
    
    // 分配内存（实际实现中会使用UB/L1内存）
    // 这里仅展示算法逻辑
    cell_list.cell_starts = new int32_t[total_cells];
    cell_list.cell_ends = new int32_t[total_cells];
    cell_list.particle_cells = new int32_t[num_particles];
    cell_list.cell_particles = new int32_t[num_particles];
    
    // 初始化细胞边界
    for (int i = 0; i < total_cells; ++i) {
        cell_list.cell_starts[i] = 0;
        cell_list.cell_ends[i] = 0;
    }
    
    // 第一遍：计数每个细胞的粒子数
    for (int i = 0; i < num_particles; ++i) {
        float x = positions[i * 3];
        float y = positions[i * 3 + 1];
        float z = positions[i * 3 + 2];
        
        // 应用周期性边界
        float bx = x - floor(x / box_size[0]) * box_size[0];
        float by = y - floor(y / box_size[1]) * box_size[1];
        float bz = z - floor(z / box_size[2]) * box_size[2];
        
        // 计算细胞索引
        int32_t cx = static_cast<int32_t>(bx / cutoff_radius);
        int32_t cy = static_cast<int32_t>(by / cutoff_radius);
        int32_t cz = static_cast<int32_t>(bz / cutoff_radius);
        
        cx = std::min(cx, cell_list.grid_size[0] - 1);
        cy = std::min(cy, cell_list.grid_size[1] - 1);
        cz = std::min(cz, cell_list.grid_size[2] - 1);
        
        int32_t cell_idx = cz * cell_list.grid_size[1] * cell_list.grid_size[0] +
                          cy * cell_list.grid_size[0] + cx;
        
        cell_list.particle_cells[i] = cell_idx;
        cell_list.cell_ends[cell_idx]++;
    }
    
    // 计算细胞起始位置
    int32_t offset = 0;
    for (int i = 0; i < total_cells; ++i) {
        cell_list.cell_starts[i] = offset;
        offset += cell_list.cell_ends[i];
        cell_list.cell_ends[i] = cell_list.cell_starts[i];
    }
    
    // 第二遍：填充细胞粒子列表
    for (int i = 0; i < num_particles; ++i) {
        int32_t cell_idx = cell_list.particle_cells[i];
        int32_t pos = cell_list.cell_ends[cell_idx];
        cell_list.cell_particles[pos] = i;
        cell_list.cell_ends[cell_idx]++;
    }
    
    return cell_list;
}

// 邻居列表构建函数（基于细胞列表）
NeighborList build_neighbor_list(
    const CellList& cell_list,
    const float* positions,
    int32_t particle_idx,
    const float box_size[3],
    float cutoff_radius) {
    
    NeighborList neighbors;
    neighbors.count = 0;
    neighbors.capacity = 100; // 初始容量
    
    // 分配内存（实际实现中使用UB内存）
    neighbors.indices = new int32_t[neighbors.capacity];
    neighbors.distances = new float[neighbors.capacity];
    
    float cutoff_sq = cutoff_radius * cutoff_radius;
    
    // 获取粒子所在细胞
    int32_t cell_idx = cell_list.particle_cells[particle_idx];
    
    // 计算细胞坐标
    int32_t grid_size[3] = {
        cell_list.grid_size[0],
        cell_list.grid_size[1],
        cell_list.grid_size[2]
    };
    
    int32_t cz = cell_idx / (grid_size[1] * grid_size[0]);
    int32_t cy = (cell_idx % (grid_size[1] * grid_size[0])) / grid_size[0];
    int32_t cx = cell_idx % grid_size[0];
    
    // 搜索相邻细胞（3×3×3区域）
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                
                // 计算相邻细胞索引（考虑周期性）
                int32_t nx = (cx + dx + grid_size[0]) % grid_size[0];
                int32_t ny = (cy + dy + grid_size[1]) % grid_size[1];
                int32_t nz = (cz + dz + grid_size[2]) % grid_size[2];
                
                int32_t neighbor_cell_idx = nz * grid_size[1] * grid_size[0] +
                                          ny * grid_size[0] + nx;
                
                // 遍历相邻细胞中的粒子
                int32_t start = cell_list.cell_starts[neighbor_cell_idx];
                int32_t end = cell_list.cell_ends[neighbor_cell_idx];
                
                for (int j = start; j < end; ++j) {
                    int32_t neighbor_idx = cell_list.cell_particles[j];
                    
                    // 跳过自身
                    if (neighbor_idx == particle_idx) continue;
                    
                    // 计算距离（考虑PBC）
                    float dx = positions[neighbor_idx * 3] - positions[particle_idx * 3];
                    float dy = positions[neighbor_idx * 3 + 1] - positions[particle_idx * 3 + 1];
                    float dz = positions[neighbor_idx * 3 + 2] - positions[particle_idx * 3 + 2];
                    
                    // 最小镜像约定
                    dx = dx - round(dx / box_size[0]) * box_size[0];
                    dy = dy - round(dy / box_size[1]) * box_size[1];
                    dz = dz - round(dz / box_size[2]) * box_size[2];
                    
                    float dist_sq = dx * dx + dy * dy + dz * dz;
                    
                    // 如果在截断半径内，添加到邻居列表
                    if (dist_sq < cutoff_sq) {
                        if (neighbors.count >= neighbors.capacity) {
                            // 扩展容量（实际实现中需要预分配足够空间）
                            neighbors.capacity *= 2;
                            int32_t* new_indices = new int32_t[neighbors.capacity];
                            float* new_distances = new float[neighbors.capacity];
                            
                            std::copy(neighbors.indices, neighbors.indices + neighbors.count, new_indices);
                            std::copy(neighbors.distances, neighbors.distances + neighbors.count, new_distances);
                            
                            delete[] neighbors.indices;
                            delete[] neighbors.distances;
                            
                            neighbors.indices = new_indices;
                            neighbors.distances = new_distances;
                        }
                        
                        neighbors.indices[neighbors.count] = neighbor_idx;
                        neighbors.distances[neighbors.count] = sqrt(dist_sq);
                        neighbors.count++;
                    }
                }
            }
        }
    }
    
    return neighbors;
}

// Tiling验证函数
bool validate_tiling(const DpdTiling& tiling) {
    if (tiling.total_particles <= 0) return false;
    if (tiling.block_size <= 0) return false;
    if (tiling.num_blocks <= 0) return false;
    if (tiling.ub_particle_capacity <= 0) return false;
    
    // 检查UB容量是否足够
    int32_t particles_per_block = tiling.block_size;
    if (particles_per_block > tiling.ub_particle_capacity) {
        return false; // UB容量不足
    }
    
    // 检查邻居列表容量
    int32_t max_neighbors_per_particle = tiling.max_neighbors;
    int32_t total_neighbor_slots = particles_per_block * max_neighbors_per_particle;
    int32_t neighbor_memory = total_neighbor_slots * sizeof(int32_t);
    
    if (neighbor_memory > UB_CAPACITY / 2) {
        return false; // 邻居列表内存不足
    }
    
    return true;
}

// 内存对齐计算
int32_t calculate_aligned_size(int32_t size, int32_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

// 获取粒子在UB中的偏移量
int32_t get_particle_ub_offset(int32_t particle_idx, const DpdTiling& tiling) {
    int32_t aligned_particle_size = calculate_aligned_size(PARTICLE_SIZE, 32);
    // 使用tiling参数避免未使用警告
    (void)tiling; // 标记为已使用
    return particle_idx * aligned_particle_size;
}

// 获取邻居列表在UB中的偏移量
int32_t get_neighbor_list_ub_offset(int32_t block_idx, const DpdTiling& tiling) {
    int32_t particle_offset = tiling.ub_particle_capacity * PARTICLE_SIZE;
    int32_t neighbor_slot_size = sizeof(int32_t) + sizeof(float);
    int32_t total_neighbor_slots = tiling.block_size * tiling.max_neighbors;
    
    return particle_offset + block_idx * total_neighbor_slots * neighbor_slot_size;
}