// dpd_host.cpp
// DPD算子主机端实现
// 实现参数检查、核函数调度、内存管理

#include "dpd_host.h"
#include "dpd_params.h"
#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <thread>
#include <atomic>

// 内部实现类
class DpdSimulatorImpl {
private:
    // 状态标志
    bool initialized_;
    bool device_initialized_;
    
    // 模拟参数
    DpdParams params_;
    ParticleData current_state_;
    HostNeighborList neighbor_list_;
    HostCellList cell_list_;
    
    // 设备内存句柄
    aclrtStream stream_;
    void* d_positions_;
    void* d_velocities_;
    void* d_forces_;
    void* d_params_;
    void* d_tiling_;
    
    // 性能统计
    std::chrono::steady_clock::time_point start_time_;
    int32_t completed_steps_;
    float total_simulation_time_;
    
    // 随机数生成器
    std::mt19937 rng_;
    std::normal_distribution<float> normal_dist_;
    
    // 错误信息
    std::string last_error_;
    
    // 输出控制
    bool enable_energy_output_;
    bool enable_trajectory_output_;
    std::vector<ParticleData> trajectory_;
    std::vector<float> energy_history_;
    
public:
    DpdSimulatorImpl() 
        : initialized_(false)
        , device_initialized_(false)
        , completed_steps_(0)
        , total_simulation_time_(0.0f)
        , enable_energy_output_(true)
        , enable_trajectory_output_(false)
        , rng_(std::random_device{}())
        , normal_dist_(0.0f, 1.0f) {
        
        // 初始化ACL
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            last_error_ = "ACL初始化失败: " + std::to_string(ret);
            return;
        }
        
        // 设置设备
        ret = aclrtSetDevice(0);
        if (ret != ACL_SUCCESS) {
            last_error_ = "设置设备失败: " + std::to_string(ret);
            return;
        }
        
        // 创建流
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            last_error_ = "创建流失败: " + std::to_string(ret);
            return;
        }
        
        device_initialized_ = true;
    }
    
    ~DpdSimulatorImpl() {
        // 释放设备内存
        free_device_memory();
        
        // 销毁流
        if (stream_) {
            aclrtDestroyStream(stream_);
        }
        
        // 重置设备
        if (device_initialized_) {
            aclrtResetDevice(0);
        }
        
        // 去初始化ACL
        aclFinalize();
    }
    
    // 初始化函数
    bool initialize(const DpdParams& params) {
        if (!device_initialized_) {
            last_error_ = "设备未初始化";
            return false;
        }
        
        // 验证参数
        if (!validate_parameters(params)) {
            last_error_ = "参数验证失败";
            return false;
        }
        
        params_ = params;
        rng_.seed(params_.random_seed);
        
        // 初始化粒子数据
        if (current_state_.size() == 0) {
            current_state_.resize(params_.num_particles);
            set_random_positions();
            set_random_velocities(params_.kBT);
        } else if (current_state_.size() != params_.num_particles) {
            last_error_ = "粒子数据大小不匹配";
            return false;
        }
        
        // 分配设备内存
        if (!allocate_device_memory()) {
            last_error_ = "设备内存分配失败";
            return false;
        }
        
        // 构建邻居列表
        rebuild_neighbor_list();
        
        // 初始化性能统计
        start_time_ = std::chrono::steady_clock::now();
        completed_steps_ = 0;
        total_simulation_time_ = 0.0f;
        
        // 清空历史数据
        trajectory_.clear();
        energy_history_.clear();
        
        if (enable_trajectory_output_) {
            trajectory_.push_back(current_state_);
        }
        
        initialized_ = true;
        return true;
    }
    
    // 运行模拟
    DpdResult run_simulation() {
        DpdResult result;
        
        if (!initialized_) {
            result.success = false;
            result.error_msg = "模拟器未初始化";
            return result;
        }
        
        auto start = std::chrono::steady_clock::now();
        
        try {
            for (int32_t step = 0; step < params_.num_steps; ++step) {
                if (!run_single_step()) {
                    result.success = false;
                    result.error_msg = "第" + std::to_string(step) + "步执行失败: " + last_error_;
                    return result;
                }
                
                completed_steps_++;
                
                // 记录能量
                if (enable_energy_output_ && (step % params_.output_freq == 0)) {
                    float ke = compute_kinetic_energy();
                    energy_history_.push_back(ke);
                    result.time_points.push_back(step * params_.dt);
                    result.kinetic_energy.push_back(ke);
                }
                
                // 记录轨迹
                if (enable_trajectory_output_ && (step % params_.output_freq == 0)) {
                    trajectory_.push_back(current_state_);
                }
                
                // 重建邻居列表（如果需要）
                if (step % 10 == 0) {  // 每10步重建一次
                    rebuild_neighbor_list();
                }
            }
            
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<float> duration = end - start;
            
            result.success = true;
            result.final_state = current_state_;
            result.total_time = duration.count();
            result.steps_per_second = params_.num_steps / duration.count();
            result.num_steps = params_.num_steps;
            
            // 计算最终温度
            result.temperature.push_back(compute_temperature());
            
        } catch (const std::exception& e) {
            result.success = false;
            result.error_msg = e.what();
        }
        
        return result;
    }
    
    // 运行单步
    DpdResult run_step(int32_t num_steps) {
        DpdResult result;
        
        if (!initialized_) {
            result.success = false;
            result.error_msg = "模拟器未初始化";
            return result;
        }
        
        if (num_steps <= 0) {
            result.success = false;
            result.error_msg = "步数必须大于0";
            return result;
        }
        
        auto start = std::chrono::steady_clock::now();
        
        try {
            for (int32_t step = 0; step < num_steps; ++step) {
                if (!run_single_step()) {
                    result.success = false;
                    result.error_msg = "第" + std::to_string(step) + "步执行失败: " + last_error_;
                    return result;
                }
                
                completed_steps_++;
            }
            
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<float> duration = end - start;
            
            result.success = true;
            result.final_state = current_state_;
            result.total_time = duration.count();
            result.steps_per_second = num_steps / duration.count();
            result.num_steps = num_steps;
            
        } catch (const std::exception& e) {
            result.success = false;
            result.error_msg = e.what();
        }
        
        return result;
    }
    
    // 获取当前状态
    ParticleData get_current_state() const {
        return current_state_;
    }
    
    // 获取参数
    DpdParams get_parameters() const {
        return params_;
    }
    
    // 检查是否初始化
    bool is_initialized() const {
        return initialized_;
    }
    
    // 获取模拟时间
    float get_simulation_time() const {
        return completed_steps_ * params_.dt;
    }
    
    // 获取已完成步数
    int32_t get_completed_steps() const {
        return completed_steps_;
    }
    
    // 获取性能指标
    float get_steps_per_second() const {
        if (total_simulation_time_ > 0) {
            return completed_steps_ / total_simulation_time_;
        }
        return 0.0f;
    }
    
    // 计算动能
    float compute_kinetic_energy() const {
        float ke = 0.0f;
        int32_t n = current_state_.size();
        
        for (int32_t i = 0; i < n; ++i) {
            float vx = current_state_.velocities[i * 3];
            float vy = current_state_.velocities[i * 3 + 1];
            float vz = current_state_.velocities[i * 3 + 2];
            
            ke += 0.5f * params_.particle_mass * (vx * vx + vy * vy + vz * vz);
        }
        
        return ke;
    }
    
    // 计算温度
    float compute_temperature() const {
        float ke = compute_kinetic_energy();
        int32_t n = current_state_.size();
        float dof = 3.0f * n;  // 自由度
        return 2.0f * ke / dof;
    }
    
    // 重建邻居列表
    void rebuild_neighbor_list() {
        // 这里实现邻居列表重建逻辑
        // 由于代码长度限制，这里展示简化版本
        neighbor_list_.resize(current_state_.size());
        
        // 实际实现中会使用细胞列表加速
        // 这里使用简单的O(N^2)算法作为示例
        float rc_sq = params_.rc * params_.rc;
        int32_t n = current_state_.size();
        
        for (int32_t i = 0; i < n; ++i) {
            neighbor_list_.indices[i].clear();
            neighbor_list_.distances[i].clear();
            
            for (int32_t j = i + 1; j < n; ++j) {
                float dx = current_state_.positions[j * 3] - current_state_.positions[i * 3];
                float dy = current_state_.positions[j * 3 + 1] - current_state_.positions[i * 3 + 1];
                float dz = current_state_.positions[j * 3 + 2] - current_state_.positions[i * 3 + 2];
                
                // 应用周期性边界
                dx = dx - round(dx / params_.box_size[0]) * params_.box_size[0];
                dy = dy - round(dy / params_.box_size[1]) * params_.box_size[1];
                dz = dz - round(dz / params_.box_size[2]) * params_.box_size[2];
                
                float dist_sq = dx * dx + dy * dy + dz * dz;
                
                if (dist_sq < rc_sq) {
                    neighbor_list_.indices[i].push_back(j);
                    neighbor_list_.distances[i].push_back(sqrt(dist_sq));
                    
                    neighbor_list_.indices[j].push_back(i);
                    neighbor_list_.distances[j].push_back(sqrt(dist_sq));
                }
            }
        }
    }
    
    // 获取邻居列表
    const HostNeighborList& get_neighbor_list() const {
        return neighbor_list_;
    }
    
    // 设置随机位置
    void set_random_positions() {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        int32_t n = current_state_.size();
        
        for (int32_t i = 0; i < n; ++i) {
            current_state_.positions[i * 3] = dist(rng_) * params_.box_size[0];
            current_state_.positions[i * 3 + 1] = dist(rng_) * params_.box_size[1];
            current_state_.positions[i * 3 + 2] = dist(rng_) * params_.box_size[2];
        }
    }
    
    // 设置随机速度
    void set_random_velocities(float temperature) {
        int32_t n = current_state_.size();
        float scale = sqrt(temperature / params_.particle_mass);
        
        // 生成随机速度
        for (int32_t i = 0; i < n; ++i) {
            current_state_.velocities[i * 3] = normal_dist_(rng_) * scale;
            current_state_.velocities[i * 3 + 1] = normal_dist_(rng_) * scale;
            current_state_.velocities[i * 3 + 2] = normal_dist_(rng_) * scale;
        }
        
        // 去除整体动量
        float total_vx = 0.0f, total_vy = 0.0f, total_vz = 0.0f;
        for (int32_t i = 0; i < n; ++i) {
            total_vx += current_state_.velocities[i * 3];
            total_vy += current_state_.velocities[i * 3 + 1];
            total_vz += current_state_.velocities[i * 3 + 2];
        }
        
        float avg_vx = total_vx / n;
        float avg_vy = total_vy / n;
        float avg_vz = total_vz / n;
        
        for (int32_t i = 0; i < n; ++i) {
            current_state_.velocities[i * 3] -= avg_vx;
            current_state_.velocities[i * 3 + 1] -= avg_vy;
            current_state_.velocities[i * 3 + 2] -= avg_vz;
        }
    }
    
    // 获取错误信息
    std::string get_last_error() const {
        return last_error_;
    }
    
    // 清空错误
    void clear_error() {
        last_error_.clear();
    }
    
private:
    // 验证参数
    bool validate_parameters(const DpdParams& params) {
        if (params.dt <= 0.0f) {
            last_error_ = "时间步长必须大于0";
            return false;
        }
        
        if (params.rc <= 0.0f) {
            last_error_ = "截断半径必须大于0";
            return false;
        }
        
        if (params.num_particles <= 0) {
            last_error_ = "粒子数必须大于0";
            return false;
        }
        
        for (int i = 0; i < 3; ++i) {
            if (params.box_size[i] <= 0.0f) {
                last_error_ = "盒子尺寸必须大于0";
                return false;
            }
        }
        
        // 检查涨落-耗散关系
        float expected_sigma = sqrt(2.0f * params.gamma * params.kBT);
        if (fabs(params.sigma - expected_sigma) > 1e-3) {
            last_error_ = "涨落-耗散关系不满足: σ² = 2γkBT";
            return false;
        }
        
        return true;
    }
    
    // 分配设备内存
    bool allocate_device_memory() {
        int32_t n = params_.num_particles;
        size_t pos_size = n * 3 * sizeof(float);
        size_t vel_size = n * 3 * sizeof(float);
        size_t force_size = n * 3 * sizeof(float);
        size_t params_size = sizeof(DpdParams);
        size_t tiling_size = sizeof(DpdTiling);
        
        aclError ret;
        
        // 分配位置内存
        ret = aclrtMalloc(&d_positions_, pos_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            last_error_ = "分配位置内存失败: " + std::to_string(ret);
            return false;
        }
        
        // 分配速度内存
        ret = aclrtMalloc(&d_velocities_, vel_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            last_error_ = "分配速度内存失败: " + std::to_string(ret);
            return false;
        }
        
        // 分配力内存
        ret = aclrtMalloc(&d_forces_, force_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            last_error_ = "分配力内存失败: " + std::to_string(ret);
            return false;
        }
        
        // 分配参数内存
        ret = aclrtMalloc(&d_params_, params_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            last_error_ = "分配参数内存失败: " + std::to_string(ret);
            return false;
        }
        
        // 分配Tiling内存
        ret = aclrtMalloc(&d_tiling_, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            last_error_ = "分配Tiling内存失败: " + std::to_string(ret);
            return false;
        }
        
        // 拷贝数据到设备
        ret = aclrtMemcpy(d_positions_, pos_size, 
                         current_state_.positions.data(), pos_size,
                         ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            last_error_ = "拷贝位置数据失败: " + std::to_string(ret);
            return false;
        }
        
        ret = aclrtMem