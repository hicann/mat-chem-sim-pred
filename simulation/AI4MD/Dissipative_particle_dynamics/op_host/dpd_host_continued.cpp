// dpd_host_continued_fixed.cpp
// DPD主机端实现 - 修复版本

#include "dpd_host.h"
#include "dpd_params.h"
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>

namespace dpd_utils {

// 文件IO函数
bool save_particle_data(const ParticleData& data, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    // 写入粒子数
    file << data.size() << std::endl;
    
    // 写入位置数据
    for (size_t i = 0; i < data.positions.size(); i += 3) {
        file << data.positions[i] << " " 
             << data.positions[i+1] << " " 
             << data.positions[i+2] << std::endl;
    }
    
    // 写入速度数据
    for (size_t i = 0; i < data.velocities.size(); i += 3) {
        file << data.velocities[i] << " " 
             << data.velocities[i+1] << " " 
             << data.velocities[i+2] << std::endl;
    }
    
    file.close();
    return true;
}

bool load_particle_data(ParticleData& data, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }
    
    int32_t num_particles;
    file >> num_particles;
    
    data.resize(num_particles);
    
    // 读取位置数据
    for (int i = 0; i < num_particles * 3; ++i) {
        file >> data.positions[i];
    }
    
    // 读取速度数据
    for (int i = 0; i < num_particles * 3; ++i) {
        file >> data.velocities[i];
    }
    
    file.close();
    return true;
}

// 数据生成函数
ParticleData create_cubic_lattice(int32_t particles_per_side, float spacing) {
    // CWE-190 fix: particles_per_side^3 overflows int32_t when
    // particles_per_side > ~1290 (1291^3 ≈ 2.16e9 > INT32_MAX). Compute the
    // total in 64-bit and validate before using it to size the particle array.
    if (particles_per_side <= 0) {
        return ParticleData(0);
    }
    int64_t total = static_cast<int64_t>(particles_per_side) *
                    particles_per_side * particles_per_side;
    if (total > INT32_MAX) {
        return ParticleData(0);  // lattice too large for int32 indexing
    }
    int32_t num_particles = static_cast<int32_t>(total);
    ParticleData data(num_particles);
    
    int idx = 0;
    for (int x = 0; x < particles_per_side; ++x) {
        for (int y = 0; y < particles_per_side; ++y) {
            for (int z = 0; z < particles_per_side; ++z) {
                data.positions[idx*3] = x * spacing;
                data.positions[idx*3 + 1] = y * spacing;
                data.positions[idx*3 + 2] = z * spacing;
                idx++;
            }
        }
    }
    
    return data;
}

ParticleData create_random_system(int32_t num_particles, const float box_size[3]) {
    ParticleData data(num_particles);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    for (int i = 0; i < num_particles; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::uniform_real_distribution<float> dist(0.0f, box_size[j]);
            data.positions[i*3 + j] = dist(gen);
        }
    }
    
    return data;
}

// 参数验证函数
bool validate_parameters(const DpdParams& params) {
    if (params.num_particles <= 0) {
        return false;
    }
    
    if (params.dt <= 0.0f || params.dt > 0.1f) {
        return false;
    }
    
    if (params.rc <= 0.0f) {
        return false;
    }
    
    for (int i = 0; i < 3; ++i) {
        if (params.box_size[i] <= 0.0f) {
            return false;
        }
    }
    
    return true;
}

bool validate_particle_data(const ParticleData& data, const DpdParams& params) {
    if (data.size() != params.num_particles) {
        return false;
    }
    
    // 检查位置是否在盒子内
    for (int i = 0; i < params.num_particles; ++i) {
        for (int j = 0; j < 3; ++j) {
            float pos = data.positions[i*3 + j];
            if (pos < 0.0f || pos > params.box_size[j]) {
                return false;
            }
        }
    }
    
    return true;
}

} // namespace dpd_utils

// DpdConfig类实现
DpdParams DpdConfig::default_config() {
    DpdParams params;
    params.dt = 0.01f;
    params.rc = 1.0f;
    params.a_ij = 25.0f;
    params.gamma = 4.5f;
    params.sigma = 3.0f;
    params.kBT = 1.0f;
    params.box_size[0] = params.box_size[1] = params.box_size[2] = 10.0f;
    params.num_particles = 1000;
    params.particle_mass = 1.0f;
    params.random_seed = 12345;
    params.num_steps = 100;
    params.output_freq = 10;
    params.enable_energy = true;
    params.enable_pressure = false;
    params.num_cores = 32;
    params.vector_len = 16;
    
    return params;
}

DpdParams DpdConfig::test_config(int32_t num_particles) {
    DpdParams params = default_config();
    params.num_particles = num_particles;
    return params;
}

DpdParams DpdConfig::performance_config(int32_t num_particles) {
    DpdParams params = default_config();
    params.num_particles = num_particles;
    params.num_steps = 1000;
    params.output_freq = 100;
    return params;
}

// 简单模拟函数（用于测试）
DpdResult run_simple_simulation(const DpdParams& params) {
    DpdResult result;
    
    // 创建模拟器
    DpdSimulator simulator;
    
    if (!simulator.initialize(params)) {
        result.success = false;
        result.error_msg = "初始化失败";
        return result;
    }
    
    // 设置初始条件
    simulator.set_random_positions();
    simulator.set_random_velocities(params.kBT);
    
    // 运行模拟
    result = simulator.run_simulation();
    
    return result;
}

// 性能测试函数
PerformanceStats run_performance_test(const DpdParams& params, int32_t num_runs) {
    PerformanceStats stats;
    stats.total_runs = num_runs;
    
    DpdSimulator simulator;
    if (!simulator.initialize(params)) {
        stats.success = false;
        return stats;
    }
    
    simulator.set_random_positions();
    simulator.set_random_velocities(params.kBT);
    
    auto start_time = std::chrono::steady_clock::now();
    
    for (int i = 0; i < num_runs; ++i) {
        DpdResult result = simulator.run_step(1);
        if (!result.success) {
            stats.success = false;
            break;
        }
        stats.successful_runs++;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats.total_time_ms = duration.count();
    if (stats.successful_runs > 0) {
        stats.avg_time_per_run_ms = static_cast<float>(stats.total_time_ms) / stats.successful_runs;
        stats.runs_per_second = 1000.0f / stats.avg_time_per_run_ms;
    }
    
    return stats;
}