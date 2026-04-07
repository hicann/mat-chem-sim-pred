// test_simple.cpp
// DPD算子简单测试程序

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

// 简单的DPD模拟（CPU版本，用于验证算法）
class SimpleDpdSimulator {
public:
    struct Params {
        float dt = 0.01f;
        float rc = 1.0f;
        float a_ij = 25.0f;
        float gamma = 4.5f;
        float sigma = 3.0f;
        float kBT = 1.0f;
        float box_size[3] = {10.0f, 10.0f, 10.0f};
        int num_particles = 100;
        int num_steps = 100;
    };
    
    struct Result {
        bool success = true;
        float total_time = 0.0f;
        float steps_per_second = 0.0f;
    };
    
    SimpleDpdSimulator(const Params& params) : params_(params) {
        // 初始化粒子数据
        positions_.resize(params.num_particles * 3);
        velocities_.resize(params.num_particles * 3);
        forces_.resize(params.num_particles * 3, 0.0f);
        
        // 设置随机初始位置
        std::srand(12345);
        for (int i = 0; i < params.num_particles; ++i) {
            for (int j = 0; j < 3; ++j) {
                positions_[i*3 + j] = static_cast<float>(std::rand()) / RAND_MAX * params.box_size[j];
                velocities_[i*3 + j] = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 0.1f;
            }
        }
    }
    
    Result run_simulation() {
        auto start = std::chrono::steady_clock::now();
        
        for (int step = 0; step < params_.num_steps; ++step) {
            if (!run_step()) {
                Result result;
                result.success = false;
                return result;
            }
        }
        
        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<float>(end - start);
        
        Result result;
        result.total_time = duration.count();
        result.steps_per_second = params_.num_steps / result.total_time;
        
        return result;
    }
    
private:
    bool run_step() {
        // 1. 计算所有力
        calculate_forces();
        
        // 2. Velocity-Verlet第一步
        for (int i = 0; i < params_.num_particles; ++i) {
            for (int j = 0; j < 3; ++j) {
                velocities_[i*3 + j] += forces_[i*3 + j] * params_.dt * 0.5f;
                positions_[i*3 + j] += velocities_[i*3 + j] * params_.dt;
                
                // 应用周期性边界条件
                if (positions_[i*3 + j] < 0) positions_[i*3 + j] += params_.box_size[j];
                if (positions_[i*3 + j] >= params_.box_size[j]) positions_[i*3 + j] -= params_.box_size[j];
            }
        }
        
        // 3. 重新计算力
        calculate_forces();
        
        // 4. Velocity-Verlet第二步
        for (int i = 0; i < params_.num_particles; ++i) {
            for (int j = 0; j < 3; ++j) {
                velocities_[i*3 + j] += forces_[i*3 + j] * params_.dt * 0.5f;
            }
        }
        
        return true;
    }
    
    void calculate_forces() {
        // 重置力
        std::fill(forces_.begin(), forces_.end(), 0.0f);
        
        // 计算所有粒子对之间的力
        for (int i = 0; i < params_.num_particles; ++i) {
            for (int j = i + 1; j < params_.num_particles; ++j) {
                float dx = positions_[j*3] - positions_[i*3];
                float dy = positions_[j*3 + 1] - positions_[i*3 + 1];
                float dz = positions_[j*3 + 2] - positions_[i*3 + 2];
                
                // 应用最小镜像约定
                for (int k = 0; k < 3; ++k) {
                    float box_size = params_.box_size[k];
                    float diff = positions_[j*3 + k] - positions_[i*3 + k];
                    if (diff > box_size * 0.5f) diff -= box_size;
                    if (diff <= -box_size * 0.5f) diff += box_size;
                    if (k == 0) dx = diff;
                    else if (k == 1) dy = diff;
                    else dz = diff;
                }
                
                float r2 = dx*dx + dy*dy + dz*dz;
                
                if (r2 < params_.rc * params_.rc && r2 > 0.0f) {
                    float r = std::sqrt(r2);
                    float inv_r = 1.0f / r;
                    
                    // 单位方向向量
                    float ex = dx * inv_r;
                    float ey = dy * inv_r;
                    float ez = dz * inv_r;
                    
                    // 相对速度
                    float dvx = velocities_[j*3] - velocities_[i*3];
                    float dvy = velocities_[j*3 + 1] - velocities_[i*3 + 1];
                    float dvz = velocities_[j*3 + 2] - velocities_[i*3 + 2];
                    
                    // 速度投影
                    float v_proj = dvx*ex + dvy*ey + dvz*ez;
                    
                    // 权重函数
                    float w_r = 1.0f - r / params_.rc;
                    
                    // 保守力
                    float fc = params_.a_ij * w_r;
                    
                    // 耗散力
                    float fd = -params_.gamma * w_r * w_r * v_proj;
                    
                    // 随机力
                    float rand_val = (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
                    float fr = params_.sigma * w_r * rand_val / std::sqrt(params_.dt);
                    
                    // 总力
                    float f_total = fc + fd + fr;
                    
                    // 应用力（牛顿第三定律）
                    forces_[i*3] += f_total * ex;
                    forces_[i*3 + 1] += f_total * ey;
                    forces_[i*3 + 2] += f_total * ez;
                    
                    forces_[j*3] -= f_total * ex;
                    forces_[j*3 + 1] -= f_total * ey;
                    forces_[j*3 + 2] -= f_total * ez;
                }
            }
        }
    }
    
    Params params_;
    std::vector<float> positions_;
    std::vector<float> velocities_;
    std::vector<float> forces_;
};

int main() {
    std::cout << "DPD算子简单测试程序" << std::endl;
    std::cout << "==================" << std::endl;
    
    // 测试不同规模的系统
    std::vector<int> particle_counts = {100, 500, 1000};
    
    for (int num_particles : particle_counts) {
        std::cout << "\n测试粒子数: " << num_particles << std::endl;
        
        SimpleDpdSimulator::Params params;
        params.num_particles = num_particles;
        params.num_steps = 100;
        
        SimpleDpdSimulator simulator(params);
        
        auto start = std::chrono::steady_clock::now();
        auto result = simulator.run_simulation();
        auto end = std::chrono::steady_clock::now();
        
        auto duration = std::chrono::duration<float>(end - start);
        
        if (result.success) {
            std::cout << "  状态: ✅ 成功" << std::endl;
            std::cout << "  模拟时间: " << duration.count() << " 秒" << std::endl;
            std::cout << "  性能: " << result.steps_per_second << " 步/秒" << std::endl;
            std::cout << "  每步时间: " << (duration.count() / params.num_steps * 1000) << " 毫秒" << std::endl;
        } else {
            std::cout << "  状态: ❌ 失败" << std::endl;
        }
    }
    
    std::cout << "\n测试完成!" << std::endl;
    
    return 0;
}