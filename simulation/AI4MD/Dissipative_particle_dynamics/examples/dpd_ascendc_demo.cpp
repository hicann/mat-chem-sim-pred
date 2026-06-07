// dpd_ascendc_demo.cpp
// DPD算子C++示例
// 演示如何直接调用host接口运行DPD模拟

#include "dpd_host.h"
#include "dpd_params.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

// 简单DPD模拟示例
void run_simple_dpd_simulation() {
    std::cout << "=" << 60 << std::endl;
    std::cout << "DPD Ascend C 示例 - 简单模拟" << std::endl;
    std::cout << "=" << 60 << std::endl;
    
    // 1. 创建模拟器
    std::cout << "\n1. 创建DPD模拟器..." << std::endl;
    DpdSimulator simulator;
    
    // 2. 设置模拟参数
    std::cout << "2. 设置模拟参数..." << std::endl;
    DpdParams params;
    params.num_particles = 1000;
    params.box_size[0] = 10.0f;
    params.box_size[1] = 10.0f;
    params.box_size[2] = 10.0f;
    params.dt = 0.01f;
    params.num_steps = 100;
    params.output_freq = 10;
    
    // 3. 初始化模拟器
    std::cout << "3. 初始化模拟器..." << std::endl;
    if (!simulator.initialize(params)) {
        std::cerr << "初始化失败: " << simulator.get_last_error() << std::endl;
        return;
    }
    
    // 4. 设置随机初始条件
    std::cout << "4. 设置初始条件..." << std::endl;
    simulator.set_random_positions();
    simulator.set_random_velocities(params.kBT);
    
    // 5. 运行模拟
    std::cout << "5. 运行模拟..." << std::endl;
    auto start_time = std::chrono::steady_clock::now();
    
    DpdResult result = simulator.run_simulation();
    
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // 6. 输出结果
    std::cout << "\n模拟完成!" << std::endl;
    std::cout << "-" << 40 << std::endl;
    
    if (result.success) {
        std::cout << "状态: ✅ 成功" << std::endl;
        std::cout << "模拟时间: " << result.total_time << " 秒" << std::endl;
        std::cout << "性能: " << result.steps_per_second << " 步/秒" << std::endl;
        std::cout << "总步数: " << result.num_steps << std::endl;
        
        // 计算最终温度
        float final_temp = simulator.compute_temperature();
        std::cout << "最终温度: " << final_temp << " kBT" << std::endl;
        
        // 计算动能
        float ke = simulator.compute_kinetic_energy();
        std::cout << "最终动能: " << ke << std::endl;
        
    } else {
        std::cout << "状态: ❌ 失败" << std::endl;
        std::cout << "错误信息: " << result.error_msg << std::endl;
    }
    
    std::cout << "总耗时: " << elapsed.count() << " 秒" << std::endl;
}

// 性能基准测试
void run_performance_benchmark() {
    std::cout << "\n" << "=" << 60 << std::endl;
    std::cout << "DPD Ascend C 示例 - 性能基准测试" << std::endl;
    std::cout << "=" << 60 << std::endl;
    
    // 测试不同系统规模
    std::vector<int> particle_counts = {100, 1000, 5000, 10000};
    std::vector<float> times;
    std::vector<float> performance;
    
    for (int num_particles : particle_counts) {
        std::cout << "\n测试粒子数: " << num_particles << std::endl;
        
        DpdSimulator simulator;
        DpdParams params;
        params.num_particles = num_particles;
        params.box_size[0] = params.box_size[1] = params.box_size[2] = 20.0f;
        params.dt = 0.01f;
        params.num_steps = 100;
        
        if (!simulator.initialize(params)) {
            std::cerr << "初始化失败" << std::endl;
            continue;
        }
        
        simulator.set_random_positions();
        simulator.set_random_velocities(params.kBT);
        
        auto start = std::chrono::steady_clock::now();
        DpdResult result = simulator.run_simulation();
        auto end = std::chrono::steady_clock::now();
        
        if (result.success) {
            float elapsed = std::chrono::duration<float>(end - start).count();
            float steps_per_second = params.num_steps / elapsed;
            
            times.push_back(elapsed);
            performance.push_back(steps_per_second);
            
            std::cout << "  时间: " << elapsed << " 秒" << std::endl;
            std::cout << "  性能: " << steps_per_second << " 步/秒" << std::endl;
            std::cout << "  每步: " << (elapsed / params.num_steps * 1000) << " 毫秒" << std::endl;
        } else {
            std::cout << "  测试失败" << std::endl;
        }
    }
    
    // 输出性能报告
    if (!performance.empty()) {
        std::cout << "\n性能总结:" << std::endl;
        std::cout << "-" << 50 << std::endl;
        std::cout << std::setw(12) << "粒子数" 
                  << std::setw(12) << "时间(秒)" 
                  << std::setw(12) << "步/秒" 
                  << std::setw(12) << "毫秒/步" << std::endl;
        std::cout << "-" << 50 << std::endl;
        
        for (size_t i = 0; i < particle_counts.size() && i < performance.size(); ++i) {
            std::cout << std::setw(12) << particle_counts[i]
                      << std::setw(12) << std::fixed << std::setprecision(3) << times[i]
                      << std::setw(12) << std::fixed << std::setprecision(1) << performance[i]
                      << std::setw(12) << std::fixed << std::setprecision(2) << (times[i] / 100 * 1000)
                      << std::endl;
        }
    }
}

// 能量守恒测试
void run_energy_conservation_test() {
    std::cout << "\n" << "=" << 60 << std::endl;
    std::cout << "DPD Ascend C 示例 - 能量守恒测试" << std::endl;
    std::cout << "=" << 60 << std::endl;
    
    DpdSimulator simulator;
    DpdParams params;
    params.num_particles = 500;
    params.box_size[0] = params.box_size[1] = params.box_size[2] = 15.0f;
    params.dt = 0.001f;  // 小时间步长保证能量守恒
    params.num_steps = 1000;
    params.output_freq = 10;
    
    if (!simulator.initialize(params)) {
        std::cerr << "初始化失败" << std::endl;
        return;
    }
    
    simulator.set_random_positions();
    simulator.set_random_velocities(params.kBT);
    
    // 记录能量历史
    std::vector<float> kinetic_energy;
    std::vector<float> time_points;
    
    std::cout << "\n运行能量守恒测试..." << std::endl;
    std::cout << "时间步长: " << params.dt << std::endl;
    std::cout << "总步数: " << params.num_steps << std::endl;
    
    for (int step = 0; step < params.num_steps; ++step) {
        DpdResult result = simulator.run_step(1);
        
        if (!result.success) {
            std::cerr << "第 " << step << " 步失败" << std::endl;
            break;
        }
        
        if (step % params.output_freq == 0) {
            float ke = simulator.compute_kinetic_energy();
            kinetic_energy.push_back(ke);
            time_points.push_back(step * params.dt);
            
            if (step % 100 == 0) {
                std::cout << "步数: " << step 
                          << ", 动能: " << ke 
                          << ", 温度: " << simulator.compute_temperature() << std::endl;
            }
        }
    }
    
    // 分析能量守恒
    if (kinetic_energy.size() > 1) {
        float initial_ke = kinetic_energy[0];
        float final_ke = kinetic_energy.back();
        float energy_drift = fabs(final_ke - initial_ke) / initial_ke * 100.0f;
        
        std::cout << "\n能量守恒分析:" << std::endl;
        std::cout << "初始动能: " << initial_ke << std::endl;
        std::cout << "最终动能: " << final_ke << std::endl;
        std::cout << "能量漂移: " << energy_drift << "%" << std::endl;
        
        if (energy_drift < 1.0f) {
            std::cout << "✅ 能量守恒良好" << std::endl;
        } else {
            std::cout << "⚠ 能量漂移较大，建议减小时间步长" << std::endl;
        }
        
        // 保存能量数据
        std::ofstream energy_file("energy_conservation.csv");
        energy_file << "time,kinetic_energy\n";
        for (size_t i = 0; i < time_points.size(); ++i) {
            energy_file << time_points[i] << "," << kinetic_energy[i] << "\n";
        }
        energy_file.close();
        
        std::cout << "能量数据已保存到: energy_conservation.csv" << std::endl;
    }
}

// 主函数
int main(int argc, char* argv[]) {
    std::cout << "DPD Ascend C 算子示例程序" << std::endl;
    std::cout << "编译时间: " << __DATE__ << " " << __TIME__ << std::endl;
    
    try {
        // 运行简单示例
        run_simple_dpd_simulation();
        
        // 运行性能测试
        run_performance_benchmark();
        
        // 运行能量守恒测试
        run_energy_conservation_test();
        
        std::cout << "\n" << "=" << 60 << std::endl;
        std::cout << "所有示例完成!" << std::endl;
        std::cout << "=" << 60 << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}