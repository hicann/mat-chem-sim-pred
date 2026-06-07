// dpd_host.h
// DPD算子主机端接口声明
// 提供C++和Python调用接口

#ifndef DPD_HOST_H
#define DPD_HOST_H

#include "dpd_params.h"
#include <memory>
#include <string>

// 前向声明
class DpdSimulatorImpl;

// DPD模拟器类（主接口）
class DpdSimulator {
public:
    // 构造函数和析构函数
    DpdSimulator();
    ~DpdSimulator();
    
    // 禁止拷贝
    DpdSimulator(const DpdSimulator&) = delete;
    DpdSimulator& operator=(const DpdSimulator&) = delete;
    
    // 移动构造和赋值
    DpdSimulator(DpdSimulator&&) noexcept;
    DpdSimulator& operator=(DpdSimulator&&) noexcept;
    
    // 初始化函数
    bool initialize(const DpdParams& params);
    bool initialize_from_file(const std::string& config_file);
    
    // 数据设置函数
    void set_particle_data(const ParticleData& data);
    void set_random_positions();  // 随机初始化位置
    void set_random_velocities(float temperature = 1.0f); // 按温度初始化速度
    
    // 运行模拟
    DpdResult run_simulation();
    DpdResult run_step(int32_t num_steps = 1);
    
    // 状态获取函数
    ParticleData get_current_state() const;
    DpdParams get_parameters() const;
    bool is_initialized() const;
    
    // 性能统计
    float get_simulation_time() const;      // 已模拟时间
    int32_t get_completed_steps() const;    // 已完成步数
    float get_steps_per_second() const;     // 性能指标
    
    // 能量计算
    float compute_kinetic_energy() const;
    float compute_potential_energy() const;
    float compute_temperature() const;
    float compute_pressure() const;
    
    // 邻居列表管理
    void rebuild_neighbor_list();
    const HostNeighborList& get_neighbor_list() const;
    
    // 输出控制
    void enable_energy_output(bool enable);
    void enable_trajectory_output(bool enable);
    void set_output_frequency(int32_t freq);
    
    // 错误处理
    std::string get_last_error() const;
    void clear_error();
    
private:
    std::unique_ptr<DpdSimulatorImpl> impl_;
};

// C风格接口（用于Python绑定）
extern "C" {
    // 创建和销毁模拟器
    void* dpd_create_simulator();
    void dpd_destroy_simulator(void* simulator);
    
    // 初始化
    int dpd_initialize(void* simulator, const DpdParams* params);
    int dpd_initialize_from_file(void* simulator, const char* config_file);
    
    // 数据设置
    void dpd_set_particle_data(void* simulator, const ParticleData* data);
    void dpd_set_random_positions(void* simulator);
    void dpd_set_random_velocities(void* simulator, float temperature);
    
    // 运行模拟
    DpdResult* dpd_run_simulation(void* simulator);
    DpdResult* dpd_run_step(void* simulator, int32_t num_steps);
    
    // 状态获取
    ParticleData* dpd_get_current_state(void* simulator);
    DpdParams* dpd_get_parameters(void* simulator);
    int dpd_is_initialized(void* simulator);
    
    // 能量计算
    float dpd_compute_kinetic_energy(void* simulator);
    float dpd_compute_potential_energy(void* simulator);
    float dpd_compute_temperature(void* simulator);
    float dpd_compute_pressure(void* simulator);
    
    // 性能统计
    float dpd_get_simulation_time(void* simulator);
    int32_t dpd_get_completed_steps(void* simulator);
    float dpd_get_steps_per_second(void* simulator);
    
    // 错误处理
    const char* dpd_get_last_error(void* simulator);
    void dpd_clear_error(void* simulator);
    
    // 内存管理
    void dpd_free_particle_data(ParticleData* data);
    void dpd_free_result(DpdResult* result);
    void dpd_free_params(DpdParams* params);
}

// 工具函数
namespace dpd_utils {
    // 文件IO
    bool save_particle_data(const ParticleData& data, const std::string& filename);
    bool load_particle_data(ParticleData& data, const std::string& filename);
    
    bool save_simulation_result(const DpdResult& result, const std::string& filename);
    bool load_simulation_result(DpdResult& result, const std::string& filename);
    
    // 数据生成
    ParticleData create_cubic_lattice(int32_t particles_per_side, float spacing);
    ParticleData create_random_system(int32_t num_particles, const float box_size[3]);
    
    // 数据分析
    float compute_radial_distribution(const ParticleData& data, 
                                     const float box_size[3],
                                     float dr = 0.1f, 
                                     float r_max = 5.0f);
    
    std::vector<float> compute_velocity_autocorrelation(const std::vector<ParticleData>& trajectory,
                                                       int32_t max_lag = 100);
    
    // 可视化数据生成
    std::string generate_xyz_format(const ParticleData& data);
    std::string generate_vtk_format(const ParticleData& data, const float box_size[3]);
    
    // 参数验证
    bool validate_parameters(const DpdParams& params);
    bool validate_particle_data(const ParticleData& data, const DpdParams& params);
    
    // 单位转换
    float lj_to_real(float lj_value, float epsilon = 1.0f, float sigma = 1.0f);
    float real_to_lj(float real_value, float epsilon = 1.0f, float sigma = 1.0f);
}

// 异常类
class DpdException : public std::exception {
private:
    std::string message_;
    
public:
    explicit DpdException(const std::string& message) : message_(message) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }
};

// 配置类
class DpdConfig {
public:
    // 从JSON文件加载配置
    static DpdParams load_from_json(const std::string& filename);
    
    // 保存配置到JSON文件
    static bool save_to_json(const DpdParams& params, const std::string& filename);
    
    // 从YAML文件加载配置
    static DpdParams load_from_yaml(const std::string& filename);
    
    // 保存配置到YAML文件
    static bool save_to_yaml(const DpdParams& params, const std::string& filename);
    
    // 生成默认配置
    static DpdParams default_config();
    
    // 生成测试配置
    static DpdParams test_config(int32_t num_particles = 100);
    
    // 生成性能测试配置
    static DpdParams benchmark_config(int32_t num_particles = 10000);
};

#endif // DPD_HOST_H