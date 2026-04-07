// dpd_kernel.h
// DPD算子内核端头文件 - AI Core计算逻辑声明
// 遵循Ascend C编程规范，支持向量化优化

#ifndef DPD_KERNEL_H
#define DPD_KERNEL_H

#include <acl/acl.h>
#include <acl/acl_op.h>
#include <acl/acl_op_compiler.h>
#include "dpd_tiling.h"

// 向量化宽度配置
#ifdef __CCE_KT_TEST__
#define VEC_LEN_VALUE 8  // 测试环境向量长度
#else
#define VEC_LEN 16 // AI Core向量长度
#endif

// 数据类型定义
using fp32_t = float;
using fp16_t = half;

// 粒子数据结构体（对齐到32字节）
struct alignas(32) Particle {
    fp32_t x, y, z;     // 位置
    fp32_t vx, vy, vz;  // 速度
    fp32_t fx, fy, fz;  // 力
    fp32_t type;        // 粒子类型（用于a_ij参数）
};

// 模拟参数结构体
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

// 核函数声明
extern "C" __global__ __aicore__ void dpd_kernel(
    uint8_t* pos_gm,        // 位置数据（输入/输出）
    uint8_t* vel_gm,        // 速度数据（输入/输出）
    uint8_t* force_gm,      // 力数据（输出）
    uint8_t* params_gm,     // 模拟参数
    uint8_t* tiling_gm      // Tiling参数
);

// 辅助函数声明
__aicore__ inline fp32_t distance_squared(
    fp32_t dx, fp32_t dy, fp32_t dz);

__aicore__ inline fp32_t pbc_distance(
    fp32_t dx, fp32_t dy, fp32_t dz,
    const fp32_t box_size[3]);

__aicore__ inline void apply_pbc(
    fp32_t& x, fp32_t& y, fp32_t& z,
    const fp32_t box_size[3]);

// 向量化计算函数
template<int VEC_LEN>
__aicore__ inline void vec_load_particles(
    __gm__ uint8_t* gm_addr,
    __ub__ Particle* ub_buf,
    int32_t offset,
    int32_t num_particles);

template<int VEC_LEN>
__aicore__ inline void vec_store_particles(
    __ub__ Particle* ub_buf,
    __gm__ uint8_t* gm_addr,
    int32_t offset,
    int32_t num_particles);

// 力计算函数
__aicore__ inline void compute_dpd_forces(
    Particle& pi,
    Particle& pj,
    const SimParams& params,
    __ub__ fp32_t* rand_buf);

// 运动积分函数
__aicore__ inline void velocity_verlet_step1(
    Particle& p,
    const SimParams& params);

__aicore__ inline void velocity_verlet_step2(
    Particle& p,
    const SimParams& params);

#endif // DPD_KERNEL_H