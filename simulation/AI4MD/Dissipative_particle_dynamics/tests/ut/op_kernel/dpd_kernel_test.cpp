// dpd_kernel_test.cpp
// DPD算子内核单元测试
// 使用ACL测试框架验证核函数正确性

#include <gtest/gtest.h>
#include <acl/acl.h>
#include <acl/acl_op.h>
#include "dpd_kernel.h"
#include "dpd_tiling.h"
#include <vector>
#include <random>
#include <cmath>

// 测试固件类
class DpdKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化ACL
        aclError ret = aclInit(nullptr);
        ASSERT_EQ(ret, ACL_SUCCESS);
        
        // 设置设备
        ret = aclrtSetDevice(0);
        ASSERT_EQ(ret, ACL_SUCCESS);
        
        // 创建流
        ret = aclrtCreateStream(&stream_);
        ASSERT_EQ(ret, ACL_SUCCESS);
        
        // 初始化测试数据
        init_test_data();
    }
    
    void TearDown() override {
        // 释放设备内存
        free_device_memory();
        
        // 销毁流
        if (stream_) {
            aclrtDestroyStream(stream_);
        }
        
        // 重置设备
        aclrtResetDevice(0);
        
        // 去初始化ACL
        aclFinalize();
    }
    
    void init_test_data() {
        // 测试参数
        params_.dt = 0.01f;
        params_.rc = 1.0f;
        params_.a_ij = 25.0f;
        params_.gamma = 4.5f;
        params_.sigma = 3.0f;
        params_.kBT = 1.0f;
        params_.box_size[0] = 10.0f;
        params_.box_size[1] = 10.0f;
        params_.box_size[2] = 10.0f;
        params_.num_particles = 64;  // 测试用较小系统
        params_.seed = 12345;
        
        // 生成测试粒子
        std::mt19937 rng(params_.seed);
        std::uniform_real_distribution<float> pos_dist(0.0f, 10.0f);
        std::normal_distribution<float> vel_dist(0.0f, 1.0f);
        
        particles_.resize(params_.num_particles);
        for (auto& p : particles_) {
            p.x = pos_dist(rng);
            p.y = pos_dist(rng);
            p.z = pos_dist(rng);
            
            p.vx = vel_dist(rng);
            p.vy = vel_dist(rng);
            p.vz = vel_dist(rng);
            
            p.fx = 0.0f;
            p.fy = 0.0f;
            p.fz = 0.0f;
            p.type = 0.0f;
        }
        
        // 计算Tiling
        tiling_ = calculate_tiling(params_.num_particles, 2, 16);
    }
    
    bool allocate_device_memory() {
        size_t particle_size = particles_.size() * sizeof(Particle);
        size_t params_size = sizeof(SimParams);
        size_t tiling_size = sizeof(DpdTiling);
        
        aclError ret;
        
        // 分配粒子内存
        ret = aclrtMalloc(&d_particles_, particle_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return false;
        
        // 分配参数内存
        ret = aclrtMalloc(&d_params_, params_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return false;
        
        // 分配Tiling内存
        ret = aclrtMalloc(&d_tiling_, tiling_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) return false;
        
        // 拷贝数据到设备
        ret = aclrtMemcpy(d_particles_, particle_size,
                         particles_.data(), particle_size,
                         ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) return false;
        
        ret = aclrtMemcpy(d_params_, params_size,
                         &params_, params_size,
                         ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) return false;
        
        ret = aclrtMemcpy(d_tiling_, tiling_size,
                         &tiling_, tiling_size,
                         ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) return false;
        
        return true;
    }
    
    void free_device_memory() {
        if (d_particles_) aclrtFree(d_particles_);
        if (d_params_) aclrtFree(d_params_);
        if (d_tiling_) aclrtFree(d_tiling_);
    }
    
    // 计算参考结果（CPU）
    std::vector<Particle> compute_reference() {
        std::vector<Particle> ref = particles_;
        float dt = params_.dt;
        float rc = params_.rc;
        float a_ij = params_.a_ij;
        float box_size[3] = {params_.box_size[0], params_.box_size[1], params_.box_size[2]};
        
        // Velocity-Verlet第一步
        float half_dt = dt * 0.5f;
        for (auto& p : ref) {
            p.vx += p.fx * half_dt;
            p.vy += p.fy * half_dt;
            p.vz += p.fz * half_dt;
            
            p.x += p.vx * dt;
            p.y += p.vy * dt;
            p.z += p.vz * dt;
            
            p.fx = 0.0f;
            p.fy = 0.0f;
            p.fz = 0.0f;
        }
        
        // 计算力（只计算保守力作为测试）
        float rc_sq = rc * rc;
        for (size_t i = 0; i < ref.size(); ++i) {
            for (size_t j = i + 1; j < ref.size(); ++j) {
                float dx = ref[j].x - ref[i].x;
                float dy = ref[j].y - ref[i].y;
                float dz = ref[j].z - ref[i].z;
                
                // 应用PBC
                dx = dx - round(dx / box_size[0]) * box_size[0];
                dy = dy - round(dy / box_size[1]) * box_size[1];
                dz = dz - round(dz / box_size[2]) * box_size[2];
                
                float r_sq = dx * dx + dy * dy + dz * dz;
                
                if (r_sq < rc_sq) {
                    float r = sqrt(r_sq);
                    float w = 1.0f - r / rc;
                    float fc_mag = a_ij * w;
                    
                    float inv_r = 1.0f / r;
                    float fx = fc_mag * dx * inv_r;
                    float fy = fc_mag * dy * inv_r;
                    float fz = fc_mag * dz * inv_r;
                    
                    ref[i].fx += fx;
                    ref[i].fy += fy;
                    ref[i].fz += fz;
                    
                    ref[j].fx -= fx;
                    ref[j].fy -= fy;
                    ref[j].fz -= fz;
                }
            }
        }
        
        // Velocity-Verlet第二步
        for (auto& p : ref) {
            p.vx += p.fx * half_dt;
            p.vy += p.fy * half_dt;
            p.vz += p.fz * half_dt;
        }
        
        // 应用PBC
        for (auto& p : ref) {
            p.x = p.x - floor(p.x / box_size[0]) * box_size[0];
            p.y = p.y - floor(p.y / box_size[1]) * box_size[1];
            p.z = p.z - floor(p.z / box_size[2]) * box_size[2];
        }
        
        return ref;
    }
    
protected:
    aclrtStream stream_;
    SimParams params_;
    DpdTiling tiling_;
    std::vector<Particle> particles_;
    
    void* d_particles_ = nullptr;
    void* d_params_ = nullptr;
    void* d_tiling_ = nullptr;
};

// 测试：Tiling计算
TEST_F(DpdKernelTest, TilingCalculation) {
    DpdTiling tiling = calculate_tiling(1000, 32, 16);
    
    EXPECT_GT(tiling.total_particles, 0);
    EXPECT_GT(tiling.block_size, 0);
    EXPECT_GT(tiling.num_blocks, 0);
    EXPECT_GT(tiling.ub_particle_capacity, 0);
    
    // 验证Tiling有效性
    EXPECT_TRUE(validate_tiling(tiling));
}

// 测试：距离计算
TEST_F(DpdKernelTest, DistanceCalculation) {
    // 测试距离平方计算
    float dx = 1.0f, dy = 2.0f, dz = 3.0f;
    float expected = dx*dx + dy*dy + dz*dz;
    
    // 注意：这里测试的是主机端函数，实际核函数需要单独测试
    EXPECT_FLOAT_EQ(distance_squared(dx, dy, dz), expected);
}

// 测试：周期性边界条件
TEST_F(DpdKernelTest, PeriodicBoundary) {
    float x = 12.5f, y = -3.2f, z = 8.7f;
    float box_size[3] = {10.0f, 10.0f, 10.0f};
    
    float orig_x = x, orig_y = y, orig_z = z;
    apply_pbc(x, y, z, box_size);
    
    // 验证结果在[0, box_size)范围内
    EXPECT_GE(x, 0.0f);
    EXPECT_LT(x, box_size[0]);
    EXPECT_GE(y, 0.0f);
    EXPECT_LT(y, box_size[1]);
    EXPECT_GE(z, 0.0f);
    EXPECT_LT(z, box_size[2]);
    
    // 验证周期性
    float diff_x = fabs(x - (orig_x - floor(orig_x / box_size[0]) * box_size[0]));
    float diff_y = fabs(y - (orig_y - floor(orig_y / box_size[1]) * box_size[1]));
    float diff_z = fabs(z - (orig_z - floor(orig_z / box_size[2]) * box_size[2]));
    
    EXPECT_LT(diff_x, 1e-6f);
    EXPECT_LT(diff_y, 1e-6f);
    EXPECT_LT(diff_z, 1e-6f);
}

// 测试：设备内存管理
TEST_F(DpdKernelTest, DeviceMemory) {
    EXPECT_TRUE(allocate_device_memory());
    
    // 验证内存分配成功
    EXPECT_NE(d_particles_, nullptr);
    EXPECT_NE(d_params_, nullptr);
    EXPECT_NE(d_tiling_, nullptr);
}

// 测试：数据拷贝
TEST_F(DpdKernelTest, DataCopy) {
    if (!allocate_device_memory()) {
        GTEST_SKIP() << "设备内存分配失败";
    }
    
    // 从设备读回数据验证
    std::vector<Particle> device_particles(particles_.size());
    size_t particle_size = particles_.size() * sizeof(Particle);
    
    aclError ret = aclrtMemcpy(device_particles.data(), particle_size,
                              d_particles_, particle_size,
                              ACL_MEMCPY_DEVICE_TO_HOST);
    
    EXPECT_EQ(ret, ACL_SUCCESS);
    
    // 比较数据
    for (size_t i = 0; i < particles_.size(); ++i) {
        EXPECT_FLOAT_EQ(device_particles[i].x, particles_[i].x);
        EXPECT_FLOAT_EQ(device_particles[i].y, particles_[i].y);
        EXPECT_FLOAT_EQ(device_particles[i].z, particles_[i].z);
    }
}

// 测试：力计算（简化）
TEST_F(DpdKernelTest, ForceCalculation) {
    // 创建两个测试粒子
    Particle p1, p2;
    p1.x = 0.0f; p1.y = 0.0f; p1.z = 0.0f;
    p2.x = 0.5f; p2.y = 0.0f; p2.z = 0.0f;  // 距离0.5 < rc=1.0
    
    p1.vx = 0.0f; p1.vy = 0.0f; p1.vz = 0.0f;
    p2.vx = 0.0f; p2.vy = 0.0f; p2.vz = 0.0f;
    
    p1.fx = 0.0f; p1.fy = 0.0f; p1.fz = 0.0f;
    p2.fx = 0.0f; p2.fy = 0.0f; p2.fz = 0.0f;
    
    SimParams params;
    params.rc = 1.0f;
    params.a_ij = 25.0f;
    params.gamma = 4.5f;
    params.sigma = 3.0f;
    params.box_size[0] = 10.0f;
    params.box_size[1] = 10.0f;
    params.box_size[2] = 10.0f;
    
    // 计算力
    float rand_buf[3] = {0.0f, 0.0f, 0.0f};  // 零随机数
    compute_dpd_forces(p1, p2, params, rand_buf);
    
    // 验证保守力
    float r = 0.5f;
    float w = 1.0f - r / params.rc;
    float expected_fc = params.a_ij * w;
    
    EXPECT_FLOAT_EQ(p1.fx, expected_fc);  // p1受到+x方向的力
    EXPECT_FLOAT_EQ(p2.fx, -expected_fc); // p2受到-x方向的力
    EXPECT_FLOAT_EQ(p1.fy, 0.0f);
    EXPECT_FLOAT_EQ(p1.fz, 0.0f);
    EXPECT_FLOAT_EQ(p2.fy, 0.0f);
    EXPECT_FLOAT_EQ(p2.fz, 0.0f);
}

// 测试：运动积分
TEST_F(DpdKernelTest, VelocityVerlet) {
    Particle p;
    p.x = 0.0f; p.y = 0.0f; p.z = 0.0f;
    p.vx = 1.0f; p.vy = 2.0f; p.vz = 3.0f;
    p.fx = 0.1f; p.fy = 0.2f; p.fz = 0.3f;
    
    SimParams params;
    params.dt = 0.01f;
    
    // 第一步
    velocity_verlet_step1(p, params);
    
    // 验证第一步结果
    float half_dt = params.dt * 0.5f;
    float expected_vx = 1.0f + 0.1f * half_dt;
    float expected_vy = 2.0f + 0.2f * half_dt;
    float expected_vz = 3.0f + 0.3f * half_dt;
    
    EXPECT_FLOAT_EQ(p.vx, expected_vx);
    EXPECT_FLOAT_EQ(p.vy, expected_vy);
    EXPECT_FLOAT_EQ(p.vz, expected_vz);
    
    float expected_x = 0.0f + p.vx * params.dt;
    float expected_y = 0.0f + p.vy * params.dt;
    float expected_z = 0.0f + p.vz * params.dt;
    
    EXPECT_FLOAT_EQ(p.x, expected_x);
    EXPECT_FLOAT_EQ(p.y, expected_y);
    EXPECT_FLOAT_EQ(p.z, expected_z);
    
    EXPECT_FLOAT_EQ(p.fx, 0.0f);
    EXPECT_FLOAT_EQ(p.fy, 0.0f);
    EXPECT_FLOAT_EQ(p.fz, 0.0f);
    
    // 假设力被重新计算为新的值
    p.fx = 0.2f; p.fy = 0.3f; p.fz = 0.4f;
    
    // 第二步
    velocity_verlet_step2(p, params);
    
    expected_vx += 0.2f * half_dt;
    expected_vy += 0.3f * half_dt;
    expected_vz += 0.4f * half_dt;
    
    EXPECT_FLOAT_EQ(p.vx, expected_vx);
    EXPECT_FLOAT_EQ(p.vy, expected_vy);
    EXPECT_FLOAT_EQ(p.vz, expected_vz);
}

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}