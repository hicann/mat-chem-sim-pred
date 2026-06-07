#!/usr/bin/env python3
"""
DPD算子Python测试脚本
包含数据生成、算子调用、结果校验
"""

import numpy as np
import sys
import os
import time
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "python"))

try:
    import dpd_op  # Python绑定模块
    HAS_DPD_OP = True
except ImportError:
    HAS_DPD_OP = False
    print("警告: 未找到dpd_op模块，使用模拟测试")

class DpdTest:
    """DPD算子测试类"""
    
    def __init__(self, test_dir="test_dpd_op"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        
        # 测试参数
        self.params = {
            "dt": 0.01,
            "rc": 1.0,
            "a_ij": 25.0,
            "gamma": 4.5,
            "sigma": 3.0,
            "kBT": 1.0,
            "box_size": [10.0, 10.0, 10.0],
            "num_particles": 100,
            "num_steps": 10,
            "random_seed": 12345
        }
        
        # 测试数据
        self.positions = None
        self.velocities = None
        self.forces = None
        
    def generate_test_data(self):
        """生成测试数据"""
        np.random.seed(self.params["random_seed"])
        n = self.params["num_particles"]
        
        # 生成随机位置（在盒子内）
        box_size = np.array(self.params["box_size"])
        self.positions = np.random.rand(n, 3) * box_size
        
        # 生成随机速度（满足麦克斯韦分布）
        temperature = self.params["kBT"]
        scale = np.sqrt(temperature)
        self.velocities = np.random.normal(0, scale, (n, 3))
        
        # 去除整体动量
        avg_velocity = np.mean(self.velocities, axis=0)
        self.velocities -= avg_velocity
        
        # 初始化力为零
        self.forces = np.zeros((n, 3))
        
        print(f"生成测试数据: {n}个粒子")
        print(f"位置范围: [{self.positions.min():.3f}, {self.positions.max():.3f}]")
        print(f"速度范围: [{self.velocities.min():.3f}, {self.velocities.max():.3f}]")
        
    def save_test_data(self):
        """保存测试数据到文件"""
        data_dir = self.test_dir / "dpd_test_data"
        data_dir.mkdir(exist_ok=True)
        
        # 保存二进制数据
        self.positions.astype(np.float32).tofile(data_dir / "input_pos.bin")
        self.velocities.astype(np.float32).tofile(data_dir / "input_vel.bin")
        
        # 保存参数
        with open(data_dir / "test_params.json", "w") as f:
            json.dump(self.params, f, indent=2)
        
        print(f"测试数据已保存到: {data_dir}")
        
    def compute_reference(self):
        """计算参考结果（CPU实现）"""
        n = self.params["num_particles"]
        dt = self.params["dt"]
        rc = self.params["rc"]
        a_ij = self.params["a_ij"]
        gamma = self.params["gamma"]
        sigma = self.params["sigma"]
        box_size = np.array(self.params["box_size"])
        
        # 初始化结果
        new_positions = self.positions.copy()
        new_velocities = self.velocities.copy()
        new_forces = np.zeros((n, 3))
        
        # Velocity-Verlet第一步
        half_dt = dt * 0.5
        new_velocities += self.forces * half_dt  # 假设质量m=1
        new_positions += new_velocities * dt
        
        # 计算力（简化版本，只计算保守力）
        rc_sq = rc * rc
        for i in range(n):
            for j in range(i + 1, n):
                # 相对位置（考虑PBC）
                rij = new_positions[j] - new_positions[i]
                rij = rij - np.round(rij / box_size) * box_size
                
                r_sq = np.sum(rij * rij)
                
                if r_sq < rc_sq:
                    r = np.sqrt(r_sq)
                    w = 1.0 - r / rc
                    
                    # 保守力
                    fc_mag = a_ij * w
                    fc = fc_mag * rij / r
                    
                    new_forces[i] += fc
                    new_forces[j] -= fc  # 牛顿第三定律
        
        # Velocity-Verlet第二步
        new_velocities += new_forces * half_dt
        
        # 应用周期性边界条件
        new_positions = new_positions - np.floor(new_positions / box_size) * box_size
        
        return new_positions, new_velocities, new_forces
    
    def run_op_test(self):
        """运行算子测试"""
        if not HAS_DPD_OP:
            print("跳过算子测试: 未找到dpd_op模块")
            return None
        
        try:
            # 创建模拟器
            simulator = dpd_op.DpdSimulator()
            
            # 设置参数
            params = dpd_op.DpdParams()
            for key, value in self.params.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            
            # 初始化
            if not simulator.initialize(params):
                print(f"初始化失败: {simulator.get_last_error()}")
                return None
            
            # 设置粒子数据
            particle_data = dpd_op.ParticleData(self.params["num_particles"])
            particle_data.positions = self.positions.flatten().tolist()
            particle_data.velocities = self.velocities.flatten().tolist()
            simulator.set_particle_data(particle_data)
            
            # 运行模拟
            start_time = time.time()
            result = simulator.run_step(self.params["num_steps"])
            elapsed = time.time() - start_time
            
            if not result.success:
                print(f"模拟失败: {result.error_msg}")
                return None
            
            # 提取结果
            final_state = result.final_state
            op_positions = np.array(final_state.positions).reshape(-1, 3)
            op_velocities = np.array(final_state.velocities).reshape(-1, 3)
            op_forces = np.array(final_state.forces).reshape(-1, 3)
            
            print(f"算子测试完成: {elapsed:.3f}秒")
            print(f"性能: {self.params['num_steps']/elapsed:.1f} 步/秒")
            
            return op_positions, op_velocities, op_forces
            
        except Exception as e:
            print(f"算子测试异常: {e}")
            return None
    
    def compare_results(self, ref_result, op_result):
        """比较参考结果和算子结果"""
        if op_result is None:
            return False
        
        ref_pos, ref_vel, ref_force = ref_result
        op_pos, op_vel, op_force = op_result
        
        # 计算误差
        pos_error = np.max(np.abs(ref_pos - op_pos))
        vel_error = np.max(np.abs(ref_vel - op_vel))
        force_error = np.max(np.abs(ref_force - op_force))
        
        print("\n结果比较:")
        print(f"位置最大误差: {pos_error:.6f}")
        print(f"速度最大误差: {vel_error:.6f}")
        print(f"力最大误差: {force_error:.6f}")
        
        # 设置容差
        pos_tol = 1e-4
        vel_tol = 1e-4
        force_tol = 1e-3
        
        pos_ok = pos_error < pos_tol
        vel_ok = vel_error < vel_tol
        force_ok = force_error < force_tol
        
        if pos_ok and vel_ok and force_ok:
            print("✅ 所有测试通过!")
            return True
        else:
            print("❌ 测试失败:")
            if not pos_ok:
                print(f"  位置误差超过容差: {pos_error:.6f} > {pos_tol}")
            if not vel_ok:
                print(f"  速度误差超过容差: {vel_error:.6f} > {vel_tol}")
            if not force_ok:
                print(f"  力误差超过容差: {force_error:.6f} > {force_tol}")
            return False
    
    def run_performance_test(self, num_particles_list=[100, 1000, 10000]):
        """运行性能测试"""
        if not HAS_DPD_OP:
            print("跳过性能测试: 未找到dpd_op模块")
            return
        
        print("\n" + "="*50)
        print("性能测试")
        print("="*50)
        
        results = []
        
        for num_particles in num_particles_list:
            print(f"\n测试粒子数: {num_particles}")
            
            # 更新参数
            self.params["num_particles"] = num_particles
            self.generate_test_data()
            
            # 创建模拟器
            simulator = dpd_op.DpdSimulator()
            params = dpd_op.DpdParams()
            for key, value in self.params.items():
                if hasattr(params, key):
                    setattr(params, key, value)
            
            # 初始化
            if not simulator.initialize(params):
                print(f"初始化失败")
                continue
            
            # 设置粒子数据
            particle_data = dpd_op.ParticleData(num_particles)
            particle_data.positions = self.positions.flatten().tolist()
            particle_data.velocities = self.velocities.flatten().tolist()
            simulator.set_particle_data(particle_data)
            
            # 预热
            simulator.run_step(5)
            
            # 正式测试
            num_steps = 100
            start_time = time.time()
            result = simulator.run_step(num_steps)
            elapsed = time.time() - start_time
            
            if result.success:
                steps_per_second = num_steps / elapsed
                results.append({
                    "num_particles": num_particles,
                    "time": elapsed,
                    "steps_per_second": steps_per_second,
                    "time_per_step": elapsed / num_steps
                })
                
                print(f"  时间: {elapsed:.3f}秒")
                print(f"  性能: {steps_per_second:.1f} 步/秒")
                print(f"  每步时间: {elapsed/num_steps*1000:.2f}毫秒")
            else:
                print(f"  测试失败: {result.error_msg}")
        
        # 输出性能报告
        if results:
            print("\n性能总结:")
            print("-"*50)
            print(f"{'粒子数':<10} {'总时间(秒)':<12} {'步/秒':<10} {'毫秒/步':<10}")
            print("-"*50)
            for r in results:
                print(f"{r['num_particles']:<10} {r['time']:<12.3f} "
                      f"{r['steps_per_second']:<10.1f} {r['time_per_step']*1000:<10.2f}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*50)
        print("DPD算子测试套件")
        print("="*50)
        
        # 1. 生成测试数据
        print("\n1. 生成测试数据...")
        self.generate_test_data()
        self.save_test_data()
        
        # 2. 计算参考结果
        print("\n2. 计算参考结果...")
        ref_result = self.compute_reference()
        
        # 保存参考结果
        ref_pos, ref_vel, ref_force = ref_result
        data_dir = self.test_dir / "dpd_test_data"
        ref_pos.astype(np.float32).tofile(data_dir / "expect_output.bin")
        
        # 3. 运行算子测试
        print("\n3. 运行算子测试...")
        op_result = self.run_op_test()
        
        # 4. 比较结果
        print("\n4. 比较结果...")
        test_passed = self.compare_results(ref_result, op_result)
        
        # 5. 性能测试
        if test_passed:
            self.run_performance_test()
        
        return test_passed

def main():
    """主函数"""
    test = DpdTest()
    
    # 运行所有测试
    success = test.run_all_tests()
    
    # 输出测试报告
    report_file = test.test_dir / "test_report.txt"
    with open(report_file, "w") as f:
        f.write("DPD算子测试报告\n")
        f.write("="*50 + "\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"测试状态: {'通过' if success else '失败'}\n")
        f.write(f"Python绑定: {'可用' if HAS_DPD_OP else '不可用'}\n")
    
    print(f"\n测试报告已保存: {report_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())