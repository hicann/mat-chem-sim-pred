"""
LJ Force Fused Operator - Python Wrapper and Benchmark

对比:
1. PyTorch CPU 实现 (多次内核调用)
2. PyTorch NPU 实现 (多次内核调用)
3. 自定义融合算子 (单次内核调用)
"""

import numpy as np
import time
import ctypes
import os
import sys

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import torch_npu
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


# ============================================================================
# LJForceNPU - NPU 算子封装（测试用）
# ============================================================================

class LJForceNPU:
    """LJ Force NPU 算子封装"""

    def __init__(self):
        self.lib = None
        self.acl_initialized = False
        self.context = None
        self.stream = None

    def init_acl(self, device_id=0):
        try:
            import acl

            ret = acl.init()
            if ret != 0:
                pass

            ret = acl.rt.set_device(device_id)
            self.context, ret = acl.rt.create_context(device_id)
            self.stream, ret = acl.rt.create_stream()

            self.acl_initialized = True
            self._load_library()

        except ImportError:
            raise RuntimeError("acl module not available")

    def _load_library(self):
        ascend_home = os.environ.get('ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest')
        acl_lib_path = os.path.join(ascend_home, 'lib64', 'libascendcl.so')
        if os.path.exists(acl_lib_path):
            ctypes.CDLL(acl_lib_path, mode=ctypes.RTLD_GLOBAL)

        lib_paths = [
            "./build/lib/liblj_force.so",
            "../build/lib/liblj_force.so",
        ]
        for path in lib_paths:
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                break

        host_paths = [
            "./build/liblj_force_host.so",
            "../build/liblj_force_host.so",
        ]
        for path in host_paths:
            if os.path.exists(path):
                self.lib = ctypes.CDLL(path)
                return

        raise RuntimeError("Cannot find liblj_force_host.so")

    def __call__(self, positions, epsilon, sigma, cutoff):
        if not self.acl_initialized:
            self.init_acl()

        import acl

        N = positions.shape[0]
        positions_flat = np.ascontiguousarray(positions.flatten(), dtype=np.float32)

        # 计算核数和对齐步长（与 Kernel 保持一致）
        max_cores = 32
        min_atoms_per_core = 16
        optimal_cores = min((N + min_atoms_per_core - 1) // min_atoms_per_core, max_cores)
        optimal_cores = max(optimal_cores, 1)
        atoms_per_core = (N + optimal_cores - 1) // optimal_cores

        # 对齐步长
        force_stride = ((atoms_per_core * 3 + 7) // 8) * 8
        force_total_size = force_stride * optimal_cores
        forces_aligned = np.zeros(force_total_size, dtype=np.float32)
        energy_buf = np.zeros(8 * optimal_cores, dtype=np.float32)

        pos_size = positions_flat.nbytes
        force_size = forces_aligned.nbytes
        energy_size = energy_buf.nbytes
        workspace_size = 256

        pos_dev, _ = acl.rt.malloc(pos_size, 0)
        force_dev, _ = acl.rt.malloc(force_size, 0)
        energy_dev, _ = acl.rt.malloc(energy_size, 0)
        workspace_dev, _ = acl.rt.malloc(workspace_size, 0)

        acl.rt.memcpy(pos_dev, pos_size, positions_flat.ctypes.data, pos_size, 1)
        acl.rt.memcpy(force_dev, force_size, forces_aligned.ctypes.data, force_size, 1)
        acl.rt.memcpy(energy_dev, energy_size, energy_buf.ctypes.data, energy_size, 1)

        self.lib.aclnnLJForceDirect(
            ctypes.c_void_p(pos_dev),
            ctypes.c_void_p(force_dev),
            ctypes.c_void_p(energy_dev),
            ctypes.c_int32(N),
            ctypes.c_float(epsilon),
            ctypes.c_float(sigma),
            ctypes.c_float(cutoff),
            ctypes.c_void_p(workspace_dev),
            ctypes.c_uint64(workspace_size),
            ctypes.c_void_p(self.stream)
        )

        acl.rt.synchronize_stream(self.stream)

        acl.rt.memcpy(forces_aligned.ctypes.data, force_size, force_dev, force_size, 2)
        acl.rt.memcpy(energy_buf.ctypes.data, energy_size, energy_dev, energy_size, 2)

        acl.rt.free(pos_dev)
        acl.rt.free(force_dev)
        acl.rt.free(energy_dev)
        acl.rt.free(workspace_dev)

        # 提取力数据
        forces = np.zeros((N, 3), dtype=np.float32)
        for core_idx in range(optimal_cores):
            start_atom = core_idx * atoms_per_core
            end_atom = min(start_atom + atoms_per_core, N)
            if start_atom >= N:
                break
            src_offset = core_idx * force_stride
            for local_idx in range(end_atom - start_atom):
                atom_idx = start_atom + local_idx
                forces[atom_idx, 0] = forces_aligned[src_offset + local_idx * 3]
                forces[atom_idx, 1] = forces_aligned[src_offset + local_idx * 3 + 1]
                forces[atom_idx, 2] = forces_aligned[src_offset + local_idx * 3 + 2]

        # 提取能量
        energy = sum(energy_buf[core_idx * 8] for core_idx in range(optimal_cores))

        return forces, energy

    def __del__(self):
        if self.acl_initialized:
            try:
                import acl
                if self.stream:
                    acl.rt.destroy_stream(self.stream)
                if self.context:
                    acl.rt.destroy_context(self.context)
                acl.rt.reset_device(0)
                acl.finalize()
            except:
                pass


# ============================================================================
# 参考实现
# ============================================================================


def lj_force_numpy(positions, epsilon, sigma, cutoff):
    """
    NumPy 参考实现 - Lennard-Jones 力场计算

    Args:
        positions: [N, 3] 原子坐标
        epsilon: 势阱深度
        sigma: 零势能距离
        cutoff: 截断距离

    Returns:
        forces: [N, 3] 每个原子受到的力
        energy: 总势能
    """
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    energy = 0.0

    cutoff_sq = cutoff * cutoff
    sigma6 = sigma ** 6
    sigma12 = sigma ** 12

    for i in range(N):
        for j in range(i + 1, N):
            r_vec = positions[i] - positions[j]
            r_sq = np.sum(r_vec ** 2)

            if r_sq < cutoff_sq and r_sq > 1e-10:
                r2_inv = 1.0 / r_sq
                r6_inv = r2_inv ** 3
                sigma6_r6 = sigma6 * r6_inv
                sigma12_r12 = sigma6_r6 ** 2

                # 势能
                potential = 4.0 * epsilon * (sigma12_r12 - sigma6_r6)
                energy += potential

                # 力
                force_scalar = 24.0 * epsilon * r2_inv * (2.0 * sigma12_r12 - sigma6_r6)
                f_vec = force_scalar * r_vec

                forces[i] += f_vec
                forces[j] -= f_vec  # 牛顿第三定律

    return forces, energy


def lj_force_pytorch(positions, epsilon, sigma, cutoff):
    """
    PyTorch 实现 - 需要多次内核调用

    这是 PyTorch 的典型实现方式，展示了为什么融合算子更快
    """
    N = positions.shape[0]
    device = positions.device

    # Step 1: 计算所有原子对的距离向量 [N, N, 3]
    r_vec = positions.unsqueeze(1) - positions.unsqueeze(0)  # 内核调用 1

    # Step 2: 计算距离平方 [N, N]
    r_sq = (r_vec ** 2).sum(dim=-1)  # 内核调用 2, 3

    # Step 3: 创建掩码（排除自身和超出截断距离的原子对）
    cutoff_sq = cutoff * cutoff
    mask = (r_sq < cutoff_sq) & (r_sq > 1e-10)  # 内核调用 4, 5
    mask = mask & ~torch.eye(N, dtype=torch.bool, device=device)  # 内核调用 6

    # Step 4: 计算 LJ 势能和力（只对有效原子对）
    r_sq_safe = torch.where(mask, r_sq, torch.ones_like(r_sq))  # 内核调用 7
    r2_inv = 1.0 / r_sq_safe  # 内核调用 8
    r6_inv = r2_inv ** 3  # 内核调用 9

    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    sigma6_r6 = sigma6 * r6_inv  # 内核调用 10
    sigma12_r12 = sigma6_r6 ** 2  # 内核调用 11

    # Step 5: 势能
    potential = 4.0 * epsilon * (sigma12_r12 - sigma6_r6)  # 内核调用 12, 13
    potential = torch.where(mask, potential, torch.zeros_like(potential))  # 内核调用 14
    energy = potential.sum() / 2.0  # 除以2避免重复计算，内核调用 15

    # Step 6: 力
    force_scalar = 24.0 * epsilon * r2_inv * (2.0 * sigma12_r12 - sigma6_r6)  # 内核调用 16, 17, 18
    force_scalar = torch.where(mask, force_scalar, torch.zeros_like(force_scalar))  # 内核调用 19

    # Step 7: 力向量
    f_vec = force_scalar.unsqueeze(-1) * r_vec  # 内核调用 20

    # Step 8: 累加每个原子受到的力
    forces = f_vec.sum(dim=1)  # 内核调用 21

    return forces, energy


def benchmark_numpy(positions, epsilon, sigma, cutoff, warmup=2, iterations=10):
    """NumPy 性能测试"""
    for _ in range(warmup):
        _ = lj_force_numpy(positions, epsilon, sigma, cutoff)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = lj_force_numpy(positions, epsilon, sigma, cutoff)
    end = time.perf_counter()

    return (end - start) / iterations * 1000


def benchmark_pytorch_cpu(positions, epsilon, sigma, cutoff, warmup=5, iterations=20):
    """PyTorch CPU 性能测试"""
    if not HAS_TORCH:
        return None

    pos_tensor = torch.from_numpy(positions)

    for _ in range(warmup):
        _ = lj_force_pytorch(pos_tensor, epsilon, sigma, cutoff)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = lj_force_pytorch(pos_tensor, epsilon, sigma, cutoff)
    end = time.perf_counter()

    return (end - start) / iterations * 1000


def benchmark_custom_npu(positions, epsilon, sigma, cutoff, op, warmup=5, iterations=20):
    """自定义算子性能测试"""
    for _ in range(warmup):
        _ = op(positions, epsilon, sigma, cutoff)

    start = time.perf_counter()
    for _ in range(iterations):
        _ = op(positions, epsilon, sigma, cutoff)
    end = time.perf_counter()

    return (end - start) / iterations * 1000


def verify_precision(positions, epsilon, sigma, cutoff, op):
    """验证精度"""
    forces_ref, energy_ref = lj_force_numpy(positions, epsilon, sigma, cutoff)
    forces_npu, energy_npu = op(positions, epsilon, sigma, cutoff)

    force_max_err = np.max(np.abs(forces_npu - forces_ref))
    force_mean_err = np.mean(np.abs(forces_npu - forces_ref))
    energy_err = abs(energy_npu - energy_ref)

    # 相对误差
    if abs(energy_ref) > 1e-6:
        energy_rel_err = energy_err / abs(energy_ref)
    else:
        energy_rel_err = 0.0

    return {
        'force_max_err': force_max_err,
        'force_mean_err': force_mean_err,
        'energy_err': energy_err,
        'energy_rel_err': energy_rel_err,
        'forces_ref': forces_ref,
        'forces_npu': forces_npu,
        'energy_ref': energy_ref,
        'energy_npu': energy_npu,
    }


def run_benchmark(num_atoms, epsilon, sigma, cutoff, op):
    """运行单个配置的性能测试"""
    print(f"\n{'='*70}")
    print(f"Configuration: N_atoms={num_atoms}, ε={epsilon}, σ={sigma}, cutoff={cutoff}")
    print(f"Atom pairs: {num_atoms * (num_atoms - 1) // 2:,}")
    print(f"{'='*70}")

    # 生成随机原子坐标（在一个盒子内）
    np.random.seed(42)
    box_size = cutoff * 2
    positions = np.random.rand(num_atoms, 3).astype(np.float32) * box_size

    # 精度验证
    result = verify_precision(positions, epsilon, sigma, cutoff, op)
    print(f"\nPrecision (vs NumPy):")
    print(f"  Force max error:   {result['force_max_err']:.2e}")
    print(f"  Force mean error:  {result['force_mean_err']:.2e}")
    print(f"  Energy error:      {result['energy_err']:.2e}")
    print(f"  Energy rel error:  {result['energy_rel_err']:.2e}")
    print(f"  Energy ref:        {result['energy_ref']:.6f}")
    print(f"  Energy NPU:        {result['energy_npu']:.6f}")

    # 性能测试
    print(f"\nPerformance:")
    results = {}

    # NumPy (只对小规模测试)
    if num_atoms <= 200:
        time_numpy = benchmark_numpy(positions, epsilon, sigma, cutoff)
        results['NumPy'] = time_numpy
        print(f"  NumPy CPU:         {time_numpy:.3f} ms")
    else:
        print(f"  NumPy CPU:         (skipped, too slow)")

    # PyTorch CPU
    time_torch = benchmark_pytorch_cpu(positions, epsilon, sigma, cutoff)
    if time_torch:
        results['PyTorch CPU'] = time_torch
        print(f"  PyTorch CPU:       {time_torch:.3f} ms")

    # Custom NPU
    time_custom = benchmark_custom_npu(positions, epsilon, sigma, cutoff, op)
    results['Custom NPU'] = time_custom
    print(f"  Custom Ascend C:   {time_custom:.3f} ms")

    # 加速比
    print(f"\nSpeedup:")
    if 'PyTorch CPU' in results:
        print(f"  Custom vs PyTorch CPU: {results['PyTorch CPU'] / time_custom:.2f}x")
    if 'NumPy' in results:
        print(f"  Custom vs NumPy:       {results['NumPy'] / time_custom:.2f}x")

    return results


def main():
    print("="*70)
    print("LJ Force Fused Operator Benchmark")
    print("="*70)
    print(f"PyTorch available: {HAS_TORCH}")
    print(f"NPU available: {HAS_NPU}")

    # LJ 参数 (Argon)
    epsilon = 0.0103  # eV
    sigma = 3.4       # Angstrom
    cutoff = 10.0     # Angstrom

    # 初始化算子
    print("\nInitializing ACL...")
    try:
        op = LJForceNPU()
    except Exception as e:
        print(f"[ERROR] Cannot initialize NPU operator: {e}")
        return

    # 测试不同规模
    configs = [
        64,    # 小规模
        128,   # 中等规模
        256,   # 较大规模
        512,   # 大规模
    ]

    all_results = []
    for num_atoms in configs:
        try:
            results = run_benchmark(num_atoms, epsilon, sigma, cutoff, op)
            all_results.append((num_atoms, results))
        except Exception as e:
            print(f"Error with N={num_atoms}: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"{'N_atoms':<10} {'Pairs':<12} {'NumPy':<12} {'PyTorch':<12} {'Custom':<12} {'Speedup':<10}")
    print("-"*70)

    for num_atoms, results in all_results:
        pairs = num_atoms * (num_atoms - 1) // 2
        numpy_t = f"{results.get('NumPy', 0):.2f}" if results.get('NumPy') else "N/A"
        torch_t = f"{results.get('PyTorch CPU', 0):.2f}" if results.get('PyTorch CPU') else "N/A"
        custom_t = f"{results.get('Custom NPU', 0):.2f}"

        if results.get('PyTorch CPU') and results.get('Custom NPU'):
            speedup = f"{results['PyTorch CPU'] / results['Custom NPU']:.2f}x"
        else:
            speedup = "N/A"

        print(f"{num_atoms:<10} {pairs:<12} {numpy_t:<12} {torch_t:<12} {custom_t:<12} {speedup:<10}")


if __name__ == "__main__":
    main()
