#!/usr/bin/env python3
"""
LJForceFused 快速测试脚本 (含 NPU)
"""
import os
import sys
import numpy as np
import time
import ctypes


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
                pass  # 可能已经初始化

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

        # 力输出：连续布局（N*3），不再分核对齐
        force_size = N * 3 * 4  # N*3 floats * 4 bytes
        forces_out = np.zeros(N * 3, dtype=np.float32)

        # 能量数组：每个核写 8 个 float
        max_cores = 32
        energy_buf = np.zeros(8 * max_cores, dtype=np.float32)

        pos_size = positions_flat.nbytes
        energy_size = energy_buf.nbytes
        workspace_size = 256

        pos_dev, _ = acl.rt.malloc(pos_size, 0)
        force_dev, _ = acl.rt.malloc(force_size, 0)
        energy_dev, _ = acl.rt.malloc(energy_size, 0)
        workspace_dev, _ = acl.rt.malloc(workspace_size, 0)

        acl.rt.memcpy(pos_dev, pos_size, positions_flat.ctypes.data, pos_size, 1)

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

        acl.rt.memcpy(forces_out.ctypes.data, force_size, force_dev, force_size, 2)
        acl.rt.memcpy(energy_buf.ctypes.data, energy_size, energy_dev, energy_size, 2)

        acl.rt.free(pos_dev)
        acl.rt.free(force_dev)
        acl.rt.free(energy_dev)
        acl.rt.free(workspace_dev)

        forces = forces_out.reshape(N, 3).copy()

        energy = energy_buf.sum()

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
# CPU 参考实现
# ============================================================================


def lj_force_numpy(positions, epsilon, sigma, cutoff):
    """NumPy 参考实现"""
    N = positions.shape[0]
    forces = np.zeros_like(positions, dtype=np.float64)
    energy = 0.0

    cutoff_sq = cutoff * cutoff
    sigma2 = sigma * sigma
    sigma6 = sigma2 * sigma2 * sigma2
    sigma12 = sigma6 * sigma6

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r_sq = dx * dx + dy * dy + dz * dz

            if r_sq < cutoff_sq and r_sq > 1e-10:
                r2_inv = 1.0 / r_sq
                r6_inv = r2_inv * r2_inv * r2_inv
                sigma6_r6 = sigma6 * r6_inv
                sigma12_r12 = sigma6_r6 * sigma6_r6

                potential = 4.0 * epsilon * (sigma12_r12 - sigma6_r6)
                energy += potential

                force_scalar = 24.0 * epsilon * r2_inv * (2.0 * sigma12_r12 - sigma6_r6)

                fx = force_scalar * dx
                fy = force_scalar * dy
                fz = force_scalar * dz

                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[i, 2] += fz
                forces[j, 0] -= fx
                forces[j, 1] -= fy
                forces[j, 2] -= fz

    return forces.astype(np.float32), energy


def lj_force_pytorch(positions, epsilon, sigma, cutoff):
    """PyTorch 向量化实现"""
    import torch
    N = positions.shape[0]
    device = positions.device

    r_vec = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_sq = (r_vec ** 2).sum(dim=-1)

    cutoff_sq = cutoff * cutoff
    valid = (r_sq < cutoff_sq) & (r_sq > 1e-10)
    upper_tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=device), diagonal=1)
    mask = valid & upper_tri

    r_sq_safe = torch.where(mask, r_sq, torch.ones_like(r_sq))
    r2_inv = 1.0 / r_sq_safe
    r6_inv = r2_inv ** 3

    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    sigma6_r6 = sigma6 * r6_inv
    sigma12_r12 = sigma6_r6 ** 2

    potential = 4.0 * epsilon * (sigma12_r12 - sigma6_r6)
    potential = torch.where(mask, potential, torch.zeros_like(potential))
    energy = potential.sum()

    force_scalar = 24.0 * epsilon * r2_inv * (2.0 * sigma12_r12 - sigma6_r6)
    force_scalar = torch.where(mask, force_scalar, torch.zeros_like(force_scalar))
    f_vec = force_scalar.unsqueeze(-1) * r_vec
    forces = f_vec.sum(dim=1) - f_vec.sum(dim=0)

    return forces, energy


def test_cpu():
    """测试 CPU 参考实现"""
    print("=" * 60)
    print("CPU 参考实现测试")
    print("=" * 60)

    sigma, epsilon, cutoff = 3.4, 0.01, 10.0

    print("\n--- 两原子测试 ---")
    positions = np.array([[0, 0, 0], [sigma, 0, 0]], dtype=np.float32)
    forces, energy = lj_force_numpy(positions, epsilon, sigma, cutoff)
    total = forces.sum(axis=0)
    status = "PASS" if np.allclose(total, 0, atol=1e-6) else "FAIL"
    print(f"  能量: {energy:.6f}, 总力: {total}, 牛顿第三: [{status}]")

    print("\n--- 20原子测试 ---")
    np.random.seed(42)
    positions = np.random.rand(20, 3).astype(np.float32) * 15.0
    forces, energy = lj_force_numpy(positions, epsilon, sigma, cutoff)
    total = forces.sum(axis=0)
    status = "PASS" if np.allclose(total, 0, atol=1e-4) else "FAIL"
    print(f"  能量: {energy:.6f}, 总力: {total}, 牛顿第三: [{status}]")


def test_npu():
    """测试 NPU 融合算子"""
    print("\n" + "=" * 60)
    print("NPU 融合算子测试")
    print("=" * 60)

    try:
        op = LJForceNPU()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    sigma, epsilon, cutoff = 3.4, 0.01, 10.0

    test_cases = [(20, "小"), (50, "中"), (100, "大")]

    for N, desc in test_cases:
        print(f"\n--- {desc}规模: {N} 原子 ---")
        np.random.seed(42)
        box = max(15.0, N ** (1/3) * 4.0)
        positions = np.random.rand(N, 3).astype(np.float32) * box

        # CPU 参考
        forces_cpu, energy_cpu = lj_force_numpy(positions, epsilon, sigma, cutoff)

        # NPU
        forces_npu, energy_npu = op(positions, epsilon, sigma, cutoff)

        # 比较
        force_err = np.abs(forces_npu - forces_cpu).max()
        energy_err = abs(energy_npu - energy_cpu)
        force_rel = force_err / (np.abs(forces_cpu).max() + 1e-10)
        energy_rel = energy_err / (abs(energy_cpu) + 1e-10)

        status = "PASS" if force_rel < 0.01 and energy_rel < 0.01 else "FAIL"
        print(f"  力误差: {force_err:.2e} (相对: {force_rel:.2e})")
        print(f"  能量: CPU={energy_cpu:.4f}, NPU={energy_npu:.4f}, 误差={energy_rel:.2e}")
        print(f"  状态: [{status}]")


def benchmark():
    """性能对比"""
    print("\n" + "=" * 60)
    print("性能对比 (NumPy vs PyTorch vs NPU)")
    print("=" * 60)

    try:
        import torch
        op = LJForceNPU()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    sigma, epsilon, cutoff = 3.4, 0.01, 10.0
    test_cases = [64, 128, 256]

    print(f"\n{'N':<8} {'Pairs':<10} {'NumPy':<12} {'PyTorch':<12} {'NPU':<12} {'加速比':<10}")
    print("-" * 64)

    for N in test_cases:
        np.random.seed(42)
        box = max(15.0, N ** (1/3) * 4.0)
        pos_np = np.random.rand(N, 3).astype(np.float32) * box
        pos_pt = torch.from_numpy(pos_np)
        pairs = N * (N - 1) // 2

        # NumPy
        start = time.perf_counter()
        for _ in range(3):
            lj_force_numpy(pos_np, epsilon, sigma, cutoff)
        numpy_ms = (time.perf_counter() - start) / 3 * 1000

        # PyTorch
        for _ in range(3):
            lj_force_pytorch(pos_pt, epsilon, sigma, cutoff)
        start = time.perf_counter()
        for _ in range(10):
            lj_force_pytorch(pos_pt, epsilon, sigma, cutoff)
        pytorch_ms = (time.perf_counter() - start) / 10 * 1000

        # NPU
        for _ in range(3):
            op(pos_np, epsilon, sigma, cutoff)
        start = time.perf_counter()
        for _ in range(10):
            op(pos_np, epsilon, sigma, cutoff)
        npu_ms = (time.perf_counter() - start) / 10 * 1000

        speedup = pytorch_ms / npu_ms if npu_ms > 0 else 0
        print(f"{N:<8} {pairs:<10} {numpy_ms:<12.2f} {pytorch_ms:<12.2f} {npu_ms:<12.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    test_cpu()
    test_npu()
    benchmark()
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
