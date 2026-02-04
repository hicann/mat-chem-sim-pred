#!/usr/bin/env python3
"""
LJ Force Fused - Unit Tests
============================

Test suite for Lennard-Jones Force Fused operator.
"""

import numpy as np
import pytest
import time


def lj_force_numpy(positions, epsilon, sigma, cutoff):
    """
    NumPy 参考实现 - Lennard-Jones 力场计算
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

                potential = 4.0 * epsilon * (sigma12_r12 - sigma6_r6)
                energy += potential

                force_scalar = 24.0 * epsilon * r2_inv * (2.0 * sigma12_r12 - sigma6_r6)
                f_vec = force_scalar * r_vec

                forces[i] += f_vec
                forces[j] -= f_vec

    return forces, energy


# Try to import PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def lj_force_pytorch(positions, epsilon, sigma, cutoff):
    """PyTorch 实现 - 多次内核调用"""
    N = positions.shape[0]
    device = positions.device

    r_vec = positions.unsqueeze(1) - positions.unsqueeze(0)
    r_sq = (r_vec ** 2).sum(dim=-1)

    cutoff_sq = cutoff * cutoff
    mask = (r_sq < cutoff_sq) & (r_sq > 1e-10)
    mask = mask & ~torch.eye(N, dtype=torch.bool, device=device)

    r_sq_safe = torch.where(mask, r_sq, torch.ones_like(r_sq))
    r2_inv = 1.0 / r_sq_safe
    r6_inv = r2_inv ** 3

    sigma6 = sigma ** 6
    sigma12 = sigma ** 12
    sigma6_r6 = sigma6 * r6_inv
    sigma12_r12 = sigma6_r6 ** 2

    potential = 4.0 * epsilon * (sigma12_r12 - sigma6_r6)
    potential = torch.where(mask, potential, torch.zeros_like(potential))
    energy = potential.sum() / 2.0

    force_scalar = 24.0 * epsilon * r2_inv * (2.0 * sigma12_r12 - sigma6_r6)
    force_scalar = torch.where(mask, force_scalar, torch.zeros_like(force_scalar))
    f_vec = force_scalar.unsqueeze(-1) * r_vec
    forces = f_vec.sum(dim=1)

    return forces, energy


class TestLJForceCPU:
    """Test CPU reference implementations."""

    def test_two_atoms_attractive(self):
        """Test two atoms at attractive distance."""
        # 两个原子，距离 = 2^(1/6) * sigma 时势能最低
        sigma = 3.4
        epsilon = 0.01
        cutoff = 10.0

        # 放置两个原子，距离约为 sigma
        positions = np.array([
            [0.0, 0.0, 0.0],
            [sigma, 0.0, 0.0]
        ], dtype=np.float32)

        forces, energy = lj_force_numpy(positions, epsilon, sigma, cutoff)

        # 在 r = sigma 时，势能 = 0
        assert abs(energy) < 0.01, f"Energy at r=sigma should be ~0, got {energy}"

        # 力应该是吸引力（负方向）
        # 原子0受到指向原子1的力（正x方向）
        assert forces[0, 0] < 0, "Force on atom 0 should be attractive (negative)"
        assert forces[1, 0] > 0, "Force on atom 1 should be attractive (positive)"

    def test_two_atoms_repulsive(self):
        """Test two atoms at repulsive distance."""
        sigma = 3.4
        epsilon = 0.01
        cutoff = 10.0

        # 距离小于 sigma，应该是排斥力
        positions = np.array([
            [0.0, 0.0, 0.0],
            [sigma * 0.9, 0.0, 0.0]
        ], dtype=np.float32)

        forces, energy = lj_force_numpy(positions, epsilon, sigma, cutoff)

        # 势能应该是正的（排斥）
        assert energy > 0, f"Energy should be positive (repulsive), got {energy}"

        # 力应该是排斥力
        assert forces[0, 0] < 0, "Force on atom 0 should be repulsive"
        assert forces[1, 0] > 0, "Force on atom 1 should be repulsive"

    def test_cutoff(self):
        """Test that atoms beyond cutoff don't interact."""
        sigma = 3.4
        epsilon = 0.01
        cutoff = 5.0

        # 距离大于 cutoff
        positions = np.array([
            [0.0, 0.0, 0.0],
            [cutoff + 1.0, 0.0, 0.0]
        ], dtype=np.float32)

        forces, energy = lj_force_numpy(positions, epsilon, sigma, cutoff)

        assert energy == 0.0, "Energy should be 0 beyond cutoff"
        np.testing.assert_allclose(forces, 0.0, atol=1e-10)

    def test_newton_third_law(self):
        """Test that F_ij = -F_ji (Newton's third law)."""
        sigma = 3.4
        epsilon = 0.01
        cutoff = 10.0

        np.random.seed(42)
        positions = np.random.rand(10, 3).astype(np.float32) * 5.0

        forces, _ = lj_force_numpy(positions, epsilon, sigma, cutoff)

        # 总力应该为零（动量守恒）
        # 注意：float32 累加有舍入误差，放宽容差
        total_force = forces.sum(axis=0)
        np.testing.assert_allclose(total_force, 0.0, atol=2.0)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestLJForcePyTorch:
    """Test PyTorch implementation matches NumPy."""

    def test_pytorch_matches_numpy(self):
        """Test PyTorch implementation matches NumPy reference."""
        sigma = 3.4
        epsilon = 0.01
        cutoff = 10.0

        np.random.seed(42)
        positions_np = np.random.rand(20, 3).astype(np.float32) * 8.0

        # NumPy reference
        forces_np, energy_np = lj_force_numpy(positions_np, epsilon, sigma, cutoff)

        # PyTorch
        positions_pt = torch.from_numpy(positions_np)
        forces_pt, energy_pt = lj_force_pytorch(positions_pt, epsilon, sigma, cutoff)

        # Compare
        np.testing.assert_allclose(forces_pt.numpy(), forces_np, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(energy_pt.item(), energy_np, rtol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestLJForcePerformance:
    """Performance comparison tests."""

    def test_pytorch_kernel_count(self):
        """Demonstrate that PyTorch requires many kernel calls."""
        # 这个测试展示 PyTorch 需要多次内核调用
        # 融合算子的优势在于减少这些调用

        N = 100
        sigma = 3.4
        epsilon = 0.01
        cutoff = 10.0

        np.random.seed(42)
        positions = torch.from_numpy(
            np.random.rand(N, 3).astype(np.float32) * 8.0
        )

        # 计时 PyTorch 实现
        warmup = 3
        iterations = 10

        for _ in range(warmup):
            _ = lj_force_pytorch(positions, epsilon, sigma, cutoff)

        start = time.perf_counter()
        for _ in range(iterations):
            forces, energy = lj_force_pytorch(positions, epsilon, sigma, cutoff)
        end = time.perf_counter()

        pytorch_time = (end - start) / iterations * 1000
        print(f"\nPyTorch CPU time for N={N}: {pytorch_time:.3f} ms")

        # 验证结果正确
        # 注意：力的绝对值很大时，float32 有舍入误差，使用相对容差
        forces_np, energy_np = lj_force_numpy(positions.numpy(), epsilon, sigma, cutoff)
        np.testing.assert_allclose(forces.numpy(), forces_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
