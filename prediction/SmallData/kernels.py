"""
高斯过程核函数 — 协方差函数集合，支持 RBF、Matern、Periodic 等常用核。

每个核函数提供 NumPy 参考实现和 PyTorch 实现，
方便后续迁移到 Ascend C 时验证正确性。
"""

import torch
import torch.nn as nn
import numpy as np


# ============================================================================
# 核函数基类
# ============================================================================

class Kernel(nn.Module):
    """Base class for GP kernels."""

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def diag(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], device=x.device)


# ============================================================================
# RBF (Radial Basis Function) / 平方指数核
# ============================================================================

class RBFKernel(Kernel):
    """
    k(x, y) = σ² * exp(-||x - y||² / (2 * l²))

    Args:
        length_scale: length scale parameter
        variance: output scale (signal variance)
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))

    @property
    def length_scale(self):
        return torch.exp(self.log_length_scale)

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        sq_dist = torch.cdist(x1 / self.length_scale, x2 / self.length_scale).pow(2)
        return self.variance * torch.exp(-0.5 * sq_dist)


def rbf_kernel_np(x1: np.ndarray, x2: np.ndarray,
                  length_scale: float = 1.0, variance: float = 1.0) -> np.ndarray:
    """NumPy reference for RBF kernel."""
    sq_dist = np.sum((x1[:, None] / length_scale - x2[None] / length_scale) ** 2, axis=-1)
    return variance * np.exp(-0.5 * sq_dist)


# ============================================================================
# Matern 核
# ============================================================================

class MaternKernel(Kernel):
    """
    Matern 核函数，支持 ν = 1/2, 3/2, 5/2

    k(x, y) = σ² * (2^{1-ν} / Γ(ν)) * (√(2ν) * d/l)^ν * K_ν(√(2ν) * d/l)
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, nu: float = 2.5):
        super().__init__()
        self.nu = nu
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))

    @property
    def length_scale(self):
        return torch.exp(self.log_length_scale)

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x1 / self.length_scale, x2 / self.length_scale)

        if self.nu == 0.5:
            return self.variance * torch.exp(-dist)
        elif self.nu == 1.5:
            sqrt3 = torch.tensor(1.7320508075688772, device=dist.device)
            scaled = sqrt3 * dist
            return self.variance * (1 + scaled) * torch.exp(-scaled)
        elif self.nu == 2.5:
            sqrt5 = torch.tensor(2.23606797749979, device=dist.device)
            scaled = sqrt5 * dist
            return self.variance * (1 + scaled + scaled ** 2 / 3) * torch.exp(-scaled)
        else:
            raise ValueError(f"Unsupported nu: {self.nu}. Use 0.5, 1.5, or 2.5.")


def matern_kernel_np(x1: np.ndarray, x2: np.ndarray,
                     length_scale: float = 1.0, variance: float = 1.0,
                     nu: float = 2.5) -> np.ndarray:
    """NumPy reference for Matern kernel."""
    dist = np.sqrt(np.sum((x1[:, None] / length_scale - x2[None] / length_scale) ** 2, axis=-1))

    if nu == 0.5:
        return variance * np.exp(-dist)
    elif nu == 1.5:
        scaled = 1.7320508075688772 * dist
        return variance * (1 + scaled) * np.exp(-scaled)
    elif nu == 2.5:
        scaled = 2.23606797749979 * dist
        return variance * (1 + scaled + scaled ** 2 / 3) * np.exp(-scaled)
    else:
        raise ValueError(f"Unsupported nu: {nu}")


# ============================================================================
# 周期核 (Periodic Kernel)
# ============================================================================

class PeriodicKernel(Kernel):
    """
    k(x, y) = σ² * exp(-2 * sin²(π * ||x - y|| / p) / l²)
    """

    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, period: float = 1.0):
        super().__init__()
        self.log_length_scale = nn.Parameter(torch.log(torch.tensor(length_scale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))
        self.log_period = nn.Parameter(torch.log(torch.tensor(period)))

    @property
    def length_scale(self):
        return torch.exp(self.log_length_scale)

    @property
    def variance(self):
        return torch.exp(self.log_variance)

    @property
    def period(self):
        return torch.exp(self.log_period)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        dist = torch.cdist(x1, x2)
        sin_term = torch.sin(torch.pi * dist / self.period).pow(2)
        return self.variance * torch.exp(-2 * sin_term / self.length_scale ** 2)


# ============================================================================
# 核函数相加 (SumKernel)
# ============================================================================

class SumKernel(Kernel):
    """Sum of two kernels."""

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.k1(x1, x2) + self.k2(x1, x2)


# ============================================================================
# 核函数相乘 (ProductKernel)
# ============================================================================

class ProductKernel(Kernel):
    """Product of two kernels."""

    def __init__(self, k1: Kernel, k2: Kernel):
        super().__init__()
        self.k1 = k1
        self.k2 = k2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.k1(x1, x2) * self.k2(x1, x2)