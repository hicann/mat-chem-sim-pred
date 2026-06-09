"""
Gaussian Process Regression (GPR) — PyTorch Reference Implementation.

Core components:
- GP 回归预测 (均值 + 方差)
- 对数边际似然优化
- 贝叶斯优化采集函数 (EI, UCB, PI)

参考实现可后续迁移到 Ascend C 的关键计算模式：
  · 核函数矩阵构建 (数据并行友好)
  · Cholesky 分解 (三角求解)
  · 采集函数计算 (逐点并行)
"""

import torch
import torch.nn as nn
import numpy as np

from .kernels import Kernel, RBFKernel


# ============================================================================
# 高斯过程回归
# ============================================================================

class GaussianProcessRegressor(nn.Module):
    """
    Exact Gaussian Process Regression.

    Args:
        kernel: Kernel instance.
        noise_scale: Initial noise (likelihood) standard deviation.
        optimize_noise: Whether to optimize the noise parameter.
    """

    def __init__(self, kernel: Kernel = None, noise_scale: float = 0.1,
                 optimize_noise: bool = True):
        super().__init__()
        self.kernel = kernel or RBFKernel()
        self.log_noise = nn.Parameter(
            torch.log(torch.tensor(noise_scale)),
            requires_grad=optimize_noise,
        )

    @property
    def noise(self):
        return torch.exp(self.log_noise)

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> "GaussianProcessRegressor":
        """
        Fit GP model. Computes and caches the Cholesky decomposition.

        Args:
            x: (n, d) — training inputs
            y: (n,) — training targets

        Returns:
            self
        """
        self.train_x = x
        self.train_y = y

        n = x.shape[0]
        K = self.kernel(x, x)
        K = K + self.noise ** 2 * torch.eye(n, device=x.device, dtype=x.dtype)

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L)

        self._L = L
        self._alpha = alpha.squeeze(-1)

        return self

    def predict(self, x: torch.Tensor, return_std: bool = False) -> tuple:
        """
        Make predictions.

        Args:
            x: (m, d) — test inputs
            return_std: whether to return predictive standard deviation

        Returns:
            mu: (m,) — predictive mean
            std: (m,) — predictive std (if return_std=True)
        """
        if not hasattr(self, 'train_x') or self.train_x is None:
            raise RuntimeError("GP model has not been fitted. Call fit() before predict().")
        K_s = self.kernel(x, self.train_x)
        mu = K_s @ self._alpha

        if not return_std:
            return mu, None

        K_ss = self.kernel(x, x)
        v = torch.linalg.solve(self._L, K_s.T)
        var = K_ss.diag() - (v ** 2).sum(0)
        var = var.clamp(min=0.0)
        return mu, var.sqrt()

    def log_marginal_likelihood(self, x: torch.Tensor = None,
                                 y: torch.Tensor = None) -> torch.Tensor:
        """
        Compute log marginal likelihood.

        Args:
            x: (n, d) — training inputs (use cached if None)
            y: (n,) — training targets (use cached if None)

        Returns:
            lml: scalar — log marginal likelihood
        """
        if x is None:
            if not hasattr(self, 'train_x') or self.train_x is None:
                raise RuntimeError(
                    "GP model has not been fitted. Call fit() first, "
                    "or pass (x, y) explicitly to log_marginal_likelihood()."
                )
            x, y = self.train_x, self.train_y

        n = x.shape[0]
        K = self.kernel(x, x)
        K = K + self.noise ** 2 * torch.eye(n, device=x.device, dtype=x.dtype)

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)

        lml = -0.5 * (y @ alpha)
        lml = lml - L.diag().log().sum()
        lml = lml - 0.5 * n * torch.log(2 * torch.tensor(torch.pi, device=x.device))

        return lml

    def negative_log_marginal_likelihood(self, x: torch.Tensor = None,
                                          y: torch.Tensor = None) -> torch.Tensor:
        """Convenience: negative LML for minimization."""
        return -self.log_marginal_likelihood(x, y)


# ============================================================================
# 贝叶斯优化采集函数
# ============================================================================

class AcquisitionFunction:
    """Base class for Bayesian optimization acquisition functions."""

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 y_best: float) -> torch.Tensor:
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """
    Expected Improvement (EI) 采集函数.

    EI(x) = E[max(f(x) - f*, 0)]
           = (μ - f*) * Φ(γ) + σ * φ(γ)
    where γ = (μ - f*) / σ
    """

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 y_best: float) -> torch.Tensor:
        sigma = sigma.clamp(min=1e-10)
        gamma = (mu - y_best) / sigma
        sqrt2 = torch.tensor(1.4142135623730951, device=mu.device, dtype=mu.dtype)
        sqrt2pi = torch.tensor(2.5066282746310002, device=mu.device, dtype=mu.dtype)
        phi = torch.exp(-0.5 * gamma ** 2) / sqrt2pi
        Phi = 0.5 * (1 + torch.erf(gamma / sqrt2))
        ei = (mu - y_best) * Phi + sigma * phi
        return ei.clamp(min=0.0)


class UpperConfidenceBound(AcquisitionFunction):
    """
    Upper Confidence Bound (UCB).

    UCB(x) = μ + β * σ
    """

    def __init__(self, beta: float = 2.0):
        self.beta = beta

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 y_best: float) -> torch.Tensor:
        return mu + self.beta * sigma


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    Probability of Improvement (PI).

    PI(x) = Φ((μ - f*) / σ)
    """

    def __call__(self, mu: torch.Tensor, sigma: torch.Tensor,
                 y_best: float) -> torch.Tensor:
        sigma = sigma.clamp(min=1e-10)
        gamma = (mu - y_best) / sigma
        sqrt2 = torch.tensor(1.4142135623730951, device=mu.device, dtype=mu.dtype)
        return 0.5 * (1 + torch.erf(gamma / sqrt2))


# ============================================================================
# 贝叶斯优化器
# ============================================================================

class BayesianOptimizer:
    """
    基于 GP 的贝叶斯优化器。

    Args:
        gp: GaussianProcessRegressor instance.
        bounds: (n_dims, 2) — search bounds [lower, upper] for each dimension.
        acquisition: Acquisition function.
        n_initial: Number of initial random points.
    """

    def __init__(
        self,
        gp: GaussianProcessRegressor,
        bounds: np.ndarray,
        acquisition: AcquisitionFunction = None,
        n_initial: int = 5,
    ):
        self.gp = gp
        self.bounds = bounds
        self.acquisition = acquisition or ExpectedImprovement()
        self.n_initial = n_initial
        self.x_observed = []
        self.y_observed = []

    def suggest(self, n_candidates: int = 1000) -> np.ndarray:
        """
        Suggest next point to evaluate.

        Args:
            n_candidates: number of random candidates to evaluate acquisition on.

        Returns:
            x_next: (d,) — next point
        """
        x_train = torch.stack(self.x_observed) if self.x_observed else None
        y_train = torch.stack(self.y_observed) if self.y_observed else None

        if x_train is None:
            x_cand = torch.rand(n_candidates, self.bounds.shape[0])
            x_cand = x_cand * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
            return x_cand[y_train.argmax() if y_train is not None else 0]

        self.gp.fit(x_train, y_train)
        y_best = y_train.max().item()

        x_cand = torch.rand(n_candidates, self.bounds.shape[0])
        x_cand = x_cand * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]

        mu, sigma = self.gp.predict(x_cand, return_std=True)
        acq = self.acquisition(mu, sigma, y_best)
        return x_cand[acq.argmax()]

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Add observed point."""
        self.x_observed.append(x)
        self.y_observed.append(y)

    def maximize(self, objective, n_iter: int = 20):
        """
        Run Bayesian optimization.

        Args:
            objective: callable that takes (d,) tensor and returns scalar.
            n_iter: number of BO iterations.

        Returns:
            best_x: best found parameters.
            best_y: best found value.
        """
        for i in range(self.n_initial):
            x = torch.rand(self.bounds.shape[0])
            x = x * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
            y = objective(x.unsqueeze(0))
            self.update(x, y[0])

        for i in range(n_iter):
            x_next = self.suggest()
            y_next = objective(x_next.unsqueeze(0))
            self.update(x_next, y_next[0])

        best_idx = torch.argmax(torch.stack(self.y_observed))
        return self.x_observed[best_idx], self.y_observed[best_idx]


# ============================================================================
# NumPy reference implementations
# ============================================================================

def gp_predict_np(x_train: np.ndarray, y_train: np.ndarray,
                  x_test: np.ndarray, length_scale: float = 1.0,
                  variance: float = 1.0, noise: float = 0.1) -> tuple:
    """NumPy reference for GP prediction."""
    n = x_train.shape[0]

    K = variance * np.exp(-0.5 * np.sum((x_train[:, None] - x_train[None]) ** 2, axis=-1)
                          / length_scale ** 2)
    K += noise ** 2 * np.eye(n)

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    K_s = variance * np.exp(-0.5 * np.sum((x_test[:, None] - x_train[None]) ** 2, axis=-1)
                            / length_scale ** 2)
    mu = K_s @ alpha

    K_ss = variance * np.ones(x_test.shape[0])
    v = np.linalg.solve(L, K_s.T)
    var = K_ss - np.sum(v ** 2, axis=0)

    return mu, np.sqrt(np.clip(var, 0, None))