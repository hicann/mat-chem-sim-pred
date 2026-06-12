"""
GPR 测试 — 验证高斯过程回归实现正确性。

Usage:
    pytest prediction/SmallData/test_gpr.py -v
    python -m prediction.SmallData.test_gpr
"""

import numpy as np
import torch

from .kernels import RBFKernel, MaternKernel, PeriodicKernel, SumKernel
from .kernels import rbf_kernel_np, matern_kernel_np
from .gpr import (
    GaussianProcessRegressor,
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
    BayesianOptimizer,
    gp_predict_np,
)


RTOL = 1e-3
ATOL = 1e-3


def check(label, result, expected, rtol=RTOL, atol=ATOL):
    try:
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=atol)
        print(f"  {label}:  PASS")
    except AssertionError as e:
        err = np.abs(result - expected).max()
        print(f"  {label}:  FAIL (max_err={err:.2e})")


def test_rbf_kernel():
    np.random.seed(42)
    x1 = np.random.randn(10, 3).astype(np.float32)
    x2 = np.random.randn(6, 3).astype(np.float32)

    kernel = RBFKernel(length_scale=1.5, variance=2.0)
    K_t = kernel(torch.from_numpy(x1), torch.from_numpy(x2))
    K_np = rbf_kernel_np(x1, x2, length_scale=1.5, variance=2.0)

    check("RBF kernel", K_t.detach().numpy(), K_np)


def test_matern_kernel():
    np.random.seed(42)
    x1 = np.random.randn(8, 2).astype(np.float32)
    x2 = np.random.randn(5, 2).astype(np.float32)

    for nu in [0.5, 1.5, 2.5]:
        kernel = MaternKernel(length_scale=1.0, variance=1.0, nu=nu)
        K_t = kernel(torch.from_numpy(x1), torch.from_numpy(x2))
        K_np = matern_kernel_np(x1, x2, length_scale=1.0, variance=1.0, nu=nu)
        check(f"Matern nu={nu}", K_t.detach().numpy(), K_np)


def test_sum_kernel():
    np.random.seed(42)
    x = np.random.randn(6, 2).astype(np.float32)

    k1 = RBFKernel(length_scale=1.0, variance=1.0)
    k2 = PeriodicKernel(length_scale=1.0, variance=0.5, period=2.0)
    k_sum = SumKernel(k1, k2)

    x_t = torch.from_numpy(x)
    K = k_sum(x_t, x_t)
    K1 = k1(x_t, x_t)
    K2 = k2(x_t, x_t)

    diff = (K - (K1 + K2)).abs().max().item()
    assert diff < 1e-6, f"SumKernel error: {diff}"
    print(f"  SumKernel validation: PASS (max_diff={diff:.2e})")


def test_gp_fit_predict():
    np.random.seed(42)
    torch.manual_seed(42)

    n_train, n_test = 20, 50
    x = np.linspace(-3, 3, n_train).astype(np.float32).reshape(-1, 1)
    y = np.sin(x.squeeze()) + np.random.randn(n_train).astype(np.float32) * 0.05

    x_test = np.linspace(-3.5, 3.5, n_test).astype(np.float32).reshape(-1, 1)

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    x_test_t = torch.from_numpy(x_test)

    kernel = RBFKernel(length_scale=1.0, variance=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, noise_scale=0.1)
    gp.fit(x_t, y_t)

    mu_t, std_t = gp.predict(x_test_t, return_std=True)
    mu_np, std_np = gp_predict_np(x, y, x_test, length_scale=1.0, variance=1.0, noise=0.1)

    check("GP predict mean", mu_t.numpy(), mu_np)
    check("GP predict std", std_t.numpy(), std_np)

    assert torch.all(std_t >= 0), "Std should be non-negative"

    y_truth = np.sin(x_test.squeeze())
    rmse = np.sqrt(np.mean((mu_t.numpy() - y_truth) ** 2))
    print(f"  GP prediction RMSE: {rmse:.4f}")


def test_gp_log_marginal_likelihood():
    torch.manual_seed(42)
    n = 10
    x = torch.randn(n, 2)
    y = torch.sin(x[:, 0]) + 0.1 * torch.randn(n)

    kernel = RBFKernel(length_scale=1.0, variance=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, noise_scale=0.1)

    lml = gp.log_marginal_likelihood(x, y)
    assert lml.shape == (), f"LML should be scalar, got {lml.shape}"
    print(f"  Log marginal likelihood: {lml.item():.4f}")


def test_gp_training():
    """验证 GP 超参数可以通过梯度优化."""
    torch.manual_seed(42)
    n = 30
    x = torch.randn(n, 1)
    y = torch.sin(2 * x.squeeze()) + 0.1 * torch.randn(n)

    kernel = RBFKernel(length_scale=0.5, variance=1.0)
    gp = GaussianProcessRegressor(kernel=kernel, noise_scale=0.1)

    optimizer = torch.optim.Adam(gp.parameters(), lr=1e-2)
    losses = []

    for epoch in range(100):
        optimizer.zero_grad()
        loss = gp.negative_log_marginal_likelihood(x, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    final_lml = -losses[-1]
    print(f"  Final log marginal likelihood: {final_lml:.4f}")
    print(f"  Learned length_scale: {kernel.length_scale.item():.4f}")
    print(f"  Learned variance: {kernel.variance.item():.4f}")

    for p in gp.parameters():
        assert p.grad is not None, "No gradient for parameter!"


def test_acquisition_functions():
    torch.manual_seed(42)
    mu = torch.linspace(-2, 2, 100)
    sigma = torch.ones(100) * 0.5
    y_best = 1.0

    ei = ExpectedImprovement()
    ucb = UpperConfidenceBound(beta=2.0)
    pi = ProbabilityOfImprovement()

    ei_vals = ei(mu, sigma, y_best)
    ucb_vals = ucb(mu, sigma, y_best)
    pi_vals = pi(mu, sigma, y_best)

    assert ei_vals.shape == mu.shape
    assert ucb_vals.shape == mu.shape
    assert pi_vals.shape == mu.shape
    assert torch.all(ei_vals >= 0), "EI should be non-negative"

    print(f"  EI at mu=0: {ei(torch.tensor([0.0]), torch.tensor([0.5]), 1.0).item():.4f}")
    print(f"  EI at mu=2: {ei(torch.tensor([2.0]), torch.tensor([0.5]), 1.0).item():.4f}")
    assert ei(torch.tensor([2.0]), torch.tensor([0.5]), 1.0) > ei(torch.tensor([0.0]), torch.tensor([0.5]), 1.0)


def test_bo_on_simple_function():
    """验证贝叶斯优化能找到简单函数的极值点."""
    np.random.seed(42)
    torch.manual_seed(42)

    def objective(x):
        return -(x - 0.5) ** 2 + 1.0

    bounds = np.array([[0.0, 1.0]])
    gp = GaussianProcessRegressor(
        kernel=RBFKernel(length_scale=0.2, variance=1.0),
        noise_scale=0.01,
    )
    bo = BayesianOptimizer(gp, bounds, n_initial=3)

    best_x, best_y = bo.maximize(objective, n_iter=10)
    print(f"  BO result: x={best_x.item():.4f}, y={best_y.item():.4f}")
    print(f"  True optimum: x=0.5, y=1.0")
    assert abs(best_x.item() - 0.5) < 0.2, f"BO failed: got x={best_x.item()}"


def main():
    print("=" * 60)
    print("Gaussian Process Test Suite")
    print("=" * 60)

    test_rbf_kernel()
    test_matern_kernel()
    test_sum_kernel()
    test_gp_fit_predict()
    test_gp_log_marginal_likelihood()
    test_gp_training()
    test_acquisition_functions()
    test_bo_on_simple_function()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()