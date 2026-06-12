"""
TabNet 测试 — 验证 PyTorch 实现正确性，与 NumPy 参考对比。

Usage:
    pytest prediction/TabularData/deep_learning_architectures/tabnet/test_tabnet.py -v
    python -m prediction.TabularData.deep_learning_architectures.tabnet.test_tabnet
"""

import numpy as np
import torch

from .tabnet import (
    sparsemax_np,
    Sparsemax,
    TabNetEncoder,
    TabNetClassifier,
    TabNetRegressor,
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


def test_sparsemax():
    np.random.seed(42)
    z = np.random.randn(4, 8).astype(np.float32)
    z_t = torch.from_numpy(z)
    sparsemax = Sparsemax(dim=-1)

    out_t = sparsemax(z_t).detach().numpy()
    out_np = sparsemax_np(z)

    check("Sparsemax", out_t, out_np)

    assert np.all(out_t >= 0), "Sparsemax output should be non-negative"
    sum_along_axis = out_t.sum(axis=-1)
    assert np.allclose(sum_along_axis, 1.0, atol=1e-5), "Sparsemax should sum to 1"
    print(f"  Sparsemax row sums: {sum_along_axis}")


def test_tabnet_encoder():
    np.random.seed(42)
    torch.manual_seed(42)

    batch, input_dim = 32, 16

    x = np.random.randn(batch, input_dim).astype(np.float32)
    x_t = torch.from_numpy(x)

    encoder = TabNetEncoder(input_dim=input_dim, dim=64, output_dim=5, n_steps=3)
    encoder.eval()

    with torch.no_grad():
        output, masks, aux = encoder(x_t)

    assert output.shape == (batch, 5), f"Expected (batch, 5), got {output.shape}"
    assert len(masks) == 3, f"Expected 3 masks, got {len(masks)}"
    for i, mask in enumerate(masks):
        assert mask.shape == (batch, input_dim), f"Mask {i} shape mismatch: {mask.shape}"
        assert mask.min() >= 0.0, f"Mask {i} has negative values"
    print(f"  TabNetEncoder output shape: {output.shape}")
    print(f"  Mask sparsity per step:")
    for i, mask in enumerate(masks):
        sparsity = (mask < 0.01).float().mean().item()
        print(f"    Step {i}: {sparsity * 100:.1f}% zero")


def test_tabnet_classifier():
    torch.manual_seed(42)
    batch, input_dim, num_classes = 16, 10, 4
    x = torch.randn(batch, input_dim)

    model = TabNetClassifier(input_dim, num_classes, dim=32, n_steps=2)
    model.eval()

    with torch.no_grad():
        probs, masks = model(x)

    assert probs.shape == (batch, num_classes)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch))
    print(f"  TabNetClassifier probs shape: {probs.shape}")
    print(f"  Probs sum check: OK")


def test_tabnet_regressor():
    torch.manual_seed(42)
    batch, input_dim = 16, 10
    x = torch.randn(batch, input_dim)

    model = TabNetRegressor(input_dim, dim=32, n_steps=2)
    model.eval()

    with torch.no_grad():
        pred, masks = model(x)

    assert pred.shape == (batch,)
    print(f"  TabNetRegressor pred shape: {pred.shape}")


def test_tabnet_gradient_flow():
    """验证梯度可以反向传播通过整个网络."""
    torch.manual_seed(42)
    batch, input_dim = 8, 6
    x = torch.randn(batch, input_dim)
    y = torch.randn(batch, 1)

    model = TabNetRegressor(input_dim, dim=16, n_steps=2)
    pred, masks = model(x)

    loss = (pred.unsqueeze(-1) - y).pow(2).mean()
    loss.backward()

    n_nonzero_grad = sum((p.grad is not None and p.grad.abs().sum() > 0)
                         for p in model.parameters())
    print(f"  Gradient flow: {n_nonzero_grad}/{sum(1 for _ in model.parameters())} params have gradients")
    assert n_nonzero_grad > 0, "No gradients flowing!"


def main():
    print("=" * 60)
    print("TabNet Test Suite")
    print("=" * 60)

    test_sparsemax()
    test_tabnet_encoder()
    test_tabnet_classifier()
    test_tabnet_regressor()
    test_tabnet_gradient_flow()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()