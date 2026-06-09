"""
TabNet: Attentive Interpretable Tabular Learning — PyTorch Reference Implementation.

Port of https://github.com/google-research/google-research/tree/master/tabnet
Designed for CANN Ascend NPU migration: each core component (Sparsemax, GLU block,
AttentiveTransformer) is isolated so it can be replaced by an Ascend C kernel later.

Reference: Arik & Pfister, AAAI 2021 (https://arxiv.org/abs/1908.07442)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Sparsemax — 核心激活函数，替代 Softmax，输出稀疏的 attention mask
# ============================================================================

def sparsemax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """NumPy reference: Sparsemax activation."""
    z = z.copy()
    z_sorted = np.sort(z, axis=axis)[:, ::-1]
    z_cumsum = np.cumsum(z_sorted, axis=axis)
    k = np.arange(1, z.shape[axis] + 1, dtype=np.float32)
    z_thresh = (z_cumsum - 1.0) / k
    k_ones = (z_sorted > z_thresh).astype(np.float32)
    k_select = np.argmax(k_ones[:, ::-1], axis=axis)
    tau = np.take_along_axis(z_thresh, k_select[:, None], axis=axis)
    return np.maximum(z - tau, 0.0)


class Sparsemax(nn.Module):
    """Sparsemax: differentiable sparse activation for feature selection."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=self.dim, keepdim=True)
        z_sorted, _ = z.sort(dim=self.dim, descending=True)
        z_cumsum = z_sorted.cumsum(dim=self.dim)
        k = torch.arange(1, z.shape[self.dim] + 1, dtype=z.dtype, device=z.device)
        z_thresh = (z_cumsum - 1.0) / k
        k_ones = (z_sorted > z_thresh).float()
        k_select = k_ones.shape[self.dim] - k_ones.flip(dims=[self.dim]).argmax(dim=self.dim) - 1
        k_select = k_select.clamp(0, z.shape[self.dim] - 1)
        tau = z_thresh.gather(self.dim, k_select.unsqueeze(self.dim)).squeeze(self.dim)
        return torch.clamp(z - tau.unsqueeze(self.dim), min=0.0)


# ============================================================================
# GLU Block — 门控线性单元，Feature Transformer 的基础构建块
# ============================================================================

class GLUBlock(nn.Module):
    """Gated Linear Unit block with batch normalization and skip connection."""

    def __init__(self, dim: int, hidden_dim: int, virtual_batch_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(dim, hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.01)
        self.fc3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(x))
        out = gate * out
        out = self.fc3(out)
        return out + x  # skip connection


class FeatureTransformer(nn.Module):
    """Stack of GLU blocks with shared/independent parameter splitting."""

    def __init__(self, dim: int, n_blocks: int = 2, shared_block: bool = False,
                 virtual_batch_size: int = 128):
        super().__init__()
        self.blocks = nn.ModuleList([
            GLUBlock(dim, dim * 2, virtual_batch_size) for _ in range(n_blocks)
        ])
        self.final_bn = nn.BatchNorm1d(dim, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_bn(x)


# ============================================================================
# Attentive Transformer — 基于 Sparsemax 的实例级特征选择
# ============================================================================

class AttentiveTransformer(nn.Module):
    """Produces sparse feature selection mask for each decision step."""

    def __init__(self, dim: int, n_features: int):
        super().__init__()
        self.fc = nn.Linear(dim, n_features, bias=False)
        self.bn = nn.BatchNorm1d(n_features, momentum=0.01)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, x: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.fc(x))
        x = x * prior
        return self.sparsemax(x)


# ============================================================================
# TabNet Encoder
# ============================================================================

class TabNetEncoder(nn.Module):
    """
    TabNet encoder with sequential multi-step attention.

    Args:
        input_dim: Total number of features (continuous + categorical).
        dim: Hidden dimension for feature transformer.
        output_dim: Output dimension (num_classes for classification, 1 for regression).
        n_steps: Number of sequential attention steps.
        n_shared: Number of shared GLU blocks in feature transformer.
        n_independent: Number of independent GLU blocks per step.
        relaxation_factor: Relaxation factor for prior scale update.
        sparsity_coefficient: Entropy regularization for sparsemax masks.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int = 64,
        output_dim: int = 1,
        n_steps: int = 3,
        n_shared: int = 2,
        n_independent: int = 2,
        relaxation_factor: float = 1.5,
        sparsity_coefficient: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient

        self.initial_bn = nn.BatchNorm1d(input_dim, momentum=0.01)

        self.shared_feature_transformer = FeatureTransformer(dim, n_shared, shared_block=True)
        self.step_feature_transformers = nn.ModuleList([
            FeatureTransformer(dim, n_independent) for _ in range(n_steps)
        ])
        self.attentive_transformers = nn.ModuleList([
            AttentiveTransformer(dim, input_dim) for _ in range(n_steps)
        ])

        self.final_fc = nn.Linear(dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (batch, input_dim) — input features.

        Returns:
            output: (batch, output_dim) — final prediction.
            masks: list of (batch, input_dim) — attention masks per step.
            aggregated: (batch, dim) — aggregated feature representations.
        """
        x = self.initial_bn(x)
        batch_size = x.shape[0]

        prior = torch.ones(batch_size, self.input_dim, device=x.device)
        attended_features = torch.zeros(batch_size, self.dim, device=x.device)
        masks = []

        for step in range(self.n_steps):
            mask = self.attentive_transformers[step](attended_features, prior)
            masks.append(mask)

            masked_x = x * mask

            step_out = self.shared_feature_transformer(masked_x)
            step_out = self.step_feature_transformers[step](step_out)

            attended_features = attended_features + step_out

            prior = prior * (self.relaxation_factor - mask)
            prior = torch.clamp(prior, min=0.0, max=1.0)

        output = self.final_fc(attended_features)

        if self.sparsity_coefficient > 0:
            sparsity_loss = self._sparsity_regularization(masks)
            return output, masks, sparsity_loss

        return output, masks, attended_features

    def _sparsity_regularization(self, masks: list) -> torch.Tensor:
        """Entropy-based sparsity regularization."""
        loss = 0.0
        for mask in masks:
            eps = 1e-10
            loss = loss + (-mask * torch.log(mask + eps) - (1 - mask) * torch.log(1 - mask + eps)).mean()
        return loss / len(masks)


class TabNetClassifier(nn.Module):
    """TabNet for classification."""

    def __init__(self, input_dim: int, num_classes: int, dim: int = 64,
                 n_steps: int = 3, **kwargs):
        super().__init__()
        self.encoder = TabNetEncoder(input_dim, dim, num_classes, n_steps, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple:
        logits, masks, aux = self.encoder(x)
        return F.softmax(logits, dim=-1), masks


class TabNetRegressor(nn.Module):
    """TabNet for regression."""

    def __init__(self, input_dim: int, dim: int = 64, n_steps: int = 3, **kwargs):
        super().__init__()
        self.encoder = TabNetEncoder(input_dim, dim, 1, n_steps, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple:
        pred, masks, aux = self.encoder(x)
        return pred.squeeze(-1), masks


# ============================================================================
# NumPy reference implementations (for cross-validation)
# ============================================================================

def tabnet_encoder_np(
    x: np.ndarray,
    W_init: dict,
    n_steps: int = 3,
    dim: int = 64,
) -> np.ndarray:
    """NumPy reference for TabNet encoder forward pass."""
    batch, input_dim = x.shape
    prior = np.ones((batch, input_dim), dtype=np.float32)
    attended = np.zeros((batch, dim), dtype=np.float32)

    for step in range(n_steps):
        score = x @ W_init[f"W_attn_{step}"]
        score = np.maximum(score, 0)
        mask = sparsemax_np(score)
        masked_x = x * mask
        h = masked_x @ W_init[f"W_feat_{step}"]
        attended = attended + h
        prior = prior * (1.5 - mask)
        prior = np.clip(prior, 0.0, 1.0)

    return attended @ W_init["W_final"]