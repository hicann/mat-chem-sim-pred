"""
PyPTO kernel: diffusion process operators for crystal structure prediction.

Port of DAO's diffusion.py and diff_utils.py:
  - SinusoidalTimeEmbeddings
  - BetaScheduler
  - SigmaScheduler
  - d_log_p_wrapped_normal
  - add_diffusion_noise
"""

import pypto as pt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbeddings  --  DAO diffusion.py
# ---------------------------------------------------------------------------

@pt.frontend.jit
def SinusoidalTimeEmbeddings(
    time: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    dim: pt.Element(pt.DT_INT32),
):
    """Sinusoidal time embedding for diffusion steps.

    time: [B] normalized timesteps
    out: [B, dim] embedding vector
    dim: embedding dimension (must be even)
    """
    half_dim = dim // 2
    steps = pt.arange(half_dim, pt.DT_FP32)
    freq = pt.exp(-pt.log(10000.0) / (half_dim - 1) * steps)
    angles = pt.unsqueeze(time, -1) * pt.unsqueeze(freq, 0)
    out[:] = pt.concat([pt.sin(angles), pt.cos(angles)], dim=-1)


# ---------------------------------------------------------------------------
# BetaScheduler  --  DAO diffusion.py
# ---------------------------------------------------------------------------

class BetaScheduler:
    """Pre-compute beta schedule and derived quantities for DDPM.

    Supports 'cosine' (default) and 'linear' schedules.
    """

    def __init__(self, timesteps=1000, beta_min=1e-4, beta_max=0.02,
                 schedule='cosine'):
        self.timesteps = timesteps
        if schedule == 'cosine':
            betas = self._cosine_schedule(timesteps)
        else:
            betas = self._linear_schedule(timesteps, beta_min, beta_max)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(self.alphas)
        self.alphas_cumprod_prev = np.concatenate(
            [np.ones(1, dtype=np.float32), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(
            1.0 / self.alphas_cumprod - 1
        )

    @staticmethod
    def _cosine_schedule(timesteps):
        """Cosine beta schedule (Nichol & Dhariwal 2021)."""
        s = 0.008
        t = np.linspace(0, timesteps, timesteps + 1, dtype=np.float64)
        f = np.cos((t / timesteps + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return np.clip(betas, 0, 0.999).astype(np.float32)

    @staticmethod
    def _linear_schedule(timesteps, beta_min, beta_max):
        """Linear beta schedule."""
        return np.linspace(beta_min, beta_max, timesteps, dtype=np.float32)

    def uniform_sample_t(self, batch_size, device='npu'):
        """Sample random timesteps uniformly."""
        return torch.randint(0, self.timesteps, (batch_size,), device=device)

    def get_params(self, t):
        """Return diffusion parameters at timestep(s) t.

        t: int or tensor of timesteps
        Returns dict of tensors on the same device as t (or CPU).
        """
        is_scalar = isinstance(t, int)
        if is_scalar:
            idx = np.array([t], dtype=np.int64)
            device = 'cpu'
        elif isinstance(t, torch.Tensor):
            device = t.device
            idx = t.cpu().numpy()
        else:
            idx = np.asarray(t)
            device = 'cpu'

        def _gather(arr):
            result = torch.from_numpy(arr[idx])
            if is_scalar:
                result = result.squeeze(0)
            return result.to(device)

        return {
            'betas': _gather(self.betas),
            'alphas': _gather(self.alphas),
            'alphas_cumprod': _gather(self.alphas_cumprod),
            'sqrt_alphas_cumprod': _gather(self.sqrt_alphas_cumprod),
            'sqrt_one_minus_alphas_cumprod': _gather(
                self.sqrt_one_minus_alphas_cumprod
            ),
            'sqrt_recip_alphas_cumprod': _gather(
                self.sqrt_recip_alphas_cumprod
            ),
            'sqrt_recipm1_alphas_cumprod': _gather(
                self.sqrt_recipm1_alphas_cumprod
            ),
        }


# ---------------------------------------------------------------------------
# SigmaScheduler  --  DAO diff_utils.py
# ---------------------------------------------------------------------------

class SigmaScheduler:
    """Geometric sigma schedule for score-based models (NCSN)."""

    def __init__(self, sigma_min=0.01, sigma_max=50.0, timesteps=1000):
        self.timesteps = timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        log_sigmas = np.linspace(
            np.log(sigma_min), np.log(sigma_max), timesteps
        )
        self.sigmas = np.exp(log_sigmas).astype(np.float32)
        self.sigmas_norm = np.cumsum(self.sigmas ** 2).astype(np.float32)

    def get_sigma(self, t):
        """Return sigma at timestep(s) t."""
        if isinstance(t, int):
            return torch.tensor(self.sigmas[t])
        if isinstance(t, torch.Tensor):
            device = t.device
            t = t.cpu().numpy()
        else:
            device = 'cpu'
            t = np.asarray(t)
        return torch.from_numpy(self.sigmas[t]).to(device)

    def get_sigma_norm(self, t):
        """Return cumulative sigma^2 norm at timestep(s) t."""
        if isinstance(t, int):
            return torch.tensor(self.sigmas_norm[t])
        if isinstance(t, torch.Tensor):
            device = t.device
            t = t.cpu().numpy()
        else:
            device = 'cpu'
            t = np.asarray(t)
        return torch.from_numpy(self.sigmas_norm[t]).to(device)


# ---------------------------------------------------------------------------
# d_log_p_wrapped_normal  --  DAO diff_utils.py
# ---------------------------------------------------------------------------

@pt.frontend.jit
def d_log_p_wrapped_normal(
    z: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    sigma: pt.Tensor([pt.DYNAMIC, 1], pt.DT_FP32),
    score_out: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
):
    """Simplified wrapped normal score: -z / sigma^2.

    z: [N, 3] noisy positions
    sigma: [N, 1] noise level
    score_out: [N, 3] score
    """
    score_out[:] = -z / (sigma ** 2)


# ---------------------------------------------------------------------------
# add_diffusion_noise  --  DDPM forward process
# ---------------------------------------------------------------------------

@pt.frontend.jit
def add_diffusion_noise(
    data: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    noise: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    alpha_cumprod: pt.Tensor([pt.DYNAMIC, 1], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
):
    """DDPM forward noising: sqrt(ac) * data + sqrt(1-ac) * noise.

    data: [N, 3] clean data
    noise: [N, 3] gaussian noise
    alpha_cumprod: [N, 1] cumulative alpha product
    out: [N, 3] noisy data
    """
    c0 = pt.sqrt(alpha_cumprod)
    c1 = pt.sqrt(1.0 - alpha_cumprod)
    out[:] = c0 * data + c1 * noise


# ---------------------------------------------------------------------------
# NumPy reference implementations  (for validation)
# ---------------------------------------------------------------------------

def sinusoidal_time_embeddings_np(time, dim):
    """NumPy reference for SinusoidalTimeEmbeddings."""
    half_dim = dim // 2
    freqs = np.exp(
        -np.log(10000.0) / (half_dim - 1)
        * np.arange(half_dim, dtype=np.float32)
    )
    angles = time[:, None] * freqs[None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)


def d_log_p_wrapped_normal_np(z, sigma):
    """NumPy reference for d_log_p_wrapped_normal."""
    return -z / (sigma ** 2)


def add_diffusion_noise_np(data, noise, alpha_cumprod):
    """NumPy reference for add_diffusion_noise."""
    c0 = np.sqrt(alpha_cumprod)
    c1 = np.sqrt(1.0 - alpha_cumprod)
    return c0 * data + c1 * noise


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sinusoidal_time_embeddings():
    np.random.seed(42)
    B, dim = 16, 128
    time = np.random.uniform(0, 1, B).astype(np.float32)
    expected = sinusoidal_time_embeddings_np(time, dim)

    t_t = torch.from_numpy(time).to('npu')
    out_t = torch.zeros(B, dim, dtype=torch.float32, device='npu')
    SinusoidalTimeEmbeddings(t_t, out_t, dim)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  SinusoidalTimeEmbeddings: {status}  (max_err={err:.2e})")


def test_beta_scheduler():
    scheduler = BetaScheduler(timesteps=1000, schedule='cosine')
    assert len(scheduler.betas) == 1000, "betas length mismatch"
    assert scheduler.betas[0] > 0, "first beta should be positive"
    print(f"  BetaScheduler (cosine): PASS  "
          f"beta[0]={scheduler.betas[0]:.6f}  "
          f"beta[-1]={scheduler.betas[-1]:.6f}")

    scheduler_lin = BetaScheduler(timesteps=1000, schedule='linear',
                                   beta_min=1e-4, beta_max=0.02)
    assert len(scheduler_lin.betas) == 1000
    print(f"  BetaScheduler (linear): PASS  "
          f"beta[0]={scheduler_lin.betas[0]:.6f}  "
          f"beta[-1]={scheduler_lin.betas[-1]:.6f}")

    t = scheduler.uniform_sample_t(8, 'cpu')
    params = scheduler.get_params(t)
    for k, v in params.items():
        assert v.shape == (8,), f"{k} shape mismatch: {v.shape}"
    print(f"  BetaScheduler.get_params: PASS  (batch=8)")

    scalar_params = scheduler.get_params(0)
    assert scalar_params['betas'].ndim == 0
    print(f"  BetaScheduler.get_params (scalar): PASS")


def test_sigma_scheduler():
    scheduler = SigmaScheduler(sigma_min=0.01, sigma_max=50.0, timesteps=1000)
    assert len(scheduler.sigmas) == 1000
    assert len(scheduler.sigmas_norm) == 1000
    assert scheduler.sigmas[0] < scheduler.sigmas[-1]
    print(f"  SigmaScheduler: PASS  "
          f"sigma[0]={scheduler.sigmas[0]:.4f}  "
          f"sigma[-1]={scheduler.sigmas[-1]:.2f}")


def test_d_log_p_wrapped_normal():
    np.random.seed(42)
    N = 32
    z = np.random.randn(N, 3).astype(np.float32)
    sigma = np.random.uniform(0.1, 2.0, (N, 1)).astype(np.float32)
    expected = d_log_p_wrapped_normal_np(z, sigma)

    z_t = torch.from_numpy(z).to('npu')
    s_t = torch.from_numpy(sigma).to('npu')
    out_t = torch.zeros(N, 3, dtype=torch.float32, device='npu')
    d_log_p_wrapped_normal(z_t, s_t, out_t)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  d_log_p_wrapped_normal: {status}  (max_err={err:.2e})")


def test_add_diffusion_noise():
    np.random.seed(42)
    N = 32
    data = np.random.randn(N, 3).astype(np.float32)
    noise = np.random.randn(N, 3).astype(np.float32)
    alpha_cumprod = np.random.uniform(0.0, 1.0, (N, 1)).astype(np.float32)
    expected = add_diffusion_noise_np(data, noise, alpha_cumprod)

    d_t = torch.from_numpy(data).to('npu')
    n_t = torch.from_numpy(noise).to('npu')
    a_t = torch.from_numpy(alpha_cumprod).to('npu')
    out_t = torch.zeros(N, 3, dtype=torch.float32, device='npu')
    add_diffusion_noise(d_t, n_t, a_t, out_t)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  add_diffusion_noise: {status}  (max_err={err:.2e})")


def main():
    print("=" * 50)
    print("PyPTO Diffusion Ops -- Tests")
    print("=" * 50)
    test_sinusoidal_time_embeddings()
    test_beta_scheduler()
    test_sigma_scheduler()
    test_d_log_p_wrapped_normal()
    test_add_diffusion_noise()
    print("=" * 50)


if __name__ == "__main__":
    main()
