"""
PyPTO kernel: basis function operators for crystal structure prediction.

Port of DAO's utils.py (SinusoidsEmbedding) and gnn.py (DimeNet++ basis layers):
  - SinusoidsEmbedding
  - BesselBasisLayer
  - SphericalBasisLayer
  - Legendre polynomial helper
"""

import pypto as pt
import numpy as np


# ---------------------------------------------------------------------------
# SinusoidsEmbedding  --  DAO utils.py
# ---------------------------------------------------------------------------

def SinusoidsEmbedding(
    x: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    n_frequencies: pt.Element(pt.DT_INT32),
):
    """Coordinate embedding via sinusoidal frequencies.

    x: [N, 3] coordinates
    out: [N, n_frequencies * 6]
    """
    N = x.shape[0]
    dim = n_frequencies * 6
    freqs = pt.arange(n_frequencies, pt.DT_FP32) * 6.283185307179586

    x_u = pt.unsqueeze(x, -1)
    f_u = pt.unsqueeze(pt.unsqueeze(freqs, 0), 0)
    x_proj = x_u * f_u

    sin_part = pt.sin(x_proj)
    cos_part = pt.cos(x_proj)

    stacked = pt.stack([sin_part, cos_part], dim=-1)
    out[:] = pt.reshape(stacked, [N, dim])


# ---------------------------------------------------------------------------
# BesselBasisLayer  --  PyG's BesselBasisLayer
# ---------------------------------------------------------------------------

@pt.frontend.jit
def BesselBasisLayer(
    distances: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    num_radial: pt.Element(pt.DT_INT32),
    cutoff: pt.Element(pt.DT_FP32),
    envelope_exponent: pt.Element(pt.DT_INT32),
):
    """Bessel radial basis with polynomial envelope.

    distances: [E]
    out: [E, num_radial]
    """
    E = distances.shape[0]
    pi = 3.141592653589793

    n_range = pt.arange(num_radial, pt.DT_FP32) + 1.0
    freq = n_range * (pi / cutoff)

    d = pt.unsqueeze(distances, -1)
    arg = d * freq
    sin_val = pt.sin(arg)
    sqrt_factor = pt.sqrt(pt.ones_like(d[..., 0:1]) * 2.0 / cutoff)

    safe_r = pt.where(distances == 0.0, pt.ones_like(distances), distances)
    safe_r = pt.unsqueeze(safe_r, -1)
    bessel = sqrt_factor * sin_val / safe_r

    p = pt.cast(envelope_exponent, pt.DT_FP32)
    r_c = d / cutoff
    env = 1.0 - (p + 1.0) * (r_c ** p) + p * (r_c ** (p + 1.0))
    env = pt.where(r_c < 1.0, env, pt.zeros_like(env))

    out[:] = bessel * env


# ---------------------------------------------------------------------------
# Legendre polynomial helper  --  P_l(x) via recurrence
# ---------------------------------------------------------------------------

@pt.frontend.jit
def legende_poly(
    x: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    l_max: pt.Element(pt.DT_INT32),
):
    """Legendre polynomials P_l(x) for l=0..l_max via recurrence.

    P_0 = 1,  P_1 = x,
    (l+1) P_{l+1} = (2l+1) x P_l - l P_{l-1}

    x: [N] evaluation points in [-1, 1]
    out: [N, l_max+1]
    """
    N = x.shape[0]
    out[:, 0] = pt.ones(N, pt.DT_FP32)
    if l_max >= 1:
        out[:, 1] = x
    for l in pt.loop(range(1, l_max)):
        l_f = pt.cast(l, pt.DT_FP32)
        p_l = out[:, l]
        p_lm1 = out[:, l - 1]
        p_lp1 = ((2.0 * l_f + 1.0) * x * p_l - l_f * p_lm1) / (l_f + 1.0)
        out[:, l + 1] = p_lp1


# ---------------------------------------------------------------------------
# SphericalBasisLayer  --  PyG's SphericalBasisLayer (simplified)
# Uses Legendre P_l(cos theta) as spherical harmonics (m=0)
# combined with Bessel radial basis via outer product.
# ---------------------------------------------------------------------------

def SphericalBasisLayer(
    distances: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    angles: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
    out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    num_spherical: pt.Element(pt.DT_INT32),
    num_radial: pt.Element(pt.DT_INT32),
    cutoff: pt.Element(pt.DT_FP32),
    envelope_exponent: pt.Element(pt.DT_INT32),
):
    """Simplified spherical basis: Legendre (m=0) x Bessel radial.

    distances: [T] for each triplet
    angles: [T] in radians
    out: [T, num_spherical * num_radial]
    """
    T = distances.shape[0]
    l_max = num_spherical - 1

    radial = pt.zeros([T, num_radial], pt.DT_FP32)
    BesselBasisLayer(distances, radial, num_radial, cutoff, envelope_exponent)

    cos_theta = pt.cos(angles)
    legendre = pt.zeros([T, l_max + 1], pt.DT_FP32)
    legende_poly(cos_theta, legendre, l_max)

    legendre_3d = pt.reshape(legendre, [T, l_max + 1, 1])
    radial_3d = pt.reshape(radial, [T, 1, num_radial])
    combined = legendre_3d * radial_3d

    out[:] = pt.reshape(combined, [T, (l_max + 1) * num_radial])


# ---------------------------------------------------------------------------
# NumPy reference implementations  (for validation)
# ---------------------------------------------------------------------------

def sinusoids_embedding_np(x, n_frequencies):
    N = x.shape[0]
    freqs = 2 * np.pi * np.arange(n_frequencies, dtype=np.float32)
    x_proj = x[:, :, None] * freqs[None, None, :]
    stacked = np.stack([np.sin(x_proj), np.cos(x_proj)], axis=-1)
    return stacked.reshape(N, -1)


def bessel_basis_np(distances, num_radial, cutoff, envelope_exponent=5):
    E = len(distances)
    n_range = np.arange(1, num_radial + 1, dtype=np.float32)
    freq = n_range * (np.pi / cutoff)

    d = distances[:, None]
    arg = d * freq[None, :]
    sin_val = np.sin(arg)
    sqrt_factor = np.sqrt(2.0 / cutoff)

    safe_r = np.where(distances == 0.0, 1.0, distances)[:, None]
    bessel = sqrt_factor * sin_val / safe_r

    p = envelope_exponent
    r_c = d / cutoff
    env = 1.0 - (p + 1.0) * (r_c ** p) + p * (r_c ** (p + 1.0))
    env = np.where(r_c < 1.0, env, 0.0)

    return bessel * env


def legende_poly_np(x, l_max):
    N = len(x)
    out = np.zeros((N, l_max + 1), dtype=np.float32)
    out[:, 0] = 1.0
    if l_max >= 1:
        out[:, 1] = x
    for l in range(1, l_max):
        out[:, l + 1] = ((2.0 * l + 1.0) * x * out[:, l] - l * out[:, l - 1]) / (l + 1.0)
    return out


def spherical_basis_np(distances, angles, num_spherical, num_radial,
                        cutoff, envelope_exponent=5):
    radial = bessel_basis_np(distances, num_radial, cutoff, envelope_exponent)
    cos_theta = np.cos(angles)
    l_max = num_spherical - 1
    legendre = legende_poly_np(cos_theta, l_max)
    combined = legendre[:, :, None] * radial[:, None, :]
    return combined.reshape(len(distances), -1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_sinusoids_embedding():
    import torch
    np.random.seed(42)
    N, n_freqs = 8, 4
    x = np.random.randn(N, 3).astype(np.float32)
    expected = sinusoids_embedding_np(x, n_freqs)

    x_t = torch.from_numpy(x).to('npu')
    out_t = torch.zeros(N, n_freqs * 6, dtype=torch.float32, device='npu')
    SinusoidsEmbedding(x_t, out_t, n_freqs)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  sinusoids_embedding: {status}")


def test_bessel_basis():
    import torch
    np.random.seed(42)
    E, num_radial, cutoff, env_exp = 32, 6, 5.0, 5
    distances = np.random.uniform(0.1, 5.0, E).astype(np.float32)
    expected = bessel_basis_np(distances, num_radial, cutoff, env_exp)

    d_t = torch.from_numpy(distances).to('npu')
    out_t = torch.zeros(E, num_radial, dtype=torch.float32, device='npu')
    BesselBasisLayer(d_t, out_t, num_radial, cutoff, env_exp)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  bessel_basis: {status}")


def test_legende_poly():
    import torch
    np.random.seed(42)
    N, l_max = 16, 5
    x = np.random.uniform(-1, 1, N).astype(np.float32)
    expected = legende_poly_np(x, l_max)

    x_t = torch.from_numpy(x).to('npu')
    out_t = torch.zeros(N, l_max + 1, dtype=torch.float32, device='npu')
    legende_poly(x_t, out_t, l_max)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  legende_poly: {status}")


def test_spherical_basis():
    import torch
    np.random.seed(42)
    T, num_sph, num_rad, cutoff, env_exp = 16, 4, 6, 5.0, 5
    distances = np.random.uniform(0.1, 5.0, T).astype(np.float32)
    angles = np.random.uniform(0, np.pi, T).astype(np.float32)
    expected = spherical_basis_np(distances, angles, num_sph, num_rad,
                                   cutoff, env_exp)

    d_t = torch.from_numpy(distances).to('npu')
    a_t = torch.from_numpy(angles).to('npu')
    out_t = torch.zeros(T, num_sph * num_rad, dtype=torch.float32,
                         device='npu')
    SphericalBasisLayer(d_t, a_t, out_t, num_sph, num_rad, cutoff, env_exp)
    result = out_t.cpu().numpy()
    err = np.abs(result - expected).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  spherical_basis: {status}")


def main():
    print("=" * 50)
    print("PyPTO Basis Ops -- Tests")
    print("=" * 50)
    test_sinusoids_embedding()
    test_bessel_basis()
    test_legende_poly()
    test_spherical_basis()
    print("=" * 50)


if __name__ == "__main__":
    main()
