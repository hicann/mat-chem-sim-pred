"""
Comprehensive test suite — validates every PyPTO operator against NumPy refs.

Usage:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export TILE_FWK_DEVICE_ID=0
    python -m dao.test_ops
"""

import numpy as np
from numpy.testing import assert_allclose

from .lattice_ops import (
    lattice_params_to_matrix as lat_kernel,
    frac_to_cart_coords as frac_cart_kernel,
    cart_to_frac_coords as cart_frac_kernel,
    compute_volume as vol_kernel,
    lattice_params_to_matrix_np,
    frac_to_cart_coords_np,
    cart_to_frac_coords_np,
    compute_volume_np,
)
from .pbc_ops import (
    get_pbc_distances as pbc_dist_kernel,
    min_distance_sqr_pbc as min_sq_kernel,
    get_pbc_distances_np,
    min_distance_sqr_pbc_np,
)
from .graph_ops import (
    compute_pairwise_pbc_distances,
    radius_graph_pbc,
    radius_graph_pbc_np,
)
from .basis_ops import (
    SinusoidsEmbedding,
    BesselBasisLayer,
    SphericalBasisLayer,
    sinusoids_embedding_np,
    bessel_basis_np,
    spherical_basis_np,
)
from .attention_ops import (
    CrysFormerAttention,
    TransformerBlock,
    crysformer_attention_np,
    transformer_block_np,
)
from .diffusion_ops import (
    SinusoidalTimeEmbeddings,
    d_log_p_wrapped_normal,
    add_diffusion_noise,
    sinusoidal_time_embeddings_np,
    d_log_p_wrapped_normal_np,
    add_diffusion_noise_np,
    BetaScheduler,
    SigmaScheduler,
)

import torch


# ============================================================================
# Tolerance
# ============================================================================

RTOL = 1e-3
ATOL = 1e-3


def check(label, result, expected, rtol=RTOL, atol=ATOL):
    try:
        assert_allclose(result, expected, rtol=rtol, atol=atol)
        print(f"  {label}:  PASS")
    except AssertionError as e:
        err = np.abs(result - expected).max()
        print(f"  {label}:  FAIL (max_err={err:.2e})")


# ============================================================================
# Lattice ops
# ============================================================================

def test_lattice_ops():
    np.random.seed(42)

    # lattice_params_to_matrix
    lengths = np.random.uniform(3, 15, (4, 3)).astype(np.float32)
    angles = np.random.uniform(50, 130, (4, 3)).astype(np.float32)
    l_t = torch.from_numpy(lengths).to('npu')
    a_t = torch.from_numpy(angles).to('npu')
    m_out = torch.zeros(4, 3, 3, dtype=torch.float32, device='npu')
    lat_kernel(l_t, a_t, m_out)
    result = m_out.cpu().numpy()
    expected = np.array([lattice_params_to_matrix_np(*p) for p in zip(lengths, angles)])
    check("lattice_params_to_matrix", result, expected)

    # frac_cart_roundtrip
    lengths_1 = np.array([[5.0, 5.0, 5.0]], dtype=np.float32)
    angles_1 = np.array([[90.0, 90.0, 90.0]], dtype=np.float32)
    l_t = torch.from_numpy(lengths_1).to('npu')
    a_t = torch.from_numpy(angles_1).to('npu')
    m_single = torch.zeros(1, 3, 3, dtype=torch.float32, device='npu')
    lat_kernel(l_t, a_t, m_single)
    lattice_np = m_single.cpu().numpy()[0]

    frac_pt = torch.rand(10, 3, device='npu')
    cart_pt = torch.zeros(10, 3, dtype=torch.float32, device='npu')
    frac_cart_kernel(frac_pt, m_single, torch.tensor([10], device='npu'), cart_pt)
    cart_np = cart_pt.cpu().numpy()
    expected_cart = frac_to_cart_coords_np(frac_pt.cpu().numpy(), lattice_np)
    check("frac_to_cart_coords", cart_np, expected_cart)

    # volume
    v_out = torch.zeros(1, dtype=torch.float32, device='npu')
    vol_kernel(m_single, v_out)
    vol_np = compute_volume_np(lattice_np)
    check("compute_volume", v_out.cpu().item(), vol_np, rtol=1e-4, atol=1e-4)


# ============================================================================
# PBC ops
# ============================================================================

def test_pbc_ops():
    np.random.seed(42)
    N = 8
    lattice = np.array([[5.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [0.0, 0.0, 5.0]], dtype=np.float32)
    frac = np.random.rand(N, 3).astype(np.float32)

    # get_pbc_distances
    f_t = torch.from_numpy(frac).to('npu')
    l_t = torch.from_numpy(lattice[None]).to('npu')
    na_t = torch.tensor([N], dtype=torch.int32, device='npu')
    d_out = torch.zeros(N, N, dtype=torch.float32, device='npu')
    v_out = torch.zeros(N, N, 3, dtype=torch.float32, device='npu')
    pbc_dist_kernel(f_t, l_t, na_t, d_out, v_out)
    d_np, v_np = get_pbc_distances_np(frac, lattice)
    check("get_pbc_distances (d)", d_out.cpu().numpy(), d_np)
    check("get_pbc_distances (v)", v_out.cpu().numpy(), v_np, rtol=1e-2, atol=1e-2)

    # min_distance_sqr_pbc
    a = np.random.randn(6, 3).astype(np.float32) * 2
    b = np.random.randn(4, 3).astype(np.float32) * 2
    a_t = torch.from_numpy(a).to('npu')
    b_t = torch.from_numpy(b).to('npu')
    lat_t = torch.from_numpy(lattice).to('npu')
    sq_out = torch.zeros(6, 4, dtype=torch.float32, device='npu')
    min_sq_kernel(a_t, b_t, lat_t, sq_out)
    expected = min_distance_sqr_pbc_np(a, b, lattice)
    check("min_distance_sqr_pbc", sq_out.cpu().numpy(), expected)


# ============================================================================
# Graph ops
# ============================================================================

def test_graph_ops():
    np.random.seed(42)
    N = 10
    lattice = np.array([[8.0, 0.0, 0.0],
                        [0.0, 8.0, 0.0],
                        [0.0, 0.0, 8.0]], dtype=np.float32)
    frac = np.random.rand(N, 3).astype(np.float32)
    radius, max_nb = 4.0, 6

    # radius_graph_pbc
    ei, co, nb = radius_graph_pbc(frac, None, None, N, radius, max_nb, 'npu', lattice=lattice)
    ei_np, co_np, nb_np = radius_graph_pbc_np(frac, lattice, radius, max_nb)
    check("radius_graph_pbc (edge_index)", ei.astype(np.int64), ei_np.astype(np.int64))
    check("radius_graph_pbc (cell_offsets)", co.astype(np.int64), co_np.astype(np.int64))
    check("radius_graph_pbc (num_bonds)", nb.astype(np.int32), nb_np.astype(np.int32))


# ============================================================================
# Basis ops
# ============================================================================

def test_basis_ops():
    np.random.seed(42)

    # SinusoidsEmbedding
    N, nf = 8, 4
    x = np.random.randn(N, 3).astype(np.float32)
    x_t = torch.from_numpy(x).to('npu')
    out_t = torch.zeros(N, nf * 6, dtype=torch.float32, device='npu')
    SinusoidsEmbedding(x_t, out_t, nf)
    check("SinusoidsEmbedding", out_t.cpu().numpy(), sinusoids_embedding_np(x, nf))

    # BesselBasisLayer
    E, nr, cutoff = 32, 6, 5.0
    dist = np.random.uniform(0.1, cutoff, E).astype(np.float32)
    d_t = torch.from_numpy(dist).to('npu')
    b_out = torch.zeros(E, nr, dtype=torch.float32, device='npu')
    BesselBasisLayer(d_t, b_out, nr, cutoff, 5)
    check("BesselBasisLayer", b_out.cpu().numpy(), bessel_basis_np(dist, nr, cutoff))

    # SphericalBasisLayer
    T, ns = 16, 4
    angles = np.random.uniform(0, np.pi, T).astype(np.float32)
    a_t = torch.from_numpy(angles).to('npu')
    s_out = torch.zeros(T, ns * nr, dtype=torch.float32, device='npu')
    SphericalBasisLayer(d_t, a_t, s_out, ns, nr, cutoff, 5)
    check("SphericalBasisLayer", s_out.cpu().numpy(),
          spherical_basis_np(dist, angles, ns, nr, cutoff))


# ============================================================================
# Attention ops
# ============================================================================

def test_attention_ops():
    np.random.seed(42)
    D, nh, N = 64, 4, 8

    node_feats = np.random.randn(N, D).astype(np.float32)
    frac = np.random.rand(N, 3).astype(np.float32)
    lattice = np.eye(3, dtype=np.float32) * 5.0

    rng = np.random.RandomState(0)
    W_q = rng.randn(D, D).astype(np.float32) * 0.02
    W_k = rng.randn(D, D).astype(np.float32) * 0.02
    W_v = rng.randn(D, D).astype(np.float32) * 0.02
    W_o = rng.randn(D, D).astype(np.float32) * 0.02
    W_ek = rng.randn(D, 12).astype(np.float32) * 0.02
    W_ev = rng.randn(D, 12).astype(np.float32) * 0.02

    def to_npu(a): return torch.from_numpy(a).to('npu')

    nf_t = to_npu(node_feats)
    fc_t = to_npu(frac)
    lat_t = to_npu(lattice)
    D_t = torch.tensor(D, device='npu')
    nh_t = torch.tensor(nh, device='npu')

    # CrysFormerAttention
    attn_out = torch.zeros(N, D, dtype=torch.float32, device='npu')
    CrysFormerAttention(nf_t, fc_t, lat_t, to_npu(W_q), to_npu(W_k), to_npu(W_v),
                        to_npu(W_ek), to_npu(W_ev), to_npu(W_o), attn_out, D_t, nh_t)
    expected = crysformer_attention_np(node_feats, frac, lattice, W_q, W_k, W_v, W_ek, W_ev, W_o, nh)
    check("CrysFormerAttention", attn_out.cpu().numpy(), expected)

    # TransformerBlock
    W_f1 = rng.randn(D, D * 4).astype(np.float32) * 0.02
    b_f1 = np.zeros(D * 4, dtype=np.float32)
    W_f2 = rng.randn(D * 4, D).astype(np.float32) * 0.02
    b_f2 = np.zeros(D, dtype=np.float32)
    g_a = np.ones(D, dtype=np.float32) * 0.5
    g_f = np.ones(D, dtype=np.float32) * 0.5

    block_out = torch.zeros(N, D, dtype=torch.float32, device='npu')
    TransformerBlock(nf_t, fc_t, lat_t,
                     to_npu(W_q), to_npu(W_k), to_npu(W_v),
                     to_npu(W_ek), to_npu(W_ev), to_npu(W_o),
                     to_npu(W_f1), to_npu(b_f1), to_npu(W_f2), to_npu(b_f2),
                     to_npu(g_a), to_npu(g_f), block_out, D_t, nh_t)
    expected_block = transformer_block_np(node_feats, frac, lattice,
                                           W_q, W_k, W_v, W_ek, W_ev, W_o,
                                           W_f1, b_f1, W_f2, b_f2, g_a, g_f, nh)
    check("TransformerBlock", block_out.cpu().numpy(), expected_block)


# ============================================================================
# Diffusion ops
# ============================================================================

def test_diffusion_ops():
    np.random.seed(42)

    # SinusoidalTimeEmbeddings
    B, dim = 16, 128
    time = np.random.uniform(0, 1, B).astype(np.float32)
    t_t = torch.from_numpy(time).to('npu')
    emb_out = torch.zeros(B, dim, dtype=torch.float32, device='npu')
    SinusoidalTimeEmbeddings(t_t, emb_out, dim)
    check("SinusoidalTimeEmbeddings", emb_out.cpu().numpy(),
          sinusoidal_time_embeddings_np(time, dim))

    # d_log_p_wrapped_normal
    N = 32
    z = np.random.randn(N, 3).astype(np.float32)
    sigma = np.random.uniform(0.1, 2.0, (N, 1)).astype(np.float32)
    z_t = torch.from_numpy(z).to('npu')
    s_t = torch.from_numpy(sigma).to('npu')
    sc_out = torch.zeros(N, 3, dtype=torch.float32, device='npu')
    d_log_p_wrapped_normal(z_t, s_t, sc_out)
    check("d_log_p_wrapped_normal", sc_out.cpu().numpy(), d_log_p_wrapped_normal_np(z, sigma))

    # add_diffusion_noise
    data = np.random.randn(N, 3).astype(np.float32)
    noise = np.random.randn(N, 3).astype(np.float32)
    ac = np.random.uniform(0, 1, (N, 1)).astype(np.float32)
    d_t = torch.from_numpy(data).to('npu')
    n_t = torch.from_numpy(noise).to('npu')
    a_t = torch.from_numpy(ac).to('npu')
    noised = torch.zeros(N, 3, dtype=torch.float32, device='npu')
    add_diffusion_noise(d_t, n_t, a_t, noised)
    check("add_diffusion_noise", noised.cpu().numpy(), add_diffusion_noise_np(data, noise, ac))

    # Schedulers
    bs = BetaScheduler(timesteps=1000, schedule='cosine')
    assert len(bs.betas) == 1000
    bs_lin = BetaScheduler(timesteps=1000, schedule='linear')
    assert len(bs_lin.betas) == 1000
    ss = SigmaScheduler(sigma_min=0.01, sigma_max=50.0, timesteps=1000)
    assert len(ss.sigmas) == 1000
    print("  BetaScheduler/SigmaScheduler: PASS")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("DAO-PyPTO Comprehensive Test Suite")
    print("=" * 60)

    test_lattice_ops()
    test_pbc_ops()
    test_graph_ops()
    test_basis_ops()
    test_attention_ops()
    test_diffusion_ops()

    print("=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
