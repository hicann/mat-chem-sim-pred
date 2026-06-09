"""
PyPTO kernel: lattice parameter conversions.

Port of dao/common/data_utils.py lattice operations.
"""

import pypto as pt
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Lattice parameters → matrix  (a,b,c,α,β,γ) → [[3,3]]
# ---------------------------------------------------------------------------

@pt.frontend.jit
def lattice_params_to_matrix(
    lengths: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    angles: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    matrix_out: pt.Tensor([pt.DYNAMIC, 3, 3], pt.DT_FP32),
):
    """Batched lattice parameters to 3×3 matrix.

    lengths: [N, 3]   (a, b, c) in Å
    angles:  [N, 3]   (α, β, γ) in degrees
    matrix_out: [N, 3, 3]
    """
    angles_r = angles * (pt.ones_like(angles) * 0.01745329252)  # π/180

    cos_a = pt.cos(angles_r[:, 0])
    cos_b = pt.cos(angles_r[:, 1])
    cos_g = pt.cos(angles_r[:, 2])
    sin_a = pt.sin(angles_r[:, 0])
    sin_b = pt.sin(angles_r[:, 1])
    sin_g = pt.sin(angles_r[:, 2])

    val = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    val = pt.clamp(val, -1.0, 1.0)
    gamma_star = pt.arccos(val)

    # vector_a: [a·sinβ,  0,  a·cosβ]
    vec_a_x = lengths[:, 0] * sin_b
    vec_a_y = pt.zeros(lengths.shape[0], pt.DT_FP32)
    vec_a_z = lengths[:, 0] * cos_b

    # vector_b: [-b·sinα·cosγ*,  b·sinα·sinγ*,  b·cosα]
    vec_b_x = -lengths[:, 1] * sin_a * pt.cos(gamma_star)
    vec_b_y = lengths[:, 1] * sin_a * pt.sin(gamma_star)
    vec_b_z = lengths[:, 1] * cos_a

    # vector_c: [0,  0,  c]
    vec_c_x = pt.zeros(lengths.shape[0], pt.DT_FP32)
    vec_c_y = pt.zeros(lengths.shape[0], pt.DT_FP32)
    vec_c_z = lengths[:, 2]

    # Stack into [N, 3, 3]
    a = pt.stack([vec_a_x, vec_a_y, vec_a_z], dim=-1)
    b = pt.stack([vec_b_x, vec_b_y, vec_b_z], dim=-1)
    c = pt.stack([vec_c_x, vec_c_y, vec_c_z], dim=-1)
    matrix_out[:] = pt.stack([a, b, c], dim=1)


@pt.frontend.jit
def lattice_params_to_matrix_single(
    lengths: pt.Tensor([3], pt.DT_FP32),
    angles: pt.Tensor([3], pt.DT_FP32),
    matrix_out: pt.Tensor([3, 3], pt.DT_FP32),
):
    """Unbatched version: single crystal."""
    deg2rad = 0.01745329252
    angles_r = angles * deg2rad

    cos_a, cos_b, cos_g = pt.cos(angles_r[0]), pt.cos(angles_r[1]), pt.cos(angles_r[2])
    sin_a, sin_b, sin_g = pt.sin(angles_r[0]), pt.sin(angles_r[1]), pt.sin(angles_r[2])

    val = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    val = pt.clamp(val, -1.0, 1.0)
    gamma_star = pt.arccos(val)

    vec_a = pt.stack([
        lengths[0] * sin_b,
        pt.zeros(1, pt.DT_FP32)[0],
        lengths[0] * cos_b,
    ], dim=0)
    vec_b = pt.stack([
        -lengths[1] * sin_a * pt.cos(gamma_star),
        lengths[1] * sin_a * pt.sin(gamma_star),
        lengths[1] * cos_a,
    ], dim=0)
    vec_c = pt.stack([
        pt.zeros(1, pt.DT_FP32)[0],
        pt.zeros(1, pt.DT_FP32)[0],
        lengths[2],
    ], dim=0)

    matrix_out[:] = pt.stack([vec_a, vec_b, vec_c], dim=0)


# ---------------------------------------------------------------------------
# Fractional coordinates → Cartesian coordinates
# ---------------------------------------------------------------------------

@pt.frontend.jit
def frac_to_cart_coords(
    frac_coords: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattices: pt.Tensor([pt.DYNAMIC, 3, 3], pt.DT_FP32),
    num_atoms: pt.Tensor([pt.DYNAMIC], pt.DT_INT32),
    cart_out: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
):
    """frac_coords [total_atoms, 3] → cartesian.

    lattice_nodes = repeat_interleave(lattices, num_atoms, dim=0)
    cart = einsum('bi,bij->bj', frac_coords, lattice_nodes)
    """
    batch_size = lattices.shape[0]
    total_atoms = frac_coords.shape[0]

    # Expand lattice matrix per atom in its batch
    lattice_nodes = pt.zeros([total_atoms, 3, 3], pt.DT_FP32)
    offset = pt.zeros(1, pt.DT_INT32)[0]
    for b in pt.loop(range(batch_size)):
        n = num_atoms[b]
        for i in pt.loop(range(n)):
            idx = offset + i
            lattice_nodes[idx] = lattices[b]
        offset = offset + n

    cart_out[:] = pt.einsum('bi,bij->bj', frac_coords, lattice_nodes)


# ---------------------------------------------------------------------------
# Cartesian coordinates → Fractional coordinates
# ---------------------------------------------------------------------------

@pt.frontend.jit
def cart_to_frac_coords(
    cart_coords: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattices: pt.Tensor([pt.DYNAMIC, 3, 3], pt.DT_FP32),
    num_atoms: pt.Tensor([pt.DYNAMIC], pt.DT_INT32),
    frac_out: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
):
    """cart_coords [total_atoms, 3] → fractional (mod 1)."""
    batch_size = lattices.shape[0]
    total_atoms = cart_coords.shape[0]

    # Compute pseudo-inverse for each lattice
    inv_lattices = pt.zeros([batch_size, 3, 3], pt.DT_FP32)
    # Using transpose as approximate inverse for orthogonal; for full inverse:
    for b in pt.loop(range(batch_size)):
        m = lattices[b]
        det = m[0, 0] * (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) \
            - m[0, 1] * (m[1, 0] * m[2, 2] - m[1, 2] * m[2, 0]) \
            + m[0, 2] * (m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0])
        inv_det = pt.reciprocal(det)
        # Adjugate matrix
        inv_lattices[b, 0, 0] = (m[1, 1] * m[2, 2] - m[1, 2] * m[2, 1]) * inv_det
        inv_lattices[b, 0, 1] = (m[0, 2] * m[2, 1] - m[0, 1] * m[2, 2]) * inv_det
        inv_lattices[b, 0, 2] = (m[0, 1] * m[1, 2] - m[0, 2] * m[1, 1]) * inv_det
        inv_lattices[b, 1, 0] = (m[1, 2] * m[2, 0] - m[1, 0] * m[2, 2]) * inv_det
        inv_lattices[b, 1, 1] = (m[0, 0] * m[2, 2] - m[0, 2] * m[2, 0]) * inv_det
        inv_lattices[b, 1, 2] = (m[1, 0] * m[0, 2] - m[0, 0] * m[1, 2]) * inv_det
        inv_lattices[b, 2, 0] = (m[1, 0] * m[2, 1] - m[2, 0] * m[1, 1]) * inv_det
        inv_lattices[b, 2, 1] = (m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1]) * inv_det
        inv_lattices[b, 2, 2] = (m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]) * inv_det

    # Expand per atom
    inv_lattice_nodes = pt.zeros([total_atoms, 3, 3], pt.DT_FP32)
    offset = pt.zeros(1, pt.DT_INT32)[0]
    for b in pt.loop(range(batch_size)):
        n = num_atoms[b]
        for i in pt.loop(range(n)):
            idx = offset + i
            inv_lattice_nodes[idx] = inv_lattices[b]
        offset = offset + n

    frac = pt.einsum('bi,bij->bj', cart_coords, inv_lattice_nodes)
    # Wrap to [0, 1)
    frac_out[:] = frac - pt.floor(frac)


# ---------------------------------------------------------------------------
# Volume computation
# ---------------------------------------------------------------------------

@pt.frontend.jit
def compute_volume(
    lattices: pt.Tensor([pt.DYNAMIC, 3, 3], pt.DT_FP32),
    volume_out: pt.Tensor([pt.DYNAMIC], pt.DT_FP32),
):
    """Volume = |a · (b × c)| for each lattice in batch."""
    a = lattices[:, 0]
    b = lattices[:, 1]
    c = lattices[:, 2]
    cross = pt.stack([
        b[:, 1] * c[:, 2] - b[:, 2] * c[:, 1],
        b[:, 2] * c[:, 0] - b[:, 0] * c[:, 2],
        b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0],
    ], dim=-1)
    vol = pt.abs(pt.sum(a * cross, dim=-1))
    volume_out[:] = vol


# ---------------------------------------------------------------------------
# NumPy reference (for validation)
# ---------------------------------------------------------------------------

def lattice_params_to_matrix_np(lengths, angles):
    a, b, c = lengths
    alpha, beta, gamma = np.radians(angles)
    cos_a, cos_b, cos_g = np.cos([alpha, beta, gamma])
    sin_a, sin_b, sin_g = np.sin([alpha, beta, gamma])

    val = (cos_a * cos_b - cos_g) / (sin_a * sin_b)
    val = np.clip(val, -1, 1)
    gamma_star = np.arccos(val)

    vector_a = np.array([a * sin_b, 0.0, a * cos_b])
    vector_b = np.array([
        -b * sin_a * np.cos(gamma_star),
        b * sin_a * np.sin(gamma_star),
        b * cos_a,
    ])
    vector_c = np.array([0.0, 0.0, c])
    return np.array([vector_a, vector_b, vector_c])


def frac_to_cart_coords_np(frac_coords, lattice):
    return frac_coords @ lattice.T


def cart_to_frac_coords_np(cart_coords, lattice):
    inv = np.linalg.inv(lattice)
    frac = cart_coords @ inv.T
    return frac % 1.0


def compute_volume_np(lattice):
    return abs(np.linalg.det(lattice))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lattice_params_to_matrix():
    np.random.seed(42)
    for _ in range(10):
        lengths = np.random.uniform(3, 15, 3).astype(np.float32)
        angles = np.random.uniform(50, 130, 3).astype(np.float32)

        l_t = torch.from_numpy(lengths[None]).to('npu')
        a_t = torch.from_numpy(angles[None]).to('npu')
        m_out = torch.zeros(1, 3, 3, dtype=torch.float32, device='npu')
        lattice_params_to_matrix(l_t, a_t, m_out)

        result = m_out.cpu().numpy()[0]
        expected = lattice_params_to_matrix_np(lengths, angles)
        err = np.abs(result - expected).max()
        status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
        print(f"  lattice_params_to_matrix: {status}")


def test_frac_cart_roundtrip():
    np.random.seed(42)
    lengths = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    angles = np.array([90.0, 90.0, 90.0], dtype=np.float32)
    lattice = lattice_params_to_matrix_np(lengths, angles)

    frac = np.random.rand(10, 3).astype(np.float32)
    cart = frac_to_cart_coords_np(frac, lattice)
    frac_back = cart_to_frac_coords_np(cart, lattice)

    err = np.abs(frac - frac_back).max()
    status = "PASS" if err < 1e-5 else f"FAIL (err={err:.2e})"
    print(f"  frac_cart_roundtrip: {status}")


def test_volume():
    np.random.seed(42)
    lengths = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    angles = np.array([90.0, 95.0, 90.0], dtype=np.float32)
    lattice = lattice_params_to_matrix_np(lengths, angles)

    l_t = torch.from_numpy(lengths[None]).to('npu')
    a_t = torch.from_numpy(angles[None]).to('npu')
    m_out = torch.zeros(1, 3, 3, dtype=torch.float32, device='npu')
    lattice_params_to_matrix(l_t, a_t, m_out)
    v_out = torch.zeros(1, dtype=torch.float32, device='npu')
    compute_volume(m_out, v_out)

    vol_npu = v_out.cpu().item()
    vol_np = compute_volume_np(lattice)
    err = abs(vol_npu - vol_np)
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  volume: {status}")


def main():
    print("=" * 50)
    print("PyPTO Lattice Ops — Tests")
    print("=" * 50)
    test_lattice_params_to_matrix()
    test_frac_cart_roundtrip()
    test_volume()
    print("=" * 50)


if __name__ == "__main__":
    main()
