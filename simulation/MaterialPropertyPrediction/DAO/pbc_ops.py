"""
PyPTO kernel: PBC distance operations.

Port of dao/common/data_utils.py periodic boundary condition operations.
"""

import pypto as pt
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Dense all-pairs PBC distance matrix  (27 periodic images)
# ---------------------------------------------------------------------------

@pt.frontend.jit
def get_pbc_distances(
    frac_coords: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattices: pt.Tensor([pt.DYNAMIC, 3, 3], pt.DT_FP32),
    num_atoms: pt.Tensor([pt.DYNAMIC], pt.DT_INT32),
    dist_matrix_out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    vec_matrix_out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC, 3], pt.DT_FP32),
):
    """Dense all-pairs PBC distance matrix.

    frac_coords    [total_atoms, 3]  fractional coordinates
    lattices       [batch_size, 3, 3]
    num_atoms      [batch_size]
    dist_matrix_out  [N, N]   min PBC distance per pair
    vec_matrix_out   [N, N, 3]  corresponding distance vector
    """
    total_atoms = frac_coords.shape[0]
    batch_size = lattices.shape[0]

    # Expand lattice matrix per atom
    lattice_nodes = pt.zeros([total_atoms, 3, 3], pt.DT_FP32)
    offset = pt.zeros(1, pt.DT_INT32)[0]
    for b in pt.loop(range(batch_size)):
        n = num_atoms[b]
        for i in pt.loop(range(n)):
            idx = offset + i
            lattice_nodes[idx] = lattices[b]
        offset = offset + n

    # frac --> cart
    cart = pt.einsum('bi,bij->bj', frac_coords, lattice_nodes)

    # 27 PBC offsets in fractional space  [27, 3]
    o = pt.cast(pt.arange(3, pt.DT_INT32), pt.DT_FP32) - 1.0
    ox = pt.reshape(o, (3, 1, 1))
    ox = pt.broadcast_to(ox, (3, 3, 3))
    ox = pt.reshape(ox, (27,))
    oy = pt.reshape(o, (1, 3, 1))
    oy = pt.broadcast_to(oy, (3, 3, 3))
    oy = pt.reshape(oy, (27,))
    oz = pt.reshape(o, (1, 1, 3))
    oz = pt.broadcast_to(oz, (3, 3, 3))
    oz = pt.reshape(oz, (27,))
    offset_frac = pt.stack([ox, oy, oz], dim=-1)

    # Cartesian offset for each atom  [total_atoms, 27, 3]
    offset_cart = pt.zeros([total_atoms, 27, 3], pt.DT_FP32)
    for i in pt.loop(range(total_atoms)):
        offset_cart[i] = pt.einsum('nk,kl->nl', offset_frac, lattice_nodes[i])

    # Pairwise distances with nearest-image convention
    for i in pt.loop(range(total_atoms)):
        ci = cart[i]
        oi = offset_cart[i]
        for j in pt.loop(range(total_atoms)):
            diff = ci - cart[j]
            vecs = diff + oi
            sq = pt.sum(vecs ** 2, dim=-1)
            min_sq = pt.min(sq, dim=0)
            mask = pt.cast(sq == min_sq, pt.DT_FP32)
            min_vec = pt.einsum('k,kl->l', mask, vecs) / pt.sum(mask)
            dist_matrix_out[i, j] = pt.sqrt(min_sq)
            vec_matrix_out[i, j] = min_vec


@pt.frontend.jit
def min_distance_sqr_pbc(
    coords_a: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    coords_b: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattice: pt.Tensor([3, 3], pt.DT_FP32),
    dist_sqr_out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
):
    """Minimum PBC squared distance between two point sets.

    coords_a     [N, 3]
    coords_b     [M, 3]
    lattice      [3, 3]
    dist_sqr_out [N, M]   min PBC squared distance per pair
    """
    N = coords_a.shape[0]
    M = coords_b.shape[0]

    # Cartesian PBC offsets  [27, 3]
    o = pt.cast(pt.arange(3, pt.DT_INT32), pt.DT_FP32) - 1.0
    ox = pt.reshape(o, (3, 1, 1))
    ox = pt.broadcast_to(ox, (3, 3, 3))
    ox = pt.reshape(ox, (27,))
    oy = pt.reshape(o, (1, 3, 1))
    oy = pt.broadcast_to(oy, (3, 3, 3))
    oy = pt.reshape(oy, (27,))
    oz = pt.reshape(o, (1, 1, 3))
    oz = pt.broadcast_to(oz, (3, 3, 3))
    oz = pt.reshape(oz, (27,))
    offset_frac = pt.stack([ox, oy, oz], dim=-1)
    offset_cart = pt.einsum('nk,kl->nl', offset_frac, lattice)

    for i in pt.loop(range(N)):
        ci = coords_a[i]
        for j in pt.loop(range(M)):
            diff = ci - coords_b[j]
            vecs = diff + offset_cart
            sq = pt.sum(vecs ** 2, dim=-1)
            dist_sqr_out[i, j] = pt.min(sq, dim=0)


# ---------------------------------------------------------------------------
# NumPy reference (for validation)
# ---------------------------------------------------------------------------

def get_pbc_distances_np(frac_coords, lattice, num_atoms=None):
    """NumPy reference for get_pbc_distances."""
    N = len(frac_coords)
    cart = frac_coords @ lattice.T

    offsets_frac = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                offsets_frac.append([ix, iy, iz])
    offsets_frac = np.array(offsets_frac, dtype=np.float32)
    offsets_cart = offsets_frac @ lattice

    dist_matrix = np.zeros((N, N), dtype=np.float32)
    vec_matrix = np.zeros((N, N, 3), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            diff = cart[i] - cart[j]
            vecs = diff + offsets_cart
            sq = np.sum(vecs ** 2, axis=-1)
            idx = np.argmin(sq)
            dist_matrix[i, j] = np.sqrt(sq[idx])
            vec_matrix[i, j] = vecs[idx]

    return dist_matrix, vec_matrix


def min_distance_sqr_pbc_np(coords_a, coords_b, lattice):
    """NumPy reference for min_distance_sqr_pbc."""
    N, M = len(coords_a), len(coords_b)

    offsets_frac = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                offsets_frac.append([ix, iy, iz])
    offsets_frac = np.array(offsets_frac, dtype=np.float32)
    offsets_cart = offsets_frac @ lattice

    result = np.zeros((N, M), dtype=np.float32)
    for i in range(N):
        for j in range(M):
            diff = coords_a[i] - coords_b[j]
            vecs = diff + offsets_cart
            sq = np.sum(vecs ** 2, axis=-1)
            result[i, j] = np.min(sq)

    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_get_pbc_distances():
    np.random.seed(42)
    N = 8
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)
    frac = np.random.rand(N, 3).astype(np.float32)

    f_t = torch.from_numpy(frac).to('npu')
    l_t = torch.from_numpy(lattice[None]).to('npu')
    na_t = torch.tensor([N], dtype=torch.int32, device='npu')
    d_out = torch.zeros(N, N, dtype=torch.float32, device='npu')
    v_out = torch.zeros(N, N, 3, dtype=torch.float32, device='npu')

    get_pbc_distances(f_t, l_t, na_t, d_out, v_out)

    d_npu = d_out.cpu().numpy()
    v_npu = v_out.cpu().numpy()
    d_np, v_np = get_pbc_distances_np(frac, lattice)

    err_d = np.abs(d_npu - d_np).max()
    err_v = np.abs(v_npu - v_np).max()
    status_d = "PASS" if err_d < 1e-3 else f"FAIL (err={err_d:.2e})"
    status_v = "PASS" if err_v < 1e-3 else f"FAIL (err={err_v:.2e})"
    print(f"  get_pbc_distances (d): {status_d}  (max_err={err_d:.2e})")
    print(f"  get_pbc_distances (v): {status_v}  (max_err={err_v:.2e})")


def test_min_distance_sqr_pbc():
    np.random.seed(42)
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [1.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)
    N, M = 6, 4
    coords_a = np.random.randn(N, 3).astype(np.float32) * 2.0
    coords_b = np.random.randn(M, 3).astype(np.float32) * 2.0

    a_t = torch.from_numpy(coords_a).to('npu')
    b_t = torch.from_numpy(coords_b).to('npu')
    l_t = torch.from_numpy(lattice).to('npu')
    d_out = torch.zeros(N, M, dtype=torch.float32, device='npu')

    min_distance_sqr_pbc(a_t, b_t, l_t, d_out)

    d_npu = d_out.cpu().numpy()
    d_np = min_distance_sqr_pbc_np(coords_a, coords_b, lattice)

    err = np.abs(d_npu - d_np).max()
    status = "PASS" if err < 1e-3 else f"FAIL (err={err:.2e})"
    print(f"  min_distance_sqr_pbc: {status}  (max_err={err:.2e})")


def main():
    print("=" * 50)
    print("PyPTO PBC Ops — Tests")
    print("=" * 50)
    test_get_pbc_distances()
    test_min_distance_sqr_pbc()
    print("=" * 50)


if __name__ == "__main__":
    main()
