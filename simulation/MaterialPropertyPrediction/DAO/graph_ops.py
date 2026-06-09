"""
PyPTO kernel: PBC radius graph construction.

Port of DAO's radius_graph_pbc() from dao/common/data_utils.py.
"""

import pypto as pt
import torch
import numpy as np
from .lattice_ops import lattice_params_to_matrix


# ---------------------------------------------------------------------------
# PyPTO kernel: dense all-pairs PBC distance matrix  (27 periodic images)
# ---------------------------------------------------------------------------

@pt.frontend.jit
def compute_pairwise_pbc_distances(
    cart_coords: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    lattice: pt.Tensor([3, 3], pt.DT_FP32),
    distances_out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC], pt.DT_FP32),
    offset_indices_out: pt.Tensor([pt.DYNAMIC, pt.DYNAMIC, 3], pt.DT_INT32),
):
    """All-pairs PBC distances with nearest-image convention.

    cart_coords        [N, 3]
    lattice            [3, 3]
    distances_out      [N, N]    min PBC distance per pair
    offset_indices_out [N, N, 3]  fractional offset of the image used
    """
    N = cart_coords.shape[0]

    # 27 fractional offsets spanning [-1, 0, 1]^3
    n = pt.cast(pt.arange(3, pt.DT_INT32), pt.DT_FP32) - 1.0

    ox = pt.reshape(pt.broadcast_to(pt.reshape(n, (3, 1, 1)), (3, 3, 3)), (27,))
    oy = pt.reshape(pt.broadcast_to(pt.reshape(n, (1, 3, 1)), (3, 3, 3)), (27,))
    oz = pt.reshape(pt.broadcast_to(pt.reshape(n, (1, 1, 3)), (3, 3, 3)), (27,))
    offset_frac = pt.stack([ox, oy, oz], dim=-1)

    offset_cart = pt.einsum('nk,kl->nl', offset_frac, lattice)

    idx_range = pt.arange(27, pt.DT_FP32)

    for i in pt.loop(range(N)):
        ci = cart_coords[i]
        for j in pt.loop(range(N)):
            diff = ci - cart_coords[j]
            vecs = diff + offset_cart
            sq = pt.sum(vecs ** 2, dim=-1)
            min_sq = pt.min(sq, dim=0)
            distances_out[i, j] = pt.sqrt(min_sq)
            mask = pt.cast(sq == min_sq, pt.DT_FP32)
            argmin_f = pt.sum(mask * idx_range, dim=0)
            argmin = pt.cast(argmin_f, pt.DT_INT32)
            offset_indices_out[i, j, 0] = offset_frac[argmin, 0]
            offset_indices_out[i, j, 1] = offset_frac[argmin, 1]
            offset_indices_out[i, j, 2] = offset_frac[argmin, 2]


# ---------------------------------------------------------------------------
# Host-side neighbor selection (CPU)
# ---------------------------------------------------------------------------

def _build_graph_from_distances(dist_matrix, offset_indices, radius, max_num_neighbors):
    """Build edge_index, cell_offsets, num_bonds from dense distance matrix.

    dist_matrix      [N, N]
    offset_indices   [N, N, 3]
    radius           float
    max_num_neighbors int

    Returns:
        edge_index   [2, E]  int64
        cell_offsets [E, 3]  int64
        num_bonds    [N]     int32
    """
    N = dist_matrix.shape[0]
    src_list, dst_list, off_list = [], [], []
    num_bonds = np.zeros(N, dtype=np.int32)

    for i in range(N):
        row = dist_matrix[i]
        valid = np.where((row > 0) & (row <= radius))[0]
        if len(valid) == 0:
            continue
        order = np.argsort(row[valid])
        valid = valid[order]
        if len(valid) > max_num_neighbors:
            valid = valid[:max_num_neighbors]
        for j in valid:
            src_list.append(i)
            dst_list.append(j)
            off_list.append(offset_indices[i, j])
        num_bonds[i] = len(valid)

    E = len(src_list)
    edge_index = np.zeros((2, E), dtype=np.int64)
    if E > 0:
        edge_index[0] = src_list
        edge_index[1] = dst_list
    cell_offsets = np.array(off_list, dtype=np.int64).reshape(-1, 3)
    return edge_index, cell_offsets, num_bonds


def radius_graph_pbc(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    radius=5.0,
    max_num_neighbors=20,
    device='npu',
    lattice=None,
):
    """PBC radius graph construction.

    Uses the compute_pairwise_pbc_distances PyPTO kernel on NPU for
    the compute-heavy distance matrix, then performs neighbor selection
    (sorting, masking) on CPU.

    Args:
        frac_coords:  [total_atoms, 3] or [N, 3]  fractional coordinates
        lengths:      [B, 3] or [3]              lattice lengths (a, b, c)
        angles:       [B, 3] or [3]              lattice angles (alpha, beta, gamma)
        num_atoms:    [B] or int                 atoms per structure
        radius:       float                      cutoff radius
        max_num_neighbors: int                   max neighbors per atom
        device:       str                        NPU device
        lattice:      [B, 3, 3] or [3, 3]        optional precomputed lattice

    Returns:
        edge_index    [2, E]  int64 numpy
        cell_offsets  [E, 3]  int64 numpy
        num_bonds     [B]     int32 numpy (sum per structure)
    """
    frac_coords = np.asarray(frac_coords, dtype=np.float32)

    if isinstance(num_atoms, (int, np.integer)):
        num_atoms = [int(num_atoms)]
    num_atoms = list(num_atoms)
    batch_size = len(num_atoms)

    # --- Build lattice matrices ---
    if lattice is not None:
        lattice = np.asarray(lattice, dtype=np.float32)
        if lattice.ndim == 2:
            lattice = lattice[None]
    else:
        lengths = np.asarray(lengths, dtype=np.float32)
        angles = np.asarray(angles, dtype=np.float32)
        if lengths.ndim == 1:
            lengths = lengths[None]
            angles = angles[None]
        lt = torch.from_numpy(lengths).to(device)
        at = torch.from_numpy(angles).to(device)
        lat_out = torch.zeros(batch_size, 3, 3, dtype=torch.float32, device=device)
        lattice_params_to_matrix(lt, at, lat_out)
        lattice = lat_out.cpu().numpy()

    # --- Process each structure ---
    all_edge_index = []
    all_cell_offsets = []
    all_num_bonds = []
    atom_offset = 0

    for b in range(batch_size):
        n = num_atoms[b]
        if n == 0:
            all_num_bonds.append(np.zeros(0, dtype=np.int32))
            continue

        frac_b = frac_coords[atom_offset:atom_offset + n]
        lat_b = lattice[b] if lattice.ndim == 3 else lattice

        # Step 1: PyPTO kernel on NPU
        frac_t = torch.from_numpy(frac_b).to(device)
        lat_t = torch.from_numpy(lat_b).to(device)
        cart_t = torch.mm(frac_t, lat_t.T)

        dist_out = torch.zeros(n, n, dtype=torch.float32, device=device)
        off_out = torch.zeros(n, n, 3, dtype=torch.int32, device=device)
        compute_pairwise_pbc_distances(cart_t, lat_t, dist_out, off_out)

        # Step 2: Copy to CPU
        dist_np = dist_out.cpu().numpy()
        off_np = off_out.cpu().numpy()

        # Step 3: Build graph on CPU
        ei, co, nb = _build_graph_from_distances(dist_np, off_np, radius, max_num_neighbors)

        # Offset indices to global atom positions
        if ei.shape[1] > 0:
            ei = ei.copy()
            ei[0] += atom_offset
            ei[1] += atom_offset

        all_edge_index.append(ei)
        all_cell_offsets.append(co)
        all_num_bonds.append(nb)

        atom_offset += n

    # --- Concatenate across batch ---
    if batch_size == 1:
        return all_edge_index[0], all_cell_offsets[0], all_num_bonds[0]

    nonzero = [i for i, ei in enumerate(all_edge_index) if ei.shape[1] > 0]
    if len(nonzero) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        cell_offsets = np.zeros((0, 3), dtype=np.int64)
    else:
        E_total = sum(all_edge_index[i].shape[1] for i in nonzero)
        edge_index = np.zeros((2, E_total), dtype=np.int64)
        cell_offsets = np.zeros((E_total, 3), dtype=np.int64)
        pos = 0
        for i in nonzero:
            ei = all_edge_index[i]
            co = all_cell_offsets[i]
            n_e = ei.shape[1]
            edge_index[:, pos:pos + n_e] = ei
            cell_offsets[pos:pos + n_e] = co
            pos += n_e

    num_bonds = np.concatenate(all_num_bonds)
    return edge_index, cell_offsets, num_bonds


# ---------------------------------------------------------------------------
# NumPy reference (for validation)
# ---------------------------------------------------------------------------

def radius_graph_pbc_np(frac_coords, lattice, radius, max_num_neighbors):
    """Pure NumPy reference for radius_graph_pbc.

    frac_coords  [N, 3]
    lattice      [3, 3]
    radius       float
    max_num_neighbors int

    Returns:
        edge_index   [2, E]
        cell_offsets [E, 3]
        num_bonds    [N]
    """
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
    offset_matrix = np.zeros((N, N, 3), dtype=np.int32)

    for i in range(N):
        for j in range(N):
            diff = cart[i] - cart[j]
            vecs = diff + offsets_cart
            sq = np.sum(vecs ** 2, axis=-1)
            idx = np.argmin(sq)
            dist_matrix[i, j] = np.sqrt(sq[idx])
            offset_matrix[i, j] = offsets_frac[idx].astype(np.int32)

    return _build_graph_from_distances(dist_matrix, offset_matrix, radius, max_num_neighbors)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compute_pairwise_pbc_distances():
    np.random.seed(42)
    N = 8
    lattice = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
    ], dtype=np.float32)
    frac = np.random.rand(N, 3).astype(np.float32)
    cart = frac @ lattice.T

    cart_t = torch.from_numpy(cart).to('npu')
    lat_t = torch.from_numpy(lattice).to('npu')
    d_out = torch.zeros(N, N, dtype=torch.float32, device='npu')
    o_out = torch.zeros(N, N, 3, dtype=torch.int32, device='npu')

    compute_pairwise_pbc_distances(cart_t, lat_t, d_out, o_out)

    d_npu = d_out.cpu().numpy()
    o_npu = o_out.cpu().numpy()

    # NumPy reference distances
    offsets_frac = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            for iz in (-1, 0, 1):
                offsets_frac.append([ix, iy, iz])
    offsets_frac = np.array(offsets_frac, dtype=np.float32)
    offsets_cart = offsets_frac @ lattice

    d_np = np.zeros((N, N), dtype=np.float32)
    o_np = np.zeros((N, N, 3), dtype=np.int32)
    for i in range(N):
        for j in range(N):
            diff = cart[i] - cart[j]
            vecs = diff + offsets_cart
            sq = np.sum(vecs ** 2, axis=-1)
            idx = np.argmin(sq)
            d_np[i, j] = np.sqrt(sq[idx])
            o_np[i, j] = offsets_frac[idx].astype(np.int32)

    err_d = np.abs(d_npu - d_np).max()
    err_o = np.abs(o_npu - o_np).max()
    status_d = "PASS" if err_d < 1e-3 else f"FAIL (err={err_d:.2e})"
    status_o = "PASS" if err_o < 1e-3 else f"FAIL (err={err_o:.2e})"
    print(f"  compute_pairwise_pbc_distances (d): {status_d}  (max_err={err_d:.2e})")
    print(f"  compute_pairwise_pbc_distances (o): {status_o}  (max_err={err_o:.2e})")


def test_radius_graph_pbc():
    np.random.seed(42)
    N = 12
    lattice = np.array([
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 8.0],
    ], dtype=np.float32)
    frac = np.random.rand(N, 3).astype(np.float32)

    radius = 4.0
    max_neighbors = 6

    # Run NPU pipeline
    ei_npu, co_npu, nb_npu = radius_graph_pbc(
        frac, None, None, N, radius, max_neighbors, 'npu', lattice=lattice,
    )

    # Run NumPy reference
    ei_np, co_np, nb_np = radius_graph_pbc_np(frac, lattice, radius, max_neighbors)

    status = "PASS"
    for label, a, b in [("edge_index", ei_npu, ei_np), ("cell_offsets", co_npu, co_np)]:
        if a.shape != b.shape:
            print(f"  {label}: FAIL (shape mismatch {a.shape} vs {b.shape})")
            status = "FAIL"
        elif not np.allclose(a, b):
            err = np.abs(a.astype(np.float64) - b.astype(np.float64)).max()
            print(f"  {label}: FAIL (max_err={err:.2e})")
            status = "FAIL"
    if not np.allclose(nb_npu, nb_np):
        print(f"  num_bonds: FAIL ({nb_npu} vs {nb_np})")
        status = "FAIL"
    if status == "PASS":
        print(f"  radius_graph_pbc: PASS  (E={ei_npu.shape[1]})")


def main():
    print("=" * 50)
    print("PyPTO Graph Ops — Tests")
    print("=" * 50)
    test_compute_pairwise_pbc_distances()
    test_radius_graph_pbc()
    print("=" * 50)


if __name__ == "__main__":
    main()
