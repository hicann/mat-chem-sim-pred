"""
PyPTO implementation of Lennard-Jones Force Operator.

This is a pure-Python Ascend C operator written with PyPTO
(Parallel Tensor/Tile Operation) framework. It compiles to a
single fused NPU kernel — no intermediate kernel launches.

Usage:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    export TILE_FWK_DEVICE_ID=0
    python pypto_lj_force.py
"""

import pypto as pt
import torch
import numpy as np
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# Core LJ computation
# ---------------------------------------------------------------------------

def lj_force_core(
    positions: pt.Tensor,
    epsilon: float,
    sigma: float,
    cutoff: float,
) -> tuple[pt.Tensor, pt.Tensor]:
    """
    Vectorised LJ force & energy from positions [N, 3].

    Returns (forces [N, 3], total_energy scalar [1]).
    """
    N = positions.shape[0]

    # Pairwise displacement vectors: [N, N, 3]
    pos_i = pt.unsqueeze(positions, 1)  # [N, 1, 3]
    pos_j = pt.unsqueeze(positions, 0)  # [1, N, 3]
    diff = pos_i - pos_j                 # [N, N, 3]  broadcast

    # Squared distances: [N, N]
    r2 = pt.sum(diff * diff, dim=-1)

    # ----- cutoff + self-exclusion masks -----
    cutoff_sq = cutoff * cutoff
    # valid pairs: within cutoff AND not exactly overlapping
    valid = (r2 < cutoff_sq) & (r2 > 1e-10)

    # Upper-triangular mask for energy (avoid double counting)
    upper_f = pt.triu(pt.ones([N, N], pt.DT_FP32), diagonal=1)
    energy_mask = pt.cast(upper_f > 0.0, pt.DT_BOOL) & valid

    # Full non-diagonal mask for forces
    idx_1d = pt.arange(N, dtype=pt.DT_INT32)       # [N]
    idx_i = pt.unsqueeze(idx_1d, 1)                # [N, 1]
    idx_j = pt.unsqueeze(idx_1d, 0)                # [1, N]
    self_mask = idx_i == idx_j                     # [N, N]  diagonal
    force_mask = valid & (~self_mask)

    # ----- safe r² (avoid division by zero) -----
    r2_safe = pt.where(valid, r2, pt.ones_like(r2))

    # ----- LJ formula -----
    r2inv = pt.reciprocal(r2_safe)
    r6inv = r2inv * r2inv * r2inv

    sigma6 = sigma ** 6
    sigma12 = sigma6 * sigma6

    s6r6 = sigma6 * r6inv
    s12r12 = s6r6 * s6r6

    # Energy (upper triangular, then no double-count)
    eps4 = 4.0 * epsilon
    potential = eps4 * (s12r12 - s6r6)
    potential_masked = pt.where(energy_mask, potential, pt.zeros_like(potential))
    total_energy = pt.sum(potential_masked)

    # Force scalar
    eps24 = 24.0 * epsilon
    fscalar = eps24 * r2inv * (2.0 * s12r12 - s6r6)
    fscalar_masked = pt.where(force_mask, fscalar, pt.zeros_like(fscalar))

    # Force vectors: fscalar[i,j] * diff[i,j,:]
    fscalar_3d = pt.unsqueeze(fscalar_masked, -1)  # [N, N, 1]
    force_pairs = fscalar_3d * diff                  # [N, N, 3]

    # Sum over j → net force on each atom i
    forces = pt.sum(force_pairs, dim=1)              # [N, 3]

    return forces, total_energy


# ---------------------------------------------------------------------------
# PyPTO kernel (JIT-compiled to NPU)
# ---------------------------------------------------------------------------

@pt.frontend.jit
def lj_force_kernel(
    positions: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    forces_out: pt.Tensor([pt.DYNAMIC, 3], pt.DT_FP32),
    energy_out: pt.Tensor([1], pt.DT_FP32),
    epsilon: pt.Element(pt.DT_FP32),
    sigma: pt.Element(pt.DT_FP32),
    cutoff: pt.Element(pt.DT_FP32),
):
    """
    Single fused kernel for LJ force + energy computation.

    - positions  : [N, 3]  atom coordinates (float32)
    - forces_out : [N, 3]  output forces (float32)
    - energy_out : [1]     output total potential energy (float32)
    """
    N = positions.shape[0]

    # Let PyPTO auto-tile; tune for the NPU's vector unit
    pt.set_vec_tile_shapes(1, 1, 1, 64)

    forces, total_energy = lj_force_core(positions, epsilon, sigma, cutoff)

    forces_out[:] = forces
    energy_out[0] = total_energy


# ---------------------------------------------------------------------------
# Reference (NumPy) for validation
# ---------------------------------------------------------------------------

def lj_force_numpy(positions, epsilon, sigma, cutoff):
    N = positions.shape[0]
    forces = np.zeros_like(positions)
    energy = 0.0
    cutoff_sq = cutoff * cutoff
    sigma6 = sigma ** 6
    sigma12 = sigma6 * sigma6

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if r2 < cutoff_sq and r2 > 1e-10:
                r2inv = 1.0 / r2
                r6inv = r2inv * r2inv * r2inv
                s6r6 = sigma6 * r6inv
                s12r12 = s6r6 * s6r6
                energy += 4.0 * epsilon * (s12r12 - s6r6)
                fscalar = 24.0 * epsilon * r2inv * (2.0 * s12r12 - s6r6)
                forces[i, 0] += fscalar * dx
                forces[i, 1] += fscalar * dy
                forces[i, 2] += fscalar * dz
                forces[j, 0] -= fscalar * dx
                forces[j, 1] -= fscalar * dy
                forces[j, 2] -= fscalar * dz
    return forces, energy


# ---------------------------------------------------------------------------
# Tests  (adapted from existing test_lj_force.py)
# ---------------------------------------------------------------------------

def test_two_atoms():
    sigma, epsilon, cutoff = 3.4, 0.01, 10.0
    positions = torch.tensor([[0.0, 0.0, 0.0], [sigma, 0.0, 0.0]],
                              dtype=torch.float32, device='npu')
    forces = torch.zeros_like(positions)
    energy = torch.zeros(1, dtype=torch.float32, device='npu')

    lj_force_kernel(positions, forces, energy, epsilon, sigma, cutoff)

    forces_cpu = forces.cpu().numpy()
    energy_val = energy.cpu().item()

    ref_f, ref_e = lj_force_numpy(positions.cpu().numpy(), epsilon, sigma, cutoff)

    assert_allclose(forces_cpu, ref_f, atol=1e-6)
    assert abs(energy_val - ref_e) < 1e-6, f"energy mismatch: {energy_val} vs {ref_e}"
    print(f"  two_atoms: energy={energy_val:.6f}  PASS")


def test_20_atoms():
    np.random.seed(42)
    pos_np = np.random.rand(20, 3).astype(np.float32) * 15.0
    sigma, epsilon, cutoff = 3.4, 0.01, 10.0

    positions = torch.from_numpy(pos_np).to('npu')
    forces = torch.zeros(20, 3, dtype=torch.float32, device='npu')
    energy = torch.zeros(1, dtype=torch.float32, device='npu')

    lj_force_kernel(positions, forces, energy, epsilon, sigma, cutoff)

    forces_npu = forces.cpu().numpy()
    energy_npu = energy.cpu().item()
    forces_ref, energy_ref = lj_force_numpy(pos_np, epsilon, sigma, cutoff)

    force_err = np.abs(forces_npu - forces_ref).max()
    energy_err = abs(energy_npu - energy_ref)
    force_rel = force_err / (np.abs(forces_ref).max() + 1e-10)
    energy_rel = energy_err / (abs(energy_ref) + 1e-10)

    status = "PASS" if force_rel < 0.01 and energy_rel < 0.01 else "FAIL"
    print(f"  20 atoms: force_rel={force_rel:.2e}  energy_rel={energy_rel:.2e}  [{status}]")


def test_newton_third_law():
    np.random.seed(42)
    pos_np = np.random.rand(50, 3).astype(np.float32) * 10.0
    sigma, epsilon, cutoff = 3.4, 0.01, 10.0

    positions = torch.from_numpy(pos_np).to('npu')
    forces = torch.zeros(50, 3, dtype=torch.float32, device='npu')
    energy = torch.zeros(1, dtype=torch.float32, device='npu')

    lj_force_kernel(positions, forces, energy, epsilon, sigma, cutoff)

    total_f = forces.cpu().sum(dim=0).numpy()
    status = "PASS" if np.allclose(total_f, 0.0, atol=1e-4) else "FAIL"
    print(f"  newton_3rd: total_force={total_f}  [{status}]")


def test_cutoff():
    sigma, epsilon, cutoff = 3.4, 0.01, 5.0
    positions = torch.tensor([[0.0, 0.0, 0.0], [cutoff + 1.0, 0.0, 0.0]],
                              dtype=torch.float32, device='npu')
    forces = torch.zeros_like(positions)
    energy = torch.zeros(1, dtype=torch.float32, device='npu')

    lj_force_kernel(positions, forces, energy, epsilon, sigma, cutoff)

    f = forces.cpu().numpy()
    e = energy.cpu().item()
    status = "PASS" if np.allclose(f, 0.0, atol=1e-10) and abs(e) < 1e-10 else "FAIL"
    print(f"  cutoff: forces={f.ravel()}  energy={e}  [{status}]")


def test_larger_system():
    N = 128
    np.random.seed(42)
    pos_np = np.random.rand(N, 3).astype(np.float32) * 15.0
    sigma, epsilon, cutoff = 3.4, 0.01, 10.0

    positions = torch.from_numpy(pos_np).to('npu')
    forces = torch.zeros(N, 3, dtype=torch.float32, device='npu')
    energy = torch.zeros(1, dtype=torch.float32, device='npu')

    lj_force_kernel(positions, forces, energy, epsilon, sigma, cutoff)

    forces_npu = forces.cpu().numpy()
    energy_npu = energy.cpu().item()
    forces_ref, energy_ref = lj_force_numpy(pos_np, epsilon, sigma, cutoff)

    force_err = np.abs(forces_npu - forces_ref).max()
    energy_err = abs(energy_npu - energy_ref)
    tot_f = forces_npu.sum(axis=0)
    print(f"  128 atoms: max_force_err={force_err:.2e}  energy_err={energy_err:.2e}")
    print(f"    total_force={tot_f}  newton_ok={np.allclose(tot_f, 0, atol=1e-3)}")


def main():
    print("=" * 55)
    print("PyPTO LJ Force Operator — Tests")
    print("=" * 55)

    device_count = torch.npu.device_count() if hasattr(torch, 'npu') else 0
    print(f"NPU devices available: {device_count}")

    test_two_atoms()
    test_cutoff()
    test_newton_third_law()
    test_20_atoms()
    test_larger_system()

    print("=" * 55)
    print("All tests completed.")
    print("=" * 55)


if __name__ == "__main__":
    main()
