"""
DAO: Siamese Foundation Models for Crystal Structure Prediction
--- PyPTO NPU kernel implementations of key operators.

Port of https://github.com/GLAD-RUC/DAO to Ascend NPU via PyPTO.
"""

from .lattice_ops import (
    lattice_params_to_matrix,
    frac_to_cart_coords,
    cart_to_frac_coords,
    compute_volume,
)
from .pbc_ops import (
    get_pbc_distances,
    min_distance_sqr_pbc,
)
from .graph_ops import (
    radius_graph_pbc,
    compute_pairwise_pbc_distances,
)
from .basis_ops import (
    SinusoidsEmbedding,
    BesselBasisLayer,
    SphericalBasisLayer,
)
from .attention_ops import (
    CrysFormerAttention,
    TransformerBlock,
)
from .diffusion_ops import (
    SinusoidalTimeEmbeddings,
    BetaScheduler,
    SigmaScheduler,
    d_log_p_wrapped_normal,
    add_diffusion_noise,
)
