"""
pmconv/spline_support.py — SplineSupport (S): tensor-valued kernel support structure.

Architectural reference
-----------------------
The overall pipeline (basis → weighting → aggregation) is inherited from
rusty1s/pytorch_spline_conv, which itself follows the SplineCNN formulation
from Fey et al. (CVPR 2018).

What S is (and how it differs from radial kernels)
---------------------------------------------------
In a *radial* kernel the filter depends only on a scalar distance
||u|| — the kernel collapses the pseudo-coordinate u ∈ [0,1]^d to a single
number and weights channels independently.

S is the opposite: it is the *tensor-valued* kernel support — the multi-index
domain Q = {0,…,k₁-1} × … × {0,…,k_d-1} over which the learnable weight
tensor W[q, Cin, Cout] is indexed.  For each q ∈ S the associated weight
W[q] ∈ ℝ^{Cin × Cout} is a full linear map from input to output feature
space.  The resulting kernel

    K_S(u)[cin, cout] = Σ_{q ∈ S}  W[q, cin, cout] · Φ_q(u)

is a continuous Cin × Cout-matrix-valued function of u, not a scalar
function.  This is a strictly richer parameterisation than any radial kernel.

Where S enters the pipeline
---------------------------
1.  spline_basis(edge_attr, S) evaluates Φ_q(u_e) for every edge e and
    every q ∈ S, yielding a dense [E, Q] basis matrix.
2.  spline_weighting(x_src, W, phi) contracts basis evaluations with W over S:
        K_S(u_e)[cin, cout] = Σ_q W[q, cin, cout] · Φ_q(u_e)
        msg[e, cout]        = Σ_cin x_src[e, cin] · K_S(u_e)[cin, cout]
"""

from __future__ import annotations

from typing import List

import torch

import pmconv_ext  # C++ extension (provides multi_index_eval_map)


class SplineSupport:
    """
    Support structure S for the tensor-valued SplineCNN kernel.

    S encapsulates:
      - The multi-index domain Q = {0,…,k₁-1} × … × {0,…,k_d-1}
      - The B-spline basis parameters (degree, k_per_dim)
      - A precomputed multi-index table  multi_index [Q, d]

    The learnable kernel weights W[q, Cin, Cout] are indexed over S:
    for each q ∈ Q the slice W[q] ∈ ℝ^{Cin × Cout} defines a tensor-valued
    linear map from input to output features.

    This mirrors the ``kernel_size`` and ``is_open_spline`` parameters of
    pytorch_spline_conv, but makes the support structure S an explicit,
    reusable first-class object rather than bare scalars.

    Parameters
    ----------
    k_per_dim : list[int]  — number of B-spline basis functions per dimension
    degree    : int        — polynomial degree  (1 = linear, 3 = cubic, …)
    """

    def __init__(self, k_per_dim: List[int], degree: int = 1) -> None:
        if not k_per_dim:
            raise ValueError("k_per_dim must be non-empty")
        if degree < 1:
            raise ValueError(f"degree must be >= 1, got {degree}")
        for i, k in enumerate(k_per_dim):
            if k < degree + 1:
                raise ValueError(
                    f"k_per_dim[{i}] = {k} must be >= degree+1 = {degree + 1}")

        self.k_per_dim: List[int] = list(k_per_dim)
        self.degree: int = degree

        # Precomputed multi-index table: q → (q₁, …, q_d)
        # Shape [Q, d], dtype int64.  This is the explicit representation of S.
        self._multi_index: torch.Tensor = pmconv_ext.multi_index_eval_map(
            k_per_dim)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def Q(self) -> int:
        """Total number of basis functions = prod(k_per_dim)."""
        return int(self._multi_index.shape[0])

    @property
    def d(self) -> int:
        """Number of pseudo-coordinate dimensions."""
        return len(self.k_per_dim)

    @property
    def multi_index(self) -> torch.Tensor:
        """Multi-index table [Q, d], dtype int64."""
        return self._multi_index

    def __repr__(self) -> str:
        return (
            f"SplineSupport(k_per_dim={self.k_per_dim}, "
            f"degree={self.degree}, Q={self.Q})"
        )
