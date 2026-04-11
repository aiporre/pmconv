"""
pmconv/basis.py — Spline basis evaluation.

Architectural reference
-----------------------
Mirrors pytorch_spline_conv's ``basis.py`` module (rusty1s/pytorch_spline_conv).
The reference module calls a C++/CUDA op that returns *sparse* basis values
and weight indices: only the (degree+1)^d non-zero terms per edge.

What changed in the tensor-kernel redesign
------------------------------------------
We return a *dense* [E, Q] basis matrix rather than the sparse
(basis_values, weight_index) pair from the reference.  Dense evaluation
simplifies the tensor contraction in spline_weighting and keeps the
implementation transparent, at the cost of Q − (degree+1)^d extra zeros.
Sparse evaluation is a straightforward future optimisation that can be
dropped in without changing the API contract.

Where S enters
--------------
S supplies the basis parameters (degree, k_per_dim) and the precomputed
multi-index table.  spline_basis delegates the numeric evaluation to the
C++ extension (``pmconv_ext.basis_tensor_product``) and returns the
multi-index alongside the basis matrix so callers can index into W correctly.
"""

from __future__ import annotations

from typing import Tuple

import torch

import pmconv_ext  # C++ extension

from .spline_support import SplineSupport


def spline_basis(
    edge_attr: torch.Tensor,   # [E, d]  pseudo-coordinates in [0, 1]^d
    S: SplineSupport,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the tensor-product B-spline basis over support S for each edge.

    Inherits from pytorch_spline_conv
    ----------------------------------
    * Uniform clamped (open) B-splines, Cox–de Boor recurrence.
    * Partition-of-unity guarantee: Σ_q Φ_q(u) = 1 for every u.
    * Basis parameters (degree, k_per_dim) are encapsulated in S.

    Tensor-kernel extension
    -----------------------
    The returned dense [E, Q] matrix is the natural representation for the
    tensor-valued kernel K_S(u)[cin, cout] = Σ_q W[q, cin, cout] · Φ_q(u).
    Each column q of phi corresponds to one element of the support S.

    Parameters
    ----------
    edge_attr : Tensor [E, d]  — pseudo-coordinates u(e) ∈ [0, 1]^d
    S         : SplineSupport  — support/domain structure (basis params + Q)

    Returns
    -------
    phi         : Tensor [E, Q]  float — basis evaluations  Φ_q(u_e)
    multi_index : Tensor [Q, d]  int64 — multi-index table q → (q₁,…,q_d)
    """
    phi = pmconv_ext.basis_tensor_product(edge_attr, S.degree, S.k_per_dim)
    return phi, S.multi_index
