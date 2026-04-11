"""
pmconv/weighting.py — Tensor-valued kernel weighting.

Architectural reference
-----------------------
Mirrors pytorch_spline_conv's ``weighting.py`` module (rusty1s/pytorch_spline_conv).
The reference function signature is::

    spline_weighting(x, weight, basis, weight_index) -> Tensor [E, Cout]

where ``basis`` and ``weight_index`` are the sparse outputs of spline_basis
(only the (degree+1)^d non-zero entries per edge).

What changed in the tensor-kernel redesign
------------------------------------------
Mathematical formulation
~~~~~~~~~~~~~~~~~~~~~~~~
Radial kernel (NOT used here)::

    g_l(u_e) = f(||u_e||)           scalar, isotropic, channel-independent

Tensor-valued kernel K_S (this implementation)::

    K_S(u_e)[cin, cout] = Σ_{q ∈ S}  W[q, cin, cout] · Φ_q(u_e)

    msg[e, cout] = Σ_cin  x_src[e, cin] · K_S(u_e)[cin, cout]

K_S(u_e) is a full Cin × Cout matrix for each edge — it mixes all input
channels into all output channels simultaneously, parameterised by the
tensor-valued weights W[q] ∈ ℝ^{Cin × Cout} that live over the support S.

Where S enters
--------------
The basis evaluations phi [E, Q] carry the support structure implicitly:
each column q corresponds to one element of S.  The contraction
``einsum("eq,qcd->ecd", phi, W)`` sums over S.

Implementation strategy
-----------------------
* Step 1: assemble the per-edge kernel matrix K_S(u_e) via einsum (or
  the C++ helper ``pmconv_ext.kernel_g_cin_cout``).
* Step 2: apply it to source features via a batched matrix product.
Both steps are fully differentiable through PyTorch autograd.

Backward dependencies
---------------------
∂L/∂W[q, cin, cout] = Σ_e  phi[e, q] · x_src[e, cin] · ∂L/∂msg[e, cout]
∂L/∂x_src[e, cin]   = Σ_{cout}  K_S(u_e)[cin, cout] · ∂L/∂msg[e, cout]
∂L/∂phi[e, q]       = Σ_{cin, cout}  W[q, cin, cout] · x_src[e, cin]
                                     · ∂L/∂msg[e, cout]

All computed automatically by PyTorch autograd.
"""

from __future__ import annotations

import torch

import pmconv_ext  # C++ extension (provides kernel_g_cin_cout)


def spline_weighting(
    x_src: torch.Tensor,   # [E, Cin]
    weight: torch.Tensor,  # [Q, Cin, Cout]
    basis: torch.Tensor,   # [E, Q]
) -> torch.Tensor:
    """
    Apply the tensor-valued kernel K_S to per-edge source features.

    Inherits from pytorch_spline_conv
    ----------------------------------
    * Two-stage pipeline: assemble per-edge kernel, then apply to features.
    * Output shape [E, Cout], later scatter-added in spline_conv.

    Tensor-kernel extension
    -----------------------
    Unlike the reference weighting (which uses sparse basis + weight_index),
    we use a dense [E, Q] basis and a full einsum contraction.  The weight
    tensor W[q, cin, cout] is genuinely tensor-valued: each q ∈ S indexes a
    Cin × Cout linear map, not a scalar coefficient per channel.

    Parameters
    ----------
    x_src  : Tensor [E, Cin]         — source node features for each edge
    weight : Tensor [Q, Cin, Cout]   — learnable tensor-valued kernel weights
                                       indexed over support S
    basis  : Tensor [E, Q]           — basis evaluations Φ_q(u_e) from
                                       spline_basis()

    Returns
    -------
    msg : Tensor [E, Cout]  — per-edge messages (differentiable w.r.t. all inputs)
    """
    # Step 1: Assemble per-edge kernel matrix K_S(u_e) ∈ ℝ^{Cin × Cout}
    #   g[e, cin, cout] = Σ_q  basis[e, q] · weight[q, cin, cout]
    #   Delegated to C++ helper (same einsum, pre-validated).
    g = pmconv_ext.kernel_g_cin_cout(basis, weight)  # [E, Cin, Cout]

    # Step 2: Apply tensor-valued kernel to source features
    #   msg[e, cout] = Σ_cin  x_src[e, cin] · g[e, cin, cout]
    #   Batched matrix-vector product: (E, 1, Cin) @ (E, Cin, Cout) → (E, 1, Cout)
    msg = torch.bmm(x_src.unsqueeze(1), g).squeeze(1)  # [E, Cout]
    return msg
