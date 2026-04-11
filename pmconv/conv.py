"""
pmconv/conv.py — Tensor-valued SplineCNN convolution operator.

Architectural reference
-----------------------
Mirrors pytorch_spline_conv's ``conv.py`` (rusty1s/pytorch_spline_conv).
The reference pipeline is::

    basis, weight_index = spline_basis(pseudo, ...)
    out = spline_weighting(x[col], weight, basis, weight_index)
    out = scatter_add(out, row, ...)   # aggregate to destination nodes
    out = out / degree                 # optional normalization

What changed in the tensor-kernel redesign
------------------------------------------
The support S is passed as an explicit ``SplineSupport`` object rather than
bare scalars (kernel_size, is_open_spline, degree).  The kernel K_S is
tensor-valued (see weighting.py).  Everything else — basis evaluation,
aggregation, normalization — is structurally identical to the reference.

Forward equation::

    y_v^{cout} = (1/|N(v)|) Σ_{w ∈ N(v)} Σ_{cin}
                   x_w^{cin} · K_S(u(v,w))[cin, cout]

    K_S(u)[cin, cout] = Σ_{q ∈ S}  W[q, cin, cout] · Φ_q(u)

Backward (autograd handles everything automatically):
    ∂L/∂W[q,cin,cout] ∝ Σ_{e=(w→v)}  Φ_q(u_e) · x_w^{cin} · ∂L/∂y_v^{cout}
    ∂L/∂x_w^{cin}     ∝ Σ_v Σ_{cout} K_S(u(v,w))[cin,cout] · ∂L/∂y_v^{cout}
"""

from __future__ import annotations

import torch

from .spline_support import SplineSupport
from .basis import spline_basis
from .weighting import spline_weighting


def spline_conv(
    graph,               # GraphInput (edge_index [2,E], edge_attr [E,d], x [N,Cin])
    weight: torch.Tensor,  # [Q, Cin, Cout]
    S: SplineSupport,
    norm: bool = True,
    root_weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Tensor-valued SplineCNN convolution.

    Inherits from pytorch_spline_conv
    ----------------------------------
    * basis → weighting → scatter-add → normalize pipeline.
    * Optional ``root_weight`` (self-loop) and ``bias`` terms, matching the
      reference ``spline_conv`` signature.
    * Fully differentiable w.r.t. ``weight``, ``x``, ``root_weight``, ``bias``.

    Tensor-kernel extension
    -----------------------
    The support S is a first-class object that bundles the basis parameters.
    The kernel K_S(u_e) ∈ ℝ^{Cin × Cout} is tensor-valued over S (see
    weighting.py).  No radial collapse: all input-output channel interactions
    are modelled jointly per edge.

    Parameters
    ----------
    graph       : GraphInput — (edge_index [2,E], edge_attr [E,d], x [N,Cin])
    weight      : Tensor [Q, Cin, Cout]  — learnable tensor-valued kernel
                  weights indexed over support S
    S           : SplineSupport          — support/domain structure
    norm        : bool                   — divide output by node in-degree
                  (default True); isolated nodes yield zero
    root_weight : Tensor [Cin, Cout]     — optional self-loop weight (not
                  used in neighbourhood aggregation); added as  x @ root_weight
    bias        : Tensor [Cout]          — optional output bias

    Returns
    -------
    y : Tensor [N, Cout]
    """
    graph.validate()

    edge_index = graph.edge_index   # [2, E]
    edge_attr  = graph.edge_attr    # [E, d]
    x          = graph.x            # [N, Cin]

    N    = graph.num_nodes()
    E    = graph.num_edges()
    Cout = weight.size(2)

    row = edge_index[0]   # source (sender)   [E]
    col = edge_index[1]   # destination (receiver) [E]

    # -----------------------------------------------------------------------
    # Stage 1 — Basis evaluation
    #   Inherits from pytorch_spline_conv: spline_basis returns the basis
    #   matrix and multi-index.  Here we use the full dense [E, Q] form.
    # -----------------------------------------------------------------------
    basis, _ = spline_basis(edge_attr, S)  # [E, Q]

    # -----------------------------------------------------------------------
    # Stage 2 — Gather source features and apply tensor-valued kernel
    #   Mirrors pytorch_spline_conv: x[source] selects source node features;
    #   spline_weighting computes the per-edge output messages.
    #   Note: our edge_index[0] = source, edge_index[1] = destination
    #   (opposite naming to pytorch_spline_conv which uses row/col inversely).
    # -----------------------------------------------------------------------
    x_src = x[row]                                  # [E, Cin]  gather source features
    out   = spline_weighting(x_src, weight, basis)  # [E, Cout]

    # -----------------------------------------------------------------------
    # Stage 3 — Scatter-add messages to destination nodes
    #   Identical to pytorch_spline_conv's scatter_add_ step.
    # -----------------------------------------------------------------------
    col_exp = col.unsqueeze(-1).expand_as(out)
    y = x.new_zeros(N, Cout).scatter_add_(0, col_exp, out)

    # -----------------------------------------------------------------------
    # Stage 4 — Normalize by node in-degree  (same as pytorch_spline_conv)
    # -----------------------------------------------------------------------
    if norm:
        ones = torch.ones(E, dtype=x.dtype, device=x.device)
        deg  = y.new_zeros(N).scatter_add_(0, col, ones)
        y    = y / deg.unsqueeze(-1).clamp_(min=1)

    # -----------------------------------------------------------------------
    # Optional self-loop weight  (matches pytorch_spline_conv interface)
    # -----------------------------------------------------------------------
    if root_weight is not None:
        y = y + x @ root_weight

    # Optional bias
    if bias is not None:
        y = y + bias

    return y  # [N, Cout]
