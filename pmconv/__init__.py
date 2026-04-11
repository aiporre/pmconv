"""
pmconv — tensor-valued SplineCNN convolution for aiporre/pmconv.

Architecture
------------
Mirrors ``rusty1s/pytorch_spline_conv`` at the Python module level:

    spline_basis     →  spline_weighting  →  scatter-add  →  normalize
         (basis.py)          (weighting.py)        (conv.py)

The key extension is replacing the radial (scalar) kernel with a
tensor-valued kernel K_S indexed over the support structure S:

    K_S(u)[cin, cout] = Σ_{q ∈ S}  W[q, cin, cout] · Φ_q(u)

See TENSOR_KERNEL.md for the full design note.

Public API
----------
SplineSupport  : support/domain structure S  (first-class object)
spline_basis   : basis evaluations  Φ_q(u_e)           [E, Q]
spline_weighting: tensor kernel application  msg[e,cout] [E, Cout]
spline_conv    : full convolution  y_v^{cout}            [N, Cout]
GraphInput     : graph data container (C++ class via pmconv_ext)

Low-level ops (via ``torch.ops.pmconv``)
-----------------------------------------
multi_index_eval_map, basis_tensor_product, kernel_g_cin_cout,
spline_convolution — all registered with TORCH_LIBRARY in pmconv_ext.
"""

import pmconv_ext  # registers torch.ops.pmconv.*  # noqa: F401

from pmconv_ext import GraphInput  # C++ class via pybind11

from .spline_support import SplineSupport
from .basis import spline_basis
from .weighting import spline_weighting
from .conv import spline_conv

__all__ = [
    "GraphInput",
    "SplineSupport",
    "spline_basis",
    "spline_weighting",
    "spline_conv",
]
