# Tensor-Valued Kernel Design Note

## Overview

This document describes the design of the tensor-valued SplineCNN kernel in
`aiporre/pmconv`, explaining:

1. What was **inherited** from `rusty1s/pytorch_spline_conv`.
2. What **changed** in the tensor-kernel redesign.
3. Where the support object **S** enters the basis/kernel pipeline.
4. How the new kernel **differs mathematically** from a radial version.
5. **Forward equation and backward dependencies**.
6. A **minimal usage example** with dummy data.

---

## 1. Mapping from pytorch_spline_conv → pmconv tensor kernel

| Concept | rusty1s/pytorch_spline_conv | pmconv (this repo) |
|---|---|---|
| Basis evaluation | `spline_basis(pseudo, kernel_size, is_open_spline, degree)` → `(basis [E,K], weight_index [E,K])` *sparse* | `spline_basis(edge_attr, S)` → `(phi [E,Q], multi_index [Q,d])` *dense* |
| Kernel / weighting | `spline_weighting(x, weight, basis, weight_index)` → `[E, Cout]` | `spline_weighting(x_src, weight, basis)` → `[E, Cout]` |
| Full convolution | `spline_conv(x, edge_index, pseudo, weight, ...)` → `[N, Cout]` | `spline_conv(graph, weight, S, ...)` → `[N, Cout]` |
| Basis parameters | bare scalars (`kernel_size`, `is_open_spline`, `degree`) | `SplineSupport(k_per_dim, degree)` — first-class support object S |
| Kernel type | Channel-wise weighted sum (per-channel scalar kernel) | Tensor-valued `K_S(u) ∈ ℝ^{Cin × Cout}` — full channel mixing |
| Basis representation | Sparse: only `(degree+1)^d` non-zero terms per edge | Dense: all `Q = ∏ k_i` terms (simpler, future sparse opt. possible) |
| Graph container | raw `x, edge_index, pseudo` tensors | `GraphInput(edge_index, edge_attr, x)` with validation |

---

## 2. The Support Structure S

**S** (`SplineSupport`) is the kernel domain — the set of multi-indices

```
S  =  Q  =  {0,…,k₁-1} × … × {0,…,k_d-1}
```

over which the learnable weight tensor `W[q, Cin, Cout]` is indexed.

S is NOT a scalar radius or radial kernel.  It is a *tensor-valued* domain:
for each `q ∈ S` the weight `W[q] ∈ ℝ^{Cin × Cout}` is a full linear map
from input to output feature space.

```python
from pmconv import SplineSupport
S = SplineSupport(k_per_dim=[4, 4], degree=1)
# S.Q = 16, S.d = 2, S.multi_index.shape = (16, 2)
```

---

## 3. Forward Equation

```
Φ_q(u_e)  =  ∏_{i=1}^{d}  B_{q_i, p}(u_e^i)      tensor-product B-spline basis

K_S(u_e)[cin, cout]  =  Σ_{q ∈ S}  W[q, cin, cout] · Φ_q(u_e)
                                                      tensor-valued kernel (Cin × Cout)

y_v^{cout}  =  (1/|N(v)|)  Σ_{w ∈ N(v)}  Σ_{cin}  x_w^{cin} · K_S(u(v,w))[cin, cout]
                                                      node-wise aggregation
```

Equivalently in tensor notation:

```
phi        [E, Q]            =  basis_tensor_product(edge_attr, S)
g          [E, Cin, Cout]    =  einsum("eq,qcd->ecd", phi, W)
msg        [E, Cout]         =  bmm(x_src[E,1,Cin], g[E,Cin,Cout]).squeeze(1)
y          [N, Cout]         =  scatter_add(msg, dst) / deg
```

---

## 4. How the Tensor Kernel Differs from a Radial Kernel

| Property | Radial kernel | Tensor kernel K_S |
|---|---|---|
| Domain | scalar ‖u‖ ∈ ℝ | u ∈ [0,1]^d, full tensor-product domain |
| Isotropy | isotropic (rotation-invariant) | anisotropic (direction-aware) |
| Channel coupling | independent per channel | joint Cin × Cout linear map |
| Parameterisation | scalar coefficients per basis function | tensor W[q] ∈ ℝ^{Cin × Cout} per basis function |
| Expressivity | least (scalar output per channel) | most (full channel mixing per edge) |

The tensor kernel K_S collapses to the per-channel formulation of the reference
only if W is constrained to be diagonal in the Cin × Cout slice, i.e.
W[q, cin, cout] = 0 for cin ≠ cout.  The general (unconstrained) form strictly
generalises the reference.

---

## 5. Backward Dependencies

PyTorch autograd differentiates through the entire pipeline automatically.
The key gradient expressions are:

```
∂L/∂W[q, cin, cout]  =  Σ_e  phi[e,q] · x_src[e,cin] · ∂L/∂msg[e,cout]
∂L/∂x_src[e, cin]    =  Σ_cout  K_S(u_e)[cin,cout] · ∂L/∂msg[e,cout]
∂L/∂phi[e, q]        =  Σ_{cin,cout}  W[q,cin,cout] · x_src[e,cin] · ∂L/∂msg[e,cout]
```

CUDA compatibility: the implementation uses only standard ATen/torch ops
(einsum, bmm, index_add_, scatter_add_) and is therefore CUDA-compatible
without any additional kernel code.  Move tensors to GPU before calling.

---

## 6. Minimal Usage Example

```python
import torch
from pmconv import GraphInput, SplineSupport, spline_conv

# Graph: 5 nodes, 8 directed edges, 2D pseudo-coordinates, 3 input features
N, E, d, Cin, Cout = 5, 8, 2, 3, 4

edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
edge_attr  = torch.rand(E, d)          # pseudo-coordinates in [0, 1]^d
x          = torch.rand(N, Cin)        # node features

graph = GraphInput(edge_index, edge_attr, x)

# Support S: 4 basis functions per dimension, linear B-splines → Q = 4² = 16
S = SplineSupport(k_per_dim=[4, 4], degree=1)

# Learnable tensor-valued kernel weights: W[q, Cin, Cout]
W = torch.rand(S.Q, Cin, Cout, requires_grad=True)

# Forward
y = spline_conv(graph, W, S, norm=True)   # [N, Cout] = [5, 4]
print(y.shape)  # torch.Size([5, 4])

# Backward
loss = y.sum()
loss.backward()
print(W.grad.shape)  # torch.Size([16, 3, 4])
```

### Using individual stages (mirrors pytorch_spline_conv module split)

```python
from pmconv import GraphInput, SplineSupport, spline_basis, spline_weighting

graph = GraphInput(edge_index, edge_attr, x)
S     = SplineSupport(k_per_dim=[4, 4], degree=1)
W     = torch.rand(S.Q, Cin, Cout)

# Stage 1: basis evaluation
phi, multi_index = spline_basis(graph.edge_attr, S)
# phi.shape         = [E, Q] = [8, 16]
# multi_index.shape = [Q, d] = [16, 2]

# Stage 2: tensor-valued kernel application
x_src = graph.x[graph.edge_index[0]]       # gather source features [E, Cin]
msg   = spline_weighting(x_src, W, phi)    # [E, Cout]

# (scatter-add + normalize handled by spline_conv)
```

### Via torch.ops (low-level, registered ops)

```python
import pmconv_ext  # registers torch.ops.pmconv.*

phi = torch.ops.pmconv.basis_tensor_product(edge_attr, 1, [4, 4])
g   = torch.ops.pmconv.kernel_g_cin_cout(phi, W)
y   = torch.ops.pmconv.spline_convolution(
        edge_index, edge_attr, x, W, 1, [4, 4], True)
```
