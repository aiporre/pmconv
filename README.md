# pmconv

Tensor-valued SplineCNN-style message-passing convolution for graphs — implemented in C++ with PyTorch / ATen and exposed to Python via **pybind11**.

The kernel follows `rusty1s/pytorch_spline_conv`'s clean basis/weighting/convolution decomposition but replaces the per-channel scalar kernel with a **tensor-valued kernel** K_S(u) ∈ ℝ^{Cin × Cout} indexed over the support structure S.  See [TENSOR_KERNEL.md](TENSOR_KERNEL.md) for the full design note.

## Overview

### High-level Python API  (`pmconv` package)

| Class / function | Description |
|---|---|
| `SplineSupport(k_per_dim, degree)` | Support structure S — kernel domain / multi-index set Q |
| `spline_basis(edge_attr, S)` | B-spline basis evaluation → `(phi [E, Q], multi_index [Q, d])` |
| `spline_weighting(x_src, weight, basis)` | Tensor-valued kernel application → `[E, Cout]` |
| `spline_conv(graph, weight, S, ...)` | Full graph convolution → `[N, Cout]` |
| `GraphInput(edge_index, edge_attr, x)` | Graph data container with validation |

### Low-level C++ / `torch.ops` API (`pmconv_ext` extension)

| Op | `torch.ops` / C++ signature | Output shape |
|---|---|---|
| `multi_index_eval_map` | `(k_per_dim)` | `[Q, d]` |
| `basis_tensor_product` | `(edge_attr, degree, k_per_dim)` | `[E, Q]` |
| `kernel_g_cin_cout` | `(phi, W)` | `[E, Cin, Cout]` |
| `spline_convolution` | `(graph, W, degree, k_per_dim)` | `[N, Cout]` |

`Q = k_1 × … × k_d`, `E` = number of edges, `N` = number of nodes.

### Mathematical formulation

```
Φ_q(u_e) = ∏_{i=1}^{d}  B_{q_i, p}(u_e^i)                 tensor-product B-spline basis

K_S(u_e)[cin, cout] = Σ_{q ∈ S}  W[q, cin, cout] · Φ_q(u_e)  tensor-valued kernel (Cin × Cout)

y_v^{cout} = (1/|N(v)|) Σ_{w∈N(v)} Σ_{cin}
               x_w^{cin} · K_S(u(v,w))[cin, cout]            node aggregation
```

B-splines used: **uniform clamped (open)** B-splines of degree `p` via the Cox–de Boor recurrence.

---

## Installation

```bash
pip install torch          # PyTorch ≥ 2.0 required
pip install .              # builds pmconv_ext.so via setup.py
```

For an editable / development install:
```bash
pip install -e .
```

---

## Quick start — high-level Python API

```python
import torch
from pmconv import GraphInput, SplineSupport, spline_conv

# Graph: 5 nodes, 8 directed edges, 2D pseudo-coordinates, 3 input features
N, E, d, Cin, Cout = 5, 8, 2, 3, 4

edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
edge_attr  = torch.rand(E, d)   # pseudo-coordinates in [0,1]^d
x          = torch.rand(N, Cin) # node features

graph = GraphInput(edge_index, edge_attr, x)

# Support S: 4 basis functions per dimension → Q = 4² = 16
S = SplineSupport(k_per_dim=[4, 4], degree=1)

# Learnable tensor-valued kernel weights W[q, Cin, Cout]
W = torch.rand(S.Q, Cin, Cout, requires_grad=True)

# Forward pass
y = spline_conv(graph, W, S, norm=True)  # [N, Cout] = [5, 4]
print(y.shape)  # torch.Size([5, 4])

# Backward pass (autograd through basis + einsum + scatter)
y.sum().backward()
print(W.grad.shape)  # torch.Size([16, 3, 4])
```

### Using individual stages (mirrors pytorch_spline_conv module split)

```python
from pmconv import GraphInput, SplineSupport, spline_basis, spline_weighting

S = SplineSupport(k_per_dim=[4, 4], degree=1)

# Stage 1: basis evaluation [E, Q] — mirrors pytorch_spline_conv basis.py
phi, multi_index = spline_basis(graph.edge_attr, S)
# phi.shape = [8, 16],  multi_index.shape = [16, 2]

# Stage 2: tensor-valued kernel application [E, Cout]
x_src = graph.x[graph.edge_index[0]]       # source features [E, Cin]
msg   = spline_weighting(x_src, W, phi)    # [E, Cout]
```

---

## Low-level API (`pmconv_ext`)

```python
import torch
from pmconv_ext import GraphInput, spline_convolution

N, E, d, Cin, Cout = 5, 8, 2, 3, 4
k, degree = 4, 1

edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
edge_attr  = torch.rand(E, d)
x          = torch.rand(N, Cin)

graph = GraphInput(edge_index, edge_attr, x)
W = torch.rand(k**d, Cin, Cout, requires_grad=True)

y = spline_convolution(graph, W, degree=degree, k_per_dim=[k]*d)
print(y.shape)  # [5, 4]
```

### Via `torch.ops`

```python
import torch
import pmconv_ext  # registers torch.ops.pmconv.*

idx = torch.ops.pmconv.multi_index_eval_map([4, 4])
phi = torch.ops.pmconv.basis_tensor_product(edge_attr, 1, [4, 4])
g   = torch.ops.pmconv.kernel_g_cin_cout(phi, W)
y   = torch.ops.pmconv.spline_convolution(
        edge_index, edge_attr, x, W, 1, [4, 4], True)
```

---

## Tensor shapes reference

```
edge_index : [2, E]          int64   – row 0 = source, row 1 = target
edge_attr  : [E, d]          float   – pseudo-coordinates u(e) ∈ [0,1]^d
x          : [N, Cin]        float   – node feature matrix
W          : [Q, Cin, Cout]  float   – learnable tensor-valued kernel weights
phi        : [E, Q]          float   – basis evaluations Φ_q(u_e)
g          : [E, Cin, Cout]  float   – per-edge kernel matrices K_S(u_e)
y          : [N, Cout]       float   – convolution output
```

---

## Running tests

```bash
LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH \
  python -m pytest tests/test_pmconv.py -v
```

---

## Source layout

```
pmconv/
  __init__.py        – package init; re-exports SplineSupport, spline_conv, …
  spline_support.py  – SplineSupport class (the S object / kernel domain)
  basis.py           – spline_basis() — mirrors pytorch_spline_conv basis.py
  weighting.py       – spline_weighting() — tensor-valued kernel application
  conv.py            – spline_conv() — full graph convolution
src/
  spline_conv.h      – public C++ API (GraphInput + 4 op declarations)
  spline_conv.cpp    – implementation (B-spline basis + all ops)
  pmconv_bindings.cpp– TORCH_LIBRARY registration + pybind11 module
  CMakeLists.txt     – CMake targets
setup.py             – Python extension build
tests/
  test_pmconv.py     – unit tests covering all ops and the Python API
TENSOR_KERNEL.md     – technical note: design, equations, mapping from reference
```
