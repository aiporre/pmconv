# pmconv

SplineCNN-style message-passing convolution for graphs — implemented in C++ with PyTorch / ATen and exposed to Python via **pybind11**.

## Overview

This repository provides four core tensor operations for SplineCNN-style message passing on graphs, following the continuous B-spline kernel formulation from [Fey et al., SplineCNN, CVPR 2018](https://arxiv.org/abs/1711.08920).

### Tensor operations

| Op | C++ / `torch.ops` signature | Output shape |
|---|---|---|
| `multi_index_eval_map` | `(k_per_dim)` | `[Q, d]` |
| `basis_tensor_product` | `(edge_attr, degree, k_per_dim)` | `[E, Q]` |
| `kernel_g_cin_cout` | `(phi, W)` | `[E, Cin, Cout]` |
| `spline_convolution` | `(graph, W, degree, k_per_dim)` | `[N, Cout]` |

`Q = k_1 × … × k_d`, `E` = number of edges, `N` = number of nodes.

### Mathematical formulation

```
Φ_q(u_e) = ∏_{i=1}^{d}  B_{q_i, p}(u_e^i)           # tensor-product basis

g_{cin,cout}(u_e) = Σ_q  W[q, cin, cout] · Φ_q(u_e)  # continuous kernel

y_v^{cout} = (1/|N(v)|) Σ_{w∈N(v)} Σ_{cin}
               x_w^{cin} · g_{cin,cout}(u(v,w))        # node aggregation
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

## Quick start

```python
import torch
from pmconv_ext import GraphInput, spline_convolution

# --- Graph: 5 nodes, 8 directed edges, 2D pseudo-coordinates, 3 features ---
N, E, d, Cin, Cout = 5, 8, 2, 3, 4
k, degree = 4, 1          # 4 basis functions per dim, degree-1 (linear) spline

edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
edge_attr  = torch.rand(E, d)   # pseudo-coordinates in [0,1]^d
x          = torch.rand(N, Cin) # node features

graph = GraphInput(edge_index, edge_attr, x)

# Learnable weight tensor: [Q, Cin, Cout],  Q = k^d = 16
W = torch.rand(k**d, Cin, Cout, requires_grad=True)

y = spline_convolution(graph, W, degree=degree, k_per_dim=[k]*d)
print(y.shape)  # [5, 4]
```

### Using individual ops

```python
from pmconv_ext import (
    multi_index_eval_map,
    basis_tensor_product,
    kernel_g_cin_cout,
    build_neighborhoods,
)

# Multi-index table for 2D, 4 basis functions per dim
idx = multi_index_eval_map([4, 4])          # [16, 2]

# Basis evaluations per edge
phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=[4, 4])  # [E, 16]

# Per-edge kernel matrices
W   = torch.rand(16, Cin, Cout)
g   = kernel_g_cin_cout(phi, W)             # [E, Cin, Cout]

# CSR-style neighbourhood structure
row_ptr, col_idx = build_neighborhoods(edge_index, N)
```

### Via `torch.ops`

All four ops are also callable through `torch.ops.pmconv`:

```python
import torch
import pmconv_ext  # registers the ops

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
W          : [Q, Cin, Cout]  float   – learnable weight tensor
phi        : [E, Q]          float   – basis evaluations
g          : [E, Cin, Cout]  float   – per-edge kernel matrices
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
src/
  spline_conv.h          – public C++ API (GraphInput + 4 op declarations)
  spline_conv.cpp        – implementation (B-spline basis + all ops)
  pmconv_bindings.cpp    – TORCH_LIBRARY registration + pybind11 module
  CMakeLists.txt         – CMake targets (existing executables + pmconv_ext)
setup.py                 – recommended Python extension build
tests/
  test_pmconv.py         – 39 unit tests covering all ops
```
