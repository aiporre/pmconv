"""
tests/test_pmconv.py — unit tests for the four SplineCNN tensor ops
and helper utilities exposed by pmconv_ext.

Run with:
    python -m pytest tests/test_pmconv.py -v
"""

import os
import sys

# Ensure the PyTorch shared libs are on LD_LIBRARY_PATH so the extension can
# load libc10.so etc. when running outside a full conda/venv that has already
# set this up.
import torch
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
ld = os.environ.get("LD_LIBRARY_PATH", "")
if torch_lib not in ld:
    os.environ["LD_LIBRARY_PATH"] = torch_lib + (":" + ld if ld else "")

import importlib
import ctypes
# Reload with updated LD_LIBRARY_PATH (needed if already cached by linker)
for lib in ["libc10.so", "libtorch_cpu.so", "libtorch.so"]:
    try:
        ctypes.CDLL(os.path.join(torch_lib, lib))
    except OSError:
        pass

import pmconv_ext
from pmconv_ext import (
    GraphInput,
    multi_index_eval_map,
    basis_tensor_product,
    kernel_g_cin_cout,
    build_neighborhoods,
    spline_convolution,
)
import pytest


# ===========================================================================
# Op 1 — multi_index_eval_map
# ===========================================================================

class TestMultiIndexEvalMap:
    def test_1d_shape(self):
        idx = multi_index_eval_map([5])
        assert idx.shape == (5, 1), f"Expected (5,1), got {idx.shape}"

    def test_2d_shape(self):
        idx = multi_index_eval_map([3, 4])
        assert idx.shape == (12, 2), f"Expected (12,2), got {idx.shape}"

    def test_3d_shape(self):
        idx = multi_index_eval_map([2, 3, 4])
        assert idx.shape == (24, 3)

    def test_1d_values(self):
        idx = multi_index_eval_map([4])
        expected = torch.tensor([[0], [1], [2], [3]], dtype=torch.int64)
        assert torch.equal(idx, expected)

    def test_2d_c_order(self):
        """Last dimension should vary fastest (C / row-major order)."""
        idx = multi_index_eval_map([2, 3])
        # Expected:  (0,0),(0,1),(0,2),(1,0),(1,1),(1,2)
        expected = torch.tensor([
            [0, 0], [0, 1], [0, 2],
            [1, 0], [1, 1], [1, 2],
        ], dtype=torch.int64)
        assert torch.equal(idx, expected)

    def test_dtype(self):
        idx = multi_index_eval_map([3, 3])
        assert idx.dtype == torch.int64

    def test_total_size_matches_product(self):
        k = [2, 3, 5]
        idx = multi_index_eval_map(k)
        Q = 1
        for ki in k:
            Q *= ki
        assert idx.shape[0] == Q

    def test_deterministic(self):
        a = multi_index_eval_map([4, 4])
        b = multi_index_eval_map([4, 4])
        assert torch.equal(a, b)


# ===========================================================================
# Op 2 — basis_tensor_product
# ===========================================================================

class TestBasisTensorProduct:
    def test_output_shape_1d(self):
        E, d, k = 10, 1, 4
        edge_attr = torch.rand(E, d)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=[k])
        assert phi.shape == (E, k), f"Expected ({E},{k}), got {phi.shape}"

    def test_output_shape_2d(self):
        E, d = 20, 2
        k = [4, 5]
        edge_attr = torch.rand(E, d)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=k)
        assert phi.shape == (E, 4 * 5)

    def test_output_shape_3d(self):
        E = 8
        k = [3, 3, 3]
        edge_attr = torch.rand(E, 3)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=k)
        assert phi.shape == (E, 27)

    def test_partition_of_unity_linear(self):
        """B-spline basis functions at any point should sum to 1."""
        E, d, k = 50, 1, 6
        edge_attr = torch.rand(E, d)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=[k])
        row_sums = phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(E), atol=1e-5), \
            f"Partition of unity violated; max err = {(row_sums - 1).abs().max()}"

    def test_partition_of_unity_cubic(self):
        E, d, k = 50, 1, 8
        edge_attr = torch.rand(E, d)
        phi = basis_tensor_product(edge_attr, degree=3, k_per_dim=[k])
        row_sums = phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(E), atol=1e-5), \
            f"Partition of unity violated; max err = {(row_sums - 1).abs().max()}"

    def test_2d_partition_of_unity(self):
        """Tensor-product basis must also sum to 1 per edge."""
        E = 30
        edge_attr = torch.rand(E, 2)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=[4, 4])
        row_sums = phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(E), atol=1e-5)

    def test_non_negative(self):
        edge_attr = torch.rand(20, 2)
        phi = basis_tensor_product(edge_attr, degree=3, k_per_dim=[6, 6])
        assert (phi >= -1e-6).all(), "Basis values must be non-negative"

    def test_dtype_preserved(self):
        edge_attr = torch.rand(5, 1, dtype=torch.float64)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=[4])
        assert phi.dtype == torch.float64

    def test_empty_edges(self):
        """Zero edges should produce empty output without error."""
        edge_attr = torch.zeros(0, 2)
        phi = basis_tensor_product(edge_attr, degree=1, k_per_dim=[4, 4])
        assert phi.shape == (0, 16)


# ===========================================================================
# Op 3 — kernel_g_cin_cout
# ===========================================================================

class TestKernelGCinCout:
    def _make_phi_W(self, E=10, Q=12, Cin=3, Cout=5):
        phi = torch.rand(E, Q)
        W   = torch.rand(Q, Cin, Cout)
        return phi, W

    def test_output_shape(self):
        phi, W = self._make_phi_W(E=10, Q=12, Cin=3, Cout=5)
        g = kernel_g_cin_cout(phi, W)
        assert g.shape == (10, 3, 5)

    def test_linear_consistency(self):
        """g = einsum('eq,qcd->ecd', phi, W) — verify against pure-Python."""
        E, Q, Cin, Cout = 8, 6, 2, 3
        phi, W = self._make_phi_W(E, Q, Cin, Cout)
        g_ref  = torch.einsum("eq,qcd->ecd", phi, W)
        g_ext  = kernel_g_cin_cout(phi, W)
        assert torch.allclose(g_ext, g_ref, atol=1e-5), \
            f"Kernel mismatch: max err = {(g_ext - g_ref).abs().max()}"

    def test_zero_weights(self):
        phi = torch.rand(5, 8)
        W   = torch.zeros(8, 4, 6)
        g   = kernel_g_cin_cout(phi, W)
        assert torch.allclose(g, torch.zeros_like(g))

    def test_dtype_float64(self):
        phi = torch.rand(4, 6, dtype=torch.float64)
        W   = torch.rand(6, 2, 3, dtype=torch.float64)
        g   = kernel_g_cin_cout(phi, W)
        assert g.dtype == torch.float64


# ===========================================================================
# Helper — build_neighborhoods
# ===========================================================================

class TestBuildNeighborhoods:
    def test_row_ptr_size(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        rp, ci = build_neighborhoods(edge_index, 3)
        assert rp.shape == (4,)   # N+1

    def test_col_idx_size(self):
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        rp, ci = build_neighborhoods(edge_index, 3)
        assert ci.shape == (3,)   # E

    def test_degree_count(self):
        # Each node has exactly 1 incoming edge in a cycle 0->1->2->0
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        rp, ci = build_neighborhoods(edge_index, 3)
        rp_a = rp.tolist()
        for v in range(3):
            deg = rp_a[v + 1] - rp_a[v]
            assert deg == 1, f"Node {v} expected degree 1, got {deg}"

    def test_multi_incoming(self):
        # Nodes 0 and 1 both point to node 2
        edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.int64)
        rp, ci = build_neighborhoods(edge_index, 3)
        rp_a = rp.tolist()
        assert rp_a[3] - rp_a[2] == 2  # node 2 has 2 incoming
        assert rp_a[1] - rp_a[0] == 0  # node 0 has 0 incoming


# ===========================================================================
# GraphInput class
# ===========================================================================

class TestGraphInput:
    def _make(self, N=5, E=8, d=2, Cin=3):
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr  = torch.rand(E, d)
        x          = torch.rand(N, Cin)
        return GraphInput(edge_index, edge_attr, x)

    def test_accessors(self):
        g = self._make(N=5, E=8, d=2, Cin=3)
        assert g.num_nodes() == 5
        assert g.num_edges() == 8
        assert g.num_node_features() == 3
        assert g.num_edge_dims() == 2

    def test_validate_passes(self):
        g = self._make()
        g.validate()  # should not raise

    def test_repr(self):
        g = self._make(N=5, E=8, d=2, Cin=3)
        r = repr(g)
        assert "GraphInput" in r


# ===========================================================================
# Op 4 — spline_convolution
# ===========================================================================

class TestSplineConvolution:
    def _small_graph(self, N=4, Cin=3, Cout=5, d=2, E=8, degree=1, k=4):
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr  = torch.rand(E, d)
        x          = torch.rand(N, Cin)
        Q          = k ** d
        W          = torch.rand(Q, Cin, Cout)
        graph      = GraphInput(edge_index, edge_attr, x)
        return graph, W, degree, [k] * d

    def test_output_shape(self):
        graph, W, degree, k_per_dim = self._small_graph(N=6, Cin=3, Cout=7)
        y = spline_convolution(graph, W, degree, k_per_dim)
        assert y.shape == (6, 7)

    def test_no_nan(self):
        graph, W, degree, k_per_dim = self._small_graph()
        y = spline_convolution(graph, W, degree, k_per_dim)
        assert not torch.isnan(y).any(), "Output contains NaN"

    def test_isolated_node_safe(self):
        """A node with no incoming edges should produce zero output."""
        N, E, Cin, Cout, d, k = 4, 3, 2, 3, 1, 4
        # Edges only touch nodes 0,1,2 — node 3 is isolated (no incoming)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        edge_attr  = torch.rand(E, d)
        x          = torch.rand(N, Cin)
        W          = torch.rand(k, Cin, Cout)
        graph      = GraphInput(edge_index, edge_attr, x)
        y = spline_convolution(graph, W, degree=1, k_per_dim=[k])
        # Node 3 has no incoming edges — output should be zero
        assert torch.allclose(y[3], torch.zeros(Cout), atol=1e-6)

    def test_hand_checkable_star_graph(self):
        """
        Star graph: hub=0 receives messages from all spokes (1,2,3).
        With W=Identity (per channel), degree=1, k=2, d=1:
        -  all edges have u=0.5  ->  phi = [0.5, 0.5] exactly at midpoint
        -  g_{cin,cout}(0.5) = 0.5 * W[0] + 0.5 * W[1]
        -  y[0] should equal mean( sum_cin x[spoke, cin] * g_{cin,cout} )
        We just check shape and no-NaN here; exact value check in comment.
        """
        N  = 4     # node 0 = hub, nodes 1,2,3 = spokes
        E  = 3     # three spokes
        Cin, Cout, k, d = 2, 2, 2, 1
        edge_index = torch.tensor([[1, 2, 3], [0, 0, 0]], dtype=torch.int64)
        edge_attr  = torch.full((E, d), 0.5)
        x          = torch.rand(N, Cin)
        W          = torch.rand(k, Cin, Cout)
        graph      = GraphInput(edge_index, edge_attr, x)
        y = spline_convolution(graph, W, degree=1, k_per_dim=[k])
        assert y.shape == (N, Cout)
        assert not torch.isnan(y).any()

    def test_no_normalize(self):
        graph, W, degree, k_per_dim = self._small_graph(N=4, E=4)
        y_norm   = spline_convolution(graph, W, degree, k_per_dim, normalize=True)
        y_nonorm = spline_convolution(graph, W, degree, k_per_dim, normalize=False)
        # With normalize=False the magnitudes should generally differ
        # (unless all degrees happen to be 1). We just check shapes.
        assert y_norm.shape == y_nonorm.shape

    def test_dtype_preserved(self):
        N, E, Cin, Cout, d, k = 4, 6, 2, 3, 1, 4
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr  = torch.rand(E, d, dtype=torch.float64)
        x          = torch.rand(N, Cin, dtype=torch.float64)
        W          = torch.rand(k, Cin, Cout, dtype=torch.float64)
        graph      = GraphInput(edge_index, edge_attr, x)
        y = spline_convolution(graph, W, degree=1, k_per_dim=[k])
        assert y.dtype == torch.float64


# ===========================================================================
# Integration — torch.ops.pmconv registration
# ===========================================================================

class TestTorchOpsRegistration:
    def test_ops_registered(self):
        import torch
        assert hasattr(torch.ops, "pmconv"), \
            "torch.ops.pmconv namespace not registered"

    def test_multi_index_via_ops(self):
        idx = torch.ops.pmconv.multi_index_eval_map([3, 3])
        assert idx.shape == (9, 2)

    def test_basis_via_ops(self):
        edge_attr = torch.rand(5, 2)
        phi = torch.ops.pmconv.basis_tensor_product(edge_attr, 1, [4, 4])
        assert phi.shape == (5, 16)

    def test_kernel_via_ops(self):
        phi = torch.rand(5, 16)
        W   = torch.rand(16, 3, 4)
        g   = torch.ops.pmconv.kernel_g_cin_cout(phi, W)
        assert g.shape == (5, 3, 4)

    def test_conv_via_ops(self):
        N, E, d, k, Cin, Cout = 4, 6, 2, 4, 3, 5
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr  = torch.rand(E, d)
        x          = torch.rand(N, Cin)
        W          = torch.rand(k ** d, Cin, Cout)
        y = torch.ops.pmconv.spline_convolution(
            edge_index, edge_attr, x, W, 1, [k, k], True
        )
        assert y.shape == (N, Cout)


# ===========================================================================
# Python-level API — pmconv package (SplineSupport / spline_basis /
# spline_weighting / spline_conv) mirroring pytorch_spline_conv structure
# ===========================================================================

from pmconv import (
    SplineSupport,
    spline_basis,
    spline_weighting,
    spline_conv,
)


class TestSplineSupport:
    """Tests for SplineSupport (the S object) — kernel domain structure."""

    def test_basic_creation(self):
        S = SplineSupport(k_per_dim=[4, 4], degree=1)
        assert S.Q == 16
        assert S.d == 2

    def test_1d_creation(self):
        S = SplineSupport(k_per_dim=[6], degree=3)
        assert S.Q == 6
        assert S.d == 1

    def test_multi_index_shape(self):
        S = SplineSupport(k_per_dim=[3, 4], degree=1)
        assert S.multi_index.shape == (12, 2)

    def test_multi_index_dtype(self):
        S = SplineSupport(k_per_dim=[4, 4], degree=1)
        assert S.multi_index.dtype == torch.int64

    def test_multi_index_deterministic(self):
        S1 = SplineSupport(k_per_dim=[3, 3], degree=1)
        S2 = SplineSupport(k_per_dim=[3, 3], degree=1)
        assert torch.equal(S1.multi_index, S2.multi_index)

    def test_repr(self):
        S = SplineSupport(k_per_dim=[4, 4], degree=1)
        r = repr(S)
        assert "SplineSupport" in r
        assert "k_per_dim=[4, 4]" in r

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError, match="degree must be >= 1"):
            SplineSupport(k_per_dim=[4], degree=0)

    def test_invalid_k_too_small_raises(self):
        with pytest.raises(ValueError, match="must be >= degree\\+1"):
            SplineSupport(k_per_dim=[2], degree=3)  # 2 < 3+1=4

    def test_empty_k_per_dim_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            SplineSupport(k_per_dim=[], degree=1)


class TestSplineBasis:
    """Tests for spline_basis — mirrors pytorch_spline_conv basis module."""

    def _make(self, E=10, d=2, k=4, degree=1):
        S = SplineSupport(k_per_dim=[k] * d, degree=degree)
        edge_attr = torch.rand(E, d)
        return edge_attr, S

    def test_output_shapes(self):
        E, d, k = 10, 2, 4
        edge_attr, S = self._make(E, d, k)
        phi, mi = spline_basis(edge_attr, S)
        assert phi.shape == (E, S.Q)
        assert mi.shape == (S.Q, d)

    def test_multi_index_same_as_support(self):
        edge_attr, S = self._make()
        _, mi = spline_basis(edge_attr, S)
        assert torch.equal(mi, S.multi_index)

    def test_partition_of_unity(self):
        """Σ_q Φ_q(u_e) = 1 for all e — partition of unity."""
        E, d, k = 40, 2, 5
        edge_attr, S = self._make(E, d, k, degree=1)
        phi, _ = spline_basis(edge_attr, S)
        row_sums = phi.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(E), atol=1e-5)

    def test_non_negative(self):
        edge_attr, S = self._make(E=30, d=2, k=6, degree=3)
        phi, _ = spline_basis(edge_attr, S)
        assert (phi >= -1e-6).all()

    def test_dtype_preserved(self):
        S = SplineSupport(k_per_dim=[4], degree=1)
        edge_attr = torch.rand(5, 1, dtype=torch.float64)
        phi, _ = spline_basis(edge_attr, S)
        assert phi.dtype == torch.float64


class TestSplineWeighting:
    """Tests for spline_weighting — tensor-valued kernel K_S application."""

    def _make(self, E=10, Q=16, Cin=3, Cout=5):
        x_src = torch.rand(E, Cin)
        weight = torch.rand(Q, Cin, Cout)
        phi = torch.rand(E, Q)
        return x_src, weight, phi

    def test_output_shape(self):
        x_src, weight, phi = self._make(E=10, Q=16, Cin=3, Cout=5)
        msg = spline_weighting(x_src, weight, phi)
        assert msg.shape == (10, 5)

    def test_consistency_with_einsum(self):
        """msg = bmm(x_src, einsum(phi, weight)) — verify against reference."""
        E, Q, Cin, Cout = 8, 12, 3, 4
        x_src, weight, phi = self._make(E, Q, Cin, Cout)
        # Reference: assemble kernel then apply
        g_ref = torch.einsum("eq,qcd->ecd", phi, weight)  # [E, Cin, Cout]
        msg_ref = torch.bmm(x_src.unsqueeze(1), g_ref).squeeze(1)  # [E, Cout]
        msg_ext = spline_weighting(x_src, weight, phi)
        assert torch.allclose(msg_ext, msg_ref, atol=1e-5)

    def test_zero_weights(self):
        x_src = torch.rand(5, 3)
        weight = torch.zeros(8, 3, 4)
        phi = torch.rand(5, 8)
        msg = spline_weighting(x_src, weight, phi)
        assert torch.allclose(msg, torch.zeros(5, 4))

    def test_autograd_w(self):
        """Gradient flows through weight."""
        E, Q, Cin, Cout = 4, 6, 2, 3
        x_src = torch.rand(E, Cin)
        phi = torch.rand(E, Q)
        weight = torch.rand(Q, Cin, Cout, requires_grad=True)
        msg = spline_weighting(x_src, weight, phi)
        msg.sum().backward()
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape

    def test_autograd_x_src(self):
        """Gradient flows through x_src."""
        E, Q, Cin, Cout = 4, 6, 2, 3
        x_src = torch.rand(E, Cin, requires_grad=True)
        phi = torch.rand(E, Q)
        weight = torch.rand(Q, Cin, Cout)
        msg = spline_weighting(x_src, weight, phi)
        msg.sum().backward()
        assert x_src.grad is not None
        assert x_src.grad.shape == x_src.shape


class TestSplineConv:
    """Tests for spline_conv — full tensor-valued SplineCNN convolution."""

    def _small_graph(self, N=4, Cin=3, Cout=5, d=2, E=8, k=4, degree=1):
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr = torch.rand(E, d)
        x = torch.rand(N, Cin)
        S = SplineSupport(k_per_dim=[k] * d, degree=degree)
        W = torch.rand(S.Q, Cin, Cout)
        graph = GraphInput(edge_index, edge_attr, x)
        return graph, W, S

    def test_output_shape(self):
        graph, W, S = self._small_graph(N=6, Cin=3, Cout=7)
        y = spline_conv(graph, W, S)
        assert y.shape == (6, 7)

    def test_no_nan(self):
        graph, W, S = self._small_graph()
        y = spline_conv(graph, W, S)
        assert not torch.isnan(y).any()

    def test_isolated_node_zero(self):
        """A node with no incoming edges must produce zero output."""
        N, E, Cin, Cout, d, k = 4, 3, 2, 3, 1, 4
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.int64)
        edge_attr = torch.rand(E, d)
        x = torch.rand(N, Cin)
        S = SplineSupport(k_per_dim=[k], degree=1)
        W = torch.rand(S.Q, Cin, Cout)
        graph = GraphInput(edge_index, edge_attr, x)
        y = spline_conv(graph, W, S)
        # Node 3 has no incoming edge
        assert torch.allclose(y[3], torch.zeros(Cout), atol=1e-6)

    def test_norm_false_vs_true_shape(self):
        graph, W, S = self._small_graph()
        y_norm = spline_conv(graph, W, S, norm=True)
        y_none = spline_conv(graph, W, S, norm=False)
        assert y_norm.shape == y_none.shape

    def test_matches_low_level_api(self):
        """spline_conv must match direct pmconv_ext.spline_convolution."""
        N, E, d, Cin, Cout, k = 5, 10, 2, 3, 4, 4
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr = torch.rand(E, d)
        x = torch.rand(N, Cin)
        S = SplineSupport(k_per_dim=[k] * d, degree=1)
        W = torch.rand(S.Q, Cin, Cout)
        graph = GraphInput(edge_index, edge_attr, x)

        y_new = spline_conv(graph, W, S, norm=True)
        y_old = spline_convolution(graph, W, degree=1, k_per_dim=[k] * d,
                                   normalize=True)
        assert torch.allclose(y_new, y_old, atol=1e-5), \
            f"Max diff: {(y_new - y_old).abs().max()}"

    def test_autograd_weight(self):
        """Gradient flows through W (tensor-valued kernel)."""
        N, E, d, Cin, Cout, k = 4, 6, 2, 3, 4, 4
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr = torch.rand(E, d)
        x = torch.rand(N, Cin)
        S = SplineSupport(k_per_dim=[k] * d, degree=1)
        W = torch.rand(S.Q, Cin, Cout, requires_grad=True)
        graph = GraphInput(edge_index, edge_attr, x)

        y = spline_conv(graph, W, S)
        y.sum().backward()
        assert W.grad is not None
        assert W.grad.shape == (S.Q, Cin, Cout)

    def test_root_weight(self):
        """Optional root_weight term is added correctly."""
        N, E, Cin, Cout, d, k = 4, 6, 3, 5, 2, 4
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr = torch.rand(E, d)
        x = torch.rand(N, Cin)
        S = SplineSupport(k_per_dim=[k] * d, degree=1)
        W = torch.rand(S.Q, Cin, Cout)
        rw = torch.rand(Cin, Cout)
        graph = GraphInput(edge_index, edge_attr, x)

        y_no_rw = spline_conv(graph, W, S)
        y_rw = spline_conv(graph, W, S, root_weight=rw)
        expected_diff = x @ rw
        assert torch.allclose(y_rw - y_no_rw, expected_diff, atol=1e-5)

    def test_bias(self):
        """Optional bias is added to every output node."""
        N, E, Cin, Cout, d, k = 4, 6, 3, 5, 2, 4
        edge_index = torch.randint(0, N, (2, E), dtype=torch.int64)
        edge_attr = torch.rand(E, d)
        x = torch.rand(N, Cin)
        S = SplineSupport(k_per_dim=[k] * d, degree=1)
        W = torch.rand(S.Q, Cin, Cout)
        b = torch.rand(Cout)
        graph = GraphInput(edge_index, edge_attr, x)

        y_no_b = spline_conv(graph, W, S)
        y_b = spline_conv(graph, W, S, bias=b)
        assert torch.allclose(y_b - y_no_b, b.unsqueeze(0).expand(N, -1),
                               atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
