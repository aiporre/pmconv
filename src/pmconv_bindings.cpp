// pmconv_bindings.cpp
//
// (1) TORCH_LIBRARY — registers four ops under the "pmconv" namespace so they
//     are callable as  torch.ops.pmconv.<name>(...)  from Python.
//
// (2) PYBIND11_MODULE — exposes GraphInput and the four ops (plus the
//     build_neighborhoods helper) as a regular Python extension module named
//     "pmconv_ext".

#include <torch/library.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // automatic std::vector <-> list conversion

#include "spline_conv.h"

namespace py = pybind11;

// ============================================================================
// Thin wrappers that accept c10::IntArrayRef (required by TORCH_LIBRARY)
// ============================================================================

static torch::Tensor tl_multi_index_eval_map(c10::IntArrayRef k_per_dim) {
    return pmconv::multi_index_eval_map(
        std::vector<int64_t>(k_per_dim.begin(), k_per_dim.end()));
}

static torch::Tensor tl_basis_tensor_product(
    const torch::Tensor& edge_attr,
    int64_t degree,
    c10::IntArrayRef k_per_dim)
{
    return pmconv::basis_tensor_product(
        edge_attr, degree,
        std::vector<int64_t>(k_per_dim.begin(), k_per_dim.end()));
}

static torch::Tensor tl_kernel_g_cin_cout(
    const torch::Tensor& phi,
    const torch::Tensor& W)
{
    return pmconv::kernel_g_cin_cout(phi, W);
}

static torch::Tensor tl_spline_convolution(
    const torch::Tensor& edge_index,
    const torch::Tensor& edge_attr,
    const torch::Tensor& x,
    const torch::Tensor& W,
    int64_t degree,
    c10::IntArrayRef k_per_dim,
    bool normalize)
{
    pmconv::GraphInput g(edge_index, edge_attr, x);
    return pmconv::spline_convolution(
        g, W, degree,
        std::vector<int64_t>(k_per_dim.begin(), k_per_dim.end()),
        normalize);
}

// ============================================================================
// TORCH_LIBRARY — register schemas
// ============================================================================
TORCH_LIBRARY(pmconv, m) {
    m.def("multi_index_eval_map(int[] k_per_dim) -> Tensor");
    m.def("basis_tensor_product(Tensor edge_attr, int degree, int[] k_per_dim) -> Tensor");
    m.def("kernel_g_cin_cout(Tensor phi, Tensor W) -> Tensor");
    m.def("spline_convolution(Tensor edge_index, Tensor edge_attr, Tensor x, Tensor W, int degree, int[] k_per_dim, bool normalize) -> Tensor");
}

// ============================================================================
// TORCH_LIBRARY_IMPL — bind implementations
//
// multi_index_eval_map has no tensor arguments so the dispatcher cannot
// determine a backend from the inputs.  We register it under
// CompositeExplicitAutograd (dispatched before backend selection) so it is
// callable as  torch.ops.pmconv.multi_index_eval_map(...)  on any device.
// The remaining three ops take tensor arguments and are registered for CPU.
// ============================================================================
TORCH_LIBRARY_IMPL(pmconv, CompositeExplicitAutograd, m) {
    m.impl("multi_index_eval_map", tl_multi_index_eval_map);
}

TORCH_LIBRARY_IMPL(pmconv, CPU, m) {
    m.impl("basis_tensor_product",  tl_basis_tensor_product);
    m.impl("kernel_g_cin_cout",     tl_kernel_g_cin_cout);
    m.impl("spline_convolution",    tl_spline_convolution);
}

// ============================================================================
// pybind11 module — "pmconv_ext"
// ============================================================================
PYBIND11_MODULE(pmconv_ext, m) {
    m.doc() = "SplineCNN tensor ops for pmconv (pybind11 interface)";

    // --- GraphInput class ---------------------------------------------------
    py::class_<pmconv::GraphInput>(m, "GraphInput",
        R"doc(
Container for the three tensors that describe one graph.

Attributes
----------
edge_index : Tensor [2, E]   int64   – row 0 = source, row 1 = target
edge_attr  : Tensor [E, d]   float   – pseudo-coordinates in [0,1]^d
x          : Tensor [N, Cin] float   – node feature matrix
)doc")
        .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor>(),
             py::arg("edge_index"), py::arg("edge_attr"), py::arg("x"))
        .def_readwrite("edge_index", &pmconv::GraphInput::edge_index)
        .def_readwrite("edge_attr",  &pmconv::GraphInput::edge_attr)
        .def_readwrite("x",          &pmconv::GraphInput::x)
        .def("num_nodes",         &pmconv::GraphInput::num_nodes)
        .def("num_edges",         &pmconv::GraphInput::num_edges)
        .def("num_node_features", &pmconv::GraphInput::num_node_features)
        .def("num_edge_dims",     &pmconv::GraphInput::num_edge_dims)
        .def("validate",          &pmconv::GraphInput::validate)
        .def("__repr__", [](const pmconv::GraphInput& g) {
            std::ostringstream ss;
            ss << "GraphInput(N=" << g.num_nodes()
               << ", E=" << g.num_edges()
               << ", Cin=" << g.num_node_features()
               << ", d="   << g.num_edge_dims() << ")";
            return ss.str();
        });

    // --- Op 1: multi_index_eval_map ----------------------------------------
    m.def("multi_index_eval_map",
        [](const std::vector<int64_t>& k_per_dim) {
            return pmconv::multi_index_eval_map(k_per_dim);
        },
        py::arg("k_per_dim"),
        R"doc(
multi_index_eval_map(k_per_dim) -> Tensor [Q, d]

Enumerate all multi-indices q in {0,…,k_1-1} x … x {0,…,k_d-1} in C order
(last dimension varies fastest).

Parameters
----------
k_per_dim : list[int]  – number of basis functions per dimension [k_1,…,k_d]

Returns
-------
Tensor of shape [Q, d], dtype int64, where Q = prod(k_per_dim)
)doc");

    // --- Op 2: basis_tensor_product ----------------------------------------
    m.def("basis_tensor_product",
        [](const torch::Tensor& edge_attr,
           int64_t degree,
           const std::vector<int64_t>& k_per_dim) {
            return pmconv::basis_tensor_product(edge_attr, degree, k_per_dim);
        },
        py::arg("edge_attr"), py::arg("degree"), py::arg("k_per_dim"),
        R"doc(
basis_tensor_product(edge_attr, degree, k_per_dim) -> Tensor [E, Q]

Evaluate tensor-product B-spline basis Φ_q(u(e)) for every edge e and
every multi-index q.  Uses uniform clamped B-splines of degree `degree`.

Parameters
----------
edge_attr : Tensor [E, d]  float  – pseudo-coordinates in [0,1]^d
degree    : int            – B-spline polynomial degree (≥ 1)
k_per_dim : list[int]      – number of basis functions per dimension

Returns
-------
Tensor of shape [E, Q], where Q = prod(k_per_dim)
)doc");

    // --- Op 3: kernel_g_cin_cout -------------------------------------------
    m.def("kernel_g_cin_cout",
        [](const torch::Tensor& phi, const torch::Tensor& W) {
            return pmconv::kernel_g_cin_cout(phi, W);
        },
        py::arg("phi"), py::arg("W"),
        R"doc(
kernel_g_cin_cout(phi, W) -> Tensor [E, Cin, Cout]

Evaluate  g_{cin,cout}(u_e) = Σ_q  W[q,cin,cout] · Φ_q(u_e).

Parameters
----------
phi : Tensor [E, Q]          float  – basis evaluations
W   : Tensor [Q, Cin, Cout]  float  – learnable weight tensor

Returns
-------
Tensor of shape [E, Cin, Cout]
)doc");

    // --- Helper: build_neighborhoods ---------------------------------------
    m.def("build_neighborhoods",
        [](const torch::Tensor& edge_index, int64_t num_nodes) {
            auto [rp, ci] = pmconv::build_neighborhoods(edge_index, num_nodes);
            return py::make_tuple(rp, ci);
        },
        py::arg("edge_index"), py::arg("num_nodes"),
        R"doc(
build_neighborhoods(edge_index, num_nodes) -> (row_ptr, col_idx)

Convert edge_index [2, E] to CSR-style representation keyed by destination
node.  row_ptr[v]..row_ptr[v+1] index into col_idx, which holds edge indices
whose destination equals v.

Returns
-------
(row_ptr [N+1], col_idx [E])  both int64
)doc");

    // --- Op 4: spline_convolution ------------------------------------------
    m.def("spline_convolution",
        [](const pmconv::GraphInput& graph,
           const torch::Tensor& W,
           int64_t degree,
           const std::vector<int64_t>& k_per_dim,
           bool normalize) {
            return pmconv::spline_convolution(graph, W, degree, k_per_dim, normalize);
        },
        py::arg("graph"), py::arg("W"),
        py::arg("degree"), py::arg("k_per_dim"),
        py::arg("normalize") = true,
        R"doc(
spline_convolution(graph, W, degree, k_per_dim, normalize=True) -> Tensor [N, Cout]

Compute node-wise convolution

    y_v^{cout} = (1/|N(v)|) Σ_{w ∈ N(v)} Σ_{cin}  x_w^{cin} · g_{cin,cout}(u(v,w))

Builds per-edge kernel internally from W and the B-spline basis.
Neighborhoods are derived from graph.edge_index.

Parameters
----------
graph     : GraphInput  (edge_index, edge_attr, x)
W         : Tensor [Q, Cin, Cout]  float  – learnable weights
degree    : int                    – B-spline degree
k_per_dim : list[int]              – basis functions per dimension
normalize : bool                   – divide by node degree (default True)

Returns
-------
Tensor of shape [N, Cout]
)doc");
}
