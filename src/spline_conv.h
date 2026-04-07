#pragma once
#include <torch/torch.h>
#include <vector>

namespace pmconv {

// ---------------------------------------------------------------------------
// GraphInput — holds the three tensors that describe one graph
// ---------------------------------------------------------------------------
//  edge_index : [2, E]   int64   – row 0 = source node, row 1 = target node
//  edge_attr  : [E, d]   float   – pseudo-coordinates u(e) ∈ [0,1]^d
//  x          : [N, Cin] float   – node feature matrix
// ---------------------------------------------------------------------------
struct GraphInput {
    torch::Tensor edge_index;  // [2, E]
    torch::Tensor edge_attr;   // [E, d]
    torch::Tensor x;           // [N, Cin]

    GraphInput(torch::Tensor ei, torch::Tensor ea, torch::Tensor xf)
        : edge_index(std::move(ei)),
          edge_attr(std::move(ea)),
          x(std::move(xf)) {}

    int64_t num_nodes() const { return x.size(0); }
    int64_t num_edges() const { return edge_attr.size(0); }
    int64_t num_node_features() const { return x.size(1); }
    int64_t num_edge_dims() const { return edge_attr.size(1); }

    void validate() const;
};

// ---------------------------------------------------------------------------
// Op 1 — multi_index_eval_map
//
//   Enumerates all multi-indices q = (q_1, …, q_d) in the Cartesian product
//   {0,…,k_1-1} × … × {0,…,k_d-1}, returning them row-by-row in C order
//   (last dimension varies fastest).
//
//   Returns: [Q, d]  int64   where Q = k_1 * … * k_d
// ---------------------------------------------------------------------------
torch::Tensor multi_index_eval_map(const std::vector<int64_t>& k_per_dim);

// ---------------------------------------------------------------------------
// Op 2 — basis_tensor_product
//
//   Evaluates tensor-product B-spline basis Φ_q(u(e)) for every edge e and
//   every multi-index q.  Uses uniform clamped (open) B-splines of given
//   degree in each dimension.
//
//   edge_attr  : [E, d]   float  – pseudo-coordinates in [0,1]^d
//   degree     : int             – B-spline degree (1 = linear, 3 = cubic …)
//   k_per_dim  : [d]      int64  – number of basis functions per dimension
//
//   Returns: [E, Q]  float   where Q = prod(k_per_dim)
// ---------------------------------------------------------------------------
torch::Tensor basis_tensor_product(
    const torch::Tensor& edge_attr,
    int64_t degree,
    const std::vector<int64_t>& k_per_dim);

// ---------------------------------------------------------------------------
// Op 3 — kernel_g_cin_cout
//
//   Evaluates the continuous kernel
//
//       g_{cin,cout}(u_e) = Σ_q  W[q, cin, cout] · Φ_q(u_e)
//
//   phi : [E, Q]          float  – basis evaluations from basis_tensor_product
//   W   : [Q, Cin, Cout]  float  – learnable weight tensor
//
//   Returns: [E, Cin, Cout]  float
// ---------------------------------------------------------------------------
torch::Tensor kernel_g_cin_cout(
    const torch::Tensor& phi,
    const torch::Tensor& W);

// ---------------------------------------------------------------------------
// Helper — build_neighborhoods
//
//   Converts edge_index [2, E] to a CSR-style representation sorted by
//   destination node so that convolution aggregation can be done without
//   sorting inside the hot path.
//
//   Returns: (row_ptr [N+1], col_idx [E])
//     row_ptr[v] … row_ptr[v+1] are the positions in col_idx (and in the
//     original edge ordering) of edges whose destination is v.
//
//   NOTE: this builds an auxiliary index; the caller must use col_idx as a
//   permutation into the original edge dimension when needed.
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> build_neighborhoods(
    const torch::Tensor& edge_index,  // [2, E]
    int64_t num_nodes);

// ---------------------------------------------------------------------------
// Op 4 — spline_convolution
//
//   Computes node-wise convolution output
//
//       y_v^{cout} = (1 / |N(v)|) Σ_{w ∈ N(v)} Σ_{cin}
//                       x_w^{cin} · g_{cin,cout}(u(v,w))
//
//   graph     : GraphInput  (edge_index, edge_attr, x)
//   W         : [Q, Cin, Cout]  float  – learnable weights
//   degree    : int              – B-spline degree
//   k_per_dim : [d]      int64   – basis functions per dimension
//   normalize : bool             – divide by node degree (default true)
//
//   Returns: [N, Cout]  float
// ---------------------------------------------------------------------------
torch::Tensor spline_convolution(
    const GraphInput& graph,
    const torch::Tensor& W,
    int64_t degree,
    const std::vector<int64_t>& k_per_dim,
    bool normalize = true);

}  // namespace pmconv
