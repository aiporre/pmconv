#include "spline_conv.h"

#include <stdexcept>
#include <cmath>

namespace pmconv {

// ---------------------------------------------------------------------------
// GraphInput::validate
// ---------------------------------------------------------------------------
void GraphInput::validate() const {
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
        "edge_index must have shape [2, E], got ", edge_index.sizes());
    TORCH_CHECK(edge_index.dtype() == torch::kInt64,
        "edge_index must be int64");

    TORCH_CHECK(edge_attr.dim() == 2,
        "edge_attr must have shape [E, d], got ", edge_attr.sizes());
    TORCH_CHECK(edge_attr.is_floating_point(),
        "edge_attr must be a floating-point tensor");

    TORCH_CHECK(x.dim() == 2,
        "x must have shape [N, Cin], got ", x.sizes());
    TORCH_CHECK(x.is_floating_point(),
        "x must be a floating-point tensor");

    int64_t E_ei = edge_index.size(1);
    int64_t E_ea = edge_attr.size(0);
    TORCH_CHECK(E_ei == E_ea,
        "edge_index (",  E_ei, " edges) and edge_attr (", E_ea,
        " rows) must have the same number of edges");

    TORCH_CHECK(edge_attr.device() == x.device(),
        "edge_attr and x must be on the same device");
    TORCH_CHECK(edge_index.device() == x.device(),
        "edge_index and x must be on the same device");
}

// ---------------------------------------------------------------------------
// Internal helper — eval_bspline_basis
//
//   Evaluates the k uniform clamped (open) B-spline basis functions of
//   degree p at each of the E scalar values in u ∈ [0,1].
//
//   Uses the Cox–de Boor triangular algorithm, fully vectorised over the E
//   edge dimension using ATen broadcasting.
//
//   Returns: [E, k]  same dtype/device as u
// ---------------------------------------------------------------------------
static torch::Tensor eval_bspline_basis(
    const torch::Tensor& u,  // [E]  values in [0,1]
    int64_t k,               // number of basis functions
    int64_t p)               // polynomial degree
{
    TORCH_CHECK(u.dim() == 1, "u must be 1-D, got ", u.dim(), "D");
    TORCH_CHECK(k >= p + 1,
        "Need k >= p+1 basis functions for degree-", p, " spline, got k=", k);

    const int64_t E = u.size(0);
    const int64_t n = k + p + 1;  // total number of knots

    // ------------------------------------------------------------------
    // Build uniform open (clamped) knot vector on CPU double, then cast
    // ------------------------------------------------------------------
    auto kopts = torch::TensorOptions().dtype(torch::kDouble).device(torch::kCPU);
    auto knots = torch::zeros({n}, kopts);
    // First p+1 knots = 0,  last p+1 knots = 1
    for (int64_t i = k; i < n; ++i) knots[i] = 1.0;
    // Interior knots uniformly spaced
    if (k > p + 1) {
        const int64_t m = k - p;  // number of interior segments
        for (int64_t i = 1; i < m; ++i) {
            knots[p + i] = static_cast<double>(i) / static_cast<double>(m);
        }
    }
    knots = knots.to(u.device()).to(u.dtype());

    // ------------------------------------------------------------------
    // Clamp u away from 1.0 to keep the closed-right endpoint in the
    // last half-open interval of the degree-0 initialisation.
    // ------------------------------------------------------------------
    constexpr double U_CLAMP_EPS = 1e-6;
    constexpr double RECURRENCE_EPS = 1e-12;

    auto uc = u.clamp(0.0, 1.0 - U_CLAMP_EPS);   // [E]
    auto u2 = uc.unsqueeze(1);            // [E, 1]  for broadcasting

    // ------------------------------------------------------------------
    // Degree-0 initialisation: B_{i,0}(u) = 1{t_i <= u < t_{i+1}}
    // Result shape: [E, n-1]
    // ------------------------------------------------------------------
    auto kl = knots.slice(0, 0, n - 1).unsqueeze(0);  // [1, n-1]
    auto kr = knots.slice(0, 1, n).unsqueeze(0);       // [1, n-1]
    auto B = ((u2 >= kl) & (u2 < kr)).to(u.dtype());  // [E, n-1]

    // ------------------------------------------------------------------
    // Cox–de Boor recurrence for degrees r = 1 … p
    // After iteration r the shape is [E, n-1-r]; after all p iterations
    // it is [E, k]  (since n-1-p = k).
    // ------------------------------------------------------------------
    auto zeros_fn = [&](int64_t cols) {
        return torch::zeros({E, cols}, u.options());
    };

    for (int64_t r = 1; r <= p; ++r) {
        const int64_t m = n - 1 - r;   // columns after this iteration
        // Knot slices for the left and right recurrence coefficients
        auto ti   = knots.slice(0, 0,     m    ).unsqueeze(0);  // [1,m]
        auto tir  = knots.slice(0, r,     r + m).unsqueeze(0);  // [1,m]
        auto ti1  = knots.slice(0, 1,     m + 1).unsqueeze(0);  // [1,m]
        auto tir1 = knots.slice(0, r + 1, r + m + 1).unsqueeze(0); // [1,m]

        auto dl = tir  - ti;   // [1,m]  left  denominator
        auto dr = tir1 - ti1;  // [1,m]  right denominator

        // alpha_i = (u - t_i) / (t_{i+r} - t_i),  0 when denom = 0
        auto alpha = torch::where(
            dl > 0.0,
            (u2 - ti) / (dl + RECURRENCE_EPS),
            zeros_fn(m)
        ).clamp(0.0, 1.0);

        // beta_i  = (t_{i+r+1} - u) / (t_{i+r+1} - t_{i+1}),  0 when denom = 0
        auto beta = torch::where(
            dr > 0.0,
            (tir1 - u2) / (dr + RECURRENCE_EPS),
            zeros_fn(m)
        ).clamp(0.0, 1.0);

        auto Bl = B.slice(1, 0, m);      // [E, m]
        auto Br = B.slice(1, 1, m + 1);  // [E, m]
        B = alpha * Bl + beta * Br;      // [E, m]
    }
    // B is now [E, k]
    return B;
}

// ---------------------------------------------------------------------------
// Op 1 — multi_index_eval_map
// ---------------------------------------------------------------------------
torch::Tensor multi_index_eval_map(const std::vector<int64_t>& k_per_dim) {
    TORCH_CHECK(!k_per_dim.empty(), "k_per_dim must be non-empty");
    for (int64_t i = 0; i < static_cast<int64_t>(k_per_dim.size()); ++i) {
        TORCH_CHECK(k_per_dim[i] >= 1,
            "k_per_dim[", i, "] must be >= 1, got ", k_per_dim[i]);
    }
    const int64_t d = static_cast<int64_t>(k_per_dim.size());

    // Q = k_1 * ... * k_d
    int64_t Q = 1;
    for (auto k : k_per_dim) Q *= k;

    // C-order strides: stride[dim] = k_{dim+1} * ... * k_{d-1}
    std::vector<int64_t> strides(d);
    strides[d - 1] = 1;
    for (int64_t dim = d - 2; dim >= 0; --dim) {
        strides[dim] = strides[dim + 1] * k_per_dim[dim + 1];
    }

    auto idx = torch::zeros({Q, d}, torch::kInt64);
    auto idx_a = idx.accessor<int64_t, 2>();
    for (int64_t q = 0; q < Q; ++q) {
        for (int64_t dim = 0; dim < d; ++dim) {
            idx_a[q][dim] = (q / strides[dim]) % k_per_dim[dim];
        }
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Op 2 — basis_tensor_product
// ---------------------------------------------------------------------------
torch::Tensor basis_tensor_product(
    const torch::Tensor& edge_attr,
    int64_t degree,
    const std::vector<int64_t>& k_per_dim)
{
    TORCH_CHECK(edge_attr.dim() == 2,
        "edge_attr must be [E, d], got ", edge_attr.sizes());
    TORCH_CHECK(edge_attr.is_floating_point(),
        "edge_attr must be floating-point");
    TORCH_CHECK(degree >= 1, "degree must be >= 1, got ", degree);
    TORCH_CHECK(!k_per_dim.empty(), "k_per_dim must be non-empty");

    const int64_t E = edge_attr.size(0);
    const int64_t d = edge_attr.size(1);
    TORCH_CHECK(static_cast<int64_t>(k_per_dim.size()) == d,
        "k_per_dim length (", k_per_dim.size(),
        ") must match edge_attr feature dimension (", d, ")");

    // Start with all-ones column; iteratively extend with each dimension
    auto phi = torch::ones({E, 1}, edge_attr.options());

    for (int64_t dim = 0; dim < d; ++dim) {
        const int64_t k_dim = k_per_dim[dim];
        TORCH_CHECK(k_dim >= degree + 1,
            "k_per_dim[", dim, "] = ", k_dim,
            " must be >= degree+1 = ", degree + 1);

        // Evaluate 1-D basis along this dimension: [E, k_dim]
        auto u_dim = edge_attr.select(1, dim).contiguous();
        auto B_dim = eval_bspline_basis(u_dim, k_dim, degree);

        // Tensor product with accumulated result
        // phi   : [E, Q_prev]
        // B_dim : [E, k_dim]
        // -> outer product along feature dimension -> [E, Q_prev, k_dim]
        // -> reshape to [E, Q_prev * k_dim]
        const int64_t Q_prev = phi.size(1);
        phi = (phi.unsqueeze(2) * B_dim.unsqueeze(1))  // [E, Q_prev, k_dim]
                  .reshape({E, Q_prev * k_dim});         // [E, Q_new]
    }
    return phi;  // [E, Q]
}

// ---------------------------------------------------------------------------
// Op 3 — kernel_g_cin_cout
// ---------------------------------------------------------------------------
torch::Tensor kernel_g_cin_cout(
    const torch::Tensor& phi,
    const torch::Tensor& W)
{
    TORCH_CHECK(phi.dim() == 2,
        "phi must be [E, Q], got ", phi.sizes());
    TORCH_CHECK(W.dim() == 3,
        "W must be [Q, Cin, Cout], got ", W.sizes());
    TORCH_CHECK(phi.size(1) == W.size(0),
        "phi Q dim (", phi.size(1),
        ") must match W Q dim (", W.size(0), ")");
    TORCH_CHECK(phi.is_floating_point() && W.is_floating_point(),
        "phi and W must be floating-point tensors");
    TORCH_CHECK(phi.device() == W.device(),
        "phi and W must be on the same device");

    // g[e, cin, cout] = Σ_q  phi[e,q] * W[q, cin, cout]
    // = einsum("eq, qcd -> ecd", phi, W)
    return torch::einsum("eq,qcd->ecd", {phi, W});  // [E, Cin, Cout]
}

// ---------------------------------------------------------------------------
// Helper — build_neighborhoods
// ---------------------------------------------------------------------------
std::pair<torch::Tensor, torch::Tensor> build_neighborhoods(
    const torch::Tensor& edge_index,
    int64_t num_nodes)
{
    TORCH_CHECK(edge_index.dim() == 2 && edge_index.size(0) == 2,
        "edge_index must be [2, E], got ", edge_index.sizes());
    TORCH_CHECK(edge_index.dtype() == torch::kInt64,
        "edge_index must be int64");
    TORCH_CHECK(num_nodes > 0, "num_nodes must be > 0");

    const int64_t E = edge_index.size(1);
    auto dst = edge_index[1].contiguous();  // [E]

    // Count incoming edges per node
    auto deg = torch::zeros({num_nodes}, torch::kInt64);
    deg.index_add_(0, dst, torch::ones({E}, torch::kInt64));

    // Exclusive prefix-sum -> row_ptr [N+1]
    auto row_ptr = torch::zeros({num_nodes + 1}, torch::kInt64);
    row_ptr.slice(0, 1, num_nodes + 1) = deg.cumsum(0);

    // Fill col_idx (edge indices sorted by destination)
    auto col_idx = torch::zeros({E}, torch::kInt64);
    auto cursor  = row_ptr.slice(0, 0, num_nodes).clone();  // working write head
    auto cursor_a = cursor.accessor<int64_t, 1>();
    auto dst_a    = dst.accessor<int64_t, 1>();
    auto col_a    = col_idx.accessor<int64_t, 1>();

    for (int64_t e = 0; e < E; ++e) {
        int64_t v = dst_a[e];
        col_a[cursor_a[v]++] = e;
    }

    return {row_ptr, col_idx};
}

// ---------------------------------------------------------------------------
// Op 4 — spline_convolution
// ---------------------------------------------------------------------------
torch::Tensor spline_convolution(
    const GraphInput& graph,
    const torch::Tensor& W,
    int64_t degree,
    const std::vector<int64_t>& k_per_dim,
    bool normalize)
{
    graph.validate();

    const int64_t N    = graph.num_nodes();
    const int64_t E    = graph.num_edges();
    const int64_t Cin  = graph.num_node_features();
    const int64_t d    = graph.num_edge_dims();

    TORCH_CHECK(W.dim() == 3,
        "W must be [Q, Cin, Cout], got ", W.sizes());
    TORCH_CHECK(W.is_floating_point(),
        "W must be floating-point");
    TORCH_CHECK(W.size(1) == Cin,
        "W Cin dim (", W.size(1), ") must match x Cin (", Cin, ")");
    TORCH_CHECK(static_cast<int64_t>(k_per_dim.size()) == d,
        "k_per_dim length (", k_per_dim.size(),
        ") must match edge_attr feature dimension (", d, ")");

    const int64_t Cout = W.size(2);

    auto& edge_index = graph.edge_index;
    auto& edge_attr  = graph.edge_attr;
    auto& x          = graph.x;

    // 1) Basis evaluations: [E, Q]
    auto phi = basis_tensor_product(edge_attr, degree, k_per_dim);

    // 2) Per-edge kernel: [E, Cin, Cout]
    auto g = kernel_g_cin_cout(phi, W);

    // 3) Source and destination node indices
    auto src = edge_index[0].contiguous();  // [E]
    auto dst = edge_index[1].contiguous();  // [E]

    // 4) Gather source features: [E, Cin]
    auto x_src = x.index_select(0, src);   // [E, Cin]

    // 5) Per-edge messages: m[e, cout] = Σ_cin  x_src[e,cin] * g[e,cin,cout]
    //    = bmm( x_src[E,1,Cin], g[E,Cin,Cout] )[E,1,Cout] -> squeeze -> [E,Cout]
    auto msg = torch::bmm(x_src.unsqueeze(1), g).squeeze(1);  // [E, Cout]

    // 6) Scatter-add messages into output
    auto y = torch::zeros({N, Cout}, x.options());
    y.index_add_(0, dst, msg);

    // 7) Optional normalization by node degree
    if (normalize) {
        auto ones = torch::ones({E}, x.options());
        auto deg  = torch::zeros({N}, x.options());
        deg.index_add_(0, dst, ones);
        deg = deg.clamp_min(1.0);      // avoid division by zero
        y = y / deg.unsqueeze(1);
    }

    return y;  // [N, Cout]
}

}  // namespace pmconv
