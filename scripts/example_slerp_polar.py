import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-8

def normalize(v, axis=-1, eps=1e-15):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(n, eps, None)

def unit(v):
    return normalize(v)

def geodesic_distance(p, q):
    p = unit(p)
    q = unit(q)
    d = np.clip(np.sum(p * q, axis=-1), -1.0, 1.0)
    return np.arccos(d)

def slerp(p, q, a, eps=EPS):
    p = unit(p)
    q = unit(q)
    dot = np.clip(np.dot(p, q), -1.0, 1.0)

    # shortest arc
    if dot < 0.0:
        q = -q
        dot = -dot

    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)

    if s < eps:
        return unit((1.0 - a) * p + a * q)

    w0 = np.sin((1.0 - a) * theta) / s
    w1 = np.sin(a * theta) / s
    return unit(w0 * p + w1 * q)

def find_span(n, p, u, U):
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid

def basis_funs(span, u, p, U):
    N = np.zeros(p + 1)
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[span + 1 - j]
        right[j] = U[span + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            temp = 0.0 if abs(denom) < 1e-14 else N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved
    return N

def bspline_basis_all(U, p, u):
    U = np.asarray(U, dtype=float)
    n = len(U) - p - 2
    span = n if u == U[n + 1] else find_span(n, p, u, U)
    Nloc = basis_funs(span, u, p, U)
    out = np.zeros(n + 1)
    out[span - p : span + 1] = Nloc
    return out

def open_uniform_knot_vector(num_ctrl, degree):
    n = num_ctrl - 1
    p = degree
    m = n + p + 1
    U = np.zeros(m + 1)
    U[: p + 1] = 0.0
    U[m - p :] = 1.0
    interior = n - p
    if interior > 0:
        U[p + 1 : m - p] = np.linspace(1, interior, interior) / (interior + 1)
    return U

def periodic_uniform_knot_vector(num_ctrl, degree):
    n = num_ctrl - 1
    p = degree
    m = n + p + 1
    return np.arange(m + 1, dtype=float)

def spherical_deboor(control_pts, knots, degree, t):
    P = unit(np.asarray(control_pts, dtype=float))
    U = np.asarray(knots, dtype=float)
    p = degree
    n = len(P) - 1
    t = float(np.clip(t, U[p], U[n + 1]))
    k = find_span(n, p, t, U)
    d = [P[j].copy() for j in range(k - p, k + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = k - p + j
            denom = U[i + p - r + 1] - U[i]
            alpha = 0.0 if abs(denom) < 1e-14 else (t - U[i]) / denom
            d[j] = slerp(d[j - 1], d[j], alpha)
    return unit(d[p])

def spherical_bspline_curve(control_pts, knots, degree, num_samples=200):
    U = np.asarray(knots, dtype=float)
    t0 = U[degree]
    t1 = U[len(control_pts)]
    ts = np.linspace(t0, t1, num_samples)
    curve = np.array([spherical_deboor(control_pts, knots, degree, t) for t in ts])
    norms = np.linalg.norm(curve, axis=1)
    assert np.max(np.abs(norms - 1.0)) < 1e-10
    return ts, curve

def spherical_knot_insert(control_pts, knots, degree, u):
    P = unit(np.asarray(control_pts, dtype=float))
    U = np.asarray(knots, dtype=float)
    p = degree
    n = len(P) - 1
    k = find_span(n, p, u, U)

    Q = np.zeros((len(P) + 1, 3), dtype=float)
    Up = np.zeros(len(U) + 1, dtype=float)

    Up[: k + 1] = U[: k + 1]
    Up[k + 1] = u
    Up[k + 2 :] = U[k + 1 :]

    for i in range(0, k - p + 1):
        Q[i] = P[i]
    for i in range(k, n + 1):
        Q[i + 1] = P[i]

    for i in range(k - p + 1, k + 1):
        denom = U[i + p] - U[i]
        alpha = 0.0 if abs(denom) < 1e-14 else (u - U[i]) / denom
        Q[i] = slerp(P[i - 1], P[i], alpha)

    return unit(Q), Up

def chord_length_params_sphere(data_pts):
    X = unit(np.asarray(data_pts, dtype=float))
    d = [0.0]
    for i in range(1, len(X)):
        d.append(d[-1] + geodesic_distance(X[i - 1], X[i]))
    d = np.array(d)
    return np.linspace(0.0, 1.0, len(X)) if d[-1] < 1e-14 else d / d[-1]

def fit_spherical_bspline(data_pts, degree, num_control_pts, max_iter=30, tol=1e-10):
    X = unit(np.asarray(data_pts, dtype=float))
    u = chord_length_params_sphere(X)
    U = open_uniform_knot_vector(num_control_pts, degree)
    A = np.vstack([bspline_basis_all(U, degree, ui) for ui in u])
    cond = np.linalg.cond(A)

    C = np.zeros((num_control_pts, 3), dtype=float)
    prev = np.inf
    for _ in range(max_iter):
        for d in range(3):
            C[:, d], *_ = np.linalg.lstsq(A, X[:, d], rcond=None)
        C = unit(C)
        Y = np.array([spherical_deboor(C, U, degree, ui) for ui in u])
        err = np.mean(geodesic_distance(X, Y) ** 2)
        if abs(prev - err) < tol:
            break
        prev = err
    return C, U, {"params": u, "collocation_cond": cond, "mean_sq_geodesic_error": prev}

def cartesian_bspline_curve(control_pts, knots, degree, num_samples=200):
    U = np.asarray(knots, dtype=float)
    t0 = U[degree]
    t1 = U[len(control_pts)]
    ts = np.linspace(t0, t1, num_samples)
    P = np.asarray(control_pts, dtype=float)
    curve = []
    for t in ts:
        N = bspline_basis_all(U, degree, t)
        curve.append(N @ P)
    return ts, np.array(curve)

def plot_sphere(ax, alpha=0.08):
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(0, np.pi, 40)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, linewidth=0, color='lightgray')

def plot_case(ax, ctrl, curve, title, data=None):
    plot_sphere(ax)
    ctrl = np.asarray(ctrl)
    curve = np.asarray(curve)
    ax.plot(ctrl[:,0], ctrl[:,1], ctrl[:,2], '--o', label='control polygon')
    ax.plot(curve[:,0], curve[:,1], curve[:,2], '-', linewidth=2, label='spherical B-spline')
    if data is not None:
        data = np.asarray(data)
        ax.scatter(data[:,0], data[:,1], data[:,2], s=18, label='data')
    ax.set_title(title)
    ax.set_box_aspect([1,1,1])
    ax.set_xlim([-1.1,1.1]); ax.set_ylim([-1.1,1.1]); ax.set_zlim([-1.1,1.1])
    ax.legend(loc='upper left', fontsize=8)

def spherical_convex_hull_hemisphere_test(ctrl, curve):
    ctrl = unit(ctrl)
    curve = unit(curve)
    h = unit(np.mean(ctrl, axis=0))
    if np.linalg.norm(h) < 1e-12:
        return None
    return np.min(curve @ h)

def figure8_on_sphere(num=200):
    t = np.linspace(0, 2*np.pi, num, endpoint=False)
    x = np.cos(t)
    y = 0.6*np.sin(2*t)
    z = 0.5*np.sin(t)
    X = np.stack([x, y, z], axis=1)
    return unit(X)

def random_noisy_points_on_sphere(num=50, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(num, 3))
    X = unit(X)
    noise = 0.08 * rng.normal(size=(num, 3))
    return unit(X + noise)

def run_tests():
    fig = plt.figure(figsize=(15, 4.8))

    # 1) Great circle arc
    ctrl1 = unit(np.array([[1,0,0], [0,1,0]], dtype=float))
    U1 = np.array([0,0,1,1], dtype=float)
    _, curve1 = spherical_bspline_curve(ctrl1, U1, degree=1, num_samples=200)
    dev1 = np.max(np.abs(np.linalg.norm(curve1, axis=1) - 1.0))
    ax1 = fig.add_subplot(131, projection='3d')
    plot_case(ax1, ctrl1, curve1, f'Great circle arc\\nmax |‖C‖-1|={dev1:.2e}')

    # 2) Figure-8 on S² (periodic-ish: wrap first p control points)
    data2 = figure8_on_sphere(12)
    p = 3
    ctrl2 = unit(np.vstack([data2, data2[:p]]))
    U2 = periodic_uniform_knot_vector(len(ctrl2), p)
    t0, t1 = U2[p], U2[len(ctrl2)]
    ts2 = np.linspace(t0, t1-1e-9, 400)
    curve2 = np.array([spherical_deboor(ctrl2, U2, p, t) for t in ts2])
    dev2 = np.max(np.abs(np.linalg.norm(curve2, axis=1) - 1.0))
    ax2 = fig.add_subplot(132, projection='3d')
    plot_case(ax2, ctrl2[:-p], curve2, f'Figure-8 on S²\\nmax |‖C‖-1|={dev2:.2e}', data=data2)

    # 3) Fit noisy points
    data3 = random_noisy_points_on_sphere(50, seed=3)
    ctrl3, U3, info = fit_spherical_bspline(data3, degree=3, num_control_pts=10)
    _, curve3 = spherical_bspline_curve(ctrl3, U3, degree=3, num_samples=400)
    dev3 = np.max(np.abs(np.linalg.norm(curve3, axis=1) - 1.0))
    ax3 = fig.add_subplot(133, projection='3d')
    plot_case(ax3, ctrl3, curve3, f'Fit 50 noisy points\\ncond(A)={info["collocation_cond"]:.2e}, max |‖C‖-1|={dev3:.2e}', data=data3)

    plt.tight_layout()
    plt.show()

    # knot insertion validation
    u_insert = 0.5
    ctrl_ins, U_ins = spherical_knot_insert(ctrl3, U3, 3, u_insert)
    ts = np.linspace(U3[3], U3[len(ctrl3)], 250)
    old_curve = np.array([spherical_deboor(ctrl3, U3, 3, t) for t in ts])
    new_curve = np.array([spherical_deboor(ctrl_ins, U_ins, 3, t) for t in ts])
    insertion_error = np.max(geodesic_distance(old_curve, new_curve))
    print("Knot insertion max geodesic difference [rad]:", insertion_error)

    # spherical convex hull / hemisphere sanity check
    hemi_margin = spherical_convex_hull_hemisphere_test(ctrl3, curve3)
    print("Hemisphere hull margin (positive means inside chosen hemisphere):", hemi_margin)

    # compare with naive Cartesian B-spline + projection
    _, cart = cartesian_bspline_curve(ctrl3, U3, 3, 400)
    cart_proj = unit(cart)
    ge_err = np.max(geodesic_distance(curve3, cart_proj))
    print("Naive Cartesian+projection max geodesic error [rad]:", ge_err)

if __name__ == "__main__":
    run_tests()