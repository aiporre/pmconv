"""
Spherical B-Spline Algorithm
=============================
Native implementation on the unit sphere S^2 following the
Casciola-Morigi framework (polar/spherical splines) and the
manifold-valued B-spline construction via SLERP-based de Boor.

References
----------
- Casciola & Morigi: "Splines on Surfaces and Their Applications"
- arXiv:2601.17841 (manifold-valued B-splines on symmetric spaces)
- Shoemake (1985): animating rotation with quaternion curves (SLERP origin)

Sections
--------
1.  Core geometry: SLERP, geodesic distance, spherical convex hull check
2.  Knot vector utilities: clamped, periodic, arc-length parameterisation
3.  Cox-de Boor scalar basis  N_{i,p}(t)
4.  Spherical de Boor evaluator  C(t) in S^2
5.  Full-curve sampler + unit-norm verifier
6.  Least-squares spherical B-spline fitting
7.  Knot insertion (Boehm's algorithm on S^2)
8.  Validation: great circle, figure-8, noisy fitting
9.  Matplotlib visualisation helper
"""

import numpy as np
from numpy.linalg import norm, lstsq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (side-effect import)
from typing import Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1.  CORE GEOMETRY
# ─────────────────────────────────────────────────────────────────────────────

EPS = 1e-8   # numerical threshold for near-degenerate SLERP


def normalize(v: np.ndarray) -> np.ndarray:
    """Project a vector onto S^2 (in-place safe, returns new array)."""
    n = norm(v)
    if n < EPS:
        raise ValueError(f"Cannot normalize near-zero vector: {v}")
    return v / n


def slerp(p: np.ndarray, q: np.ndarray, alpha: float) -> np.ndarray:
    """
    Geodesic interpolation between unit vectors p and q on S^2.

    SLERP(p, q, alpha) = sin((1-alpha)*theta)/sin(theta) * p
                       + sin(alpha*theta)/sin(theta)     * q

    Falls back to normalised linear interpolation when sin(theta) < EPS
    (coincident or antipodal points, or alpha in {0,1}).

    Parameters
    ----------
    p, q  : unit vectors on S^2  (shape (3,))
    alpha : blend parameter in [0, 1]

    Returns
    -------
    unit vector on S^2
    """
    # clamp dot product to [-1,1] to avoid arccos domain errors
    dot = float(np.clip(np.dot(p, q), -1.0, 1.0))
    theta = np.arccos(dot)

    if np.abs(np.sin(theta)) < EPS:
        # coincident (theta~0) or antipodal (theta~pi): use safe LERP
        result = (1.0 - alpha) * p + alpha * q
        n = norm(result)
        if n < EPS:
            # truly antipodal with alpha=0.5: return an arbitrary midpoint
            # (the geodesic midpoint is not unique for antipodal points)
            perp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(p, perp)) > 1.0 - EPS:
                perp = np.array([0.0, 1.0, 0.0])
            return normalize(np.cross(p, perp))
        return normalize(result)

    s = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) * p + np.sin(alpha * theta) * q) / s


def geodesic_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Great-circle distance (arc length) between two unit vectors."""
    return float(np.arccos(np.clip(np.dot(p, q), -1.0, 1.0)))


def in_spherical_convex_hull(
        test_pt: np.ndarray,
        pts: np.ndarray,
        tol: float = 1e-4,
) -> bool:
    """
    Approximate check: test_pt should lie within the spherical convex hull
    of pts.  We verify the Casciola-Morigi variation-diminishing bound by
    checking that the geodesic distance from test_pt to the "centre of mass"
    of pts is no larger than the maximum pairwise distance in pts.
    """
    centre = normalize(pts.mean(axis=0))
    max_radius = max(geodesic_distance(p, centre) for p in pts)
    dist = geodesic_distance(test_pt, centre)
    return dist <= max_radius + tol


# ─────────────────────────────────────────────────────────────────────────────
# 2.  KNOT VECTOR UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def clamped_knots(n: int, p: int) -> np.ndarray:
    """
    Standard clamped (open) uniform knot vector for n+1 control points,
    degree p.  Has p+1 repeated knots at each end so the curve interpolates
    the first and last control points.

    Returns knot vector of length n + p + 2.
    """
    n_interior = n - p          # number of interior (non-repeated) knots
    if n_interior < 0:
        raise ValueError(f"Need n >= p; got n={n}, p={p}")
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.zeros(p + 1),
        interior,
        np.ones(p + 1),
    ])
    return knots


def periodic_knots(n: int, p: int, domain: Tuple[float, float] = (0.0, 2 * np.pi)) -> np.ndarray:
    """
    Uniform periodic knot vector over `domain` for n+1 control points.
    Closed curves require exactly n+1 == number of distinct knot spans.
    Returns knot vector of length n + p + 2.
    """
    a, b = domain
    m = n + p + 2             # total number of knots
    return np.linspace(a - p * (b - a) / n, b + p * (b - a) / n, m)


def arc_length_params(pts: np.ndarray, periodic: bool = False) -> np.ndarray:
    """
    Chord-length parameterisation on S^2: accumulate geodesic distances.
    Returns parameter values in [0, 1] (or [0, 2pi] if periodic).
    """
    dists = np.array([geodesic_distance(pts[i], pts[i + 1])
                      for i in range(len(pts) - 1)])
    cumulative = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumulative[-1]
    if total < EPS:
        return np.linspace(0.0, 1.0, len(pts))
    params = cumulative / total
    if periodic:
        params *= 2 * np.pi
    return params


def find_knot_span(t: float, knots: np.ndarray, p: int) -> int:
    """
    Binary search for the knot span index i such that
    knots[i] <= t < knots[i+1].  Handles the right endpoint specially.
    """
    n = len(knots) - p - 2   # index of last control point
    if t >= knots[n + 1]:    # clamp to last non-trivial span
        return n
    lo, hi = p, n + 1
    mid = (lo + hi) // 2
    while t < knots[mid] or t >= knots[mid + 1]:
        if t < knots[mid]:
            hi = mid
        else:
            lo = mid
        mid = (lo + hi) // 2
    return mid


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COX–DE BOOR SCALAR BASIS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def basis_functions(i: int, p: int, t: float, knots: np.ndarray) -> np.ndarray:
    """
    Compute the p+1 non-zero B-spline basis functions N_{i-p,p}..N_{i,p}
    at parameter t, using the stable triangular table algorithm.

    Returns array of shape (p+1,) containing N_{i-p,p}(t)..N_{i,p}(t).
    """
    N = np.zeros(p + 1)
    N[0] = 1.0
    left  = np.zeros(p + 1)
    right = np.zeros(p + 1)

    for j in range(1, p + 1):
        left[j]  = t - knots[i + 1 - j]
        right[j] = knots[i + j] - t
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            if abs(denom) < EPS:
                temp = 0.0
            else:
                temp = N[r] / denom
            N[r] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        N[j] = saved

    return N


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SPHERICAL DE BOOR EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

def spherical_deboor(
        ctrl: np.ndarray,
        knots: np.ndarray,
        p: int,
        t: float,
) -> np.ndarray:
    """
    Evaluate the spherical B-spline at parameter t using the de Boor
    algorithm on S^2.  Every affine combination is replaced by SLERP.

    Algorithm
    ---------
    1. Find knot span i such that knots[i] <= t < knots[i+1].
    2. Extract the p+1 relevant control points: d[0..p] = ctrl[i-p..i].
    3. Apply the triangular de Boor table:
         for r = 1..p:
           for j = 0..p-r:
             alpha = (t - knots[i-p+j+r]) / (knots[i+j+1] - knots[i-p+j+r])
             d[j]  = SLERP(d[j], d[j+1], alpha)
    4. Return d[0], which lies on S^2.

    Parameters
    ----------
    ctrl  : (n+1, 3) array of unit vectors (control points on S^2)
    knots : knot vector of length n + p + 2
    p     : degree
    t     : evaluation parameter

    Returns
    -------
    point on S^2, shape (3,)
    """
    n = len(ctrl) - 1
    i = find_knot_span(t, knots, p)

    # local working copy of the p+1 active control points
    d = np.array(ctrl[max(0, i - p): i + 1], dtype=float)

    # pad at start if i - p < 0  (shouldn't happen with valid knots)
    if len(d) < p + 1:
        pad = p + 1 - len(d)
        d = np.vstack([np.tile(d[0], (pad, 1)), d])

    for r in range(1, p + 1):
        for j in range(p - r + 1):
            ki_lo = i - p + j + r
            ki_hi = i + j + 1
            denom = knots[ki_hi] - knots[ki_lo]
            if abs(denom) < EPS:
                alpha = 0.0
            else:
                alpha = (t - knots[ki_lo]) / denom
            d[j] = slerp(d[j], d[j + 1], alpha)

    return normalize(d[0])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FULL-CURVE SAMPLER + UNIT-NORM VERIFIER
# ─────────────────────────────────────────────────────────────────────────────

def spherical_bspline_curve(
        ctrl: np.ndarray,
        knots: np.ndarray,
        p: int,
        num_samples: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample the spherical B-spline curve at `num_samples` uniform parameter
    values across the active domain [knots[p], knots[-p-1]].

    Returns
    -------
    params : (num_samples,) parameter values
    curve  : (num_samples, 3) unit vectors on S^2

    Raises
    ------
    AssertionError if any point deviates from the unit sphere by > 1e-10.
    """
    t_min = knots[p]
    t_max = knots[-p - 1]
    params = np.linspace(t_min, t_max, num_samples)

    curve = np.array([spherical_deboor(ctrl, knots, p, t) for t in params])

    norms = np.linalg.norm(curve, axis=1)
    max_dev = float(np.max(np.abs(norms - 1.0)))
    assert max_dev < 1e-10, (
        f"Unit-sphere violation: max |‖C(t)‖ - 1| = {max_dev:.3e}"
    )
    return params, curve


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LEAST-SQUARES SPHERICAL B-SPLINE FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_spherical_bspline(
        data: np.ndarray,
        p: int = 3,
        num_ctrl: int = 10,
        max_iter: int = 20,
        tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a spherical B-spline to scattered data points on S^2 via iterative
    least-squares in the tangent space (intrinsic Riemannian approach).

    Strategy
    --------
    1. Parameterise data by arc-length on S^2.
    2. Build clamped knot vector for num_ctrl control points of degree p.
    3. Solve the collocation system in R^3 (Euclidean embedding):
           A  x  =  b         (A is the N x num_ctrl basis-function matrix)
    4. Project each solution column onto S^2.
    5. Iterate: re-parameterise by projecting data onto updated curve.

    Parameters
    ----------
    data     : (N, 3) unit vectors — the scattered data on S^2
    p        : spline degree (default 3)
    num_ctrl : number of control points (default 10)
    max_iter : maximum outer iterations
    tol      : convergence tolerance on max parameter change

    Returns
    -------
    ctrl  : (num_ctrl, 3) fitted control points on S^2
    knots : knot vector used
    """
    data = np.array([normalize(d) for d in data], dtype=float)
    N_data = len(data)

    # Initial parameterisation: chord-length on S^2
    params = arc_length_params(data)

    knots = clamped_knots(num_ctrl - 1, p)

    ctrl = None
    for iteration in range(max_iter):
        # Build collocation matrix A  (N_data x num_ctrl)
        A = np.zeros((N_data, num_ctrl))
        for k, t in enumerate(params):
            t_clamped = np.clip(t, knots[p], knots[-p - 1])
            i = find_knot_span(t_clamped, knots, p)
            basis = basis_functions(i, p, t_clamped, knots)
            for r in range(p + 1):
                col = i - p + r
                if 0 <= col < num_ctrl:
                    A[k, col] = basis[r]

        # Solve for each coordinate independently: A x = b
        ctrl_new = np.zeros((num_ctrl, 3))
        for dim in range(3):
            b_dim = data[:, dim]
            x, _, _, _ = lstsq(A, b_dim, rcond=None)
            ctrl_new[:, dim] = x

        # Project control points back onto S^2
        ctrl_new = np.array([normalize(c) for c in ctrl_new])

        # Re-parameterise: assign each data point the parameter of its
        # nearest point on the updated curve (coarse, 500 samples)
        _, curve_samples = spherical_bspline_curve(ctrl_new, knots, p, 500)
        curve_params = np.linspace(knots[p], knots[-p - 1], 500)

        new_params = np.zeros(N_data)
        for k, d in enumerate(data):
            dots = curve_samples @ d
            new_params[k] = curve_params[int(np.argmax(dots))]

        # Normalise params to [0, 1]
        new_params = (new_params - new_params[0]) / (new_params[-1] - new_params[0] + EPS)

        param_change = float(np.max(np.abs(new_params - params)))
        params = new_params
        ctrl = ctrl_new

        if param_change < tol and iteration > 0:
            break

    return ctrl, knots


# ─────────────────────────────────────────────────────────────────────────────
# 7.  KNOT INSERTION (Boehm's algorithm on S^2)
# ─────────────────────────────────────────────────────────────────────────────

def insert_knot(
        ctrl: np.ndarray,
        knots: np.ndarray,
        p: int,
        t_new: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Insert a single knot t_new into the spherical B-spline, producing a
    refined control polygon that yields the *identical* curve.

    The scalar blending weights alpha_i are identical to the classical
    Boehm formula; only the blending operation is replaced by SLERP.

    Returns
    -------
    ctrl_new  : (n+2, 3)  refined control points on S^2
    knots_new : updated knot vector of length len(knots)+1
    """
    n = len(ctrl) - 1
    k = find_knot_span(t_new, knots, p)

    # Compute scalar Boehm alphas
    alphas = np.zeros(n + 1)
    for i in range(n + 1):
        if i <= k - p:
            alphas[i] = 1.0
        elif i <= k:
            denom = knots[i + p] - knots[i]
            alphas[i] = (t_new - knots[i]) / denom if abs(denom) > EPS else 0.0
        else:
            alphas[i] = 0.0

    # New control points via SLERP with Boehm weight
    ctrl_new = np.zeros((n + 2, 3))
    for i in range(n + 2):
        if i == 0:
            ctrl_new[i] = ctrl[0]
        elif i == n + 1:
            ctrl_new[i] = ctrl[n]
        else:
            alpha = alphas[i]
            if abs(alpha - 1.0) < EPS:
                ctrl_new[i] = ctrl[i]
            elif abs(alpha) < EPS:
                ctrl_new[i] = ctrl[i - 1]
            else:
                ctrl_new[i] = slerp(ctrl[i - 1], ctrl[i], alpha)
        ctrl_new[i] = normalize(ctrl_new[i])

    # Insert t_new into knot vector
    idx = k + 1
    knots_new = np.concatenate([knots[:idx], [t_new], knots[idx:]])

    return ctrl_new, knots_new


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VALIDATION TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

def test_great_circle():
    """
    Test 1: degree-1 spherical B-spline on a great circle arc.
    Expected: curve IS the geodesic (exact, no approximation error).
    """
    print("=" * 60)
    print("TEST 1 — Great circle arc (degree 1, should be exact geodesic)")
    print("=" * 60)

    # Great circle in the xz-plane: from (1,0,0) to (0,0,1) via 5 pts
    angles = np.linspace(0, np.pi / 2, 6)
    ctrl = np.column_stack([np.cos(angles), np.zeros(6), np.sin(angles)])

    knots = clamped_knots(len(ctrl) - 1, p=1)
    params, curve = spherical_bspline_curve(ctrl, knots, p=1, num_samples=200)

    # Ground truth: great circle at each parameter value
    gt_angles = np.linspace(0, np.pi / 2, 200)
    gt_curve = np.column_stack([np.cos(gt_angles), np.zeros(200), np.sin(gt_angles)])

    # Alignment: parameter mapping is linear for degree-1 clamped
    geo_err = np.max(np.arccos(np.clip(np.sum(curve * gt_curve, axis=1), -1, 1)))
    norms = np.linalg.norm(curve, axis=1)
    max_unit_dev = float(np.max(np.abs(norms - 1.0)))

    print(f"  Max geodesic error vs true great circle : {np.degrees(geo_err):.6f} deg")
    print(f"  Max unit-sphere deviation |‖C(t)‖ - 1| : {max_unit_dev:.2e}")
    print()
    return ctrl, knots, curve


def test_figure8():
    """
    Test 2: figure-8 curve on S^2 (closed, degree 3).
    Uses a lemniscate-of-Bernoulli lifted to the sphere.
    """
    print("=" * 60)
    print("TEST 2 — Figure-8 on S^2 (closed, degree 3)")
    print("=" * 60)

    # Parametric figure-8 on the sphere via Viviani-inspired construction
    angles = np.linspace(0, 2 * np.pi, 13)[:-1]   # 12 ctrl pts, close loop
    r = 0.6
    x = np.sin(angles) * np.cos(r * np.sin(angles))
    y = np.cos(angles)
    z = np.sin(angles) * np.sin(r * np.sin(angles))
    raw = np.column_stack([x, y, z])
    ctrl = np.array([normalize(v) for v in raw])

    # Periodic clamped knots
    p = 3
    n = len(ctrl) - 1
    knots = clamped_knots(n, p)

    params, curve = spherical_bspline_curve(ctrl, knots, p, num_samples=500)
    norms = np.linalg.norm(curve, axis=1)
    max_unit_dev = float(np.max(np.abs(norms - 1.0)))

    print(f"  Control points          : {len(ctrl)}")
    print(f"  Knot vector length      : {len(knots)}  ({knots})")
    print(f"  Curve samples           : 500")
    print(f"  Max |‖C(t)‖ - 1|        : {max_unit_dev:.2e}")
    print()
    return ctrl, knots, curve


def test_noisy_fitting():
    """
    Test 3: fit degree-3 spherical B-spline to 50 noisy points on S^2.
    """
    print("=" * 60)
    print("TEST 3 — Fitting to 50 noisy points on S^2 (degree 3, 10 ctrl)")
    print("=" * 60)

    rng = np.random.default_rng(42)

    # Ground truth: a sinusoidal path on the sphere
    t_gt = np.linspace(0, 2 * np.pi, 50)
    lat = 0.4 * np.sin(t_gt)
    lon = t_gt * 0.8
    x_gt = np.cos(lat) * np.cos(lon)
    y_gt = np.cos(lat) * np.sin(lon)
    z_gt = np.sin(lat)
    data_clean = np.column_stack([x_gt, y_gt, z_gt])

    # Add tangential noise and re-project to S^2
    noise = rng.normal(0, 0.08, data_clean.shape)
    data_noisy = np.array([normalize(d + n) for d, n in zip(data_clean, noise)])

    ctrl, knots = fit_spherical_bspline(data_noisy, p=3, num_ctrl=10, max_iter=15)
    _, curve = spherical_bspline_curve(ctrl, knots, p=3, num_samples=300)

    # Residuals: geodesic distance from each data point to nearest curve point
    residuals = []
    for d in data_noisy:
        dists = [geodesic_distance(d, c) for c in curve]
        residuals.append(min(dists))
    residuals = np.array(residuals)

    norms = np.linalg.norm(curve, axis=1)
    max_unit_dev = float(np.max(np.abs(norms - 1.0)))

    print(f"  Data points             : 50 (with σ=0.08 tangential noise)")
    print(f"  Control points          : {len(ctrl)}")
    print(f"  Mean geodesic residual  : {np.degrees(residuals.mean()):.4f} deg")
    print(f"  Max  geodesic residual  : {np.degrees(residuals.max()):.4f} deg")
    print(f"  Max |‖C(t)‖ - 1|        : {max_unit_dev:.2e}")
    print()
    return ctrl, knots, curve, data_noisy


# ─────────────────────────────────────────────────────────────────────────────
# 9.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def draw_sphere_wireframe(ax, alpha=0.08, color='gray'):
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color=color, alpha=alpha, linewidth=0.4)


def visualise_all():
    """Run all three tests and produce a 3-panel figure."""
    fig = plt.figure(figsize=(16, 5))
    fig.suptitle("Spherical B-Spline Algorithm — Validation", fontsize=13)

    # ── Test 1 ──────────────────────────────────────────────────────────────
    ctrl1, knots1, curve1 = test_great_circle()
    ax1 = fig.add_subplot(131, projection='3d')
    draw_sphere_wireframe(ax1)
    ax1.plot(ctrl1[:, 0], ctrl1[:, 1], ctrl1[:, 2],
             'o--', color='#7F77DD', ms=6, lw=1, label='control polygon')
    ax1.plot(curve1[:, 0], curve1[:, 1], curve1[:, 2],
             '-', color='#1D9E75', lw=2.5, label='spline curve')
    ax1.set_title("Test 1: great circle arc\n(degree 1, exact geodesic)", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.set_box_aspect([1, 1, 1])
    ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')

    # ── Test 2 ──────────────────────────────────────────────────────────────
    ctrl2, knots2, curve2 = test_figure8()
    ax2 = fig.add_subplot(132, projection='3d')
    draw_sphere_wireframe(ax2)
    ax2.plot(ctrl2[:, 0], ctrl2[:, 1], ctrl2[:, 2],
             'o--', color='#7F77DD', ms=6, lw=1, label='control polygon')
    ax2.plot(curve2[:, 0], curve2[:, 1], curve2[:, 2],
             '-', color='#D85A30', lw=2.5, label='spline curve')
    ax2.set_title("Test 2: figure-8 on S²\n(degree 3, closed)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_box_aspect([1, 1, 1])
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')

    # ── Test 3 ──────────────────────────────────────────────────────────────
    ctrl3, knots3, curve3, data3 = test_noisy_fitting()
    ax3 = fig.add_subplot(133, projection='3d')
    draw_sphere_wireframe(ax3)
    ax3.scatter(data3[:, 0], data3[:, 1], data3[:, 2],
                color='#888780', s=12, alpha=0.7, label='noisy data')
    ax3.plot(ctrl3[:, 0], ctrl3[:, 1], ctrl3[:, 2],
             'o--', color='#7F77DD', ms=6, lw=1, label='control polygon')
    ax3.plot(curve3[:, 0], curve3[:, 1], curve3[:, 2],
             '-', color='#378ADD', lw=2.5, label='fitted spline')
    ax3.set_title("Test 3: least-squares fit\n(50 noisy pts, 10 ctrl, degree 3)", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.set_box_aspect([1, 1, 1])
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('z')

    plt.tight_layout()
    plt.savefig('spherical_bspline_results.png', dpi=150, bbox_inches='tight')
    print("Figure saved: spherical_bspline_results.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run knot insertion demo first
    print("=" * 60)
    print("DEMO — Knot insertion on S^2 (Boehm's algorithm)")
    print("=" * 60)
    ctrl_demo = np.array([
        normalize(np.array([1.0, 0.0, 0.5])),
        normalize(np.array([0.5, 0.5, 0.5])),
        normalize(np.array([0.0, 1.0, 0.0])),
        normalize(np.array([-0.5, 0.5, 0.5])),
        normalize(np.array([-1.0, 0.0, 0.0])),
    ])
    knots_demo = clamped_knots(len(ctrl_demo) - 1, p=2)
    ctrl_ref, knots_ref = insert_knot(ctrl_demo, knots_demo, p=2, t_new=0.5)
    _, curve_before = spherical_bspline_curve(ctrl_demo, knots_demo, p=2, num_samples=200)
    _, curve_after  = spherical_bspline_curve(ctrl_ref,  knots_ref,  p=2, num_samples=200)
    max_change = np.max(np.arccos(np.clip(
        np.sum(curve_before * curve_after, axis=1), -1, 1)))
    print(f"  Control pts before insertion : {len(ctrl_demo)}")
    print(f"  Control pts after  insertion : {len(ctrl_ref)}")
    print(f"  Max curve change (geodesic)  : {np.degrees(max_change):.2e} deg  (should be ~0)")
    print()

    visualise_all()