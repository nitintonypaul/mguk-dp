"""
Random track generator (NOT a real circuit)
A closed loop is built from random control points placed around a
circle (angularly sorted, radially jittered -> avoids self-intersection)
and connected with a periodic Catmull-Rom spline. This produces a mix
of long straights and tight hairpins, similar in spirit to simple
procedural-track generators used in racing-line RL environments.
Will implement actual track fetching later on.

Also includes  A simple (non-optimal) driving line: centerline 
offset toward the inside of a corner, scaled by local curvature. 
Purely a visual heuristic "racing line", not a minimum-curvature 
optimization. Advanced global optimization will be implemented later.
"""

import numpy as np

# ---------- HELPER FUNCTIONS ----------

def _catmull_rom_periodic(pts, samples_per_seg=60):
    """pts: (n,2) control points, closed loop. Returns fine (x,y) path."""
    n = len(pts)
    u = np.linspace(0, 1, samples_per_seg, endpoint=False)
    U = np.stack([np.ones_like(u), u, u**2, u**3], axis=1)   # (S,4)
    basis = 0.5 * np.array([[0, 2, 0, 0],
                             [-1, 0, 1, 0],
                             [2, -5, 4, -1],
                             [-1, 3, -3, 1]])                 # (4,4)
    M = U @ basis                                             # (S,4)

    xs, ys = [], []
    for i in range(n):
        p0, p1, p2, p3 = pts[(i - 1) % n], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
        seg_pts = np.stack([p0, p1, p2, p3], axis=0)          # (4,2)
        seg = M @ seg_pts                                      # (S,2)
        xs.append(seg[:, 0])
        ys.append(seg[:, 1])
    return np.concatenate(xs), np.concatenate(ys)

def _unit_normals(x, y):
    """Left-hand unit normal at each point of an (open) polyline, via
    simple centered finite differences on the tangent direction."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    return -dy / norm, dx / norm
 
 
def _relax_alpha(x, y, nx, ny, max_offset, n_iters=300, step_size=0.5,
                  tol=1e-4):
    """Discrete curve-shortening relaxation, projected onto the fixed
    centerline normals. For each point, repeatedly nudge its lateral
    offset alpha_i toward whatever would place it on the straight chord
    between its two neighbours (the local curvature-minimizing move),
    then clip to stay within the track corridor. Iterating this to
    convergence is a cheap stand-in for the full minimum-curvature QP:
    no solver dependency, and it removes single-point curvature spikes
    (the source of the jagged, mid-straight false-braking-zone artifact)
    without needing a global optimization.
    """
    n = len(x)
    alpha = np.zeros(n)
    px, py = x.copy(), y.copy()
 
    for _ in range(n_iters):
        px_prev, py_prev = np.roll(px, 1), np.roll(py, 1)
        px_next, py_next = np.roll(px, -1), np.roll(py, -1)
 
        mid_x = 0.5 * (px_prev + px_next)
        mid_y = 0.5 * (py_prev + py_next)
 
        # displacement toward the straightening midpoint, projected onto
        # the (fixed) normal direction - only lateral moves are legal
        d_alpha = (mid_x - px) * nx + (mid_y - py) * ny
 
        alpha_new = np.clip(alpha + step_size * d_alpha, -max_offset, max_offset)
        delta = np.max(np.abs(alpha_new - alpha))
        alpha = alpha_new
 
        px = x + alpha * nx
        py = y + alpha * ny
 
        if delta < tol:
            break
 
    return alpha

# ---------- CORE FUNCTIONS ----------

def generate_track(seed=None, R0=500.0, n_ctrl=16, radius_jitter=0.55,
                    samples_per_seg=60):
    rgen = np.random.default_rng(seed)

    # angularly sorted control points (guarantees a simple, non self
    # intersecting loop) with strong radial jitter -> some tight corners
    base_theta = np.linspace(0, 2 * np.pi, n_ctrl, endpoint=False)
    theta_jitter = rgen.uniform(-0.5, 0.5, n_ctrl) * (np.pi / n_ctrl) * 0.6
    ctrl_theta = base_theta + theta_jitter
    ctrl_r = R0 * (1.0 + rgen.uniform(-radius_jitter, radius_jitter, n_ctrl))
    ctrl_pts = np.stack([ctrl_r * np.cos(ctrl_theta),
                          ctrl_r * np.sin(ctrl_theta)], axis=1)

    x, y = _catmull_rom_periodic(ctrl_pts, samples_per_seg=samples_per_seg)
    n_fine = len(x)
    t_param = np.arange(n_fine)                    # uniform parameter index

    # numerical derivatives wrt parameter (curvature formula is invariant
    # to the choice of regular parameterization)
    dx = np.gradient(x, t_param, edge_order=2)
    dy = np.gradient(y, t_param, edge_order=2)
    d2x = np.gradient(dx, t_param, edge_order=2)
    d2y = np.gradient(dy, t_param, edge_order=2)

    speed = np.sqrt(dx**2 + dy**2) + 1e-9
    kappa = (dx * d2y - dy * d2x) / speed**3

    ds_fine = np.concatenate([[0.0], np.cumsum(
        0.5 * (speed[:-1] + speed[1:]))])
    DL = ds_fine[-1] + 0.5 * (speed[-1] + speed[0])   # closes the loop

    return dict(theta=t_param, x=x, y=y, kappa=kappa, s=ds_fine, DL=DL)

def resample_equal_arclength(x, y, n_seg):
    """Reparametrize a closed polyline (x[-1] ~= x[0]) to n_seg equal
    arc-length segments. Returns x, y, and a scalar ds - same convention
    as discretize_track's output."""
    seg_lengths = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    L = s[-1]
    s_new = np.linspace(0, L, n_seg + 1)
    x_new = np.interp(s_new, s, x)
    y_new = np.interp(s_new, s, y)
    return x_new, y_new, L / n_seg


def line_curvature(x, y, ds):
    """Curvature of an equal-arc-length-spaced polyline, using ds as the
    uniform step for the finite-difference derivatives."""
    dx = np.gradient(x, ds)
    dy = np.gradient(y, ds)
    d2x = np.gradient(dx, ds)
    d2y = np.gradient(dy, ds)
    speed = np.sqrt(dx**2 + dy**2) + 1e-9
    return (dx * d2y - dy * d2x) / speed**3

def discretize_track(track, n_seg=60):
    """Resample the track into n_seg segments of EQUAL arc length ds,
    as required by the spatial discretization in the PDF (Section 1)."""
    s = track["s"]
    DL = track["DL"]
    s_bounds = np.linspace(0, DL, n_seg + 1)
    theta_of_s = np.interp(s_bounds, s, track["theta"])

    x_b = np.interp(s_bounds, s, track["x"])
    y_b = np.interp(s_bounds, s, track["y"])
    kappa_b = np.interp(s_bounds, s, track["kappa"])

    ds = DL / n_seg
    return dict(s_bounds=s_bounds, x=x_b, y=y_b, kappa=kappa_b, ds=ds,
                n_seg=n_seg, DL=DL)
 
def driving_line(disc, track_half_width=6.0, n_iters=300, step_size=0.5):
    """Minimum-curvature-style driving line via iterative relaxation
    (discrete curve-shortening flow), rather than a per-segment offset
    computed purely from that segment's own local curvature. Solving it
    this way (jointly, with neighbour coupling) is what prevents a single
    noisy/near-zero curvature reading on a straight from producing an
    isolated offset spike - each point's offset is now pulled toward
    whatever keeps the WHOLE resulting line smooth, not just itself.
    """
    x_full, y_full = disc["x"], disc["y"]
    n = len(x_full) - 1                      # drop the duplicated closing point
    x, y = x_full[:n], y_full[:n]
 
    nx, ny = _unit_normals(x, y)
    max_offset = 0.8 * track_half_width
 
    alpha = _relax_alpha(x, y, nx, ny, max_offset,
                          n_iters=n_iters, step_size=step_size)
 
    line_x = x + alpha * nx
    line_y = y + alpha * ny
 
    # re-close the loop to match the (n_seg+1)-length convention used
    # elsewhere for disc["x"] / disc["y"]
    line_x = np.append(line_x, line_x[0])
    line_y = np.append(line_y, line_y[0])
    return line_x, line_y