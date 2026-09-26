"""
Stochastic Dynamic Programming for Optimal Power-Split Control in F1
=====================================================================
Demo implementation of the formulation in the project PDFs:
  - Eq (1): stage-additive lap-time objective, min sum ell(v_t,u_t)
  - Eq (2): net force Fn = F_ICE(v) + u*Pmax_MGUK/v - Fdrag(v,d) - Frolling
  - Eq (3): velocity update, clipped by segment vmax(d)
  - Eq (4): battery SoC update with regen harvesting + stochastic noise
  - Eq (5): stage cost ell(v,u) = ds / v
  - Eq (6): Bellman backward recursion, V_T(v,b) = 0 for all (v,b)
  - Extra constraints supplied separately:
        b_t = 0  =>  u_t = 0
        0 <= b_t <= b_max = 4 MJ
        V_T(v,b) = 0  for all (v,b)

This is a BALLPARK demo (not a competition lap-sim): track is randomly
generated (no real circuit), physical constants are rough public-domain
figures for 2026-era F1 cars, and the "driving line" is a simple
curvature-based heuristic (not an optimized minimum-curvature line).

Author: demo for OR project by Nitin Tony Paul
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import qss

rng = np.random.default_rng(100)

# ---------------------------------------------------------------------------
# 0. Physical constants (ballpark, 2026-era F1 hybrid power unit)
# ---------------------------------------------------------------------------
m          = 768.0        # kg, ballpark 2026 minimum car+driver mass
g          = 9.81         # m/s^2
mu_tire    = 1.6          # dry slick tyre-road friction coefficient (ballpark)
rho_air    = 1.225        # kg/m^3
CdA        = 1.20         # m^2, drag area (Cd * frontal area), ballpark
ClA        = 3.50         # m^2, downforce area (Cl * frontal area), ballpark
Crr        = 0.015        # rolling resistance coefficient, ballpark
F_rolling  = Crr * m * g  # N, constant rolling resistance force (eq. 2)

P_ice_max   = 400e3       # W, ballpark ICE peak power under 2026 regs (~536 hp)
P_mguk_max  = 350e3       # W, ballpark MGU-K peak power under 2026 regs
                           # (2026 regs roughly triple current ~120 kW MGU-K)
F_trac_max  = 15_000.0    # N, ballpark max tyre-limited tractive force (low speed)

b_max      = 4.0e6        # J = 4 MJ (given constraint: 0 <= b_t <= b_max)
eta_regen  = 0.6          # regen harvesting efficiency, ballpark

v_top      = 100.0        # m/s (~360 km/h) absolute top-speed cap
v_min      = 15.0         # m/s floor, avoids division singularities

# ---------------------------------------------------------------------------
# 1. Random track generator (NOT a real circuit)
#    A closed loop is built from random control points placed around a
#    circle (angularly sorted, radially jittered -> avoids self-intersection)
#    and connected with a periodic Catmull-Rom spline. This produces a mix
#    of long straights and tight hairpins, similar in spirit to simple
#    procedural-track generators used in racing-line RL environments.
#    No third-party track library is used.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 2. Theoretical (ballpark) v_max per segment from corner curvature,
#    including a simple downforce fixed-point iteration:
#        v_max^2 * m / r = mu * (m*g + 0.5*rho*ClA*v_max^2)
# ---------------------------------------------------------------------------
def segment_vmax(kappa, r_min=12.0):
    """Closed-form solve of  v^2*m/r = mu*(m*g + 0.5*rho*ClA*v^2)  for v,
    i.e.  v = sqrt( mu*m*g / (m/r - 0.5*mu*rho*ClA) ), capped by v_top.
    If downforce-generated grip alone would out-run the required
    centripetal force at all speeds, the corner is effectively
    speed-unlimited by this model and we just cap at v_top."""
    r = 1.0 / np.maximum(np.abs(kappa), 1.0 / 5000.0)   # curvature -> radius
    r = np.maximum(r, r_min)                             # realistic min radius
    denom = m / r - 0.5 * mu_tire * rho_air * ClA
    vmax = np.where(denom > 1e-6,
                     np.sqrt(mu_tire * m * g / np.maximum(denom, 1e-6)),
                     v_top)
    return np.minimum(vmax, v_top)


# ---------------------------------------------------------------------------
# 3. A simple (non-optimal) driving line: centerline offset toward the
#    inside of a corner, scaled by local curvature. Purely a visual /
#    heuristic "racing line", not a minimum-curvature optimization.
# ---------------------------------------------------------------------------
def driving_line(disc, track_half_width=6.0):
    x, y, kappa = disc["x"], disc["y"], disc["kappa"]
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    nx, ny = -dy / norm, dx / norm            # unit normal

    kappa_ref = 1.0 / 150.0
    max_offset = 0.8 * track_half_width
    offset = -max_offset * np.tanh(kappa / kappa_ref)

    line_x = x + offset * nx
    line_y = y + offset * ny
    return line_x, line_y


# ---------------------------------------------------------------------------
# 4. Force model (Eq. 2)
# ---------------------------------------------------------------------------
def F_ice(v):
    """Traction-limited at low speed, power-limited at high speed."""
    return np.minimum(F_trac_max, P_ice_max / np.maximum(v, 1.0))

def F_drag(v, drag_mult):
    return 0.5 * rho_air * CdA * drag_mult * v**2


# ---------------------------------------------------------------------------
# 5. Braking-zone flags & regen energy potential R(d_t) (Eq. 4)
# ---------------------------------------------------------------------------
def braking_zones_and_regen(vmax, ds):
    n_seg = len(vmax) - 1
    BZ = np.zeros(n_seg, dtype=bool)
    R = np.zeros(n_seg)
    for t in range(n_seg):
        if vmax[t + 1] < vmax[t] - 2.0:      # must shed speed -> braking zone
            BZ[t] = True
            v_avg = 0.5 * (vmax[t] + vmax[t + 1])
            t_brake = ds / max(v_avg, 1.0)
            energy_available = 0.5 * m * (vmax[t]**2 - vmax[t + 1]**2)
            R[t] = min(P_mguk_max * t_brake, energy_available)
    return BZ, R


def drag_multiplier(kappa, kappa_thresh=1.0 / 300.0):
    """Straights (low curvature) run a lower-drag / DRS-like aero setting."""
    return np.where(np.abs(kappa) < kappa_thresh, 0.80, 1.00)


# ---------------------------------------------------------------------------
# 6. Bilinear interpolation, vectorized (grid -> arbitrary query shape)
# ---------------------------------------------------------------------------
def interp2d_vec(x_grid, y_grid, table, xq, yq):
    xq, yq = np.broadcast_arrays(xq, yq)
    xq_c = np.clip(xq, x_grid[0], x_grid[-1])
    yq_c = np.clip(yq, y_grid[0], y_grid[-1])

    ix = np.clip(np.searchsorted(x_grid, xq_c) - 1, 0, len(x_grid) - 2)
    iy = np.clip(np.searchsorted(y_grid, yq_c) - 1, 0, len(y_grid) - 2)

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    wx = (xq_c - x0) / (x1 - x0)
    wy = (yq_c - y0) / (y1 - y0)

    f00 = table[ix, iy]
    f10 = table[ix + 1, iy]
    f01 = table[ix, iy + 1]
    f11 = table[ix + 1, iy + 1]

    return (f00 * (1 - wx) * (1 - wy) + f10 * wx * (1 - wy) +
            f01 * (1 - wx) * wy + f11 * wx * wy)


# ---------------------------------------------------------------------------
# 7. Stochastic Bellman backward recursion (Eq. 6) over a (v, b) grid
# ---------------------------------------------------------------------------
def solve_bellman(disc, vmax, BZ, R, n_v=31, n_b=25, n_u=11):
    n_seg = disc["n_seg"]
    ds = disc["ds"]
    kappa = disc["kappa"]
    drag_mult = drag_multiplier(kappa)

    v_grid = np.linspace(v_min, v_top, n_v)
    b_grid = np.linspace(0.0, b_max, n_b)
    u_grid = np.linspace(0.0, 1.0, n_u)

    # 3-point discrete noise approx to N(0, sigma^2) matching mean/variance
    sigma_noise = 0.01 * b_max
    noise_vals = np.array([-np.sqrt(3) * sigma_noise, 0.0, np.sqrt(3) * sigma_noise])
    noise_wts = np.array([1 / 6, 4 / 6, 1 / 6])

    V = np.zeros((n_seg + 1, n_v, n_b))          # V[n_seg] = 0  (Eq. 6 term. cond.)
    policy_u = np.zeros((n_seg, n_v, n_b))

    v_col = v_grid.reshape(n_v, 1, 1, 1)
    u_row = u_grid.reshape(1, 1, n_u, 1)
    b_pl = b_grid.reshape(1, n_b, 1, 1)
    eps = noise_vals.reshape(1, 1, 1, -1)

    dt_arr = ds / v_col                                    # (Nv,1,1,1)
    Fice = F_ice(v_col)                                     # (Nv,1,1,1)

    for t in reversed(range(n_seg)):
        V_next = V[t + 1]                                   # (Nv, Nb)

        Fmguk = u_row * P_mguk_max / v_col                   # (Nv,1,Nu,1)
        Fdrag = F_drag(v_col, drag_mult[t])                  # (Nv,1,1,1)
        Fn = Fice + Fmguk - Fdrag - F_rolling                # (Nv,1,Nu,1)

        v_next = np.minimum(v_col + Fn / m * dt_arr, vmax[t + 1])
        v_next = np.clip(v_next, v_min, v_top)               # (Nv,1,Nu,1)

        b_deploy = u_row * P_mguk_max * dt_arr               # (Nv,1,Nu,1)
        regen = float(BZ[t]) * eta_regen * R[t]

        b_next = b_pl - b_deploy + regen + eps               # (Nv,Nb,Nu,Nk)
        b_next = np.clip(b_next, 0.0, b_max)

        v_next_b = np.broadcast_to(v_next, b_next.shape)
        Vq = interp2d_vec(v_grid, b_grid, V_next, v_next_b, b_next)  # (Nv,Nb,Nu,Nk)

        EV = np.tensordot(Vq, noise_wts, axes=([3], [0]))    # (Nv,Nb,Nu)

        ell = (ds / v_grid).reshape(n_v, 1, 1)               # Eq. 5 stage cost
        cost = ell + EV                                      # (Nv,Nb,Nu)

        # Constraint: b_t = 0  =>  u_t = 0
        cost[:, 0, 1:] = np.inf

        V[t] = np.min(cost, axis=2)
        policy_u[t] = u_grid[np.argmin(cost, axis=2)]

    return dict(V=V, policy_u=policy_u, v_grid=v_grid, b_grid=b_grid,
                u_grid=u_grid, drag_mult=drag_mult,
                noise_vals=noise_vals, noise_wts=noise_wts)


# ---------------------------------------------------------------------------
# 8. Forward simulation using the optimal policy (deterministic or
#    Monte-Carlo with battery noise, to show the "stochastic" behaviour)
# ---------------------------------------------------------------------------
def simulate(disc, vmax, BZ, R, sol, v0=None, b0=None, stochastic=False, rng=None):
    n_seg = disc["n_seg"]
    ds = disc["ds"]
    drag_mult = sol["drag_mult"]
    v_grid, b_grid = sol["v_grid"], sol["b_grid"]

    v = vmax[0] if v0 is None else v0
    b = 0.5 * b_max if b0 is None else b0

    v_hist = [v]
    b_hist = [b]
    u_hist = []
    t_hist = [0.0]
    total_time = 0.0

    for t in range(n_seg):
        # bilinear-interpolate the optimal control off the DP policy grid
        u = float(interp2d_vec(v_grid, b_grid, sol["policy_u"][t],
                                np.array(v), np.array(b)))
        if b <= 1e-6:
            u = 0.0

        dt_seg = ds / max(v, 1.0)
        Fn = (F_ice(v) + u * P_mguk_max / max(v, 1.0)
              - F_drag(v, drag_mult[t]) - F_rolling)
        v_next = min(v + Fn / m * dt_seg, vmax[t + 1])
        v_next = float(np.clip(v_next, v_min, v_top))

        noise = 0.0
        if stochastic:
            noise = rng.normal(0.0, 0.01 * b_max)
        regen = eta_regen * R[t] if BZ[t] else 0.0
        b_next = np.clip(b - u * P_mguk_max * dt_seg + regen + noise, 0.0, b_max)

        total_time += dt_seg
        v, b = v_next, float(b_next)

        v_hist.append(v)
        b_hist.append(b)
        u_hist.append(u)
        t_hist.append(total_time)

    return dict(v=np.array(v_hist), b=np.array(b_hist), u=np.array(u_hist),
                t=np.array(t_hist), lap_time=total_time)


# ---------------------------------------------------------------------------
# 9. Run the full demo
# ---------------------------------------------------------------------------
def main():
    track = generate_track(seed=100)
    disc = discretize_track(track, n_seg=1000)

    # ----- QSS v_max computation -----
    vehicle = qss.VehicleParams(mass=m, mu=mu_tire, rho=rho_air, cl_a=ClA, v_top=v_top)
    vmax_seg = qss.compute_qss_vmax(disc["kappa"], disc["ds"], vehicle,
                                 a_accel_max=6.0, a_brake_max=45.0)
    # ---------------------------------E
    
    BZ, R = braking_zones_and_regen(vmax_seg, disc["ds"])
    line_x, line_y = driving_line(disc)

    print(f"Track length DL              : {disc['DL']:8.1f} m")
    print(f"Number of segments N         : {disc['n_seg']:8d}")
    print(f"Segment length ds            : {disc['ds']:8.2f} m")
    print(f"Braking zones identified     : {int(BZ.sum()):8d}")
    print("Solving stochastic Bellman recursion (this may take a few seconds)...")

    sol = solve_bellman(disc, vmax_seg, BZ, R, n_v=31, n_b=25, n_u=11)

    sim = simulate(disc, vmax_seg, BZ, R, sol, stochastic=False)
    print(f"\nOptimal (deterministic) lap time : {sim['lap_time']:.2f} s")

    # Monte-Carlo forward rollouts under battery noise to show the
    # stochastic character of the problem
    n_mc = 200
    mc_rng = np.random.default_rng(1)
    mc_times = np.array([
        simulate(disc, vmax_seg, BZ, R, sol, stochastic=True, rng=mc_rng)["lap_time"]
        for _ in range(n_mc)
    ])
    print(f"Monte-Carlo ({n_mc} runs) lap time : "
          f"mean {mc_times.mean():.2f} s, std {mc_times.std():.3f} s")

    # ---------------- plots ----------------
    s_mid = 0.5 * (disc["s_bounds"][:-1] + disc["s_bounds"][1:])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) track + driving line, colored by theoretical vmax
    ax = axes[0, 0]
    sc = ax.scatter(disc["x"], disc["y"], c=np.append(vmax_seg[:-1], vmax_seg[-1]),
                     cmap="viridis", s=10, label="centerline (vmax)")
    ax.plot(line_x, line_y, color="red", lw=1.5, label="driving line")
    ax.set_aspect("equal")
    ax.set_title("Randoml track + driving line")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(sc, ax=ax, label=r"theoretical $v_{max}$ (m/s)")

    # (b) velocity profile: theoretical ceiling vs DP-optimal trajectory
    ax = axes[0, 1]
    ax.plot(disc["s_bounds"], vmax_seg, "k--", label=r"$v_{max}(d)$ ceiling")
    ax.plot(disc["s_bounds"], sim["v"], "b-", label="optimal v_t (DP)")
    ax.set_xlabel("distance s (m)")
    ax.set_ylabel("velocity (m/s)")
    ax.set_title("Velocity profile")
    ax.legend(fontsize=8)

    # (c) control (MGU-K deployment fraction) + braking zones
    ax = axes[1, 0]
    ax.step(s_mid, sim["u"], where="mid", color="tab:orange", label=r"$u_t$ (deployment)")
    ax.fill_between(disc["s_bounds"][:-1], 0, 1, where=BZ, step="post",
                     color="grey", alpha=0.3, label="braking zone")
    ax.set_xlabel("distance s (m)")
    ax.set_ylabel(r"$u_t$ (throttle/deploy fraction)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Optimal MGU-K power-split policy")
    ax.legend(fontsize=8)

    # (d) battery SoC profile
    ax = axes[1, 1]
    ax.plot(disc["s_bounds"], sim["b"] / 1e6, "g-")
    ax.axhline(b_max / 1e6, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("distance s (m)")
    ax.set_ylabel("battery energy (MJ)")
    ax.set_title(r"Battery state of charge $b_t$")

    fig.suptitle(f"F1 power-split DP demo  |  lap time = {sim['lap_time']:.2f} s "
                 f"(MC mean {mc_times.mean():.2f} +/- {mc_times.std():.2f} s)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("f1_dp_demo.png", dpi=150)

    # separate histogram figure for the Monte-Carlo lap-time distribution
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(mc_times, bins=25, color="tab:blue", alpha=0.8)
    ax2.axvline(sim["lap_time"], color="red", ls="--", label="deterministic optimum")
    ax2.set_xlabel("lap time (s)")
    ax2.set_ylabel("count")
    ax2.set_title(f"Monte-Carlo lap-time distribution ({n_mc} stochastic rollouts)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("f1_dp_montecarlo.png", dpi=150)

    print("\nSaved plots:")
    print("f1_dp_demo.png")
    print("f1_dp_montecarlo.png")


if __name__ == "__main__":
    main()