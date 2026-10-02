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


IMMEDIATE NEXT STEPS:
- Compute QSS on driving line instead of the circuit
- Construct `models` module and implement dynamics
- Separate plotting engine for visual customization
"""
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils import qss, gettrack as gt, dynamics 

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

P_ice_max   = 400e3       # W, ballpark ICE peak power under 2026 regs (~536 hp)
P_mguk_max  = 350e3       # W, ballpark MGU-K peak power under 2026 regs
                           # (2026 regs roughly triple current ~120 kW MGU-K)
F_trac_max  = 15_000.0    # N, ballpark max tyre-limited tractive force (low speed)

b_max      = 4.0e6        # J = 4 MJ (given constraint: 0 <= b_t <= b_max)
eta_regen  = 0.6          # regen harvesting efficiency, ballpark

v_top      = 100.0        # m/s (~360 km/h) absolute top-speed cap
v_min      = 15.0         # m/s floor, avoids division singularities

# ---------------------------------------------------------------------------
#    Bilinear interpolation, vectorized (grid -> arbitrary query shape)
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
#    Stochastic Bellman backward recursion (Eq. 6) over a (v, b) grid
# ---------------------------------------------------------------------------
def solve_bellman(disc, vmax, BZ, R, n_v=31, n_b=25, n_u=11):
    n_seg = disc["n_seg"]
    ds = disc["ds"]
    kappa = disc["kappa"]
    drag_mult = dynamics.drag_multiplier(kappa)

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
    Fice = dynamics.F_ice(F_trac_max, P_ice_max, v_col)                                     # (Nv,1,1,1)

    for t in reversed(range(n_seg)):
        V_next = V[t + 1]                                   # (Nv, Nb)

        Fmguk = dynamics.F_MGUK(P_mguk_max, u_row, v_col)                        #(Nv,1,Nu,1)
        Fdrag = dynamics.F_drag(rho_air, CdA, v_col, drag_mult[t])                  # (Nv,1,1,1)
        Fn = Fice + Fmguk - Fdrag - dynamics.F_rolling(Crr, m, g)                # (Nv,1,Nu,1)

        # ---- NEXT STEP VELOCITY ----
        v_next = np.minimum(v_col + Fn / m * dt_arr, vmax[t + 1])
        v_next = np.clip(v_next, v_min, v_top)               # (Nv,1,Nu,1)

        # ---- NEXT STEP BATTERY ----
        b_deploy = u_row * P_mguk_max * dt_arr               # (Nv,1,Nu,1)
        regen = float(BZ[t]) * eta_regen * R[t]

        b_next = b_pl - b_deploy + regen + eps               # (Nv,Nb,Nu,Nk)
        b_next = np.clip(b_next, 0.0, b_max)

        v_next_b = np.broadcast_to(v_next, b_next.shape)
        Vq = interp2d_vec(v_grid, b_grid, V_next, v_next_b, b_next)  # (Nv,Nb,Nu,Nk)

        EV = np.tensordot(Vq, noise_wts, axes=([3], [0]))    # (Nv,Nb,Nu)

        # ---- PER STEP COST ----
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
#    Forward simulation using the optimal policy (deterministic or
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
        Fn = (dynamics.F_ice(F_trac_max, P_ice_max, v) + u * P_mguk_max / max(v, 1.0) - dynamics.F_drag(rho_air, CdA, v, drag_mult[t]) - dynamics.F_rolling(Crr, m, g))
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
#    Run the full demo
# ---------------------------------------------------------------------------
def main():

    track = gt.generate_track(seed=1)
    disc = gt.discretize_track(track, n_seg=1000)

    line_x, line_y = gt.driving_line(disc)
    line_x, line_y, ds_line = gt.resample_equal_arclength(line_x, line_y, disc["n_seg"])
    disc["kappa"] = gt.line_curvature(line_x, line_y, ds_line)
    disc["ds"] = ds_line

    # ----- twopass QSS v_max computation -----
    vehicle = qss.VehicleParams(mass=m, mu=mu_tire, rho=rho_air, cl_a=ClA, v_top=v_top)
    vmax_seg = qss.compute_qss_vmax(disc["kappa"], disc["ds"], vehicle, a_accel_max=6.0, a_brake_max=45.0)
    # ---------------------------------

    BZ, R = dynamics.braking_zones_and_regen(m, P_mguk_max, vmax_seg, disc["ds"])

    print(f"Track length                    : {disc['DL']:8.1f} m")
    print(f"Number of segments              : {disc['n_seg']:8d}")
    print(f"Segment length (driving line)   : {disc['ds']:8.2f} m")
    print(f"Braking zones identified        : {int(BZ.sum()):8d}")
    print("Solving stochastic Bellman recursion (this may take a few seconds)...")

    sol = solve_bellman(disc, vmax_seg, BZ, R, n_v=31, n_b=25, n_u=11)

    sim = simulate(disc, vmax_seg, BZ, R, sol, stochastic=False)
    print(f"\nOptimal (deterministic) lap time : {sim['lap_time']:.2f} s")

    # Monte-Carlo forward rollouts under battery noise to show the
    # stochastic character of the problem
    n_mc = 10
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

    ax = axes[0, 0]
    ax.plot(disc["x"][0], disc["y"][0], marker="s", color="white", markeredgecolor="black", markersize=10, zorder=3, label="start/finish")

    track_half_width = 15.0
    nx, ny = gt._unit_normals(disc["x"][:-1], disc["y"][:-1])
    left_x  = disc["x"][:-1] + track_half_width * nx
    left_y  = disc["y"][:-1] + track_half_width * ny
    right_x = disc["x"][:-1] - track_half_width * nx
    right_y = disc["y"][:-1] - track_half_width * ny

    poly_x = np.concatenate([left_x, right_x[::-1]])
    poly_y = np.concatenate([left_y, right_y[::-1]])
    ax.fill(poly_x, poly_y, color="#4a4a4a", zorder=1, label="track surface")

    points = np.array([line_x, line_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
    lc = LineCollection(segments, cmap="viridis", linewidth=2, zorder=2)
    lc.set_array(vmax_seg)
    ax.add_collection(lc)

    ax.set_xlim(poly_x.min() - 20, poly_x.max() + 20)
    ax.set_ylim(poly_y.min() - 20, poly_y.max() + 20)
    ax.set_aspect("equal")
    ax.set_title("Random track + driving line")
    plt.colorbar(lc, ax=ax, label=r"theoretical $v_{max}$ (m/s)")

    # (b) velocity profile: theoretical ceiling vs DP-optimal trajectory
    ax = axes[0, 1]
    ax.plot(disc["s_bounds"], vmax_seg, "k--", label=r"$v_{max}(d)$ ceiling")
    ax.plot(disc["s_bounds"], sim["v"], "b-", label=r"optimal $v_t$ (DP)")
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