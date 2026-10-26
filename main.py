import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from utils import qss, gettrack as gt, dynamics, realtrack as rt

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
P_mguk_max  = 350e3       # W, MGU-K peak power under 2026 regs (shaft side)
F_trac_max  = 15_000.0    # N, ballpark max tyre-limited tractive force (low speed)

b_max      = 4.0e6        # J = 4 MJ (given constraint: 0 <= b_t <= b_max)
eta_regen  = 0.6          # shaft -> battery harvesting efficiency, applied ONCE to
                           # every harvested joule (braking and ICE-loaded).
                           # TODO: cite a source; results are sensitive to it.
SIGMA_REGEN = 0.05        # std-dev of the multiplicative factor on braking regen
                           # (regen actually harvested = nominal * max(0, 1 + sigma*z))
E_REC_CAP  = 8.5e6        # J per lap, approx. 2026 recovery limit (checked after the
                           # solve on the optimal trajectory, not enforced inside the DP)

v_top      = 100.0        # m/s (~360 km/h) absolute top-speed cap
v_min      = 10.0         # m/s floor, avoids division singularities (v_max is
                           # clamped to this so the grid always covers it)

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
def solve_bellman(disc, vmax, n_v=37, n_b=25, n_u=21, sigma_regen=SIGMA_REGEN, V_T=None):
    """One backward sweep around the lap. V_T is the terminal cost-to-go on the
    (v, b) grid (None -> zeros, i.e. the old single-lap formulation)."""
    n_seg = disc["n_seg"]
    ds = disc["ds"]
    kappa = disc["kappa"]
    drag_mult = dynamics.drag_multiplier(kappa)

    v_grid = np.linspace(v_min, v_top, n_v)
    b_grid = np.linspace(0.0, b_max, n_b)
    u_grid = np.linspace(-1.0, 1.0, n_u)          # n_u odd so that u = 0 is on the grid
    u_pos = u_grid > 0
    u_nz = u_grid != 0

    # 3-point Gauss-Hermite rule for a standard normal z (matches mean 0, var 1);
    # z scales the braking regen multiplicatively: regen * max(0, 1 + sigma*z)
    noise_vals = np.array([-np.sqrt(3), 0.0, np.sqrt(3)])
    noise_wts = np.array([1 / 6, 4 / 6, 1 / 6])

    V = np.zeros((n_seg + 1, n_v, n_b))
    if V_T is not None:
        V[n_seg] = V_T
    policy_u = np.zeros((n_seg, n_v, n_b))

    v_col = v_grid.reshape(n_v, 1, 1, 1)
    u_row = u_grid.reshape(1, 1, n_u, 1)
    b_pl = b_grid.reshape(1, n_b, 1, 1)
    regen_mult = np.maximum(0.0, 1.0 + sigma_regen * noise_vals).reshape(1, 1, 1, -1)
    # battery-side cost of u: deploy drains u*P*dt, harvest (u<0) stores eta*|u|*P*dt
    flow_scale = np.where(u_row >= 0, 1.0, eta_regen)            # (1,1,Nu,1)

    dt_arr = ds / v_col                                          # (Nv,1,1,1)
    Fice = dynamics.F_ice(F_trac_max, P_ice_max, v_col)          # (Nv,1,1,1)
    Froll = dynamics.F_rolling(Crr, m, g)

    for t in reversed(range(n_seg)):
        V_next = V[t + 1]                                        # (Nv, Nb)
        v_ceil = vmax[t + 1]

        # ---- FORCES / NEXT STEP VELOCITY ----
        F_ice_only = Fice - dynamics.F_drag(rho_air, CdA, v_col, drag_mult[t]) - Froll   # (Nv,1,1,1)
        Fmguk = dynamics.F_MGUK(P_mguk_max, u_row, v_col)                                # (Nv,1,Nu,1)

        v_next = np.minimum(v_col + (F_ice_only + Fmguk) / m * dt_arr, v_ceil)
        v_next = np.clip(v_next, v_min, v_top)                   # (Nv,1,Nu,1)

        # ---- BRAKING REGEN (state dependent) ----
        v_free = v_col + F_ice_only / m * dt_arr                 # ICE-only next speed
        regen_nom, must_brake = dynamics.braking_regen(m, P_mguk_max, v_free, v_ceil, dt_arr)
        braking = must_brake[:, 0, 0, 0]                         # (Nv,)

        # ---- NEXT STEP BATTERY ----
        flow = u_row * P_mguk_max * dt_arr * flow_scale          # (Nv,1,Nu,1), <0 = charging
        b_next = b_pl - flow + eta_regen * regen_nom * regen_mult    # (Nv,Nb,Nu,Nk)
        b_next = np.clip(b_next, 0.0, b_max)

        v_next_b = np.broadcast_to(v_next, b_next.shape)
        Vq = interp2d_vec(v_grid, b_grid, V_next, v_next_b, b_next)  # (Nv,Nb,Nu,Nk)

        EV = np.tensordot(Vq, noise_wts, axes=([3], [0]))        # (Nv,Nb,Nu)

        # ---- PER STEP COST ----
        ell = (ds / v_grid).reshape(n_v, 1, 1)                   # Eq. 5 stage cost
        cost = ell + EV                                          # (Nv,Nb,Nu)

        # Constraint: b_t = 0  =>  no deployment (harvesting stays allowed)
        cost[:, 0, u_pos] = np.inf
        # Constraint: while braking the MGU-K is in regen mode -> u = 0
        cost = np.where(braking[:, None, None] & u_nz[None, None, :], np.inf, cost)

        V[t] = np.min(cost, axis=2)
        policy_u[t] = u_grid[np.argmin(cost, axis=2)]

    return dict(V=V, policy_u=policy_u, v_grid=v_grid, b_grid=b_grid,
                u_grid=u_grid, drag_mult=drag_mult,
                noise_vals=noise_vals, noise_wts=noise_wts)


def solve_periodic(disc, vmax, n_iter=40, tol=2e-3, alpha=1.0, verbose=True, **kw):
    """Steady-state (periodic) lap by relative value iteration.

    The cost-to-go at the end of the lap is the cost-to-go at the start of
    the next one: V_T = h, with h = V_0 - V_0[ref]. Iterate until h stops
    changing. alpha < 1 damps the update (aperiodicity transform) if the
    iteration oscillates.
    """
    h = None
    sol = None
    converged = False 
    for it in range(n_iter):
        sol = solve_bellman(disc, vmax, V_T=h, **kw)
        V0 = sol["V"][0]
        iv, ib = len(sol["v_grid"]) // 2, len(sol["b_grid"]) // 2
        h_new = V0 - V0[iv, ib]
        if h is not None:
            delta = float(np.max(np.abs(h_new - h)))
            if verbose:
                print(f"  periodic iteration {it + 1:2d}: max|dh| = {delta:.4f} s")
            if delta < tol:
                h = h_new
                break
            h_new = (1 - alpha) * h + alpha * h_new
        h = h_new
    sol["periodic_iters"] = it + 1
    sol["periodic_converged"] = converged  # <-- add
    if not converged:                      # <-- add
        print(f"  WARNING: periodic iteration did not converge in {n_iter} iterations")
    return sol


# ---------------------------------------------------------------------------
#    Forward simulation using the optimal policy (deterministic or
#    Monte-Carlo with noisy braking regen, to show the "stochastic" behaviour)
# ---------------------------------------------------------------------------
def simulate(disc, vmax, sol, v0=None, b0=None, stochastic=False, rng=None,
             sigma_regen=SIGMA_REGEN):
    n_seg = disc["n_seg"]
    ds = disc["ds"]
    drag_mult = sol["drag_mult"]
    v_grid, b_grid = sol["v_grid"], sol["b_grid"]

    v = vmax[0] if v0 is None else v0
    b = 0.5 * b_max if b0 is None else b0

    v_hist = [v]
    b_hist = [b]
    u_hist = []
    brake_hist = []
    t_hist = [0.0]
    total_time = 0.0
    e_deploy = e_regen_brake = e_regen_ice = 0.0     # battery-side energies, J

    for t in range(n_seg):
        vs = max(v, 1.0)
        dt_seg = ds / vs
        F_ice_only = (dynamics.F_ice(F_trac_max, P_ice_max, v)
                      - dynamics.F_drag(rho_air, CdA, v, drag_mult[t])
                      - dynamics.F_rolling(Crr, m, g))
        v_free = v + F_ice_only / m * dt_seg
        regen_nom, must_brake = dynamics.braking_regen(m, P_mguk_max, v_free, vmax[t + 1], dt_seg)
        regen_nom, must_brake = float(regen_nom), bool(must_brake)

        if must_brake:
            u = 0.0                                   # MGU-K is in regen mode
        else:
            # bilinear-interpolate the optimal control off the DP policy grid
            u = float(interp2d_vec(v_grid, b_grid, sol["policy_u"][t],
                                    np.array(v), np.array(b)))
            if b <= 1e-6:
                u = min(u, 0.0)                       # empty battery: no deployment

        v_next = min(v + (F_ice_only + u * P_mguk_max / vs) / m * dt_seg, vmax[t + 1])
        v_next = float(np.clip(v_next, v_min, v_top))

        regen_mult = 1.0
        if stochastic:
            regen_mult = max(0.0, 1.0 + sigma_regen * rng.normal())
        d_dep = max(u, 0.0) * P_mguk_max * dt_seg
        d_ice = eta_regen * max(-u, 0.0) * P_mguk_max * dt_seg
        d_brk = eta_regen * regen_nom * regen_mult
        b_next = np.clip(b - d_dep + d_ice + d_brk, 0.0, b_max)

        e_deploy += d_dep
        e_regen_ice += d_ice
        e_regen_brake += d_brk

        total_time += dt_seg
        v, b = v_next, float(b_next)

        v_hist.append(v)
        b_hist.append(b)
        u_hist.append(u)
        brake_hist.append(must_brake)
        t_hist.append(total_time)

    return dict(v=np.array(v_hist), b=np.array(b_hist), u=np.array(u_hist),
                brake=np.array(brake_hist), t=np.array(t_hist), lap_time=total_time,
                e_deploy=e_deploy, e_regen_brake=e_regen_brake, e_regen_ice=e_regen_ice)


def simulate_steady(disc, vmax, sol, n_laps=6):
    """Drive several consecutive laps, feeding each lap's end state into the
    next, and return the last one (the periodic orbit if the policy is
    consistent)."""
    v, b = vmax[0], 0.5 * b_max
    for _ in range(n_laps):
        sim = simulate(disc, vmax, sol, v0=v, b0=b)
        v, b = sim["v"][-1], sim["b"][-1]
    return sim


# ---------------------------------------------------------------------------
#    Problem builders: a track -> discretised racing line + outline for plotting
# ---------------------------------------------------------------------------
def build_random_problem(seed=1, n_seg=1000):
    """The original procedural demo track (not a real circuit)."""
    track = gt.generate_track(seed=seed)
    disc = gt.discretize_track(track, n_seg=n_seg)

    line_x, line_y = gt.driving_line(disc)
    line_x, line_y, ds_line = gt.resample_equal_arclength(line_x, line_y, disc["n_seg"])
    disc["kappa"] = gt.line_curvature(line_x, line_y, ds_line)
    disc["ds"] = ds_line

    half_w = 15.0
    nx, ny = gt._unit_normals(disc["x"][:-1], disc["y"][:-1])
    outline = (np.concatenate([disc["x"][:-1] + half_w * nx, (disc["x"][:-1] - half_w * nx)[::-1]]),
               np.concatenate([disc["y"][:-1] + half_w * ny, (disc["y"][:-1] - half_w * ny)[::-1]]))
    return dict(name=f"Random track (seed {seed})", tag="random", disc=disc,
                line_x=line_x, line_y=line_y, outline=outline,
                start=(disc["x"][0], disc["y"][0]), line_label="driving line")


def build_real_problem(name, ds_target=5.0, smooth=0.0):
    """A real circuit and its (minimum-curvature) racing line from the
    TUMFTM racetrack-database; downloaded on first use."""
    disc = rt.load_racing_line(name, ds_target=ds_target, smooth=smooth)
    return dict(name=name, tag=name.lower(), disc=disc,
                line_x=disc["x"], line_y=disc["y"], outline=rt.track_polygon(name),
                start=(disc["x"][0], disc["y"][0]), line_label="racing line")

def _panel_track(ax, problem, vmax_seg):
    line_x, line_y = problem["line_x"], problem["line_y"]
    # ax.set_facecolor("#4a4a4a")   # dark backdrop so the yellow (fast) sections stay visible
    points = np.array([line_x, line_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points, np.roll(points, -1, axis=0)], axis=1)
    lc = LineCollection(segments, cmap="viridis", linewidth=2.5, zorder=2)
    lc.set_array(vmax_seg)
    ax.add_collection(lc)
    ax.plot(*problem["start"], marker="s", color="white", markeredgecolor="black",
            markersize=10, zorder=3)
    ax.set_xlim(line_x.min() - 20, line_x.max() + 20)
    ax.set_ylim(line_y.min() - 20, line_y.max() + 20)
    ax.set_aspect("equal")
    ax.set_title(f"{problem['name']} + {problem['line_label']}")
    ax.figure.colorbar(lc, ax=ax, label=r"theoretical $v_{max}$ (m/s)")


def _panel_velocity(ax, disc, vmax_seg, sim):
    ax.plot(disc["s_bounds"], vmax_seg, "k--", label=r"$v_{max}(d)$ ceiling")
    ax.plot(disc["s_bounds"], sim["v"], "b-", label=r"optimal $v_t$ (DP)")
    ax.set_xlabel("distance s (m)")
    ax.set_ylabel("velocity (m/s)")
    ax.set_title("Velocity profile")
    ax.legend(fontsize=9)


def _panel_control(ax, disc, sim):
    s_mid = 0.5 * (disc["s_bounds"][:-1] + disc["s_bounds"][1:])
    ax.step(s_mid, sim["u"], where="mid", color="tab:orange", label=r"$u_t$ (+deploy / -harvest)")
    ax.fill_between(disc["s_bounds"][:-1], -1, 1, where=sim["brake"], step="post",
                    color="grey", alpha=0.3, label="braking (regen)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("distance s (m)")
    ax.set_ylabel(r"$u_t$ (deploy / harvest fraction)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Optimal MGU-K power-split policy")
    ax.legend(fontsize=9)


def _panel_battery(ax, disc, sim, e_rec):
    ax.plot(disc["s_bounds"], sim["b"] / 1e6, "g-")
    ax.axhline(b_max / 1e6, color="k", ls=":", lw=1)
    ax.axhline(0, color="k", ls=":", lw=1)
    ax.set_xlabel("distance s (m)")
    ax.set_ylabel("battery energy (MJ)")
    ax.set_title(f"Battery SoC  (deploy {sim['e_deploy'] / 1e6:.1f} MJ, "
                 f"harvest {e_rec / 1e6:.1f} MJ per lap)")

# ---------------------------------------------------------------------------
#    Run the full demo
# ---------------------------------------------------------------------------
def run(problem, grid, n_mc=200, sigma_regen=SIGMA_REGEN, out_dir=".", split=False):
    disc = problem["disc"]
    line_x, line_y = problem["line_x"], problem["line_y"]

    # ----- twopass QSS v_max computation on the racing line -----
    vehicle = qss.VehicleParams(mass=m, mu=mu_tire, rho=rho_air, cl_a=ClA, v_top=v_top)
    vmax_seg = qss.compute_qss_vmax(disc["kappa"], disc["ds"], vehicle, a_accel_max=6.0, a_brake_max=45.0)
    vmax_seg = np.maximum(vmax_seg, v_min)
    vmax_seg[0] = vmax_seg[-1] = min(vmax_seg[0], vmax_seg[-1])      # closed loop
    # ---------------------------------

    print(f"Track                           : {problem['name']}")
    print(f"Track length                    : {disc['DL']:8.1f} m")
    print(f"Number of segments              : {disc['n_seg']:8d}")
    print(f"Segment length ({problem['line_label']})".ljust(32) + f": {disc['ds']:8.2f} m")
    print(f"Lowest v_max on the line        : {vmax_seg.min():8.1f} m/s")
    print("Solving periodic stochastic Bellman recursion...")

    sol = solve_periodic(disc, vmax_seg, sigma_regen=sigma_regen, **grid)

    sim = simulate_steady(disc, vmax_seg, sol)
    e_rec = sim["e_regen_brake"] + sim["e_regen_ice"]
    print(f"\nSteady-state lap time           : {sim['lap_time']:.2f} s")
    print(f"Battery at lap start / end      : {sim['b'][0] / 1e6:.2f} / {sim['b'][-1] / 1e6:.2f} MJ")
    print(f"Energy deployed                 : {sim['e_deploy'] / 1e6:.2f} MJ")
    print(f"Energy harvested (battery side) : {e_rec / 1e6:.2f} MJ "
          f"(braking {sim['e_regen_brake'] / 1e6:.2f}, ICE-loaded {sim['e_regen_ice'] / 1e6:.2f}; "
          f"cap {E_REC_CAP / 1e6:.1f})")
    if e_rec > E_REC_CAP:
        print("  WARNING: trajectory exceeds the per-lap recovery cap")

    # Monte-Carlo forward rollouts under noisy braking regen, one lap each,
    # started from the deterministic steady-state (v, b)
    mc_rng = np.random.default_rng(1)
    mc = [simulate(disc, vmax_seg, sol, v0=sim["v"][0], b0=sim["b"][0],
                   stochastic=True, rng=mc_rng, sigma_regen=sigma_regen) for _ in range(n_mc)]
    mc_times = np.array([r["lap_time"] for r in mc])
    mc_drift = np.array([r["b"][-1] - r["b"][0] for r in mc])
    print(f"Monte-Carlo ({n_mc} runs) lap time : "
          f"mean {mc_times.mean():.2f} s, std {mc_times.std():.3f} s, "
          f"95% CI +/- {1.96 * mc_times.std() / np.sqrt(n_mc):.3f} s")
    print(f"Monte-Carlo battery drift/lap   : mean {mc_drift.mean() / 1e6:+.3f} MJ, "
          f"std {mc_drift.std() / 1e6:.3f} MJ")

    # ---------------- plots ----------------
    panels = [   # (file suffix, figsize, draw function)
        ("track",    (7, 8),  lambda ax: _panel_track(ax, problem, vmax_seg)),
        ("velocity", (10, 5), lambda ax: _panel_velocity(ax, disc, vmax_seg, sim)),
        ("control",  (10, 5), lambda ax: _panel_control(ax, disc, sim)),
        ("battery",  (10, 5), lambda ax: _panel_battery(ax, disc, sim, e_rec)),
    ]
    saved = []

    if split:
        for name, size, draw in panels:
            f, ax = plt.subplots(figsize=size)
            draw(ax)
            f.tight_layout()
            path = os.path.join(out_dir, f"f1_dp_{problem['tag']}_{name}.png")
            f.savefig(path, dpi=200)
            plt.close(f)
            saved.append(path)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(13, 10))
        for ax, (_, _, draw) in zip(axes.flat, panels):
            draw(ax)
        fig.suptitle(f"F1 power-split DP  |  {problem['name']}  |  steady-state lap = {sim['lap_time']:.2f} s "
                     f"(MC mean {mc_times.mean():.2f} +/- {mc_times.std():.2f} s)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path = os.path.join(out_dir, f"f1_dp_{problem['tag']}.png")
        fig.savefig(path, dpi=150)
        saved.append(path)

    # separate histogram figure for the Monte-Carlo lap-time distribution
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(mc_times, bins=25, color="tab:blue", alpha=0.8)
    ax2.axvline(sim["lap_time"], color="red", ls="--", label="deterministic steady-state lap")
    ax2.set_xlabel("lap time (s)")
    ax2.set_ylabel("count")
    ax2.set_title(f"Monte-Carlo lap-time distribution ({n_mc} stochastic rollouts)")
    ax2.legend()
    fig2.tight_layout()
    mc_png = os.path.join(out_dir, f"f1_dp_{problem['tag']}_montecarlo.png")
    fig2.savefig(mc_png, dpi=150)

    print("\nSaved plots:")
    mc_png = os.path.join(out_dir, f"f1_dp_{problem['tag']}_montecarlo.png")
    fig2.savefig(mc_png, dpi=150)
    saved.append(mc_png)
    print("\n".join(saved))


def main(argv=None):
    p = argparse.ArgumentParser(description="Stochastic DP power-split control for an F1 hybrid.")
    p.add_argument("--track", default="Spa",
                   help="circuit name from the TUMFTM racetrack-database (see --list), "
                        "or 'random' for the procedural demo track (default: Spa)")
    p.add_argument("--list", action="store_true", help="list the available circuits and exit")
    p.add_argument("--ds", type=float, default=5.0, help="target segment length in m for real tracks (default 5)")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="RMS metres the racing-line spline may move the points (default 0)")
    p.add_argument("--n-seg", type=int, default=1000, help="segments for the random track (default 1000)")
    p.add_argument("--seed", type=int, default=1, help="random-track seed (default 1)")
    p.add_argument("--mc-runs", type=int, default=200, help="Monte-Carlo rollouts (default 200)")
    p.add_argument("--sigma-regen", type=float, default=SIGMA_REGEN,
                   help=f"std-dev of the multiplicative regen noise (default {SIGMA_REGEN})")
    p.add_argument("--fast", action="store_true", help="coarse DP grid for quick iteration")
    p.add_argument("--out", default=".", help="directory for the plots (default: current directory)")
    p.add_argument("--split", action="store_true",
                   help="save each of the four panels as its own figure instead of one 2x2 figure")
    args = p.parse_args(argv)

    if args.list:
        print("\n".join(rt.available_tracks()))
        return

    if args.track.lower() == "random":
        problem = build_random_problem(seed=args.seed, n_seg=args.n_seg)
    else:
        names = {n.lower(): n for n in rt.available_tracks()}
        if args.track.lower() not in names:
            p.error(f"unknown track '{args.track}'. Use --list to see the options.")
        problem = build_real_problem(names[args.track.lower()], ds_target=args.ds, smooth=args.smooth)

    grid = dict(n_v=25, n_b=17, n_u=11) if args.fast else dict(n_v=37, n_b=25, n_u=21)
    os.makedirs(args.out, exist_ok=True)
    run(problem, grid, n_mc=args.mc_runs, sigma_regen=args.sigma_regen, out_dir=args.out, split=args.split)


if __name__ == "__main__":
    main()
