"""
qss_vmax.py
===========
Standalone quasi-steady-state (QSS) v_max(d) profile generator.

Three passes, chained (this is the standard structure used in motorsport
lap-time simulators, e.g. Heilmeier et al. 2019):

  1. cornering_vmax   - pure grip-limited ceiling from curvature alone
                         (a static equilibrium snapshot at every point;
                         no memory of neighbouring segments)
  2. forward_pass      - "how fast could I realistically be going here,
                         given the best-case acceleration available since
                         the last speed limit" -> kills unrealistic jumps
                         out of slow corners
  3. backward_pass     - "how fast can I be going here and still brake
                         down to the next corner's limit in time" -> this
                         is the lookahead-braking behaviour (a straight
                         that suddenly bends gets its ceiling pulled down
                         well before the bend, not just at the bend).

Track is treated as a closed loop: the three passes are iterated a few
times around the loop so information propagates all the way around
(e.g. a slow hairpin just after the start/finish line should also pull
down the ceiling on the last straight, which needs wraparound to see).

No dependency on the rest of the project -> safe to import anywhere.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleParams:
    mass: float          # kg
    mu: float            # tyre-road friction coefficient
    rho: float            # air density, kg/m^3
    cl_a: float            # downforce coefficient * area, m^2
    v_top: float          # absolute top-speed cap, m/s
    g: float = 9.81        # immutable by the user, matches the C++ struct


# ---------------------------------------------------------------------
# Pass 1: cornering (grip-limited) ceiling - closed-form QSS solve
# ---------------------------------------------------------------------
def cornering_vmax(curvatures, vehicle: VehicleParams):
    """v such that centripetal force == friction from (weight + downforce).
    v^2 = mu*m*g / (m*|kappa| - mu*0.5*rho*cl_a)
    Vectorized; curvature sign does NOT matter (this was the bug in the
    C++ draft - left-hand corners must be treated the same as right-hand)."""
    kappa = np.abs(np.asarray(curvatures, dtype=float))
    aero_factor = 0.5 * vehicle.mu * vehicle.rho * vehicle.cl_a
    grip_numerator = vehicle.mu * vehicle.mass * vehicle.g

    denom = vehicle.mass * kappa - aero_factor
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(np.where(denom > 0.0, grip_numerator / np.maximum(denom, 1e-12), np.inf))
    v = np.where(denom > 0.0, v, vehicle.v_top)
    return np.minimum(v, vehicle.v_top)


# ---------------------------------------------------------------------
# Pass 2: forward, acceleration-limited pass
#   v[i] = min( v_ceiling[i],  sqrt(v[i-1]^2 + 2*a_accel_max*ds[i-1]) )
# ---------------------------------------------------------------------
def forward_pass(v_ceiling, ds, a_accel_max, n_loops=3):
    n = len(v_ceiling)
    v = np.array(v_ceiling, dtype=float)
    ds = np.broadcast_to(np.asarray(ds, dtype=float), (n - 1,))

    for _ in range(n_loops):           # iterate around the closed loop
        for i in range(n):
            prev = v[i - 1] if i > 0 else v[-1]          # wraps to finish
            step_ds = ds[i - 1] if i > 0 else ds[-1]
            reachable = np.sqrt(prev**2 + 2 * a_accel_max * step_ds)
            v[i] = min(v_ceiling[i], reachable)
    return v


# ---------------------------------------------------------------------
# Pass 3: backward, braking-limited pass
#   v[i] = min( v_ceiling[i],  sqrt(v[i+1]^2 + 2*a_brake_max*ds[i]) )
# ---------------------------------------------------------------------
def backward_pass(v_ceiling, ds, a_brake_max, n_loops=3):
    n = len(v_ceiling)
    v = np.array(v_ceiling, dtype=float)
    ds = np.broadcast_to(np.asarray(ds, dtype=float), (n - 1,))

    for _ in range(n_loops):           # iterate around the closed loop
        for i in range(n - 1, -1, -1):
            nxt = v[i + 1] if i < n - 1 else v[0]         # wraps to start
            step_ds = ds[i] if i < n - 1 else ds[-1]
            reachable = np.sqrt(nxt**2 + 2 * a_brake_max * step_ds)
            v[i] = min(v_ceiling[i], reachable)
    return v


# ---------------------------------------------------------------------
# Top-level: full 3-pass QSS profile
# ---------------------------------------------------------------------
def compute_qss_vmax(curvatures, ds, vehicle: VehicleParams,
                      a_accel_max=6.0, a_brake_max=45.0, n_loops=3):
    """
    curvatures : array of length N+1, signed or unsigned curvature at each
                 segment BOUNDARY (matches discretize_track()'s kappa_b).
    ds         : scalar, or array of length N (per-segment length).
    vehicle    : VehicleParams.
    a_accel_max: ballpark best-case forward acceleration capability, m/s^2
                 (traction/power limited; ~6 m/s^2 is a reasonable
                 low-to-mid-speed F1 ballpark - lower than full launch accel
                 since this profile also has to hold near top speed).
    a_brake_max: ballpark best-case braking deceleration, m/s^2
                 (F1 cars can pull ~5-6g under heavy braking with
                 downforce -> ~45-55 m/s^2 ballpark).

    Returns v_max_eff, the same shape/semantics as the old segment_vmax().
    """
    v_corner = cornering_vmax(curvatures, vehicle)
    v_fwd = forward_pass(v_corner, ds, a_accel_max, n_loops=n_loops)
    v_eff = backward_pass(v_fwd, ds, a_brake_max, n_loops=n_loops)
    return v_eff