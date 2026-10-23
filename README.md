## Stochastic Optimal Power-Split Control in Formula 1

Modern Formula 1 hybrid power units require teams to continuously decide how much electrical
power to deploy from the MGU-K (Motor Generator Unit - Kinetic) unit alongside internal combustion
engine (ICE) output, subject to a strictly limited and only partially rechargeable battery budget. This
deployment decision is complicated by track geometry, since power injected in traction-limited corners
yields little or no speed benefit, while the same energy deployed on corner exit compounds into meaningful
gains down the following straight. We formulate this energy management problem as a sequential decision
process and solve it using dynamic programming. In the baseline formulation, the racing circuit is
discretized by distance, and a Bellman recursion determines, at each track position, s and battery
state, b, whether to deploy stored energy, subject to a traction-limited velocity ceiling derived from
track curvature and vehicle dynamics, vmax (drag, rolling resistance, power-to-force conversion). This
is extended to a continuous throttle-fraction control, $u_t \in [0, 1]$, and a stochastic battery model that
accounts for variability in braking-zone energy recovery, yielding a bi-state stochastic optimal control
problem. Physical constraints such as battery capacity, MGU-K power ceiling, and per-lap energy limits
are taken from the 2026 FIA Technical Regulations, and the resulting optimal policy projected to be validated against
real race telemetry obtained via the FastF1 API. This project demonstrates how energy-constrained sequential decision-making under uncertainty, a canonical operations research structure, governs realtime performance optimization in modern motorsport.

## Running

```bash
pip install -r requirements.txt
python main.py --track Spa          # real circuit + racing line (downloads the CSVs on first use)
python main.py --list               # available circuits
python main.py --track Monza --fast # coarse DP grid for quick iteration
python main.py --track random       # the original procedural demo track
```

Plots are written to `--out` (default: current directory) as `f1_dp_<track>.png` and
`f1_dp_<track>_montecarlo.png`.

### Model summary

- State `(v, b)`: speed and battery energy; control `u in [-1, 1]` (`+` deploy, `-` harvest by
  loading the ICE).
- Braking regen is computed from the actual state: when the ICE-only next speed would exceed the
  next node's `v_max` the car brakes, the MGU-K is in regen mode (`u = 0`) and recovers
  `min(P_mguk * dt, kinetic energy shed)`. `eta_regen` is applied once per harvested joule.
- The only stochastic element is a multiplicative factor `max(0, 1 + sigma * z)` on braking regen.
- The DP is solved for a steady-state lap (relative value iteration, `solve_periodic`), so the
  battery cannot be drained for free at the end of the lap.
- The per-lap recovery limit (~8.5 MJ) is checked on the optimal trajectory but **not** enforced
  inside the DP.

### Data and attribution

Real circuits and racing lines come from the
[TUMFTM racetrack-database](https://github.com/TUMFTM/racetrack-database) (LGPL-3.0). The files
are downloaded into `data/tumftm/` (git-ignored) and are not part of this repository. The racing
lines there are minimum-curvature lines, not minimum-time or driven lines.
