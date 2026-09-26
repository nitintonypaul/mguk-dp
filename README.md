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
