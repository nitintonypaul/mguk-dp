"""
Real circuits and their racing lines
====================================
Loader for the TUMFTM/racetrack-database (https://github.com/TUMFTM/racetrack-database,
LGPL-3.0). For each circuit it provides

  tracks/<name>.csv      centerline  x_m, y_m, w_tr_right_m, w_tr_left_m   (5 m spacing)
  racelines/<name>.csv   racing line x_m, y_m  (minimum-curvature line inside the
                         track boundaries, 5 m spacing)

The CSVs are downloaded on first use into  <repo>/data/tumftm/  (git-ignored;
they are LGPL-3.0 so they are not vendored into this MIT repo).

IMPORTANT: the raceline is a minimum-CURVATURE line, not a minimum-time line,
and it is a computed line, not a driven one. For a driven line use FastF1
telemetry (README: validation step).

Everything the DP needs comes out of load_racing_line(): equal-arclength
points on the racing line and an analytic curvature from a periodic spline
fitted through the points. The `smooth` argument (RMS metres the spline may
move the points) defaults to 0 because the database lines are already smooth:
on Spa and Monza, smoothing of 0.5 m moved the line by up to ~1.7 m and
introduced extra spurious braking events. Use smooth > 0 for noisy inputs
such as lines derived from GPS / telemetry.
"""

import os
import urllib.request

import numpy as np
from scipy.interpolate import splev, splprep

BASE_URL = "https://raw.githubusercontent.com/TUMFTM/racetrack-database/master"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "tumftm")


def available_tracks():
    """Circuits that make sense for this project (F1 circuits in the database)."""
    return ["Austin", "Budapest", "Catalunya", "Hockenheim", "Melbourne", "MexicoCity",
            "Montreal", "Monza", "Sakhir", "SaoPaulo", "Sepang", "Shanghai",
            "Silverstone", "Sochi", "Spa", "Spielberg", "Suzuka", "YasMarina"]


def _csv(kind, name):
    """Return the parsed CSV (downloading it once if it is not cached)."""
    path = os.path.join(DATA_DIR, kind, f"{name}.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"{BASE_URL}/{kind}/{name}.csv"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:                       # pragma: no cover
            if os.path.exists(path):
                os.remove(path)
            raise RuntimeError(
                f"Could not download {url} ({exc}). Clone "
                f"https://github.com/TUMFTM/racetrack-database and copy its "
                f"'tracks' and 'racelines' folders to {DATA_DIR}") from exc
    return np.loadtxt(path, delimiter=",", comments="#")


def load_racing_line(name, ds_target=5.0, smooth=0.0):
    """Racing line of circuit `name`, resampled to equal arc length.

    ds_target : desired segment length in metres (n_seg = round(DL / ds_target)).
    smooth    : RMS distance (m) the smoothing spline may move the raceline
                points. 0 interpolates exactly (noisy curvature).

    Returns a dict with the same keys the rest of the code uses for `disc`:
      n_seg, ds, DL, s_bounds, x, y (n_seg+1, closed), kappa (n_seg+1, closed,
      signed, + = left turn).
    """
    pts = _csv("racelines", name)[:, :2]
    closed = np.vstack([pts, pts[:1]])                  # splprep(per=1) wants first == last

    # smoothing spline s = sum of squared residuals = N * rms^2
    tck, _ = splprep([closed[:, 0], closed[:, 1]], s=len(pts) * smooth**2, per=True)

    # equal-arclength reparametrisation via a dense evaluation of the spline
    u_dense = np.linspace(0.0, 1.0, 40 * len(pts) + 1)
    xd, yd = splev(u_dense, tck)
    s_dense = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))])
    DL = float(s_dense[-1])

    n_seg = int(round(DL / ds_target))
    s_bounds = np.linspace(0.0, DL, n_seg + 1)
    u = np.interp(s_bounds, s_dense, u_dense)

    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    d2x, d2y = splev(u, tck, der=2)
    kappa = (dx * d2y - dy * d2x) / (dx**2 + dy**2) ** 1.5      # invariant to the parametrisation
    kappa[-1] = kappa[0]                                         # exact closure

    x, y = np.asarray(x), np.asarray(y)
    x[-1], y[-1] = x[0], y[0]
    return dict(n_seg=n_seg, ds=DL / n_seg, DL=DL, s_bounds=s_bounds,
                x=x, y=y, kappa=np.asarray(kappa))


def track_polygon(name):
    """Outline of the track surface for plotting: (x, y) arrays that trace the
    left boundary and then the right boundary backwards (use with ax.fill)."""
    t = _csv("tracks", name)
    cx, cy, w_right, w_left = t[:, 0], t[:, 1], t[:, 2], t[:, 3]
    tx = np.roll(cx, -1) - np.roll(cx, 1)                # periodic central difference
    ty = np.roll(cy, -1) - np.roll(cy, 1)
    norm = np.hypot(tx, ty) + 1e-12
    nx, ny = -ty / norm, tx / norm                       # left-hand normal
    left_x, left_y = cx + w_left * nx, cy + w_left * ny
    right_x, right_y = cx - w_right * nx, cy - w_right * ny
    return (np.concatenate([left_x, right_x[::-1]]),
            np.concatenate([left_y, right_y[::-1]]))
