"""Closed-loop simulation helpers."""

from __future__ import annotations

import time

import numpy as np

from .config import A_MAX, DT, N, Q_POS, R_U, V_MAX
from .mpc import TrackingMPC
from .plant import step
from .reference import figure_eight, horizon_from_path


def run_tracking(
    duration_s: float = 20.0,
    path: str = "eight",
    lag_alpha: float = 1.0,
) -> dict:
    dt = DT
    steps = int(round(duration_s / dt))
    t = np.arange(steps) * dt

    if path == "eight":
        path_xy = figure_eight(t)
    else:
        from .reference import line_segment

        path_xy = line_segment(t)

    # Extend path so horizon can look past the end
    pad = np.repeat(path_xy[-1][None, :], N, axis=0)
    path_ext = np.vstack([path_xy, pad])

    z = np.array([path_xy[0, 0], path_xy[0, 1], 0.0, 0.0])
    mpc = TrackingMPC(dt, N, Q_POS, R_U, A_MAX)

    traj = np.zeros((steps, 4))
    err = np.zeros(steps)
    solve_ms = np.zeros(steps)
    u_prev = np.zeros(2)
    ok_count = 0

    for i in range(steps):
        ref = horizon_from_path(path_ext, i + 1, N)
        t0 = time.perf_counter()
        u0, info = mpc.solve(z, ref)
        solve_ms[i] = (time.perf_counter() - t0) * 1e3
        if info["ok"]:
            ok_count += 1
        z, u_prev = step(z, u0, dt, V_MAX, A_MAX, u_prev=u_prev, lag_alpha=lag_alpha)
        traj[i] = z
        err[i] = np.linalg.norm(z[:2] - path_xy[i])

    return {
        "t": t,
        "path_xy": path_xy,
        "traj_xy": traj[:, :2],
        "traj": traj,
        "err": err,
        "solve_ms": solve_ms,
        "ok_frac": ok_count / steps,
        "err_rms": float(np.sqrt(np.mean(err**2))),
        "err_max": float(np.max(err)),
        "solve_ms_med": float(np.median(solve_ms)),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
    }
