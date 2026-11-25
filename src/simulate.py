"""Closed-loop simulation helpers."""

from __future__ import annotations

import time

import numpy as np

from .config import A_MAX, DT, N, Q_POS, Q_VEL, R_U, V_MAX
from .mpc import TrackingMPC
from .plant import step
from .reference import (
    figure_eight,
    horizon_from_traj,
    line_segment,
    stack_state_ref,
)


def run_tracking(
    duration_s: float = 20.0,
    path: str = "eight",
    lag_alpha: float = 1.0,
) -> dict:
    dt = DT
    steps = int(round(duration_s / dt))
    t = np.arange(steps) * dt

    if path == "eight":
        path_xy, path_v = figure_eight(t)
    else:
        path_xy, path_v = line_segment(t)

    traj_ref = stack_state_ref(path_xy, path_v)
    pad = np.repeat(traj_ref[-1][None, :], N, axis=0)
    traj_ext = np.vstack([traj_ref, pad])

    z = np.array([path_xy[0, 0], path_xy[0, 1], 0.0, 0.0])
    mpc = TrackingMPC(dt, N, Q_POS, Q_VEL, R_U, A_MAX)

    traj = np.zeros((steps, 4))
    err_pos = np.zeros(steps)
    err_vel = np.zeros(steps)
    solve_ms = np.zeros(steps)
    u_prev = np.zeros(2)
    ok_count = 0

    for i in range(steps):
        ref = horizon_from_traj(traj_ext, i + 1, N)
        t0 = time.perf_counter()
        u0, info = mpc.solve(z, ref)
        solve_ms[i] = (time.perf_counter() - t0) * 1e3
        if info["ok"]:
            ok_count += 1
        z, u_prev = step(z, u0, dt, V_MAX, A_MAX, u_prev=u_prev, lag_alpha=lag_alpha)
        traj[i] = z
        err_pos[i] = np.linalg.norm(z[:2] - path_xy[i])
        err_vel[i] = np.linalg.norm(z[2:] - path_v[i])

    return {
        "t": t,
        "path_xy": path_xy,
        "path_v": path_v,
        "traj_xy": traj[:, :2],
        "traj": traj,
        "err": err_pos,
        "err_vel": err_vel,
        "solve_ms": solve_ms,
        "ok_frac": ok_count / steps,
        "err_rms": float(np.sqrt(np.mean(err_pos**2))),
        "err_max": float(np.max(err_pos)),
        "err_vel_rms": float(np.sqrt(np.mean(err_vel**2))),
        "solve_ms_med": float(np.median(solve_ms)),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
    }
