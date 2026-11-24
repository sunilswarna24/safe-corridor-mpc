"""2D double-integrator plant (disk center)."""

from __future__ import annotations

import numpy as np


def discrete_matrices(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """ZOH double integrator: z=[x,y,vx,vy], u=[ax,ay]."""
    Ad = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    Bd = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=float,
    )
    return Ad, Bd


def step(
    z: np.ndarray,
    u: np.ndarray,
    dt: float,
    v_max: float,
    a_max: float,
    u_prev: np.ndarray | None = None,
    slew_max: float | None = None,
    lag_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One plant step. Returns (z_next, u_applied).
    lag_alpha in (0,1] blends toward commanded u (1 = no lag).
    """
    u = np.asarray(u, dtype=float).reshape(2)
    u = np.clip(u, -a_max, a_max)
    if u_prev is not None and slew_max is not None:
        du = np.clip(u - u_prev, -slew_max, slew_max)
        u = u_prev + du
    if lag_alpha < 1.0 and u_prev is not None:
        u = lag_alpha * u + (1.0 - lag_alpha) * u_prev

    Ad, Bd = discrete_matrices(dt)
    z_next = Ad @ np.asarray(z, dtype=float).reshape(4) + Bd @ u
    z_next[2:4] = np.clip(z_next[2:4], -v_max, v_max)
    return z_next, u
