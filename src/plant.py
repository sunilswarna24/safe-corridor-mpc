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
    slip_std: float = 0.0,
    accel_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One plant step. Returns (z_next, u_applied).

    slip_std: Gaussian noise on velocity after the dynamics [m/s] (wheel slip).
    accel_noise_std: Gaussian noise on applied accel [m/s^2].
    """
    u = np.asarray(u, dtype=float).reshape(2)
    u = np.clip(u, -a_max, a_max)
    if u_prev is not None and slew_max is not None:
        du = np.clip(u - u_prev, -slew_max, slew_max)
        u = u_prev + du
    if lag_alpha < 1.0 and u_prev is not None:
        u = lag_alpha * u + (1.0 - lag_alpha) * u_prev

    if accel_noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        u = u + rng.normal(0.0, accel_noise_std, size=2)
        u = np.clip(u, -a_max, a_max)

    Ad, Bd = discrete_matrices(dt)
    z_next = Ad @ np.asarray(z, dtype=float).reshape(4) + Bd @ u

    if slip_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        z_next[2:4] = z_next[2:4] + rng.normal(0.0, slip_std, size=2)

    z_next[2:4] = np.clip(z_next[2:4], -v_max, v_max)
    return z_next, u
