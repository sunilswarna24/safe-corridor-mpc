"""Unicycle AMR plant: z=[x,y,theta,v], u=[a, omega]."""

from __future__ import annotations

import numpy as np


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def f_discrete(z: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """One Euler step (no noise / limits)."""
    x, y, th, v = np.asarray(z, dtype=float).reshape(4)
    a, w = np.asarray(u, dtype=float).reshape(2)
    return np.array(
        [
            x + dt * v * np.cos(th),
            y + dt * v * np.sin(th),
            th + dt * w,
            v + dt * a,
        ],
        dtype=float,
    )


def linearize(z: np.ndarray, u: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    z+ ≈ A z + B u + c, with c = f(zr,ur) - A zr - B ur
    evaluated at the provided (z,u) operating point.
    """
    x, y, th, v = np.asarray(z, dtype=float).reshape(4)
    a, w = np.asarray(u, dtype=float).reshape(2)
    ct, st = np.cos(th), np.sin(th)
    A = np.array(
        [
            [1.0, 0.0, -dt * v * st, dt * ct],
            [0.0, 1.0, dt * v * ct, dt * st],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    B = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, dt],
            [dt, 0.0],
        ],
        dtype=float,
    )
    zr = np.array([x, y, th, v], dtype=float)
    ur = np.array([a, w], dtype=float)
    c = f_discrete(zr, ur, dt) - A @ zr - B @ ur
    return A, B, c


def step(
    z: np.ndarray,
    u: np.ndarray,
    dt: float,
    v_max: float,
    a_max: float,
    w_max: float,
    slip_std: float = 0.0,
    accel_noise_std: float = 0.0,
    yaw_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One plant step. Returns (z_next, u_applied). v clamped to [0, v_max]."""
    u = np.asarray(u, dtype=float).reshape(2).copy()
    u[0] = float(np.clip(u[0], -a_max, a_max))
    u[1] = float(np.clip(u[1], -w_max, w_max))

    if accel_noise_std > 0.0 or yaw_noise_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        if accel_noise_std > 0.0:
            u[0] += float(rng.normal(0.0, accel_noise_std))
            u[0] = float(np.clip(u[0], -a_max, a_max))
        if yaw_noise_std > 0.0:
            u[1] += float(rng.normal(0.0, yaw_noise_std))
            u[1] = float(np.clip(u[1], -w_max, w_max))

    z_next = f_discrete(z, u, dt)
    z_next[2] = wrap_angle(z_next[2])

    if slip_std > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        # Longitudinal slip + small heading jitter
        z_next[3] += float(rng.normal(0.0, slip_std))
        z_next[2] = wrap_angle(z_next[2] + float(rng.normal(0.0, 0.25 * slip_std)))

    z_next[3] = float(np.clip(z_next[3], 0.0, v_max))
    return z_next, u
