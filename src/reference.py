"""Analytic references for Milestone 1 tracking tests."""

from __future__ import annotations

import numpy as np


def figure_eight(
    t: np.ndarray, scale: float = 2.0, omega: float = 0.35
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lemniscate of Gerono + analytic velocity.
    Returns path_xy (T,2), path_v (T,2).
    """
    t = np.asarray(t, dtype=float)
    wt = omega * t
    x = scale * np.sin(wt)
    y = scale * np.sin(wt) * np.cos(wt)
    vx = scale * omega * np.cos(wt)
    vy = scale * omega * np.cos(2.0 * wt)
    return np.column_stack([x, y]), np.column_stack([vx, vy])


def line_segment(
    t: np.ndarray,
    p0: tuple[float, float] = (0.0, 0.0),
    p1: tuple[float, float] = (4.0, 0.0),
    duration: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed line from p0 to p1; velocity zero after arrival."""
    t = np.asarray(t, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    delta = p1 - p0
    dist = float(np.linalg.norm(delta))
    speed = dist / duration if duration > 0 else 0.0
    direction = delta / dist if dist > 1e-9 else np.zeros(2)

    alpha = np.clip(t / duration, 0.0, 1.0)
    path_xy = (1.0 - alpha)[:, None] * p0 + alpha[:, None] * p1
    moving = (t < duration).astype(float)[:, None]
    path_v = moving * (speed * direction)
    return path_xy, path_v


def stack_state_ref(path_xy: np.ndarray, path_v: np.ndarray) -> np.ndarray:
    """(T,2)+(T,2) -> (T,4) [x,y,vx,vy]."""
    return np.column_stack([path_xy, path_v])


def stack_unicycle_ref(path_xy: np.ndarray, path_v: np.ndarray) -> np.ndarray:
    """(T,2)+(T,2) -> (T,4) [x,y,theta,v] with unwrapped heading."""
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    path_v = np.asarray(path_v, dtype=float).reshape(-1, 2)
    speed = np.linalg.norm(path_v, axis=1)
    theta = np.arctan2(path_v[:, 1], path_v[:, 0])
    for i in range(len(theta)):
        if speed[i] < 1e-6:
            theta[i] = theta[i - 1] if i > 0 else 0.0
    theta = np.unwrap(theta)
    return np.column_stack([path_xy, theta, speed])


def horizon_from_traj(traj: np.ndarray, t0_index: int, N: int) -> np.ndarray:
    """N state samples starting at t0_index; pad with last row if needed."""
    T = traj.shape[0]
    out = np.empty((N, traj.shape[1]), dtype=float)
    for k in range(N):
        out[k] = traj[min(t0_index + k, T - 1)]
    return out
