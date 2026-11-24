"""Analytic references for Milestone 1 tracking tests."""

from __future__ import annotations

import numpy as np


def figure_eight(t: np.ndarray, scale: float = 2.0, omega: float = 0.35) -> np.ndarray:
    """Lemniscate of Gerono in the plane. Shape (T, 2) positions."""
    t = np.asarray(t, dtype=float)
    x = scale * np.sin(omega * t)
    y = scale * np.sin(omega * t) * np.cos(omega * t)
    return np.column_stack([x, y])


def line_segment(
    t: np.ndarray,
    p0: tuple[float, float] = (0.0, 0.0),
    p1: tuple[float, float] = (4.0, 0.0),
    duration: float = 8.0,
) -> np.ndarray:
    """Constant-speed line from p0 to p1 over `duration` seconds, then hold."""
    t = np.asarray(t, dtype=float)
    alpha = np.clip(t / duration, 0.0, 1.0)
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return (1.0 - alpha)[:, None] * p0 + alpha[:, None] * p1


def horizon_from_path(path_xy: np.ndarray, t0_index: int, N: int) -> np.ndarray:
    """Take N position samples starting at t0_index; pad with last point if needed."""
    T = path_xy.shape[0]
    out = np.empty((N, 2), dtype=float)
    for k in range(N):
        i = min(t0_index + k, T - 1)
        out[k] = path_xy[i]
    return out
