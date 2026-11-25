"""Time-parameterize a geometric polyline into (x,y,vx,vy) samples."""

from __future__ import annotations

import numpy as np


def profile_path(
    waypoints_xy: np.ndarray,
    dt: float,
    v_cruise: float,
    v_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constant-speed traversal along the polyline, stop at the goal.
    Returns t (T,), path_xy (T,2), path_v (T,2).
    """
    pts = np.asarray(waypoints_xy, dtype=float).reshape(-1, 2)
    if pts.shape[0] < 2:
        raise ValueError("Need at least 2 waypoints")
    v_cruise = float(np.clip(v_cruise, 1e-3, v_max))

    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    total = float(np.sum(seg_len))
    if total < 1e-9:
        t = np.array([0.0])
        return t, pts[:1].copy(), np.zeros((1, 2))

    # cumulative arc length at each vertex
    s_vertex = np.concatenate([[0.0], np.cumsum(seg_len)])
    duration = total / v_cruise
    n = max(2, int(np.ceil(duration / dt)) + 1)
    t = np.arange(n) * dt
    s_query = np.clip(t * v_cruise, 0.0, total)

    xy = np.empty((n, 2), dtype=float)
    tangents = np.empty((n, 2), dtype=float)
    for i, s in enumerate(s_query):
        # find segment
        j = int(np.searchsorted(s_vertex, s, side="right") - 1)
        j = min(max(j, 0), len(seg_len) - 1)
        if seg_len[j] < 1e-12:
            xy[i] = pts[j]
            tangents[i] = np.array([1.0, 0.0])
            continue
        a = (s - s_vertex[j]) / seg_len[j]
        a = float(np.clip(a, 0.0, 1.0))
        xy[i] = pts[j] + a * (pts[j + 1] - pts[j])
        tangents[i] = seg[j] / seg_len[j]

    speed = np.full(n, v_cruise)
    # hold at goal
    done = s_query >= total - 1e-9
    speed[done] = 0.0
    tangents[done] = tangents[np.argmax(~done)] if np.any(~done) else tangents[-1]
    vel = tangents * speed[:, None]
    return t, xy, vel
