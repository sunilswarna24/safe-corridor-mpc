"""Scripted dynamic obstacles (disk movers)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CrossingDisk:
    """
    Disk that travels a line segment — worker / forklift stand-in.

    one_shot=False: shuttle back and forth (stress reactive corridors).
    one_shot=True: p0→p1 once, then stay at p1 (clean yield demo).
    """

    p0: tuple[float, float]
    p1: tuple[float, float]
    radius: float
    speed: float  # m/s along the segment
    phase: float = 0.0  # seconds offset
    one_shot: bool = False

    def position(self, t: float) -> np.ndarray:
        p0 = np.asarray(self.p0, dtype=float)
        p1 = np.asarray(self.p1, dtype=float)
        delta = p1 - p0
        length = float(np.linalg.norm(delta))
        if length < 1e-9 or self.speed <= 0:
            return p0.copy()
        if self.one_shot:
            dist = min(max(t + self.phase, 0.0) * self.speed, length)
            a = dist / length
            return (1.0 - a) * p0 + a * p1
        period = 2.0 * length / self.speed
        tau = (t + self.phase) % period
        dist = tau * self.speed
        if dist <= length:
            a = dist / length
        else:
            a = 1.0 - (dist - length) / length
        return (1.0 - a) * p0 + a * p1

    def collides(self, xy: np.ndarray, t: float, robot_radius: float = 0.0) -> bool:
        d = np.linalg.norm(np.asarray(xy, dtype=float)[:2] - self.position(t))
        return bool(d <= self.radius + robot_radius)
