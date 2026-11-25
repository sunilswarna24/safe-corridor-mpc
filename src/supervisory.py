"""Deterministic supervisory gate around the tracking MPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import A_MAX, DT, R_ROBOT
from .movers import CrossingDisk


@dataclass
class SupervisoryConfig:
    slew_max: float = 3.0  # m/s^2 per tick
    min_corridor_half: float = 0.06  # m
    # Enter wait when dist <= stop; leave when dist >= resume (hysteresis)
    stop_clear: float = 1.10  # m beyond geometric contact
    resume_clear: float = 1.80  # m beyond geometric contact
    a_max: float = A_MAX
    dt: float = DT
    robot_radius: float = R_ROBOT


@dataclass
class SupervisoryState:
    u_prev: np.ndarray = field(default_factory=lambda: np.zeros(2))
    fallback_count: int = 0
    wait_count: int = 0
    slew_count: int = 0
    waiting: bool = False


class SupervisoryGate:
    def __init__(self, cfg: SupervisoryConfig | None = None) -> None:
        self.cfg = cfg or SupervisoryConfig()
        self.state = SupervisoryState()

    def reset(self) -> None:
        self.state = SupervisoryState()

    def _forward_unit(self, z: np.ndarray, path_ahead: np.ndarray | None) -> np.ndarray:
        v = np.asarray(z[2:4], dtype=float)
        if float(np.linalg.norm(v)) >= 0.08:
            return v / np.linalg.norm(v)
        if path_ahead is not None and len(path_ahead) > 3:
            pts = np.asarray(path_ahead, dtype=float).reshape(-1, 2)
            d = pts[min(8, len(pts) - 1)] - np.asarray(z[:2], dtype=float)
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                return d / n
        return np.array([0.0, 1.0])

    def _mover_block(
        self, z: np.ndarray, movers: Sequence[CrossingDisk], t: float, path_ahead: np.ndarray | None
    ) -> bool:
        """Reactive yield on current poses only (no predicted mover trajectories)."""
        xy = np.asarray(z[:2], dtype=float)
        ahead = self._forward_unit(z, path_ahead)
        any_close_ahead = False
        min_dist = float("inf")
        for m in movers:
            mxy = m.position(t)
            dist = float(np.linalg.norm(xy - mxy))
            min_dist = min(min_dist, dist)
            contact = float(m.radius + self.cfg.robot_radius)
            stop_d = contact + self.cfg.stop_clear
            # Forward cone (~60°): avoids re-yielding on a mover already beside/behind
            fwd = float(np.dot(mxy - xy, ahead))
            mover_ahead = fwd > 0.5 * max(dist, 1e-6)
            if mover_ahead and dist <= stop_d:
                any_close_ahead = True

        contact_resume = self.cfg.robot_radius + self.cfg.resume_clear
        # resume if far from every mover (use max mover radius among set)
        r_max = max((m.radius for m in movers), default=0.0)
        resume_d = r_max + contact_resume

        if self.state.waiting:
            # Clear when far from everyone, or nothing close ahead anymore
            if min_dist >= resume_d or not any_close_ahead:
                self.state.waiting = False
                return False
            return True
        if any_close_ahead:
            self.state.waiting = True
            return True
        return False

    def pre_check(
        self,
        z: np.ndarray,
        corridors: np.ndarray,
        movers: Sequence[CrossingDisk] | CrossingDisk | None,
        t: float,
        dt: float,
        path_ahead: np.ndarray | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (allow_mpc, reason).

        mover_close latches waiting (hold path index until clear).
        sealed_corridor denies this tick only (no latch).
        """
        del dt
        boxes = np.asarray(corridors, dtype=float).reshape(-1, 4)
        sealed = False
        for i in range(min(6, len(boxes))):
            xmin, xmax, ymin, ymax = boxes[i]
            if 0.5 * min(xmax - xmin, ymax - ymin) < self.cfg.min_corridor_half:
                sealed = True
                break

        mlist: list[CrossingDisk] = []
        if movers is not None:
            if isinstance(movers, CrossingDisk):
                mlist = [movers]
            else:
                mlist = list(movers)

        if mlist and self._mover_block(z, mlist, t, path_ahead):
            return False, "mover_close"

        if sealed:
            return False, "sealed_corridor"
        return True, "ok"

    def brake(self, z: np.ndarray) -> np.ndarray:
        """Max-decel toward zero velocity (u=0 only coasts on a double integrator)."""
        v = np.asarray(z[2:4], dtype=float)
        return np.clip(-v / self.cfg.dt, -self.cfg.a_max, self.cfg.a_max)

    def post_command(
        self,
        u_mpc: np.ndarray,
        z: np.ndarray,
        qp_ok: bool,
        allow_mpc: bool,
        reason: str,
    ) -> np.ndarray:
        if (not allow_mpc) or (not qp_ok):
            u = self.brake(z)
            self.state.fallback_count += 1
            if not allow_mpc and reason == "mover_close":
                self.state.wait_count += 1
        else:
            u = np.asarray(u_mpc, dtype=float).reshape(2).copy()

        du = np.clip(u - self.state.u_prev, -self.cfg.slew_max, self.cfg.slew_max)
        u_cmd = self.state.u_prev + du
        if np.linalg.norm(u_cmd - u) > 1e-9 and allow_mpc and qp_ok:
            self.state.slew_count += 1
        self.state.u_prev = u_cmd.copy()
        return u_cmd
