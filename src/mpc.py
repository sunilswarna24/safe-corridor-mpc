"""Condensed tracking MPC for the 2D double integrator (OSQP)."""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from .plant import discrete_matrices


class TrackingMPC:
    """
    min_U  sum_k ||z_k - r_k||^2_Q + ||u_k||^2_R
    s.t.   z_{k+1} = Ad z_k + Bd u_k
           |u_k| <= a_max
           xmin_k <= x_k <= xmax_k, ymin_k <= y_k <= ymax_k   (optional)

    Tracks position and velocity. r_k = [x,y,vx,vy] for steps 1..N.
    """

    def __init__(
        self,
        dt: float,
        N: int,
        q_pos: float,
        q_vel: float,
        r_u: float,
        a_max: float,
        use_corridors: bool = False,
    ) -> None:
        self.dt = dt
        self.N = N
        self.a_max = a_max
        self.use_corridors = use_corridors
        nu = 2
        nx = 4
        self.nu = nu
        self.nx = nx

        Ad, Bd = discrete_matrices(dt)

        Ad_pow = [np.eye(nx)]
        for _ in range(N):
            Ad_pow.append(Ad @ Ad_pow[-1])

        S = np.zeros((nx * N, nx))
        G = np.zeros((nx * N, nu * N))
        for k in range(N):
            S[nx * k : nx * (k + 1)] = Ad_pow[k + 1]
            for j in range(k + 1):
                G[nx * k : nx * (k + 1), nu * j : nu * (j + 1)] = Ad_pow[k - j] @ Bd

        self.S = S
        self.G = G

        # Position selector: stacked [x0,y0,x1,y1,...]
        Sel = np.zeros((2 * N, nx * N))
        for k in range(N):
            Sel[2 * k, nx * k + 0] = 1.0
            Sel[2 * k + 1, nx * k + 1] = 1.0
        self.Sel = Sel
        self.Sel_S = Sel @ S
        self.Sel_G = Sel @ G

        q_diag = np.array([q_pos, q_pos, q_vel, q_vel])
        Qb = np.kron(np.eye(N), np.diag(q_diag))
        Rb = np.kron(np.eye(N), r_u * np.eye(nu))
        H = G.T @ Qb @ G + Rb
        H = 0.5 * (H + H.T)
        self.P_osqp = sp.csc_matrix(2.0 * H)
        self._GTQ = G.T @ Qb

        A_u = sp.eye(nu * N, format="csc")
        if use_corridors:
            A_pos = sp.csc_matrix(self.Sel_G)
            self.A = sp.vstack([A_u, A_pos], format="csc")
            self.l = np.concatenate([-a_max * np.ones(nu * N), -np.inf * np.ones(2 * N)])
            self.u = np.concatenate([a_max * np.ones(nu * N), np.inf * np.ones(2 * N)])
        else:
            self.A = A_u
            self.l = -a_max * np.ones(nu * N)
            self.u = a_max * np.ones(nu * N)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.P_osqp,
            q=np.zeros(nu * N),
            A=self.A,
            l=self.l,
            u=self.u,
            warm_start=True,
            verbose=False,
            polish=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
        )
        self._u_prev = np.zeros(nu * N)

    def solve(
        self,
        z0: np.ndarray,
        ref_z: np.ndarray,
        corridors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        ref_z: (N, 4)
        corridors: (N, 4) [xmin, xmax, ymin, ymax] if use_corridors
        """
        z0 = np.asarray(z0, dtype=float).reshape(4)
        r = np.asarray(ref_z, dtype=float).reshape(self.N * self.nx)
        q = 2.0 * (self._GTQ @ (self.S @ z0 - r))

        if self.use_corridors:
            if corridors is None:
                raise ValueError("corridors required when use_corridors=True")
            boxes = np.asarray(corridors, dtype=float).reshape(self.N, 4)
            # predicted pos = Sel_S z0 + Sel_G U
            offset = self.Sel_S @ z0
            l_pos = np.empty(2 * self.N)
            u_pos = np.empty(2 * self.N)
            for k in range(self.N):
                xmin, xmax, ymin, ymax = boxes[k]
                l_pos[2 * k] = xmin - offset[2 * k]
                u_pos[2 * k] = xmax - offset[2 * k]
                l_pos[2 * k + 1] = ymin - offset[2 * k + 1]
                u_pos[2 * k + 1] = ymax - offset[2 * k + 1]
            l = np.concatenate([-self.a_max * np.ones(self.nu * self.N), l_pos])
            u = np.concatenate([self.a_max * np.ones(self.nu * self.N), u_pos])
            self.prob.update(q=q, l=l, u=u)
        else:
            self.prob.update(q=q)

        res = self.prob.solve()
        info = {
            "status": res.info.status,
            "iter": int(res.info.iter),
            "solve_time_s": float(res.info.solve_time),
            "ok": False,
        }
        if res.x is None or res.info.status_val not in (1, 2):
            # Zero accel (coast). SupervisoryGate hard-brakes when ok is False.
            info["ok"] = False
            return np.zeros(2), info
        u_seq = res.x
        self._u_prev = u_seq.copy()
        info["ok"] = True
        return u_seq[:2].copy(), info
