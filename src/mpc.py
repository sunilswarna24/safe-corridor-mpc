"""Condensed tracking MPC for the 2D double integrator (OSQP)."""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from .plant import discrete_matrices


class TrackingMPC:
    """
    min_U  sum_k ||p_k - r_k||^2_Q + ||u_k||^2_R
    s.t.   z_{k+1} = Ad z_k + Bd u_k
           |u_k| <= a_max

    U = [u_0; ...; u_{N-1}] in R^{2N}. Position refs are for z_1..z_N.
    """

    def __init__(
        self,
        dt: float,
        N: int,
        q_pos: float,
        r_u: float,
        a_max: float,
    ) -> None:
        self.dt = dt
        self.N = N
        self.a_max = a_max
        self.nu = 2
        nu = 2
        nx = 4

        Ad, Bd = discrete_matrices(dt)
        C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])

        Ad_pow = [np.eye(nx)]
        for _ in range(N):
            Ad_pow.append(Ad @ Ad_pow[-1])

        S = np.zeros((2 * N, nx))
        G = np.zeros((2 * N, nu * N))
        for k in range(N):
            S[2 * k : 2 * k + 2] = C @ Ad_pow[k + 1]
            for j in range(k + 1):
                G[2 * k : 2 * k + 2, nu * j : nu * j + nu] = C @ (Ad_pow[k - j] @ Bd)

        self.S = S
        self.G = G
        Qb = np.kron(np.eye(N), q_pos * np.eye(2))
        Rb = np.kron(np.eye(N), r_u * np.eye(2))
        H = G.T @ Qb @ G + Rb
        H = 0.5 * (H + H.T)

        # OSQP: 0.5 U' P U + q' U  ≡  U' H U + 2 (Sx-r)' Q G U
        self.P_osqp = sp.csc_matrix(2.0 * H)
        self._GTQ = G.T @ Qb

        self.A_u = sp.eye(nu * N, format="csc")
        self.l_u = -a_max * np.ones(nu * N)
        self.u_u = a_max * np.ones(nu * N)

        self.prob = osqp.OSQP()
        self.prob.setup(
            P=self.P_osqp,
            q=np.zeros(nu * N),
            A=self.A_u,
            l=self.l_u,
            u=self.u_u,
            warm_start=True,
            verbose=False,
            polish=False,
            eps_abs=1e-4,
            eps_rel=1e-4,
        )
        self._u_prev = np.zeros(nu * N)

    def solve(self, z0: np.ndarray, ref_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        z0 = np.asarray(z0, dtype=float).reshape(4)
        r = np.asarray(ref_xy, dtype=float).reshape(self.N * 2)
        q = 2.0 * (self._GTQ @ (self.S @ z0 - r))
        self.prob.update(q=q)
        res = self.prob.solve()
        info = {
            "status": res.info.status,
            "iter": int(res.info.iter),
            "solve_time_s": float(res.info.solve_time),
            "ok": False,
        }
        if res.x is None or res.info.status_val not in (1, 2):
            u_seq = self._u_prev
        else:
            u_seq = res.x
            self._u_prev = u_seq.copy()
            info["ok"] = True
        return u_seq[:2].copy(), info
