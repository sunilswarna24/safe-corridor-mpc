"""LTV tracking MPC for unicycle AMR (linearized about the reference)."""

from __future__ import annotations

import numpy as np
import osqp
import scipy.sparse as sp

from .plant_unicycle import linearize, wrap_angle


def _ref_inputs(ref_z: np.ndarray, dt: float) -> np.ndarray:
    """Finite-difference (a, omega) along ref [x,y,theta,v]."""
    N = ref_z.shape[0]
    U = np.zeros((N, 2), dtype=float)
    for k in range(N - 1):
        dv = ref_z[k + 1, 3] - ref_z[k, 3]
        dth = wrap_angle(ref_z[k + 1, 2] - ref_z[k, 2])
        U[k, 0] = dv / dt
        U[k, 1] = dth / dt
    if N >= 2:
        U[-1] = U[-2]
    return U


def _align_ref_heading(z0_theta: float, ref_z: np.ndarray) -> np.ndarray:
    """Unwrap reference heading so it is continuous and near the plant heading."""
    out = ref_z.copy()
    th = out[:, 2]
    th[0] = z0_theta + wrap_angle(th[0] - z0_theta)
    for i in range(1, len(th)):
        th[i] = th[i - 1] + wrap_angle(out[i, 2] - out[i - 1, 2])
    out[:, 2] = th
    return out


class UnicycleMPC:
    """
    Condensed QP on stacked U=[a,omega]_0..N-1.
    Dynamics linearized about the reference each solve.
    Optional AABB corridors on (x,y).
    """

    def __init__(
        self,
        dt: float,
        N: int,
        q_pos: float,
        q_theta: float,
        q_speed: float,
        r_u: float,
        a_max: float,
        w_max: float,
        use_corridors: bool = False,
    ) -> None:
        self.dt = dt
        self.N = N
        self.a_max = a_max
        self.w_max = w_max
        self.use_corridors = use_corridors
        self.nu = 2
        self.nx = 4
        self.q_diag = np.array([q_pos, q_pos, q_theta, q_speed], dtype=float)
        self.r_u = r_u
        self._u_warm = np.zeros(self.nu * N)

    def _build_sgd(
        self, z0: np.ndarray, ref_z: np.ndarray, u_ref: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Z stacks predicted z_1..z_N = S z0 + G U + d."""
        N, nx, nu = self.N, self.nx, self.nu
        A_list, B_list, c_list = [], [], []
        for k in range(N):
            A, B, c = linearize(ref_z[k], u_ref[k], self.dt)
            A_list.append(A)
            B_list.append(B)
            c_list.append(c)

        S = np.zeros((nx * N, nx))
        G = np.zeros((nx * N, nu * N))
        d = np.zeros(nx * N)

        for k in range(N):
            if k == 0:
                S[0:nx] = A_list[0]
                d[0:nx] = c_list[0]
            else:
                prev = (k - 1) * nx
                cur = k * nx
                S[cur : cur + nx] = A_list[k] @ S[prev : prev + nx]
                d[cur : cur + nx] = A_list[k] @ d[prev : prev + nx] + c_list[k]

            for j in range(k + 1):
                P = np.eye(nx)
                for t in range(j + 1, k + 1):
                    P = A_list[t] @ P
                G[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = P @ B_list[j]
        return S, G, d

    def solve(
        self,
        z0: np.ndarray,
        ref_z: np.ndarray,
        corridors: np.ndarray | None = None,
        u_warm: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        z0: (4,) [x,y,theta,v]
        ref_z: (N, 4)
        corridors: (N, 4) optional [xmin,xmax,ymin,ymax]
        u_warm: optional (N*2,) warm-start for OSQP
        Returns first u and info (includes u_seq).
        """
        z0 = np.asarray(z0, dtype=float).reshape(4)
        ref_z = _align_ref_heading(z0[2], np.asarray(ref_z, dtype=float).reshape(self.N, 4))
        u_ref = _ref_inputs(ref_z, self.dt)
        S, G, d = self._build_sgd(z0, ref_z, u_ref)

        N, nx, nu = self.N, self.nx, self.nu
        Qb = np.kron(np.eye(N), np.diag(self.q_diag))
        Rb = np.kron(np.eye(N), self.r_u * np.eye(nu))
        r = ref_z.reshape(N * nx)
        offset = S @ z0 + d - r
        H = G.T @ Qb @ G + Rb
        H = 0.5 * (H + H.T)
        P = sp.csc_matrix(2.0 * H)
        q = 2.0 * (G.T @ Qb @ offset)

        l_u = np.tile([-self.a_max, -self.w_max], N)
        u_u = np.tile([self.a_max, self.w_max], N)

        if self.use_corridors:
            if corridors is None:
                raise ValueError("corridors required when use_corridors=True")
            boxes = np.asarray(corridors, dtype=float).reshape(N, 4)
            Sel = np.zeros((2 * N, nx * N))
            for k in range(N):
                Sel[2 * k, nx * k] = 1.0
                Sel[2 * k + 1, nx * k + 1] = 1.0
            SG = Sel @ G
            base = Sel @ (S @ z0 + d)
            l_pos = np.empty(2 * N)
            u_pos = np.empty(2 * N)
            for k in range(N):
                xmin, xmax, ymin, ymax = boxes[k]
                l_pos[2 * k] = xmin - base[2 * k]
                u_pos[2 * k] = xmax - base[2 * k]
                l_pos[2 * k + 1] = ymin - base[2 * k + 1]
                u_pos[2 * k + 1] = ymax - base[2 * k + 1]
            A = sp.vstack([sp.eye(nu * N), sp.csc_matrix(SG)], format="csc")
            l = np.concatenate([l_u, l_pos])
            u = np.concatenate([u_u, u_pos])
        else:
            A = sp.eye(nu * N, format="csc")
            l, u = l_u, u_u

        # LTV: rebuild each tick (N is small)
        prob = osqp.OSQP()
        prob.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            warm_start=True,
            verbose=False,
            polish=False,
            eps_abs=1e-3,
            eps_rel=1e-3,
            max_iter=4000,
        )
        warm = self._u_warm if u_warm is None else np.asarray(u_warm, dtype=float).reshape(nu * N)
        prob.warm_start(x=warm)

        res = prob.solve()
        info = {
            "status": res.info.status,
            "iter": int(res.info.iter),
            "solve_time_s": float(res.info.solve_time),
            "ok": False,
            "u_seq": np.zeros(nu * N),
        }
        if res.x is None or res.info.status_val not in (1, 2):
            return np.zeros(2), info
        u_seq = np.asarray(res.x, dtype=float)
        self._u_warm = u_seq.copy()
        info["ok"] = True
        info["u_seq"] = u_seq.copy()
        return u_seq[:2].copy(), info
