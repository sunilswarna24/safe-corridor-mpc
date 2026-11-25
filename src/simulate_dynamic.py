"""Closed-loop tracking with crossing dynamic disks + reactive / predicted corridors."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from .config import (
    A_MAX,
    DT,
    N,
    PLAN_INFL,
    Q_POS,
    Q_SPEED,
    Q_THETA,
    Q_VEL,
    R_INFL,
    R_ROBOT,
    R_U,
    V_MAX,
    W_MAX,
)
from .corridor import corridors_along_path, corridors_predicted, paint_movers
from .grid import downsample, inflate
from .movers import CrossingDisk
from .mpc import TrackingMPC
from .mpc_unicycle import UnicycleMPC
from .plant import step as step_di
from .plant_unicycle import step as step_uni
from .planner import cells_to_xy, theta_star
from .profiler import profile_path
from .reference import horizon_from_traj, stack_state_ref, stack_unicycle_ref
from .ros_map import load_ros_map, nearest_free, resolve_map_yaml
from .scenarios import two_crossers
from .supervisory import SupervisoryConfig, SupervisoryGate


def run_dynamic_tracking(
    map_yaml: str = "aws_warehouse/map.yaml",
    start_xy: tuple[float, float] = (-4.0, -8.0),
    goal_xy: tuple[float, float] = (4.0, 6.0),
    v_cruise: float = 0.55,
    plan_stride: int = 2,
    corridor_max_half: float = 1.0,
    reactive: bool = True,
    use_corridors: bool = True,
    use_supervisory: bool = False,
    predict_movers: bool = False,
    plant: str = "di",
    mover: CrossingDisk | None = None,
    movers: Sequence[CrossingDisk] | None = None,
    x0_offset: tuple[float, float] = (0.0, 0.0),
    slip_std: float = 0.0,
    accel_noise_std: float = 0.0,
    yaw_noise_std: float = 0.0,
    seed: int = 0,
    warm_start_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
    cold_start: bool = False,
) -> dict:
    """
    plant: "di" (double integrator) or "unicycle"
    predict_movers: stage-wise open-loop mover paint into corridors (else myopic current pose)
    warm_start_fn: optional (z, ref, corridors) -> U flat for OSQP warm-start (unicycle)
    cold_start: if True (unicycle), force U0=0 each tick instead of shifting last solve
    """
    if plant not in ("di", "unicycle"):
        raise ValueError("plant must be 'di' or 'unicycle'")

    if movers is not None:
        mlist = list(movers)
    elif mover is not None:
        mlist = [mover]
    else:
        mlist = two_crossers()
    if not mlist:
        raise ValueError("need at least one mover")

    grid = load_ros_map(resolve_map_yaml(map_yaml))
    plan_grid = inflate(downsample(grid, plan_stride), PLAN_INFL)
    start = nearest_free(plan_grid, *plan_grid.xy_to_cell(*start_xy))
    goal = nearest_free(plan_grid, *plan_grid.xy_to_cell(*goal_xy))

    cells = theta_star(plan_grid, start, goal)
    if cells is None:
        raise RuntimeError("No path")
    waypoints = cells_to_xy(plan_grid, cells)
    t_path, path_xy, path_v = profile_path(waypoints, DT, v_cruise, V_MAX)

    base_corr = inflate(grid, R_INFL * 0.5)
    if plant == "unicycle":
        traj_ref = stack_unicycle_ref(path_xy, path_v)
    else:
        traj_ref = stack_state_ref(path_xy, path_v)
    pad = np.repeat(traj_ref[-1][None, :], N, axis=0)
    traj_ext = np.vstack([traj_ref, pad])

    frozen_grid = paint_movers(base_corr, mlist, 0.0, extra_radius=R_ROBOT)
    frozen_bounds = corridors_along_path(frozen_grid, path_xy, max_half=corridor_max_half)
    frozen_pad = np.vstack([frozen_bounds, np.repeat(frozen_bounds[-1][None, :], N, axis=0)])

    path_steps = len(t_path)
    max_ticks = int(path_steps * 1.5) + 50
    n_mov = len(mlist)

    z0_xy = path_xy[0] + np.asarray(x0_offset, dtype=float)
    zr, zc = nearest_free(base_corr, *base_corr.xy_to_cell(float(z0_xy[0]), float(z0_xy[1])))
    z0_xy = np.asarray(base_corr.cell_to_xy(zr, zc), dtype=float)

    if plant == "unicycle":
        th0 = float(traj_ref[0, 2])
        z = np.array([z0_xy[0], z0_xy[1], th0, 0.0], dtype=float)
        mpc = UnicycleMPC(
            DT, N, Q_POS, Q_THETA, Q_SPEED, R_U, A_MAX, W_MAX, use_corridors=use_corridors
        )
    else:
        z = np.array([z0_xy[0], z0_xy[1], 0.0, 0.0], dtype=float)
        mpc = TrackingMPC(DT, N, Q_POS, Q_VEL, R_U, A_MAX, use_corridors=use_corridors)

    gate_cfg = SupervisoryConfig(plant=plant)
    gate = SupervisoryGate(gate_cfg) if use_supervisory else None
    rng = np.random.default_rng(seed)
    open_box = np.array([-1e3, 1e3, -1e3, 1e3], dtype=float)

    traj = np.zeros((max_ticks, 4))
    movers_xy = np.zeros((max_ticks, n_mov, 2))
    err_pos = np.zeros(max_ticks)
    solve_ms = np.zeros(max_ticks)
    solve_iters = np.zeros(max_ticks)
    t_wall = np.zeros(max_ticks)
    waiting_flags = np.zeros(max_ticks, dtype=bool)
    sealed_ticks = 0
    corridor_snap = None
    ok_count = 0
    collisions = 0
    mover_hits = 0
    u_seqs: list[np.ndarray] = []
    feat_rows: list[np.ndarray] = []
    prog = 0
    k = 0

    while prog < path_steps - 1 and k < max_ticks:
        ti = k * DT
        t_wall[k] = ti
        for i, m in enumerate(mlist):
            movers_xy[k, i] = m.position(ti)

        if gate is None:
            prog = min(k, path_steps - 1)

        ref = horizon_from_traj(traj_ext, prog + 1, N)
        if use_corridors:
            if reactive:
                if predict_movers:
                    boxes = corridors_predicted(
                        base_corr,
                        mlist,
                        ti,
                        DT,
                        ref[:, :2],
                        max_half=corridor_max_half,
                        extra_radius=R_ROBOT,
                    )
                else:
                    g = paint_movers(base_corr, mlist, ti, extra_radius=R_ROBOT)
                    boxes = corridors_along_path(g, ref[:, :2], max_half=corridor_max_half)
            else:
                boxes = np.empty((N, 4), dtype=float)
                for j in range(N):
                    boxes[j] = frozen_pad[min(prog + 1 + j, len(frozen_pad) - 1)]
        else:
            boxes = np.repeat(open_box[None, :], N, axis=0)

        if corridor_snap is None and prog >= path_steps // 2 and use_corridors:
            corridor_snap = boxes.copy()

        allow, reason = True, "ok"
        if gate is not None:
            allow, reason = gate.pre_check(
                z, boxes, mlist, ti, DT, path_ahead=ref[:, :2]
            )
            if reason == "sealed_corridor":
                sealed_ticks += 1

        u_warm = None
        if plant == "unicycle":
            if warm_start_fn is not None:
                try:
                    u_warm = warm_start_fn(z, ref, boxes)
                except Exception:
                    u_warm = None
            elif cold_start:
                u_warm = np.zeros(N * 2, dtype=float)

        t0 = time.perf_counter()
        if plant == "unicycle":
            u_mpc, info = mpc.solve(
                z, ref, corridors=boxes if use_corridors else None, u_warm=u_warm
            )
        else:
            u_mpc, info = mpc.solve(z, ref, corridors=boxes if use_corridors else None)
        solve_ms[k] = (time.perf_counter() - t0) * 1e3
        solve_iters[k] = float(info.get("iter", 0))
        if info["ok"]:
            ok_count += 1
            if plant == "unicycle" and "u_seq" in info:
                u_seqs.append(info["u_seq"].copy())
                # compact features for NN: z(4) + ref xy (N*2) + box centers/half (N*4)
                centers = 0.5 * np.column_stack(
                    [boxes[:, 0] + boxes[:, 1], boxes[:, 2] + boxes[:, 3]]
                )
                half = 0.5 * np.column_stack(
                    [boxes[:, 1] - boxes[:, 0], boxes[:, 3] - boxes[:, 2]]
                )
                feat = np.concatenate(
                    [z, ref[:, :2].ravel(), centers.ravel(), half.ravel()]
                )
                feat_rows.append(feat)

        if gate is not None:
            u_cmd = gate.post_command(u_mpc, z, info["ok"], allow, reason)
        else:
            u_cmd = u_mpc if info["ok"] else np.zeros(2)

        if plant == "unicycle":
            z, _ = step_uni(
                z,
                u_cmd,
                DT,
                V_MAX,
                A_MAX,
                W_MAX,
                slip_std=slip_std,
                accel_noise_std=accel_noise_std,
                yaw_noise_std=yaw_noise_std,
                rng=rng,
            )
        else:
            z, _ = step_di(
                z,
                u_cmd,
                DT,
                V_MAX,
                A_MAX,
                slip_std=slip_std,
                accel_noise_std=accel_noise_std,
                rng=rng,
            )
        traj[k] = z
        err_pos[k] = np.linalg.norm(z[:2] - path_xy[min(prog, path_steps - 1)])
        if gate is not None:
            waiting_flags[k] = bool(gate.state.waiting) or (not allow)

        rr, cc = base_corr.xy_to_cell(z[0], z[1])
        if not base_corr.free(rr, cc):
            collisions += 1
        for i, m in enumerate(mlist):
            if m.collides(z[:2], ti, robot_radius=R_ROBOT):
                mover_hits += 1

        if gate is not None and allow:
            prog = min(prog + 1, path_steps - 1)
        k += 1

    n = max(k, 1)
    movers_xy_n = movers_xy[:n]
    return {
        "grid": grid,
        "waypoints": waypoints,
        "path_xy": path_xy,
        "traj_xy": traj[:n, :2],
        "traj": traj[:n],
        "t": t_wall[:n],
        "err": err_pos[:n],
        "solve_ms": solve_ms[:n],
        "solve_iters": solve_iters[:n],
        "movers_xy": movers_xy_n,
        "mover_xy": movers_xy_n[:, 0, :] if n_mov else np.zeros((n, 2)),
        "waiting": waiting_flags[:n],
        "movers": mlist,
        "mover": mlist[0] if mlist else None,
        "corridor_snap": corridor_snap,
        "reactive": reactive,
        "predict_movers": predict_movers,
        "use_corridors": use_corridors,
        "use_supervisory": use_supervisory,
        "plant": plant,
        "x0_offset": (float(x0_offset[0]), float(x0_offset[1])),
        "z0_xy": (float(z0_xy[0]), float(z0_xy[1])),
        "ok_frac": ok_count / n,
        "err_rms": float(np.sqrt(np.mean(err_pos[:n] ** 2))),
        "err_max": float(np.max(err_pos[:n])),
        "solve_ms_med": float(np.median(solve_ms[:n])),
        "solve_ms_p95": float(np.percentile(solve_ms[:n], 95)),
        "solve_iters_mean": float(np.mean(solve_iters[:n])),
        "collisions": int(collisions),
        "mover_hits": int(mover_hits),
        "sealed_ticks": int(sealed_ticks),
        "fallback_count": int(gate.state.fallback_count) if gate else 0,
        "wait_count": int(gate.state.wait_count) if gate else 0,
        "slew_count": int(gate.state.slew_count) if gate else 0,
        "ticks": n,
        "path_steps": path_steps,
        "start_xy": plan_grid.cell_to_xy(*start),
        "goal_xy": plan_grid.cell_to_xy(*goal),
        "u_seqs": u_seqs,
        "feat_rows": feat_rows,
    }
