"""Plan on warehouse map + profile + optional corridor-constrained MPC."""

from __future__ import annotations

import time

import numpy as np

from .config import A_MAX, DT, N, PLAN_INFL, Q_POS, Q_VEL, R_INFL, R_U, V_CRUISE, V_MAX
from .corridor import corridors_along_path, horizon_corridors, paint_disk_obstacle
from .grid import downsample, inflate
from .mpc import TrackingMPC
from .plant import step
from .planner import cells_to_xy, theta_star
from .profiler import profile_path
from .reference import horizon_from_traj, stack_state_ref
from .ros_map import load_ros_map, nearest_free, resolve_map_yaml


def run_map_tracking(
    map_yaml: str = "aws_warehouse/map.yaml",
    start_xy: tuple[float, float] = (-4.0, -8.0),
    goal_xy: tuple[float, float] = (4.0, 6.0),
    v_cruise: float = V_CRUISE,
    plan_stride: int = 2,
    plan_infl_m: float = PLAN_INFL,
    use_corridors: bool = False,
    corridor_max_half: float = 1.0,
    block_obstacle: tuple[float, float, float] | None = None,
    slip_std: float = 0.0,
    accel_noise_std: float = 0.0,
    seed: int = 0,
) -> dict:
    """
    block_obstacle: optional (x, y, radius) painted onto the corridor/collision grid
    after planning, to force a constrained detour inside corridors / show collisions.
    """
    grid = load_ros_map(resolve_map_yaml(map_yaml))
    coarse = downsample(grid, plan_stride)
    plan_grid = inflate(coarse, plan_infl_m)

    sr, sc = plan_grid.xy_to_cell(*start_xy)
    gr, gc = plan_grid.xy_to_cell(*goal_xy)
    start = nearest_free(plan_grid, sr, sc)
    goal = nearest_free(plan_grid, gr, gc)

    t_plan0 = time.perf_counter()
    cells = theta_star(plan_grid, start, goal)
    plan_ms = (time.perf_counter() - t_plan0) * 1e3
    if cells is None:
        raise RuntimeError(f"No path from {start} to {goal} on {map_yaml}")
    waypoints = cells_to_xy(plan_grid, cells)

    t, path_xy, path_v = profile_path(waypoints, DT, v_cruise, V_MAX)

    # Corridor / collision grid (finer map + optional blocker)
    corr_grid = inflate(grid, R_INFL * 0.5)  # light inflate for clearance
    if block_obstacle is not None:
        bx, by, br = block_obstacle
        corr_grid = paint_disk_obstacle(corr_grid, (bx, by), br)

    bounds_full = corridors_along_path(
        corr_grid, path_xy, max_half=corridor_max_half, clearance=0.0
    )

    traj_ref = stack_state_ref(path_xy, path_v)
    pad = np.repeat(traj_ref[-1][None, :], N, axis=0)
    traj_ext = np.vstack([traj_ref, pad])
    bounds_pad = np.vstack([bounds_full, np.repeat(bounds_full[-1][None, :], N, axis=0)])

    steps = len(t)
    z = np.array([path_xy[0, 0], path_xy[0, 1], 0.0, 0.0])
    mpc = TrackingMPC(DT, N, Q_POS, Q_VEL, R_U, A_MAX, use_corridors=use_corridors)
    rng = np.random.default_rng(seed)

    traj = np.zeros((steps, 4))
    err_pos = np.zeros(steps)
    err_vel = np.zeros(steps)
    solve_ms = np.zeros(steps)
    u_prev = np.zeros(2)
    ok_count = 0
    collisions = 0

    for i in range(steps):
        ref = horizon_from_traj(traj_ext, i + 1, N)
        boxes = horizon_corridors(bounds_pad, i + 1, N) if use_corridors else None
        t0 = time.perf_counter()
        u0, info = mpc.solve(z, ref, corridors=boxes)
        solve_ms[i] = (time.perf_counter() - t0) * 1e3
        if info["ok"]:
            ok_count += 1
        z, u_prev = step(
            z,
            u0,
            DT,
            V_MAX,
            A_MAX,
            u_prev=u_prev,
            slip_std=slip_std,
            accel_noise_std=accel_noise_std,
            rng=rng,
        )
        traj[i] = z
        err_pos[i] = np.linalg.norm(z[:2] - path_xy[i])
        err_vel[i] = np.linalg.norm(z[2:] - path_v[i])
        rr, cc = corr_grid.xy_to_cell(z[0], z[1])
        if not corr_grid.free(rr, cc):
            collisions += 1

    return {
        "grid": grid,
        "corr_grid": corr_grid,
        "plan_grid": plan_grid,
        "waypoints": waypoints,
        "corridor_bounds": bounds_full,
        "start": start,
        "goal": goal,
        "start_xy": plan_grid.cell_to_xy(*start),
        "goal_xy": plan_grid.cell_to_xy(*goal),
        "t": t,
        "path_xy": path_xy,
        "path_v": path_v,
        "traj_xy": traj[:, :2],
        "traj": traj,
        "err": err_pos,
        "err_vel": err_vel,
        "solve_ms": solve_ms,
        "ok_frac": ok_count / steps,
        "err_rms": float(np.sqrt(np.mean(err_pos**2))),
        "err_max": float(np.max(err_pos)),
        "err_vel_rms": float(np.sqrt(np.mean(err_vel**2))),
        "solve_ms_med": float(np.median(solve_ms)),
        "solve_ms_p95": float(np.percentile(solve_ms, 95)),
        "plan_ms": float(plan_ms),
        "n_cells": len(cells),
        "n_waypoints": len(waypoints),
        "map_yaml": map_yaml,
        "use_corridors": use_corridors,
        "collisions": int(collisions),
        "block_obstacle": block_obstacle,
        "slip_std": slip_std,
        "accel_noise_std": accel_noise_std,
        "seed": seed,
    }
