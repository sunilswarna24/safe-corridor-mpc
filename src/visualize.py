"""Plot helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .config import DT, R_ROBOT
from .grid import OccupancyGrid


def _crossing_frame(traj_xy: np.ndarray, mover_xy: np.ndarray) -> int:
    """Index where robot and mover are closest (better than raw midpoint)."""
    d = np.linalg.norm(traj_xy - mover_xy, axis=1)
    return int(np.argmin(d))


def plot_tracking(
    path_xy: np.ndarray,
    traj_xy: np.ndarray,
    out_path: Path,
    title: str = "M1 unconstrained tracking MPC",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.65", lw=1.5, label="reference")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="C0", lw=2.0, label="MPC")
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="C2", ms=6, label="start")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_errors(
    t: np.ndarray,
    err: np.ndarray,
    solve_ms: np.ndarray,
    out_path: Path,
    title: str = "Tracking error & solve time",
) -> None:
    out_path = Path(out_path)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), dpi=140, sharex=True)
    axes[0].plot(t, err, color="C3", lw=1.5)
    axes[0].set_ylabel("pos error [m]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, solve_ms, color="C0", lw=1.0)
    axes[1].set_ylabel("OSQP solve [ms]")
    axes[1].set_xlabel("t [s]")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_map_tracking(
    grid: OccupancyGrid,
    waypoints: np.ndarray,
    path_xy: np.ndarray,
    traj_xy: np.ndarray,
    out_path: Path,
    title: str = "Nav2 warehouse — Theta* + profiler + MPC",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = grid.occ.shape
    ox, oy = grid.origin_xy
    res = grid.resolution
    extent = [ox, ox + w * res, oy, oy + h * res]

    # Crop view around the trajectory for a readable portfolio figure
    pts = np.vstack([path_xy, traj_xy, waypoints])
    margin = 2.0
    xmin, ymin = pts.min(axis=0) - margin
    xmax, ymax = pts.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=140)
    ax.imshow(
        grid.occ.astype(float),
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
        alpha=0.95,
    )
    ax.plot(waypoints[:, 0], waypoints[:, 1], color="C1", lw=1.4, alpha=0.9, label="Theta*")
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.4", lw=1.0, label="profiled ref")
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="C0", lw=2.0, label="MPC")
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="C2", ms=6, label="start")
    ax.plot(path_xy[-1, 0], path_xy[-1, 1], "x", color="C3", ms=8, mew=2, label="goal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_map_corridors(
    grid: OccupancyGrid,
    path_xy: np.ndarray,
    traj_xy: np.ndarray,
    corridor_bounds: np.ndarray,
    out_path: Path,
    traj_b: np.ndarray | None = None,
    block_obstacle: tuple[float, float, float] | None = None,
    title: str = "M2 corridor-constrained MPC",
    label_a: str = "MPC + corridors",
    label_b: str = "MPC no corridors",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = grid.occ.shape
    ox, oy = grid.origin_xy
    res = grid.resolution
    extent = [ox, ox + w * res, oy, oy + h * res]

    pts = [path_xy, traj_xy]
    if traj_b is not None:
        pts.append(traj_b)
    pts = np.vstack(pts)
    margin = 2.0
    xmin, ymin = pts.min(axis=0) - margin
    xmax, ymax = pts.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=140)
    ax.imshow(
        grid.occ.astype(float),
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
        alpha=0.95,
    )

    # draw every Nth corridor box to avoid clutter
    step = max(1, len(corridor_bounds) // 25)
    first = True
    for i in range(0, len(corridor_bounds), step):
        x0, x1, y0, y1 = corridor_bounds[i]
        rect = plt.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="C1",
            lw=0.8,
            alpha=0.55,
            label="corridor" if first else None,
        )
        ax.add_patch(rect)
        first = False

    ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.45", lw=1.0, label="profiled ref")
    if traj_b is not None:
        ax.plot(traj_b[:, 0], traj_b[:, 1], color="C3", lw=1.6, alpha=0.9, label=label_b)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="C0", lw=2.0, label=label_a)
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="C2", ms=6, label="start")
    ax.plot(path_xy[-1, 0], path_xy[-1, 1], "x", color="C3", ms=8, mew=2, label="goal")

    if block_obstacle is not None:
        bx, by, br = block_obstacle
        circ = plt.Circle((bx, by), br, color="k", alpha=0.35, label="blocker")
        ax.add_patch(circ)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _as_movers_xy(
    mover_xy: np.ndarray,
    movers_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize to (T, M, 2)."""
    if movers_xy is not None:
        arr = np.asarray(movers_xy, dtype=float)
        if arr.ndim == 2:
            return arr[:, None, :]
        return arr
    arr = np.asarray(mover_xy, dtype=float)
    if arr.ndim == 2:
        return arr[:, None, :]
    return arr


def plot_dynamic(
    grid: OccupancyGrid,
    path_xy: np.ndarray,
    traj_xy: np.ndarray,
    mover_xy: np.ndarray,
    mover_radius: float,
    out_path: Path,
    corridor_snap: np.ndarray | None = None,
    traj_b: np.ndarray | None = None,
    title: str = "Dynamic crossing obstacle",
    label_a: str = "reactive corridors",
    label_b: str = "frozen corridors",
    movers_xy: np.ndarray | None = None,
    mover_radii: list[float] | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mxy = _as_movers_xy(mover_xy, movers_xy)
    n_mov = mxy.shape[1]
    radii = mover_radii if mover_radii is not None else [mover_radius] * n_mov

    h, w = grid.occ.shape
    ox, oy = grid.origin_xy
    res = grid.resolution
    extent = [ox, ox + w * res, oy, oy + h * res]

    pts = np.vstack([path_xy, traj_xy, mxy.reshape(-1, 2)])
    if traj_b is not None:
        pts = np.vstack([pts, traj_b])
    margin = 2.0
    xmin, ymin = pts.min(axis=0) - margin
    xmax, ymax = pts.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=140)
    ax.imshow(
        grid.occ.astype(float),
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
        alpha=0.95,
    )
    if corridor_snap is not None:
        step = max(1, len(corridor_snap) // 20)
        first = True
        for i in range(0, len(corridor_snap), step):
            x0, x1, y0, y1 = corridor_snap[i]
            ax.add_patch(
                plt.Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="C1",
                    lw=0.8,
                    alpha=0.5,
                    label="corridor @ mid" if first else None,
                )
            )
            first = False

    ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.45", lw=1.0, label="profiled ref")
    if traj_b is not None:
        ax.plot(traj_b[:, 0], traj_b[:, 1], color="C3", lw=1.5, alpha=0.85, label=label_b)
    ax.plot(traj_xy[:, 0], traj_xy[:, 1], color="C0", lw=2.0, label=label_a)
    for mi in range(n_mov):
        lab = "mover path" if mi == 0 else f"mover {mi + 1} path"
        ax.plot(mxy[:, mi, 0], mxy[:, mi, 1], color="k", lw=1.2, ls="--" if mi == 0 else ":", label=lab)
        mid = _crossing_frame(traj_xy, mxy[:, mi, :])
        ax.add_patch(
            plt.Circle(
                (mxy[mid, mi, 0], mxy[mid, mi, 1]),
                radii[mi],
                color="0.2",
                alpha=0.35,
                label="mover @ closest" if mi == 0 else None,
            )
        )
    ax.plot(traj_xy[0, 0], traj_xy[0, 1], "o", color="C2", ms=6, label="start")
    ax.plot(path_xy[-1, 0], path_xy[-1, 1], "x", color="C3", ms=8, mew=2, label="goal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def animate_yield(
    grid: OccupancyGrid,
    path_xy: np.ndarray,
    traj_xy: np.ndarray,
    mover_xy: np.ndarray,
    mover_radius: float,
    out_path: Path,
    waiting: np.ndarray | None = None,
    title: str = "Supervisory yield",
    stride: int = 3,
    fps: int = 12,
    movers_xy: np.ndarray | None = None,
    mover_radii: list[float] | None = None,
) -> None:
    """GIF of robot + all movers (full run, stride-subsampled)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mxy = _as_movers_xy(mover_xy, movers_xy)
    n_mov = mxy.shape[1]
    radii = mover_radii if mover_radii is not None else [mover_radius] * n_mov

    n = len(traj_xy)
    idx = np.arange(0, n, max(1, stride))
    if len(idx) < 2:
        idx = np.arange(n)

    h, w = grid.occ.shape
    ox, oy = grid.origin_xy
    res = grid.resolution
    extent = [ox, ox + w * res, oy, oy + h * res]

    pts = np.vstack([traj_xy, mxy.reshape(-1, 2), path_xy])
    margin = 1.5
    xmin, ymin = pts.min(axis=0) - margin
    xmax, ymax = pts.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=100)
    ax.imshow(
        grid.occ.astype(float),
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
        alpha=0.95,
    )
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.55", lw=1.0, label="ref")
    trail_r, = ax.plot([], [], color="C0", lw=1.5, label="robot")
    trails_m = []
    disks = []
    for mi in range(n_mov):
        (tm,) = ax.plot(
            [],
            [],
            color="k",
            lw=1.0,
            ls="--" if mi == 0 else ":",
            label="mover" if mi == 0 else f"mover {mi + 1}",
        )
        trails_m.append(tm)
        disk = plt.Circle((0, 0), radii[mi], color="0.25", alpha=0.45)
        ax.add_patch(disk)
        disks.append(disk)
    robot = plt.Circle((0, 0), R_ROBOT, color="C0", alpha=0.7)
    ax.add_patch(robot)
    status = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
    )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    def _frame(fi: int):
        k = int(idx[fi])
        trail_r.set_data(traj_xy[: k + 1, 0], traj_xy[: k + 1, 1])
        robot.center = (traj_xy[k, 0], traj_xy[k, 1])
        for mi in range(n_mov):
            trails_m[mi].set_data(mxy[: k + 1, mi, 0], mxy[: k + 1, mi, 1])
            disks[mi].center = (mxy[k, mi, 0], mxy[k, mi, 1])
        wait = bool(waiting[k]) if waiting is not None and k < len(waiting) else False
        status.set_text(f"t={k * DT:.1f}s  {'YIELD' if wait else 'track'}")
        return (trail_r, robot, status, *trails_m, *disks)

    anim = animation.FuncAnimation(fig, _frame, frames=len(idx), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def plot_ablation(
    grid: OccupancyGrid,
    path_xy: np.ndarray,
    runs: list[tuple[str, np.ndarray, str]],
    mover_xy: np.ndarray,
    mover_radius: float,
    out_path: Path,
    title: str = "Ablation",
    start_xy: tuple[float, float] | None = None,
) -> None:
    """Overlay several closed-loop trajectories on the map."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = grid.occ.shape
    ox, oy = grid.origin_xy
    res = grid.resolution
    extent = [ox, ox + w * res, oy, oy + h * res]

    pts = [path_xy, mover_xy] + [traj for _, traj, _ in runs]
    pts = np.vstack(pts)
    margin = 2.0
    xmin, ymin = pts.min(axis=0) - margin
    xmax, ymax = pts.max(axis=0) + margin

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=140)
    ax.imshow(
        grid.occ.astype(float),
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
        alpha=0.95,
    )
    ax.plot(path_xy[:, 0], path_xy[:, 1], color="0.45", lw=1.0, label="profiled ref")
    ax.plot(mover_xy[:, 0], mover_xy[:, 1], color="k", lw=1.2, ls="--", label="mover path")
    mid = _crossing_frame(runs[0][1], mover_xy) if runs else len(mover_xy) // 2
    ax.add_patch(
        plt.Circle(
            (mover_xy[mid, 0], mover_xy[mid, 1]),
            mover_radius,
            color="0.2",
            alpha=0.35,
            label="mover @ closest",
        )
    )
    for label, traj, color in runs:
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.8, label=label)
    sx, sy = start_xy if start_xy is not None else (path_xy[0, 0], path_xy[0, 1])
    ax.plot(sx, sy, "o", color="C2", ms=6, label="start")
    ax.plot(path_xy[-1, 0], path_xy[-1, 1], "x", color="C3", ms=8, mew=2, label="goal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
