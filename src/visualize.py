"""Plot helpers for Milestone 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    fig.suptitle("M1 tracking error & solve time")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
