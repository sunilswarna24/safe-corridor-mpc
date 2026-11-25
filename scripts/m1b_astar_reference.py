"""AWS warehouse + Theta* + speed profile + unconstrained MPC track."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE
from src.simulate_map import run_map_tracking
from src.visualize import plot_errors, plot_map_tracking


def main() -> None:
    out = ROOT / "results"
    stats = run_map_tracking(**AWS_AISLE, v_cruise=0.6, use_corridors=False)
    plot_map_tracking(
        stats["grid"],
        stats["waypoints"],
        stats["path_xy"],
        stats["traj_xy"],
        out / "m1b_map_tracking.png",
        title="AWS small warehouse — Theta* + profiler + MPC",
    )
    plot_errors(
        stats["t"],
        stats["err"],
        stats["solve_ms"],
        out / "m1b_errors.png",
        title="M1b AWS warehouse tracking error & solve time",
    )

    print("M1b AWS warehouse + Theta* + profiler + MPC")
    print(f"  map:                 {stats['map_yaml']}")
    print(f"  start_xy [m]:        ({float(stats['start_xy'][0]):.2f}, {float(stats['start_xy'][1]):.2f})")
    print(f"  goal_xy [m]:         ({float(stats['goal_xy'][0]):.2f}, {float(stats['goal_xy'][1]):.2f})")
    print(f"  path cells/verts:    {stats['n_cells']} / {stats['n_waypoints']}")
    print(f"  plan time [ms]:      {stats['plan_ms']:.1f}")
    print(f"  duration [s]:        {stats['t'][-1]:.1f}")
    print(f"  rms pos error [m]:   {stats['err_rms']:.4f}")
    print(f"  max pos error [m]:   {stats['err_max']:.4f}")
    print(f"  rms vel error [m/s]: {stats['err_vel_rms']:.4f}")
    print(f"  solve med [ms]:      {stats['solve_ms_med']:.3f}")
    print(f"  solve p95 [ms]:      {stats['solve_ms_p95']:.3f}")
    print(f"  OSQP ok fraction:    {stats['ok_frac']:.3f}")
    print(f"  wrote {out / 'm1b_map_tracking.png'}")


if __name__ == "__main__":
    main()
