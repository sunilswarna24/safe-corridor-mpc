"""AABB corridors + constrained MPC on AWS warehouse."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AISLE_BLOCKER, AWS_AISLE
from src.simulate_map import run_map_tracking
from src.visualize import plot_errors, plot_map_corridors


def main() -> None:
    out = ROOT / "results"
    blocker = AISLE_BLOCKER
    common = dict(**AWS_AISLE, block_obstacle=blocker)

    constrained = run_map_tracking(**common, use_corridors=True)
    open_loop = run_map_tracking(**common, use_corridors=False)

    plot_map_corridors(
        constrained["grid"],
        constrained["path_xy"],
        constrained["traj_xy"],
        constrained["corridor_bounds"],
        out / "m2_corridors.png",
        traj_b=open_loop["traj_xy"],
        block_obstacle=blocker,
        title="AWS warehouse — corridor MPC vs unconstrained",
    )
    plot_errors(
        constrained["t"],
        constrained["err"],
        constrained["solve_ms"],
        out / "m2_errors.png",
        title="M2 corridor MPC tracking error & solve time",
    )

    print("M2 AWS warehouse corridors")
    print(f"  map:                    {constrained['map_yaml']}")
    print(f"  blocker:                {blocker}")
    print(f"  path verts:             {constrained['n_waypoints']}")
    print(f"  duration [s]:           {constrained['t'][-1]:.1f}")
    print("--- constrained (Ax<=b) ---")
    print(f"  collisions (ticks):     {constrained['collisions']}")
    print(f"  rms pos error [m]:      {constrained['err_rms']:.4f}")
    print(f"  solve med/p95 [ms]:     {constrained['solve_ms_med']:.3f} / {constrained['solve_ms_p95']:.3f}")
    print(f"  OSQP ok fraction:       {constrained['ok_frac']:.3f}")
    print("--- unconstrained ---")
    print(f"  collisions (ticks):     {open_loop['collisions']}")
    print(f"  rms pos error [m]:      {open_loop['err_rms']:.4f}")
    print(f"  wrote {out / 'm2_corridors.png'}")


if __name__ == "__main__":
    main()
