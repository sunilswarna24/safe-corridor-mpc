"""Supervisory gate on two dynamic crossers: with vs without gate."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE, two_crossers
from src.simulate_dynamic import run_dynamic_tracking
from src.visualize import animate_yield, plot_dynamic, plot_errors


def main() -> None:
    out = ROOT / "results"
    movers = two_crossers()
    common = dict(**AWS_AISLE, reactive=True, movers=movers, seed=3)

    gated = run_dynamic_tracking(**common, use_supervisory=True)
    bare = run_dynamic_tracking(**common, use_supervisory=False)

    plot_dynamic(
        gated["grid"],
        gated["path_xy"],
        gated["traj_xy"],
        gated["mover_xy"],
        movers[0].radius,
        out / "m2_supervisory.png",
        corridor_snap=gated["corridor_snap"],
        traj_b=bare["traj_xy"],
        title="Two crossers — supervisory ON vs OFF",
        label_a="supervisory ON",
        label_b="supervisory OFF",
        movers_xy=gated["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )
    plot_errors(
        gated["t"],
        gated["err"],
        gated["solve_ms"],
        out / "m2_supervisory_errors.png",
        title="Supervisory ON — error & solve time",
    )
    animate_yield(
        gated["grid"],
        gated["path_xy"],
        gated["traj_xy"],
        gated["mover_xy"],
        movers[0].radius,
        out / "m2_supervisory.gif",
        waiting=gated["waiting"],
        title="Supervisory yield (ON), two movers",
        movers_xy=gated["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )

    print("M2 supervisory gate (two dynamic crossers)")
    print("--- supervisory ON ---")
    print(f"  mover hits:     {gated['mover_hits']}")
    print(f"  map collisions: {gated['collisions']}")
    print(f"  rms pos [m]:    {gated['err_rms']:.4f}")
    print(f"  fallbacks:      {gated['fallback_count']}")
    print(f"  wait ticks:     {gated['wait_count']}")
    print(f"  ticks:          {gated['ticks']} (path {gated['path_steps']})")
    print(f"  OSQP ok:        {gated['ok_frac']:.3f}")
    print("--- supervisory OFF ---")
    print(f"  mover hits:     {bare['mover_hits']}")
    print(f"  map collisions: {bare['collisions']}")
    print(f"  rms pos [m]:    {bare['err_rms']:.4f}")
    print(f"  wrote {out / 'm2_supervisory.png'}")
    print(f"  wrote {out / 'm2_supervisory.gif'}")


if __name__ == "__main__":
    main()
