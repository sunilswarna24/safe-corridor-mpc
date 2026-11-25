"""Dynamic crossing obstacle: frozen vs reactive Ax<=b corridors."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE, shuttle_mover
from src.simulate_dynamic import run_dynamic_tracking
from src.visualize import plot_dynamic, plot_errors


def main() -> None:
    out = ROOT / "results"
    mover = shuttle_mover()
    common = dict(**AWS_AISLE, mover=mover, seed=3)

    reactive = run_dynamic_tracking(**common, reactive=True)
    frozen = run_dynamic_tracking(**common, reactive=False)

    plot_dynamic(
        reactive["grid"],
        reactive["path_xy"],
        reactive["traj_xy"],
        reactive["mover_xy"],
        mover.radius,
        out / "m2_dynamic.png",
        corridor_snap=reactive["corridor_snap"],
        traj_b=frozen["traj_xy"],
        title="AWS warehouse — reactive vs frozen corridors (crossing disk)",
    )
    plot_errors(
        reactive["t"],
        reactive["err"],
        reactive["solve_ms"],
        out / "m2_dynamic_errors.png",
        title="Reactive corridors — error & solve time",
    )

    print("M2 dynamic crossing (reactive vs frozen)")
    print("--- reactive ---")
    print(f"  mover hits:     {reactive['mover_hits']}")
    print(f"  map collisions: {reactive['collisions']}")
    print(f"  rms pos [m]:    {reactive['err_rms']:.4f}")
    print(f"  OSQP ok:        {reactive['ok_frac']:.3f}")
    print("--- frozen ---")
    print(f"  mover hits:     {frozen['mover_hits']}")
    print(f"  map collisions: {frozen['collisions']}")
    print(f"  rms pos [m]:    {frozen['err_rms']:.4f}")
    print(f"  wrote {out / 'm2_dynamic.png'}")


if __name__ == "__main__":
    main()
