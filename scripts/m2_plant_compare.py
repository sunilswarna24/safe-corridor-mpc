"""Unicycle vs double-integrator on the same AWS aisle + two crossers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE, two_crossers
from src.simulate_dynamic import run_dynamic_tracking
from src.visualize import animate_yield, plot_dynamic, plot_errors


def _row(name: str, r: dict) -> str:
    return (
        f"{name:14s}  hits={r['mover_hits']:4d}  map={r['collisions']:3d}  "
        f"wait={r['wait_count']:4d}  sealed={r['sealed_ticks']:3d}  "
        f"rms={r['err_rms']:.3f}  ok={r['ok_frac']:.3f}  "
        f"p95={r['solve_ms_p95']:.2f}ms  ticks={r['ticks']}"
    )


def main() -> None:
    out = ROOT / "results"
    movers = two_crossers()
    common = dict(
        **AWS_AISLE,
        movers=movers,
        seed=3,
        reactive=True,
        use_corridors=True,
        use_supervisory=True,
        predict_movers=False,
    )

    print("Plant compare — unicycle vs DI (myopic corridors + gate)")
    uni = run_dynamic_tracking(**common, plant="unicycle")
    print(_row("unicycle", uni))
    di = run_dynamic_tracking(**common, plant="di")
    print(_row("DI", di))

    plot_dynamic(
        uni["grid"],
        uni["path_xy"],
        uni["traj_xy"],
        uni["mover_xy"],
        movers[0].radius,
        out / "m2_plant_compare.png",
        corridor_snap=uni["corridor_snap"],
        traj_b=di["traj_xy"],
        title="Unicycle vs DI — same aisle, two crossers, gate ON",
        label_a="unicycle",
        label_b="DI",
        movers_xy=uni["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )
    plot_errors(
        uni["t"],
        uni["err"],
        uni["solve_ms"],
        out / "m2_plant_compare_errors.png",
        title="Unicycle — error & solve time",
    )
    animate_yield(
        uni["grid"],
        uni["path_xy"],
        uni["traj_xy"],
        uni["mover_xy"],
        movers[0].radius,
        out / "m2_unicycle.gif",
        waiting=uni["waiting"],
        title="Unicycle AMR + gate, two movers",
        movers_xy=uni["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )

    print(f"wrote {out / 'm2_plant_compare.png'}")
    print(f"wrote {out / 'm2_unicycle.gif'}")


if __name__ == "__main__":
    main()
