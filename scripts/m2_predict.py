"""Myopic vs short-horizon predicted movers (unicycle)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE, two_crossers
from src.simulate_dynamic import run_dynamic_tracking
from src.visualize import plot_ablation, plot_dynamic, plot_errors


def _row(name: str, r: dict) -> str:
    return (
        f"{name:22s}  hits={r['mover_hits']:4d}  map={r['collisions']:3d}  "
        f"wait={r['wait_count']:4d}  sealed={r['sealed_ticks']:3d}  "
        f"rms={r['err_rms']:.3f}  ok={r['ok_frac']:.3f}  ticks={r['ticks']}"
    )


def main() -> None:
    out = ROOT / "results"
    movers = two_crossers()
    common = dict(
        **AWS_AISLE,
        movers=movers,
        seed=3,
        plant="unicycle",
        reactive=True,
        use_corridors=True,
    )

    print("Mover prediction — myopic vs predicted (unicycle)")
    my_ng = run_dynamic_tracking(**common, use_supervisory=False, predict_movers=False)
    print(_row("myopic, no gate", my_ng))
    pred_ng = run_dynamic_tracking(**common, use_supervisory=False, predict_movers=True)
    print(_row("predicted, no gate", pred_ng))
    my_g = run_dynamic_tracking(**common, use_supervisory=True, predict_movers=False)
    print(_row("myopic + gate", my_g))
    pred_g = run_dynamic_tracking(**common, use_supervisory=True, predict_movers=True)
    print(_row("predicted + gate", pred_g))

    plot_dynamic(
        pred_g["grid"],
        pred_g["path_xy"],
        pred_g["traj_xy"],
        pred_g["mover_xy"],
        movers[0].radius,
        out / "m2_predict.png",
        corridor_snap=pred_g["corridor_snap"],
        traj_b=my_g["traj_xy"],
        title="Predicted vs myopic (unicycle + gate)",
        label_a="predicted + gate",
        label_b="myopic + gate",
        movers_xy=pred_g["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )
    plot_ablation(
        pred_g["grid"],
        pred_g["path_xy"],
        [
            ("myopic, no gate", my_ng["traj_xy"], "C3"),
            ("predicted, no gate", pred_ng["traj_xy"], "C1"),
            ("myopic + gate", my_g["traj_xy"], "C0"),
            ("predicted + gate", pred_g["traj_xy"], "C2"),
        ],
        pred_g["mover_xy"],
        movers[0].radius,
        out / "m2_predict_ablation.png",
        title="Myopic vs predicted (unicycle)",
    )
    plot_errors(
        pred_g["t"],
        pred_g["err"],
        pred_g["solve_ms"],
        out / "m2_predict_errors.png",
        title="Predicted + gate — error & solve time",
    )

    table = out / "m2_predict.txt"
    table.write_text(
        "\n".join(
            [
                _row("myopic, no gate", my_ng),
                _row("predicted, no gate", pred_ng),
                _row("myopic + gate", my_g),
                _row("predicted + gate", pred_g),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out / 'm2_predict.png'}")
    print(f"wrote {out / 'm2_predict_ablation.png'}")
    print(f"wrote {table}")


if __name__ == "__main__":
    main()
