"""Faults + ablations A–C with two one-shot aisle crossers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE, two_crossers
from src.simulate_dynamic import run_dynamic_tracking
from src.visualize import plot_ablation, plot_dynamic


def _row(name: str, r: dict) -> str:
    return (
        f"{name:22s}  hits={r['mover_hits']:4d}  map={r['collisions']:3d}  "
        f"wait={r['wait_count']:4d}  fb={r['fallback_count']:4d}  "
        f"rms={r['err_rms']:.3f}  ok={r['ok_frac']:.3f}  ticks={r['ticks']}"
    )


def main() -> None:
    out = ROOT / "results"
    movers = two_crossers()
    base = dict(**AWS_AISLE, movers=movers, seed=3)

    print("Ablation A–C (two one-shot crossers)")
    a = run_dynamic_tracking(**base, reactive=True, use_corridors=True, use_supervisory=False)
    print(_row("A corridors", a))
    b = run_dynamic_tracking(**base, reactive=False, use_corridors=False, use_supervisory=True)
    print(_row("B gate only", b))
    c = run_dynamic_tracking(**base, reactive=True, use_corridors=True, use_supervisory=True)
    print(_row("C corridors+gate", c))

    plot_ablation(
        c["grid"],
        c["path_xy"],
        [
            ("A corridors", a["traj_xy"], "C3"),
            ("B gate only", b["traj_xy"], "C1"),
            ("C corridors+gate", c["traj_xy"], "C0"),
        ],
        c["mover_xy"],
        movers[0].radius,
        out / "m2_ablation.png",
        title="Ablation A–C — two aisle crossers",
    )

    print("\nBad X0 (offset from path start)")
    x0 = (0.40, -0.20)
    bad_off = run_dynamic_tracking(
        **base, reactive=True, use_corridors=True, use_supervisory=False, x0_offset=x0
    )
    print(_row("X0 gate OFF", bad_off))
    print(f"  z0=({bad_off['z0_xy'][0]:.2f}, {bad_off['z0_xy'][1]:.2f})")
    bad_on = run_dynamic_tracking(
        **base, reactive=True, use_corridors=True, use_supervisory=True, x0_offset=x0
    )
    print(_row("X0 gate ON", bad_on))
    print(f"  z0=({bad_on['z0_xy'][0]:.2f}, {bad_on['z0_xy'][1]:.2f})")

    plot_dynamic(
        bad_on["grid"],
        bad_on["path_xy"],
        bad_on["traj_xy"],
        bad_on["mover_xy"],
        movers[0].radius,
        out / "m2_fault_x0.png",
        corridor_snap=bad_on["corridor_snap"],
        traj_b=bad_off["traj_xy"],
        title="Bad X0 — supervisory ON vs OFF",
        label_a="gate ON",
        label_b="gate OFF",
        movers_xy=bad_on["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )

    print(f"\nwrote {out / 'm2_ablation.png'}")
    print(f"wrote {out / 'm2_fault_x0.png'}")


if __name__ == "__main__":
    main()
