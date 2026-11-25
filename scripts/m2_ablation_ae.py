"""Ablation A–E on the AWS aisle with two crossers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scenarios import AWS_AISLE, two_crossers
from src.simulate_dynamic import run_dynamic_tracking
from src.visualize import animate_yield, plot_ablation


def _row(name: str, r: dict) -> str:
    return (
        f"{name:28s}  hits={r['mover_hits']:4d}  map={r['collisions']:3d}  "
        f"wait={r['wait_count']:4d}  sealed={r['sealed_ticks']:3d}  "
        f"fb={r['fallback_count']:4d}  rms={r['err_rms']:.3f}  "
        f"ok={r['ok_frac']:.3f}  ticks={r['ticks']}"
    )


def main() -> None:
    out = ROOT / "results"
    movers = two_crossers()
    base = dict(**AWS_AISLE, movers=movers, seed=3)

    print("Ablation A–E (two one-shot crossers)")
    # A: myopic corridors, no gate (DI)
    a = run_dynamic_tracking(
        **base,
        reactive=True,
        plant="di",
        use_corridors=True,
        use_supervisory=False,
        predict_movers=False,
    )
    print(_row("A corridors myopic (DI)", a))

    # B: gate only
    b = run_dynamic_tracking(
        **base,
        plant="di",
        use_corridors=False,
        reactive=False,
        use_supervisory=True,
        predict_movers=False,
    )
    print(_row("B gate only (DI)", b))

    # C: corridors + gate myopic
    c = run_dynamic_tracking(
        **base,
        reactive=True,
        plant="di",
        use_corridors=True,
        use_supervisory=True,
        predict_movers=False,
    )
    print(_row("C corridors+gate myopic (DI)", c))

    # D: predicted movers + gate (unicycle)
    d = run_dynamic_tracking(
        **base,
        reactive=True,
        plant="unicycle",
        use_corridors=True,
        use_supervisory=True,
        predict_movers=True,
    )
    print(_row("D predicted+gate (uni)", d))

    # E: unicycle vs DI same scenario (myopic + gate)
    e_uni = run_dynamic_tracking(
        **base,
        reactive=True,
        plant="unicycle",
        use_corridors=True,
        use_supervisory=True,
        predict_movers=False,
    )
    e_di = c  # same as C
    print(_row("E unicycle myopic+gate", e_uni))
    print(_row("E DI myopic+gate (=C)", e_di))

    plot_ablation(
        d["grid"],
        d["path_xy"],
        [
            ("A corridors", a["traj_xy"], "C3"),
            ("B gate only", b["traj_xy"], "C1"),
            ("C myopic+gate", c["traj_xy"], "C0"),
            ("D predicted+gate", d["traj_xy"], "C2"),
            ("E unicycle", e_uni["traj_xy"], "C4"),
        ],
        d["mover_xy"],
        movers[0].radius,
        out / "m2_ablation_ae.png",
        title="Ablation A–E — warehouse aisle, two crossers",
    )

    animate_yield(
        d["grid"],
        d["path_xy"],
        d["traj_xy"],
        d["mover_xy"],
        movers[0].radius,
        out / "m2_supervisory.gif",
        waiting=d["waiting"],
        title="Unicycle + predicted movers + gate",
        movers_xy=d["movers_xy"],
        mover_radii=[m.radius for m in movers],
    )

    table_path = out / "m2_ablation_ae.txt"
    lines = [
        "ID  variant                              hits  map  wait  sealed  fb   rms    ok",
        f"A   corridors myopic, no gate (DI)       {a['mover_hits']:4d}  {a['collisions']:3d}  {a['wait_count']:4d}  {a['sealed_ticks']:6d}  {a['fallback_count']:4d}  {a['err_rms']:.3f}  {a['ok_frac']:.3f}",
        f"B   gate only (DI)                       {b['mover_hits']:4d}  {b['collisions']:3d}  {b['wait_count']:4d}  {b['sealed_ticks']:6d}  {b['fallback_count']:4d}  {b['err_rms']:.3f}  {b['ok_frac']:.3f}",
        f"C   corridors + gate myopic (DI)         {c['mover_hits']:4d}  {c['collisions']:3d}  {c['wait_count']:4d}  {c['sealed_ticks']:6d}  {c['fallback_count']:4d}  {c['err_rms']:.3f}  {c['ok_frac']:.3f}",
        f"D   predicted movers + gate (unicycle)   {d['mover_hits']:4d}  {d['collisions']:3d}  {d['wait_count']:4d}  {d['sealed_ticks']:6d}  {d['fallback_count']:4d}  {d['err_rms']:.3f}  {d['ok_frac']:.3f}",
        f"E   unicycle vs DI (myopic+gate)         uni hits={e_uni['mover_hits']} rms={e_uni['err_rms']:.3f} | DI hits={e_di['mover_hits']} rms={e_di['err_rms']:.3f}",
    ]
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {out / 'm2_ablation_ae.png'}")
    print(f"wrote {out / 'm2_supervisory.gif'}")
    print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
