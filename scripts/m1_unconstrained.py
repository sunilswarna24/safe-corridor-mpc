"""Milestone 1: unconstrained (no corridor) tracking MPC on a figure-eight."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.simulate import run_tracking
from src.visualize import plot_errors, plot_tracking


def main() -> None:
    out = ROOT / "results"
    stats = run_tracking(duration_s=20.0, path="eight")
    plot_tracking(
        stats["path_xy"],
        stats["traj_xy"],
        out / "m1_tracking.png",
        title="M1 tracking MPC — position + velocity (figure-eight)",
    )
    plot_errors(stats["t"], stats["err"], stats["solve_ms"], out / "m1_errors.png")

    print("M1 unconstrained tracking (pos + vel)")
    print(f"  rms pos error [m]:   {stats['err_rms']:.4f}")
    print(f"  max pos error [m]:   {stats['err_max']:.4f}")
    print(f"  rms vel error [m/s]: {stats['err_vel_rms']:.4f}")
    print(f"  solve med [ms]:      {stats['solve_ms_med']:.3f}")
    print(f"  solve p95 [ms]:      {stats['solve_ms_p95']:.3f}")
    print(f"  OSQP ok fraction:    {stats['ok_frac']:.3f}")
    print(f"  wrote {out / 'm1_tracking.png'}")
    print(f"  wrote {out / 'm1_errors.png'}")


if __name__ == "__main__":
    main()
