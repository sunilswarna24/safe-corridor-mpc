"""Corridor MPC with vs without Gaussian slip / accel noise."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ACCEL_NOISE_STD, SLIP_STD
from src.scenarios import AISLE_BLOCKER, AWS_AISLE
from src.simulate_map import run_map_tracking
from src.visualize import plot_errors, plot_map_corridors


def main() -> None:
    out = ROOT / "results"
    blocker = AISLE_BLOCKER
    common = dict(**AWS_AISLE, use_corridors=True, block_obstacle=blocker, seed=7)

    clean = run_map_tracking(**common, slip_std=0.0, accel_noise_std=0.0)
    noisy = run_map_tracking(
        **common, slip_std=SLIP_STD, accel_noise_std=ACCEL_NOISE_STD
    )

    plot_map_corridors(
        noisy["grid"],
        noisy["path_xy"],
        noisy["traj_xy"],
        noisy["corridor_bounds"],
        out / "m2_noise_slip.png",
        traj_b=clean["traj_xy"],
        block_obstacle=blocker,
        title="Corridor MPC — clean vs Gaussian slip/noise",
        label_a=f"noisy (slip={SLIP_STD}, aσ={ACCEL_NOISE_STD})",
        label_b="clean",
    )
    plot_errors(
        noisy["t"],
        noisy["err"],
        noisy["solve_ms"],
        out / "m2_noise_errors.png",
        title="Noisy corridor MPC — tracking error & solve time",
    )

    print("M2 Gaussian slip / accel noise (corridor MPC)")
    print(f"  slip_std [m/s]:         {SLIP_STD}")
    print(f"  accel_noise_std [m/s2]: {ACCEL_NOISE_STD}")
    print("--- clean ---")
    print(f"  collisions:             {clean['collisions']}")
    print(f"  rms pos [m]:            {clean['err_rms']:.4f}")
    print("--- noisy ---")
    print(f"  collisions:             {noisy['collisions']}")
    print(f"  rms pos [m]:            {noisy['err_rms']:.4f}")
    print(f"  OSQP ok fraction:       {noisy['ok_frac']:.3f}")
    print(f"  wrote {out / 'm2_noise_slip.png'}")


if __name__ == "__main__":
    main()
