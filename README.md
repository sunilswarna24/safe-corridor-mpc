# Safe-Corridor MPC

Warehouse-style AMR loop in Python: global path → convex \(Ax \le b\) corridors → supervisory gate → constrained tracking MPC. Neural warm-start / C++ come later on the same loop.

**Honesty:** 2D double-integrator disk (+ slip/accel noise). Reactive corridors and the gate see mover **poses now**, not predicted trajectories. Not unicycle/bicycle, not hardware, not Nav2.

| Phase | Status | Notes |
|-------|--------|-------|
| 1 Math core | Done | AWS map, Theta*, corridors, noise, dynamic disks |
| 2 Supervisory | Done | Yield / hard brake / slew; two crossers; GIF |
| 2 Faults | Done | Bad \(X_0\); ablation A–C (corridors / gate / both) |
| 3 NN + C++/ROS 2 | Planned | ONNX warm-start; thin port |
| 4 Optional | Later | Bicycle, Hybrid A*, MPPI, Nav2 plugin |

## Layout

```
src/           plant, MPC, corridors, planners, gate, sims
scripts/       runnable demos (write results/)
maps/          AWS warehouse (+ Nav2 warehouse/depot extras)
results/       plots / GIF from your runs
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python scripts/m1_unconstrained.py
python scripts/m1b_astar_reference.py
python scripts/m2_corridors.py
python scripts/m2_noise_slip.py
python scripts/m2_dynamic.py
python scripts/m2_supervisory.py
python scripts/m2_faults.py
```

## Locked numbers

- `dt = 0.05 s` (20 Hz), horizon `N = 40` (2 s)
- Disk `r = 0.30 m`; `|v| ≤ 1.0 m/s`, `|a| ≤ 1.5 m/s²`
- MPC tracks position + velocity (+ input effort); M2 adds AABB corridors

Primary map: AWS RoboMaker small warehouse (`maps/aws_warehouse/`). Attribution in `maps/ATTRIBUTION.md`.
