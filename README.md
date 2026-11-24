# Safe-Corridor MPC

Warehouse-style AMR stack: global path → convex \(Ax \le b\) corridors → supervisory gate → constrained tracking MPC. Neural warm-start comes later on the same loop.

**Honesty:** Phase 1 is a 2D Python sim (double-integrator disk). Not hardware, not Nav2, not Hybrid A* yet.

| Phase | Status | Deliverable |
|-------|--------|-------------|
| **1 Math core** | **In progress** | OSQP tracking MPC + corridors on MovingAI warehouse crops |
| 2 Supervisory + faults | Planned | Bad \(X_0\) / dynamic obstacle fallbacks + ablation A–C |
| 3 NN warm-start + C++/ROS 2 | Planned | ONNX CPU inference + ablation D |
| 4 Optional | Later | Hybrid A* / bicycle / MPPI / Nav2 plugin |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
python scripts/m1_unconstrained.py
```

Plots land in `results/`.

## Locked numbers

- `dt = 0.05 s` (20 Hz), horizon `N = 40` (2 s)
- Disk robot `r = 0.30 m`, inflation `0.35 m`, grid `0.25 m/cell`
- `|v| ≤ 1.0 m/s`, `|a| ≤ 1.5 m/s²`

Maps: MovingAI MAPF warehouse grids (static) + scripted movers for dynamics.
