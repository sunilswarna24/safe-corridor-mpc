# Safe-Corridor MPC

Warehouse AMR tracking: Theta* path → \(Ax \le b\) corridors → supervisory gate → constrained MPC. Unicycle plant; short-horizon mover prediction into stage corridors; optional NN QP warm-start. Thin C++/ROS 2 port of the same loop. Sim only — corridors on position, scripted mover forecasts, reactive gate. DI kept for the plant compare.

| Phase | Status | Notes |
|-------|--------|-------|
| 1 Math core | Done | AWS map, Theta*, corridors, noise, dynamic disks |
| 2 Supervisory | Done | Yield / hard brake / slew; two crossers |
| 2 Unicycle + predict | Done | LTV MPC; predicted movers; ablation A–E |
| 3 NN warm-start | Done | Numpy MLP → ONNX; `results/m3_warmstart.txt` |
| 4 C++ / ROS 2 | Done | `cpp/` OSQP library + `demo_line`; `ros2/safe_corridor_mpc` node |

## Layout

```
src/                 Python plant, MPC, corridors, gate, warm-start, sims
scripts/             runnable demos (write results/)
cpp/                 unicycle MPC + gate (OSQP); demo_line
ros2/safe_corridor_mpc/   20 Hz cmd_vel node (odom + path + movers)
maps/                AWS warehouse (+ Nav2 warehouse/depot extras)
results/             plots / GIF / tables from your runs
```

## Setup (Python)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run (Python)

```powershell
python scripts/m1_unconstrained.py
python scripts/m1b_astar_reference.py
python scripts/m2_corridors.py
python scripts/m2_noise_slip.py
python scripts/m2_dynamic.py
python scripts/m2_supervisory.py
python scripts/m2_faults.py
python scripts/m2_plant_compare.py
python scripts/m2_predict.py
python scripts/m2_ablation_ae.py
python scripts/m3_warmstart.py
```

## Build / run (C++)

Needs CMake ≥ 3.16, a C++17 compiler, and network once (FetchContent pulls Eigen + OSQP 0.6.3).

```powershell
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
.\cpp\build\demo_line.exe
```

## ROS 2

Package `ros2/safe_corridor_mpc` links the C++ library. Topics: `odom`, `path`, `movers` (PoseArray) → `cmd_vel` at 20 Hz.

```bash
# from a sourced ROS 2 workspace that can see this package
colcon build --packages-select safe_corridor_mpc
ros2 run safe_corridor_mpc mpc_node
```

## Locked numbers

- `dt = 0.05 s` (20 Hz), horizon `N = 40` (2 s)
- Disk `r = 0.30 m`; unicycle `|v| ≤ 1.0`, `|a| ≤ 1.5`, `|\omega| ≤ 1.2`
- Corridors: AABB on \((x,y)\); heading in dynamics

Primary map: AWS RoboMaker small warehouse (`maps/aws_warehouse/`). Attribution in `maps/ATTRIBUTION.md`.
