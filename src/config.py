"""Shared sim / controller constants."""

DT = 0.05  # s — 20 Hz
N = 40  # steps — 2.0 s horizon

# Disk robot (center is the controlled point)
R_ROBOT = 0.30  # m
R_INFL = 0.35  # m — planning / corridor margin

GRID_RES = 0.25  # m/cell (MovingAI maps)

V_MAX = 1.0  # m/s per axis
A_MAX = 1.5  # m/s^2 per axis

# Tracking weights (position, then input)
Q_POS = 20.0
R_U = 0.1
