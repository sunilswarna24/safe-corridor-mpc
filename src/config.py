"""Shared sim / controller constants."""

DT = 0.05  # s — 20 Hz
N = 40  # steps — 2.0 s horizon

# Disk robot (center is the controlled point)
R_ROBOT = 0.30  # m
R_INFL = 0.35  # m — corridor margin (M2+)
PLAN_INFL = 0.20  # m — planner dilation (aisles stay open)

V_MAX = 1.0  # m/s per axis
A_MAX = 1.5  # m/s^2 per axis
V_CRUISE = 0.6  # m/s along Theta* paths

# Tracking weights: position, velocity, input effort
Q_POS = 20.0
Q_VEL = 5.0
R_U = 0.1

# Plant disturbance (Gaussian). 0 = ideal.
SLIP_STD = 0.05  # m/s velocity slip per tick
ACCEL_NOISE_STD = 0.10  # m/s^2 execution noise
