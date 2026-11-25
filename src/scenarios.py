"""Shared AWS-aisle demo knobs (keep scripts thin)."""

from __future__ import annotations

from .movers import CrossingDisk

# Nominal path on maps/aws_warehouse
AWS_AISLE = dict(
    map_yaml="aws_warehouse/map.yaml",
    start_xy=(-4.0, -8.0),
    goal_xy=(4.0, 6.0),
    v_cruise=0.55,
    plan_stride=2,
    corridor_max_half=1.0,
)

# Static blocker for corridor-vs-open ablation (m2_corridors)
AISLE_BLOCKER = (0.0, -1.0, 0.55)


def shuttle_mover() -> CrossingDisk:
    """Along-aisle shuttle — stresses reactive vs frozen corridors."""
    return CrossingDisk(
        p0=(-5.2, -0.3),
        p1=(-2.2, -1.7),
        radius=0.40,
        speed=0.50,
        phase=12.0,
    )


def two_crossers() -> list[CrossingDisk]:
    """Two one-shot E–W crossers timed to meet the northbound robot."""
    return [
        CrossingDisk(
            p0=(-5.5, -2.55),
            p1=(1.5, -2.55),
            radius=0.50,
            speed=0.70,
            phase=-6.5,
            one_shot=True,
        ),
        CrossingDisk(
            p0=(-6.0, 1.80),
            p1=(2.5, 1.80),
            radius=0.45,
            speed=0.65,
            phase=-18.0,
            one_shot=True,
        ),
    ]
