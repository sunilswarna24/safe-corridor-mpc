"""Occupancy grid + inflation / downsample helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass
class OccupancyGrid:
    """row-major occupancy: True = blocked."""

    occ: np.ndarray  # (H, W) bool
    resolution: float
    origin_xy: tuple[float, float] = (0.0, 0.0)

    @property
    def height(self) -> int:
        return int(self.occ.shape[0])

    @property
    def width(self) -> int:
        return int(self.occ.shape[1])

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def free(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and not self.occ[r, c]

    def cell_to_xy(self, r: int, c: int) -> np.ndarray:
        x = self.origin_xy[0] + (c + 0.5) * self.resolution
        y = self.origin_xy[1] + (r + 0.5) * self.resolution
        return np.array([x, y], dtype=float)

    def xy_to_cell(self, x: float, y: float) -> tuple[int, int]:
        c = int((x - self.origin_xy[0]) / self.resolution)
        r = int((y - self.origin_xy[1]) / self.resolution)
        return r, c


def inflate(grid: OccupancyGrid, radius_m: float) -> OccupancyGrid:
    """Binary dilate obstacles by ceil(radius/resolution) cells."""
    cells = int(np.ceil(radius_m / grid.resolution))
    if cells <= 0:
        return OccupancyGrid(
            occ=grid.occ.copy(), resolution=grid.resolution, origin_xy=grid.origin_xy
        )
    yy, xx = np.ogrid[-cells : cells + 1, -cells : cells + 1]
    footprint = xx * xx + yy * yy <= cells * cells
    inflated = ndimage.binary_dilation(grid.occ, structure=footprint)
    return OccupancyGrid(
        occ=inflated, resolution=grid.resolution, origin_xy=grid.origin_xy
    )


def downsample(grid: OccupancyGrid, stride: int) -> OccupancyGrid:
    """Max-pool occupancy (blocked if any cell in the block is blocked)."""
    if stride <= 1:
        return grid
    h, w = grid.occ.shape
    hs, ws = h // stride, w // stride
    occ = grid.occ[: hs * stride, : ws * stride]
    occ = occ.reshape(hs, stride, ws, stride).max(axis=(1, 3))
    return OccupancyGrid(
        occ=occ,
        resolution=grid.resolution * stride,
        origin_xy=grid.origin_xy,
    )
