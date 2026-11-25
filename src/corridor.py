"""Local axis-aligned safe corridors along a reference path."""

from __future__ import annotations

import numpy as np

from .grid import OccupancyGrid


def _blocked(grid: OccupancyGrid, x: float, y: float) -> bool:
    r, c = grid.xy_to_cell(x, y)
    return not grid.free(r, c)


def local_aabb(
    grid: OccupancyGrid,
    x: float,
    y: float,
    max_half: float = 1.2,
    clearance: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Grow an AABB from (x,y) until hitting occupancy (or max_half).
    Returns (xmin, xmax, ymin, ymax) for the disk center.
    """
    step = 0.5 * grid.resolution
    if step <= 0:
        step = 0.05

    def expand(dx: float, dy: float) -> float:
        dist = 0.0
        while dist + step <= max_half:
            nx = x + (dist + step) * dx
            ny = y + (dist + step) * dy
            if _blocked(grid, nx, ny):
                break
            dist += step
        return max(0.0, dist - clearance)

    left = expand(-1.0, 0.0)
    right = expand(1.0, 0.0)
    down = expand(0.0, -1.0)
    up = expand(0.0, 1.0)
    # ensure nonzero width for numerical safety
    eps = 0.5 * grid.resolution
    return (x - max(left, eps), x + max(right, eps), y - max(down, eps), y + max(up, eps))


def _snap_free(grid: OccupancyGrid, x: float, y: float) -> tuple[float, float]:
    r, c = grid.xy_to_cell(x, y)
    if grid.free(r, c):
        return x, y
    for rad in range(1, 80):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                rr, cc = r + dr, c + dc
                if grid.free(rr, cc):
                    xy = grid.cell_to_xy(rr, cc)
                    return float(xy[0]), float(xy[1])
    return x, y


def corridors_along_path(
    grid: OccupancyGrid,
    path_xy: np.ndarray,
    max_half: float = 1.2,
    clearance: float = 0.0,
) -> np.ndarray:
    """
    path_xy: (T, 2) -> bounds (T, 4) as [xmin, xmax, ymin, ymax]
    Seeds blocked by obstacles are snapped to the nearest free cell first.
    """
    path_xy = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    out = np.empty((path_xy.shape[0], 4), dtype=float)
    for i, (x, y) in enumerate(path_xy):
        sx, sy = _snap_free(grid, float(x), float(y))
        out[i] = local_aabb(grid, sx, sy, max_half=max_half, clearance=clearance)
    return out


def horizon_corridors(bounds_full: np.ndarray, t0_index: int, N: int) -> np.ndarray:
    """Slice N corridor boxes starting at t0_index (pad with last)."""
    T = bounds_full.shape[0]
    out = np.empty((N, 4), dtype=float)
    for k in range(N):
        out[k] = bounds_full[min(t0_index + k, T - 1)]
    return out


def paint_disk_obstacle(
    grid: OccupancyGrid,
    center_xy: tuple[float, float],
    radius_m: float,
) -> OccupancyGrid:
    """Return a copy of grid with a disk marked occupied."""
    occ = grid.occ.copy()
    cx, cy = center_xy
    r_cells = int(np.ceil(radius_m / grid.resolution)) + 1
    cr, cc = grid.xy_to_cell(cx, cy)
    for dr in range(-r_cells, r_cells + 1):
        for dc in range(-r_cells, r_cells + 1):
            rr, cc2 = cr + dr, cc + dc
            if not grid.in_bounds(rr, cc2):
                continue
            xy = grid.cell_to_xy(rr, cc2)
            if (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2 <= radius_m**2:
                occ[rr, cc2] = True
    return OccupancyGrid(occ=occ, resolution=grid.resolution, origin_xy=grid.origin_xy)


def paint_movers(
    grid: OccupancyGrid,
    movers: list,
    t: float,
    extra_radius: float = 0.0,
) -> OccupancyGrid:
    """Paint mover disks at time t (current poses only — no foresight)."""
    g = grid
    for m in movers:
        g = paint_disk_obstacle(g, tuple(m.position(t)), float(m.radius) + extra_radius)
    return g
