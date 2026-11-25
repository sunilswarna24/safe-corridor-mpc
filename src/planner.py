"""Cost-aware A* + Theta* on inflated occupancy grids."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from .grid import OccupancyGrid

_NEIGH8 = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (1, 1, math.sqrt(2)),
)


def _h(r: int, c: int, gr: int, gc: int) -> float:
    return math.hypot(r - gr, c - gc)


def _line_of_sight(grid: OccupancyGrid, r0: int, c0: int, r1: int, c1: int) -> bool:
    """Bresenham LOS; all cells on segment must be free."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if not grid.free(r, c):
            return False
        if r == r1 and c == c1:
            return True
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def astar(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """8-connected A*. Returns cell path inclusive of start/goal, or None."""
    sr, sc = start
    gr, gc = goal
    if not grid.free(sr, sc) or not grid.free(gr, gc):
        return None

    open_heap: list[tuple[float, int, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (_h(sr, sc, gr, gc), counter, sr, sc))
    came: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    g_score = {(sr, sc): 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, r, c = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        if (r, c) == (gr, gc):
            return _reconstruct(came, (gr, gc))
        closed.add((r, c))
        for dr, dc, step in _NEIGH8:
            nr, nc = r + dr, c + dc
            if not grid.free(nr, nc):
                continue
            # prevent diagonal corner cutting
            if dr != 0 and dc != 0:
                if not grid.free(r + dr, c) or not grid.free(r, c + dc):
                    continue
            ng = g_score[(r, c)] + step
            key = (nr, nc)
            if ng + 1e-12 < g_score.get(key, math.inf):
                g_score[key] = ng
                came[key] = (r, c)
                counter += 1
                heapq.heappush(open_heap, (ng + _h(nr, nc, gr, gc), counter, nr, nc))
    return None


def theta_star(
    grid: OccupancyGrid,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """
    Basic Theta*: any-angle parent updates with LOS on the inflated grid.
    Falls back to A* path if search fails.
    """
    sr, sc = start
    gr, gc = goal
    if not grid.free(sr, sc) or not grid.free(gr, gc):
        return None

    open_heap: list[tuple[float, int, int, int]] = []
    counter = 0
    heapq.heappush(open_heap, (_h(sr, sc, gr, gc), counter, sr, sc))
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}
    g_score = {(sr, sc): 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, r, c = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        if (r, c) == (gr, gc):
            return _reconstruct(parent, (gr, gc))
        closed.add((r, c))
        p = parent[(r, c)]
        for dr, dc, _ in _NEIGH8:
            nr, nc = r + dr, c + dc
            if not grid.free(nr, nc):
                continue
            if dr != 0 and dc != 0:
                if not grid.free(r + dr, c) or not grid.free(r, c + dc):
                    continue
            # Theta* parent update
            if p is not None and _line_of_sight(grid, p[0], p[1], nr, nc):
                ng = g_score[p] + math.hypot(nr - p[0], nc - p[1])
                new_parent = p
            else:
                ng = g_score[(r, c)] + math.hypot(nr - r, nc - c)
                new_parent = (r, c)
            key = (nr, nc)
            if ng + 1e-12 < g_score.get(key, math.inf):
                g_score[key] = ng
                parent[key] = new_parent
                counter += 1
                heapq.heappush(open_heap, (ng + _h(nr, nc, gr, gc), counter, nr, nc))

    return astar(grid, start, goal)


def cells_to_xy(grid: OccupancyGrid, cells: Iterable[tuple[int, int]]) -> np.ndarray:
    return np.array([grid.cell_to_xy(r, c) for r, c in cells], dtype=float)


def _reconstruct(
    came: dict[tuple[int, int], tuple[int, int] | None],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [goal]
    cur = goal
    while came[cur] is not None:
        cur = came[cur]  # type: ignore[assignment]
        path.append(cur)
    path.reverse()
    return path
