"""Load ROS map_server occupancy grids (.pgm + .yaml)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .grid import OccupancyGrid


def _parse_yaml_simple(path: Path) -> dict:
    data: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key == "origin":
            nums = val.strip("[]").split(",")
            data[key] = [float(x) for x in nums]
        elif key in {"resolution", "occupied_thresh", "free_thresh"}:
            data[key] = float(val)
        elif key == "negate":
            data[key] = int(val)
        else:
            data[key] = val.strip().strip("'\"")
    return data


def _read_pgm(path: Path) -> np.ndarray:
    """Return image as uint8 array with shape (H, W), row 0 = top of file."""
    raw = path.read_bytes()
    if raw.startswith(b"\x89PNG"):
        raise ValueError(f"Expected PGM, got PNG: {path}")
    # Text header then binary or ascii pixels
    if not raw.startswith(b"P5") and not raw.startswith(b"P2"):
        # sometimes whitespace before magic
        i = raw.find(b"P5")
        j = raw.find(b"P2")
        if i < 0 and j < 0:
            raise ValueError(f"Not a PGM file: {path}")
        start = i if i >= 0 and (j < 0 or i < j) else j
        raw = raw[start:]

    def _read_token(buf: bytes, pos: int) -> tuple[bytes, int]:
        while pos < len(buf) and buf[pos] in b" \t\r\n":
            pos += 1
        if pos < len(buf) and buf[pos] == ord("#"):
            while pos < len(buf) and buf[pos] not in b"\n":
                pos += 1
            return _read_token(buf, pos)
        start = pos
        while pos < len(buf) and buf[pos] not in b" \t\r\n":
            pos += 1
        return buf[start:pos], pos

    magic, pos = _read_token(raw, 0)
    w_b, pos = _read_token(raw, pos)
    h_b, pos = _read_token(raw, pos)
    max_b, pos = _read_token(raw, pos)
    width, height, maxval = int(w_b), int(h_b), int(max_b)
    if magic == b"P5":
        # single whitespace after maxval
        while pos < len(raw) and raw[pos] in b" \t\r":
            pos += 1
        if pos < len(raw) and raw[pos] == ord("\n"):
            pos += 1
        pixels = np.frombuffer(raw, dtype=np.uint8, count=width * height, offset=pos)
        img = pixels.reshape((height, width))
    else:
        # P2 ascii
        rest = raw[pos:].decode("ascii", errors="ignore").split()
        pixels = np.array([int(x) for x in rest[: width * height]], dtype=np.uint8)
        img = pixels.reshape((height, width))
    if maxval != 255:
        img = (img.astype(np.float32) * (255.0 / maxval)).astype(np.uint8)
    return img


def load_ros_map(yaml_path: str | Path) -> OccupancyGrid:
    """
    Load map_server yaml+pgm into OccupancyGrid.
    occ True = blocked. Row 0 is world-y bottom (ROS image flipped).
    """
    yaml_path = Path(yaml_path)
    meta = _parse_yaml_simple(yaml_path)
    image_name = meta.get("image")
    if not image_name:
        raise ValueError(f"No image: in {yaml_path}")
    pgm_path = yaml_path.parent / image_name
    img = _read_pgm(pgm_path)  # row0 = top of file

    # Flip so row 0 is bottom (matches ROS world y-up + our cell_to_xy)
    img = np.flipud(img)

    negate = int(meta.get("negate", 0))
    occ_th = float(meta.get("occupied_thresh", 0.65))
    free_th = float(meta.get("free_thresh", 0.25))
    resolution = float(meta["resolution"])
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    origin_xy = (float(origin[0]), float(origin[1]))

    # Map pixel value -> occupancy probability (ROS convention)
    # negate=0: black (0) occupied, white (254) free
    if negate:
        prob = img.astype(np.float64) / 255.0
    else:
        prob = 1.0 - (img.astype(np.float64) / 255.0)

    if not (0.0 <= free_th <= occ_th <= 1.0):
        raise ValueError(f"bad free/occ thresholds in {yaml_path}")
    # Unknown band (free_th..occ_th) treated as free so narrow aisles stay connected
    occ = prob >= occ_th

    return OccupancyGrid(occ=occ, resolution=resolution, origin_xy=origin_xy)


def resolve_map_yaml(map_yaml: str | Path, maps_root: Path | None = None) -> Path:
    """Resolve a map yaml path relative to repo maps/ (or absolute)."""
    root = maps_root or Path(__file__).resolve().parents[1] / "maps"
    p = Path(map_yaml)
    if p.is_absolute():
        return p
    return root / map_yaml


def nearest_free(grid: OccupancyGrid, r: int, c: int, max_r: int = 40) -> tuple[int, int]:
    if grid.free(r, c):
        return r, c
    for rad in range(1, max_r + 1):
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:
                    continue
                rr, cc = r + dr, c + dc
                if grid.free(rr, cc):
                    return rr, cc
    raise RuntimeError(f"No free cell near {(r, c)}")
