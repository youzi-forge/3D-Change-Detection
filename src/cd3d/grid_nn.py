from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np


_NEIGHBOR_OFFSETS_3 = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
)


@dataclass(frozen=True)
class NnResult:
    distances: np.ndarray  # (N,) float64, inf if none within radius
    indices: np.ndarray  # (N,) int64, -1 if none within radius


def _build_cell_map(points: np.ndarray, cell_size: float) -> tuple[np.ndarray, dict[tuple[int, int, int], list[int]]]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if cell_size <= 0:
        raise ValueError("cell_size must be > 0")
    cells = np.floor(pts / float(cell_size)).astype(np.int64)
    cell_map: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx, c in enumerate(cells):
        cell_map[(int(c[0]), int(c[1]), int(c[2]))].append(idx)
    return cells, dict(cell_map)


def nearest_neighbors_within_radius(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    *,
    radius: float,
) -> NnResult:
    """
    Finds the nearest neighbor in tgt_points for each src point, but only if the neighbor is within `radius`.

    Implementation uses a uniform grid (cell size = radius) with 27 neighboring cell lookups.
    This is an exact NN search within the specified radius.

    Returns:
      - distances: inf if no neighbor within radius
      - indices: -1 if no neighbor within radius
    """
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be > 0")

    src = np.asarray(src_points, dtype=np.float64)
    tgt = np.asarray(tgt_points, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("src_points must have shape (N, 3)")
    if tgt.ndim != 2 or tgt.shape[1] != 3:
        raise ValueError("tgt_points must have shape (M, 3)")

    n = src.shape[0]
    distances2_out = np.full((n,), np.inf, dtype=np.float64)
    indices_out = np.full((n,), -1, dtype=np.int64)

    if tgt.shape[0] == 0 or n == 0:
        return NnResult(distances=np.sqrt(distances2_out), indices=indices_out)

    _, src_map = _build_cell_map(src, r)
    _, tgt_map = _build_cell_map(tgt, r)

    r2 = r * r
    for cell, src_indices_list in src_map.items():
        src_indices = np.asarray(src_indices_list, dtype=np.int64)
        src_pts = src[src_indices]
        src_sq = np.sum(src_pts * src_pts, axis=1)

        best_d2 = np.full((src_pts.shape[0],), np.inf, dtype=np.float64)
        best_j = np.full((src_pts.shape[0],), -1, dtype=np.int64)

        cx, cy, cz = cell
        for dx, dy, dz in _NEIGHBOR_OFFSETS_3:
            key = (cx + dx, cy + dy, cz + dz)
            tgt_indices_list = tgt_map.get(key)
            if not tgt_indices_list:
                continue
            tgt_indices = np.asarray(tgt_indices_list, dtype=np.int64)
            tgt_pts = tgt[tgt_indices]
            tgt_sq = np.sum(tgt_pts * tgt_pts, axis=1)

            # dist^2 = ||a||^2 + ||b||^2 - 2 a dot b
            dist2 = src_sq[:, None] + tgt_sq[None, :] - 2.0 * (src_pts @ tgt_pts.T)
            local_argmin = np.argmin(dist2, axis=1)
            local_min = dist2[np.arange(dist2.shape[0]), local_argmin]

            better = local_min < best_d2
            if np.any(better):
                best_d2[better] = local_min[better]
                best_j[better] = tgt_indices[local_argmin[better]]

        within = best_d2 <= r2
        if np.any(within):
            out_idx = src_indices[within]
            distances2_out[out_idx] = best_d2[within]
            indices_out[out_idx] = best_j[within]

    return NnResult(distances=np.sqrt(distances2_out), indices=indices_out)


def overlap_ratio(distances: np.ndarray, *, threshold: float) -> float:
    d = np.asarray(distances, dtype=np.float64)
    if d.size == 0:
        return 0.0
    return float(np.mean(d < float(threshold)))

