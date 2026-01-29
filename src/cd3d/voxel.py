from __future__ import annotations

import numpy as np


def voxel_downsample(
    points: np.ndarray,
    *,
    voxel_size: float,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Deterministic voxel downsample: keeps the first point encountered per voxel.
    `extra_arrays` (e.g., objectId) are downsampled with the same indices.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    extra_arrays = extra_arrays or {}
    for k, v in extra_arrays.items():
        v = np.asarray(v)
        if v.shape[0] != pts.shape[0]:
            raise ValueError(f"extra array '{k}' must have length N={pts.shape[0]}")

    vs = float(voxel_size)
    if vs <= 0:
        return pts, {k: np.asarray(v) for k, v in extra_arrays.items()}

    coords = np.floor(pts / vs).astype(np.int64)
    key = np.ascontiguousarray(coords).view(np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))).ravel()
    _, unique_idx = np.unique(key, return_index=True)
    unique_idx = np.sort(unique_idx)

    pts_ds = pts[unique_idx]
    extras_ds = {k: np.asarray(v)[unique_idx] for k, v in extra_arrays.items()}
    return pts_ds, extras_ds

