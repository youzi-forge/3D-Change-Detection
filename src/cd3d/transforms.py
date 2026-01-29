from __future__ import annotations

import numpy as np


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    T = np.asarray(transform, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if T.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ homog.T).T[:, :3]
    return out


def with_translation_scaled(transform: np.ndarray, *, scale: float) -> np.ndarray:
    T = np.asarray(transform, dtype=np.float64).copy()
    if T.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    T[:3, 3] *= float(scale)
    return T


def is_homogeneous(transform: np.ndarray, *, atol: float = 1e-6) -> bool:
    T = np.asarray(transform, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    return bool(np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol))

