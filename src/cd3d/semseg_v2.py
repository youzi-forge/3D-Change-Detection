from __future__ import annotations

import json
from pathlib import Path


def load_semseg_labels(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for g in payload.get("segGroups", []):
        try:
            object_id = int(g.get("objectId"))
        except Exception:
            continue
        label = g.get("label")
        if isinstance(label, str) and label:
            out[object_id] = label
        else:
            out[object_id] = "unknown"
    return out


def load_semseg_axes_lengths(path: Path) -> dict[int, tuple[float, float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, tuple[float, float, float]] = {}
    for g in payload.get("segGroups", []):
        try:
            object_id = int(g.get("objectId"))
        except Exception:
            continue
        obb = g.get("obb")
        if not isinstance(obb, dict):
            continue
        axes = obb.get("axesLengths")
        if not isinstance(axes, list) or len(axes) != 3:
            continue
        try:
            out[object_id] = (float(axes[0]), float(axes[1]), float(axes[2]))
        except Exception:
            continue
    return out

