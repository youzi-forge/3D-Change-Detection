from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RescanMeta:
    split: str
    reference_scan_id: str
    rescan_scan_id: str
    transform: np.ndarray  # (4,4) float64
    rigid: list[dict[str, Any]]
    removed: list[int]
    nonrigid: list[int]


def find_3rscan_json(datasets_root: Path) -> Path:
    candidates = [
        datasets_root / "3RScan.json",
        datasets_root / "3RScan" / "3RScan.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("Could not find 3RScan.json under the provided datasets root.")


def load_3rscan_meta(meta_path: Path) -> list[dict[str, Any]]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, list):
        raise ValueError(f"Unexpected 3RScan.json format: expected list, got {type(meta)}")
    return meta


def _matrix_from_json_list(values: list[float]) -> np.ndarray:
    if len(values) != 16:
        raise ValueError(f"Expected 16 values for a 4x4 transform, got {len(values)}")
    arr = np.asarray(values, dtype=np.float64)
    # 3RScan toolkit fills Eigen matrices in column-major order.
    return arr.reshape((4, 4), order="F")


def get_rescan_meta(
    meta: list[dict[str, Any]],
    *,
    reference_scan_id: str,
    rescan_scan_id: str,
) -> RescanMeta:
    for scene in meta:
        if scene.get("reference") != reference_scan_id:
            continue
        split = str(scene.get("type", "unknown"))
        for scan in scene.get("scans", []):
            if scan.get("reference") != rescan_scan_id:
                continue
            transform_list = scan.get("transform")
            if not isinstance(transform_list, list):
                raise ValueError("Missing transform list in rescan record.")
            transform = _matrix_from_json_list(transform_list)
            rigid = list(scan.get("rigid", []) or [])
            removed = list(scan.get("removed", []) or [])
            nonrigid = list(scan.get("nonrigid", []) or [])
            return RescanMeta(
                split=split,
                reference_scan_id=reference_scan_id,
                rescan_scan_id=rescan_scan_id,
                transform=transform,
                rigid=rigid,
                removed=removed,
                nonrigid=nonrigid,
            )

    raise KeyError(
        "Could not find (reference, rescan) pair in 3RScan.json: "
        f"{reference_scan_id} -> {rescan_scan_id}"
    )

