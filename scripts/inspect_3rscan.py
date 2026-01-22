#!/usr/bin/env python3
"""
Local 3RScan dataset inspector (pre-flight sanity checks).

What it does:
  - validates local folder layout
  - parses 3RScan.json
  - checks which scan folders are present and whether semantic files exist
  - picks a train/validation (reference, rescan) pair that is locally available
  - optionally writes configs/pairs/smoke_pair.local.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCAN_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

SEMANTIC_REQUIRED = (
    "labels.instances.annotated.v2.ply",
    "semseg.v2.json",
)

GEOM_REQUIRED = (
    "mesh.refined.v2.obj",
)


@dataclass(frozen=True)
class SmokePair:
    split: str
    reference_scan_id: str
    rescan_scan_id: str
    rigid_count: int
    removed_count: int
    nonrigid_count: int

    @property
    def change_count(self) -> int:
        return self.rigid_count + self.removed_count + self.nonrigid_count


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _find_meta_path(datasets_root: Path) -> Path:
    candidates = [
        datasets_root / "3RScan.json",
        datasets_root / "3RScan" / "3RScan.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find 3RScan.json. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _find_scans_root(datasets_root: Path) -> Path:
    # Prefer Datasets/3RScan/<scanId>/...
    preferred = datasets_root / "3RScan"
    if preferred.is_dir():
        return preferred

    # Fallback: scan folders placed directly under datasets_root
    # (Only accept if it actually looks like scan folders.)
    scan_dirs = [p for p in datasets_root.iterdir() if p.is_dir() and SCAN_ID_RE.match(p.name)]
    if scan_dirs:
        return datasets_root

    raise FileNotFoundError(
        "Could not find scans root. Expected either:\n"
        f"  - {preferred}/<scanId>/...\n"
        f"  - {datasets_root}/<scanId>/...\n"
    )


def _load_3rscan_meta(meta_path: Path) -> list[dict[str, Any]]:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, list):
        raise ValueError(f"Unexpected 3RScan.json format: expected list, got {type(meta)}")
    return meta


def _scan_dir_has(scan_dir: Path, required: Iterable[str]) -> bool:
    return all((scan_dir / name).exists() for name in required)


def _count_local_scans(scans_root: Path) -> tuple[int, int, int]:
    """Returns (total_scan_dirs, semantic_complete, mesh_only_or_partial)."""
    total = 0
    semantic_complete = 0
    partial = 0
    for p in scans_root.iterdir():
        if not p.is_dir():
            continue
        if not SCAN_ID_RE.match(p.name):
            continue
        total += 1
        has_sem = _scan_dir_has(p, SEMANTIC_REQUIRED)
        has_geom = _scan_dir_has(p, GEOM_REQUIRED)
        if has_sem and has_geom:
            semantic_complete += 1
        else:
            partial += 1
    return total, semantic_complete, partial


def _pick_smoke_pair(meta: list[dict[str, Any]], scans_root: Path) -> SmokePair | None:
    def has_full(scan_id: str) -> bool:
        d = scans_root / scan_id
        return d.is_dir() and _scan_dir_has(d, SEMANTIC_REQUIRED) and _scan_dir_has(d, GEOM_REQUIRED)

    best: SmokePair | None = None
    for scene in meta:
        split = scene.get("type")
        if split not in {"train", "validation"}:
            continue
        ref_id = scene.get("reference")
        if not isinstance(ref_id, str) or not has_full(ref_id):
            continue
        for scan in scene.get("scans", []):
            res_id = scan.get("reference")
            if not isinstance(res_id, str) or not has_full(res_id):
                continue
            rigid_count = len(scan.get("rigid", []) or [])
            removed_count = len(scan.get("removed", []) or [])
            nonrigid_count = len(scan.get("nonrigid", []) or [])
            candidate = SmokePair(
                split=split,
                reference_scan_id=ref_id,
                rescan_scan_id=res_id,
                rigid_count=rigid_count,
                removed_count=removed_count,
                nonrigid_count=nonrigid_count,
            )
            # Prefer a pair that actually has changes.
            if best is None:
                best = candidate
            elif best.change_count == 0 and candidate.change_count > 0:
                best = candidate
            elif candidate.change_count > best.change_count:
                best = candidate
    return best


def _read_ply_header(ply_path: Path) -> dict[str, Any]:
    """
    Reads ASCII header of a (binary) PLY file to extract:
      - format
      - vertex_count (if present)
      - vertex_properties (names)
    """
    fmt = None
    vertex_count = None
    vertex_properties: list[str] = []
    in_vertex_element = False

    with ply_path.open("rb") as f:
        while True:
            line_b = f.readline()
            if not line_b:
                break
            line = line_b.decode("utf-8", errors="replace").strip()
            if line.startswith("format "):
                fmt = line.split(" ", 1)[1]
            if line.startswith("element "):
                parts = line.split()
                if len(parts) >= 3 and parts[1] == "vertex":
                    in_vertex_element = True
                    vertex_count = int(parts[2])
                else:
                    in_vertex_element = False
            if in_vertex_element and line.startswith("property "):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_properties.append(parts[-1])
            if line == "end_header":
                break

    return {
        "format": fmt,
        "vertex_count": vertex_count,
        "vertex_properties": vertex_properties,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local 3RScan download and pick a smoke pair.")
    parser.add_argument(
        "--datasets-root",
        default="Datasets",
        help="Path containing 3RScan.json and the 3RScan scan folders (default: ./Datasets).",
    )
    parser.add_argument(
        "--write-smoke-config",
        action="store_true",
        help="Write configs/pairs/smoke_pair.local.json for the picked pair.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    datasets_root = (repo_root / args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)

    try:
        meta_path = _find_meta_path(datasets_root)
        scans_root = _find_scans_root(datasets_root)
    except FileNotFoundError as e:
        _eprint(str(e))
        return 2

    print("Datasets root:", datasets_root)
    print("Meta path:", meta_path)
    print("Scans root:", scans_root)

    meta = _load_3rscan_meta(meta_path)
    split_counts = {"train": 0, "validation": 0, "test": 0, "other": 0}
    total_rescans = 0
    for scene in meta:
        split = scene.get("type")
        if split in split_counts:
            split_counts[split] += 1
        else:
            split_counts["other"] += 1
        scans = scene.get("scans", [])
        if isinstance(scans, list):
            total_rescans += len(scans)

    print("Meta scenes:", len(meta))
    print("Reference scenes by split:", split_counts)
    print("Total rescans listed in meta:", total_rescans)

    total_dirs, semantic_complete, partial = _count_local_scans(scans_root)
    print("Local scan directories:", total_dirs)
    print("Local scan dirs with mesh+semantic:", semantic_complete)
    print("Local scan dirs partial/mesh-only:", partial)

    pair = _pick_smoke_pair(meta, scans_root)
    if pair is None:
        _eprint("Could not find a locally-available train/validation (reference, rescan) pair with semantic files yet.")
        _eprint("If download is still running, re-run this script later.")
        return 3

    print("\nPicked smoke pair:")
    print("  split:", pair.split)
    print("  reference:", pair.reference_scan_id)
    print("  rescan:", pair.rescan_scan_id)
    print("  meta changes: rigid=%d removed=%d nonrigid=%d (total=%d)"
          % (pair.rigid_count, pair.removed_count, pair.nonrigid_count, pair.change_count))

    # Quick sanity: show PLY header summary for both scans (no heavy parsing).
    for tag, scan_id in [("reference", pair.reference_scan_id), ("rescan", pair.rescan_scan_id)]:
        ply = scans_root / scan_id / "labels.instances.annotated.v2.ply"
        header = _read_ply_header(ply)
        props = header["vertex_properties"]
        has_object_id = "objectId" in props
        print(f"\n{tag} PLY header:")
        print("  path:", ply)
        print("  format:", header["format"])
        print("  vertex_count:", header["vertex_count"])
        print("  has objectId property:", has_object_id)

    if args.write_smoke_config:
        out_path = repo_root / "configs" / "pairs" / "smoke_pair.local.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "split": pair.split,
            "reference_scan_id": pair.reference_scan_id,
            "rescan_scan_id": pair.rescan_scan_id,
            "change_meta_counts": {
                "rigid": pair.rigid_count,
                "removed": pair.removed_count,
                "nonrigid": pair.nonrigid_count,
            },
            "source": "auto-picked by scripts/inspect_3rscan.py",
        }
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("\nWrote:", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
