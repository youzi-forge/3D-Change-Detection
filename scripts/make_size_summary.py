#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd3d.semseg_v2 import load_semseg_axes_lengths  # noqa: E402
from cd3d.three_rscan_meta import find_3rscan_json, get_rescan_meta, load_3rscan_meta  # noqa: E402


KS = (3, 5, 10)


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("Empty list")
    return out


def _tag_float(x: float) -> str:
    s = f"{x:.6g}"
    return s.replace(".", "p")


@dataclass(frozen=True)
class BucketSpec:
    name: str
    lo: float | None
    hi: float | None
    display: str


def _bucket_specs(edges: list[float]) -> list[BucketSpec]:
    edges = sorted(float(e) for e in edges)
    if any(e <= 0 for e in edges):
        raise ValueError("All size edges must be > 0")
    if len(set(edges)) != len(edges):
        raise ValueError("Size edges must be unique")

    specs: list[BucketSpec] = []
    specs.append(BucketSpec(name=f"lt{_tag_float(edges[0])}", lo=None, hi=edges[0], display=f"< {edges[0]:.3f} m"))
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        specs.append(BucketSpec(name=f"{_tag_float(lo)}_{_tag_float(hi)}", lo=lo, hi=hi, display=f"[{lo:.3f}, {hi:.3f}) m"))
    specs.append(BucketSpec(name=f"ge{_tag_float(edges[-1])}", lo=edges[-1], hi=None, display=f">= {edges[-1]:.3f} m"))
    return specs


def _bucket_name(value: float, specs: list[BucketSpec]) -> str:
    x = float(value)
    for s in specs:
        if s.lo is None and s.hi is not None and x < s.hi:
            return s.name
        if s.lo is not None and s.hi is not None and (s.lo <= x < s.hi):
            return s.name
        if s.lo is not None and s.hi is None and x >= s.lo:
            return s.name
    return specs[-1].name


def _read_objects_csv(path: Path) -> list[int]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        out: list[int] = []
        for row in r:
            try:
                oid = int(row.get("objectId", "") or "0")
            except Exception:
                continue
            if oid <= 0:
                continue
            out.append(oid)
    return out


def _gt_changed_object_ids(rescan_meta: Any) -> set[int]:
    gt: set[int] = set()
    for oid in rescan_meta.removed:
        gt.add(int(oid))
    for oid in rescan_meta.nonrigid:
        gt.add(int(oid))
    for rigid in rescan_meta.rigid:
        try:
            gt.add(int(rigid.get("instance_reference")))
        except Exception:
            continue
    return gt


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute weak-label Top-K hit rates by object size buckets (OBB axesLengths).")
    parser.add_argument("--datasets-root", default="Datasets", help="Datasets root containing 3RScan.json and scans.")
    parser.add_argument("--out-root", default="outputs", help="Output root directory containing pairs/* outputs.")
    parser.add_argument("--size-edges", default="0.3,0.6,1.0", help="Comma-separated size bucket edges in meters.")
    parser.add_argument("--reliable-only", action="store_true", help="Only include pairs with qc.json reliable=true.")
    parser.add_argument("--write-md", action="store_true", help="Also write size_summary.md next to size_summary.csv.")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    out_root = Path(args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)

    meta_path = find_3rscan_json(datasets_root)
    meta = load_3rscan_meta(meta_path)

    pairs_root = out_root / "pairs"
    if not pairs_root.is_dir():
        print(f"Missing pairs directory: {pairs_root}", file=sys.stderr)
        return 2

    edges = _parse_float_list(args.size_edges)
    specs = _bucket_specs(edges)
    spec_by_name = {s.name: s for s in specs}

    bucket_pairs_with_gt: dict[str, int] = {s.name: 0 for s in specs}
    bucket_gt_objects_total: dict[str, int] = {s.name: 0 for s in specs}
    bucket_hits: dict[str, dict[int, int]] = {s.name: {k: 0 for k in KS} for s in specs}

    axes_cache: dict[str, dict[int, tuple[float, float, float]]] = {}

    pairs_total = 0
    pairs_used = 0
    gt_objects_total = 0
    gt_objects_with_obb = 0

    for qc_path in sorted(pairs_root.glob("*/qc.json")):
        pairs_total += 1
        pair_dir = qc_path.parent
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if args.reliable_only and not bool(qc.get("reliable", False)):
            continue
        ref_id = qc.get("reference_scan_id")
        res_id = qc.get("rescan_scan_id")
        if not isinstance(ref_id, str) or not isinstance(res_id, str):
            continue

        try:
            rescan_meta = get_rescan_meta(meta, reference_scan_id=ref_id, rescan_scan_id=res_id)
        except Exception:
            continue

        gt = _gt_changed_object_ids(rescan_meta)
        if not gt:
            continue
        pred = _read_objects_csv(pair_dir / "objects.csv")

        if ref_id not in axes_cache:
            semseg_path = datasets_root / "3RScan" / ref_id / "semseg.v2.json"
            if semseg_path.is_file():
                axes_cache[ref_id] = load_semseg_axes_lengths(semseg_path)
            else:
                axes_cache[ref_id] = {}
        axes_map = axes_cache[ref_id]

        gt_by_bucket: dict[str, set[int]] = {s.name: set() for s in specs}
        gt_objects_total += len(gt)
        for oid in gt:
            axes = axes_map.get(int(oid))
            if axes is None:
                continue
            gt_objects_with_obb += 1
            size_m = max(float(axes[0]), float(axes[1]), float(axes[2]))
            b = _bucket_name(size_m, specs)
            gt_by_bucket[b].add(int(oid))

        used_any_bucket = False
        for b, gt_set in gt_by_bucket.items():
            if not gt_set:
                continue
            used_any_bucket = True
            bucket_pairs_with_gt[b] += 1
            bucket_gt_objects_total[b] += len(gt_set)
            for k in KS:
                if any(p in gt_set for p in pred[:k]):
                    bucket_hits[b][k] += 1

        if used_any_bucket:
            pairs_used += 1

    out_csv = out_root / "size_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "range", "pairs_with_gt", "gt_object_total", "hit@3", "hit@5", "hit@10"])
        for s in specs:
            denom = bucket_pairs_with_gt[s.name]
            def rate(k: int) -> float:
                return float(bucket_hits[s.name][k]) / float(denom) if denom > 0 else 0.0

            w.writerow(
                [
                    s.name,
                    s.display,
                    denom,
                    bucket_gt_objects_total[s.name],
                    f"{rate(3):.6f}",
                    f"{rate(5):.6f}",
                    f"{rate(10):.6f}",
                ]
            )

    print("Wrote:", out_csv)
    print(f"Pairs scanned: {pairs_total}")
    print(f"Pairs used: {pairs_used}")
    print(f"GT objects (weak labels): {gt_objects_total}")
    if gt_objects_total > 0:
        print(f"GT objects with OBB: {gt_objects_with_obb} ({gt_objects_with_obb / gt_objects_total:.3f})")

    if args.write_md:
        lines: list[str] = []
        lines.append("# Size-bucket summary (weak labels)")
        lines.append("")
        lines.append(f"- Output root: `{out_root}`")
        lines.append(f"- Reliable-only: `{bool(args.reliable_only)}`")
        lines.append(f"- Size metric: `max(obb.axesLengths)` from reference `semseg.v2.json`")
        lines.append(f"- Size edges (m): `{', '.join(f'{e:.3f}' for e in edges)}`")
        lines.append("")
        lines.append("| Bucket | Range | Pairs | hit@3 | hit@5 | hit@10 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for s in specs:
            denom = bucket_pairs_with_gt[s.name]
            def rate(k: int) -> float:
                return float(bucket_hits[s.name][k]) / float(denom) if denom > 0 else 0.0

            lines.append(f"| `{s.name}` | {s.display} | {denom} | {rate(3):.3f} | {rate(5):.3f} | {rate(10):.3f} |")
        lines.append("")
        lines.append("## Notes")
        lines.append("")
        lines.append("- Buckets use only weak-label GT objects that have an OBB in the reference `semseg.v2.json`.")
        lines.append("- Interpret hit rates together with QC (overlap/comparable ratios) and qualitative reports.")

        out_md = out_root / "size_summary.md"
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Wrote:", out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

