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

from cd3d.three_rscan_meta import find_3rscan_json, get_rescan_meta, load_3rscan_meta  # noqa: E402


@dataclass(frozen=True)
class SummaryRow:
    pair_id: str
    split: str
    reliable: bool
    overlap_mean: float
    comparable_ratio_ref: float
    comparable_ratio_rescan: float
    top3_hit: bool
    top5_hit: bool
    top10_hit: bool
    gt_changed_count: int
    pred_count: int


def _read_objects_csv(path: Path) -> list[int]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        out: list[int] = []
        for row in r:
            if "objectId" not in row:
                continue
            try:
                oid = int(row["objectId"])
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


def _hit_at_k(pred: list[int], gt: set[int], k: int) -> bool:
    if k <= 0:
        return False
    return any(p in gt for p in pred[:k])


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize pair outputs and compute weak-label Top-K hit rates.")
    parser.add_argument("--datasets-root", default="Datasets", help="Datasets root containing 3RScan.json and scans.")
    parser.add_argument("--out-root", default="outputs", help="Output root directory.")
    parser.add_argument("--write-md", action="store_true", help="Also write summary.md next to summary.csv.")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    out_root = Path(args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)

    meta_path = find_3rscan_json(datasets_root)
    meta = load_3rscan_meta(meta_path)

    pairs_root = out_root / "pairs"
    if not pairs_root.is_dir():
        print(f"Missing pairs directory: {pairs_root}", file=sys.stderr)
        return 2

    rows: list[SummaryRow] = []
    for qc_path in sorted(pairs_root.glob("*/qc.json")):
        pair_dir = qc_path.parent
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        ref_id = qc.get("reference_scan_id")
        res_id = qc.get("rescan_scan_id")
        if not isinstance(ref_id, str) or not isinstance(res_id, str):
            continue

        try:
            rescan_meta = get_rescan_meta(meta, reference_scan_id=ref_id, rescan_scan_id=res_id)
        except Exception:
            # Pair might not exist in metadata, or ids mismatched.
            continue

        pred = _read_objects_csv(pair_dir / "objects.csv")
        gt = _gt_changed_object_ids(rescan_meta)

        row = SummaryRow(
            pair_id=f"{ref_id}__{res_id}",
            split=str(qc.get("split", "unknown")),
            reliable=bool(qc.get("reliable", False)),
            overlap_mean=float(qc.get("overlap_mean", 0.0)),
            comparable_ratio_ref=float(qc.get("comparable_ratio_ref", 0.0)),
            comparable_ratio_rescan=float(qc.get("comparable_ratio_rescan", 0.0)),
            top3_hit=_hit_at_k(pred, gt, 3),
            top5_hit=_hit_at_k(pred, gt, 5),
            top10_hit=_hit_at_k(pred, gt, 10),
            gt_changed_count=len(gt),
            pred_count=len(pred),
        )
        rows.append(row)

    if not rows:
        print("No qc.json files found to summarize.", file=sys.stderr)
        return 3

    out_csv = out_root / "summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(SummaryRow.__annotations__.keys()))
        for r in rows:
            w.writerow([
                r.pair_id,
                r.split,
                r.reliable,
                f"{r.overlap_mean:.6f}",
                f"{r.comparable_ratio_ref:.6f}",
                f"{r.comparable_ratio_rescan:.6f}",
                r.top3_hit,
                r.top5_hit,
                r.top10_hit,
                r.gt_changed_count,
                r.pred_count,
            ])

    print("Wrote:", out_csv)

    if args.write_md:
        reliable_rows = [r for r in rows if r.reliable]
        def rate(xs: list[bool]) -> float:
            return float(sum(xs)) / float(len(xs)) if xs else 0.0

        md_lines = []
        md_lines.append("# Summary")
        md_lines.append("")
        md_lines.append(f"Pairs summarized: {len(rows)}")
        md_lines.append(f"Reliable pairs: {len(reliable_rows)}")
        md_lines.append("")
        md_lines.append("## Top-K hit rate (weak labels)")
        md_lines.append("")
        md_lines.append("| Scope | hit@3 | hit@5 | hit@10 |")
        md_lines.append("| --- | ---: | ---: | ---: |")
        md_lines.append(f"| all | {rate([r.top3_hit for r in rows]):.3f} | {rate([r.top5_hit for r in rows]):.3f} | {rate([r.top10_hit for r in rows]):.3f} |")
        md_lines.append(f"| reliable | {rate([r.top3_hit for r in reliable_rows]):.3f} | {rate([r.top5_hit for r in reliable_rows]):.3f} | {rate([r.top10_hit for r in reliable_rows]):.3f} |")
        md_lines.append("")
        md_lines.append("## Notes")
        md_lines.append("")
        md_lines.append("- Weak labels come from `3RScan.json` (`removed`, `nonrigid`, and `rigid.instance_reference`).")
        md_lines.append("- Scores are geometry-only and should be interpreted with the QC gate and comparable ratios.")

        out_md = out_root / "summary.md"
        out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print("Wrote:", out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
