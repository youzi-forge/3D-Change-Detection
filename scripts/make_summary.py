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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return float(xs[mid])
    return 0.5 * (float(xs[mid - 1]) + float(xs[mid]))


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
    qc_extras: dict[str, dict[str, Any]] = {}
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
        overlap_ref = float(qc.get("overlap_ref", 0.0))
        overlap_rescan = float(qc.get("overlap_rescan", 0.0))
        overlap_gate_value = qc.get("overlap_gate_value", None)
        if overlap_gate_value is None:
            overlap_gate_value = min(overlap_ref, overlap_rescan)
        changed_ref = float(qc.get("changed_ratio_ref", qc.get("changed_ratio_ref_obs", 0.0)))
        changed_rescan = float(qc.get("changed_ratio_rescan", qc.get("changed_ratio_rescan_obs", 0.0)))
        qc_extras[row.pair_id] = {
            "overlap_ref": overlap_ref,
            "overlap_rescan": overlap_rescan,
            "overlap_gate_value": float(overlap_gate_value),
            "gate_reason": str(qc.get("gate_reason", "")),
            "comparable_min": min(row.comparable_ratio_ref, row.comparable_ratio_rescan),
            "changed_max": max(changed_ref, changed_rescan),
        }

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
        unreliable_rows = [r for r in rows if not r.reliable]

        def rate(xs: list[bool]) -> float:
            return float(sum(xs)) / float(len(xs)) if xs else 0.0

        md_lines = []
        md_lines.append("# Summary")
        md_lines.append("")
        md_lines.append(f"Pairs summarized: {len(rows)}")
        md_lines.append(f"Reliable pairs: {len(reliable_rows)}")
        md_lines.append(f"Unreliable pairs: {len(unreliable_rows)}")
        md_lines.append("")
        md_lines.append("## Top-K hit rate (weak labels)")
        md_lines.append("")
        md_lines.append("| Scope | hit@3 | hit@5 | hit@10 |")
        md_lines.append("| --- | ---: | ---: | ---: |")
        md_lines.append(f"| all | {rate([r.top3_hit for r in rows]):.3f} | {rate([r.top5_hit for r in rows]):.3f} | {rate([r.top10_hit for r in rows]):.3f} |")
        md_lines.append(f"| reliable | {rate([r.top3_hit for r in reliable_rows]):.3f} | {rate([r.top5_hit for r in reliable_rows]):.3f} | {rate([r.top10_hit for r in reliable_rows]):.3f} |")
        md_lines.append("")
        md_lines.append("## QC overview")
        md_lines.append("")
        md_lines.append(f"- Reliable rate: {len(reliable_rows)}/{len(rows)} ({(len(reliable_rows)/len(rows)):.1%})")
        md_lines.append(f"- Median overlap_mean (reliable): {_median([r.overlap_mean for r in reliable_rows]):.3f}")
        md_lines.append(f"- Median comparable_min (reliable): {_median([float(qc_extras[r.pair_id]['comparable_min']) for r in reliable_rows if r.pair_id in qc_extras]):.3f}")
        md_lines.append(f"- Median changed_max (reliable): {_median([float(qc_extras[r.pair_id]['changed_max']) for r in reliable_rows if r.pair_id in qc_extras]):.3f}")

        if unreliable_rows:
            md_lines.append("")
            md_lines.append("## Unreliable pairs (for failure analysis)")
            md_lines.append("")
            md_lines.append("| pair_id | overlap_gate_value | overlap_mean | comparable_min | gate_reason | report |")
            md_lines.append("| --- | ---: | ---: | ---: | --- | --- |")
            for r in unreliable_rows:
                extra = qc_extras.get(r.pair_id, {})
                md_lines.append(
                    "| "
                    + " | ".join(
                        [
                            f"`{r.pair_id}`",
                            f"{float(extra.get('overlap_gate_value', 0.0)):.3f}",
                            f"{r.overlap_mean:.3f}",
                            f"{float(extra.get('comparable_min', 0.0)):.3f}",
                            str(extra.get("gate_reason", "")).replace("|", "\\|"),
                            f"`pairs/{r.pair_id}/report.html`",
                        ]
                    )
                    + " |"
                )
        md_lines.append("")
        md_lines.append("## Suggested qualitative review")
        md_lines.append("")
        md_lines.append("- Generate a shortlist of high-quality report candidates:")
        md_lines.append("  - `python3 scripts/make_hero_list.py --datasets-root Datasets --out-root <out_root> --write-md`")
        md_lines.append("- Use `configs/pairs/featured.json` for a small, representative set of success + failure cases.")
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
