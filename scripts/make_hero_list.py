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
class PairStats:
    reference_scan_id: str
    rescan_scan_id: str
    split: str
    reliable: bool
    overlap_gate_value: float
    overlap_mean: float
    comparable_min: float
    unchanged_min: float
    changed_max: float
    gt_changed_count: int | None
    gt_removed: int | None
    gt_rigid: int | None
    gt_nonrigid: int | None
    top1: str | None

    @property
    def pair_id(self) -> str:
        return f"{self.reference_scan_id}__{self.rescan_scan_id}"

    @property
    def report_relpath(self) -> str:
        return f"pairs/{self.pair_id}/report.html"


def _read_top1(objects_csv: Path) -> str | None:
    if not objects_csv.is_file():
        return None
    with objects_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        row = next(r, None)
        if not row:
            return None
        oid = (row.get("objectId") or "").strip()
        label_ref = (row.get("label_ref") or "").strip()
        label_rescan = (row.get("label_rescan") or "").strip()
        change_type = (row.get("type") or "").strip()
        score_str = (row.get("score") or "").strip()
        conf_str = (row.get("type_confidence") or "").strip()
        try:
            score = float(score_str)
        except Exception:
            score = float("nan")
        try:
            conf = float(conf_str)
        except Exception:
            conf = float("nan")
        labels = f"{label_ref}->{label_rescan}" if label_ref or label_rescan else ""
        parts = []
        if oid:
            parts.append(oid)
        if change_type:
            parts.append(change_type)
        if score == score:  # NaN check
            parts.append(f"{score:.3f}")
        if conf == conf:
            parts.append(f"c={conf:.2f}")
        if labels:
            parts.append(labels)
        return " ".join(parts) if parts else None


def _gt_counts(rescan_meta: Any) -> tuple[int, int, int, int]:
    removed = len(getattr(rescan_meta, "removed", []) or [])
    nonrigid = len(getattr(rescan_meta, "nonrigid", []) or [])
    rigid = len(getattr(rescan_meta, "rigid", []) or [])
    gt_changed = removed + nonrigid + rigid
    return gt_changed, removed, rigid, nonrigid


def _load_stats(
    out_root: Path,
    *,
    meta: list[dict[str, Any]] | None,
) -> list[PairStats]:
    pairs_root = out_root / "pairs"
    stats: list[PairStats] = []
    for qc_path in sorted(pairs_root.glob("*/qc.json")):
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        ref_id = qc.get("reference_scan_id")
        res_id = qc.get("rescan_scan_id")
        if not isinstance(ref_id, str) or not isinstance(res_id, str):
            continue

        comparable_ref = float(qc.get("comparable_ratio_ref", 0.0))
        comparable_rescan = float(qc.get("comparable_ratio_rescan", 0.0))
        unchanged_ref = float(qc.get("unchanged_ratio_ref", qc.get("unchanged_ratio_ref_obs", 0.0)))
        unchanged_rescan = float(qc.get("unchanged_ratio_rescan", qc.get("unchanged_ratio_rescan_obs", 0.0)))
        changed_ref = float(qc.get("changed_ratio_ref", qc.get("changed_ratio_ref_obs", 0.0)))
        changed_rescan = float(qc.get("changed_ratio_rescan", qc.get("changed_ratio_rescan_obs", 0.0)))

        overlap_gate_value = qc.get("overlap_gate_value", None)
        if overlap_gate_value is None:
            try:
                overlap_gate_value = min(float(qc.get("overlap_ref", 0.0)), float(qc.get("overlap_rescan", 0.0)))
            except Exception:
                overlap_gate_value = 0.0

        gt_changed = removed = rigid = nonrigid = None
        if meta is not None:
            try:
                rescan_meta = get_rescan_meta(meta, reference_scan_id=ref_id, rescan_scan_id=res_id)
                gt_changed, removed, rigid, nonrigid = _gt_counts(rescan_meta)
            except Exception:
                gt_changed = removed = rigid = nonrigid = None

        pair_dir = qc_path.parent
        top1 = _read_top1(pair_dir / "objects.csv")
        stats.append(
            PairStats(
                reference_scan_id=ref_id,
                rescan_scan_id=res_id,
                split=str(qc.get("split", "unknown")),
                reliable=bool(qc.get("reliable", False)),
                overlap_gate_value=float(overlap_gate_value),
                overlap_mean=float(qc.get("overlap_mean", 0.0)),
                comparable_min=min(comparable_ref, comparable_rescan),
                unchanged_min=min(unchanged_ref, unchanged_rescan),
                changed_max=max(changed_ref, changed_rescan),
                gt_changed_count=gt_changed,
                gt_removed=removed,
                gt_rigid=rigid,
                gt_nonrigid=nonrigid,
                top1=top1,
            )
        )
    return stats


def _select_top(
    items: list[PairStats],
    *,
    key,
    top_n: int,
    max_per_reference: int,
    reverse: bool,
) -> list[PairStats]:
    out: list[PairStats] = []
    per_ref: dict[str, int] = {}
    for s in sorted(items, key=key, reverse=reverse):
        if max_per_reference > 0:
            n = per_ref.get(s.reference_scan_id, 0)
            if n >= max_per_reference:
                continue
            per_ref[s.reference_scan_id] = n + 1
        out.append(s)
        if len(out) >= top_n:
            break
    return out


def _format_table(rows: list[PairStats]) -> list[str]:
    lines: list[str] = []
    lines.append(
        "| pair_id | split | overlap_mean | comparable_min | unchanged_min | changed_max | gt_changed | top1 | report |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for r in rows:
        gt = "" if r.gt_changed_count is None else str(r.gt_changed_count)
        top1 = r.top1 or ""
        report = f"`{r.report_relpath}`"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r.pair_id}`",
                    r.split,
                    f"{r.overlap_mean:.3f}",
                    f"{r.comparable_min:.3f}",
                    f"{r.unchanged_min:.3f}",
                    f"{r.changed_max:.3f}",
                    gt,
                    top1,
                    report,
                ]
            )
            + " |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a short list of report candidates from an output root.")
    parser.add_argument("--datasets-root", default="Datasets", help="Datasets root containing 3RScan.json (optional).")
    parser.add_argument("--out-root", default="outputs/showcase", help="Output root directory containing pairs/*/qc.json.")
    parser.add_argument("--split", default="", help="Optional split filter (train/validation/test).")
    parser.add_argument("--top-reliable", type=int, default=10, help="How many reliable candidates to list.")
    parser.add_argument("--top-unreliable", type=int, default=5, help="How many unreliable candidates to list.")
    parser.add_argument("--max-per-reference", type=int, default=2, help="Max listed pairs per reference scan id (0 disables).")
    parser.add_argument("--min-overlap", type=float, default=0.65, help="Filter reliable candidates by overlap_mean.")
    parser.add_argument("--min-comparable", type=float, default=0.90, help="Filter reliable candidates by comparable_min.")
    parser.add_argument("--min-changed", type=float, default=0.02, help="Filter reliable candidates by changed_max.")
    parser.add_argument("--max-changed", type=float, default=0.25, help="Filter reliable candidates by changed_max.")
    parser.add_argument("--write-md", action="store_true", help="Write hero_candidates.md under the output root.")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    out_root = Path(args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)

    meta: list[dict[str, Any]] | None = None
    try:
        meta_path = find_3rscan_json(datasets_root)
        meta = load_3rscan_meta(meta_path)
    except Exception:
        meta = None

    stats = _load_stats(out_root, meta=meta)
    if args.split:
        stats = [s for s in stats if s.split == args.split]

    reliable = [s for s in stats if s.reliable]
    unreliable = [s for s in stats if not s.reliable]

    reliable = [
        s
        for s in reliable
        if (s.overlap_mean >= args.min_overlap)
        and (s.comparable_min >= args.min_comparable)
        and (args.min_changed <= s.changed_max <= args.max_changed)
    ]

    top_reliable = _select_top(
        reliable,
        key=lambda s: (s.overlap_mean, s.comparable_min, s.unchanged_min),
        top_n=max(0, int(args.top_reliable)),
        max_per_reference=int(args.max_per_reference),
        reverse=True,
    )
    top_unreliable = _select_top(
        unreliable,
        key=lambda s: (s.overlap_gate_value, s.overlap_mean),
        top_n=max(0, int(args.top_unreliable)),
        max_per_reference=int(args.max_per_reference),
        reverse=False,
    )

    lines: list[str] = []
    lines.append("# Hero candidates")
    lines.append("")
    lines.append(f"Output root: `{out_root}`")
    lines.append("")
    lines.append("## Reliable")
    lines.append("")
    if top_reliable:
        lines.extend(_format_table(top_reliable))
    else:
        lines.append("_No reliable candidates matched the current filters._")
    lines.append("")
    lines.append("## Unreliable")
    lines.append("")
    if top_unreliable:
        lines.extend(_format_table(top_unreliable))
    else:
        lines.append("_No unreliable candidates found._")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Reliable list is filtered by overlap/comparable/changing ratios; adjust flags to broaden or narrow.")
    lines.append("- Report paths are relative to the output root.")
    if meta is None:
        lines.append("- Weak-label counts are omitted because 3RScan.json was not found under the provided datasets root.")

    content = "\n".join(lines) + "\n"
    if args.write_md:
        out_path = out_root / "hero_candidates.md"
        out_path.write_text(content, encoding="utf-8")
        print("Wrote:", out_path)
    else:
        print(content)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
