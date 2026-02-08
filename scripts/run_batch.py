#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "src"))
from cd3d.three_rscan_meta import find_3rscan_json, load_3rscan_meta  # noqa: E402


@dataclass(frozen=True)
class Pair:
    reference_scan_id: str
    rescan_scan_id: str
    split: str
    change_count: int

    @property
    def pair_id(self) -> str:
        return f"{self.reference_scan_id}__{self.rescan_scan_id}"


def _scan_has_semantics(scan_dir: Path) -> bool:
    return (scan_dir / "labels.instances.annotated.v2.ply").is_file() and (scan_dir / "semseg.v2.json").is_file()


def _collect_pairs(
    *,
    meta: list[dict[str, Any]],
    scans_root: Path,
    split: str,
) -> list[Pair]:
    out: list[Pair] = []
    for scene in meta:
        scene_split = str(scene.get("type", "unknown"))
        if split != "all" and scene_split != split:
            continue
        reference_id = scene.get("reference")
        if not isinstance(reference_id, str):
            continue
        ref_dir = scans_root / reference_id
        if not _scan_has_semantics(ref_dir):
            continue
        for scan in scene.get("scans", []):
            rescan_id = scan.get("reference")
            if not isinstance(rescan_id, str):
                continue
            res_dir = scans_root / rescan_id
            if not _scan_has_semantics(res_dir):
                continue
            rigid = scan.get("rigid", []) or []
            removed = scan.get("removed", []) or []
            nonrigid = scan.get("nonrigid", []) or []
            change_count = int(len(rigid) + len(removed) + len(nonrigid))
            out.append(Pair(reference_scan_id=reference_id, rescan_scan_id=rescan_id, split=scene_split, change_count=change_count))
    return out


def _apply_selection_constraints(
    pairs: list[Pair],
    *,
    max_per_reference: int,
    max_references: int,
) -> list[Pair]:
    max_per_reference = int(max_per_reference)
    max_references = int(max_references)
    if max_per_reference <= 0 and max_references <= 0:
        return pairs

    selected: list[Pair] = []
    per_ref: dict[str, int] = {}
    refs: set[str] = set()

    for p in pairs:
        ref = p.reference_scan_id
        if max_references > 0 and ref not in refs and len(refs) >= max_references:
            continue
        if max_per_reference > 0 and per_ref.get(ref, 0) >= max_per_reference:
            continue
        selected.append(p)
        refs.add(ref)
        per_ref[ref] = per_ref.get(ref, 0) + 1

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scripts/run_pair.py for a batch of 3RScan pairs.")
    parser.add_argument("--datasets-root", default="Datasets", help="Datasets root containing 3RScan.json and scans.")
    parser.add_argument("--out-root", default="outputs", help="Output root directory.")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train", help="Which split to sample pairs from.")
    parser.add_argument("--strategy", choices=["most_changes", "random"], default="most_changes", help="Pair selection strategy.")
    parser.add_argument("--limit", type=int, default=20, help="Max number of pairs to run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (for strategy=random).")
    parser.add_argument("--max-per-reference", type=int, default=0, help="Max pairs per reference scan id (0=unlimited).")
    parser.add_argument("--max-references", type=int, default=0, help="Max unique reference scan ids (0=unlimited).")
    parser.add_argument("--resume", action="store_true", help="Skip pairs that already have qc.json under the output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected pairs and exit without running.")
    parser.add_argument("--write-pairs", help="Write selected pair ids to the given JSON file and exit.")

    # Pass-through parameters for run_pair.py
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--overlap-delta", type=float, default=0.05)
    parser.add_argument("--overlap-min", type=float, default=0.30)
    parser.add_argument("--heat-cap-factor", type=float, default=5.0)
    parser.add_argument("--min-object-support", type=int, default=50)
    parser.add_argument("--min-object-total", type=int, default=20)
    parser.add_argument("--move-translation-min", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--plot-max-points", type=int, default=60000)
    parser.add_argument("--scale-sample-size", type=int, default=8000)
    parser.add_argument("--skip-ply", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-report", action="store_true")

    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    out_root = Path(args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)

    meta_path = find_3rscan_json(datasets_root)
    meta = load_3rscan_meta(meta_path)
    scans_root = datasets_root / "3RScan"

    pairs = _collect_pairs(meta=meta, scans_root=scans_root, split=args.split)
    if not pairs:
        print("No eligible pairs found (check split selection and local semantics availability).", file=sys.stderr)
        return 2

    if args.strategy == "most_changes":
        pairs.sort(key=lambda p: (p.change_count, p.pair_id), reverse=True)
    else:
        rng = __import__("random").Random(int(args.seed))
        rng.shuffle(pairs)

    pairs = _apply_selection_constraints(
        pairs,
        max_per_reference=int(args.max_per_reference),
        max_references=int(args.max_references),
    )

    pairs = pairs[: max(0, int(args.limit))]
    print(f"Selected {len(pairs)} pairs (strategy={args.strategy}, split={args.split}).")

    if args.dry_run or args.write_pairs:
        payload = [
            {
                "reference_scan_id": p.reference_scan_id,
                "rescan_scan_id": p.rescan_scan_id,
                "split": p.split,
                "change_count": p.change_count,
            }
            for p in pairs
        ]
        if args.write_pairs:
            out_path = Path(args.write_pairs)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            print("Wrote:", out_path)
        else:
            for p in pairs:
                print(p.pair_id, "changes=", p.change_count)
        return 0

    run_pair = REPO_ROOT / "scripts" / "run_pair.py"
    success = 0
    failed: list[tuple[str, int]] = []

    for i, pair in enumerate(pairs, start=1):
        out_pair_dir = out_root / "pairs" / pair.pair_id
        qc_path = out_pair_dir / "qc.json"
        if args.resume and qc_path.is_file():
            print(f"[{i}/{len(pairs)}] SKIP {pair.pair_id} (qc.json exists)")
            success += 1
            continue

        cmd = [
            sys.executable,
            str(run_pair),
            "--datasets-root",
            str(datasets_root),
            "--out-root",
            str(out_root),
            "--reference",
            pair.reference_scan_id,
            "--rescan",
            pair.rescan_scan_id,
            "--voxel-size",
            str(args.voxel_size),
            "--tau",
            str(args.tau),
            "--overlap-delta",
            str(args.overlap_delta),
            "--overlap-min",
            str(args.overlap_min),
            "--heat-cap-factor",
            str(args.heat_cap_factor),
            "--min-object-support",
            str(args.min_object_support),
            "--min-object-total",
            str(args.min_object_total),
            "--move-translation-min",
            str(args.move_translation_min),
            "--top-k",
            str(args.top_k),
            "--plot-max-points",
            str(args.plot_max_points),
            "--scale-sample-size",
            str(args.scale_sample_size),
        ]
        if args.skip_ply:
            cmd.append("--skip-ply")
        if args.skip_figures:
            cmd.append("--skip-figures")
        if args.skip_report:
            cmd.append("--skip-report")

        print(f"[{i}/{len(pairs)}] RUN  {pair.pair_id} (changes={pair.change_count})")
        p = subprocess.run(cmd, text=True)
        if p.returncode == 0:
            success += 1
        else:
            failed.append((pair.pair_id, p.returncode))

    print(f"Done. success={success} failed={len(failed)}")
    if failed:
        print("Failures:")
        for pid, code in failed:
            print(f"  {pid} (exit={code})")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
