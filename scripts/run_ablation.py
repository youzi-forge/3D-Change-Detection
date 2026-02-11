#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    # Stable, filesystem-friendly tag.
    s = f"{x:.6g}"
    return s.replace(".", "p")


@dataclass(frozen=True)
class AblationConfig:
    voxel_size: float
    tau: float

    @property
    def tag(self) -> str:
        return f"voxel{_tag_float(self.voxel_size)}_tau{_tag_float(self.tau)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a voxel_size x tau ablation and write per-setting summaries.")
    parser.add_argument("--datasets-root", default="Datasets", help="Datasets root containing 3RScan.json and scans.")
    parser.add_argument("--out-root", default="outputs/ablation", help="Base output root for ablation runs.")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--strategy", choices=["most_changes", "random"], default="most_changes")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--voxel-sizes", required=True, help="Comma-separated voxel sizes in meters, e.g. 0.01,0.02,0.05")
    parser.add_argument("--taus", required=True, help="Comma-separated tau thresholds in meters, e.g. 0.05,0.1,0.2")

    parser.add_argument("--overlap-delta", type=float, default=0.05)
    parser.add_argument("--overlap-min", type=float, default=0.30)
    parser.add_argument("--heat-cap-factor", type=float, default=5.0)
    parser.add_argument("--min-object-support", type=int, default=50)
    parser.add_argument("--min-object-total", type=int, default=20)
    parser.add_argument("--move-translation-min", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--scale-sample-size", type=int, default=8000)
    parser.add_argument("--exclude-labels", default="", help="Comma-separated reference labels to exclude from Top Objects.")

    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    base_out_root = Path(args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)
    base_out_root.mkdir(parents=True, exist_ok=True)

    voxel_sizes = _parse_float_list(args.voxel_sizes)
    taus = _parse_float_list(args.taus)

    run_batch = REPO_ROOT / "scripts" / "run_batch.py"
    make_summary = REPO_ROOT / "scripts" / "make_summary.py"

    settings = [AblationConfig(voxel_size=v, tau=t) for v in voxel_sizes for t in taus]
    print(f"Running {len(settings)} settings.")

    for i, s in enumerate(settings, start=1):
        out_root = base_out_root / s.tag
        out_root.mkdir(parents=True, exist_ok=True)
        print(f"[{i}/{len(settings)}] {s.tag} -> {out_root}")

        batch_cmd = [
            sys.executable,
            str(run_batch),
            "--datasets-root",
            str(datasets_root),
            "--out-root",
            str(out_root),
            "--split",
            str(args.split),
            "--strategy",
            str(args.strategy),
            "--limit",
            str(args.limit),
            "--seed",
            str(args.seed),
            "--voxel-size",
            str(s.voxel_size),
            "--tau",
            str(s.tau),
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
            "--scale-sample-size",
            str(args.scale_sample_size),
            "--exclude-labels",
            str(args.exclude_labels),
            "--skip-ply",
            "--skip-figures",
            "--skip-report",
        ]
        if args.resume:
            batch_cmd.append("--resume")

        p = subprocess.run(batch_cmd, text=True)
        if p.returncode != 0:
            print(f"Batch run failed for {s.tag} (exit={p.returncode}).", file=sys.stderr)
            return p.returncode

        summary_cmd = [
            sys.executable,
            str(make_summary),
            "--datasets-root",
            str(datasets_root),
            "--out-root",
            str(out_root),
            "--write-md",
        ]
        p2 = subprocess.run(summary_cmd, text=True)
        if p2.returncode != 0:
            print(f"Summary failed for {s.tag} (exit={p2.returncode}).", file=sys.stderr)
            return p2.returncode

    print("Ablation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
