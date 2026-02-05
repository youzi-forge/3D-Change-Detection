#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Allow running without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cd3d.grid_nn import nearest_neighbors_within_radius, overlap_ratio  # noqa: E402
from cd3d.ply_ascii import read_3rscan_instance_ply, write_ply_ascii  # noqa: E402
from cd3d.three_rscan_meta import (  # noqa: E402
    find_3rscan_json,
    get_rescan_meta,
    load_3rscan_meta,
)
from cd3d.transforms import apply_transform, is_homogeneous, with_translation_scaled  # noqa: E402
from cd3d.voxel import voxel_downsample  # noqa: E402


@dataclass(frozen=True)
class PairConfig:
    reference_scan_id: str
    rescan_scan_id: str


def _load_pair_config(path: Path) -> PairConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PairConfig(
        reference_scan_id=str(payload["reference_scan_id"]),
        rescan_scan_id=str(payload["rescan_scan_id"]),
    )


def _load_semseg_labels(path: Path) -> dict[int, str]:
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


def _pick_translation_scale(
    *,
    ref_points: np.ndarray,
    res_points: np.ndarray,
    transform: np.ndarray,
    overlap_delta: float,
    sample_size: int,
) -> tuple[float, dict[str, Any]]:
    rng = np.random.default_rng(0)
    n_ref = ref_points.shape[0]
    n_res = res_points.shape[0]
    ref_idx = rng.choice(n_ref, size=min(sample_size, n_ref), replace=False)
    res_idx = rng.choice(n_res, size=min(sample_size, n_res), replace=False)
    ref_s = ref_points[ref_idx]
    res_s = res_points[res_idx]

    candidates = [1.0, 0.001]
    scores: dict[float, float] = {}
    for scale in candidates:
        T = with_translation_scaled(transform, scale=scale)
        res_aligned = apply_transform(res_s, T)
        d_ref = nearest_neighbors_within_radius(ref_s, res_aligned, radius=overlap_delta).distances
        d_res = nearest_neighbors_within_radius(res_aligned, ref_s, radius=overlap_delta).distances
        ov = 0.5 * (overlap_ratio(d_ref, threshold=overlap_delta) + overlap_ratio(d_res, threshold=overlap_delta))
        scores[scale] = float(ov)

    best_scale = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0]
    debug = {"candidate_overlap_mean": {str(k): v for k, v in scores.items()}}
    return best_scale, debug


def _heat_from_distance(dist: np.ndarray, *, tau: float) -> np.ndarray:
    tau = float(tau)
    if tau <= 0:
        raise ValueError("tau must be > 0")
    d = np.asarray(dist, dtype=np.float64)
    heat = np.clip(d / tau, 0.0, 1.0)
    return heat


def _colors_from_heat(heat: np.ndarray, *, comparable: np.ndarray) -> np.ndarray:
    import matplotlib

    h = np.asarray(heat, dtype=np.float64)
    comp = np.asarray(comparable, dtype=bool)
    cmap = matplotlib.colormaps.get_cmap("inferno")
    rgba = cmap(np.clip(h, 0.0, 1.0))  # (N,4) floats
    rgb = (rgba[:, :3] * 255.0).astype(np.uint8)
    rgb[~comp] = np.array([200, 200, 200], dtype=np.uint8)
    return rgb


def _compute_object_stats(
    *,
    object_ids: np.ndarray,
    distances: np.ndarray,
    observed: np.ndarray,
    tau: float,
    min_support: int,
) -> dict[int, dict[str, Any]]:
    obj = np.asarray(object_ids, dtype=np.int64)
    dist = np.asarray(distances, dtype=np.float64)
    obs = np.asarray(observed, dtype=bool)
    changed = obs & (dist > float(tau))

    out: dict[int, dict[str, Any]] = {}

    unique, counts = np.unique(obj, return_counts=True)
    total_counts = {int(k): int(v) for k, v in zip(unique, counts)}

    unique_o, counts_o = np.unique(obj[obs], return_counts=True) if np.any(obs) else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    obs_counts = {int(k): int(v) for k, v in zip(unique_o, counts_o)}

    unique_ch, counts_ch = np.unique(obj[changed], return_counts=True) if np.any(changed) else (np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    changed_counts = {int(k): int(v) for k, v in zip(unique_ch, counts_ch)}

    # Collect changed-point distances for robust percentiles.
    changed_dist_by_obj: dict[int, list[float]] = {}
    if np.any(changed):
        for oid, d in zip(obj[changed], dist[changed], strict=False):
            oid_i = int(oid)
            changed_dist_by_obj.setdefault(oid_i, []).append(float(d))

    for oid, support_total in total_counts.items():
        support_obs = obs_counts.get(oid, 0)
        changed_obs = changed_counts.get(oid, 0)
        ratio = float(changed_obs) / float(support_obs) if support_obs > 0 else 0.0
        p95 = 0.0
        if oid in changed_dist_by_obj and len(changed_dist_by_obj[oid]) >= 1:
            p95 = float(np.percentile(np.asarray(changed_dist_by_obj[oid], dtype=np.float64), 95))
        out[oid] = {
            "support_total": int(support_total),
            "support_observed": int(support_obs),
            "changed_observed": int(changed_obs),
            "ratio_changed_observed": ratio,
            "p95_changed_distance": p95,
            "is_low_support": bool(support_obs < int(min_support)),
        }
    return out


def _centroid(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.zeros((3,), dtype=np.float64)
    return np.mean(points, axis=0)


def _assign_change_type(
    *,
    object_id: int,
    ref_stats: dict[int, dict[str, Any]],
    res_stats: dict[int, dict[str, Any]],
    reliable: bool,
    ref_points: np.ndarray,
    res_points: np.ndarray,
    ref_object_ids: np.ndarray,
    res_object_ids: np.ndarray,
    ref_observed: np.ndarray,
    res_observed: np.ndarray,
    move_translation_min: float,
    min_support_total: int,
) -> tuple[str, float]:
    ref_support_total = int(ref_stats.get(object_id, {}).get("support_total", 0))
    res_support_total = int(res_stats.get(object_id, {}).get("support_total", 0))

    if ref_support_total >= min_support_total and res_support_total < min_support_total:
        return ("disappeared", 0.9 if reliable else 0.4)
    if ref_support_total < min_support_total and res_support_total >= min_support_total:
        return ("appeared", 0.9 if reliable else 0.4)

    if not reliable:
        return ("unknown", 0.2)

    # Both present: use a conservative centroid-shift heuristic.
    ref_mask = (ref_object_ids == object_id) & ref_observed
    res_mask = (res_object_ids == object_id) & res_observed
    if np.sum(ref_mask) < min_support_total or np.sum(res_mask) < min_support_total:
        return ("unknown", 0.3)

    c_ref = _centroid(ref_points[ref_mask])
    c_res = _centroid(res_points[res_mask])
    shift = float(np.linalg.norm(c_res - c_ref))
    if shift >= float(move_translation_min):
        return ("moved_rigid", 0.7)
    return ("nonrigid_or_recon", 0.5)


def _write_objects_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_report_html(
    path: Path,
    *,
    title: str,
    summary: dict[str, Any],
    object_rows: list[dict[str, Any]],
    images: list[tuple[str, str]],
    notes: list[str],
) -> None:
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    rows_html = "\n".join(
        "<tr>"
        + "".join(f"<td>{esc(str(v))}</td>" for v in row.values())
        + "</tr>"
        for row in object_rows
    )
    header_html = ""
    if object_rows:
        header_html = "<tr>" + "".join(f"<th>{esc(k)}</th>" for k in object_rows[0].keys()) + "</tr>"

    images_html = "\n".join(
        f"<figure><img src=\"{esc(src)}\" alt=\"{esc(caption)}\" /><figcaption>{esc(caption)}</figcaption></figure>"
        for src, caption in images
    )

    summary_html = "\n".join(
        f"<tr><th>{esc(str(k))}</th><td>{esc(str(v))}</td></tr>" for k, v in summary.items()
    )

    notes_html = "\n".join(f"<li>{esc(n)}</li>" for n in notes)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{esc(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin: 0 0 8px 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 13px; }}
    th {{ background: #f6f6f6; text-align: left; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #555; margin-top: 6px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #eee; }}
    .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .ok {{ background: #e7f6ea; color: #0f5b1c; }}
    .bad {{ background: #fde8e8; color: #7a1b1b; }}
    code {{ background: #f6f6f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>{esc(title)}</h1>
  <div class="grid">
    <section>
      <h2>Summary</h2>
      <table>
        {summary_html}
      </table>
    </section>
    <section>
      <h2>Figures</h2>
      {images_html}
    </section>
    <section>
      <h2>Top Objects</h2>
      <table>
        {header_html}
        {rows_html}
      </table>
    </section>
    <section>
      <h2>Notes</h2>
      <ul>
        {notes_html}
      </ul>
    </section>
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _save_three_view_overlay_png(
    path: Path,
    *,
    ref_points: np.ndarray,
    res_points: np.ndarray,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    def sample(points: np.ndarray) -> np.ndarray:
        if points.shape[0] <= max_points:
            return points
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        return points[idx]

    ref = sample(ref_points)
    res = sample(res_points)

    views = [
        (0, 1, "XY"),
        (0, 2, "XZ"),
        (1, 2, "YZ"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, (a, b, title) in zip(axes, views, strict=False):
        ax.scatter(ref[:, a], ref[:, b], s=0.6, c="#1f77b4", alpha=0.35, linewidths=0)
        ax.scatter(res[:, a], res[:, b], s=0.6, c="#ff7f0e", alpha=0.35, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_three_view_heatmap_png(
    path: Path,
    *,
    points: np.ndarray,
    heat: np.ndarray,
    comparable: np.ndarray,
    max_points: int,
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=np.float64)
    h = np.asarray(heat, dtype=np.float64)
    comp = np.asarray(comparable, dtype=bool)
    if pts.shape[0] != h.shape[0] or pts.shape[0] != comp.shape[0]:
        raise ValueError("points/heat/comparable lengths mismatch")

    if pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        h = h[idx]
        comp = comp[idx]

    views = [
        (0, 1, f"{title_prefix} XY"),
        (0, 2, f"{title_prefix} XZ"),
        (1, 2, f"{title_prefix} YZ"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    for ax, (a, b, title) in zip(axes, views, strict=False):
        ax.scatter(pts[~comp, a], pts[~comp, b], s=0.6, c="#c8c8c8", alpha=0.15, linewidths=0)
        sc = ax.scatter(pts[comp, a], pts[comp, b], s=0.6, c=h[comp], cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.7, linewidths=0)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, location="right", pad=0.01)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.02)
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single 3RScan (reference, rescan) pair and export artifacts.")
    parser.add_argument("--datasets-root", default="Datasets", help="Datasets root containing 3RScan.json and scans.")
    parser.add_argument("--pair-config", help="Path to a JSON file with reference_scan_id and rescan_scan_id.")
    parser.add_argument("--reference", help="Reference scan id (if --pair-config is not provided).")
    parser.add_argument("--rescan", help="Rescan scan id (if --pair-config is not provided).")
    parser.add_argument("--out-root", default="outputs", help="Output root directory (default: outputs).")

    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size in meters for downsampling.")
    parser.add_argument("--tau", type=float, default=0.10, help="Change threshold tau in meters.")
    parser.add_argument("--overlap-delta", type=float, default=0.05, help="Overlap/comparable threshold in meters.")
    parser.add_argument("--overlap-min", type=float, default=0.30, help="Reliability gate threshold on overlap_mean.")
    parser.add_argument("--heat-cap-factor", type=float, default=5.0, help="Cap NN search radius at heat_cap_factor * tau.")
    parser.add_argument("--min-object-support", type=int, default=50, help="Min observed support points for stable object stats.")
    parser.add_argument("--min-object-total", type=int, default=20, help="Min total points to consider object present.")
    parser.add_argument("--move-translation-min", type=float, default=0.20, help="Centroid shift threshold for moved_rigid typing.")
    parser.add_argument("--top-k", type=int, default=15, help="Number of objects to report.")
    parser.add_argument("--plot-max-points", type=int, default=60000, help="Max points to plot per figure (subsampled).")
    parser.add_argument("--scale-sample-size", type=int, default=8000, help="Sample size used for translation-scale detection.")
    parser.add_argument("--skip-ply", action="store_true", help="Skip writing heatmap PLY files.")
    parser.add_argument("--skip-figures", action="store_true", help="Skip writing PNG figures.")
    parser.add_argument("--skip-report", action="store_true", help="Skip writing the HTML report.")

    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve() if not Path(args.datasets_root).is_absolute() else Path(args.datasets_root)
    out_root = Path(args.out_root).resolve() if not Path(args.out_root).is_absolute() else Path(args.out_root)

    if args.pair_config:
        cfg = _load_pair_config(Path(args.pair_config))
        reference_id = cfg.reference_scan_id
        rescan_id = cfg.rescan_scan_id
    else:
        if not args.reference or not args.rescan:
            parser.error("Provide --pair-config or both --reference and --rescan.")
        reference_id = args.reference
        rescan_id = args.rescan

    scan_root = datasets_root / "3RScan"
    ref_dir = scan_root / reference_id
    res_dir = scan_root / rescan_id
    ref_ply = ref_dir / "labels.instances.annotated.v2.ply"
    res_ply = res_dir / "labels.instances.annotated.v2.ply"
    ref_semseg = ref_dir / "semseg.v2.json"
    res_semseg = res_dir / "semseg.v2.json"

    missing = [p for p in (ref_ply, res_ply, ref_semseg, res_semseg) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing required file: {p}", file=sys.stderr)
        return 2

    meta_path = find_3rscan_json(datasets_root)
    meta = load_3rscan_meta(meta_path)
    rescan_meta = get_rescan_meta(meta, reference_scan_id=reference_id, rescan_scan_id=rescan_id)

    if not is_homogeneous(rescan_meta.transform):
        print("WARNING: transform does not look homogeneous (last row not [0,0,0,1]). Proceeding anyway.", file=sys.stderr)

    t0 = time.time()

    ref_points, ref_object_ids = read_3rscan_instance_ply(ref_ply)
    res_points, res_object_ids = read_3rscan_instance_ply(res_ply)

    # Detect whether translation appears to be stored in meters or millimeters.
    scale, scale_debug = _pick_translation_scale(
        ref_points=ref_points,
        res_points=res_points,
        transform=rescan_meta.transform,
        overlap_delta=float(args.overlap_delta),
        sample_size=int(args.scale_sample_size),
    )
    T = with_translation_scaled(rescan_meta.transform, scale=scale)

    res_points_aligned = apply_transform(res_points, T)

    # Downsample for stable runtime.
    ref_points_ds, ref_extras = voxel_downsample(
        ref_points,
        voxel_size=float(args.voxel_size),
        extra_arrays={"objectId": ref_object_ids},
    )
    res_points_ds, res_extras = voxel_downsample(
        res_points_aligned,
        voxel_size=float(args.voxel_size),
        extra_arrays={"objectId": res_object_ids},
    )
    ref_object_ids_ds = ref_extras["objectId"]
    res_object_ids_ds = res_extras["objectId"]

    max_radius = float(max(args.overlap_delta, args.heat_cap_factor * args.tau))

    nn_ref = nearest_neighbors_within_radius(ref_points_ds, res_points_ds, radius=max_radius)
    nn_res = nearest_neighbors_within_radius(res_points_ds, ref_points_ds, radius=max_radius)
    d_ref = nn_ref.distances
    d_res = nn_res.distances

    observed_ref = np.isfinite(d_ref)
    observed_res = np.isfinite(d_res)
    close_ref = observed_ref & (d_ref < float(args.overlap_delta))
    close_res = observed_res & (d_res < float(args.overlap_delta))

    overlap_ref = overlap_ratio(d_ref, threshold=float(args.overlap_delta))
    overlap_res = overlap_ratio(d_res, threshold=float(args.overlap_delta))
    overlap_mean = 0.5 * (overlap_ref + overlap_res)
    comparable_ratio_ref = float(np.mean(observed_ref)) if observed_ref.size else 0.0
    comparable_ratio_res = float(np.mean(observed_res)) if observed_res.size else 0.0

    reliable = overlap_mean >= float(args.overlap_min)
    gate_reason = ""
    if not reliable:
        gate_reason = f"overlap_mean={overlap_mean:.3f} < overlap_min={float(args.overlap_min):.3f}"

    # Cap inf distances for visualization/heat computation.
    d_ref_vis = np.where(np.isfinite(d_ref), d_ref, max_radius)
    d_res_vis = np.where(np.isfinite(d_res), d_res, max_radius)
    heat_ref = _heat_from_distance(d_ref_vis, tau=float(args.tau))
    heat_res = _heat_from_distance(d_res_vis, tau=float(args.tau))

    ref_labels = _load_semseg_labels(ref_semseg)
    res_labels = _load_semseg_labels(res_semseg)

    ref_stats = _compute_object_stats(
        object_ids=ref_object_ids_ds,
        distances=d_ref_vis,
        observed=observed_ref,
        tau=float(args.tau),
        min_support=int(args.min_object_support),
    )
    res_stats = _compute_object_stats(
        object_ids=res_object_ids_ds,
        distances=d_res_vis,
        observed=observed_res,
        tau=float(args.tau),
        min_support=int(args.min_object_support),
    )

    all_object_ids = sorted(set(ref_stats.keys()) | set(res_stats.keys()))
    rows: list[dict[str, Any]] = []
    for oid in all_object_ids:
        r = ref_stats.get(oid, {})
        s = res_stats.get(oid, {})
        score = max(float(r.get("ratio_changed_observed", 0.0)), float(s.get("ratio_changed_observed", 0.0)))
        if score <= 0.0:
            continue

        change_type, type_conf = _assign_change_type(
            object_id=oid,
            ref_stats=ref_stats,
            res_stats=res_stats,
            reliable=reliable,
            ref_points=ref_points_ds,
            res_points=res_points_ds,
            ref_object_ids=ref_object_ids_ds,
            res_object_ids=res_object_ids_ds,
            ref_observed=observed_ref,
            res_observed=observed_res,
            move_translation_min=float(args.move_translation_min),
            min_support_total=int(args.min_object_total),
        )

        rows.append(
            {
                "objectId": int(oid),
                "label_ref": ref_labels.get(int(oid), "unknown"),
                "label_rescan": res_labels.get(int(oid), "unknown"),
                "score": f"{score:.4f}",
                "type": change_type,
                "type_confidence": f"{type_conf:.2f}",
                "ref_support_total": int(r.get("support_total", 0)),
                "ref_support_observed": int(r.get("support_observed", 0)),
                "ref_ratio_changed_obs": f"{float(r.get('ratio_changed_observed', 0.0)):.4f}",
                "ref_p95_changed_dist": f"{float(r.get('p95_changed_distance', 0.0)):.4f}",
                "res_support_total": int(s.get("support_total", 0)),
                "res_support_observed": int(s.get("support_observed", 0)),
                "res_ratio_changed_obs": f"{float(s.get('ratio_changed_observed', 0.0)):.4f}",
                "res_p95_changed_dist": f"{float(s.get('p95_changed_distance', 0.0)):.4f}",
                "low_support": bool(r.get("is_low_support", True) and s.get("is_low_support", True)),
            }
        )

    rows.sort(key=lambda rr: float(rr["score"]), reverse=True)
    rows = rows[: int(args.top_k)]

    pair_id = f"{reference_id}__{rescan_id}"
    out_pair = out_root / "pairs" / pair_id
    out_pair.mkdir(parents=True, exist_ok=True)
    figures_dir = out_pair / "figures"
    if not args.skip_figures:
        figures_dir.mkdir(parents=True, exist_ok=True)

    alignment_payload = {
        "reference_scan_id": reference_id,
        "rescan_scan_id": rescan_id,
        "split": rescan_meta.split,
        "transform_storage": "column-major list (Eigen default); reshaped with order='F'",
        "translation_scale_applied": scale,
        "transform_rescan_to_reference": T.tolist(),
        "scale_debug": scale_debug,
    }
    (out_pair / "alignment.json").write_text(json.dumps(alignment_payload, indent=2) + "\n", encoding="utf-8")

    qc_payload = {
        "reference_scan_id": reference_id,
        "rescan_scan_id": rescan_id,
        "split": rescan_meta.split,
        "voxel_size_m": float(args.voxel_size),
        "tau_m": float(args.tau),
        "overlap_delta_m": float(args.overlap_delta),
        "nn_search_radius_m": max_radius,
        "translation_scale_applied": scale,
        "comparable_ratio_ref": comparable_ratio_ref,
        "comparable_ratio_rescan": comparable_ratio_res,
        "overlap_ref": overlap_ref,
        "overlap_rescan": overlap_res,
        "overlap_mean": overlap_mean,
        "reliable": bool(reliable),
        "gate_reason": gate_reason,
        "point_counts": {
            "ref_points": int(ref_points.shape[0]),
            "rescan_points": int(res_points.shape[0]),
            "ref_points_downsampled": int(ref_points_ds.shape[0]),
            "rescan_points_downsampled": int(res_points_ds.shape[0]),
        },
        "runtime_sec": float(time.time() - t0),
        "command": " ".join(sys.argv),
    }
    (out_pair / "qc.json").write_text(json.dumps(qc_payload, indent=2) + "\n", encoding="utf-8")

    _write_objects_csv(out_pair / "objects.csv", rows)

    if not args.skip_ply:
        # Export heatmap PLYs.
        colors_ref = _colors_from_heat(heat_ref, comparable=observed_ref)
        colors_res = _colors_from_heat(heat_res, comparable=observed_res)

        write_ply_ascii(
            out_pair / "heatmap_ref.ply",
            points=ref_points_ds,
            colors_rgb=colors_ref,
            extra_properties={
                "distance": d_ref_vis,
                "heat": heat_ref,
                "observed": observed_ref.astype(np.uint8),
                "close": close_ref.astype(np.uint8),
                "changed": (observed_ref & (d_ref_vis > float(args.tau))).astype(np.uint8),
                "objectId": ref_object_ids_ds.astype(np.int64),
            },
            comments=[f"tau_m={float(args.tau)}", f"overlap_delta_m={float(args.overlap_delta)}"],
        )
        write_ply_ascii(
            out_pair / "heatmap_rescan.ply",
            points=res_points_ds,
            colors_rgb=colors_res,
            extra_properties={
                "distance": d_res_vis,
                "heat": heat_res,
                "observed": observed_res.astype(np.uint8),
                "close": close_res.astype(np.uint8),
                "changed": (observed_res & (d_res_vis > float(args.tau))).astype(np.uint8),
                "objectId": res_object_ids_ds.astype(np.int64),
            },
            comments=[f"tau_m={float(args.tau)}", f"overlap_delta_m={float(args.overlap_delta)}"],
        )

    overlay_png = figures_dir / "overlay.png"
    heat_ref_png = figures_dir / "heatmap_reference.png"
    heat_res_png = figures_dir / "heatmap_rescan.png"
    if not args.skip_figures:
        _save_three_view_overlay_png(
            overlay_png,
            ref_points=ref_points_ds,
            res_points=res_points_ds,
            max_points=int(args.plot_max_points),
        )
        _save_three_view_heatmap_png(
            heat_ref_png,
            points=ref_points_ds,
            heat=heat_ref,
            comparable=observed_ref,
            max_points=int(args.plot_max_points),
            title_prefix="Reference",
        )
        _save_three_view_heatmap_png(
            heat_res_png,
            points=res_points_ds,
            heat=heat_res,
            comparable=observed_res,
            max_points=int(args.plot_max_points),
            title_prefix="Rescan",
        )

    # Report
    title = f"3RScan pair report: {reference_id} vs {rescan_id}"
    badge = "reliable" if reliable else "unreliable"
    notes = []
    if not reliable:
        notes.append(f"Marked as unreliable by gate: {gate_reason}")
    notes.append("Weak labels are treated as reference signals, not absolute ground truth.")
    notes.append(f"Weak-label counts: rigid={len(rescan_meta.rigid)} removed={len(rescan_meta.removed)} nonrigid={len(rescan_meta.nonrigid)}")

    summary = {
        "split": rescan_meta.split,
        "reference_scan_id": reference_id,
        "rescan_scan_id": rescan_id,
        "reliability": badge,
        "comparable_ratio_ref": f"{comparable_ratio_ref:.3f}",
        "comparable_ratio_rescan": f"{comparable_ratio_res:.3f}",
        "overlap_ref": f"{overlap_ref:.3f}",
        "overlap_rescan": f"{overlap_res:.3f}",
        "overlap_mean": f"{overlap_mean:.3f}",
        "voxel_size_m": float(args.voxel_size),
        "tau_m": float(args.tau),
        "overlap_delta_m": float(args.overlap_delta),
        "translation_scale_applied": scale,
        "runtime_sec": f"{qc_payload['runtime_sec']:.2f}",
    }
    if not args.skip_report:
        images: list[tuple[str, str]] = []
        if not args.skip_figures:
            images = [
                (str((Path("figures") / overlay_png.name).as_posix()), "Aligned overlay (three orthographic views)"),
                (str((Path("figures") / heat_ref_png.name).as_posix()), "Reference heatmap (three orthographic views)"),
                (str((Path("figures") / heat_res_png.name).as_posix()), "Rescan heatmap (three orthographic views)"),
            ]
        _write_report_html(
            out_pair / "report.html",
            title=title,
            summary=summary,
            object_rows=rows,
            images=images,
            notes=notes,
        )

    print("Wrote:", out_pair)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
