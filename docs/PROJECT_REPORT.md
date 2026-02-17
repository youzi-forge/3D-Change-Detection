# Project report: Multi-session 3D Change Detection Baseline (3RScan)

This document is a short, human-readable write-up meant for first-time readers of the repo.
It complements `README.md` (which focuses on usage).

## Problem framing

Multi-session 3D change detection asks: given two captures of the same environment at different times, what changed?
In practice, the harder question is often **verification of condition**:

- What can be confidently verified as unchanged?
- When are results unreliable due to limited overlap, drift, or reconstruction differences?
- Can change evidence be attributed to objects with an explainable “evidence chain”?

This repo implements a **CUDA-free**, geometry-only baseline on the 3RScan dataset with a strong emphasis on:

- explicit definitions (units, coordinate conventions, “comparable regions”)
- QC metrics and reliability gating
- object-level attribution and explainable change typing
- reproducible artifacts and per-pair reports

## Method summary

Given a reference scan and a rescan:

1) **Alignment**
- Use the metadata transform `T` from `3RScan.json` (parsed in column-major order).
- Automatically validate whether translation is stored in meters or millimeters by checking overlap under candidate scales.

2) **Comparable region**
- Define a point as *observed/comparable* if it has a nearest neighbor within a bounded search radius.
- Only report “unchanged” and “changed” ratios inside the comparable region.

3) **QC and reliability gate**
- Compute overlap ratios `overlap_ref` and `overlap_rescan` using a distance threshold `overlap_delta`.
- Default gate: `reliable = min(overlap_ref, overlap_rescan) >= overlap_min`.
- When unreliable, the report explicitly states the gate reason; downstream outputs are treated as diagnostics.

4) **Geometry change heatmaps**
- Compute bidirectional nearest-neighbor distances (reference->rescan and rescan->reference) within the search radius.
- Visualize normalized distances via `clip(distance / tau, 0..1)`; unobserved points are rendered in gray.

5) **Object-level attribution**
- Each point carries an `objectId` from `labels.instances.annotated.v2.ply`.
- Change evidence votes onto objects, producing a Top-K table with scores, support, and a lightweight change type:
  - `appeared`, `disappeared`, `moved_rigid`, `nonrigid_or_recon`, `unknown`

6) **Reports and summaries**
- Per pair: `report.html` (QC + overlay + heatmaps + Top objects).
- Per run: `summary.csv` and `summary.md` (weak-label Top-K hit rates).
- Optional: `size_summary.md` (weak-label hit rates bucketed by object size using OBB proxies).

## Results snapshot (reference run)

The pipeline is designed to be reproducible, but numeric results depend on the chosen pair list and parameters.
As an example, a reference run (2026-03-01) on the committed `configs/pairs/showcase.json` (60 pairs) produced:

- Reliable pairs: 59 / 60 (one intentionally low-overlap failure case)
- Weak-label Top-K hit rate (reliable pairs): hit@3 = 0.831, hit@5 = 0.983, hit@10 = 1.000

Size-bucket trends (reliable-only, size proxy = `max(obb.axesLengths)`):

- Smaller objects are harder to retrieve reliably at low K (consistent with a geometry-only baseline + downsampling).

To reproduce these numbers locally, run:

```bash
python3 scripts/run_batch.py --datasets-root Datasets --pairs-json configs/pairs/showcase.json --out-root outputs/showcase --exclude-labels wall,floor,ceiling --resume
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs/showcase --write-md
python3 scripts/make_size_summary.py --datasets-root Datasets --out-root outputs/showcase --reliable-only --write-md
```

## Representative cases

This repo includes a small set of representative qualitative cases:

- `configs/pairs/featured.json`

After running them, open:

- `outputs/featured/pairs/<reference>__<rescan>/report.html`

The report pages are the intended “portfolio artifact”: each one is a compact evidence chain that shows:

- whether the pair is reliable (and why)
- where the comparable region is (gray vs colored points)
- where geometric change evidence concentrates
- which object instances are most implicated and how they are typed

## Failure mode (why QC gating matters)

A common failure mode in multi-session scans is **partial overlap / coverage mismatch**:

- If only a small fraction of the scene is observed in both sessions, naive geometric differencing will report large “changes”.
- In the reports, this typically manifests as a large fraction of `appeared`/`disappeared` objects driven by unobserved regions.

This repo treats such cases as *unreliable* by default and keeps outputs as diagnostics rather than claims.

## Limitations and next steps

- This is a geometry-only baseline; it cannot reliably separate non-rigid change from reconstruction differences.
- The current baseline does not run ICP refinement by default; adding a bounded ICP step is plausible future work but must be QC-aware.
- Weak labels from `3RScan.json` are treated as reference signals, not absolute truth.

## Citation

If you use the 3RScan dataset, please cite:

```bibtex
@inproceedings{wald2019,
    title={RIO: 3D Object Instance Re-Localization in Changing Indoor Environments},
    author={Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, Matthias Niessner},
    booktitle={Proceedings IEEE International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```
