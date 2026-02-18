# Project Report

Technical writeup for the multi-session 3D change detection baseline. For setup and usage, see the main [`README`](../README.md).

## Problem

Given two 3D reconstructions of the same indoor space captured at different times, identify what changed — and be explicit about what you can't identify.

The hard part is not finding differences. Aligned point clouds always differ due to reconstruction noise, viewpoint variation, and partial coverage. The hard part is deciding which differences are real and which regions are reliable enough to make any claim about.

## Method

The pipeline runs six stages per scan pair. Every intermediate result is saved to disk.

**1. Alignment.** Apply the transform `T` from `3RScan.json` (column-major, Eigen convention). The translation component may be stored in meters or millimeters depending on the dataset release — the pipeline tests both scales against a subsampled overlap check and picks the better one.

**2. QC gate.** Compute bidirectional overlap ratios (fraction of points with a nearest neighbor within `overlap_delta`). The gate uses `min(overlap_ref, overlap_rescan)` rather than the mean — mean can mask asymmetric failures (e.g. ref 0.9, rescan 0.2 averages to 0.55, which looks passable but means the rescan barely covers the scene). If the min falls below `overlap_min`, the pair is marked unreliable. Unreliable pairs still produce outputs, but the report flags them and downstream consumers should treat the results as diagnostics.

**3. Comparable region.** A point is "comparable" if it has a nearest neighbor within the NN search radius. This is distinct from the overlap ratio in step 2: overlap is a per-pair scalar used to decide whether the pair is trustworthy at all; the comparable region is a per-point mask that controls where the pipeline is allowed to claim "unchanged". A pair can pass the overlap gate while still having localized unobserved patches — those patches show up as gray in the heatmaps, not as false positives.

**4. Change heatmaps.** Bidirectional NN distances on voxel-downsampled point clouds. Distances are normalized as `clip(d / tau, 0, 1)` for visualization. The NN search uses a uniform grid (cell size = search radius, 27-cell neighborhood lookup) rather than a KD-tree — it has zero external dependencies, gives exact results within the radius, and is fast enough for the point counts here (typically 50k–200k after downsampling).

**5. Object attribution.** Each point carries an `objectId` from `labels.instances.annotated.v2.ply`. Per object, the pipeline computes: fraction of observed points exceeding `tau`, the 95th-percentile changed-point distance, and support count. Objects are ranked by change score and assigned a type:

| Type | Heuristic |
|---|---|
| `appeared` | Present in rescan, absent or below support threshold in reference |
| `disappeared` | Present in reference, absent in rescan |
| `moved_rigid` | Both present, centroid shift exceeds threshold |
| `nonrigid_or_recon` | High change score but no large centroid shift — could be deformation or reconstruction noise |

**6. Reporting.** Per-pair HTML reports show: reliability badge, QC metrics, aligned overlay (three orthographic views), reference/rescan heatmaps, and the Top-K object table. Batch summaries aggregate weak-label hit rates.

**Default parameters.** The defaults (`voxel_size=0.02 m`, `tau=0.1 m`, `overlap_delta=0.05 m`) were selected via a `voxel_size x tau` grid ablation on train-split pairs — see `scripts/run_ablation.py`. `overlap_min=0.3` is conservative: it passes most well-captured pairs while catching severe coverage mismatches. All parameters are configurable via CLI flags.

## Evaluation

Weak labels from `3RScan.json` — `removed`, `nonrigid`, and `rigid` instance lists — serve as noisy reference signals. The primary metric is **Top-K hit rate**: does any of the top K predicted objects appear in the weak-label change set?

### Reference run (2026-03-01)

Dataset: `configs/pairs/showcase.json`, 60 pairs.

**Hit rates (reliable pairs, N=59):**

| hit@3 | hit@5 | hit@10 |
|---:|---:|---:|
| 0.831 | 0.983 | 1.000 |

**Size-bucket trend** (object size = `max(obb.axesLengths)` from `semseg.v2.json`, reliable pairs only):

Small objects (< 0.3 m) are harder to retrieve at low K. This is expected — voxel downsampling at 2 cm reduces point support for small objects, and geometry-only NN distances are noisier at that scale. By hit@10, retrieval is near-perfect across all sizes.

To reproduce:

```bash
python3 scripts/run_batch.py --datasets-root Datasets \
  --pairs-json configs/pairs/showcase.json \
  --out-root outputs/showcase \
  --exclude-labels wall,floor,ceiling --resume

python3 scripts/make_summary.py --datasets-root Datasets \
  --out-root outputs/showcase --write-md

python3 scripts/make_size_summary.py --datasets-root Datasets \
  --out-root outputs/showcase --reliable-only --write-md
```

## Failure modes

**Partial overlap.** The most common failure. If only a fraction of the scene is observed in both sessions, naive NN differencing flags large regions as changed. The QC gate catches this — in the reference run, the single intentionally low-overlap pair was correctly marked unreliable (overlap gate value 0.246, threshold 0.300).

When a pair is unreliable, the report header says so and prints the gate values. The object table is still populated but should not be read as trusted predictions.

**Reconstruction noise on large surfaces.** Walls and floors can show non-zero NN distances from scan-to-scan variation in surface reconstruction. The `--exclude-labels wall,floor,ceiling` flag mitigates this for the object ranking, but it does not affect the heatmaps themselves.

## Limitations

- **Geometry-only.** Without appearance or semantic cues, the pipeline cannot distinguish non-rigid physical change from reconstruction artifacts. The `nonrigid_or_recon` label is deliberately ambiguous about this.
- **No ICP refinement.** The metadata transform is used as-is. A bounded ICP step could improve alignment for borderline pairs but would need its own QC to avoid degrading good alignments.
- **Downsampling drops small objects.** At `voxel_size = 0.02 m`, objects smaller than ~5 cm may retain too few points for stable attribution.
