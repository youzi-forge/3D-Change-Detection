# Multi-session 3D Change Detection Baseline (3RScan)

Reproducible, **CUDA-free** baseline pipeline for **multi-session 3D change detection** on the 3RScan dataset:

- Metadata-based alignment (`T` from `3RScan.json`) with automatic translation-scale handling (m vs mm)
- QC metrics + reliability gate (to handle partial overlap / drift); default gate uses `min(overlap_ref, overlap_rescan) >= overlap_min`
- Comparable-region definition (only claim “unchanged” where both scans are comparable)
- Geometry-only change heatmaps + object-level change attribution
- Weak-label Top-K evaluation + size-bucket analysis (OBB proxy)
- Per-pair 1-page HTML reports + summary tables (ablation-friendly)

This repo **does not** include the dataset. You must download 3RScan yourself under their Terms of Use.

## Dataset layout (expected)

Example layout that works out-of-the-box with the scripts below:

```
Datasets/
  3RScan.json
  3RScan/
    <scanId>/
      labels.instances.annotated.v2.ply
      semseg.v2.json
      mesh.refined.v2.obj
      mesh.refined.mtl
      mesh.refined_0.png
      sequence.zip
```

Notes:
- The dataset is **flat by scanId**. Reference↔rescan relationships live in `3RScan.json`.
- `labels.instances.annotated.v2.ply` can contain `objectId==0` (background/unlabeled). This pipeline excludes it from "Top Objects".
- Some **test rescans** may not include semantic files; this pipeline targets **train/validation** for object-level attribution.
  (In `3RScan.json`, the split field is typically `train`, `validation`, `test`.)

## Quickstart (data sanity check)

1) Put `3RScan.json` under `Datasets/` and scans under `Datasets/3RScan/<scanId>/`.

2) Run the local inspector (picks a valid train/validation reference+rescan pair if available):

```bash
python3 scripts/inspect_3rscan.py --datasets-root Datasets --write-smoke-config
```

This writes a local smoke-pair config:

- `configs/pairs/smoke_pair.local.json`

## Run a single pair

If you have a local smoke pair config:

```bash
python3 scripts/run_pair.py --datasets-root Datasets --pair-config configs/pairs/smoke_pair.local.json
```

To keep the Top Objects table focused on non-structural instances:

```bash
python3 scripts/run_pair.py --datasets-root Datasets --pair-config configs/pairs/smoke_pair.local.json --exclude-labels wall,floor,ceiling
```

Artifacts are written under `outputs/pairs/<reference>__<rescan>/`:
- `qc.json`, `alignment.json`
- `heatmap_ref.ply`, `heatmap_rescan.ply`
- `objects.csv`
- `report.html` (with `figures/*.png`)

## Run a batch and summarize

Run a small batch (default: picks pairs with the most weak-labeled changes from the chosen split):

```bash
python3 scripts/run_batch.py --datasets-root Datasets --split train --limit 20 --resume
```

Run an explicit, reproducible list of pairs from a JSON file:

```bash
python3 scripts/run_batch.py --datasets-root Datasets --pairs-json configs/pairs/examples.json --out-root outputs/examples --exclude-labels wall,floor,ceiling --resume
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs/examples --write-md
```

Run the curated showcase set (includes a single intentionally-unreliable pair to demonstrate the QC gate):

```bash
python3 scripts/run_batch.py --datasets-root Datasets --pairs-json configs/pairs/showcase.json --out-root outputs/showcase --exclude-labels wall,floor,ceiling --resume
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs/showcase --write-md
python3 scripts/make_size_summary.py --datasets-root Datasets --out-root outputs/showcase --reliable-only --write-md
```

Pick a shortlist of representative reliable/unreliable cases for qualitative review:

```bash
python3 scripts/make_hero_list.py --datasets-root Datasets --out-root outputs/showcase --write-md
```

This writes `outputs/showcase/hero_candidates.md` with report paths and key QC stats.

To ensure coverage across multiple environments, constrain selection by reference scan id:

```bash
python3 scripts/run_batch.py --datasets-root Datasets --split train --limit 20 --max-per-reference 2 --resume
```

To keep Top Objects focused on non-structural instances:

```bash
python3 scripts/run_batch.py --datasets-root Datasets --split train --limit 20 --max-per-reference 2 --exclude-labels wall,floor,ceiling --resume
```

Then generate a summary table and a short Markdown report:

```bash
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs --write-md
```

## Size-bucket summary (weak labels)

Bucket weak-label GT objects by size using `obb.axesLengths` from the reference `semseg.v2.json`:

```bash
python3 scripts/make_size_summary.py --datasets-root Datasets --out-root outputs --reliable-only --write-md
```

## Ablations

Run a small `voxel_size x tau` ablation (metrics-only; skips PLY/figures/report):

```bash
python3 scripts/run_ablation.py --datasets-root Datasets --out-root outputs/ablation --split train --limit 20 --voxel-sizes 0.01,0.02,0.05 --taus 0.05,0.1,0.2 --resume
```

Aggregate per-setting summaries into one table:

```bash
python3 scripts/make_ablation_table.py --ablation-root outputs/ablation --write-md
```

## Tests

```bash
python3 -m unittest discover -s tests
```
