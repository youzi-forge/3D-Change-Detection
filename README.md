# Multi-session 3D Change Detection Baseline (3RScan)

Reproducible, **CUDA-free** baseline for **multi-session 3D change detection** and **condition verification** on the 3RScan dataset.

This project focuses on an evidence pipeline (QC + comparable regions + heatmaps + object-level attribution + reports),
not on training a large model.

For a higher-level write-up (methods, results snapshot, failure mode), see `docs/PROJECT_REPORT.md`.

## What this repo does

Given a reference scan and a rescan of the same environment, the pipeline:

- Aligns scans using the metadata transform `T` from `3RScan.json` (with automatic translation-scale handling: meters vs millimeters)
- Computes QC signals and applies a reliability gate (to handle partial overlap / drift)
- Defines the **comparable region** (only claim “unchanged” where both scans are comparable)
- Computes geometry-only change heatmaps (bidirectional NN distances)
- Attributes change evidence to object instances via per-point `objectId`
- Writes a 1-page HTML report per pair plus batch-level summary tables

This repo **does not** include 3RScan. You must download it yourself under their Terms of Use.

## What you get (outputs)

Per pair under `outputs/<run_name>/pairs/<reference>__<rescan>/`:

- `qc.json` and `alignment.json`
- `objects.csv` (Top-K objects with scores and explainable change typing)
- `report.html` with `figures/*.png` (overlay + heatmaps)
- Optional: `heatmap_ref.ply`, `heatmap_rescan.ply` (colored point clouds)

## Featured cases (qualitative)

This repo ships a small, representative set of cases (two reliable successes + one QC-gated failure) under:

- `configs/pairs/featured.json`

Run them and open the local report pages:

```bash
python3 scripts/run_batch.py --datasets-root Datasets --pairs-json configs/pairs/featured.json --out-root outputs/featured --exclude-labels wall,floor,ceiling --resume
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs/featured --write-md
```

Then open:

- `outputs/featured/pairs/<reference>__<rescan>/report.html`

Note: the included failure case is expected to be marked as `unreliable` by the overlap gate (partial overlap / coverage mismatch).
In that regime, heatmaps and Top Objects are diagnostics, not trustworthy change claims.

## Results snapshot (reference run)

As an example, a reference run (2026-03-01) on the committed `configs/pairs/showcase.json` (60 pairs) produced:

- Reliable pairs: 59 / 60
- Weak-label Top-K hit rate on reliable pairs: hit@3 = 0.831, hit@5 = 0.983, hit@10 = 1.000
- Size-bucket trend (reliable-only): smaller objects are harder to retrieve reliably at low K

You can reproduce these numbers locally with the `showcase.json` commands below and inspect:

- `outputs/showcase/summary.md`
- `outputs/showcase/size_summary.md`

## Dataset layout (expected)

Example layout that works out-of-the-box:

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

## Quickstart (data sanity check)

1) Put `3RScan.json` under `Datasets/` and scans under `Datasets/3RScan/<scanId>/`.

2) Run the local inspector (picks a valid train/validation pair if available):

```bash
python3 scripts/inspect_3rscan.py --datasets-root Datasets --write-smoke-config
```

This writes:
- `configs/pairs/smoke_pair.local.json`

## Run a single pair

```bash
python3 scripts/run_pair.py --datasets-root Datasets --pair-config configs/pairs/smoke_pair.local.json --exclude-labels wall,floor,ceiling
```

Artifacts are written under `outputs/pairs/<reference>__<rescan>/`.

## Run a batch and summarize

Run a small batch (default: picks pairs with the most weak-labeled changes from the chosen split):

```bash
python3 scripts/run_batch.py --datasets-root Datasets --split train --limit 20 --resume
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs --write-md
```

Run an explicit, reproducible list of pairs from a JSON file:

```bash
python3 scripts/run_batch.py --datasets-root Datasets --pairs-json configs/pairs/examples.json --out-root outputs/examples --exclude-labels wall,floor,ceiling --resume
python3 scripts/make_summary.py --datasets-root Datasets --out-root outputs/examples --write-md
```

Run the curated showcase set (60 pairs; includes a single intentionally-unreliable pair to demonstrate the QC gate):

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

## Size-bucket summary (weak labels)

Bucket weak-label GT objects by size using `obb.axesLengths` from the reference `semseg.v2.json`:

```bash
python3 scripts/make_size_summary.py --datasets-root Datasets --out-root outputs/showcase --reliable-only --write-md
```

## Ablations

Run a small `voxel_size x tau` ablation (metrics-only; skips PLY/figures/report):

```bash
python3 scripts/run_ablation.py --datasets-root Datasets --out-root outputs/ablation --split train --limit 20 --voxel-sizes 0.01,0.02,0.05 --taus 0.05,0.1,0.2 --resume
python3 scripts/make_ablation_table.py --ablation-root outputs/ablation --write-md
```

## Limitations and scope

- Geometry-only baseline: reconstruction noise and partial overlap can dominate without QC.
- Weak labels: `3RScan.json` change annotations are treated as reference signals, not absolute ground truth.
- Reports are designed for evidence and debugging, not as definitive change claims.

## Tests

```bash
python3 -m unittest discover -s tests
```
