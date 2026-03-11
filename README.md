# RSO Maneuver Intent World Model

Lightweight PyTorch implementation of an RSO behavior world model built around a Temporal Convolutional Network (TCN). The deterministic propagator remains authoritative for orbital state; the learned model predicts maneuver likelihood, maneuver class/purpose, delta-v bucket, and remaining propulsion estimate.

The repository vendors the base TCN implementation from `locuslab/TCN` under [`third_party/TCN`](./third_party/TCN) and adapts it for multivariate orbital time series plus static metadata.

## Layout

```text
configs/                     Example training configuration
data/raw/                    Cached Space-Track, CelesTrak, and SATCAT downloads
data/processed/              Prepared sequence windows (.npz + manifest)
scripts/                     CLI entrypoints for download, feature prep, training, and export
src/rso_world_model/         Package source
third_party/TCN/             Vendored Temporal Convolutional Network reference
tests/                       Basic shape and dataset tests
```

## Workflow

1. Download raw data:

```bash
python3 scripts/download_spacetrack_history.py --norad-ids 25544 43013
python3 scripts/download_celestrak_gp.py --groups active starlink geo
python3 scripts/download_satcat.py
```

`download_spacetrack_history.py` expects `SPACETRACK_ID` and `SPACETRACK_PASSWORD` in the environment.

For larger Space-Track historical backfills, prefer archive-style ingestion instead of aggressive per-object `GP_History` crawling. The repo includes:

- `scripts/download_spacetrack_bulk.py` for resumable current-GP snapshots plus cautious per-object history pulls
- `scripts/download_spacetrack_historical_archives.py` for project-local download of authenticated historical archive assets when you have the archive URLs

Recommended split:

- use current `GP` snapshots for live/current RSO coverage
- use per-object `GP_History` only for targeted experiments or small cohorts
- use archive downloads for large historical backfills

2. Build prepared temporal windows:

```bash
python3 scripts/build_feature_dataset.py \
  --spacetrack-dir data/raw/spacetrack \
  --celestrak-dir data/raw/celestrak \
  --satcat-path data/raw/satcat/satcat.csv \
  --output-dir data/processed/train_windows
```

3. Train the model:

```bash
python3 scripts/train_world_model.py --config configs/train.example.yaml
```

4. Export to ONNX:

```bash
python3 scripts/export_onnx.py \
  --checkpoint artifacts/checkpoints/best.pt \
  --config configs/train.example.yaml \
  --output artifacts/export/rso_world_model.onnx
```

## Notes

- Environmental Sun/Moon features use Skyfield when a local ephemeris file is supplied. Without one, the pipeline emits NaNs plus mask flags instead of failing.
- The maneuver labels produced by `build_feature_dataset.py` are heuristic bootstrapping labels derived from propagation residuals and orbital deltas. They are sufficient for initial experimentation but should be replaced with analyst-reviewed labels when available.
- The default TCN stack is 5 layers, kernel size 3, and 1.1M parameters for the default feature dimensions, keeping the model within Jetson Orin NX deployment constraints.
