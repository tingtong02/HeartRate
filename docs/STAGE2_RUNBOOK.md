# Stage 2 Runbook

## Scope

Stage 2 first round only covers:

- beat detection
- IBI extraction
- IBI cleaning
- basic time-domain PRV/HRV features
- Stage 2 evaluation

This round does not include SQI, event detection, irregular pulse screening, respiration, or deep learning.

Stage 3 has not started.

## Environment

Reuse the current environment:

```bash
conda activate HeartRate_env
pip install -e .
```

## Dataset Setup

Reuse the existing local dataset configs:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Set the local dataset root:

```yaml
dataset:
  root_dir: /path/to/dataset
```

Then run:

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

## Stage 2 Defaults

- beat detection runs on Stage 1 style preprocessed PPG
- beats / IBI / features are computed independently inside long analysis windows
- default analysis window is `60 s`
- default analysis step is `30 s`
- Stage 2 keeps `stage1_frequency` as the current strongest HR baseline background

## Outputs

Stage 2 writes:

- `outputs/{dataset}_stage2_beats.csv`
- `outputs/{dataset}_stage2_features.csv`
- `outputs/{dataset}_stage2_metrics.csv`

Use `outputs/{dataset}_stage2_metrics.csv` as the primary summary file when reproducing Stage 2 results.

## Real First-Round Results

`PPG-DaLiA`
- beat detection: precision `0.4854`, recall `0.3510`, f1 `0.4074`, beat_count_error `26.8503`
- IBI error: MAE `65.6322 ms`, RMSE `88.3973 ms`, valid IBI pairs `18372`
- feature highlights:
  - `mean_ibi_ms`: MAE `153.5016`, Pearson `0.4771`
  - `median_ibi_ms`: MAE `129.7529`, Pearson `0.4655`
  - `sdnn_ms`: MAE `72.2186`, Pearson `0.0576`
  - `rmssd_ms`: MAE `89.1385`, Pearson `0.1360`
  - `ibi_cv`: MAE `0.0869`, Pearson `0.1005`

`WESAD`
- beat detection: precision `0.4768`, recall `0.3820`, f1 `0.4242`, beat_count_error `17.3936`
- IBI error: MAE `59.9759 ms`, RMSE `84.2615 ms`, valid IBI pairs `16033`
- feature highlights:
  - `mean_ibi_ms`: MAE `97.5454`, Pearson `0.6355`
  - `median_ibi_ms`: MAE `83.9194`, Pearson `0.6592`
  - `sdnn_ms`: MAE `63.7618`, Pearson `0.0982`
  - `rmssd_ms`: MAE `98.9918`, Pearson `-0.0234`
  - `ibi_cv`: MAE `0.0804`, Pearson `0.0459`

Current interpretation:

- Stage 2 first round is runnable, measurable, and reproducible
- mean / median IBI style outputs are relatively more usable
- variability features such as `sdnn_ms`, `rmssd_ms`, and `ibi_cv` are still weak
- the next improvement should focus on beat detection and IBI cleaning before adding more feature types

## Notes

- Beat detection is direct peak finding plus local refinement, not a Hilbert-envelope-first design.
- The first-round feature set is strictly time-domain only.
- Stage 3 has not started.
