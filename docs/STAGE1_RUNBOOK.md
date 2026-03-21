# Stage 1 Runbook

## Scope

Stage 1 only covers a stronger heart-rate main pipeline:

- enhanced preprocessing
- Stage 1 frequency chain
- Stage 1 time-domain chain
- minimal rule-based fusion
- fair comparison against the Stage 0 baseline

This round does not include SQI, beat detection, IBI/HRV, event detection, respiration, or deep learning.

Stage 2 has not started.

## Environment

Reuse the Stage 0 setup:

```bash
conda activate HeartRate_env
pip install -e .
```

## Dataset Setup

Reuse the Stage 0 local dataset configs:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Set the local dataset root:

```yaml
dataset:
  root_dir: /path/to/dataset
```

## Run Stage 1 Comparison

PPG-DaLiA:

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

WESAD:

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

For reproducible comparisons:

- keep the same dataset config and split seed when comparing runs
- use the single Stage 1 script output so all four methods are evaluated on the same windows
- treat `outputs/{dataset}_stage1_metrics.csv` as the primary comparison summary

## Outputs

Stage 1 writes:

- `outputs/{dataset}_stage1_predictions.csv`
- `outputs/{dataset}_stage1_metrics.csv`

The comparison includes:

- `stage0_baseline`
- `stage1_frequency`
- `stage1_time`
- `stage1_fusion`

## Real First-Round Results

`PPG-DaLiA`
- `stage0_baseline`: MAE `23.7573`, RMSE `34.8267`, MAPE `22.5490`, Pearson `0.3895`
- `stage1_frequency`: MAE `9.5227`, RMSE `18.8806`, MAPE `9.6125`, Pearson `0.7679`
- `stage1_time`: MAE `23.5926`, RMSE `31.5837`, MAPE `21.8004`, Pearson `0.3310`
- `stage1_fusion`: MAE `21.8655`, RMSE `30.3448`, MAPE `20.3324`, Pearson `0.4220`

`WESAD`
- `stage0_baseline`: MAE `19.6101`, RMSE `31.6620`, MAPE `18.9734`, Pearson `0.2644`
- `stage1_frequency`: MAE `10.8137`, RMSE `22.3642`, MAPE `10.0833`, Pearson `0.5348`
- `stage1_time`: MAE `18.6811`, RMSE `28.6999`, MAPE `18.9044`, Pearson `0.1262`
- `stage1_fusion`: MAE `16.4980`, RMSE `27.1314`, MAPE `15.6883`, Pearson `0.3265`

Current interpretation:

- `stage1_frequency` is the strongest first-round method
- `stage1_fusion` improves on `stage0_baseline`
- `stage1_fusion` does not yet outperform `stage1_frequency`

## Notes

- Frequency uses an enhanced Welch chain only. Stage 1 does not introduce STFT.
- The default Stage 1 bandpass is configurable and starts at `0.6-3.5 Hz`.
- Fusion is rule-based and intentionally minimal for reproducibility.
- This repository is still at Stage 1 for HR baseline work; no Stage 2 beat-level pipeline has been started.
