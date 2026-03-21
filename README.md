# HeartRate_CNN

HeartRate_CNN is a public-dataset PPG heart rate analysis project.  
The current repository state now includes **Stage 1 first-round HR baseline work** on top of the Stage 0 data and evaluation foundation.

## Current Status

Implemented so far:

- unified subject-level data loading for `PPG-DaLiA` and `WESAD`
- unified data structures for subject records and window samples
- minimal preprocessing: resampling, common-duration alignment, and sliding-window segmentation
- subject-wise train/test split
- ECG-derived window-level reference HR
- a minimal frequency-domain HR baseline using Welch PSD
- a Stage 1 enhanced preprocessing path
- a Stage 1 frequency chain based on enhanced Welch analysis
- a Stage 1 time-domain chain based on pulse peak detection
- a minimal rule-based Stage 1 fusion path
- a Stage 1 comparison script that evaluates Stage 0 baseline, Stage 1 frequency, Stage 1 time, and Stage 1 fusion on the same windows
- basic evaluation metrics
- smoke test and pytest coverage

Still not included:

- CNN / TCN / deep learning training
- SQI
- beat detection / IBI / PRV / HRV
- event detection
- respiration estimation

Stage 2 has **not** started yet.

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate HeartRate_env
pip install -e .
```

## Dataset Setup

Public datasets are **not** committed with this repository. You need to download them manually.

Recommended local placement:

```text
HeartRate_CNN/
  dataset/
    PPG_DaLiA/
    WESAD/
```

The `dataset/` directory is ignored by Git.

Tracked dataset config templates keep `root_dir: ""`.  
Create local configs first:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Then edit the local config you want to use:

```yaml
dataset:
  root_dir: /path/to/dataset
```

Optional subject filtering:

```yaml
dataset:
  subject_include: ["S1", "S2"]
```

`*.local.yaml` files are ignored by Git.

## How To Run

Run tests:

```bash
pytest
```

Run the synthetic smoke path:

```bash
python scripts/run_stage0_smoke.py
```

Run the real baseline on PPG-DaLiA:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the real baseline on WESAD:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 1 comparison on PPG-DaLiA:

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 1 comparison on WESAD:

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

To reproduce the Stage 1 comparison fairly:

- use the same dataset config style as Stage 0
- keep the default subject-wise split seed unless you are intentionally running a new comparison
- compare all four methods from the same `run_stage1_baseline.py` execution
- use the generated `outputs/{dataset}_stage1_metrics.csv` as the source of record

## Stage 1 Results

First-round real comparison results:

`PPG-DaLiA`
- `stage0_baseline`: MAE `23.7573`, RMSE `34.8267`, Pearson `0.3895`
- `stage1_frequency`: MAE `9.5227`, RMSE `18.8806`, Pearson `0.7679`
- `stage1_time`: MAE `23.5926`, RMSE `31.5837`, Pearson `0.3310`
- `stage1_fusion`: MAE `21.8655`, RMSE `30.3448`, Pearson `0.4220`

`WESAD`
- `stage0_baseline`: MAE `19.6101`, RMSE `31.6620`, Pearson `0.2644`
- `stage1_frequency`: MAE `10.8137`, RMSE `22.3642`, Pearson `0.5348`
- `stage1_time`: MAE `18.6811`, RMSE `28.6999`, Pearson `0.1262`
- `stage1_fusion`: MAE `16.4980`, RMSE `27.1314`, Pearson `0.3265`

Current takeaway:

- the strongest first-round Stage 1 method is `stage1_frequency`
- `stage1_fusion` improves over `stage0_baseline` on both datasets
- `stage1_fusion` does not yet beat `stage1_frequency`

## Dataset Notes

Stage 0 assumes the official subject-pickle style layout for both datasets.

- `PPG-DaLiA`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- `WESAD`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- official pickles are loaded with Python 3 `latin1` compatibility
- reference HR is unified at the window level by reconstructing it from chest ECG peaks

For more detail, see:

- [docs/STAGE0_RUNBOOK.md](.docs/STAGE0_RUNBOOK.md)
- [docs/STAGE1_RUNBOOK.md](.docs/STAGE1_RUNBOOK.md)
- [docs/DATASETS.md](.docs/DATASETS.md)

## Current Limitations

- Stage 0 provides only a minimal frequency-domain HR baseline.
- Stage 1 is still a lightweight classical-signal baseline system, not a final robust estimator.
- Stage 1 frequency is currently the best-performing path; fusion is mainly improving coverage and robustness relative to Stage 0, not surpassing the frequency chain yet.
- Current baseline quality is intended for reproducible validation, not final performance.
- Dataset configs in the repository are templates only; local paths must be set in ignored `*.local.yaml` files.
- If a dataset variant changes pickle keys or file layout, the loader may need a small compatibility update.
