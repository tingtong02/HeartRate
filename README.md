# HeartRate_CNN

HeartRate_CNN is a public-dataset PPG heart rate analysis project.  
The current repository state now includes **Stage 2 first-round beat / IBI / time-domain PRV-HRV work** on top of the Stage 0 and Stage 1 foundations.

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
- a Stage 2 beat detection path based on direct peak finding over Stage 1 style preprocessed PPG
- a Stage 2 IBI extraction and cleaning path
- a Stage 2 basic time-domain PRV/HRV feature path
- a Stage 2 evaluation script for beat / IBI / feature comparison
- basic evaluation metrics
- smoke test and pytest coverage

Still not included:

- CNN / TCN / deep learning training
- SQI
- event detection
- respiration estimation
- frequency-domain or nonlinear HRV features
- irregular pulse screening

Stage 3 has **not** started yet.

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

Run the Stage 2 evaluation on PPG-DaLiA:

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 2 evaluation on WESAD:

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

To reproduce the Stage 2 evaluation:

- create local dataset configs the same way as Stage 0 / Stage 1
- keep the default subject-wise split seed unless you are intentionally running a new comparison
- run `run_stage2_baseline.py` once per dataset
- use `outputs/{dataset}_stage2_metrics.csv` as the source of record

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

## Stage 2 Scope

Stage 2 first round only includes:

- beat detection
- IBI extraction
- IBI cleaning
- basic time-domain PRV/HRV features

Stage 2 still does not include:

- SQI
- event detection
- irregular pulse screening
- frequency-domain HRV
- nonlinear HRV
- deep learning models

## Stage 2 Results

First-round real evaluation results:

`PPG-DaLiA`
- beat detection: precision `0.4854`, recall `0.3510`, f1 `0.4074`, beat_count_error `26.8503`
- IBI error: MAE `65.6322 ms`, RMSE `88.3973 ms`, valid IBI pairs `18372`
- feature highlights:
  - `mean_ibi_ms`: MAE `153.5016`, Pearson `0.4771`
  - `median_ibi_ms`: MAE `129.7529`, Pearson `0.4655`
  - `mean_hr_bpm_from_ibi`: MAE `19.8757`, Pearson `0.2717`
  - `sdnn_ms`: MAE `72.2186`, Pearson `0.0576`
  - `rmssd_ms`: MAE `89.1385`, Pearson `0.1360`
  - `ibi_cv`: MAE `0.0869`, Pearson `0.1005`

`WESAD`
- beat detection: precision `0.4768`, recall `0.3820`, f1 `0.4242`, beat_count_error `17.3936`
- IBI error: MAE `59.9759 ms`, RMSE `84.2615 ms`, valid IBI pairs `16033`
- feature highlights:
  - `mean_ibi_ms`: MAE `97.5454`, Pearson `0.6355`
  - `median_ibi_ms`: MAE `83.9194`, Pearson `0.6592`
  - `mean_hr_bpm_from_ibi`: MAE `12.2677`, Pearson `0.4882`
  - `sdnn_ms`: MAE `63.7618`, Pearson `0.0982`
  - `rmssd_ms`: MAE `98.9918`, Pearson `-0.0234`
  - `ibi_cv`: MAE `0.0804`, Pearson `0.0459`

Current Stage 2 takeaway:

- Stage 2 first round is already runnable, measurable, and reproducible
- mean / median IBI related outputs are relatively more usable than variability features
- `sdnn_ms`, `rmssd_ms`, and `ibi_cv` are still weak in this first-round implementation
- the next Stage 2 improvement should focus on beat detection and IBI cleaning, not on expanding the feature set first

## Dataset Notes

Stage 0 assumes the official subject-pickle style layout for both datasets.

- `PPG-DaLiA`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- `WESAD`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- official pickles are loaded with Python 3 `latin1` compatibility
- reference HR is unified at the window level by reconstructing it from chest ECG peaks

For more detail, see:

- [docs/STAGE0_RUNBOOK.md](.docs/STAGE0_RUNBOOK.md)
- [docs/STAGE1_RUNBOOK.md](.docs/STAGE1_RUNBOOK.md)
- [docs/STAGE2_RUNBOOK.md](.docs/STAGE2_RUNBOOK.md)
- [docs/DATASETS.md](.docs/DATASETS.md)

## Current Limitations

- Stage 0 provides only a minimal frequency-domain HR baseline.
- Stage 1 is still a lightweight classical-signal baseline system, not a final robust estimator.
- Stage 1 frequency is currently the best-performing path; fusion is mainly improving coverage and robustness relative to Stage 0, not surpassing the frequency chain yet.
- Stage 2 first round is limited to beat / IBI / basic time-domain PRV-HRV only.
- Stage 2 first round is more reliable for mean / median IBI style summaries than for variability features such as `sdnn_ms`, `rmssd_ms`, and `ibi_cv`.
- Current baseline quality is intended for reproducible validation, not final performance.
- Dataset configs in the repository are templates only; local paths must be set in ignored `*.local.yaml` files.
- If a dataset variant changes pickle keys or file layout, the loader may need a small compatibility update.
