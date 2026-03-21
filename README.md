# HeartRate_CNN

HeartRate_CNN is a public-dataset PPG heart rate analysis project.  
The current repository state is focused on **Stage 0**: building a minimal, reproducible data loading and evaluation foundation before moving on to stronger heart-rate algorithms.

## Stage 0 Status

Stage 0 currently implements:

- unified subject-level data loading for `PPG-DaLiA` and `WESAD`
- unified data structures for subject records and window samples
- minimal preprocessing: resampling, common-duration alignment, and sliding-window segmentation
- subject-wise train/test split
- ECG-derived window-level reference HR
- a minimal frequency-domain HR baseline using Welch PSD
- basic evaluation metrics
- smoke test and pytest coverage

Stage 0 does not include:

- CNN / TCN / deep learning training
- SQI
- beat detection / IBI / PRV / HRV
- event detection
- respiration estimation

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

## Dataset Notes

Stage 0 assumes the official subject-pickle style layout for both datasets.

- `PPG-DaLiA`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- `WESAD`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- official pickles are loaded with Python 3 `latin1` compatibility
- reference HR is unified at the window level by reconstructing it from chest ECG peaks

For more detail, see:

- [docs/STAGE0_RUNBOOK.md](.docs/STAGE0_RUNBOOK.md)
- [docs/DATASETS.md](.docs/DATASETS.md)

## Current Limitations

- Stage 0 provides only a minimal frequency-domain HR baseline.
- Current baseline quality is intended for reproducible validation, not final performance.
- Dataset configs in the repository are templates only; local paths must be set in ignored `*.local.yaml` files.
- If a dataset variant changes pickle keys or file layout, the loader may need a small compatibility update.
