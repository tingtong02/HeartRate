# Stage 0 Runbook

## 1. Create the environment

```bash
conda env create -f environment.yml
conda activate HeartRate_env
pip install -e .
```

## 2. Configure dataset paths

Public datasets are not committed to this repository. Download the dataset locally first, then create a local dataset config from the tracked template:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Local dataset configs are ignored by Git via `.gitignore`.

Edit the local dataset config you want to run:

- `configs/datasets/ppg_dalia.local.yaml`
- `configs/datasets/wesad.local.yaml`

Set:

```yaml
dataset:
  root_dir: /path/to/dataset
```

You may optionally limit subjects:

```yaml
dataset:
  subject_include: ["S1", "S2"]
```

## 3. Run the smoke path

The smoke script does not require public datasets. It uses synthetic signals to validate the full Stage 0 path:

```bash
python scripts/run_stage0_smoke.py
```

This checks:
- window generation
- ECG-derived reference HR
- Welch frequency baseline
- metric aggregation

## 4. Run the real dataset baseline

Example for PPG-DaLiA:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Example for WESAD:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Optional output saving is controlled in `configs/eval/hr_baseline.yaml` and `configs/base.yaml`.

## 5. Common failure cases

- `Dataset root does not exist`
  - Check `dataset.root_dir` in the local dataset config.
- `No pickle file found`
  - Confirm the subject directory contains the expected `Sx.pkl` file. Other sidecar files are ignored by Stage 0.
- `Missing required key 'BVP'` or `Missing required key 'ECG'`
  - Verify the pickle payload matches the official wrist/chest key layout.
- `pickle` decoding errors on Python 3
  - Stage 0 loaders already use `latin1` compatibility mode for the official dataset pickles.
- `No subjects available for evaluation`
  - Confirm `subject_include` is not filtering out all subjects.

## 6. Current boundaries

- Stage 0 only provides a minimal frequency-domain HR baseline.
- No SQI, beat detection, deep learning, or multitask pipeline is included in this round.
- ECG is used only to construct reference HR for evaluation.
- Tracked dataset configs intentionally keep `root_dir: ""`; set paths only in local `*.local.yaml` files.
