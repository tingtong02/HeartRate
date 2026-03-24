# Usage Guide

## What This Repository Does

`HeartRate_CNN` is a public-dataset physiological analysis repository built around wrist PPG.

Current capabilities:

- Stage 0: unified loading, alignment, ECG-backed HR references, smoke path
- Stage 1: short-window HR estimation
- Stage 2: beat detection, IBI cleaning, time-domain PRV features
- Stage 3: quality-aware HR gating and a robust HR policy prototype
- Stage 4: event detection, irregular pulse screening, anomaly scoring, unified suspiciousness
- Stage 5: CNN-based respiration estimation plus a multitask output interface that carries HR / quality / event / anomaly context forward

The repository is CPU-first and public-data-only. It is not a clinical diagnosis system.

## Environment Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate HeartRate_env
pip install -e .
```

Key dependencies:

- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `PyWavelets`
- `joblib`
- `pytest`
- `torch` for Stage 5

Run tests:

```bash
pytest -q
```

If you prefer not to activate the environment in your shell, use `conda run -n HeartRate_env ...` instead.

## Dataset Setup

Supported datasets:

- `PPG-DaLiA`
- `WESAD`

Expected workflow:

1. Download the public datasets locally.
2. Keep dataset roots outside Git-tracked source files.
3. Point a local dataset config at the local path.

Tracked dataset templates:

- `configs/datasets/ppg_dalia.yaml`
- `configs/datasets/wesad.yaml`

Current workspace also includes:

- `configs/datasets/ppg_dalia.local.yaml`
- `configs/datasets/wesad.local.yaml`

Those `.local.yaml` files currently contain machine-specific absolute paths for this workspace. On another machine, update them or copy from the tracked template.

Typical pattern:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Then edit:

```yaml
dataset:
  root_dir: /path/to/dataset
  subject_include: null
```

Optional subject filtering:

```yaml
dataset:
  subject_include: ["S1", "S2"]
```

The repository expects official pickle payloads with:

- wrist `BVP`
- chest `ECG`
- optional wrist `ACC`
- optional chest `Resp`

## Repository Workflow Overview

The stage flow is cumulative:

1. Stage 1 estimates HR on `8 s / 2 s` windows.
2. Stage 2 builds beat / IBI / PRV support on longer analysis windows.
3. Stage 3 adds signal quality gating and a robust HR policy on top of Stage 1.
4. Stage 4 builds event / irregular / anomaly outputs on top of Stage 3-aware windows.
5. Stage 5 builds respiration estimation and a multitask interface on top of Stage 4 context.

High-level dependency map:

- Stage 0 supports all later stages.
- Stage 1 is the base HR estimation layer.
- Stage 2 supports beat-level analysis, Stage 3 beat-quality logic, Stage 4 irregular screening features, and Stage 5 respiration surrogates.
- Stage 3 supports Stage 4 and Stage 5 quality-aware context.
- Stage 4 supports Stage 5 multitask context.

## How To Run Each Stage

### Stage 0

Synthetic smoke test:

```bash
python scripts/run_stage0_smoke.py
```

Real baseline:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Notes:

- `configs/eval/hr_baseline.yaml` currently sets `output.save_csv: false`
- Stage 0 primarily prints metrics; it only writes `outputs/{dataset}_stage0_predictions.csv` if CSV saving is enabled

### Stage 1

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Primary outputs:

- `outputs/{dataset}_stage1_predictions.csv`
- `outputs/{dataset}_stage1_metrics.csv`

### Stage 2

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Primary outputs:

- `outputs/{dataset}_stage2_beats.csv`
- `outputs/{dataset}_stage2_features.csv`
- `outputs/{dataset}_stage2_metrics.csv`
- `outputs/{dataset}_stage2_beat_quality.csv`
- `outputs/{dataset}_stage2_beat_quality_sweep.csv`

### Stage 3

Rule-based baseline:

```bash
python scripts/run_stage3_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage3_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Enhanced / ML / robust comparison:

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Primary outputs:

- `outputs/{dataset}_stage3_predictions.csv`
- `outputs/{dataset}_stage3_metrics.csv`
- `outputs/{dataset}_stage3_enhanced_predictions.csv`
- `outputs/{dataset}_stage3_enhanced_metrics.csv`
- threshold / operating-point / policy sweep CSVs

### Stage 4

Prepare reusable Stage 4 caches:

```bash
python scripts/prepare_stage4_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 4A event-only run:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 4B irregular-only run:

```bash
python scripts/run_stage4_irregular_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Full Stage 4 canonical run:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Validation-only Stage 4 run:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --output-scope validation \
  --output-label my_stage4_run
```

Force cache rebuild:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --rebuild-cache
```

### Stage 5

Prepare reusable Stage 5 window packages:

```bash
python scripts/prepare_stage5_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Tune and save only the Stage 5 CNN checkpoint:

```bash
python scripts/run_stage5_train_cnn.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Full Stage 5 canonical run:

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Validation-only Stage 5 run:

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --output-scope validation \
  --output-label stage5_analysis
```

Equivalent commands work for `WESAD` by swapping the dataset config path.

## How To Interpret Outputs

### Canonical Vs Validation Vs Cache

Canonical outputs:

- live in `outputs/`
- use unsuffixed filenames
- are the source of record

Validation or analysis outputs:

- live in `outputs/validation/<label>/`
- must not be confused with canonical outputs

Cache artifacts:

- Stage 4: `outputs/cache/stage4/<dataset>/...`
- Stage 5: `outputs/cache/stage5/<dataset>/...`
- are reusable intermediates, not results

### Key Canonical Files

Stage 1:

- `outputs/{dataset}_stage1_predictions.csv`
- `outputs/{dataset}_stage1_metrics.csv`

Stage 2:

- `outputs/{dataset}_stage2_beats.csv`
- `outputs/{dataset}_stage2_features.csv`
- `outputs/{dataset}_stage2_metrics.csv`

Stage 3:

- `outputs/{dataset}_stage3_predictions.csv`
- `outputs/{dataset}_stage3_metrics.csv`
- `outputs/{dataset}_stage3_enhanced_predictions.csv`
- `outputs/{dataset}_stage3_enhanced_metrics.csv`

Stage 4:

- `outputs/{dataset}_stage4_event_predictions.csv`
- `outputs/{dataset}_stage4_event_metrics.csv`
- `outputs/{dataset}_stage4_irregular_predictions.csv`
- `outputs/{dataset}_stage4_irregular_metrics.csv`
- `outputs/{dataset}_stage4_anomaly_predictions.csv`
- `outputs/{dataset}_stage4_anomaly_metrics.csv`
- `outputs/{dataset}_stage4_full_predictions.csv`
- `outputs/{dataset}_stage4_full_metrics.csv`

Stage 5:

- `outputs/{dataset}_stage5_predictions.csv`
- `outputs/{dataset}_stage5_metrics.csv`
- `outputs/{dataset}_stage5_tuning_results.csv`
- `outputs/models/stage5/{dataset}_stage5_resp_multitask_cnn_v1_best.pt`
- `outputs/models/stage5/{dataset}_stage5_resp_multitask_cnn_v1_best_config.json`

### Representative Schema Conventions

Stage 3 enhanced predictions:

- one row per `8 s` window
- include rule, ML, motion-refined, robust-policy, and DWT branch fields

Stage 4 full predictions:

- one row per `8 s` window
- include Stage 3 context plus:
  - `hr_event_flag`
  - `irregular_pulse_flag`
  - `anomaly_score`
  - `stage4_suspicion_flag`
  - `stage4_suspicion_score`

Stage 5 predictions:

- one row per respiration window
- include:
  - `resp_rate_ref_bpm`
  - `resp_rate_pred_bpm`
  - `resp_confidence`
  - `resp_validity_flag`
  - carried-forward Stage 4 context
  - Stage 2 support summaries

## How To Use Stage 4 Correctly

Stage 4 has four conceptual layers:

- Stage 4A events:
  - quality-gated rule-based tachycardia, bradycardia, and abrupt-change episodes
- Stage 4B irregular pulse screening:
  - quality-gated, feature-based screening
  - default model is `hist_gbdt_irregular`
- Stage 4C anomaly:
  - quality-gated `IsolationForest` anomaly score
- Stage 4 full:
  - unified row-wise suspiciousness and comparison outputs

What Stage 4 does mean:

- richer physiological interpretation on top of Stage 3
- auditable event / screening / anomaly outputs
- better stratification of suspicious segments than quality-only labels alone

What Stage 4 does not mean:

- clinical arrhythmia diagnosis
- true rhythm labels
- a proven universally better ranking layer than Stage 3-only quality suspiciousness

Important interpretation rule:

- treat the anomaly layer as the strongest standalone Stage 4 signal family
- treat the unified suspiciousness layer as an auditable stratification layer unless you produce new evidence showing stronger ranking performance

## How To Use Stage 5 Correctly

Stage 5 adds respiration, not a replacement HR model.

Default Stage 5 model:

- `stage5_resp_multitask_cnn_v1`

Label source:

- chest `Resp` waveform already present in both supported datasets

Reference RR derivation:

- respiratory-band filter
- spectral RR estimate
- breath-peak RR estimate
- validity check requiring sufficient support and agreement

Stage 5 workflow:

1. Build or reuse Stage 4 context and Stage 5 window packages.
2. Run tuning.
3. Train the selected CNN candidate.
4. Write Stage 5 predictions and metrics.
5. Save the best checkpoint and config.

How Stage 5 relates to HR:

- it carries forward Stage 4 HR / quality / event / anomaly outputs
- it does not retrain the HR stack
- HR carry-forward match metrics in `stage5_metrics.csv` should remain `1.0`

How to interpret Stage 5 confidence:

- `resp_confidence` is the CNN validity-head probability-like output
- `resp_validity_flag` is thresholded from that confidence
- use `high_quality_ref_valid` and `predicted_valid` subsets in `stage5_metrics.csv` when judging practical usability

## Common Workflows

### I Just Want To Run Tests

```bash
pytest -q
```

### I Want To Reproduce Canonical Stage 3 Outputs

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Then inspect:

- `outputs/ppg_dalia_stage3_enhanced_predictions.csv`
- `outputs/ppg_dalia_stage3_enhanced_metrics.csv`

### I Want To Run Stage 4 Without Touching Canonical Outputs

```bash
python scripts/prepare_stage4_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --output-scope validation \
  --output-label stage4_dev
```

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --output-scope validation \
  --output-label stage4_dev
```

### I Want To Train And Evaluate Stage 5

```bash
python scripts/prepare_stage5_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

If you only want the checkpoint:

```bash
python scripts/run_stage5_train_cnn.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

### I Want To Inspect Source-Of-Record Results

Look in `outputs/`, not `outputs/validation/`.

For Stage 4:

- `outputs/{dataset}_stage4_full_metrics.csv`
- `outputs/{dataset}_stage4_full_predictions.csv`

For Stage 5:

- `outputs/{dataset}_stage5_metrics.csv`
- `outputs/{dataset}_stage5_predictions.csv`

If you need exact split provenance, inspect:

- `outputs/cache/stage4/<dataset>/source/*.json`
- `outputs/cache/stage5/<dataset>/window_package/*.json`

## Common Mistakes And Troubleshooting

### Wrong dataset path

Symptom:

- loader errors such as “Dataset root does not exist” or “No pickle file found”

Fix:

- correct `dataset.root_dir` in your `.local.yaml`
- verify the subject directories contain the expected `Sx.pkl` files

### Confusing validation outputs with canonical outputs

Symptom:

- you inspect `outputs/validation/...` and assume those are the official results

Fix:

- canonical source-of-record outputs are always the unsuffixed files in `outputs/`
- validation and analysis runs belong under `outputs/validation/<label>/`

### Stale caches

Symptom:

- a rerun is reusing old artifacts after code or config changes

Fix:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --rebuild-cache
```

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --rebuild-cache
```

### Long runtimes

Expected:

- Stage 4 canonical runs can be heavy
- Stage 5 tuning and training can be much heavier

Recommendation:

- prepare caches first
- reuse cached packages for repeated experiments
- use validation output scope for analysis runs

### CPU-only expectations

Stage 5 uses PyTorch, but the repository is still CPU-first.

- there is no GPU assumption in the scripts or docs
- long runtimes are normal

### Over-interpreting proxy labels

Stage 3 quality targets, Stage 4 event targets, and Stage 4 irregular targets are not clinical truth.

Use them for repository evaluation, not clinical claims.

## What Not To Overclaim

Do not overclaim any of the following:

- Stage 3 quality pseudo-labels as true quality ground truth
- Stage 4 event or irregular proxy labels as clinical truth
- Stage 4 unified suspiciousness as a proven superior ranking layer over the Stage 3-only baseline
- Stage 4 outputs as diagnosis
- Stage 5 respiration outputs as clinical-grade validated respiration estimation
- analysis-only or validation-only runs as canonical results

The strongest safe current claims are:

- Stage 3 gives a practical quality-aware HR layer
- Stage 4 gives a richer, auditable suspicious-segment interpretation layer
- Stage 4 anomaly is the strongest standalone Stage 4 signal family
- Stage 5’s tuned CNN materially improves respiration estimation over the classical surrogate baseline on both supported datasets without changing HR outputs
