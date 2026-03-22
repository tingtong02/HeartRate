# Stage 3 Runbook

## Scope

Stage 3 currently covers a narrow, CPU-friendly quality-gating track for the existing Stage 1 HR pipeline.

This round includes:

- window-level quality scoring for the current 8 s Stage 1 windows
- a `good` / `poor` quality decision
- a `validity_flag` used to gate `stage1_frequency`
- an auxiliary `motion_flag` based on optional ACC
- gated-vs-ungated HR comparison against ECG-backed reference HR
- a lightweight ML quality path using `LogisticRegression`
- train-only threshold selection and threshold / retention operating-point analysis
- a Stage 3B1 motion-aware strengthened comparison layered on top of the existing ML-gated path

This round does not include:

- event detection
- irregular pulse screening
- respiration
- deep learning training
- GPU-dependent methods
- Stage 2 feature-family expansion

## Environment

Reuse the existing environment:

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

## Run Stage 3

PPG-DaLiA:

```bash
python scripts/run_stage3_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

WESAD:

```bash
python scripts/run_stage3_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

## Outputs

Stage 3 writes:

- `outputs/{dataset}_stage3_predictions.csv`
- `outputs/{dataset}_stage3_metrics.csv`

Use `outputs/{dataset}_stage3_metrics.csv` as the primary summary file.

## Enhancement Round

Stage 3 enhancement round keeps the rule-based baseline and adds a lightweight ML comparison path.

Run the enhancement comparison on PPG-DaLiA:

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the enhancement comparison on WESAD:

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Enhancement round writes:

- `outputs/{dataset}_stage3_enhanced_predictions.csv`
- `outputs/{dataset}_stage3_enhanced_metrics.csv`
- `outputs/{dataset}_stage3_enhanced_threshold_sweep.csv`
- `outputs/{dataset}_stage3_enhanced_operating_points.csv`

These tables compare:

- ungated `stage1_frequency`
- rule-based Stage 3 gating
- ML-based Stage 3 gating using `LogisticRegression`
- motion-aware strengthened Stage 3 gating built on top of the same ML quality score

The threshold-analysis outputs stay strictly within Stage 3:

- `*_threshold_sweep.csv` records train-only threshold sweeps and report-only test sweeps
- `*_operating_points.csv` records the selected threshold, reference operating points, and a compact status such as `stable`, `fragile`, or `suboptimal`
- the predictions and metrics CSVs also include a motion-aware strengthened path for side-by-side comparison inside the same Stage 3 enhancement run

## Notes

- Pseudo-labels are derived from ECG-backed Stage 1 frequency HR error.
- Good / poor target thresholds are config-driven in `configs/eval/hr_stage3.yaml`.
- `motion_flag` is diagnostic in round 1 and does not independently override the main quality decision.
- The enhancement round uses the same pseudo-labels as the rule baseline and selects the ML gating threshold on train subjects only.
- Stage 3B1 does not add denoising, adaptive filtering, beat-level SQI, or a new model family; it only strengthens motion/noise-related reasoning within the existing Stage 3A framework.
- Test-set threshold rows are for reporting only and are not used to choose the deployed threshold.
