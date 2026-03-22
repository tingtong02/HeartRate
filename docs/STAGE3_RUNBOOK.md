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
- a Stage 3B2 DWT-denoised comparison branch layered on top of the same Stage 3 enhanced evaluation flow
- a validated minimum viable beat-level quality proxy branch through the Stage 2 enhanced outputs, with threshold / retention analysis retained in Stage 2 result tables
- a narrow Stage 3C2 robust-HR policy branch that integrates Stage 3 window quality with local 8 s beat-derived fallback HR and a limited auditable hold action

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
- DWT-denoised ML gating built from the same Stage 1 window pipeline
- a Stage 3C2 robust-HR policy path that adds local beat fallback and limited `hold_previous` behavior on top of the same enhanced evaluation flow

The threshold-analysis outputs stay strictly within Stage 3:

- `*_threshold_sweep.csv` records train-only threshold sweeps and report-only test sweeps
- `*_operating_points.csv` records the selected threshold, reference operating points, and a compact status such as `stable`, `fragile`, or `suboptimal`
- the predictions and metrics CSVs also include a motion-aware strengthened path for side-by-side comparison inside the same Stage 3 enhancement run
- threshold-analysis tables include a `branch` column so raw ML and DWT ML operating points can be compared inside the same CSV family
- the predictions CSV also includes Stage 3C2 policy-audit fields such as `robust_hr_action`, `robust_hr_source`, `beat_fallback_hr_bpm`, `hold_applied`, and `policy_reason_code`

For Stage 3C2 output interpretation:

- `gated_stage3_ml_logreg` remains the direct Stage 3 ML-gated comparison path
- `robust_stage3c2_policy` is an additive robust-output path, not a replacement for the other Stage 3 comparison rows
- local beat-derived fallback HR is computed directly on the current Stage 1 8 s window using the existing Stage 2 beat / IBI functions
- `hold_previous` is intentionally limited, resets at subject boundaries, and should be treated as a short auditable recovery action rather than smoothing

## Notes

- Pseudo-labels are derived from ECG-backed Stage 1 frequency HR error.
- Good / poor target thresholds are config-driven in `configs/eval/hr_stage3.yaml`.
- `motion_flag` is diagnostic in round 1 and does not independently override the main quality decision.
- The enhancement round uses the same pseudo-labels as the rule baseline and selects the ML gating threshold on train subjects only.
- Stage 3B1 does not add denoising, adaptive filtering, beat-level SQI, or a new model family; it only strengthens motion/noise-related reasoning within the existing Stage 3A framework.
- Stage 3B2 adds a single DWT denoising branch using `PyWavelets`; it does not add SSA, adaptive filtering, beat-level SQI, or deep-learning denoising.
- Test-set threshold rows are for reporting only and are not used to choose the deployed threshold.
- Beat-level quality closure is still partial: Stage 3 currently includes a minimum viable, validated beat-level quality proxy via Stage 2 outputs, but not full beat-level SQI closure.
- For the beat-quality branch, `enhanced_beat_quality` is the official baseline operating point and `enhanced_beat_quality_refined` is analysis-only.
- Stage 3C2 does not use 60 s Stage 2 aggregation as its primary fallback path in this first round.
