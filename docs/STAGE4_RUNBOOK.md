# Stage 4 Runbook

## Scope

Stage 4 currently includes two narrow, CPU-first repository baselines:

- Stage 4A: quality-gated, rule-based HR event detection
- Stage 4B: quality-gated irregular pulse screening

Implemented Stage 4A event families:

- `tachycardia_event`
- `bradycardia_event`
- `abrupt_change_event`

Implemented Stage 4B screening outputs:

- `hist_gbdt_irregular`
- `irregular_rule_baseline`

This repository still does not include:

- clinical event labels
- clinical rhythm diagnosis
- Stage 4C anomaly scoring
- TCN / autoencoder / GPU-dependent methods

## Stage 4A Defaults

Round-1 Stage 4A defaults remain unchanged:

- `tachycardia_event`: `gated_stage3_ml_logreg`
- `bradycardia_event`: `gated_stage3_ml_logreg`
- `abrupt_change_event`: `robust_stage3c2_policy`

Default thresholds in `configs/eval/hr_stage4.yaml`:

- tachycardia: HR `>= 100 bpm`
- bradycardia: HR `<= 50 bpm`
- abrupt change: absolute HR delta `>= 20 bpm`

Stage 4A predictions are written to:

- `outputs/{dataset}_stage4_event_predictions.csv`
- `outputs/{dataset}_stage4_event_metrics.csv`

## Stage 4B Defaults

Round-2 Stage 4B adds a shared Stage 4 feature frame on the same Stage 1 `8 s / 2 s` window timeline.

Default feature-source policy:

- selected HR source for local HR context: `robust_stage3c2_policy`
- beat / IBI feature path: Stage 2 `enhanced`
- Stage 3 quality reference for gating: `ml_validity_flag`

Default model:

- `HistGradientBoostingClassifier`

Default comparison baseline:

- `irregular_rule_baseline`

The default tree model was chosen because it stays inside the current sklearn-only dependency set and remains CPU-friendly.

## Stage 4B Feature Frame

The shared Stage 4 feature frame combines:

- Stage 2 beat / IBI / PRV summaries:
  - `mean_ibi_ms`
  - `median_ibi_ms`
  - `mean_hr_bpm_from_ibi`
  - `sdnn_ms`
  - `rmssd_ms`
  - `pnn50`
  - `ibi_cv`
- local irregularity summaries:
  - successive IBI jump mean / max
  - local IBI deviation ratio mean / max
  - IBI MAD
  - turning-point ratio
  - `selected_hr_delta_bpm`
- support and missingness fields:
  - `num_beats`
  - `num_ibi_raw`
  - `num_ibi_clean`
  - `ibi_is_valid`
  - `ibi_removed_ratio`
  - insufficient-support flags
- beat-quality summaries:
  - `beat_quality_mean_score`
  - `beat_quality_good_ratio`
  - `beat_quality_good_count`
- Stage 3 quality context:
  - `ml_signal_quality_score`
  - `ml_validity_flag`
  - `rule_signal_quality_score`
  - `rule_validity_flag`
  - `motion_flag`
- robust-policy audit context:
  - `robust_hr_source`
  - `robust_hr_action`
  - `hold_applied`
  - `hold_age_windows`
  - `hr_jump_bpm_from_previous`
  - one-hot source / action indicators for modeling

## Stage 4B Proxy Labels

This repository does not contain clinical irregular-rhythm labels.

Stage 4B therefore evaluates against repository-specific ECG/reference-side irregularity proxy labels.

Default proxy-label logic in `configs/eval/hr_stage4_irregular.yaml`:

- source: ECG-derived reference beats inside the same Stage 1 window
- derive reference IBI series
- clean reference IBI with the same Stage 2 `enhanced` cleaning logic
- require minimum clean reference IBI support
- mark positive when any configured irregularity heuristic fires:
  - `ref_rmssd_ms >= irregular_rmssd_ms`
  - `ref_pnn50 >= irregular_pnn50`
  - `ref_ibi_cv >= irregular_ibi_cv`
  - `ref_local_deviation_ratio_max >= irregular_local_deviation_ratio`

These proxy labels are evaluation aids only. They must not be described as clinical truth.

## Stage 4B Quality Gating

Stage 4B keeps explicit suppress-but-audit behavior.

Default quality-gate policy:

- require `selected_hr_is_valid == True`
- require Stage 3 quality pass through `ml_validity_flag`
- require sufficient local beat / clean-IBI support
- suppress screening outputs if robust policy is in disallowed continuity states:
  - `robust_hr_source in {"none", "hold_previous"}`
  - `robust_hr_action in {"hold", "reject"}`

Important Stage 4B audit fields:

- `selected_hr_source`
- `selected_hr_bpm`
- `selected_hr_is_valid`
- `quality_gate_passed`
- `quality_gate_reason`
- `screening_candidate_flag`
- `irregular_pulse_flag`
- `screening_validity_flag`
- `screening_reason_code`

## Outputs

Stage 4B writes:

- `outputs/{dataset}_stage4_irregular_predictions.csv`
- `outputs/{dataset}_stage4_irregular_metrics.csv`

Optional debug export:

- `outputs/{dataset}_stage4_feature_frame.csv`

Stage 4B predictions are one row per `window x method`.

Important prediction columns:

- identity: `split`, `dataset`, `subject_id`, `window_index`, `start_time_s`, `duration_s`
- label audit: `screening_proxy_target`, `proxy_label_support_flag`, `proxy_label_reason`
- score / decision: `model_name`, `screening_score`, `irregular_pulse_score`, `screening_threshold`, `screening_candidate_flag`, `irregular_pulse_flag`
- gating: `screening_validity_flag`, `screening_reason_code`, `quality_gate_passed`, `quality_gate_reason`
- support / context: `selected_hr_source`, `selected_hr_bpm`, `selected_hr_is_valid`, `num_beats`, `num_ibi_clean`, `ibi_is_valid`, `ml_signal_quality_score`, `robust_hr_source`, `robust_hr_action`

Stage 4B metrics include:

- `num_eval_windows`
- `num_positive_targets`
- `num_positive_predictions`
- `accuracy`
- `precision`
- `recall`
- `f1`
- `auroc`
- `auprc`
- `selected_hr_valid_fraction`
- `quality_gate_pass_fraction`
- `support_sufficient_fraction`
- `suppressed_positive_count`
- `valid_prediction_fraction`

## How To Run

Stage 4A on PPG-DaLiA:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 4A on WESAD:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Stage 4B on PPG-DaLiA:

```bash
python scripts/run_stage4_irregular_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 4B on WESAD:

```bash
python scripts/run_stage4_irregular_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

## Limitations

- Stage 4A and Stage 4B are both repository-specific baselines.
- Proxy labels are not clinical ground truth.
- Stage 4B should be interpreted as non-diagnostic screening for irregularity suspicion, not diagnosis.
- Stage 4B currently uses fixed round-2 thresholds rather than train-only decision-threshold tuning.
- Stage 4C anomaly scoring is still unimplemented.
