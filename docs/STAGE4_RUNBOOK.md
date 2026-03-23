# Stage 4 Runbook

## Scope

Stage 4 round 1 currently includes only a quality-gated, rule-based HR event baseline:

- `tachycardia_event`
- `bradycardia_event`
- `abrupt_change_event`

This round does not include:

- irregular pulse screening
- anomaly scoring
- TCN / autoencoder / GPU-dependent methods
- clinical event labels or clinical diagnosis

## Default Sources

Stage 4A reuses Stage 3 quality-aware HR outputs and keeps the source configurable per event family.

Round-1 defaults:

- `tachycardia_event`: `gated_stage3_ml_logreg`
- `bradycardia_event`: `gated_stage3_ml_logreg`
- `abrupt_change_event`: `robust_stage3c2_policy`

The Stage 4 predictions CSV keeps the selected source explicit per row through `selected_hr_source`.

## Thresholds And Episode Rules

Round-1 default thresholds in `configs/eval/hr_stage4.yaml`:

- tachycardia: HR `>= 100 bpm`
- bradycardia: HR `<= 50 bpm`
- abrupt change: absolute HR delta `>= 20 bpm`

Round-1 persistence / episode rules:

- tachycardia minimum persistence: `2` valid windows
- bradycardia minimum persistence: `2` valid windows
- abrupt-change minimum persistence: `2` valid windows
- episode merge gap: `1` window

Abrupt-change uses a narrow confirmation rule:

- the first abrupt window is triggered by a large delta from the previous reliable HR
- the next window can confirm the same episode if it remains displaced from the pre-change anchor by at least `50%` of the abrupt threshold

## Proxy Targets

This repository does not currently contain true clinical event labels.

Stage 4A therefore evaluates against repository-specific proxy targets built from ECG-backed reference HR on the same Stage 1 `8 s / 2 s` window timeline.

Proxy-target policy:

- reference source: `ref_hr_bpm`
- same thresholds as the detector
- same persistence and episode-merging rules as the detector

These proxy targets are evaluation aids only. They must not be described as clinical ground truth.

## Quality Gating Policy

Stage 4A keeps explicit quality-aware suppression.

Round-1 policy:

- a raw rule candidate may still be recorded through `event_candidate_flag`
- a candidate is emitted as a valid event only if `quality_gate_passed == True`
- low-quality or unavailable source windows are suppressed, not silently emitted as normal events
- abrupt-change also requires a previous reliable HR from the same subject

Important audit fields:

- `selected_hr_source`
- `selected_hr_bpm`
- `selected_hr_is_valid`
- `quality_gate_passed`
- `quality_gate_reason`
- `event_candidate_flag`
- `event_validity_flag`
- `event_reason_code`

## Outputs

Stage 4A writes:

- `outputs/{dataset}_stage4_event_predictions.csv`
- `outputs/{dataset}_stage4_event_metrics.csv`

Predictions CSV is a long-format table with one row per `window x event_type`.

Important columns:

- identity: `split`, `dataset`, `subject_id`, `window_index`, `start_time_s`, `duration_s`
- source audit: `selected_hr_source`, `selected_hr_bpm`, `selected_hr_is_valid`
- abrupt audit: `selected_hr_prev_bpm`, `selected_hr_delta_bpm`, `event_anchor_hr_bpm`
- quality audit: `quality_gate_passed`, `quality_gate_reason`
- proxy target: `proxy_event_target`
- event decision: `event_candidate_flag`, `event_trigger_rule`, `event_threshold_bpm`, `event_severity_score`, `event_validity_flag`, `event_reason_code`
- episode audit: `episode_id`, `episode_start_flag`, `episode_end_flag`

Metrics CSV reports one row per `split x event_type` plus an `all_events` aggregate row.

Key metrics fields:

- `num_eval_windows`
- `num_eval_events`
- `precision`
- `recall`
- `f1`
- `accuracy`
- `selected_hr_valid_fraction`
- `quality_gate_pass_fraction`
- `candidate_fraction`
- `suppressed_candidate_count`
- `valid_event_fraction`
- `num_pred_events`

## How To Run

PPG-DaLiA:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

WESAD:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

## Limitations

- Stage 4A is still heuristic and repository-specific.
- Proxy targets are not clinical truth.
- Abrupt-change detection is sensitive to source continuity and uses a deliberately simple confirmation rule in round 1.
- Stage 4B irregular pulse screening and Stage 4C anomaly scoring are still unimplemented.
