# Stage 4 Runbook

## Scope

Stage 4 is now complete for this repository's current CPU-first implemented scope.

Implemented Stage 4 components:

- Stage 4A: quality-gated, rule-based HR event detection
- Stage 4B: quality-gated irregular pulse screening
- Stage 4C: quality-gated anomaly scoring
- final unified Stage 4 output layer with combined suspiciousness and Stage 3-vs-Stage 4 comparison rows

Implemented Stage 4A event families:

- `tachycardia_event`
- `bradycardia_event`
- `abrupt_change_event`

Implemented Stage 4B screening outputs:

- `hist_gbdt_irregular`
- `irregular_rule_baseline`

Implemented Stage 4C anomaly output:

- `isolation_forest_anomaly`

This repository still does not include:

- clinical event labels
- clinical rhythm diagnosis
- deep sequence models for Stage 4
- autoencoder anomaly models
- GPU-dependent methods

## Final Stage 4 Defaults

Stage 4A defaults remain:

- `tachycardia_event`: `gated_stage3_ml_logreg`
- `bradycardia_event`: `gated_stage3_ml_logreg`
- `abrupt_change_event`: `robust_stage3c2_policy`

Stage 4B defaults:

- shared Stage 4 feature frame on the Stage 1 `8 s / 2 s` window timeline
- selected HR context source: `robust_stage3c2_policy`
- beat / IBI feature path: Stage 2 `enhanced`
- default model: `HistGradientBoostingClassifier`
- default screening method name: `hist_gbdt_irregular`
- comparison baseline: `irregular_rule_baseline`

Stage 4C defaults:

- anomaly model: `IsolationForest`
- default method name: `isolation_forest_anomaly`
- fit reference set: train-only, quality-passed, support-sufficient, proxy-regular windows
- anomaly alert cutoff: train-only `alert_quantile = 0.95`

Final unified Stage 4 defaults:

- comparison target: `proxy_abnormal_union`
- Stage 3-only suspiciousness baseline:
  - `stage3_quality_suspicious_score = 1 - ml_signal_quality_score`
  - `stage3_quality_suspicious_flag = ~ml_validity_flag`
- combined Stage 4 suspiciousness:
  - base = max(valid event component, valid irregular score, valid anomaly score)
  - event component floor = `0.60`
  - add `0.10` for exactly two valid signal families
  - add `0.20` for three valid signal families

Stage 4 workflow defaults:

- cache-enabled reusable source / feature preparation
- canonical output scope: `outputs/`
- bounded validation output scope: `outputs/validation/<label>/`
- cache root: `outputs/cache/stage4`
- full-layer default variant name: `default`

## Stage 4A Event Logic

Default thresholds in `configs/eval/hr_stage4_full.yaml`:

- tachycardia: HR `>= 100 bpm`
- bradycardia: HR `<= 50 bpm`
- abrupt change: absolute HR delta `>= 20 bpm`

Stage 4A includes:

- explicit source selection per event family
- suppress-but-audit quality gating
- proxy event targets built from ECG-backed reference HR
- minimum persistence rules
- episode grouping and merge-gap logic

## Shared Stage 4 Feature Frame

The shared Stage 4 feature frame is the common upstream input for Stage 4B and Stage 4C.

Feature categories:

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
  - `ibi_mad_ms`
  - `turning_point_ratio`
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
  - `robust_hr_is_valid`
  - `hold_applied`
  - `hold_age_windows`
  - `hr_jump_bpm_from_previous`
  - one-hot source / action indicators for modeling

## Proxy Labels And Evaluation Targets

This repository does not contain clinical event or rhythm labels.

Stage 4A evaluation uses ECG-backed proxy HR event targets.

Stage 4B evaluation uses ECG/reference-side irregularity proxy labels:

- derive ECG reference beats inside the same Stage 1 window
- derive reference IBI
- clean the reference IBI with the Stage 2 `enhanced` cleaning logic
- require minimum clean reference IBI support
- mark positive when any configured irregularity heuristic fires:
  - `ref_rmssd_ms >= irregular_rmssd_ms`
  - `ref_pnn50 >= irregular_pnn50`
  - `ref_ibi_cv >= irregular_ibi_cv`
  - `ref_local_deviation_ratio_max >= irregular_local_deviation_ratio`

Final Stage 4 comparison target:

- `proxy_abnormal_target = proxy_hr_event_target_any OR screening_proxy_target`
- evaluate comparison rows only where both event and irregular proxy support are available:
  - `proxy_abnormal_support_flag = proxy_hr_event_support_flag AND proxy_label_support_flag`

These proxy targets are repository-specific evaluation aids only. They must not be described as clinical truth.

## Quality Gating

Stage 4 remains explicitly quality-aware and uses suppress-but-audit behavior.

Default Stage 4B / Stage 4C gating policy:

- require `selected_hr_is_valid == True`
- require Stage 3 quality pass through `ml_validity_flag`
- require sufficient local beat / clean-IBI support
- suppress outputs when robust continuity is in disallowed states:
  - `robust_hr_source in {"none", "hold_previous"}`
  - `robust_hr_action in {"hold", "reject"}`

Important audit fields preserved across Stage 4 outputs:

- `selected_hr_source`
- `selected_hr_bpm`
- `selected_hr_is_valid`
- `quality_gate_passed`
- `quality_gate_reason`
- Stage 4A: `event_validity_flag`, `event_reason_code`, episode fields
- Stage 4B: `screening_validity_flag`, `screening_reason_code`
- Stage 4C: `anomaly_validity_flag`, `anomaly_reason_code`, `anomaly_fit_reference_flag`

## Stage 4C Anomaly Layer

Stage 4C uses `IsolationForest` over the shared Stage 4 feature frame.

Anomaly scoring logic:

- fit on train-only reference rows selected as regular, quality-passed windows
- compute `raw_anomaly_score = -decision_function(...)`
- normalize to `anomaly_score` against the train reference-score distribution
- threshold with a train-only quantile cutoff
- apply the shared suppress-but-audit quality gate

Important anomaly fields:

- `raw_anomaly_score`
- `anomaly_score`
- `anomaly_threshold`
- `anomaly_candidate_flag`
- `anomaly_flag`
- `anomaly_validity_flag`
- `anomaly_reason_code`
- `anomaly_fit_reference_flag`

## Final Unified Stage 4 Output

The final Stage 4 interface is one row per window.

Key fields include:

- identity:
  - `split`, `dataset`, `subject_id`, `window_index`, `start_time_s`, `duration_s`
- Stage 3 context:
  - `selected_hr_source`
  - `selected_hr_bpm`
  - `selected_hr_is_valid`
  - `ml_signal_quality_score`
  - `stage3_quality_suspicious_score`
  - `stage3_quality_suspicious_flag`
- shared gate:
  - `quality_gate_passed`
  - `quality_gate_reason`
- Stage 4A summary:
  - `hr_event_flag`
  - `hr_event_type`
  - `hr_event_type_summary`
  - `hr_event_severity_score`
  - `event_validity_flag`
- Stage 4B summary:
  - `irregular_pulse_flag`
  - `irregular_pulse_score`
  - `screening_validity_flag`
- Stage 4C summary:
  - `anomaly_score`
  - `anomaly_flag`
  - `anomaly_validity_flag`
- final Stage 4 layer:
  - `stage4_suspicion_flag`
  - `stage4_suspicion_score`
  - `stage4_suspicion_type_summary`
  - `stage4_reason_code`
- proxy targets:
  - `proxy_hr_event_target_any`
  - `screening_proxy_target`
  - `proxy_abnormal_target`
  - `proxy_abnormal_support_flag`

`stage4_suspicion_type_summary` is a sorted `|`-joined summary such as:

- `event:tachycardia_event|irregular|anomaly`

`stage4_reason_code` priority:

- `multi_signal_suspicion`
- `hr_event_suspicion`
- `irregular_pulse_suspicion`
- `anomaly_suspicion`
- `low_quality_suppressed`
- `no_stage4_signal`

## Output Files

Stage 4A:

- `outputs/{dataset}_stage4_event_predictions.csv`
- `outputs/{dataset}_stage4_event_metrics.csv`

Stage 4B:

- `outputs/{dataset}_stage4_irregular_predictions.csv`
- `outputs/{dataset}_stage4_irregular_metrics.csv`

Stage 4C:

- `outputs/{dataset}_stage4_anomaly_predictions.csv`
- `outputs/{dataset}_stage4_anomaly_metrics.csv`

Final full Stage 4 layer:

- `outputs/{dataset}_stage4_full_predictions.csv`
- `outputs/{dataset}_stage4_full_metrics.csv`

`stage4_full_metrics.csv` uses flat `metric_group` rows for:

- `event`
- `irregular`
- `anomaly`
- `unified`
- `stage3_comparison`
- `stratification`

Comparison rows explicitly contrast:

- `stage3_quality_only`
- `stage4_irregular_default`
- `stage4_anomaly_default`
- `stage4_full_default`

## Cache-Backed Workflow

Stage 4 reruns are expensive mainly because they otherwise rebuild:

- Stage 3-quality-aware source frames
- Stage 4 shared feature frames

This repository now supports two explicit reusable cache artifacts:

- `quality_aware_source_package`
- `stage4_feature_package`

Artifact location:

- `outputs/cache/stage4/<dataset>/source/<cache_key>.joblib`
- `outputs/cache/stage4/<dataset>/feature/<cache_key>.joblib`
- matching JSON manifests are written alongside each artifact

Cache manifests record:

- cache schema version
- config hash
- dataset metadata
- train / eval subject lists
- row counts
- selected Stage 3 ML threshold
- build timestamp

Cache key inputs include:

- dataset name and root path
- sorted train / eval subject lists
- relevant preprocessing, evaluation, Stage 1, and Stage 3 config sections for the source package
- source package key plus `stage4_shared` settings for the feature package
- cache schema version

Cache behavior:

- `scripts/prepare_stage4_sources.py` builds or reuses both packages explicitly
- Stage 4 scripts auto-reuse matching cache artifacts unless `cache.rebuild: true` or `--rebuild-cache` is passed
- runners print whether each package was built or reused and how long it took

These cache artifacts are reusable implementation intermediates only. They are not source-of-record analysis outputs.

## Output Scope Separation

Stage 4 output hygiene is now explicit:

- `output.scope: canonical`
  - writes unsuffixed source-of-record files to `outputs/`
- `output.scope: validation`
  - requires a non-empty `output.label`
  - writes to `outputs/validation/<label>/`

This separation prevents bounded subject-slice runs from silently overwriting canonical output files.

## How To Run

Prepare reusable Stage 4 source and feature packages:

```bash
python scripts/prepare_stage4_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Prepare bounded-validation packages explicitly:

```bash
python scripts/prepare_stage4_sources.py \
  --dataset-config /tmp/ppg_dalia_stage4_medium6.yaml \
  --output-scope validation \
  --output-label bounded_medium6_seed42
```

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

Final full Stage 4 pipeline on PPG-DaLiA:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Final full Stage 4 pipeline on WESAD:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Final full Stage 4 bounded validation on PPG-DaLiA:

```bash
python scripts/run_stage4_full.py \
  --dataset-config /tmp/ppg_dalia_stage4_medium6.yaml \
  --output-scope validation \
  --output-label bounded_medium6_seed42
```

Analysis-only fusion check using cached inputs:

```bash
python scripts/run_stage4_full.py \
  --config configs/eval/hr_stage4_full.yaml \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --eval-config configs/eval/hr_stage4_full_balanced_v1_analysis.yaml \
  --output-scope validation \
  --output-label fusion_balanced_v1_canonical
```

## Validation And Canonical Closure Summary

The bounded `medium6` validation runs remain useful for quicker iteration and analysis-only comparisons.

`PPG-DaLiA`

- subjects: `S1,S10,S11,S12,S13,S14`
- train: `S11,S12,S13,S14`
- eval: `S1,S10`
- one-time prep:
  - source package build `296.66 s`
  - feature package build `227.60 s`
  - total prep `524.31 s`
- cached rerun:
  - source package reuse `0.01 s`
  - feature package reuse `0.01 s`
  - full pipeline runtime `23.64 s`
- eval comparison:
  - `stage3_quality_only`: `AUPRC 0.5729`, `AUROC 0.6203`, `precision 0.5755`, `recall 0.6949`
  - `stage4_anomaly_default`: `AUPRC 0.6090`, `AUROC 0.6584`, `precision 0.7008`, `recall 0.0562`
  - `stage4_full_default`: `AUPRC 0.5309`, `AUROC 0.4338`, `precision 0.4978`, `recall 0.2288`

`WESAD`

- subjects: `S10,S11,S13,S14,S15,S16`
- train: `S13,S14,S15,S16`
- eval: `S10,S11`
- one-time prep:
  - source package build `176.58 s`
  - feature package build `134.60 s`
  - total prep `311.21 s`
- cached rerun:
  - source package reuse `0.01 s`
  - feature package reuse `0.01 s`
  - full pipeline runtime `14.46 s`
- eval comparison:
  - `stage3_quality_only`: `AUPRC 0.6142`, `AUROC 0.6600`, `precision 0.6372`, `recall 0.6442`
  - `stage4_anomaly_default`: `AUPRC 0.6548`, `AUROC 0.6933`, `precision 0.6486`, `recall 0.0976`
  - `stage4_full_default`: `AUPRC 0.5352`, `AUROC 0.4057`, `precision 0.6444`, `recall 0.2426`

Canonical full-dataset Stage 4 closure runs were then completed and refreshed the unsuffixed source-of-record files in `outputs/`.

`PPG-DaLiA` canonical

- train subjects: `S10,S11,S12,S13,S15,S2,S3,S5,S7,S9`
- eval subjects: `S1,S14,S4,S6,S8`
- one-time canonical prep:
  - source package build `707.96 s`
  - feature package build `538.25 s`
  - total prep `1246.29 s`
- cached canonical rerun:
  - source package reuse `0.02 s`
  - feature package reuse `0.02 s`
  - full pipeline runtime `54.58 s`
- eval comparison:
  - `stage3_quality_only`: `AUPRC 0.6834`, `AUROC 0.5646`, `precision 0.6717`, `recall 0.7286`, `alert_rate 0.6833`
  - `stage4_anomaly_default`: `AUPRC 0.6902`, `AUROC 0.6064`, `precision 0.7911`, `recall 0.0744`, `alert_rate 0.0592`
  - `stage4_full_default`: `AUPRC 0.6581`, `AUROC 0.4666`, `precision 0.6487`, `recall 0.2169`, `alert_rate 0.2107`

`WESAD` canonical

- train subjects: `S11,S13,S14,S15,S17,S2,S3,S5,S7,S9`
- eval subjects: `S10,S16,S4,S6,S8`
- one-time canonical prep:
  - source package build `463.79 s`
  - feature package build `347.84 s`
  - total prep `811.69 s`
- cached canonical rerun:
  - source package reuse `0.02 s`
  - feature package reuse `0.01 s`
  - full pipeline runtime `36.00 s`
- eval comparison:
  - `stage3_quality_only`: `AUPRC 0.6098`, `AUROC 0.5720`, `precision 0.5925`, `recall 0.5355`, `alert_rate 0.5047`
  - `stage4_anomaly_default`: `AUPRC 0.6004`, `AUROC 0.5698`, `precision 0.7467`, `recall 0.0739`, `alert_rate 0.0553`
  - `stage4_full_default`: `AUPRC 0.5997`, `AUROC 0.4940`, `precision 0.6667`, `recall 0.3206`, `alert_rate 0.2685`

Conservative analysis-only fusion check:

- variant name: `balanced_v1_analysis`
- config-only changes:
  - `event_min_score = 0.45`
  - `two_signal_bonus = 0.05`
  - `three_signal_bonus = 0.10`
- result:
  - bounded `medium6`:
    - `PPG-DaLiA`: `AUPRC 0.5254`, `AUROC 0.4321`
    - `WESAD`: `AUPRC 0.5345`, `AUROC 0.4053`
  - canonical:
    - `PPG-DaLiA`: `AUPRC 0.6571`, `AUROC 0.4660`
    - `WESAD`: `AUPRC 0.5998`, `AUROC 0.4940`
- interpretation:
  - this variant underperformed the current default on both datasets
  - it did not meet the promotion rule and remains analysis-only

## Current Best-Supported Conclusions

- Stage 4 cache-backed preparation materially improves repeated-run practicality.
- Canonical full-dataset Stage 4 outputs now exist and are the source-of-record artifacts in `outputs/`.
- Stage 4C anomaly scoring is still the strongest standalone Stage 4 component, but canonical evidence is mixed across datasets rather than a uniform win over the Stage 3-only quality baseline.
  - it improves over `stage3_quality_only` on canonical `PPG-DaLiA`
  - it is slightly below `stage3_quality_only` on canonical `WESAD`
- The final unified Stage 4 suspiciousness default remains informative and auditable, but it still underperforms the simple Stage 3-only quality suspiciousness baseline on canonical eval for both datasets.
- The tested `balanced_v1_analysis` fusion adjustment did not improve that outcome and was not promoted.
- The repository should therefore continue to present:
  - anomaly-layer results as the strongest standalone Stage 4 signal family, while acknowledging the mixed cross-dataset canonical evidence
  - unified-fusion results as useful stratification output, not as a demonstrated ranking improvement over the Stage 3-only baseline

## Interpretation And Limitations

- Stage 4 is complete only for this repository's current CPU-first implemented scope.
- Stage 4 is not a replacement for the Stage 3 HR estimator.
- Stage 4 outputs are non-diagnostic.
- Proxy targets are not clinical ground truth.
- Stage 4C uses an unsupervised anomaly model over proxy-regular train windows, not clinical anomaly labels.
- The final Stage 4 suspiciousness layer is intentionally interpretable and auditable, not a hidden learned fusion model.
- Canonical ranking evidence does not support claiming that the unified Stage 4 suspiciousness layer is universally stronger than the Stage 3-only quality baseline.
- Broader future work such as richer anomaly models, clinical labels, respiration, and deep sequence modeling remains deferred.
