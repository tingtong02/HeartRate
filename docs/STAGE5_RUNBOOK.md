# Stage 5 Runbook

## Scope

Stage 5 is now complete for this repository's current CPU-first implemented scope.

Implemented Stage 5 components:

- direct respiration reference extraction from the public chest `Resp` waveform in `PPG-DaLiA` and `WESAD`
- a classical respiration surrogate baseline using `RIAV`, `RIFV`, and `RIBV`
- a tuned 1D CNN multitask respiration model with:
  - RR regression head
  - respiration-validity / confidence head
- a Stage 5 unified multitask interface that carries forward Stage 4 context and adds respiration outputs
- cache-backed Stage 5 window preparation for repeated tuning and evaluation runs

Stage 5 still does not include:

- TCN-based respiration modeling
- end-to-end joint HR+RR retraining
- clinical respiration labels or clinical respiration claims
- GPU-dependent training

## Respiration References

Stage 5 does not use a proxy-only default label.

Default Stage 5 reference source:

- the existing public chest `Resp` waveform already present in both supported datasets

Reference RR derivation per Stage 5 window:

- resample chest Resp to a lower working rate
- bandpass in the respiratory band `0.08–0.70 Hz`
- estimate RR with:
  - spectral peak RR
  - breath-peak counting RR
- keep a reference RR only when:
  - at least `3` breaths are present
  - respiratory spectral peak support is strong enough
  - spectral and breath-peak estimates agree within `3 bpm`

This makes the default Stage 5 label a direct public-signal-derived repository reference, not a synthetic proxy label.

## Default Models

Classical baseline:

- method name: `resp_surrogate_fusion_baseline`
- derive `RIAV`, `RIFV`, and `RIBV` from Stage 2 enhanced beats
- interpolate surrogate series to a uniform grid
- estimate RR with Welch PSD inside the respiratory band
- fuse surrogate RR candidates using spectral prominence / support

Default Stage 5 CNN:

- model name: `stage5_resp_multitask_cnn_v1`
- input window timeline:
  - default Stage 5 candidate windows: `32 s` and `48 s`
  - step: `4 s`
  - CNN input rate: `16 Hz`
- time-series channels:
  - `PPG`
  - `ACC` magnitude
  - `RIAV`
  - `RIFV`
  - `RIBV`
- scalar context:
  - surrogate-fusion baseline RR
  - Stage 3 / Stage 4 quality summaries
  - beat and clean-IBI support counts
  - PRV support features
- architecture:
  - shared 1D CNN encoder
  - small scalar-context MLP
  - fused shared MLP
  - RR regression head
  - respiration-validity confidence head

Stage 5 leaves the HR pipeline unchanged:

- Stage 5 inherits Stage 4 HR/quality/event/anomaly outputs
- Stage 5 does not retrain or replace the Stage 4 HR estimator

## Self-Optimization Performed

Stage 5 tuning was run per dataset, not pooled.

Tuning phases:

- Phase A structural sweep:
  - window length: `32 s` vs `48 s`
  - channels: `ppg_acc` vs `ppg_acc_ri`
- Phase B hyperparameter sweep on the top structures:
  - base width: `32` vs `64`
  - dropout: `0.10` vs `0.20`
  - learning rate: `1e-3` vs `3e-4`
  - batch size: `64` vs `128`
  - weight decay: `0` vs `1e-4`
- Phase C confidence threshold sweep:
  - `0.4`, `0.5`, `0.6`

Selection rule:

- primary: lowest inner-validation high-quality RR MAE
- tie-breaks:
  - lower RMSE
  - higher correlation
  - simpler model

Canonical winners in this round:

- `PPG-DaLiA`:
  - `48 s`
  - `ppg_acc`
  - base width `32`
  - dropout `0.20`
  - learning rate `3e-4`
  - batch size `64`
  - threshold `0.5`
- `WESAD`:
  - `32 s`
  - `ppg_acc_ri`
  - base width `64`
  - dropout `0.10`
  - learning rate `1e-3`
  - batch size `128`
  - threshold `0.4`

## Canonical Results

Canonical Stage 5 eval results versus the classical baseline:

- `PPG-DaLiA`, high-quality reference-valid subset:
  - baseline: `MAE 9.16 bpm`, `RMSE 10.25 bpm`, `r 0.10`, `within_3 0.089`
  - CNN: `MAE 2.37 bpm`, `RMSE 3.38 bpm`, `r 0.61`, `within_3 0.744`
- `WESAD`, high-quality reference-valid subset:
  - baseline: `MAE 6.41 bpm`, `RMSE 7.52 bpm`, `r 0.13`, `within_3 0.260`
  - CNN: `MAE 2.85 bpm`, `RMSE 3.50 bpm`, `r 0.43`, `within_3 0.591`

Respiration-validity confidence quality on eval:

- `PPG-DaLiA`: `AUROC 0.823`, `AUPRC 0.639`
- `WESAD`: `AUROC 0.709`, `AUPRC 0.782`

Best-supported Stage 5 interpretation:

- the tuned CNN clearly outperforms the classical surrogate baseline on both datasets
- RR estimates are usable on high-quality segments on both datasets
- `PPG-DaLiA` is the stronger of the two current Stage 5 datasets

## HR Non-Degradation

Stage 5 does not change Stage 4 HR predictions.

Reported carry-forward checks in `stage5_metrics.csv`:

- `hr_selected_hr_match_rate = 1.0`
- `stage4_suspicion_match_rate = 1.0`

Interpretation:

- Stage 5 adds respiration and multitask aggregation
- Stage 5 does not materially degrade the existing HR output layer because it does not replace it

## Unified Stage 5 Output Interface

Stage 5 predictions are one row per respiration window.

Key fields include:

- identity:
  - `dataset`, `split`, `subject_id`, `window_index`, `start_time_s`, `duration_s`
- respiration:
  - `resp_rate_ref_bpm`
  - `resp_rate_ref_valid_flag`
  - `resp_reference_reason`
  - `resp_rate_baseline_bpm`
  - `resp_rate_pred_bpm`
  - `resp_confidence`
  - `resp_validity_flag`
- inherited Stage 4 context:
  - `selected_hr_bpm`
  - `selected_hr_source`
  - `selected_hr_is_valid`
  - `ml_signal_quality_score`
  - `motion_flag`
  - `validity_flag`
  - `hr_event_flag`
  - `irregular_pulse_flag`
  - `anomaly_score`
  - `stage4_suspicion_flag`
  - `stage4_suspicion_score`
  - `stage4_suspicion_type_summary`
- Stage 2 support summaries:
  - `num_beats`
  - `num_ibi_clean`
  - `mean_ibi_ms`
  - `rmssd_ms`
  - `sdnn_ms`
  - `pnn50`
  - serialized `beat_positions_s`
  - serialized `ibi_series_ms`

## Output Files

Canonical Stage 5 outputs:

- `outputs/{dataset}_stage5_predictions.csv`
- `outputs/{dataset}_stage5_metrics.csv`
- `outputs/{dataset}_stage5_tuning_results.csv`
- `outputs/models/stage5/{dataset}_stage5_resp_multitask_cnn_v1_best.pt`
- `outputs/models/stage5/{dataset}_stage5_resp_multitask_cnn_v1_best_config.json`

Validation-mode outputs use the same scope routing as Stage 4:

- canonical outputs stay in `outputs/`
- bounded or analysis-only outputs go to `outputs/validation/<label>/`

Reusable cache artifacts:

- `outputs/cache/stage5/{dataset}/window_package/*.joblib`
- `outputs/cache/stage5/{dataset}/window_package/*.json`

These cache artifacts are reusable intermediates, not source-of-record outputs.

## Workflow

Prepare reusable Stage 5 packages first:

```bash
python scripts/prepare_stage5_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the full Stage 5 pipeline:

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Train and save only the tuned CNN checkpoint:

```bash
python scripts/run_stage5_train_cnn.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run bounded / analysis-only experiments without touching canonical outputs:

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml \
  --output-scope validation \
  --output-label stage5_analysis
```

## Limitations

- Stage 5 respiration references come from public chest Resp signals, not clinical adjudication
- the repository does not yet implement TCN-based or more advanced multimodal respiration models
- the current multitask interface is aggregation-based; it does not jointly retrain HR and RR in one network
- WESAD remains harder than PPG-DaLiA for this Stage 5 setup
- Stage 5 is useful and reproducible for the repository's practical CPU-first scope, but it should not be over-interpreted as clinical respiration validation
