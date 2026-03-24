# Project Handoff

## Repository Identity And Current Scope

`HeartRate_CNN` is a CPU-first public-dataset physiological analysis repository built around wrist PPG, chest ECG, optional wrist ACC, and now chest respiration signals where available.

Current practical scope:

- supported datasets: `PPG-DaLiA` and `WESAD`
- public-data only workflow
- subject-wise evaluation
- no GPU requirement
- canonical pipeline coverage from Stage 0 through Stage 5

What the repository now includes:

- Stage 0: loading, alignment, windowing, ECG-backed HR reference, smoke path
- Stage 1: stronger HR estimation over `8 s / 2 s` windows
- Stage 2: beat detection, IBI cleaning, time-domain PRV features
- Stage 3: quality-aware HR gating and robust HR policy
- Stage 4: event detection, irregular pulse screening, anomaly scoring, unified suspiciousness
- Stage 5: CNN-based respiration estimation and a multitask output layer that carries Stage 4 context forward

This is not a clinical product. The repository is best understood as a reproducible, modular physiological analysis framework with conservative, audit-friendly outputs.

## Current Project Status By Stage

### Stage 0

- Implemented: yes
- Validated: yes
- Main entrypoints: `scripts/run_stage0_smoke.py`, `scripts/run_stage0_baseline.py`
- Main outputs: Stage 0 prints metrics; `outputs/{dataset}_stage0_predictions.csv` is only written if `configs/eval/hr_baseline.yaml` enables saving
- Complete vs limited:
  - complete for loading, alignment, segmentation, and ECG-backed window-level HR references
  - limited to a minimal Welch frequency HR baseline

### Stage 1

- Implemented: yes
- Validated: yes
- Safe default: `stage1_frequency`
- Comparison methods: `stage0_baseline`, `stage1_frequency`, `stage1_time`, `stage1_fusion`
- Main outputs:
  - `outputs/{dataset}_stage1_predictions.csv`
  - `outputs/{dataset}_stage1_metrics.csv`
- Complete vs limited:
  - complete for the repository’s basic HR-estimation baseline layer
  - `stage1_fusion` is reproducible and useful, but it is not the best current HR path

### Stage 2

- Implemented: yes
- Validated: yes
- Safe default: `enhanced`
- Official beat-quality operating point: `enhanced_beat_quality`
- Analysis-only Stage 2 operating point: `enhanced_beat_quality_refined`
- Main outputs:
  - `outputs/{dataset}_stage2_beats.csv`
  - `outputs/{dataset}_stage2_features.csv`
  - `outputs/{dataset}_stage2_metrics.csv`
  - `outputs/{dataset}_stage2_beat_quality.csv`
  - `outputs/{dataset}_stage2_beat_quality_sweep.csv`
- Complete vs limited:
  - complete as a practical beat / IBI / time-domain PRV foundation
  - still limited to time-domain PRV features; no frequency-domain or nonlinear HRV closure

### Stage 3

- Implemented: yes
- Validated: yes
- Accuracy-oriented default: `gated_stage3_ml_logreg`
- Robust-output prototype: `robust_stage3c2_policy`
- Analysis-only Stage 3 outputs:
  - `enhanced_beat_quality_refined`
  - `robust_stage3c2_policy_refined`
- Exploratory comparison branches:
  - motion-aware strengthening
  - DWT-denoised ML gate
- Main outputs:
  - `outputs/{dataset}_stage3_predictions.csv`
  - `outputs/{dataset}_stage3_metrics.csv`
  - `outputs/{dataset}_stage3_enhanced_predictions.csv`
  - `outputs/{dataset}_stage3_enhanced_metrics.csv`
  - threshold / operating-point / policy sweep CSVs
- Complete vs limited:
  - practically complete for this repository’s CPU-first Stage 3 scope
  - full beat-level SQI closure, SSA, adaptive filtering, and deep denoising remain deferred

### Stage 4

- Implemented: yes
- Validated: yes
- Stage 4A event families:
  - `tachycardia_event`
  - `bradycardia_event`
  - `abrupt_change_event`
- Stage 4B default model: `hist_gbdt_irregular`
- Stage 4B comparison baseline: `irregular_rule_baseline`
- Stage 4C default model: `isolation_forest_anomaly`
- Final unified Stage 4 output: `stage4_full`
- Main outputs:
  - `outputs/{dataset}_stage4_event_predictions.csv`
  - `outputs/{dataset}_stage4_event_metrics.csv`
  - `outputs/{dataset}_stage4_irregular_predictions.csv`
  - `outputs/{dataset}_stage4_irregular_metrics.csv`
  - `outputs/{dataset}_stage4_anomaly_predictions.csv`
  - `outputs/{dataset}_stage4_anomaly_metrics.csv`
  - `outputs/{dataset}_stage4_full_predictions.csv`
  - `outputs/{dataset}_stage4_full_metrics.csv`
- Complete vs limited:
  - complete for the repository’s current CPU-first Stage 4 scope
  - still proxy-label-based and non-clinical
  - unified suspiciousness is useful for stratification and auditability, but not demonstrated as a better ranking layer than the Stage 3-only baseline

### Stage 5

- Implemented: yes
- Validated: yes
- Default model family: `stage5_resp_multitask_cnn_v1`
- Baseline comparison: `resp_surrogate_fusion_baseline`
- Label source: direct chest `Resp` waveform already present in `PPG-DaLiA` and `WESAD`
- Main outputs:
  - `outputs/{dataset}_stage5_predictions.csv`
  - `outputs/{dataset}_stage5_metrics.csv`
  - `outputs/{dataset}_stage5_tuning_results.csv`
  - `outputs/models/stage5/{dataset}_stage5_resp_multitask_cnn_v1_best.pt`
  - `outputs/models/stage5/{dataset}_stage5_resp_multitask_cnn_v1_best_config.json`
- Complete vs limited:
  - complete for the repository’s current practical CPU-first Stage 5 scope
  - still not a joint HR+RR retraining stack
  - no TCN path
  - no clinical respiration validation

## Key Default Paths And Safe Carry-Forward Conclusions

Current defaults worth preserving unless there is new evidence:

- Stage 1 default HR estimator: `stage1_frequency`
- Stage 2 default beat / IBI path: `enhanced`
- Stage 2 official beat-quality operating point: `enhanced_beat_quality`
- Stage 3 accuracy-oriented default: `gated_stage3_ml_logreg`
- Stage 3 robust-output prototype: `robust_stage3c2_policy`
- Stage 4B default irregular model: `hist_gbdt_irregular`
- Stage 4C default anomaly model: `isolation_forest_anomaly`
- Stage 5 default respiration model family: `stage5_resp_multitask_cnn_v1`

Safe carry-forward conclusions:

- Stage 1: `stage1_frequency` remains the best Stage 1 HR path.
- Stage 2: `enhanced` is a meaningful improvement over the Stage 2 baseline and is the correct foundation for later stages.
- Stage 3: `gated_stage3_ml_logreg` is the accuracy-oriented default; `robust_stage3c2_policy` is a separate robust-output prototype, not a replacement default.
- Stage 4: the strongest supported value is the richer output layer and the standalone anomaly signal; do not claim the unified suspiciousness layer is a better ranking model than Stage 3-only quality.
- Stage 5: the tuned CNN clearly beats the classical respiration surrogate baseline on both datasets, RR is usable on high-quality segments, and Stage 5 does not change HR outputs.

## Repository Architecture Map

### Data Loading

- `src/heart_rate_cnn/data/base.py` defines the loader interface.
- `src/heart_rate_cnn/data/ppg_dalia.py` and `src/heart_rate_cnn/data/wesad.py` load:
  - wrist `BVP`
  - chest `ECG`
  - optional chest `Resp`
  - optional wrist `ACC`
- `SubjectRecord` in `src/heart_rate_cnn/types.py` is the subject-level contract.

### Preprocessing And Windowing

- `src/heart_rate_cnn/preprocess.py` handles:
  - resampling
  - Stage 1 PPG preprocessing
  - optional DWT denoising
  - common-duration trimming across PPG / ECG / ACC / Resp
  - ECG peak detection
  - sliding `WindowSample` creation
- Default short-window timeline for Stages 0, 1, 3, and 4:
  - `8 s` windows
  - `2 s` step

### HR Pipeline

- Stage 0 baseline: `src/heart_rate_cnn/baseline_hr.py`
- Stage 1: `src/heart_rate_cnn/stage1_hr.py`
  - frequency path
  - time path
  - simple rule-based fusion

### Beat / IBI / PRV Pipeline

- `src/heart_rate_cnn/stage2_beat.py`
  - beat detection
  - IBI extraction
  - IBI cleaning
  - beat-quality proxy
  - time-domain PRV features
- Default Stage 2 analysis windows:
  - `60 s` windows
  - `30 s` step

### Quality Pipeline

- `src/heart_rate_cnn/stage3_quality.py`
  - ECG-backed pseudo-target generation for quality labels
  - rule-based quality scoring
  - logistic-regression ML gating
  - train-only threshold selection
  - robust HR policy with local beat fallback and limited hold behavior

### Stage 4 Pipeline

- `src/heart_rate_cnn/stage4_features.py`
  - reusable Stage 3-aware source package
  - reusable Stage 4 shared feature package
  - output-scope routing
  - cache management
- `src/heart_rate_cnn/stage4_events.py`
  - quality-gated rule-based event detection
- `src/heart_rate_cnn/stage4_irregular.py`
  - irregular screening feature use, proxy labels, and default HistGBDT screening path
- `src/heart_rate_cnn/stage4_anomaly.py`
  - IsolationForest anomaly scoring
- `src/heart_rate_cnn/stage4_full.py`
  - row-wise unified Stage 4 output and comparison metrics

### Stage 5 Pipeline

- `src/heart_rate_cnn/stage5_respiration.py`
  - chest-Resp-derived RR references
  - RIAV / RIFV / RIBV baseline
  - Stage 5 cache-backed window package
  - tuning loop
  - 1D CNN multitask RR model
- `src/heart_rate_cnn/stage5_multitask.py`
  - aggregation of Stage 4 context into Stage 5 respiration windows
  - final multitask prediction frame
  - Stage 5 metrics

### Script Flow

- Stages 0–3 are mostly self-contained one-shot evaluation scripts.
- Stage 4 uses reusable source/feature preparation plus canonical vs validation output routing.
- Stage 5 builds on Stage 4 default context, then prepares cached Stage 5 window packages for tuning and final inference.

## Source-Of-Record Outputs

### Canonical Outputs

Canonical outputs live in `outputs/` and use unsuffixed filenames. Treat these as the source of record.

Examples:

- `outputs/ppg_dalia_stage4_full_predictions.csv`
- `outputs/wesad_stage4_full_metrics.csv`
- `outputs/ppg_dalia_stage5_predictions.csv`
- `outputs/wesad_stage5_metrics.csv`

### Validation Outputs

Bounded or analysis-only outputs live under `outputs/validation/<label>/`.

Examples already present in this workspace:

- `outputs/validation/bounded_medium6_seed42/`
- `outputs/validation/fusion_balanced_v1/`
- `outputs/validation/fusion_balanced_v1_canonical/`

Do not treat those as canonical unless you intentionally rerun and promote a result.

### Cache Outputs

Reusable cache artifacts live under:

- `outputs/cache/stage4/<dataset>/`
- `outputs/cache/stage5/<dataset>/`

These are not source-of-record outputs. They are reproducible intermediates.

What they currently record:

- exact train/eval subject lists
- selected thresholds or upstream package keys
- row counts
- schema version
- build timestamp

Current canonical manifests in this workspace record these Stage 4/5 splits:

- `ppg_dalia`
  - train: `S10,S11,S12,S13,S15,S2,S3,S5,S7,S9`
  - eval: `S1,S14,S4,S6,S8`
- `wesad`
  - train: `S11,S13,S14,S15,S17,S2,S3,S5,S7,S9`
  - eval: `S10,S16,S4,S6,S8`

## Current Evidence-Backed Conclusions

What the repository can honestly claim now:

- Stage 3 provides a practical, reproducible quality-aware HR layer with a validated ML gate and an auditable robust-HR prototype.
- Stage 4 adds a richer physiological interpretation layer on top of Stage 3:
  - events
  - irregular pulse suspicion
  - anomaly score
  - unified suspiciousness and audit fields
- Stage 4 proxy targets are repository-specific and non-clinical.
- Canonical Stage 4 evidence is mixed:
  - the standalone anomaly layer is the strongest Stage 4 component
  - on `PPG-DaLiA`, anomaly ranking beat the Stage 3-only suspiciousness baseline
  - on `WESAD`, anomaly was slightly below the Stage 3-only baseline
  - the unified Stage 4 suspiciousness layer did not beat the Stage 3-only baseline on canonical runs
- Stage 5 is the clearest new capability:
  - direct chest-Resp-derived references are available in both supported datasets
  - the tuned CNN clearly beats the classical RIAV/RIFV/RIBV baseline on both datasets
  - high-quality RR estimates are usable on both datasets
  - `PPG-DaLiA` is stronger than `WESAD`
  - Stage 5 does not materially change HR outputs because it inherits Stage 4 HR context unchanged

Useful canonical Stage 5 summary points:

- `PPG-DaLiA`, high-quality eval:
  - baseline MAE `9.16 bpm`
  - CNN MAE `2.37 bpm`
- `WESAD`, high-quality eval:
  - baseline MAE `6.41 bpm`
  - CNN MAE `2.85 bpm`

## Current Caveats And Pitfalls

- Proxy labels are not clinical truth.
  - Stage 3 quality labels are ECG-backed pseudo-labels.
  - Stage 4 event and irregular targets are repository-specific proxy targets.
- Stage 4 anomaly is the strongest standalone Stage 4 component, but its ranking gain is not uniform across datasets.
- Stage 4 unified suspiciousness is useful for stratification and auditability, not yet a demonstrated superior ranking layer.
- Stage 5 respiration references come from public chest `Resp`, not clinical adjudication.
- Stage 5 does not jointly retrain HR and RR; it aggregates Stage 4 outputs into respiration windows.
- Long runtimes are normal for Stage 4 canonical builds and especially Stage 5 tuning/training. Use caches.
- Some earlier docs preserve historical wording from the stage in which they were written.
  - `docs/STAGE1_RUNBOOK.md` still contains stage-era wording that no longer reflects the full repo state.
  - `docs/DATASETS.md` and `docs/STAGE0_DATA_CONTRACT.md` are still Stage 0-centric.
  - `pyproject.toml` still describes the project as a Stage 0 foundation.
- The `.local.yaml` dataset configs in this workspace point to absolute local paths. They are not portable as-is.

## Recommended Next Work Directions

Most natural next steps from the current state:

- Stage 5 refinement on harder data:
  - improve `WESAD` respiration robustness
  - test additional Stage 5 confidence policies
  - consider dataset expansion if new public respiration-capable data is added
- Stage 5 modeling extensions:
  - optional TCN or other CPU-feasible temporal upgrades
  - only if they materially improve over the current tuned CNN
- Stage 4 refinement:
  - improve unified suspiciousness ranking without losing interpretability
  - keep anomaly metrics separate and visible
- Label quality and evaluation:
  - add more direct rhythm / respiration reference sources if public-data support expands
- Future planning:
  - Stage 6 should only start after deciding whether Stage 5 is being extended or frozen

## Quick Takeover Checklist

1. Read `README.md`, then `docs/PROJECT_TASKS.md`, then `docs/STAGE4_RUNBOOK.md` and `docs/STAGE5_RUNBOOK.md`.
2. Confirm environment setup:
   - `conda activate HeartRate_env`
   - `pip install -e .`
3. Verify local dataset roots in `configs/datasets/*.local.yaml`.
4. Run `pytest -q` before changing anything.
5. Decide whether you need canonical outputs, validation outputs, or only cached prep.
6. Inspect `outputs/` before rerunning long jobs; canonical files may already exist.
7. Inspect cache manifests under `outputs/cache/stage4/` and `outputs/cache/stage5/` if you need exact split provenance.
8. When rerunning Stage 4 or Stage 5, use `--output-scope validation --output-label ...` for experiments so you do not pollute canonical outputs.
9. Rebuild caches only when code, configs, or raw-data assumptions changed enough to invalidate them.
10. Keep the conservative claims intact:
   - Stage 4 unified suspiciousness is not proven superior to Stage 3-only ranking
   - Stage 5 respiration is useful and reproducible, but not clinically validated
