# HeartRate_CNN Project Tasks

## 1. Project Overview

This project aims to build a research-grade PPG-based heart rate analysis framework using only public datasets.

Current scope:
- No real IoT device data available
- No embedded deployment constraints
- Focus on accurate and robust heart rate estimation
- SpO2 is out of scope for now
- Development will be performed mainly with Codex
- Main workspace: `~/learning/HeartRate_CNN`

Target capabilities:
- Robust heart rate estimation
- Beat detection and IBI extraction
- Signal quality assessment (SQI)
- Motion artifact detection and suppression
- Heart rate abnormal event detection
- Irregular pulse / rhythm anomaly screening
- PRV / HRV feature extraction
- Respiration rate estimation
- Unified multitask output interface

---

## 2. Project Objectives

### Primary objective
Build a robust PPG-based heart rate estimation pipeline that performs well across public datasets, especially under noisy and motion-corrupted conditions.

### Secondary objectives
- Build beat-level physiological analysis capability
- Build quality-aware output logic
- Build event/screening modules on top of beat-level features
- Build a modular research framework for future extension

---

## 3. In-Scope Tasks

### Core mandatory tasks
1. Robust heart rate estimation
2. Signal quality assessment (SQI)
3. Motion artifact detection / suppression
4. Beat-level pulse detection
5. Heart rate confidence output
6. Invalid / hold / reject logic for low-quality segments

### Recommended tasks
7. Heart rate abnormal event detection
8. Irregular pulse / rhythm anomaly screening
9. PRV / HRV feature extraction
10. Respiration rate estimation
11. Unified multitask output interface

---

## 4. Out-of-Scope Tasks

1. SpO2 estimation
2. Blood pressure estimation
3. Clinical-grade arrhythmia diagnosis
4. Embedded optimization or MCU deployment
5. Hardware-specific optical compensation requiring real device measurements

---

## 5. Dataset Constraints

Only public datasets are allowed at the current stage.

Candidate datasets include:
- PPG-DaLiA
- WESAD
- BIDMC / MIMIC-derived PPG datasets
- Public datasets with respiration references
- Public datasets with rhythm irregularity labels or ECG references

All experiments should support subject-wise train/validation/test split where applicable.

---

## 6. Development Stages

### Stage 0: Data and evaluation foundation
Goals:
- Build unified dataset loading
- Build resampling / segmentation / alignment pipeline
- Build evaluation framework
- Establish reproducible experiment protocol

Deliverables:
- Dataset loaders
- Unified sample format
- Evaluation scripts
- Baseline benchmark report

Acceptance criteria:
- At least 2 public datasets fully loadable
- HR baseline can run end-to-end
- Metrics are reproducible

---

### Stage 1: Robust heart rate baseline
Goals:
- Build a stronger baseline than simple S-G + FFT
- Implement dual-path HR estimation and fusion

Current status:
- Stage 1 first-round baseline is implemented with enhanced preprocessing, an enhanced Welch frequency chain, a time-domain peak chain, and minimal rule-based fusion
- Later stages have now also been implemented in narrower repository-specific forms; see the Stage 2 and Stage 3 sections below for current status

Algorithms:
- Detrending
- Zero-phase bandpass filtering
- Savitzky-Golay smoothing
- Welch PSD
- STFT-based dominant frequency tracking
- Adaptive peak detection
- Rule-based or gradient-boosted fusion

Deliverables:
- Window-level HR estimator
- HR confidence score
- Baseline result tables

Acceptance criteria:
- Outperforms plain FFT baseline
- Stable HR outputs under clean and mildly noisy conditions

---

### Stage 2: Beat detection and PRV/HRV foundation
Goals:
- Move from average HR to beat-level analysis
- Extract reliable IBI sequences

Current status:
- Stage 2 implementation is still focused only on beat detection, IBI extraction/cleaning, and basic time-domain PRV/HRV feature evaluation
- Stage 2 enhancement-round results improved the highest-priority metrics on both PPG-DaLiA and WESAD: beat F1 increased and IBI RMSE decreased relative to the Stage 2 baseline variant
- Mean / median IBI style outputs are now clearly more usable than in the first round, while variability features still benefit from further beat detection and IBI cleaning improvements
- Stage 2 is now strong enough to serve as the foundation for Stage 3, so deeper Stage 2 tuning is lower priority than beginning the next stage
- Stage 3 has started through a narrow quality-aware branch while later stages remain unstarted

Algorithms:
- Adaptive peak detection
- Derivative-assisted beat localization
- Template-assisted beat refinement
- IBI cleaning
- Time-domain PRV features
- Frequency-domain PRV features
- Nonlinear features: Poincaré, sample entropy

Deliverables:
- Beat detector
- IBI extraction pipeline
- PRV/HRV feature extraction module

Acceptance criteria:
- Beat detection reaches usable precision/recall
- IBI error is acceptable on high-quality segments

---

### Stage 3: SQI and motion robustness
Goals:
- Make the system quality-aware and robust under motion/noise

Current status:
- Stage 3 is now practically complete for this repository's CPU-first implemented scope
- Completed repository Stage 3 scope includes:
  - Stage 3A window-level SQI / quality gating
  - rule-based and ML-gated Stage 3 HR quality paths
  - train-only threshold selection and operating-point refinement for the ML gate
  - Stage 3C1 minimum viable beat-level quality proxy via Stage 2 outputs
  - Stage 3C1.1 beat-quality threshold / retention analysis
  - Stage 3C2 robust-HR policy prototype with local 8 s beat fallback and limited auditable hold behavior
  - Stage 3C2.1 robust-HR policy operating-point refinement
- Exploratory Stage 3 add-ons include:
  - Stage 3B1 motion-aware strengthening
  - Stage 3B2 DWT denoising comparison
- Current default / analysis interpretation:
  - `gated_stage3_ml_logreg` is the current accuracy-oriented default Stage 3 path
  - `robust_stage3c2_policy` is the current robust-output prototype path
  - `enhanced_beat_quality_refined` is analysis-only
  - `robust_stage3c2_policy_refined` is analysis-only
- Deferred Stage 3 roadmap items include:
  - full beat-level SQI closure beyond the current minimum viable proxy
  - SSA denoising
  - adaptive filtering such as `NLMS` / `RLS`
  - deep-learning denoising
  - broader non-CPU-first Stage 3 extensions

Algorithms:
- Window-level SQI classifier
- Beat-level SQI scoring
- Motion detection from signal statistics and optional IMU
- DWT / SSA denoising
- NLMS / RLS adaptive filtering when IMU is available
- Optional denoising autoencoder / 1D U-Net

Deliverables:
- SQI model
- Motion artifact detector
- Signal validity logic
- Robust HR update policy

Acceptance criteria:
- Low-quality windows are correctly rejected or downgraded
- HR robustness improves under motion-corrupted segments

---

### Stage 4: Event detection and irregular pulse screening
Goals:
- Detect clinically relevant but non-diagnostic events/suspicions

Current status:
- Stage 4 is now complete for this repository's current CPU-first implemented scope
- Implemented Stage 4 scope includes:
  - Stage 4A quality-gated, rule-based HR event detection
  - Stage 4B quality-gated irregular pulse screening on a shared Stage 4 feature frame
  - Stage 4C quality-gated anomaly scoring with `IsolationForest`
  - a final unified Stage 4 row-wise output layer with combined suspiciousness
  - explicit Stage 3-only versus Stage 4 comparison rows on a proxy abnormal target
- Implemented event families:
  - `tachycardia_event`
  - `bradycardia_event`
  - `abrupt_change_event`
- Current repository default / interpretation:
  - Stage 4A event defaults remain source-configurable and quality-gated
  - Stage 4B default model is `HistGradientBoostingClassifier`
  - Stage 4C default model is `IsolationForest`
  - the final Stage 4 suspiciousness layer is interpretable, auditable, and non-diagnostic
  - cache-backed reusable Stage 4 source / feature preparation and explicit canonical-vs-validation output routing are now implemented
- Evaluation remains proxy-based and repository-specific:
  - Stage 4A uses ECG-backed proxy HR event targets
  - Stage 4B uses ECG/reference-side irregularity proxy labels
  - final Stage 4 comparison rows use `proxy_abnormal_target = proxy_hr_event_target_any OR screening_proxy_target`
  - current best-supported evidence shows Stage 4C anomaly ranking outperforming the Stage 3-only quality baseline on stronger bounded validation, while the unified suspiciousness layer still remains more conservative and has not yet demonstrated a ranking gain over the Stage 3-only baseline
- Deferred beyond current Stage 4 scope:
  - clinical labels / diagnosis
  - deep sequence models
  - autoencoder anomaly models
  - broader non-CPU-first Stage 4 extensions

Algorithms:
- Rule-based tachycardia / bradycardia / abrupt-change event detection
- HistGradientBoosting-based irregular pulse screening
- Isolation Forest anomaly scoring
- interpretable combined suspiciousness fusion over event / irregular / anomaly outputs

Deliverables:
- HR event detector
- Irregular pulse screening module
- Anomaly score output
- Unified Stage 4 output interface

Acceptance criteria:
- Event detection is stable and does not overfire on noisy segments
- Screening outputs remain quality-gated
- Stage 4 adds a more useful suspicious-segment output layer than Stage 3 quality gating alone without making clinical claims

---

### Stage 5: Respiration and multitask fusion
Goals:
- Expand from HR system to a multitask physiological analysis framework

Algorithms:
- RIAV / RIFV / RIBV respiration estimation
- Fusion model for respiration rate
- CNN + TCN multitask architecture
- Unified feature and output interface

Deliverables:
- Respiration estimation module
- Unified multitask inference interface

Acceptance criteria:
- RR estimates are usable on high-quality segments
- Multitask expansion does not significantly degrade HR performance

---

## 7. Output Interface Requirements

The unified system should eventually output:

- `hr`
- `hr_confidence`
- `signal_quality`
- `motion_flag`
- `validity_flag`
- `beat_positions`
- `ibi_series`
- `prv_features`
- `hr_event_flag`
- `irregular_pulse_flag`
- `anomaly_score`
- `resp_rate`
- `resp_confidence`

Not all outputs are required from Stage 1. They should be introduced progressively by stage.

---

## 8. Evaluation Protocol

### HR estimation
Metrics:
- MAE
- RMSE
- MAPE
- Correlation
- Bland-Altman analysis

### Beat / IBI
Metrics:
- Precision
- Recall
- F1
- IBI MAE / RMSE

### Classification / detection
Metrics:
- Accuracy
- Precision
- Recall
- F1
- AUROC
- AUPRC

### Evaluation conditions
- Subject-wise split whenever possible
- Report by scenario:
  - clean / rest
  - mild motion
  - heavy motion
- Report both aggregate and dataset-specific results

---

## 9. Engineering Requirements

### Environment
- Conda environment name: `HeartRate_env`

### Code requirements
- Modular package structure
- Reproducible random seeds
- Config-driven experiments
- Clear separation between:
  - data loading
  - preprocessing
  - models
  - evaluation
  - training scripts
  - notebooks

### Repository expectations
Suggested structure:

````text
HeartRate_CNN/
├── configs/
├── data/
├── datasets/
├── docs/
├── evaluation/
├── models/
├── notebooks/
├── preprocess/
├── quality/
├── hr/
├── respiration/
├── events/
├── fusion/
├── scripts/
└── tests/
````

---

## 10. Codex Working Principles

Codex must:

* Work only inside this repository
* Prefer incremental changes
* Keep modules small and testable
* Add docstrings and comments where useful
* Avoid breaking existing scripts
* Update relevant docs when major changes are introduced

Codex should not:

* Add unnecessary dependencies
* Implement embedded-specific optimizations
* Claim unsupported performance
* Skip evaluation code for new algorithms



## 11. Definition of Done

A stage is considered complete only if:

1. Code is implemented
2. Configuration is runnable
3. Evaluation script exists
4. Results are saved
5. Documentation is updated
6. Outputs match the expected interface for that stage



---
