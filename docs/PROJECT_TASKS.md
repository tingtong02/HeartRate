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
- Stage 2 and later stages have not started yet

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
- Stage 3A window-level SQI / quality gating is implemented and validated
- Stage 3B1 motion-aware strengthening and Stage 3B2 DWT denoising are implemented as exploratory comparison branches
- Stage 3C1 minimum viable beat-level quality proxy is implemented and validated through the Stage 2 pipeline outputs
- Stage 3C1.1 threshold / retention refinement is implemented as analysis-only operating-point comparison
- `enhanced_beat_quality` remains the official baseline operating point for the current beat-quality branch
- `enhanced_beat_quality_refined` is analysis-only and is not the new default threshold
- Full Stage 3 remains incomplete: beat-level SQI closure, robust HR update policy, adaptive filtering, and other roadmap items below are still not fully implemented

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

Algorithms:
- Rule-based tachycardia / bradycardia / abrupt-change event detection
- TCN-based temporal event detector
- XGBoost-based irregular pulse screening
- Isolation Forest / Autoencoder anomaly scoring

Deliverables:
- HR event detector
- Irregular pulse screening module
- Anomaly score output

Acceptance criteria:
- Event detection is stable and does not overfire on noisy segments
- Screening outputs remain quality-gated

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
