# HeartRate_CNN

`HeartRate_CNN` is a CPU-first, public-dataset PPG physiological analysis repository.

The current repository state implements a practical Stage 0–5 stack for:

- heart-rate estimation
- beat / IBI / time-domain PRV analysis
- quality-aware HR gating and a robust HR policy prototype
- Stage 4 suspicious-segment outputs:
  - rule-based HR events
  - irregular pulse screening
  - anomaly scoring
  - unified suspiciousness / audit fields
- Stage 5 CNN-based respiration estimation and a multitask output interface

Supported datasets:

- `PPG-DaLiA`
- `WESAD`

This is a research/reproducibility repository, not a clinical product.

## Current Repository Status

The repository is complete through Stage 5 for its current practical CPU-first scope.

| Stage | Current status | Main default / interpretation |
| --- | --- | --- |
| Stage 0 | Implemented and stable | loading, alignment, ECG-backed HR references, smoke path |
| Stage 1 | Implemented and validated | `stage1_frequency` is the best Stage 1 HR baseline |
| Stage 2 | Implemented and validated | `enhanced` is the practical default beat / IBI / PRV path |
| Stage 3 | Practically complete for current scope | `gated_stage3_ml_logreg` is the accuracy-oriented default; `robust_stage3c2_policy` is a robust-output prototype |
| Stage 4 | Complete for current CPU-first scope | interpretable event / irregular / anomaly layer with canonical vs validation output separation |
| Stage 5 | Complete for current practical CPU-first scope | `stage5_resp_multitask_cnn_v1` is the default respiration model |

## Current Default Paths

Current defaults and safe interpretations:

- Stage 1 default: `stage1_frequency`
- Stage 2 default: `enhanced`
- Stage 2 official beat-quality operating point: `enhanced_beat_quality`
- Stage 2 analysis-only operating point: `enhanced_beat_quality_refined`
- Stage 3 accuracy-oriented default: `gated_stage3_ml_logreg`
- Stage 3 robust-output prototype: `robust_stage3c2_policy`
- Stage 3 analysis-only paths:
  - `enhanced_beat_quality_refined`
  - `robust_stage3c2_policy_refined`
- Stage 4B default irregular model: `hist_gbdt_irregular`
- Stage 4C default anomaly model: `isolation_forest_anomaly`
- Stage 5 default respiration model: `stage5_resp_multitask_cnn_v1`

Safest carry-forward conclusions from the canonical outputs:

- `stage1_frequency` remains the strongest Stage 1 HR method.
- Stage 2 `enhanced` is the correct beat / IBI foundation for later stages.
- Stage 3 `gated_stage3_ml_logreg` is the current default when you want the best accuracy-oriented quality-aware HR path.
- Stage 3 `robust_stage3c2_policy` is useful when you want auditable fallback/hold behavior, but it is not the accuracy-oriented default.
- Stage 4 adds a richer physiological output layer beyond Stage 3 quality gating, but the current unified Stage 4 suspiciousness score is not yet a clearly better ranking baseline than the simple Stage 3-only suspiciousness baseline.
- Stage 4 standalone evidence is mixed by dataset:
  - on `PPG-DaLiA`, anomaly is the strongest standalone Stage 4 signal
  - on `WESAD`, irregular screening is the strongest standalone Stage 4 signal
- Stage 5’s tuned CNN respiration model clearly outperforms the classical RIAV / RIFV / RIBV surrogate baseline on both datasets, with stronger results on `PPG-DaLiA`.

## What Is Implemented

### Stage 0–1: HR Foundation

- subject-level loading for `PPG-DaLiA` and `WESAD`
- common-duration trimming across available channels
- sliding-window segmentation
- ECG-backed window-level reference HR
- Stage 0 minimal Welch baseline
- Stage 1 enhanced frequency HR path
- Stage 1 time-domain pulse path
- simple rule-based Stage 1 fusion

### Stage 2: Beat / IBI / PRV Foundation

- beat detection on preprocessed PPG
- IBI extraction and cleaning
- time-domain PRV features:
  - mean / median IBI
  - mean HR from IBI
  - SDNN
  - RMSSD
  - pNN50
  - IBI CV
- rule-based beat-quality proxy
- threshold / retention analysis for beat-quality operating points

### Stage 3: Quality-Aware HR Layer

- rule-based window-level quality scoring
- ML quality gating with `LogisticRegression`
- train-only threshold selection and refinement
- motion-aware comparison branch
- DWT-denoised comparison branch
- robust HR policy with:
  - local 8 s beat-derived fallback
  - limited auditable `hold_previous`
  - subject-boundary reset behavior

### Stage 4: Event / Irregular / Anomaly Layer

- Stage 4A quality-gated rule-based HR event detection:
  - `tachycardia_event`
  - `bradycardia_event`
  - `abrupt_change_event`
- Stage 4B quality-gated irregular pulse screening on a shared feature frame
- Stage 4C quality-gated anomaly scoring with `IsolationForest`
- unified Stage 4 row-wise output with:
  - event flags
  - irregular flags / scores
  - anomaly scores / flags
  - combined suspiciousness
  - audit fields
- reusable Stage 4 source / feature caches
- explicit canonical vs validation output routing

### Stage 5: CNN Respiration And Multitask Interface

- direct RR reference derivation from the public chest `Resp` waveform already present in both supported datasets
- RIAV / RIFV / RIBV surrogate-fusion baseline
- tuned 1D CNN multitask respiration model with:
  - RR regression head
  - respiration-validity / confidence head
- cache-backed Stage 5 window preparation
- Stage 5 multitask predictions that carry forward Stage 4 context:
  - HR
  - quality
  - motion
  - events
  - irregular pulse flag
  - anomaly score
  - Stage 4 suspiciousness
  - respiration outputs

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate HeartRate_env
pip install -e .
```

Run the test suite:

```bash
pytest -q
```

Key dependencies:

- `numpy`, `scipy`, `pandas`, `scikit-learn`
- `PyWavelets`
- `joblib`
- `pytest`
- `torch` for Stage 5

## Dataset Setup

Public datasets are not committed to this repository.

Supported datasets:

- `PPG-DaLiA`
- `WESAD`

Recommended local layout:

```text
HeartRate_CNN/
  dataset/
    PPG_DaLiA/
      S1/
        S1.pkl
      ...
    WESAD/
      S2/
        S2.pkl
      ...
```

Tracked dataset templates:

- `configs/datasets/ppg_dalia.yaml`
- `configs/datasets/wesad.yaml`

Local workflow:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Then edit the local config:

```yaml
dataset:
  root_dir: /path/to/dataset
  subject_include: null
```

Optional subject filter:

```yaml
dataset:
  subject_include: ["S1", "S2"]
```

Notes:

- `*.local.yaml` files are ignored by Git.
- The current workspace already contains local configs with machine-specific absolute paths.
- Loaders expect the official wrist/chest pickle layout with:
  - wrist `BVP`
  - chest `ECG`
  - optional wrist `ACC`
  - optional chest `Resp`

## How To Run

### Stage 0

Synthetic smoke path:

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

Note: `configs/eval/hr_baseline.yaml` currently sets `output.save_csv: false`, so Stage 0 mainly prints metrics unless you enable CSV saving.

### Stage 1

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

### Stage 2

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

### Stage 3

Rule-based Stage 3 baseline:

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

### Stage 4

Prepare reusable Stage 4 caches:

```bash
python scripts/prepare_stage4_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 4A event baseline:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 4B irregular baseline:

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

Rebuild Stage 4 caches:

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

The Stage 5 scripts use `configs/eval/hr_stage5.yaml` plus the CNN overlay `configs/eval/hr_stage5_cnn.yaml` by default.

## Results Site

This repository now includes a static, English-first technical dashboard under `web/`.

The browser does not read the large raw CSV artifacts directly. Instead, build compact JSON snapshots first:

```bash
conda run -n HeartRate_env python scripts/build_results_site_data.py
```

This exports site data to:

- `web/public/data/`

The frontend source lives in:

- `web/src/`

To run or build the site, install a local Node.js toolchain, then use:

```bash
cd web
npm install
npm run dev
```

```bash
npm run build
```

Use the dashboard to present:

- Stage 0–5 capability progression
- canonical Stage 1–5 metrics
- Stage 4 suspicious-segment evidence and caveats
- Stage 5 respiration and multitask results
- validation / analysis-only experiments without mixing them into canonical conclusions

## Output Conventions

### Canonical Outputs

Canonical source-of-record outputs live in `outputs/` and use unsuffixed filenames.

Examples:

- `outputs/ppg_dalia_stage4_full_predictions.csv`
- `outputs/wesad_stage4_full_metrics.csv`
- `outputs/ppg_dalia_stage5_predictions.csv`
- `outputs/wesad_stage5_metrics.csv`

### Validation Outputs

Bounded or analysis-only outputs live under:

- `outputs/validation/<label>/`

Use validation scope when you do not want to overwrite the canonical results.

### Cache Artifacts

Reusable caches live under:

- `outputs/cache/stage4/<dataset>/`
- `outputs/cache/stage5/<dataset>/`

These are intermediate artifacts, not source-of-record outputs.

### Useful Canonical Files

Stage 1:

- `outputs/{dataset}_stage1_predictions.csv`
- `outputs/{dataset}_stage1_metrics.csv`

Stage 2:

- `outputs/{dataset}_stage2_beats.csv`
- `outputs/{dataset}_stage2_features.csv`
- `outputs/{dataset}_stage2_metrics.csv`
- `outputs/{dataset}_stage2_beat_quality.csv`
- `outputs/{dataset}_stage2_beat_quality_sweep.csv`

Stage 3:

- `outputs/{dataset}_stage3_predictions.csv`
- `outputs/{dataset}_stage3_metrics.csv`
- `outputs/{dataset}_stage3_enhanced_predictions.csv`
- `outputs/{dataset}_stage3_enhanced_metrics.csv`
- threshold / operating-point / policy sweep CSVs

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

## Key Current Conclusions

These are the strongest evidence-backed takeaways from the current canonical outputs.

### Stage 1

- `stage1_frequency` is the strongest Stage 1 HR baseline on both supported datasets.
- Canonical Stage 1 best rows:
  - `PPG-DaLiA`: MAE `9.52`, RMSE `18.88`, `r 0.768`
  - `WESAD`: MAE `10.81`, RMSE `22.36`, `r 0.535`

### Stage 2

- Stage 2 `enhanced` improves the key beat / IBI metrics and is the correct default foundation for later stages.
- Mean / median IBI style features are more robust than higher-variance variability features.

### Stage 3

- `gated_stage3_ml_logreg` is the current accuracy-oriented default Stage 3 path.
- `robust_stage3c2_policy` is best understood as a robust-output prototype path with auditable fallback and hold behavior.

### Stage 4

- Stage 4 adds practical value as a richer suspicious-segment layer on top of Stage 3.
- The canonical evidence is mixed by standalone signal family:
  - `PPG-DaLiA` eval, proxy-abnormal ranking:
    - Stage 3-only baseline AUPRC `0.683`
    - Stage 4 anomaly AUPRC `0.690`
    - Stage 4 unified default AUPRC `0.658`
  - `WESAD` eval, proxy-abnormal ranking:
    - Stage 3-only baseline AUPRC `0.610`
    - Stage 4 irregular default AUPRC `0.670`
    - Stage 4 anomaly AUPRC `0.600`
    - Stage 4 unified default AUPRC `0.600`
- The safest current Stage 4 statement is:
  - standalone Stage 4 signals are useful
  - the best standalone signal differs by dataset
  - the current unified suspiciousness fusion is helpful for interpretation and stratification, but it is not yet a clearly superior ranking baseline to the simple Stage 3-only suspiciousness baseline

### Stage 5

- The tuned Stage 5 CNN clearly outperforms the classical respiration surrogate baseline on both datasets.
- Canonical high-quality eval subset:
  - `PPG-DaLiA`
    - baseline MAE `9.16 bpm`
    - CNN MAE `2.37 bpm`
    - CNN `within_3_bpm_rate 0.744`
  - `WESAD`
    - baseline MAE `6.41 bpm`
    - CNN MAE `2.85 bpm`
    - CNN `within_3_bpm_rate 0.591`
- Stage 5 carries forward Stage 4 HR outputs unchanged, so it does not materially degrade the existing HR layer.

## Current Limitations

- Public-data evaluation only; no private or device-specific data.
- Stage 3 quality labels are pseudo-labels, not clinical truth.
- Stage 4 event and irregular targets are proxy labels, not clinical labels.
- Stage 4 is a post-Stage-3 physiological output layer, not a replacement HR estimator.
- The current Stage 4 unified suspiciousness layer is not yet a proven superior ranking baseline to Stage 3-only quality suspiciousness.
- Stage 5 respiration references come from public chest `Resp` signals and repository-specific derivation logic, not clinical adjudication.
- Stage 5 is useful and reproducible, but it is not clinical respiration validation.
- The current Stage 5 implementation is CPU-first 1D CNN only:
  - no TCN path
  - no end-to-end joint HR+RR retraining
- Frequency-domain and nonlinear HRV feature families are still not implemented in the main pipeline.

## Detailed Docs

Detailed repository docs:

- [Project tasks](docs/PROJECT_TASKS.md)
- [Stage 0 runbook](docs/STAGE0_RUNBOOK.md)
- [Stage 1 runbook](docs/STAGE1_RUNBOOK.md)
- [Stage 2 runbook](docs/STAGE2_RUNBOOK.md)
- [Stage 3 runbook](docs/STAGE3_RUNBOOK.md)
- [Stage 4 runbook](docs/STAGE4_RUNBOOK.md)
- [Stage 5 runbook](docs/STAGE5_RUNBOOK.md)
- [Usage guide](docs/USAGE_GUIDE.md)
- [Project handoff](docs/HANDOFF.md)

If you are taking over the repository, start with:

1. this `README.md`
2. `docs/HANDOFF.md`
3. `docs/STAGE4_RUNBOOK.md`
4. `docs/STAGE5_RUNBOOK.md`
