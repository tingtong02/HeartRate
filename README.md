# HeartRate_CNN

HeartRate_CNN is a public-dataset PPG physiological analysis project.  
The current repository state now includes **a complete Stage 5 multitask respiration layer for this repository's current CPU-first scope** on top of the Stage 0, Stage 1, Stage 2, Stage 3, and Stage 4 foundations.

## Stage 3 Closure Status

Stage 3 is now **practically complete for this repository's CPU-first scope**.

- Completed Stage 3 core:
  - window-level SQI / quality gating
  - ML gating with train-only threshold selection and operating-point refinement
  - minimum viable beat-level quality proxy through Stage 2 outputs
  - robust-HR policy prototype with local 8 s beat fallback and limited auditable hold behavior
- Current accuracy-oriented default Stage 3 path:
  - `gated_stage3_ml_logreg`
- Current robust-output prototype path:
  - `robust_stage3c2_policy`
- Analysis-only outputs:
  - `enhanced_beat_quality_refined`
  - `robust_stage3c2_policy_refined`
- Deferred Stage 3 roadmap items:
  - full beat-level SQI closure
  - SSA / adaptive filtering / deep-learning denoising
  - broader non-CPU-first Stage 3 extensions

This closure is repository-specific and should not be read as “every broad original Stage 3 roadmap item is fully implemented.”

## Current Status

Implemented so far:

- unified subject-level data loading for `PPG-DaLiA` and `WESAD`
- unified data structures for subject records and window samples
- minimal preprocessing: resampling, common-duration alignment, and sliding-window segmentation
- subject-wise train/test split
- ECG-derived window-level reference HR
- a minimal frequency-domain HR baseline using Welch PSD
- a Stage 1 enhanced preprocessing path
- a Stage 1 frequency chain based on enhanced Welch analysis
- a Stage 1 time-domain chain based on pulse peak detection
- a minimal rule-based Stage 1 fusion path
- a Stage 1 comparison script that evaluates Stage 0 baseline, Stage 1 frequency, Stage 1 time, and Stage 1 fusion on the same windows
- a Stage 2 beat detection path based on direct peak finding over Stage 1 style preprocessed PPG
- a Stage 2 IBI extraction and cleaning path
- a Stage 2 basic time-domain PRV/HRV feature path
- a Stage 2 evaluation script that compares `baseline` and `enhanced` beat / IBI / feature pipelines on the same analysis windows
- a validated minimum viable Stage 3C1 beat-level quality proxy layered on top of the Stage 2 enhanced beat / IBI path
- a Stage 2 threshold / retention analysis path for the rule-based beat-quality proxy, with an analysis-only `enhanced_beat_quality_refined` operating point
- optional lightweight Stage 2 worst-window CSV summaries for error review
- a Stage 3 round-1 rule-based quality-gating baseline for `stage1_frequency`
- a Stage 3 evaluation script that compares ungated and quality-gated `stage1_frequency` on the same Stage 1 windows
- a Stage 3 enhancement path with lightweight ML gating and motion-aware strengthened comparison on the same Stage 1 windows
- a Stage 3B2 DWT-denoised comparison branch inside the same Stage 3 enhanced runner
- a narrow Stage 3C2 robust-HR policy branch that adds local 8 s beat-derived fallback and a limited auditable `hold_previous` action on top of the same Stage 3 enhanced outputs
- a narrow Stage 3C2.1 operating-point refinement path that keeps `robust_stage3c2_policy` as baseline and adds an analysis-only `robust_stage3c2_policy_refined` comparison row
- a Stage 4A baseline for quality-gated rule-based HR event detection with tachycardia, bradycardia, and abrupt-change episodes
- a Stage 4B baseline for quality-gated irregular pulse screening with a shared Stage 4 feature frame, a default `HistGradientBoostingClassifier`, and a rule baseline comparison
- a Stage 4C baseline for quality-gated anomaly scoring with `IsolationForest`
- a final Stage 4 unified row-wise output layer with event, irregularity, anomaly, and combined suspiciousness fields
- explicit Stage 3-only versus Stage 4 comparison rows on a proxy abnormal target
- a Stage 5 respiration reference pipeline derived from the public chest `Resp` waveform already present in `PPG-DaLiA` and `WESAD`
- a Stage 5 classical respiration surrogate baseline using `RIAV`, `RIFV`, and `RIBV`
- a tuned CPU-first Stage 5 1D CNN multitask respiration model with RR regression and respiration-validity confidence heads
- a cache-backed Stage 5 window package workflow for repeated tuning and evaluation runs
- a Stage 5 unified multitask output interface that carries forward Stage 4 HR/quality/event/anomaly context and adds respiration outputs
- basic evaluation metrics
- smoke test and pytest coverage

Still not included:

- frequency-domain or nonlinear HRV features
- clinical rhythm diagnosis
- TCN-based respiration modeling
- end-to-end joint HR+RR retraining
- clinical respiration validation

Current Stage 3 scope should now be treated as practically complete for this repository's narrow, CPU-first quality-aware HR layer.

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate HeartRate_env
pip install -e .
```

## Dataset Setup

Public datasets are **not** committed with this repository. You need to download them manually.

Recommended local placement:

```text
HeartRate_CNN/
  dataset/
    PPG_DaLiA/
    WESAD/
```

The `dataset/` directory is ignored by Git.

Tracked dataset config templates keep `root_dir: ""`.  
Create local configs first:

```bash
cp configs/datasets/ppg_dalia.yaml configs/datasets/ppg_dalia.local.yaml
cp configs/datasets/wesad.yaml configs/datasets/wesad.local.yaml
```

Then edit the local config you want to use:

```yaml
dataset:
  root_dir: /path/to/dataset
```

Optional subject filtering:

```yaml
dataset:
  subject_include: ["S1", "S2"]
```

`*.local.yaml` files are ignored by Git.

## How To Run

Run tests:

```bash
pytest
```

Run the synthetic smoke path:

```bash
python scripts/run_stage0_smoke.py
```

Run the real baseline on PPG-DaLiA:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the real baseline on WESAD:

```bash
python scripts/run_stage0_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 1 comparison on PPG-DaLiA:

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 1 comparison on WESAD:

```bash
python scripts/run_stage1_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 2 evaluation on PPG-DaLiA:

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 2 evaluation on WESAD:

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 3 quality-gating baseline on PPG-DaLiA:

```bash
python scripts/run_stage3_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 3 quality-gating baseline on WESAD:

```bash
python scripts/run_stage3_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 3 enhancement-round comparison on PPG-DaLiA:

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 3 enhancement-round comparison on WESAD:

```bash
python scripts/run_stage3_enhanced.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 4A event baseline on PPG-DaLiA:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 4A event baseline on WESAD:

```bash
python scripts/run_stage4_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the Stage 4B irregular screening baseline on PPG-DaLiA:

```bash
python scripts/run_stage4_irregular_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the Stage 4B irregular screening baseline on WESAD:

```bash
python scripts/run_stage4_irregular_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Run the final full Stage 4 pipeline on PPG-DaLiA:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the final full Stage 4 pipeline on WESAD:

```bash
python scripts/run_stage4_full.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Prepare reusable Stage 4 source and feature packages before repeated validation runs:

```bash
python scripts/prepare_stage4_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run a bounded validation without touching canonical output files:

```bash
python scripts/run_stage4_full.py \
  --dataset-config /tmp/ppg_dalia_stage4_medium6.yaml \
  --output-scope validation \
  --output-label bounded_medium6_seed42
```

Stage 4 output routing now follows two explicit scopes:

- canonical runs write unsuffixed source-of-record files to `outputs/`
- bounded validation runs write to `outputs/validation/<label>/`
- reusable cache artifacts live under `outputs/cache/stage4/<dataset>/`

The Stage 4 prep script and all Stage 4 runners will reuse matching cached source and feature packages unless `--rebuild-cache` is passed.

## Stage 5 Workflow

Prepare reusable Stage 5 respiration window packages before repeated tuning runs:

```bash
python scripts/prepare_stage5_sources.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the final full Stage 5 pipeline on PPG-DaLiA:

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Run the final full Stage 5 pipeline on WESAD:

```bash
python scripts/run_stage5_full.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Train and save only the tuned Stage 5 CNN checkpoint:

```bash
python scripts/run_stage5_train_cnn.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

Stage 5 outputs follow the same routing discipline as Stage 4:

- canonical outputs stay in `outputs/`
- bounded or analysis-only outputs go to `outputs/validation/<label>/`
- reusable Stage 5 cache artifacts live under `outputs/cache/stage5/<dataset>/`

Current best-supported Stage 5 conclusion:

- the tuned CNN respiration model clearly outperforms the classical `RIAV` / `RIFV` / `RIBV` surrogate baseline on both datasets
- RR estimates are usable on high-quality segments on both datasets, with stronger results on `PPG-DaLiA`
- Stage 5 does not materially degrade the existing HR pipeline because it carries the Stage 4 HR layer forward unchanged

To reproduce the Stage 2 evaluation:

- create local dataset configs the same way as Stage 0 / Stage 1
- keep the default subject-wise split seed unless you are intentionally running a new comparison
- run `run_stage2_baseline.py` once per dataset; it will emit `baseline`, `enhanced`, the configured `enhanced_beat_quality` baseline operating point, and an analysis-only threshold sweep / refined operating-point comparison when enabled
- use `outputs/{dataset}_stage2_metrics.csv` as the source of record
- use `outputs/{dataset}_stage2_beat_quality_sweep.csv` to inspect the threshold / retention tradeoff explicitly
- optionally set `stage2.debug.save_error_cases: true` in `configs/eval/hr_stage2.yaml` to write a lightweight `outputs/{dataset}_stage2_error_cases.csv`

To reproduce the Stage 1 comparison fairly:

- use the same dataset config style as Stage 0
- keep the default subject-wise split seed unless you are intentionally running a new comparison
- compare all four methods from the same `run_stage1_baseline.py` execution
- use the generated `outputs/{dataset}_stage1_metrics.csv` as the source of record

To reproduce the Stage 3 round-1 gating comparison:

- use the same dataset config style as Stage 0 / Stage 1
- keep the default subject-wise split seed unless you are intentionally running a new comparison
- run `run_stage3_baseline.py` once per dataset
- use `outputs/{dataset}_stage3_metrics.csv` as the source of record

To reproduce the Stage 3 enhancement round fairly:

- use the same dataset config style as Stage 0 / Stage 1 / Stage 3 baseline
- keep the default subject-wise split seed unless you are intentionally running a new comparison
- run `run_stage3_enhanced.py` once per dataset
- use `outputs/{dataset}_stage3_enhanced_metrics.csv` as the source of record
- use `gated_stage3_ml_logreg` as the current accuracy-oriented default Stage 3 comparison path
- `robust_stage3c2_policy` is an additive comparison path inside the same Stage 3 enhanced run
- `robust_stage3c2_policy` is the current robust-output prototype path for coverage / recovery / auditable fallback behavior
- `robust_stage3c2_policy_refined` is analysis-only and must not be read as a silently adopted new default
- the Stage 3C2 beat fallback is computed locally on each Stage 1 8 s window using the existing Stage 2 beat / IBI functions
- `hold_previous` is intentionally limited, resets at subject boundaries, and is reported explicitly in the predictions CSV rather than acting as hidden smoothing
- use `outputs/{dataset}_stage3_enhanced_policy_sweep.csv` to inspect the Stage 3C2.1 error / coverage / jump tradeoff explicitly

## Stage 4 Validation Workflow

Recommended order for Stage 4 reruns:

1. prepare reusable source and feature packages once with `prepare_stage4_sources.py`
2. rerun `run_stage4_full.py` for the same dataset / split / config and confirm cache reuse
3. use `--output-scope validation --output-label <label>` for bounded validation
4. reserve plain `outputs/` for canonical full-dataset runs only

Current output hygiene:

- canonical full-dataset source-of-record outputs live in `outputs/`
- bounded validation outputs live in `outputs/validation/<label>/`
- cache artifacts live in `outputs/cache/stage4/<dataset>/` and are reusable intermediates, not source-of-record outputs

The current stronger bounded validation label is `bounded_medium6_seed42`.

Medium6 subject sets used in the latest Stage 4 refinement round:

- `PPG-DaLiA`: `S1,S10,S11,S12,S13,S14`
  - train: `S11,S12,S13,S14`
  - eval: `S1,S10`
- `WESAD`: `S10,S11,S13,S14,S15,S16`
  - train: `S13,S14,S15,S16`
  - eval: `S10,S11`

Bounded validation remains useful for quick reruns and fusion analysis, but canonical full-dataset Stage 4 reruns have now been completed for both datasets.

Canonical full-dataset subject-wise Stage 4 closure runs used these splits:

- `PPG-DaLiA`
  - train: `S10,S11,S12,S13,S15,S2,S3,S5,S7,S9`
  - eval: `S1,S14,S4,S6,S8`
- `WESAD`
  - train: `S11,S13,S14,S15,S17,S2,S3,S5,S7,S9`
  - eval: `S10,S16,S4,S6,S8`

Canonical closure runtime summary:

- `PPG-DaLiA`
  - fresh canonical prep `1246.29 s`
  - cached canonical full rerun `54.58 s`
- `WESAD`
  - fresh canonical prep `811.69 s`
  - cached canonical full rerun `36.00 s`

Current best-supported Stage 4 conclusions from canonical full-dataset evidence:

- cache-backed reruns are materially cheaper than rebuilds on canonical full-dataset scope
- Stage 4C anomaly is still the strongest standalone Stage 4 component, but canonical evidence is mixed across datasets rather than uniformly stronger than Stage 3-only quality suspiciousness
  - `PPG-DaLiA` eval: Stage 3 baseline `AUPRC 0.6834`, `AUROC 0.5646`; Stage 4 anomaly `AUPRC 0.6902`, `AUROC 0.6064`
  - `WESAD` eval: Stage 3 baseline `AUPRC 0.6098`, `AUROC 0.5720`; Stage 4 anomaly `AUPRC 0.6004`, `AUROC 0.5698`
- the current unified Stage 4 suspiciousness default still underperforms the simple Stage 3-only quality baseline on canonical full-dataset eval
  - `PPG-DaLiA` eval: Stage 4 unified `AUPRC 0.6581`, `AUROC 0.4666`
  - `WESAD` eval: Stage 4 unified `AUPRC 0.5997`, `AUROC 0.4940`
- one conservative analysis-only fusion variant (`balanced_v1_analysis`) was rechecked on canonical cached inputs and still underperformed the current default on both datasets, so it was not promoted
- bounded validation artifacts should continue to be treated as non-canonical comparison outputs only

## Stage 1 Results

First-round real comparison results:

`PPG-DaLiA`
- `stage0_baseline`: MAE `23.7573`, RMSE `34.8267`, Pearson `0.3895`
- `stage1_frequency`: MAE `9.5227`, RMSE `18.8806`, Pearson `0.7679`
- `stage1_time`: MAE `23.5926`, RMSE `31.5837`, Pearson `0.3310`
- `stage1_fusion`: MAE `21.8655`, RMSE `30.3448`, Pearson `0.4220`

`WESAD`
- `stage0_baseline`: MAE `19.6101`, RMSE `31.6620`, Pearson `0.2644`
- `stage1_frequency`: MAE `10.8137`, RMSE `22.3642`, Pearson `0.5348`
- `stage1_time`: MAE `18.6811`, RMSE `28.6999`, Pearson `0.1262`
- `stage1_fusion`: MAE `16.4980`, RMSE `27.1314`, Pearson `0.3265`

Current takeaway:

- the strongest first-round Stage 1 method is `stage1_frequency`
- `stage1_fusion` improves over `stage0_baseline` on both datasets
- `stage1_fusion` does not yet beat `stage1_frequency`

## Stage 2 Scope

Stage 2 first round only includes:

- beat detection
- IBI extraction
- IBI cleaning
- basic time-domain PRV/HRV features
- an exploratory beat-level quality proxy path on top of the enhanced beat pipeline

Stage 2 still does not include:

- SQI
- event detection
- irregular pulse screening
- frequency-domain HRV
- nonlinear HRV
- deep learning models

Stage 3C1 closure status:

- Stage 3C1 implemented and validated a minimum viable rule-based beat-level quality proxy on top of the Stage 2 enhanced pipeline
- `enhanced_beat_quality` is the official baseline operating point for this branch at threshold `0.55`
- Stage 3C1.1 added threshold / retention tradeoff analysis and an additive analysis-only `enhanced_beat_quality_refined` comparison row
- the score is reproducible and decision-useful, but the default threshold is conservative and reduces recall / kept-beat ratio / valid IBI pairs substantially
- the refined operating point shows that better tradeoffs exist, especially on WESAD, but no new default threshold is adopted here

## Stage 3 Scope

Stage 3 round 1 currently includes:

- window-level quality scoring for Stage 1 windows
- rule-based quality gating for `stage1_frequency`
- an auxiliary ACC-based `motion_flag`
- gated-vs-ungated HR comparison against ECG-backed reference HR

Stage 3 enhancement round currently adds:

- a lightweight `LogisticRegression` quality model on top of the Stage 3 rule baseline
- fair comparison of ungated, rule-gated, and ML-gated `stage1_frequency`
- threshold / retention tradeoff analysis and operating-point summary CSVs for the ML-gated path
- a motion-aware strengthened comparison path that refines the existing ML-gated quality score using optional ACC summaries and PPG instability features
- a DWT-denoised comparison path that re-runs the same Stage 1 / Stage 3 enhanced evaluation flow on wavelet-denoised windows

Stage 3 round 1 does not include:

- event detection
- irregular pulse screening
- respiration
- deep learning training
- GPU-dependent methods

## Stage 2 Results

Enhancement-round real evaluation results:

`PPG-DaLiA`
- `baseline`: beat f1 `0.4073`, `ibi_rmse_ms 82.7283`, `mean_ibi_ms` MAE `150.6063`, `median_ibi_ms` MAE `126.9604`, `sdnn_ms` MAE `71.9162`, `rmssd_ms` MAE `89.5698`, `ibi_cv` MAE `0.0869`
- `enhanced`: beat f1 `0.4479`, `ibi_rmse_ms 73.0118`, `mean_ibi_ms` MAE `71.5637`, `median_ibi_ms` MAE `65.5073`, `sdnn_ms` MAE `48.9403`, `rmssd_ms` MAE `46.8822`, `ibi_cv` MAE `0.0778`

`WESAD`
- `baseline`: beat f1 `0.4241`, `ibi_rmse_ms 81.1065`, `mean_ibi_ms` MAE `96.1951`, `median_ibi_ms` MAE `83.5158`, `sdnn_ms` MAE `62.6055`, `rmssd_ms` MAE `98.8140`, `ibi_cv` MAE `0.0795`
- `enhanced`: beat f1 `0.4477`, `ibi_rmse_ms 67.0309`, `mean_ibi_ms` MAE `66.8864`, `median_ibi_ms` MAE `62.6820`, `sdnn_ms` MAE `38.7060`, `rmssd_ms` MAE `50.6693`, `ibi_cv` MAE `0.0603`

Current Stage 2 takeaway:

- Stage 2 enhancement round is runnable, measurable, and reproducible
- the enhancement path improves the two highest-priority metrics on both datasets: beat `f1` and `ibi_rmse_ms`
- `mean_ibi_ms` and `median_ibi_ms` are now clearly more usable than in the first round
- variability features such as `sdnn_ms`, `rmssd_ms`, and `ibi_cv` also improve, but they are still weaker and remain more sensitive to beat / IBI errors than mean / median summaries
- beat detection and IBI cleaning are now strong enough for Stage 2 to serve as a reasonable base for Stage 3
- Stage 2 now also supports a minimum viable beat-level quality proxy comparison without introducing a new learned model family
- if more Stage 2 work is done later, it should still focus on beat detection and IBI cleaning, but its priority is now lower than starting Stage 3

## Dataset Notes

Stage 0 assumes the official subject-pickle style layout for both datasets.

- `PPG-DaLiA`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- `WESAD`: reads `signal["wrist"]["BVP"]`, optional `signal["wrist"]["ACC"]`, and `signal["chest"]["ECG"]`
- official pickles are loaded with Python 3 `latin1` compatibility
- reference HR is unified at the window level by reconstructing it from chest ECG peaks

For more detail, see:

- [docs/STAGE0_RUNBOOK.md](.docs/STAGE0_RUNBOOK.md)
- [docs/STAGE1_RUNBOOK.md](.docs/STAGE1_RUNBOOK.md)
- [docs/STAGE2_RUNBOOK.md](.docs/STAGE2_RUNBOOK.md)
- [docs/DATASETS.md](.docs/DATASETS.md)

## Current Limitations

- Stage 0 provides only a minimal frequency-domain HR baseline.
- Stage 1 is still a lightweight classical-signal baseline system, not a final robust estimator.
- Stage 1 frequency is currently the best-performing path; fusion is mainly improving coverage and robustness relative to Stage 0, not surpassing the frequency chain yet.
- Stage 2 enhancement round is still limited to beat / IBI / basic time-domain PRV-HRV only.
- The new beat-level quality proxy is narrow, rule-based, and exploratory; it is not full beat-level SQI closure.
- Stage 3 is still a narrow window-level SQI / quality-gating track centered on `stage1_frequency`, not full Stage 3 closure.
- Stage 3B2 adds `PyWavelets` only for a single DWT comparison branch; it does not introduce SSA, adaptive filtering, beat-level SQI, or deep learning denoising.
- Stage 2 is more reliable for mean / median IBI style summaries than for variability features such as `sdnn_ms`, `rmssd_ms`, and `ibi_cv`.
- Current baseline quality is intended for reproducible validation, not final performance.
- Dataset configs in the repository are templates only; local paths must be set in ignored `*.local.yaml` files.
- If a dataset variant changes pickle keys or file layout, the loader may need a small compatibility update.
