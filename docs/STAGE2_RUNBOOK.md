# Stage 2 Runbook

## Scope

Stage 2 enhancement round still only covers:

- beat detection
- IBI extraction
- IBI cleaning
- basic time-domain PRV/HRV features
- an optional exploratory beat-level quality proxy on top of the enhanced beat path
- Stage 2 evaluation
- lightweight numeric error-case review

This round does not include SQI, event detection, irregular pulse screening, respiration, or deep learning.

## Environment

Reuse the current environment:

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

Then run:

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/ppg_dalia.local.yaml
```

```bash
python scripts/run_stage2_baseline.py \
  --dataset-config configs/datasets/wesad.local.yaml
```

Each run now writes both `baseline` and `enhanced` results for the same subject split and the same analysis windows.

## Stage 2 Defaults

- beat detection runs on Stage 1 style preprocessed PPG
- beats / IBI / features are computed independently inside long analysis windows
- default analysis window is `60 s`
- default analysis step is `30 s`
- Stage 2 keeps `stage1_frequency` as the current strongest HR baseline background
- `run_stage2_baseline.py` compares `baseline` and `enhanced` variants in one pass
- when `stage2.beat_quality.enabled: true`, `run_stage2_baseline.py` also evaluates an `enhanced_beat_quality` variant
- when `stage2.beat_quality_refine.enabled: true`, `run_stage2_baseline.py` also exports a threshold-sweep analysis and an `enhanced_beat_quality_refined` analysis-only operating point
- lightweight error-case export is off by default and can be enabled from `configs/eval/hr_stage2.yaml`

## Outputs

Stage 2 writes:

- `outputs/{dataset}_stage2_beats.csv`
- `outputs/{dataset}_stage2_beat_quality.csv`
- `outputs/{dataset}_stage2_beat_quality_sweep.csv`
- `outputs/{dataset}_stage2_features.csv`
- `outputs/{dataset}_stage2_metrics.csv`
- optional `outputs/{dataset}_stage2_error_cases.csv`

Use `outputs/{dataset}_stage2_metrics.csv` as the primary summary file when reproducing Stage 2 results.  
If `stage2.debug.save_error_cases: true`, the extra CSV contains only the worst few analysis windows per variant as a numeric summary. It does not create images or a separate visualization system.

For the beat-quality branch:

- `outputs/{dataset}_stage2_beat_quality.csv` is the primary beat-level inspection table
- `outputs/{dataset}_stage2_beat_quality_sweep.csv` is the threshold / retention tradeoff analysis table
- `enhanced_beat_quality` is the official baseline operating point at the configured threshold
- `enhanced_beat_quality_refined` is analysis-only and must not be treated as the new default threshold
- this beat-quality branch now serves as the minimum viable Stage 3 beat-level quality layer in the repository's practically complete Stage 3 scope

## Real Enhancement-Round Results

`PPG-DaLiA`
- `baseline`: beat precision `0.4855`, recall `0.3508`, f1 `0.4073`, beat_count_error `26.9110`
- `baseline`: IBI MAE `61.3743 ms`, RMSE `82.7283 ms`, valid IBI pairs `16343`
- `enhanced`: beat precision `0.4874`, recall `0.4143`, f1 `0.4479`, beat_count_error `15.5033`
- `enhanced`: IBI MAE `53.1301 ms`, RMSE `73.0118 ms`, valid IBI pairs `19029`
- feature highlights:
  - `mean_ibi_ms`: baseline MAE `150.6063`, enhanced MAE `71.5637`
  - `median_ibi_ms`: baseline MAE `126.9604`, enhanced MAE `65.5073`
  - `sdnn_ms`: baseline MAE `71.9162`, enhanced MAE `48.9403`
  - `rmssd_ms`: baseline MAE `89.5698`, enhanced MAE `46.8822`
  - `ibi_cv`: baseline MAE `0.0869`, enhanced MAE `0.0778`

`WESAD`
- `baseline`: beat precision `0.4766`, recall `0.3819`, f1 `0.4241`, beat_count_error `17.3876`
- `baseline`: IBI MAE `57.6451 ms`, RMSE `81.1065 ms`, valid IBI pairs `15091`
- `enhanced`: beat precision `0.4815`, recall `0.4182`, f1 `0.4477`, beat_count_error `12.6335`
- `enhanced`: IBI MAE `46.4785 ms`, RMSE `67.0309 ms`, valid IBI pairs `15466`
- feature highlights:
  - `mean_ibi_ms`: baseline MAE `96.1951`, enhanced MAE `66.8864`
  - `median_ibi_ms`: baseline MAE `83.5158`, enhanced MAE `62.6820`
  - `sdnn_ms`: baseline MAE `62.6055`, enhanced MAE `38.7060`
  - `rmssd_ms`: baseline MAE `98.8140`, enhanced MAE `50.6693`
  - `ibi_cv`: baseline MAE `0.0795`, enhanced MAE `0.0603`

Current interpretation:

- Stage 2 enhancement round is runnable, measurable, and reproducible
- the enhanced variant improves the top-priority metrics on both datasets: beat `f1` and `ibi_rmse_ms`
- `mean_ibi_ms` and `median_ibi_ms` also improve strongly on both datasets
- `sdnn_ms`, `rmssd_ms`, and `ibi_cv` improve, but they remain more fragile than mean / median IBI summaries
- beat detection and IBI cleaning are now strong enough for Stage 2 to act as a practical foundation for Stage 3
- if Stage 2 is revisited later, the next enhancement should still focus on beat detection and IBI cleaning rather than on adding new feature families first

## Notes

- Beat detection is direct peak finding plus local refinement, not a Hilbert-envelope-first design.
- The feature set is still strictly time-domain only.
- The beat-level quality proxy is implemented and validated as a minimum viable rule-based quality layer, not full beat-level SQI closure.
- Stage 3C1 adds per-beat `beat_quality_score`, `beat_quality_label`, and `beat_is_kept_by_quality` outputs plus beat-level CSV export.
- Stage 3C1.1 adds threshold / retention tradeoff analysis and an additive `enhanced_beat_quality_refined` comparison row.
- `enhanced_beat_quality` remains the official baseline operating point at the configured threshold `0.55`.
- `enhanced_beat_quality_refined` is analysis-only and exists to make the threshold/retention tradeoff explicit; it is not a silently adopted new default.
- Current conclusion: the score is reproducible and decision-useful, the default threshold is conservative, and the refined operating point is useful for analysis but is not adopted as the new default.
- For final Stage 3 interpretation, this branch should be read as a validated minimum viable beat-level quality component, not as full beat-level SQI closure.
