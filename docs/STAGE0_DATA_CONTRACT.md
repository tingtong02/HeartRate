# Stage 0 Data Contract

## Scope

Stage 0 only defines the minimum data structures needed for:
- dataset loading
- resampling / alignment / segmentation
- window-level reference HR construction
- frequency-domain HR baseline evaluation

## `SubjectRecord`

Required fields:
- `dataset`: dataset name, currently `ppg_dalia` or `wesad`
- `subject_id`: subject identifier such as `S1`
- `ppg`: continuous raw PPG array from wrist `BVP`
- `ppg_fs`: original PPG sampling rate
- `ecg`: continuous raw ECG array from chest `ECG`
- `ecg_fs`: original ECG sampling rate

Optional fields:
- `acc`: continuous wrist `ACC` array
- `acc_fs`: original ACC sampling rate
- `metadata`: dataset-specific lightweight metadata

Notes:
- Loader output is always continuous subject-level data.
- Loaders do not perform segmentation or label construction.
- Paths are not embedded in the contract; they are resolved from config.

## `WindowSample`

Required fields:
- `dataset`
- `subject_id`
- `window_index`
- `start_time_s`
- `duration_s`
- `ppg`
- `ppg_fs`
- `is_valid`

Optional fields:
- `acc`
- `ref_hr_bpm`

Semantics:
- `ppg` is the resampled, fixed-length window used by the Stage 0 baseline.
- `acc` is optional and resampled to the same rate if present.
- `ref_hr_bpm` is the window-level reference HR derived from ECG peaks.
- `is_valid=False` means the window does not contain enough reliable ECG peak information to form a reference HR and is excluded from metric aggregation.

## Window-Level Reference HR

Unified protocol for both datasets:
1. Read subject-level chest ECG from the official pickle payload.
2. Detect ECG peaks on the continuous ECG stream.
3. Segment the resampled PPG into fixed windows.
4. For each PPG window, collect ECG peaks within the same time range.
5. Compute `ref_hr_bpm = 60 / mean(RR)` using peaks inside the window.
6. Mark the window invalid if fewer than two ECG peaks are available.

This keeps evaluation consistent even if the raw annotation conventions differ across datasets.
