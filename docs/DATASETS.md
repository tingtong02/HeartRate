# Dataset Assumptions for Stage 0

## Shared assumptions

- Public datasets are expected to be downloaded manually by the user and are not committed with this repository.
- Implementation targets the official public dataset directory style.
- Dataset root contains subject directories such as `S1/`, `S2/`.
- Each subject directory may contain multiple companion files, but Stage 0 only reads the subject pickle file such as `S1.pkl`.
- Only the minimum channels needed for Stage 0 are read.
- Official pickles are loaded with Python 3 using `latin1` compatibility decoding.
- Repository dataset templates keep `root_dir` empty; set local paths in ignored `configs/datasets/*.local.yaml` files.

## PPG-DaLiA

Assumed directory layout:

```text
PPG-DaLiA/
  README.pdf
  S1/
    S1.pkl
    S1_E4.zip
    S1_RespiBAN.h5
    S1_activity.csv
    S1_quest.csv
  S2/
    S2.pkl
```

Channels used:
- PPG: `signal["wrist"]["BVP"]`
- Optional ACC: `signal["wrist"]["ACC"]`
- Reference ECG: `signal["chest"]["ECG"]`

Sampling-rate assumptions in code:
- `BVP`: 64 Hz
- `ACC`: 32 Hz
- `ECG`: 700 Hz

Observed top-level pickle keys on real data:
- `activity`
- `label`
- `questionnaire`
- `rpeaks`
- `signal`
- `subject`

## WESAD

Assumed directory layout:

```text
WESAD/
  wesad_readme.pdf
  S2/
    S2.pkl
    S2_E4_Data.zip
    S2_quest.csv
    S2_readme.txt
    S2_respiban.txt
  S3/
    S3.pkl
```

Channels used:
- PPG: `signal["wrist"]["BVP"]`
- Optional ACC: `signal["wrist"]["ACC"]`
- Reference ECG: `signal["chest"]["ECG"]`

Sampling-rate assumptions in code:
- `BVP`: 64 Hz
- `ACC`: 32 Hz
- `ECG`: 700 Hz

Observed top-level pickle keys on real data:
- `label`
- `signal`
- `subject`

## Reference HR unification

- Stage 0 does not use dataset-specific HR labels as the primary reference.
- For both datasets, window-level reference HR is reconstructed from chest ECG peaks.
- Baseline evaluation uses the same window protocol on both datasets.

## Current limitations

- The loader assumes the official pickle payload exposes `signal -> wrist/chest`.
- If a dataset variant renames keys or stores files differently, the current loader will need a small compatibility update.
- ACC is loaded only as an optional aligned field and is not used by the Stage 0 baseline.
