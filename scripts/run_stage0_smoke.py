from __future__ import annotations

import math

import numpy as np

from heart_rate_cnn.baseline_hr import predict_windows
from heart_rate_cnn.metrics import compute_hr_metrics
from heart_rate_cnn.preprocess import build_window_samples
from heart_rate_cnn.types import SubjectRecord


def make_synthetic_subject(subject_id: str, bpm: float, duration_s: float = 30.0) -> SubjectRecord:
    ppg_fs = 64.0
    ecg_fs = 256.0
    time_ppg = np.arange(0.0, duration_s, 1.0 / ppg_fs)
    time_ecg = np.arange(0.0, duration_s, 1.0 / ecg_fs)

    heart_hz = bpm / 60.0
    ppg = np.sin(2 * np.pi * heart_hz * time_ppg) + 0.05 * np.random.default_rng(42).normal(size=time_ppg.size)
    ecg = np.zeros_like(time_ecg)
    beat_times = np.arange(0.5, duration_s, 1.0 / heart_hz)
    for beat_time in beat_times:
        center = int(round(beat_time * ecg_fs))
        if 1 <= center < ecg.size - 1:
            ecg[center - 1 : center + 2] = [0.8, 1.2, 0.8]

    acc = np.zeros((time_ppg.size, 3), dtype=float)
    return SubjectRecord(
        dataset="synthetic",
        subject_id=subject_id,
        ppg=ppg,
        ppg_fs=ppg_fs,
        ecg=ecg,
        ecg_fs=ecg_fs,
        acc=acc,
        acc_fs=ppg_fs,
        metadata={"synthetic_bpm": bpm},
    )


def main() -> None:
    records = [
        make_synthetic_subject("SYN001", 72.0),
        make_synthetic_subject("SYN002", 90.0),
    ]
    windows = []
    for record in records:
        windows.extend(build_window_samples(record, target_ppg_fs=64.0, window_seconds=8.0, step_seconds=2.0))

    predictions = predict_windows(windows)
    valid_rows = [
        row for row in predictions if row["is_valid"] and row["ref_hr_bpm"] is not None and not math.isnan(row["pred_hr_bpm"])
    ]
    y_true = np.array([row["ref_hr_bpm"] for row in valid_rows], dtype=float)
    y_pred = np.array([row["pred_hr_bpm"] for row in valid_rows], dtype=float)
    metrics = compute_hr_metrics(y_true, y_pred)

    print("Stage 0 smoke test completed.")
    print(f"Subjects: {len(records)}")
    print(f"Windows: {len(windows)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"{key}: {value}")


if __name__ == "__main__":
    main()
