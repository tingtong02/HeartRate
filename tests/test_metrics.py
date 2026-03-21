from __future__ import annotations

import numpy as np

from heart_rate_cnn.baseline_hr import estimate_hr_frequency_domain
from heart_rate_cnn.metrics import compute_hr_metrics


def test_compute_hr_metrics_known_values() -> None:
    y_true = np.array([60.0, 80.0, 100.0])
    y_pred = np.array([62.0, 78.0, 101.0])
    metrics = compute_hr_metrics(y_true, y_pred)

    assert np.isclose(metrics["mae"], 5.0 / 3.0)
    assert np.isclose(metrics["rmse"], np.sqrt(3.0))
    assert metrics["num_valid_windows"] == 3.0


def test_frequency_baseline_estimates_sine_hr() -> None:
    fs = 64.0
    bpm = 72.0
    time = np.arange(0.0, 16.0, 1.0 / fs)
    ppg = np.sin(2 * np.pi * (bpm / 60.0) * time)

    estimated = estimate_hr_frequency_domain(ppg, fs=fs, hr_band_bpm=(40.0, 180.0))
    assert abs(estimated - bpm) < 3.0
