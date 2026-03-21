from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.metrics import summarize_method_metrics
from heart_rate_cnn.preprocess import preprocess_ppg_stage1
from heart_rate_cnn.stage1_hr import (
    estimate_hr_frequency_stage1,
    estimate_hr_time_stage1,
    fuse_hr_estimates,
)


def test_preprocess_ppg_stage1_keeps_length_and_finite_values() -> None:
    fs = 64.0
    time = np.arange(0.0, 8.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.2 * np.sin(2 * np.pi * 4.5 * time)

    processed = preprocess_ppg_stage1(signal, fs=fs)
    assert processed.shape == signal.shape
    assert np.all(np.isfinite(processed))


def test_stage1_frequency_estimates_sine_hr() -> None:
    fs = 64.0
    bpm = 78.0
    time = np.arange(0.0, 10.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * (bpm / 60.0) * time)

    result = estimate_hr_frequency_stage1(signal, fs=fs, hr_band_bpm=(40.0, 180.0))
    assert result["freq_is_valid"]
    assert abs(float(result["freq_pred_hr_bpm"]) - bpm) < 3.0
    assert 0.0 <= float(result["freq_confidence"]) <= 1.0


def test_stage1_time_estimates_pulse_train_hr() -> None:
    fs = 64.0
    bpm = 72.0
    duration_s = 12.0
    time = np.arange(0.0, duration_s, 1.0 / fs)
    signal = np.zeros_like(time)
    beat_times = np.arange(0.5, duration_s, 60.0 / bpm)
    for beat_time in beat_times:
        signal += np.exp(-0.5 * ((time - beat_time) / 0.06) ** 2)

    result = estimate_hr_time_stage1(signal, fs=fs, hr_band_bpm=(40.0, 180.0))
    assert result["time_is_valid"]
    assert abs(float(result["time_pred_hr_bpm"]) - bpm) < 3.0
    assert 0.0 <= float(result["time_confidence"]) <= 1.0


def test_fusion_logic_handles_common_cases() -> None:
    blended = fuse_hr_estimates(
        {"freq_pred_hr_bpm": 80.0, "freq_confidence": 0.7, "freq_is_valid": True},
        {"time_pred_hr_bpm": 82.0, "time_confidence": 0.6, "time_num_peaks": 5, "time_is_valid": True},
    )
    assert blended["fusion_source"] == "blended"

    freq_only = fuse_hr_estimates(
        {"freq_pred_hr_bpm": 76.0, "freq_confidence": 0.8, "freq_is_valid": True},
        {"time_pred_hr_bpm": np.nan, "time_confidence": 0.0, "time_num_peaks": 0, "time_is_valid": False},
    )
    assert freq_only["fusion_source"] == "frequency"

    time_only = fuse_hr_estimates(
        {"freq_pred_hr_bpm": np.nan, "freq_confidence": 0.0, "freq_is_valid": False},
        {"time_pred_hr_bpm": 88.0, "time_confidence": 0.7, "time_num_peaks": 5, "time_is_valid": True},
    )
    assert time_only["fusion_source"] == "time"

    conflict = fuse_hr_estimates(
        {"freq_pred_hr_bpm": 72.0, "freq_confidence": 0.7, "freq_is_valid": True},
        {"time_pred_hr_bpm": 120.0, "time_confidence": 0.5, "time_num_peaks": 5, "time_is_valid": True},
    )
    assert conflict["fusion_source"] == "frequency"


def test_summarize_method_metrics_builds_four_method_table() -> None:
    frame = pd.DataFrame(
        {
            "ref_hr_bpm": [70.0, 80.0, 90.0],
            "stage0_pred_hr_bpm": [71.0, 81.0, 88.0],
            "freq_pred_hr_bpm": [70.0, 79.0, 91.0],
            "time_pred_hr_bpm": [69.0, 82.0, 89.0],
            "fusion_pred_hr_bpm": [70.0, 80.0, 90.0],
            "stage0_is_valid": [True, True, True],
            "freq_is_valid": [True, True, True],
            "time_is_valid": [True, True, True],
            "fusion_is_valid": [True, True, True],
        }
    )

    metrics_frame = summarize_method_metrics(
        frame,
        {
            "stage0_baseline": {"pred_col": "stage0_pred_hr_bpm", "valid_col": "stage0_is_valid"},
            "stage1_frequency": {"pred_col": "freq_pred_hr_bpm", "valid_col": "freq_is_valid"},
            "stage1_time": {"pred_col": "time_pred_hr_bpm", "valid_col": "time_is_valid"},
            "stage1_fusion": {"pred_col": "fusion_pred_hr_bpm", "valid_col": "fusion_is_valid"},
        },
    )
    assert list(metrics_frame["method"]) == [
        "stage0_baseline",
        "stage1_frequency",
        "stage1_time",
        "stage1_fusion",
    ]
