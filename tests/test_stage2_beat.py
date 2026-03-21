from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.metrics import summarize_feature_metrics
from heart_rate_cnn.stage2_beat import (
    clean_ibi_series,
    compute_time_domain_prv_features,
    detect_beats_in_window,
    evaluate_beat_detection,
    extract_ibi_from_beats,
    match_beats_by_tolerance,
    preprocess_ppg_for_beats,
    refine_beats_in_window,
)


def _make_ppg_like_signal(fs: float, bpm: float, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.arange(0.0, duration_s, 1.0 / fs)
    signal = np.zeros_like(time)
    for beat_time in np.arange(0.6, duration_s, 60.0 / bpm):
        signal += np.exp(-0.5 * ((time - beat_time) / 0.05) ** 2)
    return time, signal


def test_detect_beats_in_window_finds_expected_pulse_count() -> None:
    fs = 64.0
    bpm = 75.0
    _, signal = _make_ppg_like_signal(fs, bpm, duration_s=12.0)
    beats = detect_beats_in_window(signal, fs=fs)
    assert 12 <= beats.size <= 16


def test_refine_beats_moves_peaks_to_local_maxima() -> None:
    fs = 64.0
    signal = np.zeros(256, dtype=float)
    signal[48] = 0.4
    signal[50] = 1.0
    signal[120] = 0.5
    signal[123] = 1.0
    coarse = np.array([47, 121])
    refined = refine_beats_in_window(signal, coarse, fs=fs, config={"refine_radius_seconds": 0.05})
    assert np.array_equal(refined, np.array([50, 123]))


def test_extract_ibi_from_beats_returns_expected_values() -> None:
    beats = np.array([0, 64, 128, 192])
    ibi = extract_ibi_from_beats(beats, fs=64.0)
    assert np.allclose(ibi, np.array([1.0, 1.0, 1.0]))


def test_clean_ibi_series_removes_outlier_interval() -> None:
    ibi = np.array([0.80, 0.82, 1.40, 0.81, 0.79], dtype=float)
    cleaned = clean_ibi_series(ibi, {"max_deviation_ratio": 0.20})
    assert cleaned["ibi_is_valid"]
    assert cleaned["ibi_clean_s"].size == 4


def test_compute_time_domain_prv_features_returns_expected_keys() -> None:
    ibi = np.array([0.80, 0.82, 0.79, 0.81], dtype=float)
    features = compute_time_domain_prv_features(ibi, num_beats=5, num_ibi_raw=4, num_ibi_clean=4)
    assert features["num_beats"] == 5.0
    assert features["num_ibi_raw"] == 4.0
    assert features["num_ibi_clean"] == 4.0
    assert np.isfinite(features["mean_ibi_ms"])
    assert np.isfinite(features["rmssd_ms"])


def test_match_beats_by_tolerance_is_one_to_one() -> None:
    pred_s = np.array([1.00, 2.02, 3.00])
    ref_s = np.array([1.01, 2.00, 2.99])
    matches = match_beats_by_tolerance(pred_s, ref_s, tolerance_seconds=0.05)
    assert matches == [(0, 0), (1, 1), (2, 2)]


def test_evaluate_beat_detection_computes_precision_recall() -> None:
    pred = np.array([64, 128, 192])
    ref = np.array([65, 128, 190, 256])
    metrics = evaluate_beat_detection(pred, ref, pred_fs=64.0, ref_fs=64.0, tolerance_seconds=0.05)
    assert metrics["tp"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 1


def test_preprocess_ppg_for_beats_keeps_signal_finite() -> None:
    fs = 64.0
    time = np.arange(0.0, 8.0, 1.0 / fs)
    signal = np.sin(2 * np.pi * 1.1 * time)
    processed = preprocess_ppg_for_beats(signal, fs=fs)
    assert processed.shape == signal.shape
    assert np.all(np.isfinite(processed))


def test_summarize_feature_metrics_returns_rows() -> None:
    frame = pd.DataFrame(
        {
            "pred_mean_ibi_ms": [800.0, 810.0],
            "ref_mean_ibi_ms": [790.0, 805.0],
            "pred_rmssd_ms": [35.0, 40.0],
            "ref_rmssd_ms": [30.0, 42.0],
        }
    )
    summary = summarize_feature_metrics(frame, ["mean_ibi_ms", "rmssd_ms"], ref_prefix="ref_", pred_prefix="pred_")
    assert list(summary["feature"]) == ["mean_ibi_ms", "rmssd_ms"]
