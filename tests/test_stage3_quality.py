from __future__ import annotations

import numpy as np

from heart_rate_cnn.stage3_quality import (
    apply_rule_based_quality_decision,
    build_quality_target,
    compute_binary_classification_summary,
    compute_motion_summary,
    extract_quality_features,
)
from heart_rate_cnn.types import WindowSample


def _make_window(with_acc: bool = True) -> WindowSample:
    fs = 64.0
    time = np.arange(0.0, 8.0, 1.0 / fs)
    ppg = np.sin(2 * np.pi * 1.2 * time)
    acc = None
    if with_acc:
        acc = np.stack(
            [
                0.1 * np.sin(2 * np.pi * 0.3 * time),
                0.1 * np.cos(2 * np.pi * 0.3 * time),
                np.zeros_like(time),
            ],
            axis=1,
        )
    return WindowSample(
        dataset="synthetic",
        subject_id="SYN001",
        window_index=0,
        start_time_s=0.0,
        duration_s=8.0,
        ppg=ppg,
        ppg_fs=fs,
        acc=acc,
        ref_hr_bpm=72.0,
        is_valid=True,
    )


def test_build_quality_target_assigns_expected_labels() -> None:
    good = build_quality_target(
        ref_hr_bpm=72.0,
        freq_pred_hr_bpm=74.0,
        window_is_valid=True,
        freq_is_valid=True,
        good_error_bpm=5.0,
        poor_error_bpm=10.0,
    )
    borderline = build_quality_target(
        ref_hr_bpm=72.0,
        freq_pred_hr_bpm=79.0,
        window_is_valid=True,
        freq_is_valid=True,
        good_error_bpm=5.0,
        poor_error_bpm=10.0,
    )
    poor = build_quality_target(
        ref_hr_bpm=72.0,
        freq_pred_hr_bpm=84.0,
        window_is_valid=True,
        freq_is_valid=True,
        good_error_bpm=5.0,
        poor_error_bpm=10.0,
    )
    assert good["quality_target_label"] == "good"
    assert borderline["quality_target_label"] == "borderline"
    assert poor["quality_target_label"] == "poor"


def test_compute_motion_summary_handles_missing_acc() -> None:
    summary = compute_motion_summary(None, {"accel_std_threshold": 0.1, "accel_range_threshold": 0.2})
    assert not summary["has_acc"]
    assert not summary["motion_flag"]


def test_compute_motion_summary_flags_large_motion() -> None:
    acc = np.stack(
        [
            np.linspace(-1.0, 1.0, 64),
            np.zeros(64),
            np.zeros(64),
        ],
        axis=1,
    )
    summary = compute_motion_summary(acc, {"accel_std_threshold": 0.2, "accel_range_threshold": 0.5})
    assert summary["has_acc"]
    assert summary["motion_flag"]


def test_extract_quality_features_returns_expected_fields() -> None:
    window = _make_window()
    features = extract_quality_features(
        window,
        freq_result={
            "freq_pred_hr_bpm": 72.0,
            "freq_confidence": 0.8,
            "freq_peak_ratio": 2.5,
            "freq_is_valid": True,
        },
        time_result={
            "time_pred_hr_bpm": 73.0,
            "time_confidence": 0.6,
            "time_num_peaks": 9,
            "time_is_valid": True,
        },
        fusion_result={
            "fusion_pred_hr_bpm": 72.5,
            "fusion_confidence": 0.7,
            "fusion_source": "blended",
            "fusion_is_valid": True,
        },
        preprocess_config={},
        motion_config={},
    )
    assert np.isfinite(features["ppg_centered_std"])
    assert np.isfinite(features["ppg_peak_to_peak"])
    assert np.isfinite(features["ppg_processed_diff_std"])
    assert np.isfinite(features["hr_agreement_bpm"])


def test_rule_quality_decision_marks_invalid_frequency_as_poor() -> None:
    decision = apply_rule_based_quality_decision(
        window_is_valid=True,
        features={
            "freq_is_valid": False,
            "freq_confidence": 0.9,
            "freq_peak_ratio": 3.0,
            "time_is_valid": True,
            "time_confidence": 0.8,
            "ppg_processed_diff_std": 0.05,
            "hr_agreement_bpm": 1.0,
            "motion_flag": False,
        },
        config={"good_score_threshold": 0.5},
    )
    assert decision["signal_quality_label"] == "poor"
    assert not decision["validity_flag"]


def test_rule_quality_decision_keeps_motion_auxiliary() -> None:
    decision = apply_rule_based_quality_decision(
        window_is_valid=True,
        features={
            "freq_is_valid": True,
            "freq_confidence": 0.95,
            "freq_peak_ratio": 3.0,
            "time_is_valid": True,
            "time_confidence": 0.8,
            "ppg_processed_diff_std": 0.05,
            "hr_agreement_bpm": 1.0,
            "motion_flag": True,
        },
        config={
            "good_score_threshold": 0.5,
            "peak_ratio_bad": 1.1,
            "peak_ratio_good": 2.5,
            "agreement_good_bpm": 3.0,
            "agreement_bad_bpm": 12.0,
            "diff_std_good": 0.12,
            "diff_std_bad": 0.35,
        },
    )
    assert decision["signal_quality_label"] == "good"
    assert decision["validity_flag"]
    assert decision["motion_flag"]


def test_compute_binary_classification_summary_returns_expected_counts() -> None:
    summary = compute_binary_classification_summary(
        ["good", "poor", "good", "poor"],
        ["good", "poor", "poor", "poor"],
    )
    assert summary["num_eval_windows"] == 4.0
    assert np.isclose(summary["accuracy"], 0.75)
