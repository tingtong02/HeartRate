from __future__ import annotations

import numpy as np

from heart_rate_cnn.stage3_quality import (
    apply_rule_based_quality_decision,
    apply_ml_quality_decision,
    build_refined_threshold_grid,
    build_ml_feature_matrix,
    build_ml_feature_row,
    build_quality_target,
    compute_binary_classification_summary,
    compute_motion_summary,
    evaluate_ml_threshold_grid,
    extract_quality_features,
    fit_quality_logistic_regression,
    predict_quality_logistic_regression,
    select_best_ml_threshold,
    summarize_operating_point_status,
    summarize_threshold_selection,
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


def test_build_ml_feature_row_adds_derived_fields() -> None:
    row = build_ml_feature_row(
        {
            "freq_pred_hr_bpm": 80.0,
            "freq_confidence": 0.8,
            "freq_peak_ratio": 2.2,
            "freq_is_valid": True,
            "time_confidence": 0.6,
            "time_num_peaks": 9.0,
            "time_is_valid": True,
            "fusion_confidence": 0.7,
            "fusion_source": "blended",
            "hr_agreement_bpm": 4.0,
            "ppg_centered_std": 1.0,
            "ppg_peak_to_peak": 3.0,
            "ppg_processed_diff_std": 0.1,
            "has_acc": True,
            "acc_axis_std_norm": 0.2,
            "acc_mag_range": 0.5,
        }
    )
    assert np.isclose(row["hr_agreement_ratio"], 0.05)
    assert row["fusion_is_blended"] == 1.0


def test_build_ml_feature_matrix_returns_expected_shape() -> None:
    matrix = build_ml_feature_matrix(
        [
            {
                "freq_pred_hr_bpm": 75.0,
                "freq_confidence": 0.8,
                "freq_peak_ratio": 2.0,
                "freq_is_valid": True,
                "time_confidence": 0.6,
                "time_num_peaks": 9.0,
                "time_is_valid": True,
                "fusion_confidence": 0.7,
                "fusion_source": "blended",
                "hr_agreement_bpm": 3.0,
                "ppg_centered_std": 1.0,
                "ppg_peak_to_peak": 2.5,
                "ppg_processed_diff_std": 0.1,
                "has_acc": False,
                "acc_axis_std_norm": np.nan,
                "acc_mag_range": np.nan,
            }
        ]
    )
    assert matrix.shape[0] == 1
    assert matrix.shape[1] >= 10


def test_fit_and_predict_quality_logistic_regression() -> None:
    feature_rows = [
        {
            "freq_pred_hr_bpm": 72.0,
            "freq_confidence": 0.9,
            "freq_peak_ratio": 2.8,
            "freq_is_valid": True,
            "time_confidence": 0.7,
            "time_num_peaks": 9.0,
            "time_is_valid": True,
            "fusion_confidence": 0.8,
            "fusion_source": "blended",
            "hr_agreement_bpm": 1.0,
            "ppg_centered_std": 1.0,
            "ppg_peak_to_peak": 2.0,
            "ppg_processed_diff_std": 0.08,
            "has_acc": False,
            "acc_axis_std_norm": np.nan,
            "acc_mag_range": np.nan,
        },
        {
            "freq_pred_hr_bpm": 72.0,
            "freq_confidence": 0.1,
            "freq_peak_ratio": 1.1,
            "freq_is_valid": False,
            "time_confidence": 0.2,
            "time_num_peaks": 4.0,
            "time_is_valid": False,
            "fusion_confidence": 0.1,
            "fusion_source": "frequency",
            "hr_agreement_bpm": 18.0,
            "ppg_centered_std": 1.5,
            "ppg_peak_to_peak": 4.0,
            "ppg_processed_diff_std": 0.3,
            "has_acc": False,
            "acc_axis_std_norm": np.nan,
            "acc_mag_range": np.nan,
        },
    ]
    model = fit_quality_logistic_regression(feature_rows, ["good", "poor"], random_seed=42)
    probabilities = predict_quality_logistic_regression(model, feature_rows)
    assert probabilities.shape == (2,)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))


def test_apply_ml_quality_decision_uses_threshold() -> None:
    decision = apply_ml_quality_decision(
        signal_quality_score=0.72,
        threshold=0.60,
        window_is_valid=True,
        freq_is_valid=True,
        motion_flag=True,
    )
    assert decision["signal_quality_label"] == "good"
    assert decision["validity_flag"]
    assert decision["motion_flag"]


def test_select_best_ml_threshold_prefers_low_mae_with_retention_floor() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "quality_target_label": ["good", "good", "poor", "poor"],
            "ml_signal_quality_score": [0.90, 0.85, 0.40, 0.20],
            "ref_hr_bpm": [70.0, 72.0, 68.0, 75.0],
            "ungated_pred_hr_bpm": [70.0, 72.0, 90.0, 95.0],
            "ungated_is_valid": [True, True, True, True],
        }
    )
    summary = select_best_ml_threshold(
        frame,
        score_col="ml_signal_quality_score",
        pred_col="ungated_pred_hr_bpm",
        valid_col="ungated_is_valid",
        threshold_grid=[0.3, 0.5, 0.8],
        min_retention_ratio=0.5,
    )
    assert summary["selected_threshold"] in {0.3, 0.5, 0.8}
    assert summary["retention_ratio"] >= 0.5


def test_evaluate_ml_threshold_grid_returns_expected_columns() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "quality_target_label": ["good", "poor"],
            "ml_signal_quality_score": [0.9, 0.2],
            "ref_hr_bpm": [70.0, 80.0],
            "ungated_pred_hr_bpm": [70.0, 95.0],
            "ungated_is_valid": [True, True],
        }
    )
    summary = evaluate_ml_threshold_grid(
        frame,
        score_col="ml_signal_quality_score",
        pred_col="ungated_pred_hr_bpm",
        valid_col="ungated_is_valid",
        threshold_grid=[0.3, 0.6],
        min_retention_ratio=0.5,
        split_name="train_select",
        sweep_stage="coarse",
    )
    assert list(summary["threshold"]) == [0.3, 0.6]
    assert "retention_ratio" in summary.columns
    assert "is_feasible_retention" in summary.columns


def test_build_refined_threshold_grid_returns_centered_grid() -> None:
    grid = build_refined_threshold_grid(center_threshold=0.45, refinement_radius=0.04, refinement_step=0.02)
    assert grid == [0.41, 0.43, 0.45, 0.47, 0.49]


def test_summarize_threshold_selection_matches_best_row() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "threshold": [0.3, 0.5, 0.7],
            "retention_ratio": [1.0, 0.8, 0.6],
            "mae": [10.0, 8.0, 8.5],
            "f1": [0.7, 0.8, 0.9],
            "is_feasible_retention": [True, True, False],
        }
    )
    summary = summarize_threshold_selection(frame)
    assert np.isclose(summary["selected_threshold"], 0.5)


def test_summarize_operating_point_status_marks_stable_band() -> None:
    import pandas as pd

    fine_frame = pd.DataFrame(
        {
            "threshold": [0.41, 0.43, 0.45],
            "retention_ratio": [0.98, 0.97, 0.96],
            "mae": [8.05, 8.01, 8.00],
            "f1": [0.84, 0.85, 0.86],
            "is_feasible_retention": [True, True, True],
        }
    )
    summary = summarize_operating_point_status(
        fine_frame,
        selected_threshold=0.45,
        stability_mae_tolerance=0.10,
        stable_min_threshold_count=3,
    )
    assert summary["operating_point_status"] == "stable"
