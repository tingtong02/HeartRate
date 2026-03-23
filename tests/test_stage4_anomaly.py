from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.stage4_anomaly import (
    PREDICTION_COLUMNS,
    build_anomaly_predictions,
    fit_isolation_forest_anomaly_model,
    normalize_anomaly_scores,
    select_anomaly_fit_reference_rows,
    summarize_stage4_anomaly_metrics,
)
from heart_rate_cnn.stage4_irregular import MODEL_FEATURE_COLUMNS


def _anomaly_frame() -> pd.DataFrame:
    rows = []
    for index in range(8):
        row = {
            "split": "train" if index < 5 else "eval",
            "dataset": "synthetic",
            "subject_id": "S1",
            "window_index": index,
            "start_time_s": float(index * 2),
            "duration_s": 8.0,
            "selected_hr_source": "robust_stage3c2_policy",
            "selected_hr_bpm": 70.0 + index,
            "selected_hr_is_valid": True,
            "ml_signal_quality_score": 0.85,
            "ml_validity_flag": True,
            "robust_hr_source": "frequency",
            "robust_hr_action": "direct_update",
            "num_beats": 8.0,
            "num_ibi_raw": 7.0,
            "num_ibi_clean": 6.0,
            "ibi_removed_ratio": 0.1,
            "ibi_is_valid": True,
            "selected_hr_missing_flag": False,
            "insufficient_beats_flag": False,
            "insufficient_clean_ibi_flag": False,
            "quality_gate_passed": True,
            "quality_gate_reason": "pass",
            "support_sufficient_flag": True,
            "proxy_hr_event_target_any": False,
            "screening_proxy_target": False,
            "proxy_label_support_flag": True,
            "proxy_abnormal_target": False,
            "proxy_abnormal_support_flag": True,
        }
        if index in (2, 6):
            row["proxy_abnormal_target"] = True
            row["screening_proxy_target"] = True
            row["rmssd_ms"] = 120.0
            row["pnn50"] = 0.45
            row["local_deviation_ratio_max"] = 0.28
        else:
            row["rmssd_ms"] = 20.0
            row["pnn50"] = 0.05
            row["local_deviation_ratio_max"] = 0.04
        rows.append(row)

    frame = pd.DataFrame(rows)
    defaults = {
        "mean_ibi_ms": 800.0,
        "median_ibi_ms": 790.0,
        "mean_hr_bpm_from_ibi": 75.0,
        "sdnn_ms": 18.0,
        "rmssd_ms": 22.0,
        "pnn50": 0.04,
        "ibi_cv": 0.03,
        "ibi_mad_ms": 8.0,
        "successive_ibi_jump_mean_ms": 20.0,
        "successive_ibi_jump_max_ms": 30.0,
        "local_deviation_ratio_mean": 0.03,
        "local_deviation_ratio_max": 0.05,
        "turning_point_ratio": 0.20,
        "selected_hr_delta_bpm": 2.0,
        "beat_quality_mean_score": 0.75,
        "beat_quality_good_ratio": 0.8,
        "beat_quality_good_count": 6.0,
        "beat_fallback_available": True,
        "beat_fallback_num_beats": 8.0,
        "beat_fallback_num_clean_ibi": 6.0,
        "beat_fallback_kept_ratio": 0.8,
        "rule_signal_quality_score": 0.8,
        "rule_validity_flag": True,
        "motion_flag": False,
        "has_acc": False,
        "acc_axis_std_norm": np.nan,
        "acc_mag_range": np.nan,
        "freq_confidence": 0.7,
        "freq_peak_ratio": 2.4,
        "time_confidence": 0.6,
        "time_num_peaks": 8.0,
        "hr_agreement_bpm": 2.0,
        "ppg_processed_diff_std": 0.10,
        "robust_hr_is_valid": True,
        "hold_applied": False,
        "hold_age_windows": 0.0,
        "hr_jump_bpm_from_previous": 2.0,
        "robust_source_is_frequency": 1.0,
        "robust_source_is_beat_fallback": 0.0,
        "robust_source_is_hold_previous": 0.0,
        "robust_source_is_none": 0.0,
        "robust_action_is_direct_update": 1.0,
        "robust_action_is_fallback_update": 0.0,
        "robust_action_is_hold": 0.0,
        "robust_action_is_reject": 0.0,
    }
    for column_name in MODEL_FEATURE_COLUMNS:
        if column_name not in frame.columns:
            frame[column_name] = defaults.get(column_name, 0.0)
    return frame


def test_select_anomaly_fit_reference_rows_uses_train_quality_passed_proxy_regular_rows() -> None:
    frame = _anomaly_frame()
    frame.loc[1, "quality_gate_passed"] = False
    mask = select_anomaly_fit_reference_rows(
        frame,
        config={
            "require_quality_gate_pass": True,
            "require_support_sufficient": True,
            "require_proxy_support": True,
        },
    )
    assert mask.tolist() == [True, False, False, True, True, False, False, False]


def test_normalize_anomaly_scores_is_monotonic_and_bounded() -> None:
    normalized = normalize_anomaly_scores(
        np.asarray([0.10, 0.20, 0.50, 1.00], dtype=float),
        np.asarray([0.15, 0.25, 0.35, 0.45], dtype=float),
    )
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))
    assert normalized.tolist() == sorted(normalized.tolist())


def test_build_anomaly_predictions_emits_schema_and_suppresses_low_quality_candidate() -> None:
    frame = _anomaly_frame()
    frame.loc[6, "selected_hr_is_valid"] = False
    model, reference_mask = fit_isolation_forest_anomaly_model(
        frame,
        config={"min_reference_rows": 2, "n_estimators": 32, "random_seed": 42},
    )
    predictions = build_anomaly_predictions(
        frame,
        model=model,
        fit_reference_mask=reference_mask,
        config={
            "model_name": "isolation_forest_anomaly",
            "alert_quantile": 0.80,
            "quality_gate": {
                "mode": "suppress",
                "require_selected_hr_valid": True,
                "require_stage3_quality_pass": True,
                "stage3_quality_flag_column": "ml_validity_flag",
                "require_support_sufficient": True,
                "disallowed_robust_sources": ["none", "hold_previous"],
                "disallowed_robust_actions": ["hold", "reject"],
            },
        },
    )
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    row = predictions.loc[predictions["window_index"] == 6].iloc[0]
    assert not bool(row["quality_gate_passed"])
    assert not bool(row["anomaly_flag"])
    assert row["quality_gate_reason"] == "selected_hr_invalid"


def test_constant_anomaly_model_path_handles_degenerate_reference_set() -> None:
    frame = _anomaly_frame().iloc[[0]].copy()
    model, reference_mask = fit_isolation_forest_anomaly_model(frame, config={"min_reference_rows": 4, "random_seed": 42})
    predictions = build_anomaly_predictions(
        frame,
        model=model,
        fit_reference_mask=reference_mask,
        config={"model_name": "isolation_forest_anomaly", "quality_gate": {"mode": "suppress"}},
    )
    assert np.allclose(predictions["raw_anomaly_score"].to_numpy(dtype=float), 0.0)
    assert np.allclose(predictions["anomaly_score"].to_numpy(dtype=float), 0.0)


def test_summarize_stage4_anomaly_metrics_returns_expected_columns() -> None:
    frame = _anomaly_frame()
    model, reference_mask = fit_isolation_forest_anomaly_model(
        frame,
        config={"min_reference_rows": 2, "n_estimators": 32, "random_seed": 42},
    )
    predictions = build_anomaly_predictions(
        frame,
        model=model,
        fit_reference_mask=reference_mask,
        config={
            "model_name": "isolation_forest_anomaly",
            "alert_quantile": 0.80,
            "quality_gate": {"mode": "suppress"},
        },
    )
    metrics = summarize_stage4_anomaly_metrics(predictions)
    expected_columns = {
        "task",
        "method",
        "split",
        "num_eval_windows",
        "num_positive_targets",
        "num_positive_predictions",
        "auroc",
        "fit_reference_fraction",
    }
    assert expected_columns.issubset(metrics.columns)
