from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.stage4_events import build_stage4_event_predictions
from heart_rate_cnn.stage4_full import (
    FULL_METRIC_COLUMNS,
    FULL_PREDICTION_COLUMNS,
    build_stage4_full_predictions,
    collapse_stage4_event_predictions,
    summarize_stage4_full_metrics,
)


def _event_config() -> dict:
    return {
        "source": {
            "tachycardia_event": "gated_stage3_ml_logreg",
            "bradycardia_event": "gated_stage3_ml_logreg",
            "abrupt_change_event": "robust_stage3c2_policy",
        },
        "thresholds": {
            "tachy_hr_bpm": 100.0,
            "brady_hr_bpm": 50.0,
            "abrupt_delta_hr_bpm": 20.0,
            "abrupt_confirmation_ratio": 0.50,
        },
        "persistence": {
            "tachy_min_valid_windows": 2,
            "brady_min_valid_windows": 2,
            "abrupt_min_valid_windows": 2,
            "episode_merge_gap_windows": 1,
        },
        "quality_gate": {"mode": "suppress"},
    }


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 8.0,
                "ref_hr_bpm": 90.0,
                "window_is_valid": True,
                "ungated_pred_hr_bpm": 90.0,
                "ungated_is_valid": True,
                "ml_gated_is_valid": True,
                "robust_hr_bpm": 90.0,
                "robust_hr_is_valid": True,
                "policy_reason_code": "quality_good_direct",
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_bpm": 90.0,
                "selected_hr_is_valid": True,
                "ml_signal_quality_score": 0.95,
                "ml_validity_flag": True,
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 1,
                "start_time_s": 8.0,
                "duration_s": 8.0,
                "ref_hr_bpm": 104.0,
                "window_is_valid": True,
                "ungated_pred_hr_bpm": 105.0,
                "ungated_is_valid": True,
                "ml_gated_is_valid": True,
                "robust_hr_bpm": 105.0,
                "robust_hr_is_valid": True,
                "policy_reason_code": "quality_good_direct",
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_bpm": 105.0,
                "selected_hr_is_valid": True,
                "ml_signal_quality_score": 0.90,
                "ml_validity_flag": True,
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 2,
                "start_time_s": 16.0,
                "duration_s": 8.0,
                "ref_hr_bpm": 102.0,
                "window_is_valid": True,
                "ungated_pred_hr_bpm": 103.0,
                "ungated_is_valid": True,
                "ml_gated_is_valid": True,
                "robust_hr_bpm": 103.0,
                "robust_hr_is_valid": True,
                "policy_reason_code": "quality_good_direct",
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_bpm": 103.0,
                "selected_hr_is_valid": True,
                "ml_signal_quality_score": 0.88,
                "ml_validity_flag": True,
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 3,
                "start_time_s": 24.0,
                "duration_s": 8.0,
                "ref_hr_bpm": 70.0,
                "window_is_valid": True,
                "ungated_pred_hr_bpm": 70.0,
                "ungated_is_valid": True,
                "ml_gated_is_valid": False,
                "robust_hr_bpm": np.nan,
                "robust_hr_is_valid": False,
                "policy_reason_code": "reject_no_reliable_source",
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_bpm": np.nan,
                "selected_hr_is_valid": False,
                "ml_signal_quality_score": 0.10,
                "ml_validity_flag": False,
            },
        ]
    )


def _feature_frame() -> pd.DataFrame:
    frame = _base_frame().copy()
    frame["selected_hr_delta_bpm"] = [0.0, 15.0, 2.0, np.nan]
    frame["quality_gate_passed"] = [True, True, True, False]
    frame["quality_gate_reason"] = ["pass", "pass", "pass", "selected_hr_invalid"]
    frame["screening_proxy_target"] = [False, True, True, False]
    frame["proxy_label_support_flag"] = [True, True, True, True]
    return frame


def _irregular_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 8.0,
                "screening_proxy_target": False,
                "proxy_label_support_flag": True,
                "irregular_pulse_score": 0.10,
                "irregular_pulse_flag": False,
                "screening_validity_flag": True,
                "quality_gate_passed": True,
                "quality_gate_reason": "pass",
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 1,
                "start_time_s": 8.0,
                "duration_s": 8.0,
                "screening_proxy_target": True,
                "proxy_label_support_flag": True,
                "irregular_pulse_score": 0.72,
                "irregular_pulse_flag": True,
                "screening_validity_flag": True,
                "quality_gate_passed": True,
                "quality_gate_reason": "pass",
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 2,
                "start_time_s": 16.0,
                "duration_s": 8.0,
                "screening_proxy_target": True,
                "proxy_label_support_flag": True,
                "irregular_pulse_score": 0.30,
                "irregular_pulse_flag": False,
                "screening_validity_flag": True,
                "quality_gate_passed": True,
                "quality_gate_reason": "pass",
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 3,
                "start_time_s": 24.0,
                "duration_s": 8.0,
                "screening_proxy_target": False,
                "proxy_label_support_flag": True,
                "irregular_pulse_score": 0.95,
                "irregular_pulse_flag": False,
                "screening_validity_flag": False,
                "quality_gate_passed": False,
                "quality_gate_reason": "selected_hr_invalid",
            },
        ]
    )


def _anomaly_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 8.0,
                "anomaly_score": 0.15,
                "anomaly_flag": False,
                "anomaly_validity_flag": True,
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 1,
                "start_time_s": 8.0,
                "duration_s": 8.0,
                "anomaly_score": 0.82,
                "anomaly_flag": True,
                "anomaly_validity_flag": True,
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 2,
                "start_time_s": 16.0,
                "duration_s": 8.0,
                "anomaly_score": 0.93,
                "anomaly_flag": True,
                "anomaly_validity_flag": True,
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 3,
                "start_time_s": 24.0,
                "duration_s": 8.0,
                "anomaly_score": 0.90,
                "anomaly_flag": False,
                "anomaly_validity_flag": False,
            },
        ]
    )


def _event_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "event_detection",
                "method": "stage4_rule_events_v1",
                "event_type": "all_events",
                "split": "eval",
                "num_eval_windows": 12.0,
                "num_eval_events": 2.0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "accuracy": 1.0,
                "quality_gate_pass_fraction": 0.75,
                "valid_event_fraction": 0.1667,
                "num_pred_events": 2.0,
            }
        ]
    )


def _irregular_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "irregular_pulse_screening",
                "method": "hist_gbdt_irregular",
                "split": "eval",
                "num_eval_windows": 4.0,
                "num_positive_targets": 2.0,
                "num_positive_predictions": 1.0,
                "accuracy": 0.75,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.6667,
                "auroc": 0.75,
                "auprc": 0.75,
                "quality_gate_pass_fraction": 0.75,
                "valid_prediction_fraction": 0.75,
            }
        ]
    )


def _anomaly_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "task": "anomaly_scoring",
                "method": "isolation_forest_anomaly",
                "split": "eval",
                "num_eval_windows": 4.0,
                "num_positive_targets": 2.0,
                "num_positive_predictions": 2.0,
                "accuracy": 0.75,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "auroc": 0.70,
                "auprc": 0.72,
                "quality_gate_pass_fraction": 0.75,
                "valid_prediction_fraction": 0.75,
            }
        ]
    )


def test_collapse_stage4_event_predictions_and_build_full_predictions() -> None:
    event_predictions = build_stage4_event_predictions(_base_frame().drop(columns=["split"]), split_name="eval", config=_event_config())
    event_summary = collapse_stage4_event_predictions(event_predictions)
    full_predictions = build_stage4_full_predictions(
        feature_frame=_feature_frame(),
        event_summary=event_summary,
        irregular_predictions=_irregular_predictions(),
        anomaly_predictions=_anomaly_predictions(),
        config={"suspicion": {"event_min_score": 0.60, "two_signal_bonus": 0.10, "three_signal_bonus": 0.20}},
    )
    assert set(FULL_PREDICTION_COLUMNS).issubset(full_predictions.columns)
    row1 = full_predictions.loc[full_predictions["window_index"] == 1].iloc[0]
    assert bool(row1["hr_event_flag"])
    assert bool(row1["irregular_pulse_flag"])
    assert bool(row1["anomaly_flag"])
    assert np.isclose(float(row1["stage4_suspicion_score"]), 1.0)
    assert row1["stage4_reason_code"] == "multi_signal_suspicion"
    assert row1["stage4_suspicion_type_summary"] == "event:tachycardia_event|irregular|anomaly"

    row2 = full_predictions.loc[full_predictions["window_index"] == 2].iloc[0]
    assert bool(row2["proxy_abnormal_target"])
    assert bool(row2["hr_event_flag"])
    assert bool(row2["anomaly_flag"])
    assert row2["stage4_reason_code"] == "multi_signal_suspicion"
    assert row2["stage4_suspicion_type_summary"] == "event:tachycardia_event|anomaly"

    row3 = full_predictions.loc[full_predictions["window_index"] == 3].iloc[0]
    assert not bool(row3["stage4_suspicion_flag"])
    assert row3["stage4_reason_code"] == "low_quality_suppressed"


def test_summarize_stage4_full_metrics_returns_expected_groups() -> None:
    event_predictions = build_stage4_event_predictions(_base_frame().drop(columns=["split"]), split_name="eval", config=_event_config())
    event_summary = collapse_stage4_event_predictions(event_predictions)
    full_predictions = build_stage4_full_predictions(
        feature_frame=_feature_frame(),
        event_summary=event_summary,
        irregular_predictions=_irregular_predictions(),
        anomaly_predictions=_anomaly_predictions(),
        config={"suspicion": {"event_min_score": 0.60, "two_signal_bonus": 0.10, "three_signal_bonus": 0.20}},
    )
    metrics = summarize_stage4_full_metrics(
        full_predictions=full_predictions,
        event_metrics=_event_metrics(),
        irregular_metrics=_irregular_metrics(),
        anomaly_metrics=_anomaly_metrics(),
    )
    assert set(FULL_METRIC_COLUMNS).issubset(metrics.columns)
    assert {"event", "irregular", "anomaly", "unified", "stage3_comparison", "stratification"}.issubset(set(metrics["metric_group"]))
    stage4_row = metrics.loc[(metrics["metric_group"] == "stage3_comparison") & (metrics["method"] == "stage4_full_default")].iloc[0]
    assert stage4_row["target_name"] == "proxy_abnormal_union"
