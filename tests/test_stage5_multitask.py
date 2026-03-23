from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.stage5_multitask import (
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    STAGE5_IDENTITY_COLUMNS,
    aggregate_stage4_context_to_windows,
    build_stage5_multitask_predictions,
    summarize_stage5_metrics,
)


def _stage4_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 8.0,
                "selected_hr_bpm": 70.0,
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_is_valid": True,
                "ml_signal_quality_score": 0.8,
                "motion_flag": False,
                "quality_gate_passed": True,
                "hr_event_flag": False,
                "irregular_pulse_flag": True,
                "anomaly_score": 0.25,
                "stage4_suspicion_flag": True,
                "stage4_suspicion_score": 0.40,
                "stage4_suspicion_type_summary": "irregular",
            },
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 1,
                "start_time_s": 8.0,
                "duration_s": 8.0,
                "selected_hr_bpm": 74.0,
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_is_valid": True,
                "ml_signal_quality_score": 0.6,
                "motion_flag": True,
                "quality_gate_passed": False,
                "hr_event_flag": True,
                "irregular_pulse_flag": False,
                "anomaly_score": 0.55,
                "stage4_suspicion_flag": True,
                "stage4_suspicion_score": 0.80,
                "stage4_suspicion_type_summary": "event:tachycardia_event",
            },
        ]
    )


def test_aggregate_stage4_context_to_windows_applies_overlap_rules() -> None:
    stage5_identity = pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 16.0,
            }
        ]
    )
    aggregated = aggregate_stage4_context_to_windows(_stage4_frame(), stage5_identity)
    assert aggregated.shape[0] == 1
    row = aggregated.iloc[0]
    assert abs(float(row["selected_hr_bpm"]) - 72.0) < 1e-6
    assert bool(row["motion_flag"]) is True
    assert bool(row["hr_event_flag"]) is True
    assert bool(row["irregular_pulse_flag"]) is True
    assert abs(float(row["anomaly_score"]) - 0.55) < 1e-6
    assert abs(float(row["stage4_suspicion_score"]) - 0.80) < 1e-6


def test_build_stage5_multitask_predictions_has_stable_schema_and_serialization() -> None:
    frame = pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 32.0,
                "resp_rate_ref_bpm": 15.0,
                "resp_rate_ref_valid_flag": True,
                "resp_reference_reason": "reference_valid",
                "resp_rate_baseline_bpm": 14.2,
                "resp_rate_pred_bpm": 15.6,
                "resp_confidence": 0.8,
                "selected_hr_bpm": 72.0,
                "selected_hr_source": "robust_stage3c2_policy",
                "selected_hr_is_valid": True,
                "ml_signal_quality_score": 0.75,
                "motion_flag": False,
                "validity_flag": True,
                "hr_event_flag": False,
                "irregular_pulse_flag": True,
                "anomaly_score": 0.3,
                "stage4_suspicion_flag": True,
                "stage4_suspicion_score": 0.45,
                "stage4_suspicion_type_summary": "irregular",
                "num_beats": 8.0,
                "num_ibi_clean": 7.0,
                "mean_ibi_ms": 810.0,
                "rmssd_ms": 20.0,
                "sdnn_ms": 18.0,
                "pnn50": 0.08,
                "beat_positions_s": [0.5, 1.3, 2.1],
                "ibi_series_ms": [790.0, 815.0],
            }
        ]
    )
    predictions = build_stage5_multitask_predictions(frame, resp_validity_threshold=0.5)
    assert list(PREDICTION_COLUMNS) == predictions.columns.tolist()
    assert bool(predictions.loc[0, "resp_validity_flag"]) is True
    assert predictions.loc[0, "beat_positions_s"].startswith("[")
    assert predictions.loc[0, "ibi_series_ms"].startswith("[")


def test_summarize_stage5_metrics_returns_expected_rows_and_columns() -> None:
    predictions = build_stage5_multitask_predictions(
        pd.DataFrame(
            [
                {
                    "split": "train",
                    "dataset": "synthetic",
                    "subject_id": "S1",
                    "window_index": 0,
                    "start_time_s": 0.0,
                    "duration_s": 32.0,
                    "resp_rate_ref_bpm": 15.0,
                    "resp_rate_ref_valid_flag": True,
                    "resp_reference_reason": "reference_valid",
                    "resp_rate_baseline_bpm": 13.0,
                    "resp_rate_pred_bpm": 15.2,
                    "resp_confidence": 0.9,
                    "selected_hr_bpm": 70.0,
                    "selected_hr_source": "robust_stage3c2_policy",
                    "selected_hr_is_valid": True,
                    "ml_signal_quality_score": 0.8,
                    "motion_flag": False,
                    "validity_flag": True,
                    "hr_event_flag": False,
                    "irregular_pulse_flag": False,
                    "anomaly_score": 0.1,
                    "stage4_suspicion_flag": False,
                    "stage4_suspicion_score": 0.1,
                    "stage4_suspicion_type_summary": "",
                    "num_beats": 8.0,
                    "num_ibi_clean": 7.0,
                    "mean_ibi_ms": 800.0,
                    "rmssd_ms": 18.0,
                    "sdnn_ms": 16.0,
                    "pnn50": 0.05,
                    "beat_positions_s": [0.5, 1.3],
                    "ibi_series_ms": [790.0],
                },
                {
                    "split": "eval",
                    "dataset": "synthetic",
                    "subject_id": "S2",
                    "window_index": 0,
                    "start_time_s": 0.0,
                    "duration_s": 32.0,
                    "resp_rate_ref_bpm": 18.0,
                    "resp_rate_ref_valid_flag": True,
                    "resp_reference_reason": "reference_valid",
                    "resp_rate_baseline_bpm": 16.5,
                    "resp_rate_pred_bpm": 17.8,
                    "resp_confidence": 0.7,
                    "selected_hr_bpm": 72.0,
                    "selected_hr_source": "robust_stage3c2_policy",
                    "selected_hr_is_valid": True,
                    "ml_signal_quality_score": 0.7,
                    "motion_flag": False,
                    "validity_flag": True,
                    "hr_event_flag": True,
                    "irregular_pulse_flag": False,
                    "anomaly_score": 0.2,
                    "stage4_suspicion_flag": True,
                    "stage4_suspicion_score": 0.6,
                    "stage4_suspicion_type_summary": "event:tachycardia_event",
                    "num_beats": 8.0,
                    "num_ibi_clean": 7.0,
                    "mean_ibi_ms": 790.0,
                    "rmssd_ms": 19.0,
                    "sdnn_ms": 17.0,
                    "pnn50": 0.04,
                    "beat_positions_s": [0.4, 1.2],
                    "ibi_series_ms": [780.0],
                },
            ]
        ),
        resp_validity_threshold=0.5,
    )
    metrics = summarize_stage5_metrics(predictions)
    assert list(METRIC_COLUMNS) == metrics.columns.tolist()
    assert {"resp_surrogate_fusion_baseline", "stage5_resp_multitask_cnn_v1"} == set(metrics["method"])
    assert {"all_ref_valid", "high_quality_ref_valid", "predicted_valid"} == set(metrics["subset"])
    assert np.allclose(metrics["hr_selected_hr_match_rate"].to_numpy(dtype=float), 1.0)
