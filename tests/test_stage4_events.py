from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.stage4_events import (
    EVENT_TYPES,
    PREDICTION_COLUMNS,
    apply_quality_gated_event_logic,
    build_stage4_event_predictions,
    consolidate_event_episodes,
    detect_window_event_candidates,
    select_stage4_hr_source,
    summarize_stage4_event_metrics,
)


def _stage4_config() -> dict:
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
        "quality_gate": {
            "mode": "suppress",
        },
    }


def _base_frame() -> pd.DataFrame:
    rows = [
        {
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
        },
        {
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
        },
        {
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
        },
        {
            "dataset": "synthetic",
            "subject_id": "S1",
            "window_index": 3,
            "start_time_s": 24.0,
            "duration_s": 8.0,
            "ref_hr_bpm": 107.0,
            "window_is_valid": True,
            "ungated_pred_hr_bpm": 108.0,
            "ungated_is_valid": True,
            "ml_gated_is_valid": False,
            "robust_hr_bpm": 104.0,
            "robust_hr_is_valid": True,
            "policy_reason_code": "quality_good_direct",
        },
        {
            "dataset": "synthetic",
            "subject_id": "S1",
            "window_index": 4,
            "start_time_s": 32.0,
            "duration_s": 8.0,
            "ref_hr_bpm": 48.0,
            "window_is_valid": True,
            "ungated_pred_hr_bpm": 48.0,
            "ungated_is_valid": True,
            "ml_gated_is_valid": True,
            "robust_hr_bpm": 48.0,
            "robust_hr_is_valid": True,
            "policy_reason_code": "quality_good_direct",
        },
        {
            "dataset": "synthetic",
            "subject_id": "S1",
            "window_index": 5,
            "start_time_s": 40.0,
            "duration_s": 8.0,
            "ref_hr_bpm": 46.0,
            "window_is_valid": True,
            "ungated_pred_hr_bpm": 46.0,
            "ungated_is_valid": True,
            "ml_gated_is_valid": True,
            "robust_hr_bpm": 46.0,
            "robust_hr_is_valid": True,
            "policy_reason_code": "quality_good_direct",
        },
        {
            "dataset": "synthetic",
            "subject_id": "S2",
            "window_index": 0,
            "start_time_s": 0.0,
            "duration_s": 8.0,
            "ref_hr_bpm": 70.0,
            "window_is_valid": True,
            "ungated_pred_hr_bpm": 70.0,
            "ungated_is_valid": True,
            "ml_gated_is_valid": True,
            "robust_hr_bpm": 70.0,
            "robust_hr_is_valid": True,
            "policy_reason_code": "quality_good_direct",
        },
        {
            "dataset": "synthetic",
            "subject_id": "S2",
            "window_index": 1,
            "start_time_s": 8.0,
            "duration_s": 8.0,
            "ref_hr_bpm": 94.0,
            "window_is_valid": True,
            "ungated_pred_hr_bpm": 94.0,
            "ungated_is_valid": True,
            "ml_gated_is_valid": True,
            "robust_hr_bpm": 95.0,
            "robust_hr_is_valid": True,
            "policy_reason_code": "beat_fallback_used",
        },
        {
            "dataset": "synthetic",
            "subject_id": "S2",
            "window_index": 2,
            "start_time_s": 16.0,
            "duration_s": 8.0,
            "ref_hr_bpm": 95.0,
            "window_is_valid": True,
            "ungated_pred_hr_bpm": 95.0,
            "ungated_is_valid": True,
            "ml_gated_is_valid": True,
            "robust_hr_bpm": 96.0,
            "robust_hr_is_valid": True,
            "policy_reason_code": "beat_fallback_used",
        },
    ]
    return pd.DataFrame(rows)


def test_select_stage4_hr_source_uses_ml_gate_validity_but_keeps_hr_value() -> None:
    frame = _base_frame().iloc[[3]].copy()
    selected = select_stage4_hr_source(
        frame,
        event_type="tachycardia_event",
        source_config=_stage4_config()["source"],
    )
    assert np.isclose(float(selected.iloc[0]["selected_hr_bpm"]), 108.0)
    assert not bool(selected.iloc[0]["selected_hr_is_valid"])
    assert selected.iloc[0]["quality_gate_reason"] == "stage3_ml_gate_blocked"


def test_detect_and_gate_tachy_candidate_suppresses_low_quality_window() -> None:
    frame = _base_frame().iloc[[3]].copy()
    selected = select_stage4_hr_source(frame, event_type="tachycardia_event", source_config=_stage4_config()["source"])
    detected = detect_window_event_candidates(selected, event_type="tachycardia_event", config=_stage4_config())
    gated = apply_quality_gated_event_logic(detected, event_type="tachycardia_event", config=_stage4_config())
    assert bool(gated.iloc[0]["event_candidate_flag"])
    assert not bool(gated.iloc[0]["quality_gate_passed"])
    assert not bool(gated.iloc[0]["event_validity_flag"])
    assert gated.iloc[0]["event_reason_code"] == "suppressed_low_quality"


def test_build_stage4_event_predictions_creates_expected_long_schema() -> None:
    predictions = build_stage4_event_predictions(_base_frame(), split_name="eval", config=_stage4_config())
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    assert set(predictions["event_type"]) == set(EVENT_TYPES)
    assert predictions.shape[0] == _base_frame().shape[0] * len(EVENT_TYPES)


def test_build_stage4_event_predictions_creates_tachy_and_brady_episodes() -> None:
    predictions = build_stage4_event_predictions(_base_frame(), split_name="eval", config=_stage4_config())

    tachy = predictions.loc[
        (predictions["subject_id"] == "S1")
        & (predictions["event_type"] == "tachycardia_event")
        & (predictions["window_index"].isin([1, 2]))
    ].copy()
    assert tachy["event_validity_flag"].tolist() == [True, True]
    assert tachy["episode_id"].nunique() == 1
    assert tachy["episode_start_flag"].tolist() == [True, False]
    assert tachy["episode_end_flag"].tolist() == [False, True]

    brady = predictions.loc[
        (predictions["subject_id"] == "S1")
        & (predictions["event_type"] == "bradycardia_event")
        & (predictions["window_index"].isin([4, 5]))
    ].copy()
    assert brady["event_validity_flag"].tolist() == [True, True]
    assert brady["episode_id"].nunique() == 1


def test_build_stage4_event_predictions_creates_abrupt_change_confirmation_episode() -> None:
    predictions = build_stage4_event_predictions(_base_frame(), split_name="eval", config=_stage4_config())
    abrupt = predictions.loc[
        (predictions["subject_id"] == "S2")
        & (predictions["event_type"] == "abrupt_change_event")
        & (predictions["window_index"].isin([1, 2]))
    ].copy()
    assert abrupt["event_validity_flag"].tolist() == [True, True]
    assert abrupt["event_trigger_rule"].tolist() == ["abrupt_delta", "abrupt_change_confirmation"]
    assert abrupt["episode_id"].nunique() == 1
    assert np.isclose(float(abrupt.iloc[0]["selected_hr_prev_bpm"]), 70.0)
    assert np.isclose(float(abrupt.iloc[1]["selected_hr_prev_bpm"]), 70.0)


def test_consolidate_event_episodes_merges_gap_within_tolerance() -> None:
    frame = pd.DataFrame(
        {
            "split": ["eval", "eval"],
            "dataset": ["synthetic", "synthetic"],
            "subject_id": ["S1", "S1"],
            "window_index": [1, 3],
            "start_time_s": [8.0, 24.0],
            "duration_s": [8.0, 8.0],
            "event_type": ["tachycardia_event", "tachycardia_event"],
            "event_candidate_flag": [True, True],
            "event_validity_flag": [True, True],
            "event_reason_code": ["candidate_pending_episode", "candidate_pending_episode"],
        }
    )
    consolidated = consolidate_event_episodes(frame, event_type="tachycardia_event", config=_stage4_config())
    assert consolidated["event_validity_flag"].tolist() == [True, True]
    assert consolidated["episode_id"].nunique() == 1
    assert consolidated["episode_start_flag"].tolist() == [True, False]
    assert consolidated["episode_end_flag"].tolist() == [False, True]


def test_summarize_stage4_event_metrics_reports_expected_rows() -> None:
    predictions = build_stage4_event_predictions(_base_frame(), split_name="eval", config=_stage4_config())
    metrics = summarize_stage4_event_metrics(predictions)
    eval_metrics = metrics.loc[metrics["split"] == "eval"].copy()
    assert set(eval_metrics["event_type"]) == set(EVENT_TYPES) | {"all_events"}
    assert "selected_hr_valid_fraction" in eval_metrics.columns
    assert "suppressed_candidate_count" in eval_metrics.columns
    tachy_row = eval_metrics.loc[eval_metrics["event_type"] == "tachycardia_event"].iloc[0]
    assert int(tachy_row["num_pred_events"]) == 1
