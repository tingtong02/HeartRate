from __future__ import annotations

import numpy as np
import pandas as pd

from heart_rate_cnn.stage4_features import build_stage4_shared_feature_frame
from heart_rate_cnn.stage4_irregular import (
    MODEL_FEATURE_COLUMNS,
    PREDICTION_COLUMNS,
    build_irregular_proxy_labels,
    build_rule_baseline_candidates,
    build_screening_predictions,
    fit_hist_gbdt_irregular_classifier,
    predict_hist_gbdt_irregular_scores,
    summarize_stage4_irregular_metrics,
)
from heart_rate_cnn.types import SubjectRecord


def _stage4_shared_config() -> dict:
    return {
        "selected_hr_source": "robust_stage3c2_policy",
        "beat_variant_mode": "enhanced",
        "min_beats": 4,
        "min_ibi_clean": 3,
        "min_ref_ibi_clean": 3,
    }


def _stage3_config() -> dict:
    return {
        "robust_hr_policy": {
            "fallback_variant_mode": "enhanced",
            "fallback_min_ibi_s": 0.33,
            "fallback_max_ibi_s": 1.5,
            "fallback_local_median_radius": 2,
            "fallback_max_deviation_ratio": 0.25,
            "fallback_adjacent_jump_ratio": 0.22,
            "fallback_jump_anchor_ratio": 0.12,
            "fallback_short_series_threshold": 5,
            "fallback_min_clean_ibi": 3,
            "fallback_beat_quality_threshold": 0.55,
            "fallback_plausibility_margin_s": 0.08,
            "fallback_jump_good_ratio": 0.08,
            "fallback_jump_bad_ratio": 0.25,
            "fallback_crowding_good_scale": 1.10,
            "fallback_missing_ibi_score": 0.50,
            "fallback_weight_base_peak_quality": 0.60,
            "fallback_weight_ibi_plausibility": 0.20,
            "fallback_weight_ibi_stability": 0.10,
            "fallback_weight_crowding": 0.05,
            "fallback_weight_clean_pair_bonus": 0.05,
        }
    }


class _SyntheticLoader:
    def __init__(self, record: SubjectRecord):
        self._record = record

    def load_subject(self, subject_id: str) -> SubjectRecord:
        if subject_id != self._record.subject_id:
            raise KeyError(subject_id)
        return self._record


def _make_record() -> SubjectRecord:
    ppg_fs = 64.0
    ecg_fs = 128.0
    duration_s = 16.0
    ppg_t = np.arange(int(duration_s * ppg_fs)) / ppg_fs
    ecg_t = np.arange(int(duration_s * ecg_fs)) / ecg_fs
    ppg = 0.8 * np.sin(2.0 * np.pi * 1.1 * ppg_t) + 0.15 * np.sin(2.0 * np.pi * 2.2 * ppg_t)
    ecg = np.zeros_like(ecg_t)
    for beat_time in np.arange(0.75, duration_s, 0.95):
        beat_index = int(round(beat_time * ecg_fs))
        if 2 <= beat_index < ecg.size - 2:
            ecg[beat_index - 2 : beat_index + 3] = np.array([0.0, 0.8, 1.6, 0.8, 0.0])
    return SubjectRecord(
        dataset="synthetic",
        subject_id="S1",
        ppg=ppg.astype(float),
        ppg_fs=ppg_fs,
        ecg=ecg.astype(float),
        ecg_fs=ecg_fs,
        acc=None,
        acc_fs=None,
    )


def _source_frame() -> pd.DataFrame:
    rows = []
    for window_index, start_time_s in enumerate([0.0, 2.0, 4.0, 6.0, 8.0]):
        rows.append(
            {
                "split": "train",
                "dataset": "synthetic",
                "subject_id": "S1",
                "window_index": window_index,
                "start_time_s": start_time_s,
                "duration_s": 8.0,
                "ref_hr_bpm": 68.0 + window_index,
                "window_is_valid": True,
                "freq_confidence": 0.7,
                "freq_peak_ratio": 2.4,
                "time_confidence": 0.6,
                "time_num_peaks": 8.0,
                "hr_agreement_bpm": 2.0,
                "ppg_centered_std": 0.4,
                "ppg_peak_to_peak": 1.2,
                "ppg_processed_diff_std": 0.09,
                "has_acc": False,
                "acc_axis_std_norm": np.nan,
                "acc_mag_range": np.nan,
                "motion_flag": False,
                "rule_signal_quality_score": 0.7,
                "rule_validity_flag": True,
                "ml_signal_quality_score": 0.8,
                "ml_validity_flag": True,
                "beat_fallback_available": True,
                "beat_fallback_num_beats": 8.0,
                "beat_fallback_num_clean_ibi": 7.0,
                "beat_fallback_kept_ratio": 0.8,
                "beat_fallback_reason": "available",
                "robust_hr_bpm": 68.0 + window_index,
                "robust_hr_is_valid": True,
                "robust_hr_source": "frequency",
                "robust_hr_action": "direct_update",
                "hold_applied": False,
                "hold_age_windows": 0.0,
                "hr_jump_bpm_from_previous": 1.0,
                "policy_reason_code": "quality_good_direct",
                "subject_boundary_reset": False,
                "ungated_pred_hr_bpm": 68.0 + window_index,
                "ungated_is_valid": True,
                "ml_gated_is_valid": True,
            }
        )
    return pd.DataFrame(rows)


def _irregular_frame() -> pd.DataFrame:
    rows = []
    for index in range(6):
        row = {
            "split": "train" if index < 4 else "eval",
            "dataset": "synthetic",
            "subject_id": "S1",
            "window_index": index,
            "start_time_s": float(index * 2),
            "duration_s": 8.0,
            "selected_hr_source": "robust_stage3c2_policy",
            "selected_hr_bpm": 70.0 + index,
            "selected_hr_is_valid": True,
            "selected_hr_delta_bpm": 2.0,
            "selected_hr_missing_flag": False,
            "ml_signal_quality_score": 0.9,
            "ml_validity_flag": True,
            "rule_signal_quality_score": 0.8,
            "rule_validity_flag": True,
            "motion_flag": False,
            "has_acc": False,
            "acc_axis_std_norm": np.nan,
            "acc_mag_range": np.nan,
            "freq_confidence": 0.7,
            "freq_peak_ratio": 2.5,
            "time_confidence": 0.6,
            "time_num_peaks": 8.0,
            "hr_agreement_bpm": 2.0,
            "ppg_processed_diff_std": 0.10,
            "robust_hr_source": "frequency",
            "robust_hr_action": "direct_update",
            "robust_hr_is_valid": True,
            "hold_applied": False,
            "hold_age_windows": 0.0,
            "hr_jump_bpm_from_previous": 2.0,
            "num_beats": 8.0,
            "num_ibi_raw": 7.0,
            "num_ibi_clean": 6.0,
            "ibi_removed_ratio": 0.1,
            "ibi_is_valid": True,
            "insufficient_beats_flag": False,
            "insufficient_clean_ibi_flag": False,
            "beat_quality_mean_score": 0.7,
            "beat_quality_good_ratio": 0.75,
            "beat_quality_good_count": 6.0,
            "beat_fallback_available": True,
            "beat_fallback_num_beats": 8.0,
            "beat_fallback_num_clean_ibi": 6.0,
            "beat_fallback_kept_ratio": 0.75,
            "mean_ibi_ms": 800.0,
            "median_ibi_ms": 790.0,
            "mean_hr_bpm_from_ibi": 75.0,
            "sdnn_ms": 20.0,
            "rmssd_ms": 22.0,
            "pnn50": 0.05,
            "ibi_cv": 0.03,
            "ibi_mad_ms": 10.0,
            "successive_ibi_jump_mean_ms": 20.0,
            "successive_ibi_jump_max_ms": 35.0,
            "local_deviation_ratio_mean": 0.04,
            "local_deviation_ratio_max": 0.06,
            "turning_point_ratio": 0.20,
            "ref_num_ibi_clean": 6.0,
            "ref_ibi_is_valid": True,
            "ref_rmssd_ms": 18.0,
            "ref_pnn50": 0.05,
            "ref_ibi_cv": 0.03,
            "ref_local_deviation_ratio_max": 0.05,
            "robust_source_is_frequency": 1.0,
            "robust_source_is_beat_fallback": 0.0,
            "robust_source_is_hold_previous": 0.0,
            "robust_source_is_none": 0.0,
            "robust_action_is_direct_update": 1.0,
            "robust_action_is_fallback_update": 0.0,
            "robust_action_is_hold": 0.0,
            "robust_action_is_reject": 0.0,
        }
        if index in (1, 4):
            row["rmssd_ms"] = 95.0
            row["pnn50"] = 0.42
            row["ibi_cv"] = 0.16
            row["local_deviation_ratio_max"] = 0.24
            row["turning_point_ratio"] = 0.70
            row["ref_rmssd_ms"] = 110.0
            row["ref_pnn50"] = 0.50
            row["ref_ibi_cv"] = 0.18
            row["ref_local_deviation_ratio_max"] = 0.23
        rows.append(row)

    frame = pd.DataFrame(rows)
    for column_name in MODEL_FEATURE_COLUMNS:
        if column_name not in frame.columns:
            frame[column_name] = 0.0
    return frame


def test_build_stage4_shared_feature_frame_includes_expected_columns() -> None:
    loader = _SyntheticLoader(_make_record())
    feature_frame = build_stage4_shared_feature_frame(
        loader=loader,
        subject_ids=["S1"],
        split_name="train",
        preprocess_cfg={"target_ppg_fs": 64, "window_seconds": 8.0, "step_seconds": 2.0},
        stage3_cfg=_stage3_config(),
        stage4_shared_cfg=_stage4_shared_config(),
        source_frame=_source_frame(),
    )
    assert feature_frame.shape[0] == 5
    expected_columns = {
        "selected_hr_source",
        "num_beats",
        "num_ibi_clean",
        "rmssd_ms",
        "beat_quality_mean_score",
        "ref_rmssd_ms",
        "robust_hr_source",
    }
    assert expected_columns.issubset(feature_frame.columns)


def test_build_irregular_proxy_labels_uses_reference_irregularity_thresholds() -> None:
    frame = _irregular_frame().iloc[[0, 1]].copy()
    labeled = build_irregular_proxy_labels(
        frame,
        config={
            "label": {
                "positive_rule": "any",
                "min_ref_ibi_clean": 3,
                "irregular_rmssd_ms": 80.0,
                "irregular_pnn50": 0.35,
                "irregular_ibi_cv": 0.12,
                "irregular_local_deviation_ratio": 0.18,
            }
        },
    )
    assert labeled["screening_proxy_target"].tolist() == [False, True]
    assert labeled["proxy_label_support_flag"].tolist() == [True, True]
    assert labeled.iloc[1]["proxy_label_reason"] != "reference_regular"


def test_quality_gate_suppresses_low_quality_positive_candidate() -> None:
    frame = build_irregular_proxy_labels(_irregular_frame().iloc[[1]].copy(), config={"label": {"min_ref_ibi_clean": 3}})
    frame.loc[:, "ml_validity_flag"] = False
    predictions = build_screening_predictions(
        frame,
        model_name="hist_gbdt_irregular",
        scores=np.asarray([0.95], dtype=float),
        threshold=0.50,
        candidate_reasons=["score_threshold_met"],
        candidate_indicator_counts=[0],
        quality_gate_config={
            "mode": "suppress",
            "require_selected_hr_valid": True,
            "require_stage3_quality_pass": True,
            "stage3_quality_flag_column": "ml_validity_flag",
            "require_support_sufficient": True,
            "disallowed_robust_sources": ["none", "hold_previous"],
            "disallowed_robust_actions": ["hold", "reject"],
        },
    )
    assert bool(predictions.iloc[0]["screening_candidate_flag"])
    assert not bool(predictions.iloc[0]["quality_gate_passed"])
    assert not bool(predictions.iloc[0]["irregular_pulse_flag"])
    assert predictions.iloc[0]["quality_gate_reason"] == "stage3_quality_blocked"


def test_rule_baseline_and_hist_gbdt_training_work_on_synthetic_frame() -> None:
    labeled = build_irregular_proxy_labels(_irregular_frame().copy(), config={"label": {"min_ref_ibi_clean": 3}})
    rule_scores, rule_reasons, rule_counts = build_rule_baseline_candidates(
        labeled,
        config={
            "min_positive_indicators": 2,
            "thresholds": {
                "rmssd_ms": 70.0,
                "pnn50": 0.30,
                "ibi_cv": 0.10,
                "local_deviation_ratio_max": 0.16,
                "turning_point_ratio": 0.55,
            },
        },
    )
    assert rule_scores.shape[0] == labeled.shape[0]
    assert max(rule_counts) >= 2
    assert any(reason != "no_rule_trigger" for reason in rule_reasons)

    model = fit_hist_gbdt_irregular_classifier(
        labeled,
        config={"learning_rate": 0.05, "max_depth": 3, "max_iter": 50, "min_samples_leaf": 1, "random_seed": 42},
    )
    scores = predict_hist_gbdt_irregular_scores(model, labeled)
    assert scores.shape[0] == labeled.shape[0]
    assert np.all(np.isfinite(scores))


def test_constant_model_path_handles_all_negative_training_labels() -> None:
    labeled = build_irregular_proxy_labels(_irregular_frame().copy(), config={"label": {"min_ref_ibi_clean": 3}})
    labeled.loc[:, "screening_proxy_target"] = False
    model = fit_hist_gbdt_irregular_classifier(labeled, config={"max_iter": 20, "random_seed": 42})
    scores = predict_hist_gbdt_irregular_scores(model, labeled)
    assert np.allclose(scores, 0.0)


def test_prediction_schema_and_metrics_summary_are_stable() -> None:
    labeled = build_irregular_proxy_labels(_irregular_frame().copy(), config={"label": {"min_ref_ibi_clean": 3}})
    predictions = build_screening_predictions(
        labeled,
        model_name="hist_gbdt_irregular",
        scores=np.asarray([0.1, 0.9, 0.2, 0.3, 0.8, 0.1], dtype=float),
        threshold=0.50,
        candidate_reasons=["score_below_threshold", "score_threshold_met", "score_below_threshold", "score_below_threshold", "score_threshold_met", "score_below_threshold"],
        candidate_indicator_counts=[0, 0, 0, 0, 0, 0],
        quality_gate_config={
            "mode": "suppress",
            "require_selected_hr_valid": True,
            "require_stage3_quality_pass": True,
            "stage3_quality_flag_column": "ml_validity_flag",
            "require_support_sufficient": True,
            "disallowed_robust_sources": ["none", "hold_previous"],
            "disallowed_robust_actions": ["hold", "reject"],
        },
    )
    assert set(PREDICTION_COLUMNS).issubset(predictions.columns)
    metrics = summarize_stage4_irregular_metrics(predictions)
    expected_metric_columns = {
        "task",
        "method",
        "split",
        "num_eval_windows",
        "num_positive_targets",
        "num_positive_predictions",
        "f1",
        "quality_gate_pass_fraction",
    }
    assert expected_metric_columns.issubset(metrics.columns)
