from __future__ import annotations

import math
from typing import Any

import numpy as np

from heart_rate_cnn.metrics import compute_precision_recall_f1
from heart_rate_cnn.preprocess import preprocess_ppg_stage1
from heart_rate_cnn.types import WindowSample


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _linear_score(
    value: float,
    *,
    good: float,
    bad: float,
    higher_is_better: bool,
) -> float:
    if not np.isfinite(value):
        return 0.0
    if math.isclose(good, bad):
        return 1.0 if (value >= good if higher_is_better else value <= good) else 0.0

    if higher_is_better:
        if value <= bad:
            return 0.0
        if value >= good:
            return 1.0
        return _clip01((value - bad) / (good - bad))

    if value <= good:
        return 1.0
    if value >= bad:
        return 0.0
    return _clip01((bad - value) / (bad - good))


def build_quality_target(
    *,
    ref_hr_bpm: float | None,
    freq_pred_hr_bpm: float | None,
    window_is_valid: bool,
    freq_is_valid: bool,
    good_error_bpm: float,
    poor_error_bpm: float,
) -> dict[str, float | str]:
    if poor_error_bpm < good_error_bpm:
        raise ValueError("poor_error_bpm must be greater than or equal to good_error_bpm.")

    if not window_is_valid or not freq_is_valid or ref_hr_bpm is None or freq_pred_hr_bpm is None:
        return {
            "quality_target_label": "poor",
            "hr_abs_error_bpm": math.nan,
        }

    ref_hr = float(ref_hr_bpm)
    pred_hr = float(freq_pred_hr_bpm)
    if not np.isfinite(ref_hr) or not np.isfinite(pred_hr):
        return {
            "quality_target_label": "poor",
            "hr_abs_error_bpm": math.nan,
        }

    hr_abs_error_bpm = abs(pred_hr - ref_hr)
    if hr_abs_error_bpm <= good_error_bpm:
        label = "good"
    elif hr_abs_error_bpm >= poor_error_bpm:
        label = "poor"
    else:
        label = "borderline"
    return {
        "quality_target_label": label,
        "hr_abs_error_bpm": float(hr_abs_error_bpm),
    }


def compute_motion_summary(
    acc_window: np.ndarray | None,
    config: dict[str, Any] | None = None,
) -> dict[str, float | bool]:
    cfg = config or {}
    if acc_window is None:
        return {
            "has_acc": False,
            "acc_axis_std_norm": math.nan,
            "acc_mag_range": math.nan,
            "motion_flag": False,
        }

    acc = np.asarray(acc_window, dtype=float)
    if acc.size == 0:
        return {
            "has_acc": False,
            "acc_axis_std_norm": math.nan,
            "acc_mag_range": math.nan,
            "motion_flag": False,
        }
    if acc.ndim == 1:
        acc = acc.reshape(-1, 1)

    centered = acc - np.mean(acc, axis=0, keepdims=True)
    axis_std = np.std(centered, axis=0)
    acc_axis_std_norm = float(np.linalg.norm(axis_std))
    magnitude = np.linalg.norm(centered, axis=1)
    acc_mag_range = float(np.ptp(magnitude)) if magnitude.size else math.nan

    motion_flag = bool(
        acc_axis_std_norm >= float(cfg.get("accel_std_threshold", 0.35))
        or acc_mag_range >= float(cfg.get("accel_range_threshold", 1.5))
    )
    return {
        "has_acc": True,
        "acc_axis_std_norm": acc_axis_std_norm,
        "acc_mag_range": acc_mag_range,
        "motion_flag": motion_flag,
    }


def extract_quality_features(
    window: WindowSample,
    *,
    freq_result: dict[str, float | bool],
    time_result: dict[str, float | int | bool],
    fusion_result: dict[str, float | str | bool],
    preprocess_config: dict[str, Any] | None = None,
    motion_config: dict[str, Any] | None = None,
) -> dict[str, float | str | bool]:
    preprocess_cfg = preprocess_config or {}
    processed = preprocess_ppg_stage1(
        window.ppg,
        fs=window.ppg_fs,
        bandpass_low_hz=float(preprocess_cfg.get("bandpass_low_hz", 0.6)),
        bandpass_high_hz=float(preprocess_cfg.get("bandpass_high_hz", 3.5)),
        bandpass_order=int(preprocess_cfg.get("bandpass_order", 3)),
        smooth_window_seconds=float(preprocess_cfg.get("smooth_window_seconds", 0.20)),
        smooth_polyorder=int(preprocess_cfg.get("smooth_polyorder", 2)),
        extra_smoothing=bool(preprocess_cfg.get("extra_smoothing", False)),
    )

    raw_ppg = np.asarray(window.ppg, dtype=float).reshape(-1)
    centered = raw_ppg - np.mean(raw_ppg)
    diff_std = float(np.std(np.diff(processed))) if processed.size >= 2 else math.nan
    hr_agreement_bpm = math.nan
    freq_hr = float(freq_result.get("freq_pred_hr_bpm", math.nan))
    time_hr = float(time_result.get("time_pred_hr_bpm", math.nan))
    if np.isfinite(freq_hr) and np.isfinite(time_hr):
        hr_agreement_bpm = float(abs(freq_hr - time_hr))

    motion_summary = compute_motion_summary(window.acc, motion_config)
    return {
        "freq_pred_hr_bpm": freq_hr,
        "freq_confidence": float(freq_result.get("freq_confidence", 0.0)),
        "freq_peak_ratio": float(freq_result.get("freq_peak_ratio", 0.0)),
        "freq_is_valid": bool(freq_result.get("freq_is_valid", False)),
        "time_pred_hr_bpm": time_hr,
        "time_confidence": float(time_result.get("time_confidence", 0.0)),
        "time_num_peaks": float(time_result.get("time_num_peaks", 0)),
        "time_is_valid": bool(time_result.get("time_is_valid", False)),
        "fusion_pred_hr_bpm": float(fusion_result.get("fusion_pred_hr_bpm", math.nan)),
        "fusion_confidence": float(fusion_result.get("fusion_confidence", 0.0)),
        "fusion_source": str(fusion_result.get("fusion_source", "none")),
        "ppg_centered_std": float(np.std(centered)),
        "ppg_peak_to_peak": float(np.ptp(centered)) if centered.size else math.nan,
        "ppg_processed_diff_std": diff_std,
        "hr_agreement_bpm": hr_agreement_bpm,
        **motion_summary,
    }


def score_quality_rule_based(
    *,
    window_is_valid: bool,
    features: dict[str, float | str | bool],
    config: dict[str, Any] | None = None,
) -> float:
    cfg = config or {}
    if not window_is_valid or not bool(features.get("freq_is_valid", False)):
        return 0.0

    weights = cfg.get("weights", {})
    weight_freq_conf = float(weights.get("freq_confidence", 0.45))
    weight_peak_ratio = float(weights.get("peak_ratio", 0.25))
    weight_agreement = float(weights.get("agreement", 0.15))
    weight_time_conf = float(weights.get("time_confidence", 0.10))
    weight_smoothness = float(weights.get("smoothness", 0.05))
    total_weight = weight_freq_conf + weight_peak_ratio + weight_agreement + weight_time_conf + weight_smoothness
    if total_weight <= 0:
        raise ValueError("Stage 3 rule weights must sum to a positive value.")

    freq_conf_score = _clip01(float(features.get("freq_confidence", 0.0)))
    peak_ratio_score = _linear_score(
        float(features.get("freq_peak_ratio", math.nan)),
        good=float(cfg.get("peak_ratio_good", 2.5)),
        bad=float(cfg.get("peak_ratio_bad", 1.1)),
        higher_is_better=True,
    )
    agreement_score = _linear_score(
        float(features.get("hr_agreement_bpm", math.nan)),
        good=float(cfg.get("agreement_good_bpm", 3.0)),
        bad=float(cfg.get("agreement_bad_bpm", 12.0)),
        higher_is_better=False,
    )
    if not bool(features.get("time_is_valid", False)):
        agreement_score = float(cfg.get("missing_time_agreement_score", 0.0))
    time_conf_score = _clip01(float(features.get("time_confidence", 0.0))) if bool(features.get("time_is_valid", False)) else 0.0
    smoothness_score = _linear_score(
        float(features.get("ppg_processed_diff_std", math.nan)),
        good=float(cfg.get("diff_std_good", 0.12)),
        bad=float(cfg.get("diff_std_bad", 0.35)),
        higher_is_better=False,
    )

    score = (
        weight_freq_conf * freq_conf_score
        + weight_peak_ratio * peak_ratio_score
        + weight_agreement * agreement_score
        + weight_time_conf * time_conf_score
        + weight_smoothness * smoothness_score
    ) / total_weight
    return _clip01(score)


def apply_rule_based_quality_decision(
    *,
    window_is_valid: bool,
    features: dict[str, float | str | bool],
    config: dict[str, Any] | None = None,
) -> dict[str, float | str | bool]:
    cfg = config or {}
    score = score_quality_rule_based(window_is_valid=window_is_valid, features=features, config=cfg)
    label = "good" if score >= float(cfg.get("good_score_threshold", 0.55)) else "poor"
    validity_flag = bool(window_is_valid and bool(features.get("freq_is_valid", False)) and label == "good")
    return {
        "signal_quality_score": score,
        "signal_quality_label": label,
        "validity_flag": validity_flag,
        "motion_flag": bool(features.get("motion_flag", False)),
    }


def compute_binary_classification_summary(
    target_labels: list[str],
    predicted_labels: list[str],
    *,
    positive_label: str = "good",
) -> dict[str, float]:
    if len(target_labels) != len(predicted_labels):
        raise ValueError("target_labels and predicted_labels must have the same length.")
    if not target_labels:
        return {
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "num_eval_windows": 0.0,
        }

    tp = 0
    fp = 0
    fn = 0
    correct = 0
    for target, predicted in zip(target_labels, predicted_labels):
        if target == predicted:
            correct += 1
        if predicted == positive_label and target == positive_label:
            tp += 1
        elif predicted == positive_label and target != positive_label:
            fp += 1
        elif predicted != positive_label and target == positive_label:
            fn += 1

    metrics = compute_precision_recall_f1(tp, fp, fn)
    return {
        "accuracy": float(correct / len(target_labels)),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "num_eval_windows": float(len(target_labels)),
    }
