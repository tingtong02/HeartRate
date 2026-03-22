from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from heart_rate_cnn.metrics import compute_hr_metrics, compute_precision_recall_f1
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


ML_FEATURE_NAMES = [
    "freq_confidence",
    "freq_peak_ratio",
    "freq_is_valid",
    "time_confidence",
    "time_num_peaks",
    "time_is_valid",
    "fusion_confidence",
    "hr_agreement_bpm",
    "hr_agreement_ratio",
    "fusion_is_blended",
    "ppg_centered_std",
    "ppg_peak_to_peak",
    "ppg_processed_diff_std",
    "has_acc",
    "acc_axis_std_norm",
    "acc_mag_range",
]


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def build_ml_feature_row(features: dict[str, float | str | bool]) -> dict[str, float]:
    freq_pred_hr_bpm = _safe_numeric(features.get("freq_pred_hr_bpm", math.nan), default=math.nan)
    hr_agreement_bpm = _safe_numeric(features.get("hr_agreement_bpm", math.nan), default=math.nan)
    if np.isfinite(freq_pred_hr_bpm) and abs(freq_pred_hr_bpm) > 1e-8 and np.isfinite(hr_agreement_bpm):
        hr_agreement_ratio = float(hr_agreement_bpm / abs(freq_pred_hr_bpm))
    else:
        hr_agreement_ratio = 0.0

    fusion_source = str(features.get("fusion_source", "none"))
    row = {
        "freq_confidence": _safe_numeric(features.get("freq_confidence", 0.0)),
        "freq_peak_ratio": _safe_numeric(features.get("freq_peak_ratio", 0.0)),
        "freq_is_valid": _safe_numeric(features.get("freq_is_valid", False)),
        "time_confidence": _safe_numeric(features.get("time_confidence", 0.0)),
        "time_num_peaks": _safe_numeric(features.get("time_num_peaks", 0.0)),
        "time_is_valid": _safe_numeric(features.get("time_is_valid", False)),
        "fusion_confidence": _safe_numeric(features.get("fusion_confidence", 0.0)),
        "hr_agreement_bpm": _safe_numeric(features.get("hr_agreement_bpm", 0.0)),
        "hr_agreement_ratio": hr_agreement_ratio,
        "fusion_is_blended": float(fusion_source == "blended"),
        "ppg_centered_std": _safe_numeric(features.get("ppg_centered_std", 0.0)),
        "ppg_peak_to_peak": _safe_numeric(features.get("ppg_peak_to_peak", 0.0)),
        "ppg_processed_diff_std": _safe_numeric(features.get("ppg_processed_diff_std", 0.0)),
        "has_acc": _safe_numeric(features.get("has_acc", False)),
        "acc_axis_std_norm": _safe_numeric(features.get("acc_axis_std_norm", 0.0)),
        "acc_mag_range": _safe_numeric(features.get("acc_mag_range", 0.0)),
    }
    return row


def build_ml_feature_matrix(feature_rows: list[dict[str, float | str | bool]]) -> np.ndarray:
    if not feature_rows:
        return np.empty((0, len(ML_FEATURE_NAMES)), dtype=float)
    matrix = []
    for feature_row in feature_rows:
        ml_row = build_ml_feature_row(feature_row)
        matrix.append([ml_row[name] for name in ML_FEATURE_NAMES])
    return np.asarray(matrix, dtype=float)


def fit_quality_logistic_regression(
    feature_rows: list[dict[str, float | str | bool]],
    target_labels: list[str],
    *,
    random_seed: int = 42,
    c_value: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline | dict[str, float | str]:
    if len(feature_rows) != len(target_labels):
        raise ValueError("feature_rows and target_labels must have the same length.")
    if not feature_rows:
        raise ValueError("At least one training row is required.")

    y = np.asarray([1 if label == "good" else 0 for label in target_labels], dtype=int)
    unique = np.unique(y)
    if unique.size < 2:
        return {
            "model_type": "constant",
            "probability_good": float(unique[0]) if unique.size == 1 else 0.0,
        }

    x = build_ml_feature_matrix(feature_rows)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=random_seed,
                    C=float(c_value),
                    max_iter=int(max_iter),
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def predict_quality_logistic_regression(
    model: Pipeline | dict[str, float | str],
    feature_rows: list[dict[str, float | str | bool]],
) -> np.ndarray:
    if not feature_rows:
        return np.array([], dtype=float)
    if isinstance(model, dict):
        probability_good = float(model.get("probability_good", 0.0))
        return np.full(len(feature_rows), probability_good, dtype=float)

    x = build_ml_feature_matrix(feature_rows)
    probabilities = model.predict_proba(x)[:, 1]
    return np.asarray(probabilities, dtype=float)


def apply_ml_quality_decision(
    *,
    signal_quality_score: float,
    threshold: float,
    window_is_valid: bool,
    freq_is_valid: bool,
    motion_flag: bool,
) -> dict[str, float | str | bool]:
    score = _clip01(signal_quality_score)
    label = "good" if score >= threshold else "poor"
    validity_flag = bool(window_is_valid and freq_is_valid and label == "good")
    return {
        "signal_quality_score": score,
        "signal_quality_label": label,
        "validity_flag": validity_flag,
        "motion_flag": motion_flag,
    }


def select_best_ml_threshold(
    frame: pd.DataFrame,
    *,
    score_col: str,
    pred_col: str,
    valid_col: str,
    ref_col: str = "ref_hr_bpm",
    target_col: str = "quality_target_label",
    threshold_grid: list[float] | tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    min_retention_ratio: float = 0.9,
) -> dict[str, float]:
    if frame.empty:
        raise ValueError("select_best_ml_threshold requires a non-empty frame.")

    ungated_valid_count = int(frame[valid_col].fillna(False).astype(bool).sum())
    if ungated_valid_count <= 0:
        return {
            "selected_threshold": float(threshold_grid[0]),
            "retention_ratio": 0.0,
            "mae": math.nan,
            "f1": math.nan,
        }

    classification_frame = frame.loc[frame[target_col].isin(["good", "poor"])].copy()
    rows: list[dict[str, float | bool]] = []
    for threshold in threshold_grid:
        predicted_labels = np.where(frame[score_col].to_numpy(dtype=float) >= threshold, "good", "poor")
        valid_mask = (
            frame[ref_col].notna()
            & frame[pred_col].notna()
            & frame[valid_col].astype(bool)
            & (frame[score_col].to_numpy(dtype=float) >= threshold)
        )
        gated_frame = frame.loc[valid_mask]
        hr_metrics = compute_hr_metrics(
            gated_frame[ref_col].to_numpy(dtype=float),
            gated_frame[pred_col].to_numpy(dtype=float),
        )

        if classification_frame.empty:
            f1 = math.nan
        else:
            class_predictions = np.where(
                classification_frame[score_col].to_numpy(dtype=float) >= threshold,
                "good",
                "poor",
            )
            class_metrics = compute_binary_classification_summary(
                classification_frame[target_col].tolist(),
                class_predictions.tolist(),
            )
            f1 = float(class_metrics["f1"]) if not math.isnan(float(class_metrics["f1"])) else math.nan

        retention_ratio = float(gated_frame.shape[0] / ungated_valid_count)
        rows.append(
            {
                "threshold": float(threshold),
                "retention_ratio": retention_ratio,
                "mae": float(hr_metrics["mae"]) if not math.isnan(float(hr_metrics["mae"])) else math.inf,
                "f1": f1 if not math.isnan(f1) else -1.0,
                "feasible": bool(retention_ratio >= min_retention_ratio),
            }
        )

    threshold_frame = pd.DataFrame(rows)
    feasible = threshold_frame.loc[threshold_frame["feasible"]].copy()
    candidate_frame = feasible if not feasible.empty else threshold_frame
    candidate_frame = candidate_frame.sort_values(
        by=["mae", "retention_ratio", "f1", "threshold"],
        ascending=[True, False, False, True],
    )
    best = candidate_frame.iloc[0].to_dict()
    selected_mae = float(best["mae"])
    return {
        "selected_threshold": float(best["threshold"]),
        "retention_ratio": float(best["retention_ratio"]),
        "mae": selected_mae if math.isfinite(selected_mae) else math.nan,
        "f1": float(best["f1"]) if float(best["f1"]) >= 0 else math.nan,
    }
