from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from heart_rate_cnn.metrics import compute_precision_recall_f1
from heart_rate_cnn.stage4_features import STAGE4_IDENTITY_COLUMNS, safe_bool, safe_float


DEFAULT_MODEL_NAME = "hist_gbdt_irregular"
RULE_BASELINE_NAME = "irregular_rule_baseline"

MODEL_FEATURE_COLUMNS: tuple[str, ...] = (
    "mean_ibi_ms",
    "median_ibi_ms",
    "mean_hr_bpm_from_ibi",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "ibi_cv",
    "ibi_mad_ms",
    "successive_ibi_jump_mean_ms",
    "successive_ibi_jump_max_ms",
    "local_deviation_ratio_mean",
    "local_deviation_ratio_max",
    "turning_point_ratio",
    "selected_hr_delta_bpm",
    "num_beats",
    "num_ibi_raw",
    "num_ibi_clean",
    "ibi_removed_ratio",
    "ibi_is_valid",
    "beat_quality_mean_score",
    "beat_quality_good_ratio",
    "beat_quality_good_count",
    "beat_fallback_available",
    "beat_fallback_num_beats",
    "beat_fallback_num_clean_ibi",
    "beat_fallback_kept_ratio",
    "ml_signal_quality_score",
    "rule_signal_quality_score",
    "ml_validity_flag",
    "rule_validity_flag",
    "motion_flag",
    "has_acc",
    "acc_axis_std_norm",
    "acc_mag_range",
    "freq_confidence",
    "freq_peak_ratio",
    "time_confidence",
    "time_num_peaks",
    "hr_agreement_bpm",
    "ppg_processed_diff_std",
    "robust_hr_is_valid",
    "hold_applied",
    "hold_age_windows",
    "hr_jump_bpm_from_previous",
    "insufficient_beats_flag",
    "insufficient_clean_ibi_flag",
    "selected_hr_missing_flag",
    "robust_source_is_frequency",
    "robust_source_is_beat_fallback",
    "robust_source_is_hold_previous",
    "robust_source_is_none",
    "robust_action_is_direct_update",
    "robust_action_is_fallback_update",
    "robust_action_is_hold",
    "robust_action_is_reject",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
    "model_name",
    "screening_proxy_target",
    "proxy_label_support_flag",
    "proxy_label_reason",
    "screening_score",
    "irregular_pulse_score",
    "screening_threshold",
    "screening_candidate_flag",
    "candidate_reason_code",
    "candidate_indicator_count",
    "irregular_pulse_flag",
    "screening_validity_flag",
    "screening_reason_code",
    "quality_gate_passed",
    "quality_gate_reason",
    "support_sufficient_flag",
    "selected_hr_source",
    "selected_hr_bpm",
    "selected_hr_is_valid",
    "selected_hr_delta_bpm",
    "num_beats",
    "num_ibi_clean",
    "ibi_is_valid",
    "ml_signal_quality_score",
    "robust_hr_source",
    "robust_hr_action",
)


def _safe_clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def compute_support_sufficient_flags(feature_frame: pd.DataFrame) -> np.ndarray:
    if feature_frame.empty:
        return np.array([], dtype=bool)
    return (
        (feature_frame["num_beats"].to_numpy(dtype=float) >= 1.0)
        & (feature_frame["num_ibi_clean"].to_numpy(dtype=float) >= 1.0)
        & feature_frame["ibi_is_valid"].astype(bool).to_numpy()
        & ~feature_frame["selected_hr_missing_flag"].astype(bool).to_numpy()
        & ~feature_frame["insufficient_beats_flag"].astype(bool).to_numpy()
        & ~feature_frame["insufficient_clean_ibi_flag"].astype(bool).to_numpy()
    )


def build_irregular_proxy_labels(
    feature_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    label_cfg = cfg.get("label", {})
    positive_rule = str(label_cfg.get("positive_rule", "any"))
    min_ref_ibi_clean = int(label_cfg.get("min_ref_ibi_clean", 3))
    rmssd_threshold = float(label_cfg.get("irregular_rmssd_ms", 80.0))
    pnn50_threshold = float(label_cfg.get("irregular_pnn50", 0.35))
    ibi_cv_threshold = float(label_cfg.get("irregular_ibi_cv", 0.12))
    local_dev_threshold = float(label_cfg.get("irregular_local_deviation_ratio", 0.18))

    labeled = feature_frame.copy()
    proxy_targets: list[bool] = []
    support_flags: list[bool] = []
    reasons: list[str] = []

    for row in labeled.to_dict(orient="records"):
        support_flag = bool(row.get("ref_ibi_is_valid", False)) and int(safe_float(row.get("ref_num_ibi_clean"), default=0.0)) >= min_ref_ibi_clean
        triggered_rules: list[str] = []
        if safe_float(row.get("ref_rmssd_ms"), default=math.nan) >= rmssd_threshold:
            triggered_rules.append("ref_rmssd_high")
        if safe_float(row.get("ref_pnn50"), default=math.nan) >= pnn50_threshold:
            triggered_rules.append("ref_pnn50_high")
        if safe_float(row.get("ref_ibi_cv"), default=math.nan) >= ibi_cv_threshold:
            triggered_rules.append("ref_ibi_cv_high")
        if safe_float(row.get("ref_local_deviation_ratio_max"), default=math.nan) >= local_dev_threshold:
            triggered_rules.append("ref_local_deviation_high")

        positive = False
        reason = "reference_regular"
        if not support_flag:
            reason = "insufficient_reference_ibi_support"
        else:
            if positive_rule == "at_least_two":
                positive = len(triggered_rules) >= 2
            else:
                positive = len(triggered_rules) >= 1
            if positive:
                reason = "|".join(triggered_rules)

        proxy_targets.append(bool(positive))
        support_flags.append(bool(support_flag))
        reasons.append(reason)

    labeled["screening_proxy_target"] = proxy_targets
    labeled["proxy_label_support_flag"] = support_flags
    labeled["proxy_label_reason"] = reasons
    return labeled


def build_rule_baseline_candidates(
    feature_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], list[int]]:
    cfg = config or {}
    threshold_cfg = cfg.get("thresholds", {})
    min_indicators = int(cfg.get("min_positive_indicators", 2))
    rmssd_threshold = float(threshold_cfg.get("rmssd_ms", 70.0))
    pnn50_threshold = float(threshold_cfg.get("pnn50", 0.30))
    ibi_cv_threshold = float(threshold_cfg.get("ibi_cv", 0.10))
    local_dev_threshold = float(threshold_cfg.get("local_deviation_ratio_max", 0.16))
    turning_point_threshold = float(threshold_cfg.get("turning_point_ratio", 0.55))

    scores: list[float] = []
    reasons: list[str] = []
    counts: list[int] = []
    for row in feature_frame.to_dict(orient="records"):
        triggered: list[str] = []
        if safe_float(row.get("rmssd_ms"), default=math.nan) >= rmssd_threshold:
            triggered.append("rmssd_high")
        if safe_float(row.get("pnn50"), default=math.nan) >= pnn50_threshold:
            triggered.append("pnn50_high")
        if safe_float(row.get("ibi_cv"), default=math.nan) >= ibi_cv_threshold:
            triggered.append("ibi_cv_high")
        if safe_float(row.get("local_deviation_ratio_max"), default=math.nan) >= local_dev_threshold:
            triggered.append("local_deviation_high")
        if safe_float(row.get("turning_point_ratio"), default=math.nan) >= turning_point_threshold:
            triggered.append("turning_point_high")
        indicator_count = len(triggered)
        counts.append(indicator_count)
        if indicator_count >= min_indicators:
            positive_span = max(5 - min_indicators, 1)
            score = 0.5 + 0.5 * float((indicator_count - min_indicators) / positive_span)
        else:
            score = 0.49 * float(indicator_count / max(min_indicators, 1))
        scores.append(score)
        reasons.append("|".join(triggered) if triggered else "no_rule_trigger")
    return np.asarray(scores, dtype=float), reasons, counts


def build_model_matrix(feature_frame: pd.DataFrame) -> np.ndarray:
    if feature_frame.empty:
        return np.empty((0, len(MODEL_FEATURE_COLUMNS)), dtype=float)
    matrix = []
    for row in feature_frame.to_dict(orient="records"):
        matrix.append([safe_float(row.get(column_name), default=math.nan) for column_name in MODEL_FEATURE_COLUMNS])
    return np.asarray(matrix, dtype=float)


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.array([], dtype=float)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return np.ones(labels.size, dtype=float)
    positive_weight = labels.size / (2.0 * positives)
    negative_weight = labels.size / (2.0 * negatives)
    return np.where(labels == 1, positive_weight, negative_weight).astype(float)


def fit_hist_gbdt_irregular_classifier(
    feature_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> HistGradientBoostingClassifier | dict[str, float | str]:
    cfg = config or {}
    train_frame = feature_frame.loc[feature_frame["proxy_label_support_flag"].astype(bool)].copy()
    if train_frame.empty:
        return {"model_type": "constant", "probability_positive": 0.0}

    y = train_frame["screening_proxy_target"].astype(bool).astype(int).to_numpy(dtype=int)
    unique = np.unique(y)
    if unique.size < 2:
        return {
            "model_type": "constant",
            "probability_positive": float(unique[0]) if unique.size == 1 else 0.0,
        }

    x = build_model_matrix(train_frame)
    model = HistGradientBoostingClassifier(
        learning_rate=float(cfg.get("learning_rate", 0.05)),
        max_depth=int(cfg.get("max_depth", 3)),
        max_iter=int(cfg.get("max_iter", 200)),
        min_samples_leaf=int(cfg.get("min_samples_leaf", 20)),
        l2_regularization=float(cfg.get("l2_regularization", 0.0)),
        random_state=int(cfg.get("random_seed", 42)),
    )
    model.fit(x, y, sample_weight=_balanced_sample_weights(y))
    return model


def predict_hist_gbdt_irregular_scores(
    model: HistGradientBoostingClassifier | dict[str, float | str],
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    if feature_frame.empty:
        return np.array([], dtype=float)
    if isinstance(model, dict):
        probability_positive = float(model.get("probability_positive", 0.0))
        return np.full(feature_frame.shape[0], probability_positive, dtype=float)
    x = build_model_matrix(feature_frame)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def evaluate_screening_quality_gate(
    row: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    cfg = config or {}
    if str(cfg.get("mode", "suppress")) != "suppress":
        raise ValueError(f"Unsupported Stage 4B quality-gate mode: {cfg.get('mode')}")

    if bool(cfg.get("require_selected_hr_valid", True)) and not safe_bool(row.get("selected_hr_is_valid", False)):
        return False, "selected_hr_invalid"

    stage3_flag_column = str(cfg.get("stage3_quality_flag_column", "ml_validity_flag"))
    if bool(cfg.get("require_stage3_quality_pass", True)) and not safe_bool(row.get(stage3_flag_column, False)):
        return False, "stage3_quality_blocked"

    if bool(cfg.get("require_support_sufficient", True)):
        if not safe_bool(row.get("support_sufficient_flag", False)):
            if safe_bool(row.get("insufficient_beats_flag", False)):
                return False, "insufficient_beats"
            if safe_bool(row.get("insufficient_clean_ibi_flag", False)) or not safe_bool(row.get("ibi_is_valid", False)):
                return False, "insufficient_clean_ibi"
            return False, "insufficient_support"

    disallowed_sources = {str(value) for value in cfg.get("disallowed_robust_sources", ["none", "hold_previous"])}
    robust_source = str(row.get("robust_hr_source", ""))
    if robust_source in disallowed_sources:
        return False, f"robust_source_{robust_source}"

    disallowed_actions = {str(value) for value in cfg.get("disallowed_robust_actions", ["hold", "reject"])}
    robust_action = str(row.get("robust_hr_action", ""))
    if robust_action in disallowed_actions:
        return False, f"robust_action_{robust_action}"

    return True, "pass"


def build_screening_predictions(
    feature_frame: pd.DataFrame,
    *,
    model_name: str,
    scores: np.ndarray,
    threshold: float,
    candidate_reasons: list[str],
    candidate_indicator_counts: list[int],
    quality_gate_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    predictions = feature_frame.copy()
    predictions["model_name"] = model_name
    predictions["screening_score"] = np.asarray(scores, dtype=float)
    predictions["irregular_pulse_score"] = np.asarray(scores, dtype=float)
    predictions["screening_threshold"] = float(threshold)
    predictions["candidate_reason_code"] = candidate_reasons
    predictions["candidate_indicator_count"] = np.asarray(candidate_indicator_counts, dtype=float)
    predictions["screening_candidate_flag"] = predictions["screening_score"].to_numpy(dtype=float) >= float(threshold)

    predictions["support_sufficient_flag"] = compute_support_sufficient_flags(predictions)

    quality_passed: list[bool] = []
    quality_reasons: list[str] = []
    validity_flags: list[bool] = []
    emitted_flags: list[bool] = []
    screening_reasons: list[str] = []
    for row in predictions.to_dict(orient="records"):
        quality_pass, quality_reason = evaluate_screening_quality_gate(row, config=quality_gate_config)
        quality_passed.append(bool(quality_pass))
        quality_reasons.append(quality_reason)

        candidate_flag = bool(row["screening_candidate_flag"])
        if quality_pass:
            validity_flags.append(True)
            emitted_flags.append(candidate_flag)
            screening_reasons.append("positive_emitted" if candidate_flag else "negative_valid")
        else:
            validity_flags.append(False)
            emitted_flags.append(False)
            screening_reasons.append("suppressed_positive" if candidate_flag else "invalid_window_suppressed")

    predictions["quality_gate_passed"] = quality_passed
    predictions["quality_gate_reason"] = quality_reasons
    predictions["screening_validity_flag"] = validity_flags
    predictions["irregular_pulse_flag"] = emitted_flags
    predictions["screening_reason_code"] = screening_reasons

    for column_name in PREDICTION_COLUMNS:
        if column_name not in predictions.columns:
            if column_name.endswith("_flag") or column_name.endswith("_target") or column_name.endswith("_passed"):
                predictions[column_name] = False
            elif column_name.endswith("_reason") or column_name.endswith("_code") or column_name.endswith("_source") or column_name == "model_name":
                predictions[column_name] = ""
            else:
                predictions[column_name] = math.nan
    return predictions.loc[:, list(PREDICTION_COLUMNS)]


def summarize_stage4_irregular_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (split_name, model_name), group in predictions.groupby(["split", "model_name"], sort=False):
        eval_group = group.loc[group["proxy_label_support_flag"].astype(bool)].copy()
        y_true = eval_group["screening_proxy_target"].astype(bool).to_numpy(dtype=bool)
        y_pred = eval_group["irregular_pulse_flag"].astype(bool).to_numpy(dtype=bool)
        y_score = eval_group["irregular_pulse_score"].to_numpy(dtype=float)

        tp = int(np.sum(y_true & y_pred))
        fp = int(np.sum(~y_true & y_pred))
        fn = int(np.sum(y_true & ~y_pred))
        tn = int(np.sum(~y_true & ~y_pred))
        metrics = compute_precision_recall_f1(tp, fp, fn)
        accuracy = float((tp + tn) / max(eval_group.shape[0], 1)) if not eval_group.empty else math.nan

        auroc = math.nan
        auprc = math.nan
        if eval_group.shape[0] > 0 and np.unique(y_true.astype(int)).size == 2:
            auroc = float(roc_auc_score(y_true.astype(int), y_score))
            auprc = float(average_precision_score(y_true.astype(int), y_score))

        rows.append(
            {
                "task": "irregular_pulse_screening",
                "method": str(model_name),
                "split": str(split_name),
                "num_eval_windows": float(eval_group.shape[0]),
                "num_positive_targets": float(np.sum(y_true)),
                "num_positive_predictions": float(np.sum(y_pred)),
                "accuracy": accuracy,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "auroc": auroc,
                "auprc": auprc,
                "selected_hr_valid_fraction": float(np.mean(group["selected_hr_is_valid"].astype(bool))) if not group.empty else math.nan,
                "quality_gate_pass_fraction": float(np.mean(group["quality_gate_passed"].astype(bool))) if not group.empty else math.nan,
                "support_sufficient_fraction": float(np.mean(group["support_sufficient_flag"].astype(bool))) if not group.empty else math.nan,
                "suppressed_positive_count": float(
                    np.sum(group["screening_candidate_flag"].astype(bool) & ~group["quality_gate_passed"].astype(bool))
                ),
                "valid_prediction_fraction": float(np.mean(group["screening_validity_flag"].astype(bool))) if not group.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)
