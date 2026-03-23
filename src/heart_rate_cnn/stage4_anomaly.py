from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

from heart_rate_cnn.metrics import compute_precision_recall_f1
from heart_rate_cnn.stage4_irregular import (
    MODEL_FEATURE_COLUMNS,
    compute_support_sufficient_flags,
    evaluate_screening_quality_gate,
)
from heart_rate_cnn.stage4_features import safe_float


DEFAULT_MODEL_NAME = "isolation_forest_anomaly"

PREDICTION_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
    "model_name",
    "proxy_abnormal_target",
    "proxy_abnormal_support_flag",
    "raw_anomaly_score",
    "anomaly_score",
    "anomaly_threshold",
    "anomaly_candidate_flag",
    "anomaly_flag",
    "anomaly_validity_flag",
    "anomaly_reason_code",
    "anomaly_fit_reference_flag",
    "quality_gate_passed",
    "quality_gate_reason",
    "support_sufficient_flag",
    "selected_hr_source",
    "selected_hr_bpm",
    "selected_hr_is_valid",
    "ml_signal_quality_score",
    "robust_hr_source",
    "robust_hr_action",
)


def build_model_matrix(feature_frame: pd.DataFrame) -> np.ndarray:
    if feature_frame.empty:
        return np.empty((0, len(MODEL_FEATURE_COLUMNS)), dtype=float)
    matrix = []
    for row in feature_frame.to_dict(orient="records"):
        matrix.append([safe_float(row.get(column_name), default=math.nan) for column_name in MODEL_FEATURE_COLUMNS])
    return np.asarray(matrix, dtype=float)


def select_anomaly_fit_reference_rows(
    feature_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> pd.Series:
    cfg = config or {}
    require_quality_gate = bool(cfg.get("require_quality_gate_pass", True))
    require_support = bool(cfg.get("require_support_sufficient", True))
    require_proxy_support = bool(cfg.get("require_proxy_support", True))

    mask = feature_frame["split"].astype(str) == "train"
    if require_quality_gate and "quality_gate_passed" in feature_frame.columns:
        mask &= feature_frame["quality_gate_passed"].astype(bool)
    if require_support:
        support_flags = (
            feature_frame["support_sufficient_flag"].astype(bool)
            if "support_sufficient_flag" in feature_frame.columns
            else pd.Series(compute_support_sufficient_flags(feature_frame), index=feature_frame.index)
        )
        mask &= support_flags
    if "proxy_hr_event_target_any" in feature_frame.columns:
        mask &= ~feature_frame["proxy_hr_event_target_any"].astype(bool)
    if "screening_proxy_target" in feature_frame.columns:
        mask &= ~feature_frame["screening_proxy_target"].astype(bool)
    if require_proxy_support and "proxy_abnormal_support_flag" in feature_frame.columns:
        mask &= feature_frame["proxy_abnormal_support_flag"].astype(bool)
    elif require_proxy_support and "proxy_label_support_flag" in feature_frame.columns:
        mask &= feature_frame["proxy_label_support_flag"].astype(bool)
    return mask.astype(bool)


def _fit_reference_fallback_mask(feature_frame: pd.DataFrame) -> pd.Series:
    mask = feature_frame["split"].astype(str) == "train"
    if "quality_gate_passed" in feature_frame.columns:
        mask &= feature_frame["quality_gate_passed"].astype(bool)
    if "support_sufficient_flag" in feature_frame.columns:
        mask &= feature_frame["support_sufficient_flag"].astype(bool)
    elif feature_frame.shape[0] > 0:
        mask &= pd.Series(compute_support_sufficient_flags(feature_frame), index=feature_frame.index)
    return mask.astype(bool)


def fit_isolation_forest_anomaly_model(
    feature_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> tuple[IsolationForest | dict[str, float | str], pd.Series]:
    cfg = config or {}
    reference_mask = select_anomaly_fit_reference_rows(feature_frame, config=cfg)
    if int(np.sum(reference_mask)) < int(cfg.get("min_reference_rows", 16)):
        fallback_mask = _fit_reference_fallback_mask(feature_frame)
        if int(np.sum(fallback_mask)) >= 2:
            reference_mask = fallback_mask

    reference_frame = feature_frame.loc[reference_mask].copy()
    if reference_frame.shape[0] < 2:
        return {"model_type": "constant", "raw_score": 0.0}, reference_mask

    x_ref = build_model_matrix(reference_frame)
    if x_ref.shape[0] < 2:
        return {"model_type": "constant", "raw_score": 0.0}, reference_mask

    model = IsolationForest(
        n_estimators=int(cfg.get("n_estimators", 200)),
        max_samples=cfg.get("max_samples", "auto"),
        max_features=float(cfg.get("max_features", 1.0)),
        bootstrap=bool(cfg.get("bootstrap", False)),
        random_state=int(cfg.get("random_seed", 42)),
        contamination=str(cfg.get("contamination", "auto")),
    )
    model.fit(x_ref)
    return model, reference_mask


def predict_raw_anomaly_scores(
    model: IsolationForest | dict[str, float | str],
    feature_frame: pd.DataFrame,
) -> np.ndarray:
    if feature_frame.empty:
        return np.array([], dtype=float)
    if isinstance(model, dict):
        return np.full(feature_frame.shape[0], float(model.get("raw_score", 0.0)), dtype=float)
    x = build_model_matrix(feature_frame)
    return -np.asarray(model.decision_function(x), dtype=float)


def normalize_anomaly_scores(
    raw_scores: np.ndarray,
    reference_raw_scores: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=float)
    ref = np.asarray(reference_raw_scores, dtype=float)
    if scores.size == 0:
        return np.array([], dtype=float)
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        return np.zeros(scores.size, dtype=float)
    if float(np.max(ref) - np.min(ref)) <= 1e-12:
        return (scores > float(np.max(ref))).astype(float)
    sorted_ref = np.sort(ref)
    ranks = np.searchsorted(sorted_ref, scores, side="right")
    return np.clip(ranks / max(sorted_ref.size, 1), 0.0, 1.0).astype(float)


def build_anomaly_predictions(
    feature_frame: pd.DataFrame,
    *,
    model: IsolationForest | dict[str, float | str],
    fit_reference_mask: pd.Series,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    predictions = feature_frame.copy()
    predictions["model_name"] = str(cfg.get("model_name", DEFAULT_MODEL_NAME))

    if "support_sufficient_flag" not in predictions.columns:
        predictions["support_sufficient_flag"] = compute_support_sufficient_flags(predictions)

    raw_scores = predict_raw_anomaly_scores(model, predictions)
    reference_raw_scores = raw_scores[np.asarray(fit_reference_mask, dtype=bool)] if raw_scores.size else np.array([], dtype=float)
    normalized_scores = normalize_anomaly_scores(raw_scores, reference_raw_scores)
    if reference_raw_scores.size == 0:
        threshold = math.inf
    else:
        threshold = float(np.quantile(reference_raw_scores, float(cfg.get("alert_quantile", 0.95))))

    predictions["raw_anomaly_score"] = raw_scores
    predictions["anomaly_score"] = normalized_scores
    predictions["anomaly_threshold"] = threshold
    predictions["anomaly_candidate_flag"] = raw_scores >= threshold if np.isfinite(threshold) else False
    predictions["anomaly_fit_reference_flag"] = np.asarray(fit_reference_mask, dtype=bool)

    quality_passed: list[bool] = []
    quality_reasons: list[str] = []
    anomaly_flags: list[bool] = []
    validity_flags: list[bool] = []
    reason_codes: list[str] = []
    for row in predictions.to_dict(orient="records"):
        quality_pass, quality_reason = evaluate_screening_quality_gate(row, config=cfg.get("quality_gate", {}))
        quality_passed.append(bool(quality_pass))
        quality_reasons.append(quality_reason)
        candidate_flag = bool(row["anomaly_candidate_flag"])
        if quality_pass:
            validity_flags.append(True)
            anomaly_flags.append(candidate_flag)
            reason_codes.append("positive_emitted" if candidate_flag else "negative_valid")
        else:
            validity_flags.append(False)
            anomaly_flags.append(False)
            reason_codes.append("suppressed_positive" if candidate_flag else "invalid_window_suppressed")

    predictions["quality_gate_passed"] = quality_passed
    predictions["quality_gate_reason"] = quality_reasons
    predictions["anomaly_validity_flag"] = validity_flags
    predictions["anomaly_flag"] = anomaly_flags
    predictions["anomaly_reason_code"] = reason_codes

    for column_name in PREDICTION_COLUMNS:
        if column_name not in predictions.columns:
            if column_name.endswith("_flag") or column_name.endswith("_target") or column_name.endswith("_passed"):
                predictions[column_name] = False
            elif column_name.endswith("_reason") or column_name.endswith("_code") or column_name.endswith("_source") or column_name == "model_name":
                predictions[column_name] = ""
            else:
                predictions[column_name] = math.nan
    return predictions.loc[:, list(PREDICTION_COLUMNS)]


def summarize_stage4_anomaly_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (split_name, model_name), group in predictions.groupby(["split", "model_name"], sort=False):
        if "proxy_abnormal_support_flag" in group.columns:
            eval_group = group.loc[group["proxy_abnormal_support_flag"].astype(bool)].copy()
        else:
            eval_group = group.copy()
        y_true = eval_group["proxy_abnormal_target"].astype(bool).to_numpy(dtype=bool) if not eval_group.empty else np.array([], dtype=bool)
        y_pred = eval_group["anomaly_flag"].astype(bool).to_numpy(dtype=bool) if not eval_group.empty else np.array([], dtype=bool)
        y_score = eval_group["anomaly_score"].to_numpy(dtype=float) if not eval_group.empty else np.array([], dtype=float)

        tp = int(np.sum(y_true & y_pred)) if y_true.size else 0
        fp = int(np.sum(~y_true & y_pred)) if y_true.size else 0
        fn = int(np.sum(y_true & ~y_pred)) if y_true.size else 0
        tn = int(np.sum(~y_true & ~y_pred)) if y_true.size else 0
        metrics = compute_precision_recall_f1(tp, fp, fn)
        accuracy = float((tp + tn) / max(eval_group.shape[0], 1)) if not eval_group.empty else math.nan

        auroc = math.nan
        auprc = math.nan
        if eval_group.shape[0] > 0 and np.unique(y_true.astype(int)).size == 2:
            auroc = float(roc_auc_score(y_true.astype(int), y_score))
            auprc = float(average_precision_score(y_true.astype(int), y_score))

        rows.append(
            {
                "task": "anomaly_scoring",
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
                "suppressed_candidate_count": float(
                    np.sum(group["anomaly_candidate_flag"].astype(bool) & ~group["quality_gate_passed"].astype(bool))
                ),
                "valid_prediction_fraction": float(np.mean(group["anomaly_validity_flag"].astype(bool))) if not group.empty else math.nan,
                "fit_reference_fraction": float(np.mean(group["anomaly_fit_reference_flag"].astype(bool))) if not group.empty else math.nan,
            }
        )
    return pd.DataFrame(rows)
