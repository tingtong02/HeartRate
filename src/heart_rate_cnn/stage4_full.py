from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from heart_rate_cnn.metrics import compute_precision_recall_f1
from heart_rate_cnn.stage4_features import STAGE4_IDENTITY_COLUMNS


FULL_PREDICTION_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
    "selected_hr_source",
    "selected_hr_bpm",
    "selected_hr_is_valid",
    "ml_signal_quality_score",
    "ml_validity_flag",
    "stage3_quality_suspicious_score",
    "stage3_quality_suspicious_flag",
    "quality_gate_passed",
    "quality_gate_reason",
    "hr_event_flag",
    "hr_event_type",
    "hr_event_type_summary",
    "hr_event_severity_score",
    "event_validity_flag",
    "irregular_pulse_flag",
    "irregular_pulse_score",
    "screening_validity_flag",
    "anomaly_score",
    "anomaly_flag",
    "anomaly_validity_flag",
    "stage4_suspicion_flag",
    "stage4_suspicion_score",
    "stage4_suspicion_type_summary",
    "stage4_reason_code",
    "proxy_hr_event_target_any",
    "screening_proxy_target",
    "proxy_abnormal_target",
    "proxy_abnormal_support_flag",
)

FULL_METRIC_COLUMNS: tuple[str, ...] = (
    "task",
    "metric_group",
    "method",
    "subgroup",
    "split",
    "target_name",
    "num_eval_windows",
    "num_positive_targets",
    "num_positive_predictions",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auroc",
    "auprc",
    "alert_rate",
    "quality_gate_pass_fraction",
    "valid_fraction",
    "proxy_abnormal_rate",
)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return bool(value)


def collapse_stage4_event_predictions(event_predictions: pd.DataFrame) -> pd.DataFrame:
    if event_predictions.empty:
        return pd.DataFrame(columns=[*STAGE4_IDENTITY_COLUMNS, "hr_event_flag", "hr_event_type"])

    rows: list[dict[str, float | str | bool]] = []
    ordered = event_predictions.sort_values(by=[*STAGE4_IDENTITY_COLUMNS, "event_type"]).reset_index(drop=True)
    for _, group in ordered.groupby(list(STAGE4_IDENTITY_COLUMNS), sort=False):
        valid_group = group.loc[group["event_validity_flag"].astype(bool)].copy()
        proxy_any = bool(group["proxy_event_target"].astype(bool).any())
        hr_event_flag = not valid_group.empty
        if hr_event_flag:
            valid_group = valid_group.sort_values(by=["event_severity_score", "event_type"], ascending=[False, True])
            primary_type = str(valid_group.iloc[0]["event_type"])
            severity = float(valid_group["event_severity_score"].max())
            summary = "|".join(sorted(set(valid_group["event_type"].astype(str).tolist())))
        else:
            primary_type = ""
            severity = 0.0
            summary = ""

        rows.append(
            {
                "split": str(group["split"].iloc[0]),
                "dataset": str(group["dataset"].iloc[0]),
                "subject_id": str(group["subject_id"].iloc[0]),
                "window_index": int(group["window_index"].iloc[0]),
                "start_time_s": float(group["start_time_s"].iloc[0]),
                "duration_s": float(group["duration_s"].iloc[0]),
                "hr_event_flag": hr_event_flag,
                "hr_event_type": primary_type,
                "hr_event_type_summary": summary,
                "hr_event_severity_score": severity,
                "event_validity_flag": hr_event_flag,
                "proxy_hr_event_target_any": proxy_any,
            }
        )
    return pd.DataFrame(rows)


def _build_type_summary(row: pd.Series) -> str:
    parts: list[str] = []
    if _safe_bool(row.get("hr_event_flag", False)):
        event_summary = str(row.get("hr_event_type_summary", "")).strip()
        parts.append(f"event:{event_summary}" if event_summary else "event")
    if _safe_bool(row.get("irregular_pulse_flag", False)):
        parts.append("irregular")
    if _safe_bool(row.get("anomaly_flag", False)):
        parts.append("anomaly")
    return "|".join(parts)


def build_stage4_full_predictions(
    *,
    feature_frame: pd.DataFrame,
    event_summary: pd.DataFrame,
    irregular_predictions: pd.DataFrame,
    anomaly_predictions: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cfg = config or {}
    suspicion_cfg = cfg.get("suspicion", {})
    full = feature_frame.copy()
    full["stage3_quality_suspicious_score"] = 1.0 - np.clip(full["ml_signal_quality_score"].to_numpy(dtype=float), 0.0, 1.0)
    full["stage3_quality_suspicious_flag"] = ~full["ml_validity_flag"].astype(bool)

    irregular_subset = irregular_predictions.loc[
        :,
        [
            *STAGE4_IDENTITY_COLUMNS,
            "irregular_pulse_score",
            "irregular_pulse_flag",
            "screening_validity_flag",
            "quality_gate_passed",
            "quality_gate_reason",
        ],
    ].copy()
    anomaly_subset = anomaly_predictions.loc[
        :,
        [
            *STAGE4_IDENTITY_COLUMNS,
            "anomaly_score",
            "anomaly_flag",
            "anomaly_validity_flag",
        ],
    ].copy()

    full = full.merge(event_summary, on=list(STAGE4_IDENTITY_COLUMNS), how="left", validate="one_to_one")
    full = full.merge(irregular_subset, on=list(STAGE4_IDENTITY_COLUMNS), how="left", validate="one_to_one")
    full = full.merge(anomaly_subset, on=list(STAGE4_IDENTITY_COLUMNS), how="left", validate="one_to_one")

    if "screening_proxy_target" not in full.columns:
        full["screening_proxy_target"] = False
    if "proxy_label_support_flag" not in full.columns:
        full["proxy_label_support_flag"] = False

    bool_fill_columns = [
        "hr_event_flag",
        "event_validity_flag",
        "proxy_hr_event_target_any",
        "screening_proxy_target",
        "proxy_label_support_flag",
        "irregular_pulse_flag",
        "screening_validity_flag",
        "quality_gate_passed",
        "anomaly_flag",
        "anomaly_validity_flag",
    ]
    for column_name in bool_fill_columns:
        if column_name not in full.columns:
            full[column_name] = False
        full[column_name] = full[column_name].fillna(False).astype(bool)

    for column_name in ["hr_event_type", "hr_event_type_summary", "quality_gate_reason"]:
        if column_name not in full.columns:
            full[column_name] = ""
        full[column_name] = full[column_name].fillna("").astype(str)

    for column_name in ["hr_event_severity_score", "irregular_pulse_score", "anomaly_score"]:
        if column_name not in full.columns:
            full[column_name] = 0.0
        full[column_name] = full[column_name].fillna(0.0).astype(float)

    full["proxy_hr_event_support_flag"] = full["window_is_valid"].astype(bool) & full["ref_hr_bpm"].notna()
    full["proxy_abnormal_target"] = full["proxy_hr_event_target_any"].astype(bool) | full["screening_proxy_target"].astype(bool)
    full["proxy_abnormal_support_flag"] = full["proxy_hr_event_support_flag"].astype(bool) & full["proxy_label_support_flag"].astype(bool)

    event_component = np.where(
        full["hr_event_flag"].astype(bool),
        np.maximum(float(suspicion_cfg.get("event_min_score", 0.60)), np.clip(full["hr_event_severity_score"].to_numpy(dtype=float), 0.0, 1.0)),
        0.0,
    )
    irregular_component = np.where(
        full["screening_validity_flag"].astype(bool),
        np.clip(full["irregular_pulse_score"].to_numpy(dtype=float), 0.0, 1.0),
        0.0,
    )
    anomaly_component = np.where(
        full["anomaly_validity_flag"].astype(bool),
        np.clip(full["anomaly_score"].to_numpy(dtype=float), 0.0, 1.0),
        0.0,
    )
    base_score = np.maximum.reduce([event_component, irregular_component, anomaly_component])
    active_count = (
        full["hr_event_flag"].astype(int)
        + full["irregular_pulse_flag"].astype(int)
        + full["anomaly_flag"].astype(int)
    )
    bonus = np.where(
        active_count >= 3,
        float(suspicion_cfg.get("three_signal_bonus", 0.20)),
        np.where(active_count == 2, float(suspicion_cfg.get("two_signal_bonus", 0.10)), 0.0),
    )
    full["stage4_suspicion_score"] = np.clip(base_score + bonus, 0.0, 1.0)
    full["stage4_suspicion_flag"] = (
        full["hr_event_flag"].astype(bool)
        | full["irregular_pulse_flag"].astype(bool)
        | full["anomaly_flag"].astype(bool)
    )
    full["stage4_suspicion_type_summary"] = [
        _build_type_summary(row) for _, row in full.iterrows()
    ]

    reason_codes: list[str] = []
    for _, row in full.iterrows():
        if _safe_bool(row["hr_event_flag"]) and (_safe_bool(row["irregular_pulse_flag"]) or _safe_bool(row["anomaly_flag"])):
            reason_codes.append("multi_signal_suspicion")
        elif _safe_bool(row["irregular_pulse_flag"]) and _safe_bool(row["anomaly_flag"]):
            reason_codes.append("multi_signal_suspicion")
        elif _safe_bool(row["hr_event_flag"]):
            reason_codes.append("hr_event_suspicion")
        elif _safe_bool(row["irregular_pulse_flag"]):
            reason_codes.append("irregular_pulse_suspicion")
        elif _safe_bool(row["anomaly_flag"]):
            reason_codes.append("anomaly_suspicion")
        elif not _safe_bool(row["quality_gate_passed"]):
            reason_codes.append("low_quality_suppressed")
        else:
            reason_codes.append("no_stage4_signal")
    full["stage4_reason_code"] = reason_codes

    for column_name in FULL_PREDICTION_COLUMNS:
        if column_name not in full.columns:
            if column_name.endswith("_flag") or column_name.endswith("_target") or column_name.endswith("_passed"):
                full[column_name] = False
            elif column_name.endswith("_reason") or column_name.endswith("_summary") or column_name.endswith("_type") or column_name.endswith("_source"):
                full[column_name] = ""
            else:
                full[column_name] = math.nan

    full = full.sort_values(by=list(STAGE4_IDENTITY_COLUMNS)).reset_index(drop=True)
    return full.loc[:, list(FULL_PREDICTION_COLUMNS)]


def _binary_metric_row(
    frame: pd.DataFrame,
    *,
    metric_group: str,
    method: str,
    split_name: str,
    target_name: str,
    target_col: str,
    pred_col: str,
    score_col: str,
    mask_col: str,
    subgroup: str = "",
    valid_col: str | None = None,
) -> dict[str, float | str]:
    eval_group = frame.loc[frame[mask_col].astype(bool)].copy() if mask_col in frame.columns else frame.copy()
    if eval_group.empty:
        return {
            "task": "stage4_full",
            "metric_group": metric_group,
            "method": method,
            "subgroup": subgroup,
            "split": split_name,
            "target_name": target_name,
            "num_eval_windows": 0.0,
            "num_positive_targets": 0.0,
            "num_positive_predictions": 0.0,
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "f1": math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
            "alert_rate": math.nan,
            "quality_gate_pass_fraction": math.nan,
            "valid_fraction": math.nan,
            "proxy_abnormal_rate": math.nan,
        }

    y_true = eval_group[target_col].astype(bool).to_numpy(dtype=bool)
    y_pred = eval_group[pred_col].astype(bool).to_numpy(dtype=bool)
    y_score = eval_group[score_col].to_numpy(dtype=float)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    metrics = compute_precision_recall_f1(tp, fp, fn)
    accuracy = float((tp + tn) / max(eval_group.shape[0], 1))

    auroc = math.nan
    auprc = math.nan
    if eval_group.shape[0] > 0 and np.unique(y_true.astype(int)).size == 2:
        auroc = float(roc_auc_score(y_true.astype(int), y_score))
        auprc = float(average_precision_score(y_true.astype(int), y_score))

    return {
        "task": "stage4_full",
        "metric_group": metric_group,
        "method": method,
        "subgroup": subgroup,
        "split": split_name,
        "target_name": target_name,
        "num_eval_windows": float(eval_group.shape[0]),
        "num_positive_targets": float(np.sum(y_true)),
        "num_positive_predictions": float(np.sum(y_pred)),
        "accuracy": accuracy,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "auroc": auroc,
        "auprc": auprc,
        "alert_rate": float(np.mean(y_pred.astype(float))) if y_pred.size else math.nan,
        "quality_gate_pass_fraction": float(np.mean(eval_group["quality_gate_passed"].astype(bool))),
        "valid_fraction": float(np.mean(eval_group[valid_col].astype(bool))) if valid_col is not None else math.nan,
        "proxy_abnormal_rate": float(np.mean(eval_group["proxy_abnormal_target"].astype(bool))) if "proxy_abnormal_target" in eval_group.columns else math.nan,
    }


def _adapt_existing_metric_row(
    row: pd.Series,
    *,
    metric_group: str,
    subgroup: str,
    target_name: str,
    positive_target_col: str,
    positive_pred_col: str,
    valid_fraction_col: str,
) -> dict[str, float | str]:
    num_eval_windows = float(row.get("num_eval_windows", math.nan))
    num_positive_targets = float(row.get(positive_target_col, math.nan))
    num_positive_predictions = float(row.get(positive_pred_col, math.nan))
    alert_rate = num_positive_predictions / num_eval_windows if np.isfinite(num_eval_windows) and num_eval_windows > 0 else math.nan
    return {
        "task": "stage4_full",
        "metric_group": metric_group,
        "method": str(row.get("method", "")),
        "subgroup": subgroup,
        "split": str(row.get("split", "")),
        "target_name": target_name,
        "num_eval_windows": num_eval_windows,
        "num_positive_targets": num_positive_targets,
        "num_positive_predictions": num_positive_predictions,
        "accuracy": _safe_float(row.get("accuracy")),
        "precision": _safe_float(row.get("precision")),
        "recall": _safe_float(row.get("recall")),
        "f1": _safe_float(row.get("f1")),
        "auroc": _safe_float(row.get("auroc")),
        "auprc": _safe_float(row.get("auprc")),
        "alert_rate": alert_rate,
        "quality_gate_pass_fraction": _safe_float(row.get("quality_gate_pass_fraction")),
        "valid_fraction": _safe_float(row.get(valid_fraction_col)),
        "proxy_abnormal_rate": math.nan,
    }


def summarize_stage4_full_metrics(
    *,
    full_predictions: pd.DataFrame,
    event_metrics: pd.DataFrame,
    irregular_metrics: pd.DataFrame,
    anomaly_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    for _, row in event_metrics.iterrows():
        rows.append(
            _adapt_existing_metric_row(
                row,
                metric_group="event",
                subgroup=str(row.get("event_type", "")),
                target_name="proxy_hr_event",
                positive_target_col="num_eval_events",
                positive_pred_col="num_pred_events",
                valid_fraction_col="valid_event_fraction",
            )
        )

    for _, row in irregular_metrics.iterrows():
        rows.append(
            _adapt_existing_metric_row(
                row,
                metric_group="irregular",
                subgroup="",
                target_name="screening_proxy_target",
                positive_target_col="num_positive_targets",
                positive_pred_col="num_positive_predictions",
                valid_fraction_col="valid_prediction_fraction",
            )
        )

    for _, row in anomaly_metrics.iterrows():
        rows.append(
            _adapt_existing_metric_row(
                row,
                metric_group="anomaly",
                subgroup="",
                target_name="proxy_abnormal_union",
                positive_target_col="num_positive_targets",
                positive_pred_col="num_positive_predictions",
                valid_fraction_col="valid_prediction_fraction",
            )
        )

    for split_name, split_group in full_predictions.groupby("split", sort=False):
        rows.append(
            _binary_metric_row(
                split_group,
                metric_group="unified",
                method="stage4_full_default",
                subgroup="",
                split_name=str(split_name),
                target_name="proxy_abnormal_union",
                target_col="proxy_abnormal_target",
                pred_col="stage4_suspicion_flag",
                score_col="stage4_suspicion_score",
                mask_col="proxy_abnormal_support_flag",
                valid_col="quality_gate_passed",
            )
        )
        rows.append(
            _binary_metric_row(
                split_group,
                metric_group="stage3_comparison",
                method="stage3_quality_only",
                subgroup="score_and_flag",
                split_name=str(split_name),
                target_name="proxy_abnormal_union",
                target_col="proxy_abnormal_target",
                pred_col="stage3_quality_suspicious_flag",
                score_col="stage3_quality_suspicious_score",
                mask_col="proxy_abnormal_support_flag",
                valid_col="ml_validity_flag",
            )
        )
        rows.append(
            _binary_metric_row(
                split_group,
                metric_group="stage3_comparison",
                method="stage4_irregular_default",
                subgroup="score_and_flag",
                split_name=str(split_name),
                target_name="proxy_abnormal_union",
                target_col="proxy_abnormal_target",
                pred_col="irregular_pulse_flag",
                score_col="irregular_pulse_score",
                mask_col="proxy_abnormal_support_flag",
                valid_col="screening_validity_flag",
            )
        )
        rows.append(
            _binary_metric_row(
                split_group,
                metric_group="stage3_comparison",
                method="stage4_anomaly_default",
                subgroup="score_and_flag",
                split_name=str(split_name),
                target_name="proxy_abnormal_union",
                target_col="proxy_abnormal_target",
                pred_col="anomaly_flag",
                score_col="anomaly_score",
                mask_col="proxy_abnormal_support_flag",
                valid_col="anomaly_validity_flag",
            )
        )
        rows.append(
            _binary_metric_row(
                split_group,
                metric_group="stage3_comparison",
                method="stage4_full_default",
                subgroup="score_and_flag",
                split_name=str(split_name),
                target_name="proxy_abnormal_union",
                target_col="proxy_abnormal_target",
                pred_col="stage4_suspicion_flag",
                score_col="stage4_suspicion_score",
                mask_col="proxy_abnormal_support_flag",
                valid_col="quality_gate_passed",
            )
        )

        stage3_valid_group = split_group.loc[split_group["ml_validity_flag"].astype(bool)].copy()
        if not stage3_valid_group.empty:
            categories = []
            for _, row in stage3_valid_group.iterrows():
                if not _safe_bool(row["quality_gate_passed"]):
                    categories.append("low_quality_suppressed")
                else:
                    active = int(_safe_bool(row["hr_event_flag"])) + int(_safe_bool(row["irregular_pulse_flag"])) + int(_safe_bool(row["anomaly_flag"]))
                    if active >= 2:
                        categories.append("multi_flag_valid")
                    elif _safe_bool(row["hr_event_flag"]):
                        categories.append("event_like_valid")
                    elif _safe_bool(row["irregular_pulse_flag"]):
                        categories.append("irregular_valid")
                    elif _safe_bool(row["anomaly_flag"]):
                        categories.append("anomaly_only_valid")
                    else:
                        categories.append("normal_valid")
            stage3_valid_group["stage4_category"] = categories
            for category_name, category_group in stage3_valid_group.groupby("stage4_category", sort=False):
                rows.append(
                    {
                        "task": "stage4_full",
                        "metric_group": "stratification",
                        "method": "stage4_stratification",
                        "subgroup": str(category_name),
                        "split": str(split_name),
                        "target_name": "proxy_abnormal_union",
                        "num_eval_windows": float(category_group.shape[0]),
                        "num_positive_targets": float(np.sum(category_group["proxy_abnormal_target"].astype(bool))),
                        "num_positive_predictions": float(np.sum(category_group["stage4_suspicion_flag"].astype(bool))),
                        "accuracy": math.nan,
                        "precision": math.nan,
                        "recall": math.nan,
                        "f1": math.nan,
                        "auroc": math.nan,
                        "auprc": math.nan,
                        "alert_rate": float(np.mean(category_group["stage4_suspicion_flag"].astype(bool))),
                        "quality_gate_pass_fraction": float(np.mean(category_group["quality_gate_passed"].astype(bool))),
                        "valid_fraction": float(np.mean(category_group["quality_gate_passed"].astype(bool))),
                        "proxy_abnormal_rate": float(np.mean(category_group["proxy_abnormal_target"].astype(bool))),
                    }
                )

    metrics = pd.DataFrame(rows)
    for column_name in FULL_METRIC_COLUMNS:
        if column_name not in metrics.columns:
            if column_name in {"task", "metric_group", "method", "subgroup", "split", "target_name"}:
                metrics[column_name] = ""
            else:
                metrics[column_name] = math.nan
    return metrics.loc[:, list(FULL_METRIC_COLUMNS)]
