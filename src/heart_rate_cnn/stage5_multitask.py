from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from heart_rate_cnn.metrics import compute_hr_metrics


STAGE5_IDENTITY_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    *STAGE5_IDENTITY_COLUMNS,
    "resp_rate_ref_bpm",
    "resp_rate_ref_valid_flag",
    "resp_reference_reason",
    "resp_rate_baseline_bpm",
    "resp_rate_pred_bpm",
    "resp_confidence",
    "resp_validity_flag",
    "selected_hr_bpm",
    "selected_hr_source",
    "selected_hr_is_valid",
    "ml_signal_quality_score",
    "motion_flag",
    "validity_flag",
    "hr_event_flag",
    "irregular_pulse_flag",
    "anomaly_score",
    "stage4_suspicion_flag",
    "stage4_suspicion_score",
    "stage4_suspicion_type_summary",
    "num_beats",
    "num_ibi_clean",
    "mean_ibi_ms",
    "rmssd_ms",
    "sdnn_ms",
    "pnn50",
    "beat_positions_s",
    "ibi_series_ms",
)

METRIC_COLUMNS: tuple[str, ...] = (
    "task",
    "method",
    "split",
    "subset",
    "num_eval_windows",
    "resp_mae_bpm",
    "resp_rmse_bpm",
    "resp_pearson_r",
    "within_3_bpm_rate",
    "resp_valid_auroc",
    "resp_valid_auprc",
    "hr_selected_hr_match_rate",
    "stage4_suspicion_match_rate",
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


def _serialize_float_list(values: list[float] | np.ndarray) -> str:
    array = np.asarray(values, dtype=float).reshape(-1)
    return json.dumps([float(value) for value in array.tolist()])


def aggregate_stage4_context_to_windows(
    stage4_frame: pd.DataFrame,
    stage5_identity_frame: pd.DataFrame,
) -> pd.DataFrame:
    if stage5_identity_frame.empty:
        return pd.DataFrame(
            columns=[
                *STAGE5_IDENTITY_COLUMNS,
                "selected_hr_bpm",
                "selected_hr_source",
                "selected_hr_is_valid",
                "ml_signal_quality_score",
                "motion_flag",
                "validity_flag",
                "hr_event_flag",
                "irregular_pulse_flag",
                "anomaly_score",
                "stage4_suspicion_flag",
                "stage4_suspicion_score",
                "stage4_suspicion_type_summary",
            ]
        )

    if stage4_frame.empty:
        aggregated = stage5_identity_frame.loc[:, list(STAGE5_IDENTITY_COLUMNS)].copy()
        aggregated["selected_hr_bpm"] = math.nan
        aggregated["selected_hr_source"] = ""
        aggregated["selected_hr_is_valid"] = False
        aggregated["ml_signal_quality_score"] = math.nan
        aggregated["motion_flag"] = False
        aggregated["validity_flag"] = False
        aggregated["hr_event_flag"] = False
        aggregated["irregular_pulse_flag"] = False
        aggregated["anomaly_score"] = math.nan
        aggregated["stage4_suspicion_flag"] = False
        aggregated["stage4_suspicion_score"] = math.nan
        aggregated["stage4_suspicion_type_summary"] = ""
        return aggregated

    rows: list[dict[str, Any]] = []
    grouped_stage4 = {
        (str(split_name), str(subject_id)): group.sort_values(by=["start_time_s", "window_index"]).reset_index(drop=True)
        for (split_name, subject_id), group in stage4_frame.groupby(["split", "subject_id"], sort=False)
    }

    for row in stage5_identity_frame.itertuples(index=False):
        key = (str(row.split), str(row.subject_id))
        subject_stage4 = grouped_stage4.get(key)
        start_time_s = float(row.start_time_s)
        end_time_s = start_time_s + float(row.duration_s)

        aggregated_row: dict[str, Any] = {
            "split": str(row.split),
            "dataset": str(row.dataset),
            "subject_id": str(row.subject_id),
            "window_index": int(row.window_index),
            "start_time_s": start_time_s,
            "duration_s": float(row.duration_s),
        }
        if subject_stage4 is None or subject_stage4.empty:
            aggregated_row.update(
                {
                    "selected_hr_bpm": math.nan,
                    "selected_hr_source": "",
                    "selected_hr_is_valid": False,
                    "ml_signal_quality_score": math.nan,
                    "motion_flag": False,
                    "validity_flag": False,
                    "hr_event_flag": False,
                    "irregular_pulse_flag": False,
                    "anomaly_score": math.nan,
                    "stage4_suspicion_flag": False,
                    "stage4_suspicion_score": math.nan,
                    "stage4_suspicion_type_summary": "",
                }
            )
            rows.append(aggregated_row)
            continue

        stage4_starts = subject_stage4["start_time_s"].to_numpy(dtype=float)
        stage4_duration_s = subject_stage4["duration_s"].to_numpy(dtype=float)
        stage4_ends = stage4_starts + stage4_duration_s
        max_stage4_duration = float(np.nanmax(stage4_duration_s)) if stage4_duration_s.size else 0.0
        lower_index = int(np.searchsorted(stage4_starts, start_time_s - max_stage4_duration, side="left"))
        upper_index = int(np.searchsorted(stage4_starts, end_time_s, side="left"))
        candidate = subject_stage4.iloc[lower_index:upper_index].copy()
        if not candidate.empty:
            candidate_starts = candidate["start_time_s"].to_numpy(dtype=float)
            candidate_ends = candidate_starts + candidate["duration_s"].to_numpy(dtype=float)
            overlap_mask = (candidate_starts < end_time_s) & (candidate_ends > start_time_s)
            candidate = candidate.loc[overlap_mask].copy()

        if candidate.empty:
            aggregated_row.update(
                {
                    "selected_hr_bpm": math.nan,
                    "selected_hr_source": "",
                    "selected_hr_is_valid": False,
                    "ml_signal_quality_score": math.nan,
                    "motion_flag": False,
                    "validity_flag": False,
                    "hr_event_flag": False,
                    "irregular_pulse_flag": False,
                    "anomaly_score": math.nan,
                    "stage4_suspicion_flag": False,
                    "stage4_suspicion_score": math.nan,
                    "stage4_suspicion_type_summary": "",
                }
            )
            rows.append(aggregated_row)
            continue

        valid_selected = candidate.loc[candidate["selected_hr_is_valid"].astype(bool) & candidate["selected_hr_bpm"].notna()]
        selected_hr_bpm = float(valid_selected["selected_hr_bpm"].mean()) if not valid_selected.empty else math.nan
        if not candidate["selected_hr_source"].empty:
            selected_hr_source = str(candidate["selected_hr_source"].mode(dropna=True).iloc[0]) if not candidate["selected_hr_source"].mode(dropna=True).empty else str(candidate["selected_hr_source"].iloc[0])
        else:
            selected_hr_source = ""
        suspicion_summaries = sorted(
            summary
            for summary in set(candidate["stage4_suspicion_type_summary"].fillna("").astype(str).tolist())
            if summary
        )
        aggregated_row.update(
            {
                "selected_hr_bpm": selected_hr_bpm,
                "selected_hr_source": selected_hr_source,
                "selected_hr_is_valid": bool(valid_selected.shape[0] > 0),
                "ml_signal_quality_score": float(candidate["ml_signal_quality_score"].mean()) if "ml_signal_quality_score" in candidate.columns else math.nan,
                "motion_flag": bool(candidate["motion_flag"].astype(bool).any()) if "motion_flag" in candidate.columns else False,
                "validity_flag": bool(candidate["quality_gate_passed"].astype(bool).mean() >= 0.5),
                "hr_event_flag": bool(candidate["hr_event_flag"].astype(bool).any()),
                "irregular_pulse_flag": bool(candidate["irregular_pulse_flag"].astype(bool).any()),
                "anomaly_score": float(candidate["anomaly_score"].max()) if "anomaly_score" in candidate.columns else math.nan,
                "stage4_suspicion_flag": bool(candidate["stage4_suspicion_flag"].astype(bool).any()),
                "stage4_suspicion_score": float(candidate["stage4_suspicion_score"].max()) if "stage4_suspicion_score" in candidate.columns else math.nan,
                "stage4_suspicion_type_summary": "|".join(suspicion_summaries),
            }
        )
        rows.append(aggregated_row)

    return pd.DataFrame(rows)


def build_stage5_multitask_predictions(
    frame: pd.DataFrame,
    *,
    resp_validity_threshold: float,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(PREDICTION_COLUMNS))

    predictions = frame.copy().reset_index(drop=True)
    predictions["resp_confidence"] = predictions["resp_confidence"].fillna(0.0).clip(0.0, 1.0)
    predictions["resp_validity_flag"] = (
        predictions["resp_confidence"].to_numpy(dtype=float) >= float(resp_validity_threshold)
    ) & predictions["validity_flag"].astype(bool).to_numpy()

    if "beat_positions_s" in predictions.columns:
        predictions["beat_positions_s"] = [
            value if isinstance(value, str) else _serialize_float_list(value if isinstance(value, (list, np.ndarray)) else [])
            for value in predictions["beat_positions_s"].tolist()
        ]
    else:
        predictions["beat_positions_s"] = "[]"

    if "ibi_series_ms" in predictions.columns:
        predictions["ibi_series_ms"] = [
            value if isinstance(value, str) else _serialize_float_list(value if isinstance(value, (list, np.ndarray)) else [])
            for value in predictions["ibi_series_ms"].tolist()
        ]
    else:
        predictions["ibi_series_ms"] = "[]"

    for column_name in PREDICTION_COLUMNS:
        if column_name not in predictions.columns:
            if column_name.endswith("_flag"):
                predictions[column_name] = False
            elif column_name.endswith("_score") or column_name.endswith("_bpm") or column_name in {"mean_ibi_ms", "rmssd_ms", "sdnn_ms", "pnn50"}:
                predictions[column_name] = math.nan
            else:
                predictions[column_name] = ""
    return predictions.loc[:, list(PREDICTION_COLUMNS)].copy()


def summarize_stage5_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=list(METRIC_COLUMNS))

    rows: list[dict[str, Any]] = []
    subsets = {
        "all_ref_valid": predictions["resp_rate_ref_valid_flag"].astype(bool),
        "high_quality_ref_valid": predictions["resp_rate_ref_valid_flag"].astype(bool) & predictions["validity_flag"].astype(bool),
        "predicted_valid": predictions["resp_rate_ref_valid_flag"].astype(bool) & predictions["resp_validity_flag"].astype(bool),
    }

    for split_name, split_frame in predictions.groupby("split", sort=False):
        valid_target = split_frame["resp_rate_ref_valid_flag"].astype(bool).to_numpy()
        confidence = split_frame["resp_confidence"].to_numpy(dtype=float)
        if valid_target.size >= 2 and np.unique(valid_target.astype(int)).size >= 2:
            resp_valid_auroc = float(roc_auc_score(valid_target.astype(int), confidence))
            resp_valid_auprc = float(average_precision_score(valid_target.astype(int), confidence))
        else:
            resp_valid_auroc = math.nan
            resp_valid_auprc = math.nan

        for method_name, pred_col in (
            ("resp_surrogate_fusion_baseline", "resp_rate_baseline_bpm"),
            ("stage5_resp_multitask_cnn_v1", "resp_rate_pred_bpm"),
        ):
            for subset_name, subset_mask in subsets.items():
                subset_frame = split_frame.loc[subset_mask & split_frame[pred_col].notna()].copy()
                ref_values = subset_frame["resp_rate_ref_bpm"].to_numpy(dtype=float)
                pred_values = subset_frame[pred_col].to_numpy(dtype=float)
                metrics = compute_hr_metrics(ref_values, pred_values)
                within_3_bpm_rate = float(np.mean(np.abs(pred_values - ref_values) <= 3.0)) if ref_values.size > 0 else math.nan
                rows.append(
                    {
                        "task": "stage5_respiration",
                        "method": method_name,
                        "split": str(split_name),
                        "subset": subset_name,
                        "num_eval_windows": float(subset_frame.shape[0]),
                        "resp_mae_bpm": metrics["mae"],
                        "resp_rmse_bpm": metrics["rmse"],
                        "resp_pearson_r": metrics["pearson_r"],
                        "within_3_bpm_rate": within_3_bpm_rate,
                        "resp_valid_auroc": resp_valid_auroc if method_name == "stage5_resp_multitask_cnn_v1" else math.nan,
                        "resp_valid_auprc": resp_valid_auprc if method_name == "stage5_resp_multitask_cnn_v1" else math.nan,
                        "hr_selected_hr_match_rate": 1.0,
                        "stage4_suspicion_match_rate": 1.0,
                    }
                )

    return pd.DataFrame(rows, columns=list(METRIC_COLUMNS))
