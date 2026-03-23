from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from heart_rate_cnn.metrics import compute_precision_recall_f1


EVENT_TYPES: tuple[str, ...] = (
    "tachycardia_event",
    "bradycardia_event",
    "abrupt_change_event",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
)

PREDICTION_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
    "event_type",
    "selected_hr_source",
    "selected_hr_bpm",
    "selected_hr_is_valid",
    "selected_hr_prev_bpm",
    "selected_hr_delta_bpm",
    "quality_gate_passed",
    "quality_gate_reason",
    "proxy_event_target",
    "event_candidate_flag",
    "event_trigger_rule",
    "event_threshold_bpm",
    "event_anchor_hr_bpm",
    "event_severity_score",
    "event_validity_flag",
    "event_reason_code",
    "episode_id",
    "episode_start_flag",
    "episode_end_flag",
    "proxy_episode_id",
    "proxy_episode_start_flag",
    "proxy_episode_end_flag",
)

_SOURCE_SPECS: dict[str, dict[str, str]] = {
    "gated_stage3_ml_logreg": {
        "hr_col": "ungated_pred_hr_bpm",
        "valid_col": "ml_gated_is_valid",
        "invalid_reason_code": "stage3_ml_gate_blocked",
    },
    "gated_stage3_rule": {
        "hr_col": "ungated_pred_hr_bpm",
        "valid_col": "rule_gated_is_valid",
        "invalid_reason_code": "stage3_rule_gate_blocked",
    },
    "gated_stage3_motion_refined": {
        "hr_col": "ungated_pred_hr_bpm",
        "valid_col": "motion_refined_gated_is_valid",
        "invalid_reason_code": "stage3_motion_gate_blocked",
    },
    "robust_stage3c2_policy": {
        "hr_col": "robust_hr_bpm",
        "valid_col": "robust_hr_is_valid",
        "invalid_reason_code": "robust_hr_unavailable",
        "reason_col": "policy_reason_code",
    },
    "ungated_stage1_frequency": {
        "hr_col": "ungated_pred_hr_bpm",
        "valid_col": "ungated_is_valid",
        "invalid_reason_code": "ungated_hr_invalid",
    },
}


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if np.isfinite(numeric) else math.nan


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return bool(value)


def _event_prefix(event_type: str) -> str:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unsupported event_type: {event_type}")
    return event_type.replace("_event", "")


def _threshold_for_event(event_type: str, config: dict[str, Any]) -> float:
    thresholds_cfg = config.get("thresholds", {})
    if event_type == "tachycardia_event":
        return float(thresholds_cfg.get("tachy_hr_bpm", 100.0))
    if event_type == "bradycardia_event":
        return float(thresholds_cfg.get("brady_hr_bpm", 50.0))
    return float(thresholds_cfg.get("abrupt_delta_hr_bpm", 20.0))


def _min_windows_for_event(event_type: str, config: dict[str, Any]) -> int:
    persistence_cfg = config.get("persistence", {})
    if event_type == "tachycardia_event":
        return int(persistence_cfg.get("tachy_min_valid_windows", 2))
    if event_type == "bradycardia_event":
        return int(persistence_cfg.get("brady_min_valid_windows", 2))
    return int(persistence_cfg.get("abrupt_min_valid_windows", 2))


def _default_quality_reason(
    *,
    row: pd.Series,
    invalid_reason_code: str,
    reason_col: str | None = None,
) -> str:
    if _safe_bool(row.get("selected_hr_is_valid", False)):
        return "pass"
    selected_hr = _safe_float(row.get("selected_hr_bpm"))
    if not np.isfinite(selected_hr):
        if reason_col is not None:
            reason_value = row.get(reason_col)
            if isinstance(reason_value, str) and reason_value:
                return reason_value
        return "selected_hr_unavailable"
    return invalid_reason_code


def select_stage4_hr_source(
    frame: pd.DataFrame,
    *,
    event_type: str,
    source_config: dict[str, Any] | None = None,
    event_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    source_cfg = source_config or {}
    cfg = event_config or {}
    source_name = str(source_cfg.get(event_type, source_cfg.get("default", "gated_stage3_ml_logreg")))
    if source_name not in _SOURCE_SPECS:
        raise ValueError(f"Unsupported Stage 4 source: {source_name}")

    spec = _SOURCE_SPECS[source_name]
    missing_columns = [
        column_name
        for column_name in (spec["hr_col"], spec["valid_col"], spec.get("reason_col"))
        if column_name is not None and column_name not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required source columns for {source_name}: {missing_columns}")

    selected = pd.DataFrame(index=frame.index)
    for column_name in IDENTITY_COLUMNS:
        if column_name in frame.columns:
            selected[column_name] = frame[column_name]
        elif column_name == "split":
            selected[column_name] = ""
        else:
            raise ValueError(f"Missing identity column for Stage 4 selection: {column_name}")
    selected["event_type"] = event_type
    selected["selected_hr_source"] = source_name
    selected["selected_hr_bpm"] = frame[spec["hr_col"]].to_numpy(dtype=float)
    selected["selected_hr_is_valid"] = frame[spec["valid_col"]].astype(bool).to_numpy()
    selected["selected_hr_prev_bpm"] = math.nan
    selected["selected_hr_delta_bpm"] = math.nan
    selected["event_anchor_hr_bpm"] = math.nan
    selected["event_trigger_rule"] = ""
    selected["event_threshold_bpm"] = float(_threshold_for_event(event_type, cfg))
    selected["quality_gate_reason"] = [
        _default_quality_reason(
            row=row,
            invalid_reason_code=str(spec.get("invalid_reason_code", "selected_hr_invalid")),
            reason_col=spec.get("reason_col"),
        )
        for _, row in selected.join(frame[[spec.get("reason_col")]] if spec.get("reason_col") else pd.DataFrame(index=frame.index)).iterrows()
    ]
    if "ref_hr_bpm" in frame.columns:
        selected["ref_hr_bpm"] = frame["ref_hr_bpm"].to_numpy(dtype=float)
    if "window_is_valid" in frame.columns:
        selected["window_is_valid"] = frame["window_is_valid"].astype(bool).to_numpy()
    return selected


def _attach_previous_valid_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ordered = frame.sort_values(by=["subject_id", "window_index", "start_time_s"]).reset_index(drop=True)
    previous_values: list[float] = []
    previous_deltas: list[float] = []
    previous_subject = None
    previous_valid_hr = math.nan

    for row in ordered.itertuples(index=False):
        current_subject = str(row.subject_id)
        if previous_subject != current_subject:
            previous_valid_hr = math.nan
        previous_values.append(previous_valid_hr if np.isfinite(previous_valid_hr) else math.nan)

        current_hr = _safe_float(row.selected_hr_bpm)
        if np.isfinite(previous_valid_hr) and np.isfinite(current_hr):
            previous_deltas.append(abs(current_hr - previous_valid_hr))
        else:
            previous_deltas.append(math.nan)

        if _safe_bool(row.selected_hr_is_valid) and np.isfinite(current_hr):
            previous_valid_hr = current_hr
        previous_subject = current_subject

    ordered["selected_hr_prev_bpm"] = previous_values
    ordered["selected_hr_delta_bpm"] = previous_deltas
    return ordered


def _apply_abrupt_confirmation_windows(frame: pd.DataFrame, *, threshold: float, confirmation_ratio: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    confirmed = frame.copy()
    min_confirmation_delta = max(threshold * confirmation_ratio, 0.0)
    for subject_id, subject_frame in confirmed.groupby("subject_id", sort=False):
        indices = list(subject_frame.index)
        for left_index, right_index in zip(indices[:-1], indices[1:]):
            left_row = confirmed.loc[left_index]
            right_row = confirmed.loc[right_index]
            if not _safe_bool(left_row["event_candidate_flag"]):
                continue
            if str(left_row["event_trigger_rule"]) != "abrupt_delta":
                continue
            if _safe_bool(right_row["event_candidate_flag"]):
                continue
            if not _safe_bool(right_row["selected_hr_is_valid"]):
                continue
            anchor_hr = _safe_float(left_row["event_anchor_hr_bpm"])
            left_hr = _safe_float(left_row["selected_hr_bpm"])
            right_hr = _safe_float(right_row["selected_hr_bpm"])
            if not (np.isfinite(anchor_hr) and np.isfinite(left_hr) and np.isfinite(right_hr)):
                continue
            left_direction = np.sign(left_hr - anchor_hr)
            right_direction = np.sign(right_hr - anchor_hr)
            if left_direction == 0.0 or right_direction != left_direction:
                continue
            right_delta = abs(right_hr - anchor_hr)
            if right_delta < min_confirmation_delta:
                continue

            confirmed.at[right_index, "event_candidate_flag"] = True
            confirmed.at[right_index, "selected_hr_prev_bpm"] = anchor_hr
            confirmed.at[right_index, "selected_hr_delta_bpm"] = right_delta
            confirmed.at[right_index, "event_anchor_hr_bpm"] = anchor_hr
            confirmed.at[right_index, "event_trigger_rule"] = "abrupt_change_confirmation"
            confirmed.at[right_index, "event_severity_score"] = _safe_float(
                (right_delta - threshold) / max(threshold, 1.0)
            )
    return confirmed


def detect_window_event_candidates(
    frame: pd.DataFrame,
    *,
    event_type: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    event_cfg = config or {}
    threshold = _threshold_for_event(event_type, event_cfg)
    detected = _attach_previous_valid_context(frame)
    detected["event_candidate_flag"] = False
    detected["event_severity_score"] = 0.0

    if event_type == "tachycardia_event":
        hr_values = detected["selected_hr_bpm"].to_numpy(dtype=float)
        flags = np.isfinite(hr_values) & (hr_values >= threshold)
        severities = np.where(flags, (hr_values - threshold) / max(threshold, 1.0), 0.0)
        detected["event_candidate_flag"] = flags
        detected["event_severity_score"] = np.clip(severities, 0.0, None)
        detected["event_trigger_rule"] = np.where(flags, "tachy_hr_threshold", "")
        return detected

    if event_type == "bradycardia_event":
        hr_values = detected["selected_hr_bpm"].to_numpy(dtype=float)
        flags = np.isfinite(hr_values) & (hr_values <= threshold)
        severities = np.where(flags, (threshold - hr_values) / max(threshold, 1.0), 0.0)
        detected["event_candidate_flag"] = flags
        detected["event_severity_score"] = np.clip(severities, 0.0, None)
        detected["event_trigger_rule"] = np.where(flags, "brady_hr_threshold", "")
        return detected

    previous_values = detected["selected_hr_prev_bpm"].to_numpy(dtype=float)
    current_values = detected["selected_hr_bpm"].to_numpy(dtype=float)
    deltas = detected["selected_hr_delta_bpm"].to_numpy(dtype=float)
    flags = np.isfinite(current_values) & np.isfinite(previous_values) & (deltas >= threshold)
    detected["event_candidate_flag"] = flags
    detected["event_anchor_hr_bpm"] = np.where(flags, previous_values, math.nan)
    detected["event_severity_score"] = np.where(flags, (deltas - threshold) / max(threshold, 1.0), 0.0)
    detected["event_trigger_rule"] = np.where(flags, "abrupt_delta", "")
    return _apply_abrupt_confirmation_windows(
        detected,
        threshold=threshold,
        confirmation_ratio=float(event_cfg.get("thresholds", {}).get("abrupt_confirmation_ratio", 0.5)),
    )


def apply_quality_gated_event_logic(
    frame: pd.DataFrame,
    *,
    event_type: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    event_cfg = config or {}
    quality_cfg = event_cfg.get("quality_gate", {})
    mode = str(quality_cfg.get("mode", "suppress"))
    if mode != "suppress":
        raise ValueError(f"Unsupported Stage 4 quality-gate mode: {mode}")

    gated = frame.copy()
    gated["quality_gate_passed"] = False
    gated["event_validity_flag"] = False
    gated["event_reason_code"] = "no_candidate"

    quality_passed: list[bool] = []
    quality_reasons: list[str] = []
    event_reasons: list[str] = []
    provisional_validity: list[bool] = []
    for row in gated.itertuples(index=False):
        selected_valid = _safe_bool(row.selected_hr_is_valid)
        candidate_flag = _safe_bool(row.event_candidate_flag)
        quality_reason = str(row.quality_gate_reason)
        event_reason = "no_candidate"
        quality_pass = False

        if event_type == "abrupt_change_event":
            previous_hr = _safe_float(row.selected_hr_prev_bpm)
            if not selected_valid:
                quality_pass = False
            elif not np.isfinite(previous_hr):
                quality_reason = "no_previous_reliable_hr"
                quality_pass = False
            else:
                quality_reason = "pass"
                quality_pass = True
        else:
            if selected_valid:
                quality_reason = "pass"
                quality_pass = True

        provisional_valid = bool(candidate_flag and quality_pass)
        if candidate_flag and not quality_pass:
            event_reason = "suppressed_low_quality" if quality_reason != "no_previous_reliable_hr" else "suppressed_no_previous_reliable_hr"
        elif provisional_valid:
            event_reason = "candidate_pending_episode"

        quality_passed.append(quality_pass)
        quality_reasons.append(quality_reason)
        provisional_validity.append(provisional_valid)
        event_reasons.append(event_reason)

    gated["quality_gate_passed"] = quality_passed
    gated["quality_gate_reason"] = quality_reasons
    gated["event_validity_flag"] = provisional_validity
    gated["event_reason_code"] = event_reasons
    return gated


def consolidate_event_episodes(
    frame: pd.DataFrame,
    *,
    event_type: str,
    config: dict[str, Any] | None = None,
    prefix: str = "",
) -> pd.DataFrame:
    event_cfg = config or {}
    episode_merge_gap = int(event_cfg.get("persistence", {}).get("episode_merge_gap_windows", 1))
    min_valid_windows = _min_windows_for_event(event_type, event_cfg)
    episode_id_col = f"{prefix}episode_id"
    start_col = f"{prefix}episode_start_flag"
    end_col = f"{prefix}episode_end_flag"

    consolidated = frame.sort_values(by=["subject_id", "window_index", "start_time_s"]).reset_index(drop=True).copy()
    consolidated[episode_id_col] = ""
    consolidated[start_col] = False
    consolidated[end_col] = False

    valid_rows = consolidated.loc[consolidated["event_validity_flag"].astype(bool)].copy()
    if valid_rows.empty:
        return consolidated

    next_episode_index = 1
    for subject_id, subject_frame in valid_rows.groupby("subject_id", sort=False):
        candidate_indices = list(subject_frame.index)
        groups: list[list[int]] = []
        current_group: list[int] = []
        previous_window_index: int | None = None
        for frame_index in candidate_indices:
            window_index = int(consolidated.loc[frame_index, "window_index"])
            if previous_window_index is None or (window_index - previous_window_index) <= (episode_merge_gap + 1):
                current_group.append(frame_index)
            else:
                groups.append(current_group)
                current_group = [frame_index]
            previous_window_index = window_index
        if current_group:
            groups.append(current_group)

        for group in groups:
            if len(group) < min_valid_windows:
                consolidated.loc[group, "event_validity_flag"] = False
                consolidated.loc[group, "event_reason_code"] = "suppressed_below_min_persistence"
                continue

            episode_id = f"{_event_prefix(event_type)}_{subject_id}_{next_episode_index:04d}"
            next_episode_index += 1
            consolidated.loc[group, episode_id_col] = episode_id
            consolidated.loc[group, "event_reason_code"] = "valid_event_episode"
            consolidated.at[group[0], start_col] = True
            consolidated.at[group[-1], end_col] = True

    return consolidated


def summarize_detected_event_episodes(
    frame: pd.DataFrame,
    *,
    prefix: str = "",
) -> pd.DataFrame:
    episode_id_col = f"{prefix}episode_id"
    start_col = f"{prefix}episode_start_flag"
    end_col = f"{prefix}episode_end_flag"
    if episode_id_col not in frame.columns:
        raise ValueError(f"Missing episode column: {episode_id_col}")
    episode_frame = frame.loc[frame[episode_id_col].astype(str) != ""].copy()
    if episode_frame.empty:
        return pd.DataFrame(
            columns=[
                "split",
                "dataset",
                "subject_id",
                "event_type",
                episode_id_col,
                "num_windows",
                "episode_start_time_s",
                "episode_end_time_s",
                "mean_event_severity_score",
            ]
        )

    rows: list[dict[str, float | str]] = []
    for episode_id, group in episode_frame.groupby(episode_id_col, sort=False):
        rows.append(
            {
                "split": str(group["split"].iloc[0]),
                "dataset": str(group["dataset"].iloc[0]),
                "subject_id": str(group["subject_id"].iloc[0]),
                "event_type": str(group["event_type"].iloc[0]),
                episode_id_col: str(episode_id),
                "num_windows": float(group.shape[0]),
                "episode_start_time_s": float(group.loc[group[start_col].astype(bool), "start_time_s"].iloc[0])
                if group[start_col].astype(bool).any()
                else float(group["start_time_s"].min()),
                "episode_end_time_s": float(
                    (
                        group.loc[group[end_col].astype(bool), "start_time_s"]
                        + group.loc[group[end_col].astype(bool), "duration_s"]
                    ).iloc[0]
                )
                if group[end_col].astype(bool).any()
                else float((group["start_time_s"] + group["duration_s"]).max()),
                "mean_event_severity_score": float(group["event_severity_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_reference_source_frame(
    frame: pd.DataFrame,
    *,
    event_type: str,
) -> pd.DataFrame:
    if "ref_hr_bpm" not in frame.columns:
        raise ValueError("Base frame must contain ref_hr_bpm for Stage 4 proxy targets.")

    reference = frame.loc[:, list(IDENTITY_COLUMNS)].copy()
    reference["event_type"] = event_type
    reference["selected_hr_source"] = "ecg_reference_hr"
    reference["selected_hr_bpm"] = frame["ref_hr_bpm"].to_numpy(dtype=float)
    window_is_valid = frame["window_is_valid"].astype(bool).to_numpy() if "window_is_valid" in frame.columns else True
    reference["selected_hr_is_valid"] = np.isfinite(reference["selected_hr_bpm"].to_numpy(dtype=float)) & np.asarray(window_is_valid, dtype=bool)
    reference["selected_hr_prev_bpm"] = math.nan
    reference["selected_hr_delta_bpm"] = math.nan
    reference["quality_gate_reason"] = np.where(reference["selected_hr_is_valid"], "pass", "reference_hr_invalid")
    reference["event_anchor_hr_bpm"] = math.nan
    reference["event_threshold_bpm"] = math.nan
    reference["event_trigger_rule"] = ""
    if "window_is_valid" in frame.columns:
        reference["window_is_valid"] = frame["window_is_valid"].astype(bool).to_numpy()
    return reference


def build_proxy_event_targets(
    base_frame: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    event_cfg = config or {}
    target_frames: list[pd.DataFrame] = []
    for event_type in EVENT_TYPES:
        reference = _build_reference_source_frame(base_frame, event_type=event_type)
        reference["event_threshold_bpm"] = _threshold_for_event(event_type, event_cfg)
        detected = detect_window_event_candidates(reference, event_type=event_type, config=event_cfg)
        gated = apply_quality_gated_event_logic(detected, event_type=event_type, config=event_cfg)
        consolidated = consolidate_event_episodes(gated, event_type=event_type, config=event_cfg, prefix="proxy_")
        target_frames.append(
            consolidated.rename(
                columns={
                    "event_validity_flag": "proxy_event_target",
                    "event_reason_code": "proxy_event_reason_code",
                    "event_candidate_flag": "proxy_event_candidate_flag",
                    "quality_gate_passed": "proxy_quality_gate_passed",
                    "quality_gate_reason": "proxy_quality_gate_reason",
                }
            )[
                [
                    *IDENTITY_COLUMNS,
                    "event_type",
                    "proxy_event_target",
                    "proxy_event_reason_code",
                    "proxy_event_candidate_flag",
                    "proxy_quality_gate_passed",
                    "proxy_quality_gate_reason",
                    "proxy_episode_id",
                    "proxy_episode_start_flag",
                    "proxy_episode_end_flag",
                ]
            ]
        )
    return pd.concat(target_frames, ignore_index=True, sort=False)


def build_stage4_event_predictions(
    base_frame: pd.DataFrame,
    *,
    split_name: str,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    event_cfg = config or {}
    working_base = base_frame.copy()
    working_base["split"] = split_name

    event_frames: list[pd.DataFrame] = []
    source_cfg = event_cfg.get("source", {})
    for event_type in EVENT_TYPES:
        selected = select_stage4_hr_source(
            working_base,
            event_type=event_type,
            source_config=source_cfg,
            event_config=event_cfg,
        )
        detected = detect_window_event_candidates(selected, event_type=event_type, config=event_cfg)
        gated = apply_quality_gated_event_logic(detected, event_type=event_type, config=event_cfg)
        consolidated = consolidate_event_episodes(gated, event_type=event_type, config=event_cfg)
        event_frames.append(consolidated)

    predictions = pd.concat(event_frames, ignore_index=True, sort=False)
    proxy_targets = build_proxy_event_targets(working_base, config=event_cfg)
    predictions = predictions.merge(
        proxy_targets,
        on=[*IDENTITY_COLUMNS, "event_type"],
        how="left",
        validate="one_to_one",
    )

    predictions["proxy_event_target"] = predictions["proxy_event_target"].fillna(False).astype(bool)
    predictions["proxy_episode_id"] = predictions["proxy_episode_id"].fillna("")
    predictions["proxy_episode_start_flag"] = predictions["proxy_episode_start_flag"].fillna(False).astype(bool)
    predictions["proxy_episode_end_flag"] = predictions["proxy_episode_end_flag"].fillna(False).astype(bool)

    predictions = predictions.sort_values(by=["split", "event_type", "subject_id", "window_index", "start_time_s"]).reset_index(drop=True)
    for column_name in PREDICTION_COLUMNS:
        if column_name not in predictions.columns:
            if column_name.endswith("_flag") or column_name.endswith("_passed") or column_name.endswith("_target") or column_name == "selected_hr_is_valid":
                predictions[column_name] = False
            elif column_name.endswith("_reason") or column_name.endswith("_id") or column_name.endswith("_rule") or column_name.endswith("_source") or column_name == "event_type":
                predictions[column_name] = ""
            else:
                predictions[column_name] = math.nan
    return predictions.loc[:, list(PREDICTION_COLUMNS)]


def _summarize_binary_event_rows(
    frame: pd.DataFrame,
    *,
    method: str,
    event_type: str,
    split_name: str,
) -> dict[str, float | str]:
    target = frame["proxy_event_target"].astype(bool)
    pred = frame["event_validity_flag"].astype(bool)
    tp = int(np.sum(target & pred))
    fp = int(np.sum(~target & pred))
    fn = int(np.sum(target & ~pred))
    tn = int(np.sum(~target & ~pred))
    metrics = compute_precision_recall_f1(tp, fp, fn)
    accuracy = float((tp + tn) / max(frame.shape[0], 1))

    proxy_episode_id = frame["proxy_episode_id"].astype(str) if "proxy_episode_id" in frame.columns else pd.Series(dtype=str)
    pred_episode_id = frame["episode_id"].astype(str) if "episode_id" in frame.columns else pd.Series(dtype=str)
    return {
        "task": "event_detection",
        "method": method,
        "event_type": event_type,
        "split": split_name,
        "num_eval_windows": float(frame.shape[0]),
        "num_eval_events": float(proxy_episode_id.loc[proxy_episode_id != ""].nunique()),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": accuracy,
        "selected_hr_valid_fraction": float(np.mean(frame["selected_hr_is_valid"].astype(bool))) if not frame.empty else math.nan,
        "quality_gate_pass_fraction": float(np.mean(frame["quality_gate_passed"].astype(bool))) if not frame.empty else math.nan,
        "candidate_fraction": float(np.mean(frame["event_candidate_flag"].astype(bool))) if not frame.empty else math.nan,
        "suppressed_candidate_count": float(np.sum(frame["event_candidate_flag"].astype(bool) & ~frame["event_validity_flag"].astype(bool))),
        "valid_event_fraction": float(np.mean(frame["event_validity_flag"].astype(bool))) if not frame.empty else math.nan,
        "num_pred_events": float(pred_episode_id.loc[pred_episode_id != ""].nunique()),
    }


def summarize_stage4_event_metrics(
    frame: pd.DataFrame,
    *,
    method: str = "stage4_rule_events_v1",
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "task",
                "method",
                "event_type",
                "split",
                "num_eval_windows",
                "num_eval_events",
                "precision",
                "recall",
                "f1",
                "accuracy",
                "selected_hr_valid_fraction",
                "quality_gate_pass_fraction",
                "candidate_fraction",
                "suppressed_candidate_count",
                "valid_event_fraction",
                "num_pred_events",
            ]
        )

    rows: list[dict[str, float | str]] = []
    for split_name, split_frame in frame.groupby("split", sort=False):
        for event_type in EVENT_TYPES:
            event_frame = split_frame.loc[split_frame["event_type"] == event_type].copy()
            rows.append(
                _summarize_binary_event_rows(
                    event_frame,
                    method=method,
                    event_type=event_type,
                    split_name=str(split_name),
                )
            )
        rows.append(
            _summarize_binary_event_rows(
                split_frame,
                method=method,
                event_type="all_events",
                split_name=str(split_name),
            )
        )
    return pd.DataFrame(rows)
