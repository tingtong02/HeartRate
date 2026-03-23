from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.preprocess import build_window_samples, detect_ecg_peaks, trim_record_to_common_duration
from heart_rate_cnn.stage1_hr import (
    estimate_hr_frequency_stage1,
    estimate_hr_time_stage1,
    fuse_hr_estimates,
)
from heart_rate_cnn.stage2_beat import (
    clean_ibi_series,
    compute_beat_quality_proxy,
    compute_time_domain_prv_features,
    detect_beats_in_window,
    extract_ibi_from_beats,
)
from heart_rate_cnn.stage3_quality import (
    apply_ml_quality_decision,
    apply_robust_hr_policy_sequence,
    apply_rule_based_quality_decision,
    build_quality_target,
    build_refined_threshold_grid,
    compute_local_beat_fallback_hr,
    evaluate_ml_threshold_grid,
    extract_quality_features,
    fit_quality_logistic_regression,
    predict_quality_logistic_regression,
    summarize_threshold_selection,
)
from heart_rate_cnn.types import WindowSample


STAGE4_IDENTITY_COLUMNS: tuple[str, ...] = (
    "split",
    "dataset",
    "subject_id",
    "window_index",
    "start_time_s",
    "duration_s",
)

STAGE4_SOURCE_SPECS: dict[str, dict[str, str]] = {
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

ROBUST_SOURCE_CATEGORIES: tuple[str, ...] = (
    "frequency",
    "beat_fallback",
    "hold_previous",
    "none",
)

ROBUST_ACTION_CATEGORIES: tuple[str, ...] = (
    "direct_update",
    "fallback_update",
    "hold",
    "reject",
)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return bool(value)


def make_loader(dataset_name: str, root_dir: str):
    if dataset_name == "ppg_dalia":
        return PPGDaliaLoader(root_dir)
    if dataset_name == "wesad":
        return WESADLoader(root_dir)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _default_quality_reason(
    *,
    row: pd.Series,
    invalid_reason_code: str,
    reason_col: str | None = None,
) -> str:
    if safe_bool(row.get("selected_hr_is_valid", False)):
        return "pass"
    selected_hr = safe_float(row.get("selected_hr_bpm"))
    if not np.isfinite(selected_hr):
        if reason_col is not None:
            reason_value = row.get(reason_col)
            if isinstance(reason_value, str) and reason_value:
                return reason_value
        return "selected_hr_unavailable"
    return invalid_reason_code


def select_stage4_signal_source(
    frame: pd.DataFrame,
    *,
    source_name: str,
    split_name: str | None = None,
) -> pd.DataFrame:
    if source_name not in STAGE4_SOURCE_SPECS:
        raise ValueError(f"Unsupported Stage 4 source: {source_name}")

    spec = STAGE4_SOURCE_SPECS[source_name]
    missing_columns = [
        column_name
        for column_name in (spec["hr_col"], spec["valid_col"], spec.get("reason_col"))
        if column_name is not None and column_name not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required source columns for {source_name}: {missing_columns}")

    selected = pd.DataFrame(index=frame.index)
    for column_name in STAGE4_IDENTITY_COLUMNS:
        if column_name in frame.columns:
            selected[column_name] = frame[column_name]
        elif column_name == "split":
            selected[column_name] = split_name if split_name is not None else ""
        else:
            raise ValueError(f"Missing identity column for Stage 4 selection: {column_name}")

    selected["selected_hr_source"] = source_name
    selected["selected_hr_bpm"] = frame[spec["hr_col"]].to_numpy(dtype=float)
    selected["selected_hr_is_valid"] = frame[spec["valid_col"]].astype(bool).to_numpy()
    selected["selected_hr_prev_bpm"] = math.nan
    selected["selected_hr_delta_bpm"] = math.nan
    selected["quality_gate_reason"] = [
        _default_quality_reason(
            row=row,
            invalid_reason_code=str(spec.get("invalid_reason_code", "selected_hr_invalid")),
            reason_col=spec.get("reason_col"),
        )
        for _, row in selected.join(frame[[spec.get("reason_col")]] if spec.get("reason_col") else pd.DataFrame(index=frame.index)).iterrows()
    ]

    passthrough_columns = [
        "window_is_valid",
        "ref_hr_bpm",
        "ml_validity_flag",
        "rule_validity_flag",
        "motion_refined_validity_flag",
        "robust_hr_source",
        "robust_hr_action",
        "robust_hr_is_valid",
        "policy_reason_code",
        "hold_applied",
        "hold_age_windows",
        "subject_boundary_reset",
    ]
    for column_name in passthrough_columns:
        if column_name in frame.columns:
            selected[column_name] = frame[column_name].to_numpy()
    return selected


def attach_previous_valid_hr_context(frame: pd.DataFrame) -> pd.DataFrame:
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

        current_hr = safe_float(row.selected_hr_bpm)
        if np.isfinite(previous_valid_hr) and np.isfinite(current_hr):
            previous_deltas.append(abs(current_hr - previous_valid_hr))
        else:
            previous_deltas.append(math.nan)

        if safe_bool(row.selected_hr_is_valid) and np.isfinite(current_hr):
            previous_valid_hr = current_hr
        previous_subject = current_subject

    ordered["selected_hr_prev_bpm"] = previous_values
    ordered["selected_hr_delta_bpm"] = previous_deltas
    return ordered


def _build_stage3_source_rows(
    *,
    loader,
    subject_ids: list[str],
    preprocess_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    stage1_cfg: dict[str, Any],
    stage3_cfg: dict[str, Any],
) -> tuple[list[dict[str, float | int | str | bool | None]], list[WindowSample]]:
    hr_band_bpm = tuple(float(value) for value in eval_cfg["hr_band_bpm"])
    rows: list[dict[str, float | int | str | bool | None]] = []
    windows_out: list[WindowSample] = []
    for subject_id in subject_ids:
        record = loader.load_subject(subject_id)
        windows = build_window_samples(
            record=record,
            target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
            window_seconds=float(preprocess_cfg["window_seconds"]),
            step_seconds=float(preprocess_cfg["step_seconds"]),
        )
        windows_out.extend(windows)
        for window in windows:
            freq_result = estimate_hr_frequency_stage1(window.ppg, window.ppg_fs, hr_band_bpm, stage1_cfg["frequency"])
            time_result = estimate_hr_time_stage1(window.ppg, window.ppg_fs, hr_band_bpm, stage1_cfg["time"])
            fusion_result = fuse_hr_estimates(
                freq_result,
                time_result,
                agreement_threshold_bpm=float(stage1_cfg["fusion"]["agreement_threshold_bpm"]),
                conflict_threshold_bpm=float(stage1_cfg["fusion"]["conflict_threshold_bpm"]),
            )
            feature_row = extract_quality_features(
                window,
                freq_result=freq_result,
                time_result=time_result,
                fusion_result=fusion_result,
                preprocess_config=stage1_cfg["frequency"],
                motion_config=stage3_cfg.get("motion", {}),
            )
            target_row = build_quality_target(
                ref_hr_bpm=window.ref_hr_bpm,
                freq_pred_hr_bpm=float(freq_result["freq_pred_hr_bpm"]),
                window_is_valid=bool(window.is_valid),
                freq_is_valid=bool(freq_result["freq_is_valid"]),
                good_error_bpm=float(stage3_cfg["target"]["good_error_bpm"]),
                poor_error_bpm=float(stage3_cfg["target"]["poor_error_bpm"]),
            )
            rule_row = apply_rule_based_quality_decision(
                window_is_valid=bool(window.is_valid),
                features=feature_row,
                config=stage3_cfg["rule"],
            )
            beat_fallback_row = compute_local_beat_fallback_hr(
                window,
                config=stage3_cfg.get("robust_hr_policy", {}),
            )
            rows.append(
                {
                    "dataset": window.dataset,
                    "subject_id": window.subject_id,
                    "window_index": window.window_index,
                    "start_time_s": window.start_time_s,
                    "duration_s": window.duration_s,
                    "ref_hr_bpm": window.ref_hr_bpm,
                    "window_is_valid": bool(window.is_valid),
                    **feature_row,
                    **target_row,
                    "rule_signal_quality_score": float(rule_row["signal_quality_score"]),
                    "rule_signal_quality_label": str(rule_row["signal_quality_label"]),
                    "rule_validity_flag": bool(rule_row["validity_flag"]),
                    "motion_flag": bool(rule_row["motion_flag"]),
                    "ungated_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"]),
                    "ungated_is_valid": bool(window.is_valid and freq_result["freq_is_valid"]),
                    "rule_gated_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"]) if bool(rule_row["validity_flag"]) else math.nan,
                    "rule_gated_is_valid": bool(window.is_valid and rule_row["validity_flag"]),
                    **beat_fallback_row,
                }
            )
    return rows, windows_out


def _apply_ml_decisions(frame: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    decisions = [
        apply_ml_quality_decision(
            signal_quality_score=float(score),
            threshold=threshold,
            window_is_valid=bool(row["window_is_valid"]),
            freq_is_valid=bool(row["freq_is_valid"]),
            motion_flag=bool(row["motion_flag"]),
        )
        for score, row in zip(frame["ml_signal_quality_score"].tolist(), frame.to_dict(orient="records"))
    ]
    updated = frame.copy()
    updated["ml_signal_quality_label"] = [str(row["signal_quality_label"]) for row in decisions]
    updated["ml_validity_flag"] = [bool(row["validity_flag"]) for row in decisions]
    updated["ml_gated_pred_hr_bpm"] = [
        float(pred_hr) if bool(validity_flag) else math.nan
        for pred_hr, validity_flag in zip(updated["ungated_pred_hr_bpm"].tolist(), updated["ml_validity_flag"].tolist())
    ]
    updated["ml_gated_is_valid"] = [
        bool(validity_flag and ungated_valid)
        for validity_flag, ungated_valid in zip(updated["ml_validity_flag"].tolist(), updated["ungated_is_valid"].tolist())
    ]
    return updated


def build_quality_aware_source_frames(
    *,
    loader,
    train_subjects: list[str],
    eval_subjects: list[str],
    preprocess_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    stage1_cfg: dict[str, Any],
    stage3_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[WindowSample], list[WindowSample], float]:
    train_rows, train_windows = _build_stage3_source_rows(
        loader=loader,
        subject_ids=train_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
    )
    eval_rows, eval_windows = _build_stage3_source_rows(
        loader=loader,
        subject_ids=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
    )
    train_frame = pd.DataFrame(train_rows)
    eval_frame = pd.DataFrame(eval_rows)

    train_ml_frame = train_frame.loc[train_frame["quality_target_label"].isin(["good", "poor"])].copy()
    ml_model = fit_quality_logistic_regression(
        train_ml_frame.to_dict(orient="records"),
        train_ml_frame["quality_target_label"].tolist(),
        random_seed=int(eval_cfg["random_seed"]),
        c_value=float(stage3_cfg["ml"]["c_value"]),
        max_iter=int(stage3_cfg["ml"]["max_iter"]),
    )
    train_frame["ml_signal_quality_score"] = predict_quality_logistic_regression(ml_model, train_frame.to_dict(orient="records"))
    eval_frame["ml_signal_quality_score"] = predict_quality_logistic_regression(ml_model, eval_frame.to_dict(orient="records"))

    coarse_grid = [float(value) for value in stage3_cfg["ml"]["threshold_grid"]]
    min_retention_ratio = float(stage3_cfg["ml"]["min_retention_ratio"])
    coarse_train_sweep = evaluate_ml_threshold_grid(
        train_frame,
        score_col="ml_signal_quality_score",
        pred_col="ungated_pred_hr_bpm",
        valid_col="ungated_is_valid",
        threshold_grid=coarse_grid,
        min_retention_ratio=min_retention_ratio,
        split_name="train_select",
        sweep_stage="coarse",
    )
    threshold_summary = summarize_threshold_selection(coarse_train_sweep)
    selected_threshold = float(threshold_summary["selected_threshold"])

    if bool(stage3_cfg["ml"].get("refine_threshold", True)):
        fine_grid = build_refined_threshold_grid(
            center_threshold=selected_threshold,
            refinement_radius=float(stage3_cfg["ml"].get("refinement_radius", 0.10)),
            refinement_step=float(stage3_cfg["ml"].get("refinement_step", 0.02)),
        )
        fine_train_sweep = evaluate_ml_threshold_grid(
            train_frame,
            score_col="ml_signal_quality_score",
            pred_col="ungated_pred_hr_bpm",
            valid_col="ungated_is_valid",
            threshold_grid=fine_grid,
            min_retention_ratio=min_retention_ratio,
            split_name="train_select",
            sweep_stage="fine",
        )
        threshold_summary = summarize_threshold_selection(fine_train_sweep)
        selected_threshold = float(threshold_summary["selected_threshold"])

    train_frame = _apply_ml_decisions(train_frame, threshold=selected_threshold)
    eval_frame = _apply_ml_decisions(eval_frame, threshold=selected_threshold)

    train_robust = apply_robust_hr_policy_sequence(train_frame, config=stage3_cfg.get("robust_hr_policy", {}))
    eval_robust = apply_robust_hr_policy_sequence(eval_frame, config=stage3_cfg.get("robust_hr_policy", {}))
    for column_name in train_robust.columns:
        train_frame[column_name] = train_robust[column_name].tolist()
    for column_name in eval_robust.columns:
        eval_frame[column_name] = eval_robust[column_name].tolist()

    return train_frame, eval_frame, train_windows, eval_windows, selected_threshold


def _build_ibi_configs(
    *,
    stage3_cfg: dict[str, Any],
    stage4_shared_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    shared_cfg = stage4_shared_cfg or {}
    robust_cfg = stage3_cfg.get("robust_hr_policy", {})
    beat_variant_mode = str(shared_cfg.get("beat_variant_mode", robust_cfg.get("fallback_variant_mode", "enhanced")))
    beat_cfg = {"variant_mode": beat_variant_mode}
    ibi_cfg = {
        "variant_mode": beat_variant_mode,
        "min_ibi_s": float(shared_cfg.get("min_ibi_s", robust_cfg.get("fallback_min_ibi_s", 0.33))),
        "max_ibi_s": float(shared_cfg.get("max_ibi_s", robust_cfg.get("fallback_max_ibi_s", 1.5))),
        "local_median_radius": int(shared_cfg.get("local_median_radius", robust_cfg.get("fallback_local_median_radius", 2))),
        "max_deviation_ratio": float(shared_cfg.get("max_deviation_ratio", robust_cfg.get("fallback_max_deviation_ratio", 0.25))),
        "adjacent_jump_ratio": float(shared_cfg.get("adjacent_jump_ratio", robust_cfg.get("fallback_adjacent_jump_ratio", 0.22))),
        "jump_anchor_ratio": float(shared_cfg.get("jump_anchor_ratio", robust_cfg.get("fallback_jump_anchor_ratio", 0.12))),
        "short_series_threshold": int(shared_cfg.get("short_series_threshold", robust_cfg.get("fallback_short_series_threshold", 5))),
        "min_clean_ibi": int(shared_cfg.get("min_ibi_clean", robust_cfg.get("fallback_min_clean_ibi", 3))),
    }
    quality_cfg = {
        "good_score_threshold": float(shared_cfg.get("beat_quality_threshold", robust_cfg.get("fallback_beat_quality_threshold", 0.55))),
        "plausibility_margin_s": float(shared_cfg.get("plausibility_margin_s", robust_cfg.get("fallback_plausibility_margin_s", 0.08))),
        "jump_good_ratio": float(shared_cfg.get("jump_good_ratio", robust_cfg.get("fallback_jump_good_ratio", 0.08))),
        "jump_bad_ratio": float(shared_cfg.get("jump_bad_ratio", robust_cfg.get("fallback_jump_bad_ratio", 0.25))),
        "crowding_good_scale": float(shared_cfg.get("crowding_good_scale", robust_cfg.get("fallback_crowding_good_scale", 1.10))),
        "missing_ibi_score": float(shared_cfg.get("missing_ibi_score", robust_cfg.get("fallback_missing_ibi_score", 0.50))),
        "weights": {
            "base_peak_quality": float(shared_cfg.get("weight_base_peak_quality", robust_cfg.get("fallback_weight_base_peak_quality", 0.60))),
            "ibi_plausibility": float(shared_cfg.get("weight_ibi_plausibility", robust_cfg.get("fallback_weight_ibi_plausibility", 0.20))),
            "ibi_stability": float(shared_cfg.get("weight_ibi_stability", robust_cfg.get("fallback_weight_ibi_stability", 0.10))),
            "crowding": float(shared_cfg.get("weight_crowding", robust_cfg.get("fallback_weight_crowding", 0.05))),
            "clean_pair_bonus": float(shared_cfg.get("weight_clean_pair_bonus", robust_cfg.get("fallback_weight_clean_pair_bonus", 0.05))),
        },
    }
    return beat_cfg, ibi_cfg, quality_cfg


def _compute_local_deviation_ratios(ibi_values: np.ndarray, radius: int = 1) -> np.ndarray:
    ibi = np.asarray(ibi_values, dtype=float)
    ratios: list[float] = []
    for index, value in enumerate(ibi):
        start = max(0, index - radius)
        end = min(ibi.size, index + radius + 1)
        neighborhood = ibi[start:end]
        if neighborhood.size < 2:
            continue
        center = float(np.median(neighborhood))
        if not np.isfinite(center) or center <= 1e-8:
            continue
        ratios.append(abs(float(value) - center) / center)
    return np.asarray(ratios, dtype=float)


def _compute_turning_point_ratio(ibi_values: np.ndarray) -> float:
    ibi = np.asarray(ibi_values, dtype=float)
    if ibi.size < 3:
        return math.nan
    diff = np.diff(ibi)
    if diff.size < 2:
        return math.nan
    turning_points = 0
    valid_pairs = 0
    for left, right in zip(diff[:-1], diff[1:]):
        if np.isclose(left, 0.0) or np.isclose(right, 0.0):
            continue
        valid_pairs += 1
        if np.sign(left) != np.sign(right):
            turning_points += 1
    if valid_pairs == 0:
        return math.nan
    return float(turning_points / valid_pairs)


def _compute_ibi_irregularity_features(ibi_clean_s: np.ndarray, *, prefix: str = "") -> dict[str, float]:
    ibi = np.asarray(ibi_clean_s, dtype=float)
    features = {
        f"{prefix}successive_ibi_jump_mean_ms": math.nan,
        f"{prefix}successive_ibi_jump_max_ms": math.nan,
        f"{prefix}local_deviation_ratio_mean": math.nan,
        f"{prefix}local_deviation_ratio_max": math.nan,
        f"{prefix}ibi_mad_ms": math.nan,
        f"{prefix}turning_point_ratio": math.nan,
    }
    if ibi.size == 0:
        return features

    ibi_ms = ibi * 1000.0
    features[f"{prefix}ibi_mad_ms"] = float(np.median(np.abs(ibi_ms - np.median(ibi_ms))))

    diff_ms = np.abs(np.diff(ibi_ms))
    if diff_ms.size > 0:
        features[f"{prefix}successive_ibi_jump_mean_ms"] = float(np.mean(diff_ms))
        features[f"{prefix}successive_ibi_jump_max_ms"] = float(np.max(diff_ms))

    local_ratios = _compute_local_deviation_ratios(ibi, radius=1)
    if local_ratios.size > 0:
        features[f"{prefix}local_deviation_ratio_mean"] = float(np.mean(local_ratios))
        features[f"{prefix}local_deviation_ratio_max"] = float(np.max(local_ratios))

    features[f"{prefix}turning_point_ratio"] = _compute_turning_point_ratio(ibi)
    return features


def _compute_reference_window_features(
    *,
    ecg_peaks: np.ndarray,
    ecg_fs: float,
    start_time_s: float,
    duration_s: float,
    ibi_cfg: dict[str, Any],
) -> dict[str, float | bool]:
    start_index = int(round(start_time_s * ecg_fs))
    end_index = int(round((start_time_s + duration_s) * ecg_fs))
    window_beats = np.asarray(ecg_peaks, dtype=int)
    window_beats = window_beats[(window_beats >= start_index) & (window_beats < end_index)]
    ref_ibi_s = extract_ibi_from_beats(window_beats, fs=ecg_fs)
    ref_clean = clean_ibi_series(ref_ibi_s, ibi_cfg)
    ref_clean_ibi = np.asarray(ref_clean["ibi_clean_s"], dtype=float)
    ref_prv = compute_time_domain_prv_features(
        ref_clean_ibi,
        num_beats=int(window_beats.size),
        num_ibi_raw=int(ref_ibi_s.size),
        num_ibi_clean=int(ref_clean_ibi.size),
    )
    ref_irregularity = _compute_ibi_irregularity_features(ref_clean_ibi, prefix="ref_")
    return {
        "ref_num_beats": float(window_beats.size),
        "ref_num_ibi_raw": float(ref_ibi_s.size),
        "ref_num_ibi_clean": float(ref_clean_ibi.size),
        "ref_ibi_is_valid": bool(ref_clean.get("ibi_is_valid", False)),
        "ref_ibi_removed_ratio": safe_float(ref_clean.get("ibi_removed_ratio"), default=math.nan),
        "ref_mean_ibi_ms": safe_float(ref_prv.get("mean_ibi_ms"), default=math.nan),
        "ref_median_ibi_ms": safe_float(ref_prv.get("median_ibi_ms"), default=math.nan),
        "ref_mean_hr_bpm_from_ibi": safe_float(ref_prv.get("mean_hr_bpm_from_ibi"), default=math.nan),
        "ref_sdnn_ms": safe_float(ref_prv.get("sdnn_ms"), default=math.nan),
        "ref_rmssd_ms": safe_float(ref_prv.get("rmssd_ms"), default=math.nan),
        "ref_pnn50": safe_float(ref_prv.get("pnn50"), default=math.nan),
        "ref_ibi_cv": safe_float(ref_prv.get("ibi_cv"), default=math.nan),
        **ref_irregularity,
    }


def _compute_beat_quality_summary(
    *,
    window: WindowSample,
    beat_cfg: dict[str, Any],
    ibi_cfg: dict[str, Any],
    quality_cfg: dict[str, Any],
    raw_beats: np.ndarray,
) -> dict[str, float]:
    if raw_beats.size == 0:
        return {
            "beat_quality_mean_score": math.nan,
            "beat_quality_good_ratio": math.nan,
            "beat_quality_good_count": 0.0,
        }
    quality = compute_beat_quality_proxy(
        window.ppg,
        raw_beats,
        fs=window.ppg_fs,
        beat_config=beat_cfg,
        ibi_config=ibi_cfg,
        quality_config=quality_cfg,
    )
    scores = np.asarray(quality["beat_quality_score"], dtype=float)
    keep_mask = np.asarray(quality["beat_is_kept_by_quality"], dtype=bool)
    return {
        "beat_quality_mean_score": float(np.mean(scores)) if scores.size else math.nan,
        "beat_quality_good_ratio": float(np.mean(keep_mask)) if keep_mask.size else math.nan,
        "beat_quality_good_count": float(np.sum(keep_mask)),
    }


def _build_stage4_feature_row(
    *,
    window: WindowSample,
    source_row: dict[str, Any],
    selected_row: dict[str, Any],
    ecg_peaks: np.ndarray,
    ecg_fs: float,
    beat_cfg: dict[str, Any],
    ibi_cfg: dict[str, Any],
    quality_cfg: dict[str, Any],
    stage4_shared_cfg: dict[str, Any],
) -> dict[str, float | str | bool]:
    raw_beats = detect_beats_in_window(window.ppg, fs=window.ppg_fs, config=beat_cfg)
    pred_ibi_s = extract_ibi_from_beats(raw_beats, fs=window.ppg_fs)
    pred_clean = clean_ibi_series(pred_ibi_s, ibi_cfg)
    pred_clean_ibi = np.asarray(pred_clean["ibi_clean_s"], dtype=float)
    pred_prv = compute_time_domain_prv_features(
        pred_clean_ibi,
        num_beats=int(raw_beats.size),
        num_ibi_raw=int(pred_ibi_s.size),
        num_ibi_clean=int(pred_clean_ibi.size),
    )
    pred_irregularity = _compute_ibi_irregularity_features(pred_clean_ibi, prefix="")
    beat_quality_summary = _compute_beat_quality_summary(
        window=window,
        beat_cfg=beat_cfg,
        ibi_cfg=ibi_cfg,
        quality_cfg=quality_cfg,
        raw_beats=raw_beats,
    )
    ref_features = _compute_reference_window_features(
        ecg_peaks=ecg_peaks,
        ecg_fs=ecg_fs,
        start_time_s=float(window.start_time_s),
        duration_s=float(window.duration_s),
        ibi_cfg=ibi_cfg,
    )

    min_beats = int(stage4_shared_cfg.get("min_beats", 4))
    min_ibi_clean = int(stage4_shared_cfg.get("min_ibi_clean", 3))
    min_ref_ibi_clean = int(stage4_shared_cfg.get("min_ref_ibi_clean", 3))
    num_beats = int(raw_beats.size)
    num_ibi_clean = int(pred_clean_ibi.size)
    ref_num_ibi_clean = int(safe_float(ref_features["ref_num_ibi_clean"], default=0.0))

    row = {
        "split": str(source_row.get("split", "")),
        "dataset": str(window.dataset),
        "subject_id": str(window.subject_id),
        "window_index": int(window.window_index),
        "start_time_s": float(window.start_time_s),
        "duration_s": float(window.duration_s),
        "selected_hr_source": str(selected_row["selected_hr_source"]),
        "selected_hr_bpm": safe_float(selected_row.get("selected_hr_bpm"), default=math.nan),
        "selected_hr_is_valid": bool(selected_row.get("selected_hr_is_valid", False)),
        "selected_hr_prev_bpm": safe_float(selected_row.get("selected_hr_prev_bpm"), default=math.nan),
        "selected_hr_delta_bpm": safe_float(selected_row.get("selected_hr_delta_bpm"), default=math.nan),
        "selected_hr_missing_flag": not np.isfinite(safe_float(selected_row.get("selected_hr_bpm"), default=math.nan)),
        "window_is_valid": bool(source_row.get("window_is_valid", False)),
        "ref_hr_bpm": safe_float(source_row.get("ref_hr_bpm"), default=math.nan),
        "freq_confidence": safe_float(source_row.get("freq_confidence"), default=0.0),
        "freq_peak_ratio": safe_float(source_row.get("freq_peak_ratio"), default=0.0),
        "time_confidence": safe_float(source_row.get("time_confidence"), default=0.0),
        "time_num_peaks": safe_float(source_row.get("time_num_peaks"), default=0.0),
        "hr_agreement_bpm": safe_float(source_row.get("hr_agreement_bpm"), default=math.nan),
        "ppg_centered_std": safe_float(source_row.get("ppg_centered_std"), default=math.nan),
        "ppg_peak_to_peak": safe_float(source_row.get("ppg_peak_to_peak"), default=math.nan),
        "ppg_processed_diff_std": safe_float(source_row.get("ppg_processed_diff_std"), default=math.nan),
        "has_acc": bool(source_row.get("has_acc", False)),
        "acc_axis_std_norm": safe_float(source_row.get("acc_axis_std_norm"), default=math.nan),
        "acc_mag_range": safe_float(source_row.get("acc_mag_range"), default=math.nan),
        "motion_flag": bool(source_row.get("motion_flag", False)),
        "rule_signal_quality_score": safe_float(source_row.get("rule_signal_quality_score"), default=0.0),
        "rule_validity_flag": bool(source_row.get("rule_validity_flag", False)),
        "ml_signal_quality_score": safe_float(source_row.get("ml_signal_quality_score"), default=0.0),
        "ml_validity_flag": bool(source_row.get("ml_validity_flag", False)),
        "beat_fallback_available": bool(source_row.get("beat_fallback_available", False)),
        "beat_fallback_num_beats": safe_float(source_row.get("beat_fallback_num_beats"), default=0.0),
        "beat_fallback_num_clean_ibi": safe_float(source_row.get("beat_fallback_num_clean_ibi"), default=0.0),
        "beat_fallback_kept_ratio": safe_float(source_row.get("beat_fallback_kept_ratio"), default=math.nan),
        "beat_fallback_reason": str(source_row.get("beat_fallback_reason", "")),
        "robust_hr_source": str(source_row.get("robust_hr_source", "")),
        "robust_hr_action": str(source_row.get("robust_hr_action", "")),
        "robust_hr_is_valid": bool(source_row.get("robust_hr_is_valid", False)),
        "hold_applied": bool(source_row.get("hold_applied", False)),
        "hold_age_windows": safe_float(source_row.get("hold_age_windows"), default=0.0),
        "hr_jump_bpm_from_previous": safe_float(source_row.get("hr_jump_bpm_from_previous"), default=math.nan),
        "policy_reason_code": str(source_row.get("policy_reason_code", "")),
        "subject_boundary_reset": bool(source_row.get("subject_boundary_reset", False)),
        "num_beats": float(num_beats),
        "num_ibi_raw": float(pred_ibi_s.size),
        "num_ibi_clean": float(num_ibi_clean),
        "ibi_is_valid": bool(pred_clean.get("ibi_is_valid", False)),
        "ibi_removed_ratio": safe_float(pred_clean.get("ibi_removed_ratio"), default=math.nan),
        "insufficient_beats_flag": bool(num_beats < min_beats),
        "insufficient_clean_ibi_flag": bool(num_ibi_clean < min_ibi_clean),
        "insufficient_ref_ibi_flag": bool(ref_num_ibi_clean < min_ref_ibi_clean),
        **{name: safe_float(value, default=math.nan) for name, value in pred_prv.items()},
        **pred_irregularity,
        **beat_quality_summary,
        **ref_features,
    }
    for category in ROBUST_SOURCE_CATEGORIES:
        row[f"robust_source_is_{category}"] = float(row["robust_hr_source"] == category)
    for category in ROBUST_ACTION_CATEGORIES:
        row[f"robust_action_is_{category}"] = float(row["robust_hr_action"] == category)
    return row


def build_stage4_shared_feature_frame(
    *,
    loader,
    subject_ids: list[str],
    split_name: str,
    preprocess_cfg: dict[str, Any],
    stage3_cfg: dict[str, Any],
    stage4_shared_cfg: dict[str, Any],
    source_frame: pd.DataFrame,
) -> pd.DataFrame:
    if source_frame.empty:
        return pd.DataFrame(columns=list(STAGE4_IDENTITY_COLUMNS))

    working_source = source_frame.copy()
    working_source["split"] = split_name
    source_name = str(stage4_shared_cfg.get("selected_hr_source", "robust_stage3c2_policy"))
    selected_frame = select_stage4_signal_source(working_source, source_name=source_name, split_name=split_name)
    selected_frame = attach_previous_valid_hr_context(selected_frame)
    selected_lookup = {
        (str(row["subject_id"]), int(row["window_index"])): row
        for row in selected_frame.to_dict(orient="records")
    }
    source_lookup = {
        (str(row["subject_id"]), int(row["window_index"])): row
        for row in working_source.to_dict(orient="records")
    }

    beat_cfg, ibi_cfg, quality_cfg = _build_ibi_configs(stage3_cfg=stage3_cfg, stage4_shared_cfg=stage4_shared_cfg)
    rows: list[dict[str, float | str | bool]] = []
    for subject_id in subject_ids:
        record = trim_record_to_common_duration(loader.load_subject(subject_id))
        windows = build_window_samples(
            record=record,
            target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
            window_seconds=float(preprocess_cfg["window_seconds"]),
            step_seconds=float(preprocess_cfg["step_seconds"]),
        )
        ecg_peaks = detect_ecg_peaks(record.ecg, record.ecg_fs)
        for window in windows:
            key = (str(window.subject_id), int(window.window_index))
            if key not in source_lookup or key not in selected_lookup:
                continue
            rows.append(
                _build_stage4_feature_row(
                    window=window,
                    source_row=source_lookup[key],
                    selected_row=selected_lookup[key],
                    ecg_peaks=ecg_peaks,
                    ecg_fs=float(record.ecg_fs),
                    beat_cfg=beat_cfg,
                    ibi_cfg=ibi_cfg,
                    quality_cfg=quality_cfg,
                    stage4_shared_cfg=stage4_shared_cfg,
                )
            )
    feature_frame = pd.DataFrame(rows)
    if not feature_frame.empty:
        feature_frame = feature_frame.sort_values(by=["split", "subject_id", "window_index", "start_time_s"]).reset_index(drop=True)
    return feature_frame
