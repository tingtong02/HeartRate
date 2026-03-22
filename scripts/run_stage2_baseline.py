from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.metrics import (
    compute_ibi_error_metrics,
    compute_precision_recall_f1,
    summarize_feature_metrics,
)
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage2_beat import (
    build_analysis_windows,
    clean_ibi_series,
    compute_beat_quality_proxy,
    compute_time_domain_prv_features,
    detect_beats_in_window,
    detect_reference_beats_in_window,
    evaluate_beat_detection,
    extract_ibi_from_beats,
    extract_matched_ibi_pairs_with_indices_ms,
)


FEATURE_NAMES = [
    "num_beats",
    "num_ibi_raw",
    "num_ibi_clean",
    "mean_ibi_ms",
    "median_ibi_ms",
    "mean_hr_bpm_from_ibi",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "ibi_cv",
]

FEATURE_HIGHLIGHTS = ["mean_ibi_ms", "median_ibi_ms", "sdnn_ms", "rmssd_ms", "ibi_cv"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 2 beat/IBI/feature baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage2.yaml", help="Stage 2 eval config path.")
    return parser.parse_args()


def make_loader(dataset_name: str, root_dir: str):
    if dataset_name == "ppg_dalia":
        return PPGDaliaLoader(root_dir)
    if dataset_name == "wesad":
        return WESADLoader(root_dir)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def build_variant_configs(stage2_cfg: dict) -> tuple[dict[str, dict], dict]:
    shared_beat = {
        "bandpass_low_hz": float(stage2_cfg["beat"].get("bandpass_low_hz", 0.6)),
        "bandpass_high_hz": float(stage2_cfg["beat"].get("bandpass_high_hz", 3.5)),
        "bandpass_order": int(stage2_cfg["beat"].get("bandpass_order", 3)),
        "smooth_window_seconds": float(stage2_cfg["beat"].get("smooth_window_seconds", 0.20)),
        "smooth_polyorder": int(stage2_cfg["beat"].get("smooth_polyorder", 2)),
        "extra_smoothing": bool(stage2_cfg["beat"].get("extra_smoothing", True)),
        "hr_max_bpm": float(stage2_cfg["beat"].get("hr_max_bpm", 180.0)),
        "min_prominence": 0.05,
        "prominence_scale": 0.35,
        "min_width_seconds": 0.08,
        "refine_radius_seconds": 0.08,
        "refractory_scale": 1.5,
    }
    baseline_beat = {**shared_beat, "variant_mode": "baseline"}
    enhanced_beat = copy.deepcopy(stage2_cfg["beat"])
    enhanced_beat["variant_mode"] = "enhanced"

    shared_ibi = {
        "min_ibi_s": float(stage2_cfg["ibi"].get("min_ibi_s", 0.33)),
        "max_ibi_s": float(stage2_cfg["ibi"].get("max_ibi_s", 1.5)),
        "local_median_radius": int(stage2_cfg["ibi"].get("local_median_radius", 2)),
        "max_deviation_ratio": 0.30,
        "min_clean_ibi": int(stage2_cfg["ibi"].get("min_clean_ibi", 3)),
    }
    baseline_ibi = {**shared_ibi, "variant_mode": "baseline"}
    enhanced_ibi = copy.deepcopy(stage2_cfg["ibi"])
    enhanced_ibi["variant_mode"] = "enhanced"

    return (
        {
            "baseline": {"beat": baseline_beat, "ibi": baseline_ibi, "beat_quality": {}},
            "enhanced": {"beat": enhanced_beat, "ibi": enhanced_ibi, "beat_quality": {}},
            **(
                {
                    "enhanced_beat_quality": {
                        "beat": copy.deepcopy(enhanced_beat),
                        "ibi": copy.deepcopy(enhanced_ibi),
                        "beat_quality": copy.deepcopy(stage2_cfg.get("beat_quality", {})),
                    }
                }
                if bool(stage2_cfg.get("beat_quality", {}).get("enabled", False))
                else {}
            ),
        },
        baseline_ibi,
    )


def summarize_error_cases(beat_frame: pd.DataFrame, max_cases_per_variant: int) -> pd.DataFrame:
    error_columns = [
        "variant",
        "subject_id",
        "analysis_window_index",
        "num_pred_beats",
        "num_ref_beats",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "beat_count_error",
        "ibi_mae_ms",
        "ibi_rmse_ms",
        "pred_num_ibi_clean",
        "ref_num_ibi_clean",
    ]
    frames: list[pd.DataFrame] = []
    for variant in sorted(beat_frame["variant"].unique()):
        variant_frame = beat_frame.loc[beat_frame["variant"] == variant, error_columns].copy()
        if variant_frame.empty:
            continue
        variant_frame["sort_f1"] = variant_frame["f1"].fillna(-1.0)
        variant_frame["sort_ibi_rmse"] = variant_frame["ibi_rmse_ms"].fillna(-1.0)
        variant_frame = variant_frame.sort_values(
            by=["sort_f1", "sort_ibi_rmse"],
            ascending=[True, False],
        ).head(max_cases_per_variant)
        frames.append(variant_frame.drop(columns=["sort_f1", "sort_ibi_rmse"]))
    if not frames:
        return pd.DataFrame(columns=error_columns)
    return pd.concat(frames, ignore_index=True)


def summarize_beat_quality_proxy(beat_quality_frame: pd.DataFrame) -> pd.DataFrame:
    if beat_quality_frame.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "task",
                "metric_group",
                "num_pred_beats",
                "kept_beat_ratio",
                "good_label_ratio",
                "precision_among_kept",
                "mean_score_matched",
                "mean_score_unmatched",
            ]
        )

    rows: list[dict[str, float | str]] = []
    for variant in sorted(beat_quality_frame["variant"].unique()):
        variant_frame = beat_quality_frame.loc[beat_quality_frame["variant"] == variant].copy()
        if variant_frame.empty:
            continue
        kept_mask = variant_frame["beat_is_kept_by_quality"].astype(bool)
        matched_mask = variant_frame["beat_is_matched_to_ref"].astype(bool)
        kept_count = int(kept_mask.sum())
        rows.append(
            {
                "variant": variant,
                "task": "beat_quality_proxy",
                "metric_group": "summary",
                "num_pred_beats": float(variant_frame.shape[0]),
                "kept_beat_ratio": float(kept_count / max(variant_frame.shape[0], 1)),
                "good_label_ratio": float(np.mean(variant_frame["beat_quality_label"] == "good")),
                "precision_among_kept": float(np.mean(matched_mask[kept_mask])) if kept_count > 0 else math.nan,
                "mean_score_matched": float(variant_frame.loc[matched_mask, "beat_quality_score"].mean()) if matched_mask.any() else math.nan,
                "mean_score_unmatched": float(variant_frame.loc[~matched_mask, "beat_quality_score"].mean()) if (~matched_mask).any() else math.nan,
            }
        )
    return pd.DataFrame(rows)


def build_threshold_grid(
    min_threshold: float,
    max_threshold: float,
    step_threshold: float,
    *,
    include_thresholds: list[float] | None = None,
) -> np.ndarray:
    if step_threshold <= 0.0:
        raise ValueError("Threshold step must be positive.")
    if max_threshold < min_threshold:
        raise ValueError("Threshold range must satisfy max_threshold >= min_threshold.")

    threshold_values: set[float] = set()
    current = float(min_threshold)
    while current <= float(max_threshold) + 1e-8:
        threshold_values.add(round(current, 6))
        current += float(step_threshold)
    for value in include_thresholds or []:
        threshold_values.add(round(float(value), 6))
    return np.array(sorted(threshold_values), dtype=float)


def select_beat_quality_analysis_threshold(
    sweep_frame: pd.DataFrame,
    *,
    selection_metric: str,
    min_kept_beat_ratio: float,
) -> pd.Series:
    if sweep_frame.empty:
        raise ValueError("Cannot select a threshold from an empty sweep frame.")

    feasible = sweep_frame.loc[sweep_frame["kept_beat_ratio"] >= float(min_kept_beat_ratio)].copy()
    selected_frame = feasible if not feasible.empty else sweep_frame.copy()
    selected_frame = selected_frame.sort_values(
        by=[selection_metric, "kept_beat_ratio", "f1", "threshold"],
        ascending=[True, False, False, True],
    )
    selected = selected_frame.iloc[0].copy()
    selected["meets_min_kept_beat_ratio"] = bool(float(selected["kept_beat_ratio"]) >= float(min_kept_beat_ratio))
    return selected


def evaluate_beat_quality_threshold_records(
    records: list[dict[str, object]],
    *,
    threshold: float,
    variant_name: str,
    matching_tolerance_seconds: float,
    ibi_config: dict,
    reference_ibi_cfg: dict,
    collect_rows: bool,
) -> dict[str, object]:
    beat_rows: list[dict[str, float | str | int]] = []
    feature_rows: list[dict[str, float | str | int]] = []
    variant_state: dict[str, list[float] | int] = {
        "all_ref_ibi_ms": [],
        "all_pred_ibi_ms": [],
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }
    total_raw_beats = 0
    total_kept_beats = 0
    beat_count_errors: list[float] = []
    all_scores_matched: list[float] = []
    all_scores_unmatched: list[float] = []
    kept_scores_matched: list[float] = []

    for record in records:
        raw_pred_beats = np.asarray(record["raw_pred_beats"], dtype=int)
        ref_beats = np.asarray(record["ref_beats"], dtype=int)
        beat_quality_score = np.asarray(record["beat_quality_score"], dtype=float)
        matched_raw_flags = np.asarray(record["matched_raw_flags"], dtype=bool)
        keep_mask = beat_quality_score >= float(threshold)
        pred_beats = raw_pred_beats[keep_mask]

        total_raw_beats += int(raw_pred_beats.size)
        total_kept_beats += int(pred_beats.size)
        all_scores_matched.extend(beat_quality_score[matched_raw_flags].tolist())
        all_scores_unmatched.extend(beat_quality_score[~matched_raw_flags].tolist())
        kept_scores_matched.extend(matched_raw_flags[keep_mask].astype(float).tolist())

        beat_eval = evaluate_beat_detection(
            pred_beats,
            ref_beats,
            pred_fs=float(record["ppg_fs"]),
            ref_fs=float(record["ecg_fs"]),
            tolerance_seconds=matching_tolerance_seconds,
        )
        beat_count_errors.append(float(beat_eval["beat_count_error"]))
        variant_state["tp"] = int(variant_state["tp"]) + int(beat_eval["tp"])
        variant_state["fp"] = int(variant_state["fp"]) + int(beat_eval["fp"])
        variant_state["fn"] = int(variant_state["fn"]) + int(beat_eval["fn"])

        ref_ibi_s = extract_ibi_from_beats(ref_beats, float(record["ecg_fs"]))
        ref_clean = clean_ibi_series(ref_ibi_s, reference_ibi_cfg)
        pred_ibi_s = extract_ibi_from_beats(pred_beats, float(record["ppg_fs"]))
        pred_clean = clean_ibi_series(pred_ibi_s, ibi_config)
        ref_ibi_ms, pred_ibi_ms, ref_pair_indices, pred_pair_indices = extract_matched_ibi_pairs_with_indices_ms(
            pred_beats,
            ref_beats,
            pred_fs=float(record["ppg_fs"]),
            ref_fs=float(record["ecg_fs"]),
            tolerance_seconds=matching_tolerance_seconds,
        )
        if ref_ibi_ms.size and pred_ibi_ms.size:
            valid_pair_mask = (
                ref_clean["ibi_mask"][ref_pair_indices].astype(bool)
                & pred_clean["ibi_mask"][pred_pair_indices].astype(bool)
            )
            ref_ibi_ms = ref_ibi_ms[valid_pair_mask]
            pred_ibi_ms = pred_ibi_ms[valid_pair_mask]
        if ref_ibi_ms.size and pred_ibi_ms.size:
            variant_state["all_ref_ibi_ms"].extend(ref_ibi_ms.tolist())
            variant_state["all_pred_ibi_ms"].extend(pred_ibi_ms.tolist())
            ibi_metrics = compute_ibi_error_metrics(ref_ibi_ms, pred_ibi_ms)
        else:
            ibi_metrics = {
                "ibi_mae_ms": math.nan,
                "ibi_rmse_ms": math.nan,
                "num_valid_ibi_pairs": 0.0,
            }

        if collect_rows:
            pred_features = compute_time_domain_prv_features(
                pred_clean["ibi_clean_s"],
                num_beats=len(pred_beats),
                num_ibi_raw=len(pred_ibi_s),
                num_ibi_clean=len(pred_clean["ibi_clean_s"]),
            )
            ref_features = record["ref_features"]

            beat_rows.append(
                {
                    "variant": variant_name,
                    "dataset": record["dataset"],
                    "subject_id": record["subject_id"],
                    "analysis_window_index": record["analysis_window_index"],
                    "start_time_s": record["start_time_s"],
                    "duration_s": record["duration_s"],
                    "num_pred_beats": float(len(pred_beats)),
                    "num_ref_beats": float(len(ref_beats)),
                    "pred_num_ibi_clean": float(len(pred_clean["ibi_clean_s"])),
                    "ref_num_ibi_clean": float(len(ref_clean["ibi_clean_s"])),
                    "pred_ibi_removed_ratio": float(pred_clean["ibi_removed_ratio"]) if not math.isnan(float(pred_clean["ibi_removed_ratio"])) else math.nan,
                    "tp": float(beat_eval["tp"]),
                    "fp": float(beat_eval["fp"]),
                    "fn": float(beat_eval["fn"]),
                    "precision": float(beat_eval["precision"]) if not math.isnan(float(beat_eval["precision"])) else math.nan,
                    "recall": float(beat_eval["recall"]) if not math.isnan(float(beat_eval["recall"])) else math.nan,
                    "f1": float(beat_eval["f1"]) if not math.isnan(float(beat_eval["f1"])) else math.nan,
                    "beat_count_error": float(beat_eval["beat_count_error"]),
                    "ibi_mae_ms": float(ibi_metrics["ibi_mae_ms"]) if not math.isnan(float(ibi_metrics["ibi_mae_ms"])) else math.nan,
                    "ibi_rmse_ms": float(ibi_metrics["ibi_rmse_ms"]) if not math.isnan(float(ibi_metrics["ibi_rmse_ms"])) else math.nan,
                    "num_valid_ibi_pairs": float(ibi_metrics["num_valid_ibi_pairs"]),
                }
            )

            feature_row: dict[str, float | str | int] = {
                "variant": variant_name,
                "dataset": record["dataset"],
                "subject_id": record["subject_id"],
                "analysis_window_index": record["analysis_window_index"],
                "start_time_s": record["start_time_s"],
                "duration_s": record["duration_s"],
            }
            for feature_name in FEATURE_NAMES:
                feature_row[f"pred_{feature_name}"] = float(pred_features[feature_name])
                feature_row[f"ref_{feature_name}"] = float(ref_features[feature_name])
            feature_rows.append(feature_row)

    beat_summary = compute_precision_recall_f1(
        int(variant_state["tp"]),
        int(variant_state["fp"]),
        int(variant_state["fn"]),
    )
    ibi_summary = compute_ibi_error_metrics(
        np.asarray(variant_state["all_ref_ibi_ms"], dtype=float),
        np.asarray(variant_state["all_pred_ibi_ms"], dtype=float),
    )
    summary_row = {
        "variant_source": "enhanced_beat_quality",
        "threshold": float(threshold),
        "num_pred_beats": float(total_raw_beats),
        "kept_beat_ratio": float(total_kept_beats / max(total_raw_beats, 1)),
        "precision": float(beat_summary["precision"]) if not math.isnan(float(beat_summary["precision"])) else math.nan,
        "recall": float(beat_summary["recall"]) if not math.isnan(float(beat_summary["recall"])) else math.nan,
        "f1": float(beat_summary["f1"]) if not math.isnan(float(beat_summary["f1"])) else math.nan,
        "beat_count_error": float(np.mean(beat_count_errors)) if beat_count_errors else math.nan,
        "ibi_mae_ms": float(ibi_summary["ibi_mae_ms"]) if not math.isnan(float(ibi_summary["ibi_mae_ms"])) else math.nan,
        "ibi_rmse_ms": float(ibi_summary["ibi_rmse_ms"]) if not math.isnan(float(ibi_summary["ibi_rmse_ms"])) else math.nan,
        "num_valid_ibi_pairs": float(ibi_summary["num_valid_ibi_pairs"]),
        "mean_score_matched": float(np.mean(all_scores_matched)) if all_scores_matched else math.nan,
        "mean_score_unmatched": float(np.mean(all_scores_unmatched)) if all_scores_unmatched else math.nan,
        "good_label_ratio": float(total_kept_beats / max(total_raw_beats, 1)),
        "precision_among_kept": float(np.mean(kept_scores_matched)) if kept_scores_matched else math.nan,
    }
    return {
        "beat_rows": beat_rows,
        "feature_rows": feature_rows,
        "variant_state": variant_state,
        "summary_row": summary_row,
    }


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage2_cfg = config["stage2"]
    output_cfg = config["output"]
    variant_cfgs, reference_ibi_cfg = build_variant_configs(stage2_cfg)
    debug_cfg = stage2_cfg.get("debug", {})
    beat_quality_refine_cfg = stage2_cfg.get("beat_quality_refine", {})

    loader = make_loader(dataset_cfg["name"], dataset_cfg["root_dir"])
    subjects = loader.list_subjects()
    if dataset_cfg.get("subject_include"):
        allowed = set(dataset_cfg["subject_include"])
        subjects = [subject for subject in subjects if subject in allowed]
    if not subjects:
        raise RuntimeError("No subjects available for evaluation.")

    split = train_test_subject_split(
        subjects,
        test_size=float(eval_cfg["test_size"]),
        random_seed=int(eval_cfg["random_seed"]),
    )
    eval_subjects = split.test_subjects if split.test_subjects else split.train_subjects

    beat_rows: list[dict[str, float | str | int]] = []
    feature_rows: list[dict[str, float | str | int]] = []
    beat_quality_rows: list[dict[str, float | str | int | bool]] = []
    beat_quality_analysis_records: list[dict[str, object]] = []
    beat_quality_refined_summary_row: dict[str, float | str] | None = None
    beat_quality_threshold_selection_row: dict[str, float | str] | None = None
    beat_quality_sweep_frame = pd.DataFrame()
    variant_state: dict[str, dict[str, list[float] | int]] = {
        variant: {
            "all_ref_ibi_ms": [],
            "all_pred_ibi_ms": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for variant in variant_cfgs
    }

    for subject_id in eval_subjects:
        record = loader.load_subject(subject_id)
        windows = build_analysis_windows(
            record=record,
            target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
            analysis_window_seconds=float(stage2_cfg["analysis_window_seconds"]),
            analysis_step_seconds=float(stage2_cfg["analysis_step_seconds"]),
        )
        for window in windows:
            ref_beats = detect_reference_beats_in_window(window["ecg_window"], window["ecg_fs"])
            ref_ibi_s = extract_ibi_from_beats(ref_beats, window["ecg_fs"])
            ref_clean = clean_ibi_series(ref_ibi_s, reference_ibi_cfg)
            ref_features = compute_time_domain_prv_features(
                ref_clean["ibi_clean_s"],
                num_beats=len(ref_beats),
                num_ibi_raw=len(ref_ibi_s),
                num_ibi_clean=len(ref_clean["ibi_clean_s"]),
            )

            for variant, variant_cfg in variant_cfgs.items():
                raw_pred_beats = detect_beats_in_window(window["ppg_window"], window["ppg_fs"], variant_cfg["beat"])
                pred_beats = raw_pred_beats
                beat_quality_proxy = None
                kept_raw_indices = np.arange(raw_pred_beats.size, dtype=int)
                if bool(variant_cfg.get("beat_quality", {}).get("enabled", False)):
                    beat_quality_proxy = compute_beat_quality_proxy(
                        window["ppg_window"],
                        raw_pred_beats,
                        fs=window["ppg_fs"],
                        beat_config=variant_cfg["beat"],
                        ibi_config=variant_cfg["ibi"],
                        quality_config=variant_cfg["beat_quality"],
                    )
                    kept_raw_indices = np.flatnonzero(beat_quality_proxy["beat_is_kept_by_quality"])
                    pred_beats = raw_pred_beats[kept_raw_indices]
                beat_eval = evaluate_beat_detection(
                    pred_beats,
                    ref_beats,
                    pred_fs=window["ppg_fs"],
                    ref_fs=window["ecg_fs"],
                    tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                )
                variant_state[variant]["tp"] = int(variant_state[variant]["tp"]) + int(beat_eval["tp"])
                variant_state[variant]["fp"] = int(variant_state[variant]["fp"]) + int(beat_eval["fp"])
                variant_state[variant]["fn"] = int(variant_state[variant]["fn"]) + int(beat_eval["fn"])

                pred_ibi_s = extract_ibi_from_beats(pred_beats, window["ppg_fs"])
                pred_clean = clean_ibi_series(pred_ibi_s, variant_cfg["ibi"])
                ref_ibi_ms, pred_ibi_ms, ref_pair_indices, pred_pair_indices = extract_matched_ibi_pairs_with_indices_ms(
                    pred_beats,
                    ref_beats,
                    pred_fs=window["ppg_fs"],
                    ref_fs=window["ecg_fs"],
                    tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                )
                if ref_ibi_ms.size and pred_ibi_ms.size:
                    valid_pair_mask = (
                        ref_clean["ibi_mask"][ref_pair_indices].astype(bool)
                        & pred_clean["ibi_mask"][pred_pair_indices].astype(bool)
                    )
                    ref_ibi_ms = ref_ibi_ms[valid_pair_mask]
                    pred_ibi_ms = pred_ibi_ms[valid_pair_mask]
                    ref_pair_indices = ref_pair_indices[valid_pair_mask]
                    pred_pair_indices = pred_pair_indices[valid_pair_mask]
                if ref_ibi_ms.size and pred_ibi_ms.size:
                    variant_state[variant]["all_ref_ibi_ms"].extend(ref_ibi_ms.tolist())
                    variant_state[variant]["all_pred_ibi_ms"].extend(pred_ibi_ms.tolist())
                    ibi_metrics = compute_ibi_error_metrics(ref_ibi_ms, pred_ibi_ms)
                else:
                    ibi_metrics = {
                        "ibi_mae_ms": math.nan,
                        "ibi_rmse_ms": math.nan,
                        "num_valid_ibi_pairs": 0.0,
                    }

                pred_features = compute_time_domain_prv_features(
                    pred_clean["ibi_clean_s"],
                    num_beats=len(pred_beats),
                    num_ibi_raw=len(pred_ibi_s),
                    num_ibi_clean=len(pred_clean["ibi_clean_s"]),
                )

                beat_rows.append(
                    {
                        "variant": variant,
                        "dataset": window["dataset"],
                        "subject_id": window["subject_id"],
                        "analysis_window_index": window["analysis_window_index"],
                        "start_time_s": window["start_time_s"],
                        "duration_s": window["duration_s"],
                        "num_pred_beats": float(len(pred_beats)),
                        "num_ref_beats": float(len(ref_beats)),
                        "pred_num_ibi_clean": float(len(pred_clean["ibi_clean_s"])),
                        "ref_num_ibi_clean": float(len(ref_clean["ibi_clean_s"])),
                        "pred_ibi_removed_ratio": float(pred_clean["ibi_removed_ratio"]) if not math.isnan(float(pred_clean["ibi_removed_ratio"])) else math.nan,
                        "tp": float(beat_eval["tp"]),
                        "fp": float(beat_eval["fp"]),
                        "fn": float(beat_eval["fn"]),
                        "precision": float(beat_eval["precision"]) if not math.isnan(float(beat_eval["precision"])) else math.nan,
                        "recall": float(beat_eval["recall"]) if not math.isnan(float(beat_eval["recall"])) else math.nan,
                        "f1": float(beat_eval["f1"]) if not math.isnan(float(beat_eval["f1"])) else math.nan,
                        "beat_count_error": float(beat_eval["beat_count_error"]),
                        "ibi_mae_ms": float(ibi_metrics["ibi_mae_ms"]) if not math.isnan(float(ibi_metrics["ibi_mae_ms"])) else math.nan,
                        "ibi_rmse_ms": float(ibi_metrics["ibi_rmse_ms"]) if not math.isnan(float(ibi_metrics["ibi_rmse_ms"])) else math.nan,
                        "num_valid_ibi_pairs": float(ibi_metrics["num_valid_ibi_pairs"]),
                    }
                )

                feature_row: dict[str, float | str | int] = {
                    "variant": variant,
                    "dataset": window["dataset"],
                    "subject_id": window["subject_id"],
                    "analysis_window_index": window["analysis_window_index"],
                    "start_time_s": window["start_time_s"],
                    "duration_s": window["duration_s"],
                }
                for feature_name in FEATURE_NAMES:
                    feature_row[f"pred_{feature_name}"] = float(pred_features[feature_name])
                    feature_row[f"ref_{feature_name}"] = float(ref_features[feature_name])
                feature_rows.append(feature_row)

                if beat_quality_proxy is not None:
                    raw_eval = evaluate_beat_detection(
                        raw_pred_beats,
                        ref_beats,
                        pred_fs=window["ppg_fs"],
                        ref_fs=window["ecg_fs"],
                        tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                    )
                    matched_raw_indices = {int(pred_idx) for pred_idx, _ in raw_eval["matches"]}
                    beat_quality_analysis_records.append(
                        {
                            "dataset": window["dataset"],
                            "subject_id": window["subject_id"],
                            "analysis_window_index": window["analysis_window_index"],
                            "start_time_s": window["start_time_s"],
                            "duration_s": window["duration_s"],
                            "ppg_fs": float(window["ppg_fs"]),
                            "ecg_fs": float(window["ecg_fs"]),
                            "ref_beats": np.asarray(ref_beats, dtype=int),
                            "raw_pred_beats": np.asarray(raw_pred_beats, dtype=int),
                            "beat_quality_score": np.asarray(beat_quality_proxy["beat_quality_score"], dtype=float),
                            "matched_raw_flags": np.array(
                                [raw_index in matched_raw_indices for raw_index in range(raw_pred_beats.size)],
                                dtype=bool,
                            ),
                            "ref_features": ref_features,
                        }
                    )

                    quality_clean_pair_raw_flags = np.zeros(raw_pred_beats.size, dtype=bool)
                    pred_clean_mask = np.asarray(pred_clean["ibi_mask"], dtype=bool)
                    if pred_clean_mask.size:
                        for filtered_start_idx, keep in enumerate(pred_clean_mask):
                            if keep and filtered_start_idx < kept_raw_indices.size:
                                raw_start_idx = int(kept_raw_indices[filtered_start_idx])
                                quality_clean_pair_raw_flags[raw_start_idx] = True
                                if raw_start_idx + 1 < raw_pred_beats.size:
                                    quality_clean_pair_raw_flags[raw_start_idx + 1] = True

                    matched_clean_pair_raw_flags = np.zeros(raw_pred_beats.size, dtype=bool)
                    if pred_pair_indices.size:
                        for filtered_start_idx in pred_pair_indices:
                            if filtered_start_idx < kept_raw_indices.size:
                                raw_start_idx = int(kept_raw_indices[filtered_start_idx])
                                matched_clean_pair_raw_flags[raw_start_idx] = True
                                if raw_start_idx + 1 < raw_pred_beats.size:
                                    matched_clean_pair_raw_flags[raw_start_idx + 1] = True

                    for raw_index, beat_sample_index in enumerate(raw_pred_beats.tolist()):
                        beat_quality_rows.append(
                            {
                                "variant": variant,
                                "dataset": window["dataset"],
                                "subject_id": window["subject_id"],
                                "analysis_window_index": window["analysis_window_index"],
                                "start_time_s": window["start_time_s"],
                                "duration_s": window["duration_s"],
                                "beat_index_in_window": float(raw_index),
                                "beat_sample_index": float(beat_sample_index),
                                "beat_time_s": float(window["start_time_s"] + beat_sample_index / window["ppg_fs"]),
                                "beat_quality_score": float(beat_quality_proxy["beat_quality_score"][raw_index]),
                                "beat_quality_label": str(beat_quality_proxy["beat_quality_label"][raw_index]),
                                "beat_is_kept_by_quality": bool(beat_quality_proxy["beat_is_kept_by_quality"][raw_index]),
                                "beat_is_matched_to_ref": bool(raw_index in matched_raw_indices),
                                "beat_has_clean_pred_ibi_pair": bool(beat_quality_proxy["beat_has_clean_pred_ibi_pair"][raw_index]),
                                "beat_is_used_in_quality_clean_ibi_pair": bool(quality_clean_pair_raw_flags[raw_index]),
                                "beat_is_in_matched_clean_ibi_pair": bool(matched_clean_pair_raw_flags[raw_index]),
                                "beat_base_quality_score": float(beat_quality_proxy["beat_base_quality_score"][raw_index]),
                                "beat_ibi_plausibility_score": float(beat_quality_proxy["beat_ibi_plausibility_score"][raw_index]),
                                "beat_ibi_stability_score": float(beat_quality_proxy["beat_ibi_stability_score"][raw_index]),
                                "beat_crowding_score": float(beat_quality_proxy["beat_crowding_score"][raw_index]),
                                "prev_ibi_ms": float(beat_quality_proxy["prev_ibi_s"][raw_index] * 1000.0)
                                if not math.isnan(float(beat_quality_proxy["prev_ibi_s"][raw_index]))
                                else math.nan,
                                "next_ibi_ms": float(beat_quality_proxy["next_ibi_s"][raw_index] * 1000.0)
                                if not math.isnan(float(beat_quality_proxy["next_ibi_s"][raw_index]))
                                else math.nan,
                            }
                        )

    refined_variant_name = "enhanced_beat_quality_refined"
    if (
        bool(stage2_cfg.get("beat_quality", {}).get("enabled", False))
        and bool(beat_quality_refine_cfg.get("enabled", False))
        and beat_quality_analysis_records
    ):
        baseline_threshold = float(stage2_cfg.get("beat_quality", {}).get("good_score_threshold", 0.55))
        coarse_thresholds = build_threshold_grid(
            float(beat_quality_refine_cfg.get("coarse_min_threshold", 0.30)),
            float(beat_quality_refine_cfg.get("coarse_max_threshold", 0.75)),
            float(beat_quality_refine_cfg.get("coarse_step", 0.05)),
            include_thresholds=[baseline_threshold],
        )
        selection_metric = str(beat_quality_refine_cfg.get("selection_metric", "ibi_rmse_ms"))
        min_kept_beat_ratio = float(beat_quality_refine_cfg.get("min_kept_beat_ratio", 0.35))

        sweep_rows: list[dict[str, float | str | bool]] = []
        for threshold in coarse_thresholds.tolist():
            sweep_result = evaluate_beat_quality_threshold_records(
                beat_quality_analysis_records,
                threshold=float(threshold),
                variant_name=refined_variant_name,
                matching_tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                ibi_config=variant_cfgs["enhanced_beat_quality"]["ibi"],
                reference_ibi_cfg=reference_ibi_cfg,
                collect_rows=False,
            )
            summary_row = dict(sweep_result["summary_row"])
            summary_row["sweep_stage"] = "coarse"
            summary_row["is_baseline_threshold"] = bool(abs(float(threshold) - baseline_threshold) <= 1e-8)
            summary_row["meets_min_kept_beat_ratio"] = bool(float(summary_row["kept_beat_ratio"]) >= min_kept_beat_ratio)
            summary_row["is_feasible"] = bool(summary_row["meets_min_kept_beat_ratio"])
            sweep_rows.append(summary_row)

        coarse_sweep_frame = pd.DataFrame(sweep_rows)
        coarse_selected = select_beat_quality_analysis_threshold(
            coarse_sweep_frame,
            selection_metric=selection_metric,
            min_kept_beat_ratio=min_kept_beat_ratio,
        )

        if bool(beat_quality_refine_cfg.get("fine_enabled", True)):
            fine_thresholds = build_threshold_grid(
                float(coarse_selected["threshold"]) - float(beat_quality_refine_cfg.get("fine_radius", 0.04)),
                float(coarse_selected["threshold"]) + float(beat_quality_refine_cfg.get("fine_radius", 0.04)),
                float(beat_quality_refine_cfg.get("fine_step", 0.01)),
                include_thresholds=[baseline_threshold, float(coarse_selected["threshold"])],
            )
            seen_thresholds = {round(float(row["threshold"]), 6) for row in sweep_rows}
            for threshold in fine_thresholds.tolist():
                rounded_threshold = round(float(threshold), 6)
                if rounded_threshold in seen_thresholds:
                    continue
                sweep_result = evaluate_beat_quality_threshold_records(
                    beat_quality_analysis_records,
                    threshold=float(threshold),
                    variant_name=refined_variant_name,
                    matching_tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                    ibi_config=variant_cfgs["enhanced_beat_quality"]["ibi"],
                    reference_ibi_cfg=reference_ibi_cfg,
                    collect_rows=False,
                )
                summary_row = dict(sweep_result["summary_row"])
                summary_row["sweep_stage"] = "fine"
                summary_row["is_baseline_threshold"] = bool(abs(float(threshold) - baseline_threshold) <= 1e-8)
                summary_row["meets_min_kept_beat_ratio"] = bool(float(summary_row["kept_beat_ratio"]) >= min_kept_beat_ratio)
                summary_row["is_feasible"] = bool(summary_row["meets_min_kept_beat_ratio"])
                sweep_rows.append(summary_row)
                seen_thresholds.add(rounded_threshold)

        beat_quality_sweep_frame = pd.DataFrame(sweep_rows).sort_values(
            by=["threshold", "sweep_stage"],
            ascending=[True, True],
        ).reset_index(drop=True)
        selected_row = select_beat_quality_analysis_threshold(
            beat_quality_sweep_frame,
            selection_metric=selection_metric,
            min_kept_beat_ratio=min_kept_beat_ratio,
        )
        selected_threshold = float(selected_row["threshold"])
        refined_result = evaluate_beat_quality_threshold_records(
            beat_quality_analysis_records,
            threshold=selected_threshold,
            variant_name=refined_variant_name,
            matching_tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
            ibi_config=variant_cfgs["enhanced_beat_quality"]["ibi"],
            reference_ibi_cfg=reference_ibi_cfg,
            collect_rows=True,
        )
        beat_rows.extend(refined_result["beat_rows"])
        feature_rows.extend(refined_result["feature_rows"])
        variant_state[refined_variant_name] = refined_result["variant_state"]

        beat_quality_refined_summary_row = {
            "variant": refined_variant_name,
            "task": "beat_quality_proxy",
            "metric_group": "summary",
            "num_pred_beats": float(selected_row["num_pred_beats"]),
            "kept_beat_ratio": float(selected_row["kept_beat_ratio"]),
            "good_label_ratio": float(selected_row["good_label_ratio"]),
            "precision_among_kept": float(selected_row["precision_among_kept"]) if not math.isnan(float(selected_row["precision_among_kept"])) else math.nan,
            "mean_score_matched": float(selected_row["mean_score_matched"]) if not math.isnan(float(selected_row["mean_score_matched"])) else math.nan,
            "mean_score_unmatched": float(selected_row["mean_score_unmatched"]) if not math.isnan(float(selected_row["mean_score_unmatched"])) else math.nan,
        }
        beat_quality_threshold_selection_row = {
            "variant": refined_variant_name,
            "task": "beat_quality_threshold_selection",
            "metric_group": "summary",
            "baseline_threshold": baseline_threshold,
            "selected_threshold": selected_threshold,
            "selection_metric": selection_metric,
            "selection_constraint": f"analysis_only|min_kept_beat_ratio>={min_kept_beat_ratio:.2f}",
            "selected_kept_beat_ratio": float(selected_row["kept_beat_ratio"]),
            "operating_point_role": "analysis_only",
        }

    beat_frame = pd.DataFrame(beat_rows)
    feature_frame = pd.DataFrame(feature_rows)
    beat_quality_frame = pd.DataFrame(beat_quality_rows)

    metrics_parts: list[pd.DataFrame] = []
    variant_order = list(variant_cfgs.keys())
    if refined_variant_name in variant_state:
        variant_order.append(refined_variant_name)
    for variant in variant_order:
        variant_beat_frame = beat_frame.loc[beat_frame["variant"] == variant].copy()
        variant_feature_frame = feature_frame.loc[feature_frame["variant"] == variant].copy()
        beat_summary = compute_precision_recall_f1(
            int(variant_state[variant]["tp"]),
            int(variant_state[variant]["fp"]),
            int(variant_state[variant]["fn"]),
        )
        beat_summary_row = {
            "variant": variant,
            "task": "beat_detection",
            "metric_group": "summary",
            "precision": beat_summary["precision"],
            "recall": beat_summary["recall"],
            "f1": beat_summary["f1"],
            "beat_count_error": float(variant_beat_frame["beat_count_error"].mean()) if not variant_beat_frame.empty else math.nan,
        }
        ibi_summary = compute_ibi_error_metrics(
            np.asarray(variant_state[variant]["all_ref_ibi_ms"], dtype=float),
            np.asarray(variant_state[variant]["all_pred_ibi_ms"], dtype=float),
        )
        ibi_summary_row = {
            "variant": variant,
            "task": "ibi_error",
            "metric_group": "summary",
            "ibi_mae_ms": ibi_summary["ibi_mae_ms"],
            "ibi_rmse_ms": ibi_summary["ibi_rmse_ms"],
            "num_valid_ibi_pairs": ibi_summary["num_valid_ibi_pairs"],
        }
        feature_summary = summarize_feature_metrics(
            variant_feature_frame,
            feature_names=FEATURE_NAMES,
            ref_prefix="ref_",
            pred_prefix="pred_",
        )
        feature_summary["variant"] = variant
        feature_summary["task"] = "feature_comparison"
        feature_summary["metric_group"] = "summary"
        metrics_parts.extend([pd.DataFrame([beat_summary_row]), pd.DataFrame([ibi_summary_row]), feature_summary])

    beat_quality_summary = summarize_beat_quality_proxy(beat_quality_frame)
    if not beat_quality_summary.empty:
        metrics_parts.append(beat_quality_summary)
    if beat_quality_refined_summary_row is not None:
        metrics_parts.append(pd.DataFrame([beat_quality_refined_summary_row]))
    if beat_quality_threshold_selection_row is not None:
        metrics_parts.append(pd.DataFrame([beat_quality_threshold_selection_row]))

    metrics_frame = pd.concat(metrics_parts, ignore_index=True, sort=False)

    print("Stage 2 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Subjects evaluated: {len(eval_subjects)}")
    analysis_window_count = 0
    if variant_order:
        analysis_window_count = int(beat_frame.loc[beat_frame["variant"] == variant_order[0]].shape[0])
    print(f"Analysis windows: {analysis_window_count}")
    for variant in variant_order:
        variant_metrics = metrics_frame.loc[
            (metrics_frame["variant"] == variant) & (metrics_frame["metric_group"] == "summary")
        ].copy()
        beat_summary_row = variant_metrics.loc[variant_metrics["task"] == "beat_detection"].iloc[0].to_dict()
        ibi_summary_row = variant_metrics.loc[variant_metrics["task"] == "ibi_error"].iloc[0].to_dict()
        print(f"variant: {variant}")
        print("  task: beat_detection")
        for key in ("precision", "recall", "f1", "beat_count_error"):
            value = beat_summary_row[key]
            print(f"    {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"    {key}: {value}")
        print("  task: ibi_error")
        for key in ("ibi_mae_ms", "ibi_rmse_ms", "num_valid_ibi_pairs"):
            value = ibi_summary_row[key]
            print(f"    {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"    {key}: {value}")
        feature_subset = variant_metrics.loc[
            (variant_metrics["task"] == "feature_comparison")
            & (variant_metrics["feature"].isin(FEATURE_HIGHLIGHTS))
        ]
        print("  task: feature_comparison")
        for _, row in feature_subset.iterrows():
            mae = row["mae"]
            pearson_r = row["pearson_r"]
            print(
                f"    {row['feature']}: mae={mae:.4f}, pearson_r={pearson_r:.4f}"
                if isinstance(mae, float) and not math.isnan(mae)
                else f"    {row['feature']}: mae={mae}, pearson_r={pearson_r}"
            )
        quality_subset = variant_metrics.loc[variant_metrics["task"] == "beat_quality_proxy"]
        if not quality_subset.empty:
            quality_row = quality_subset.iloc[0].to_dict()
            print("  task: beat_quality_proxy")
            for key in ("num_pred_beats", "kept_beat_ratio", "good_label_ratio", "precision_among_kept", "mean_score_matched", "mean_score_unmatched"):
                value = quality_row.get(key, math.nan)
                print(f"    {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"    {key}: {value}")
        threshold_subset = variant_metrics.loc[variant_metrics["task"] == "beat_quality_threshold_selection"]
        if not threshold_subset.empty:
            threshold_row = threshold_subset.iloc[0].to_dict()
            print("  task: beat_quality_threshold_selection")
            for key in ("baseline_threshold", "selected_threshold", "selection_metric", "selection_constraint", "selected_kept_beat_ratio", "operating_point_role"):
                value = threshold_row.get(key, math.nan)
                print(f"    {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"    {key}: {value}")

    save_csv = bool(output_cfg.get("save_csv", False))
    save_error_cases = bool(debug_cfg.get("save_error_cases", False))
    if save_csv or save_error_cases:
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

    if save_csv:
        beats_path = output_dir / f"{dataset_cfg['name']}_stage2_beats.csv"
        beat_quality_path = output_dir / f"{dataset_cfg['name']}_stage2_beat_quality.csv"
        beat_quality_sweep_path = output_dir / f"{dataset_cfg['name']}_stage2_beat_quality_sweep.csv"
        features_path = output_dir / f"{dataset_cfg['name']}_stage2_features.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage2_metrics.csv"
        beat_frame.to_csv(beats_path, index=False)
        beat_quality_frame.to_csv(beat_quality_path, index=False)
        if not beat_quality_sweep_frame.empty:
            beat_quality_sweep_frame.to_csv(beat_quality_sweep_path, index=False)
        feature_frame.to_csv(features_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved beats to: {beats_path}")
        print(f"Saved beat quality to: {beat_quality_path}")
        if not beat_quality_sweep_frame.empty:
            print(f"Saved beat quality sweep to: {beat_quality_sweep_path}")
        print(f"Saved features to: {features_path}")
        print(f"Saved metrics to: {metrics_path}")

    if save_error_cases:
        error_frame = summarize_error_cases(beat_frame, max_cases_per_variant=int(debug_cfg.get("max_cases_per_variant", 10)))
        error_path = output_dir / f"{dataset_cfg['name']}_stage2_error_cases.csv"
        error_frame.to_csv(error_path, index=False)
        print(f"Saved error cases to: {error_path}")


if __name__ == "__main__":
    main()
