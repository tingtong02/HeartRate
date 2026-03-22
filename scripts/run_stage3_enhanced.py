from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.metrics import compute_hr_metrics
from heart_rate_cnn.preprocess import build_window_samples, dwt_denoise_ppg
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage1_hr import (
    estimate_hr_frequency_stage1,
    estimate_hr_time_stage1,
    fuse_hr_estimates,
)
from heart_rate_cnn.stage3_quality import (
    apply_robust_hr_policy_sequence,
    apply_ml_quality_decision,
    apply_motion_aware_quality_decision,
    apply_rule_based_quality_decision,
    build_refined_threshold_grid,
    build_quality_target,
    compute_local_beat_fallback_hr,
    compute_binary_classification_summary,
    extract_quality_features,
    evaluate_ml_threshold_grid,
    fit_quality_logistic_regression,
    predict_quality_logistic_regression,
    summarize_robust_hr_policy_behavior,
    summarize_operating_point_status,
    summarize_threshold_selection,
)
from heart_rate_cnn.types import WindowSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 3 enhancement-round ML comparison.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage3_enhanced.yaml", help="Stage 3 enhanced eval config path.")
    return parser.parse_args()


def make_loader(dataset_name: str, root_dir: str):
    if dataset_name == "ppg_dalia":
        return PPGDaliaLoader(root_dir)
    if dataset_name == "wesad":
        return WESADLoader(root_dir)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _make_analysis_window(
    window: WindowSample,
    *,
    denoise_cfg: dict | None = None,
) -> WindowSample:
    if not denoise_cfg or not bool(denoise_cfg.get("enabled", False)):
        return window
    if str(denoise_cfg.get("method", "dwt")) != "dwt":
        raise ValueError(f"Unsupported denoising method: {denoise_cfg.get('method')}")

    denoised_ppg = dwt_denoise_ppg(
        window.ppg,
        wavelet=str(denoise_cfg.get("wavelet", "db4")),
        max_level=int(denoise_cfg.get("max_level", 4)),
        threshold_mode=str(denoise_cfg.get("threshold_mode", "soft")),
        threshold_scale=float(denoise_cfg.get("threshold_scale", 1.0)),
    )
    return replace(window, ppg=denoised_ppg)


def _build_stage3_rows(
    *,
    loader,
    subject_ids: list[str],
    preprocess_cfg: dict,
    eval_cfg: dict,
    stage1_cfg: dict,
    stage3_cfg: dict,
    branch: str = "raw",
    denoise_cfg: dict | None = None,
) -> list[dict[str, float | int | str | bool | None]]:
    hr_band_bpm = tuple(float(value) for value in eval_cfg["hr_band_bpm"])
    rows: list[dict[str, float | int | str | bool | None]] = []
    for subject_id in subject_ids:
        record = loader.load_subject(subject_id)
        windows = build_window_samples(
            record=record,
            target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
            window_seconds=float(preprocess_cfg["window_seconds"]),
            step_seconds=float(preprocess_cfg["step_seconds"]),
        )
        for window in windows:
            analysis_window = _make_analysis_window(window, denoise_cfg=denoise_cfg)
            freq_result = estimate_hr_frequency_stage1(analysis_window.ppg, analysis_window.ppg_fs, hr_band_bpm, stage1_cfg["frequency"])
            time_result = estimate_hr_time_stage1(analysis_window.ppg, analysis_window.ppg_fs, hr_band_bpm, stage1_cfg["time"])
            fusion_result = fuse_hr_estimates(
                freq_result,
                time_result,
                agreement_threshold_bpm=float(stage1_cfg["fusion"]["agreement_threshold_bpm"]),
                conflict_threshold_bpm=float(stage1_cfg["fusion"]["conflict_threshold_bpm"]),
            )
            feature_row = extract_quality_features(
                analysis_window,
                freq_result=freq_result,
                time_result=time_result,
                fusion_result=fusion_result,
                preprocess_config=stage1_cfg["frequency"],
                motion_config=stage3_cfg.get("motion", {}),
            )
            target_row = build_quality_target(
                ref_hr_bpm=analysis_window.ref_hr_bpm,
                freq_pred_hr_bpm=float(freq_result["freq_pred_hr_bpm"]),
                window_is_valid=bool(analysis_window.is_valid),
                freq_is_valid=bool(freq_result["freq_is_valid"]),
                good_error_bpm=float(stage3_cfg["target"]["good_error_bpm"]),
                poor_error_bpm=float(stage3_cfg["target"]["poor_error_bpm"]),
            )
            rule_row = apply_rule_based_quality_decision(
                window_is_valid=bool(analysis_window.is_valid),
                features=feature_row,
                config=stage3_cfg["rule"],
            )
            robust_policy_cfg = stage3_cfg.get("robust_hr_policy", {})
            beat_fallback_row = compute_local_beat_fallback_hr(
                analysis_window,
                config=robust_policy_cfg,
            )
            rows.append(
                {
                    "branch": branch,
                    "dataset": analysis_window.dataset,
                    "subject_id": analysis_window.subject_id,
                    "window_index": analysis_window.window_index,
                    "start_time_s": analysis_window.start_time_s,
                    "duration_s": analysis_window.duration_s,
                    "ref_hr_bpm": analysis_window.ref_hr_bpm,
                    "window_is_valid": bool(analysis_window.is_valid),
                    **feature_row,
                    **target_row,
                    "rule_signal_quality_score": float(rule_row["signal_quality_score"]),
                    "rule_signal_quality_label": str(rule_row["signal_quality_label"]),
                    "rule_validity_flag": bool(rule_row["validity_flag"]),
                    "motion_flag": bool(rule_row["motion_flag"]),
                    "ungated_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"]),
                    "ungated_is_valid": bool(analysis_window.is_valid and freq_result["freq_is_valid"]),
                    "rule_gated_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"])
                    if bool(rule_row["validity_flag"])
                    else math.nan,
                    "rule_gated_is_valid": bool(analysis_window.is_valid and rule_row["validity_flag"]),
                    **beat_fallback_row,
                }
            )
    return rows


def _summarize_hr_method(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    valid_col: str,
    method: str,
    ungated_valid_count: int,
    selected_threshold: float | None = None,
) -> dict[str, float | str]:
    valid_mask = frame["ref_hr_bpm"].notna() & frame[pred_col].notna() & frame[valid_col].astype(bool)
    valid_frame = frame.loc[valid_mask]
    metrics = compute_hr_metrics(
        valid_frame["ref_hr_bpm"].to_numpy(dtype=float),
        valid_frame[pred_col].to_numpy(dtype=float),
    )
    retention_ratio = float(valid_frame.shape[0] / ungated_valid_count) if ungated_valid_count > 0 else math.nan
    return {
        "task": "hr_comparison",
        "method": method,
        "accuracy": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "f1": math.nan,
        "num_eval_windows": math.nan,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mape": metrics["mape"],
        "pearson_r": metrics["pearson_r"],
        "num_valid_windows": metrics["num_valid_windows"],
        "retention_ratio": retention_ratio,
        "selected_threshold": selected_threshold if selected_threshold is not None else math.nan,
    }


def _summarize_quality_method(
    frame: pd.DataFrame,
    *,
    label_col: str,
    method: str,
    selected_threshold: float | None = None,
) -> dict[str, float | str]:
    classification_frame = frame.loc[frame["quality_target_label"].isin(["good", "poor"])].copy()
    metrics = compute_binary_classification_summary(
        classification_frame["quality_target_label"].tolist(),
        classification_frame[label_col].tolist(),
    )
    return {
        "task": "quality_classification",
        "method": method,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "num_eval_windows": metrics["num_eval_windows"],
        "mae": math.nan,
        "rmse": math.nan,
        "mape": math.nan,
        "pearson_r": math.nan,
        "num_valid_windows": math.nan,
        "retention_ratio": math.nan,
        "selected_threshold": selected_threshold if selected_threshold is not None else math.nan,
    }


def _ensure_window_alignment(reference_frame: pd.DataFrame, branch_frame: pd.DataFrame) -> None:
    key_cols = ["dataset", "subject_id", "window_index", "start_time_s", "duration_s"]
    if reference_frame.shape[0] != branch_frame.shape[0]:
        raise ValueError("Branch frames must contain the same number of windows.")
    for key in key_cols:
        if not reference_frame[key].equals(branch_frame[key]):
            raise ValueError(f"Branch frames are not aligned on {key}.")


def _build_operating_points_frame(
    *,
    branch: str,
    selected_threshold: float,
    selection_mode: str,
    retention_floor: float,
    final_train_summary: dict[str, float],
    eval_selected_summary: dict[str, float],
    test_rule_mae: float,
    fine_status: dict[str, float | str],
    best_train_rmse_row: dict[str, float],
    best_eval_rule_tradeoff: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "branch": branch,
                "summary_type": "selected_operating_point",
                "threshold": selected_threshold,
                "split_used": "train_select",
                "selection_mode": selection_mode,
                "retention_floor": retention_floor,
                "train_retention_ratio": float(final_train_summary["retention_ratio"]),
                "train_mae": float(final_train_summary["mae"]),
                "train_f1": float(final_train_summary["f1"]),
                "test_retention_ratio": float(eval_selected_summary.get("retention_ratio", math.nan)),
                "test_mae": float(eval_selected_summary.get("mae", math.nan)),
                "test_rmse": float(eval_selected_summary.get("rmse", math.nan)),
                "test_rule_mae": test_rule_mae,
                "stable_threshold_count": float(fine_status["stable_threshold_count"]),
                "selected_threshold_rank": float(fine_status["selected_threshold_rank"]),
                "operating_point_status": str(fine_status["operating_point_status"]),
            },
            {
                "branch": branch,
                "summary_type": "best_train_mae_feasible",
                "threshold": float(final_train_summary["selected_threshold"]),
                "split_used": "train_select",
                "selection_mode": "mae",
                "retention_floor": retention_floor,
                "train_retention_ratio": float(final_train_summary["retention_ratio"]),
                "train_mae": float(final_train_summary["mae"]),
                "train_f1": float(final_train_summary["f1"]),
                "test_retention_ratio": math.nan,
                "test_mae": math.nan,
                "test_rmse": math.nan,
                "test_rule_mae": test_rule_mae,
                "stable_threshold_count": math.nan,
                "selected_threshold_rank": math.nan,
                "operating_point_status": "reference",
            },
            {
                "branch": branch,
                "summary_type": "best_train_rmse_feasible",
                "threshold": float(best_train_rmse_row["threshold"]),
                "split_used": "train_select",
                "selection_mode": "rmse",
                "retention_floor": retention_floor,
                "train_retention_ratio": float(best_train_rmse_row["retention_ratio"]),
                "train_mae": float(best_train_rmse_row["mae"]),
                "train_f1": float(best_train_rmse_row["f1"]) if not math.isnan(float(best_train_rmse_row["f1"])) else math.nan,
                "test_retention_ratio": math.nan,
                "test_mae": math.nan,
                "test_rmse": math.nan,
                "test_rule_mae": test_rule_mae,
                "stable_threshold_count": math.nan,
                "selected_threshold_rank": math.nan,
                "operating_point_status": "reference",
            },
            {
                "branch": branch,
                "summary_type": "highest_retention_test_beating_rule_mae",
                "threshold": float(best_eval_rule_tradeoff.get("threshold", math.nan)),
                "split_used": "test_report",
                "selection_mode": "report_only",
                "retention_floor": retention_floor,
                "train_retention_ratio": math.nan,
                "train_mae": math.nan,
                "train_f1": math.nan,
                "test_retention_ratio": float(best_eval_rule_tradeoff.get("retention_ratio", math.nan)),
                "test_mae": float(best_eval_rule_tradeoff.get("mae", math.nan)),
                "test_rmse": float(best_eval_rule_tradeoff.get("rmse", math.nan)),
                "test_rule_mae": test_rule_mae,
                "stable_threshold_count": math.nan,
                "selected_threshold_rank": math.nan,
                "operating_point_status": "report_only",
            },
        ]
    )


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage1_cfg = config["stage1"]
    stage3_cfg = config["stage3"]
    output_cfg = config["output"]
    denoise_cfg = stage3_cfg.get("denoise", {})

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
    train_subjects = split.train_subjects
    eval_subjects = split.test_subjects if split.test_subjects else split.train_subjects

    train_frame = pd.DataFrame(
        _build_stage3_rows(
            loader=loader,
            subject_ids=train_subjects,
            preprocess_cfg=preprocess_cfg,
            eval_cfg=eval_cfg,
            stage1_cfg=stage1_cfg,
            stage3_cfg=stage3_cfg,
            branch="raw",
        )
    )
    eval_frame = pd.DataFrame(
        _build_stage3_rows(
            loader=loader,
            subject_ids=eval_subjects,
            preprocess_cfg=preprocess_cfg,
            eval_cfg=eval_cfg,
            stage1_cfg=stage1_cfg,
            stage3_cfg=stage3_cfg,
            branch="raw",
        )
    )

    dwt_train_frame = pd.DataFrame()
    dwt_eval_frame = pd.DataFrame()
    if bool(denoise_cfg.get("enabled", False)):
        dwt_train_frame = pd.DataFrame(
            _build_stage3_rows(
                loader=loader,
                subject_ids=train_subjects,
                preprocess_cfg=preprocess_cfg,
                eval_cfg=eval_cfg,
                stage1_cfg=stage1_cfg,
                stage3_cfg=stage3_cfg,
                branch="dwt",
                denoise_cfg=denoise_cfg,
            )
        )
        dwt_eval_frame = pd.DataFrame(
            _build_stage3_rows(
                loader=loader,
                subject_ids=eval_subjects,
                preprocess_cfg=preprocess_cfg,
                eval_cfg=eval_cfg,
                stage1_cfg=stage1_cfg,
                stage3_cfg=stage3_cfg,
                branch="dwt",
                denoise_cfg=denoise_cfg,
            )
        )
        _ensure_window_alignment(eval_frame, dwt_eval_frame)

    ml_train_frame = train_frame.loc[train_frame["quality_target_label"].isin(["good", "poor"])].copy()
    train_feature_rows = ml_train_frame.to_dict(orient="records")
    train_target_labels = ml_train_frame["quality_target_label"].tolist()
    ml_model = fit_quality_logistic_regression(
        train_feature_rows,
        train_target_labels,
        random_seed=int(eval_cfg["random_seed"]),
        c_value=float(stage3_cfg["ml"]["c_value"]),
        max_iter=int(stage3_cfg["ml"]["max_iter"]),
    )

    train_scores = predict_quality_logistic_regression(ml_model, train_frame.to_dict(orient="records"))
    eval_scores = predict_quality_logistic_regression(ml_model, eval_frame.to_dict(orient="records"))
    train_frame["ml_signal_quality_score"] = train_scores
    eval_frame["ml_signal_quality_score"] = eval_scores

    coarse_grid = [float(value) for value in stage3_cfg["ml"]["threshold_grid"]]
    coarse_train_sweep = evaluate_ml_threshold_grid(
        train_frame,
        score_col="ml_signal_quality_score",
        pred_col="ungated_pred_hr_bpm",
        valid_col="ungated_is_valid",
        threshold_grid=coarse_grid,
        min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
        split_name="train_select",
        sweep_stage="coarse",
    )
    coarse_threshold_summary = summarize_threshold_selection(coarse_train_sweep)
    selected_threshold = float(coarse_threshold_summary["selected_threshold"])

    fine_train_sweep = pd.DataFrame(columns=coarse_train_sweep.columns)
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
            min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
            split_name="train_select",
            sweep_stage="fine",
        )
        fine_threshold_summary = summarize_threshold_selection(fine_train_sweep)
        selected_threshold = float(fine_threshold_summary["selected_threshold"])

    eval_report_grid = sorted(
        set(coarse_grid) | set(fine_train_sweep["threshold"].tolist() if not fine_train_sweep.empty else [])
    )
    eval_report_sweep = evaluate_ml_threshold_grid(
        eval_frame,
        score_col="ml_signal_quality_score",
        pred_col="ungated_pred_hr_bpm",
        valid_col="ungated_is_valid",
        threshold_grid=eval_report_grid,
        min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
        split_name="test_report",
        sweep_stage="report",
    )
    threshold_sweep_frame = pd.concat(
        [
            coarse_train_sweep.assign(branch="raw_ml"),
            fine_train_sweep.assign(branch="raw_ml"),
            eval_report_sweep.assign(branch="raw_ml"),
        ],
        ignore_index=True,
        sort=False,
    )

    ml_decisions = [
        apply_ml_quality_decision(
            signal_quality_score=float(score),
            threshold=selected_threshold,
            window_is_valid=bool(row["window_is_valid"]),
            freq_is_valid=bool(row["freq_is_valid"]),
            motion_flag=bool(row["motion_flag"]),
        )
        for score, row in zip(eval_frame["ml_signal_quality_score"].tolist(), eval_frame.to_dict(orient="records"))
    ]
    eval_frame["ml_signal_quality_label"] = [row["signal_quality_label"] for row in ml_decisions]
    eval_frame["ml_validity_flag"] = [row["validity_flag"] for row in ml_decisions]
    eval_frame["ml_gated_pred_hr_bpm"] = [
        float(pred_hr) if bool(validity_flag) else math.nan
        for pred_hr, validity_flag in zip(eval_frame["ungated_pred_hr_bpm"].tolist(), eval_frame["ml_validity_flag"].tolist())
    ]
    eval_frame["ml_gated_is_valid"] = [
        bool(validity_flag and ungated_valid)
        for validity_flag, ungated_valid in zip(eval_frame["ml_validity_flag"].tolist(), eval_frame["ungated_is_valid"].tolist())
    ]

    motion_refined_decisions = [
        apply_motion_aware_quality_decision(
            base_signal_quality_score=float(score),
            window_is_valid=bool(row["window_is_valid"]),
            freq_is_valid=bool(row["freq_is_valid"]),
            features=row,
            config=stage3_cfg.get("motion_refine", {}),
            fallback_threshold=selected_threshold,
        )
        for score, row in zip(eval_frame["ml_signal_quality_score"].tolist(), eval_frame.to_dict(orient="records"))
    ]
    eval_frame["motion_aux_score"] = [float(row["motion_aux_score"]) for row in motion_refined_decisions]
    eval_frame["motion_refined_quality_score"] = [
        float(row["motion_refined_quality_score"]) for row in motion_refined_decisions
    ]
    eval_frame["motion_refined_quality_label"] = [
        str(row["motion_refined_quality_label"]) for row in motion_refined_decisions
    ]
    eval_frame["motion_refined_validity_flag"] = [
        bool(row["motion_refined_validity_flag"]) for row in motion_refined_decisions
    ]
    motion_refined_threshold = float(motion_refined_decisions[0]["quality_threshold"]) if motion_refined_decisions else selected_threshold
    eval_frame["motion_refined_gated_pred_hr_bpm"] = [
        float(pred_hr) if bool(validity_flag) else math.nan
        for pred_hr, validity_flag in zip(
            eval_frame["ungated_pred_hr_bpm"].tolist(),
            eval_frame["motion_refined_validity_flag"].tolist(),
        )
    ]
    eval_frame["motion_refined_gated_is_valid"] = [
        bool(validity_flag and ungated_valid)
        for validity_flag, ungated_valid in zip(
            eval_frame["motion_refined_validity_flag"].tolist(),
            eval_frame["ungated_is_valid"].tolist(),
        )
    ]

    robust_policy_frame = apply_robust_hr_policy_sequence(
        eval_frame,
        config=stage3_cfg.get("robust_hr_policy", {}),
    )
    for column in robust_policy_frame.columns:
        eval_frame[column] = robust_policy_frame[column].tolist()

    rule_test_mask = (
        eval_frame["ref_hr_bpm"].notna()
        & eval_frame["rule_gated_pred_hr_bpm"].notna()
        & eval_frame["rule_gated_is_valid"].astype(bool)
    )
    rule_test_frame = eval_frame.loc[rule_test_mask]
    test_rule_mae = compute_hr_metrics(
        rule_test_frame["ref_hr_bpm"].to_numpy(dtype=float),
        rule_test_frame["rule_gated_pred_hr_bpm"].to_numpy(dtype=float),
    )["mae"]

    dwt_selected_threshold = math.nan
    dwt_threshold_sweep_frame = pd.DataFrame()
    dwt_operating_points_frame = pd.DataFrame()
    if not dwt_train_frame.empty and not dwt_eval_frame.empty:
        dwt_ml_train_frame = dwt_train_frame.loc[dwt_train_frame["quality_target_label"].isin(["good", "poor"])].copy()
        dwt_model = fit_quality_logistic_regression(
            dwt_ml_train_frame.to_dict(orient="records"),
            dwt_ml_train_frame["quality_target_label"].tolist(),
            random_seed=int(eval_cfg["random_seed"]),
            c_value=float(stage3_cfg["ml"]["c_value"]),
            max_iter=int(stage3_cfg["ml"]["max_iter"]),
        )
        dwt_train_scores = predict_quality_logistic_regression(dwt_model, dwt_train_frame.to_dict(orient="records"))
        dwt_eval_scores = predict_quality_logistic_regression(dwt_model, dwt_eval_frame.to_dict(orient="records"))
        dwt_train_frame["dwt_ml_signal_quality_score"] = dwt_train_scores
        dwt_eval_frame["dwt_ml_signal_quality_score"] = dwt_eval_scores

        dwt_coarse_train_sweep = evaluate_ml_threshold_grid(
            dwt_train_frame,
            score_col="dwt_ml_signal_quality_score",
            pred_col="ungated_pred_hr_bpm",
            valid_col="ungated_is_valid",
            threshold_grid=coarse_grid,
            min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
            split_name="train_select",
            sweep_stage="coarse",
        )
        dwt_coarse_summary = summarize_threshold_selection(dwt_coarse_train_sweep)
        dwt_selected_threshold = float(dwt_coarse_summary["selected_threshold"])

        dwt_fine_train_sweep = pd.DataFrame(columns=dwt_coarse_train_sweep.columns)
        if bool(stage3_cfg["ml"].get("refine_threshold", True)):
            dwt_fine_grid = build_refined_threshold_grid(
                center_threshold=dwt_selected_threshold,
                refinement_radius=float(stage3_cfg["ml"].get("refinement_radius", 0.10)),
                refinement_step=float(stage3_cfg["ml"].get("refinement_step", 0.02)),
            )
            dwt_fine_train_sweep = evaluate_ml_threshold_grid(
                dwt_train_frame,
                score_col="dwt_ml_signal_quality_score",
                pred_col="ungated_pred_hr_bpm",
                valid_col="ungated_is_valid",
                threshold_grid=dwt_fine_grid,
                min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
                split_name="train_select",
                sweep_stage="fine",
            )
            dwt_fine_summary = summarize_threshold_selection(dwt_fine_train_sweep)
            dwt_selected_threshold = float(dwt_fine_summary["selected_threshold"])

        dwt_eval_report_grid = sorted(
            set(coarse_grid) | set(dwt_fine_train_sweep["threshold"].tolist() if not dwt_fine_train_sweep.empty else [])
        )
        dwt_eval_report_sweep = evaluate_ml_threshold_grid(
            dwt_eval_frame,
            score_col="dwt_ml_signal_quality_score",
            pred_col="ungated_pred_hr_bpm",
            valid_col="ungated_is_valid",
            threshold_grid=dwt_eval_report_grid,
            min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
            split_name="test_report",
            sweep_stage="report",
        )
        dwt_threshold_sweep_frame = pd.concat(
            [
                dwt_coarse_train_sweep.assign(branch="dwt_ml"),
                dwt_fine_train_sweep.assign(branch="dwt_ml"),
                dwt_eval_report_sweep.assign(branch="dwt_ml"),
            ],
            ignore_index=True,
            sort=False,
        )

        dwt_decisions = [
            apply_ml_quality_decision(
                signal_quality_score=float(score),
                threshold=dwt_selected_threshold,
                window_is_valid=bool(row["window_is_valid"]),
                freq_is_valid=bool(row["freq_is_valid"]),
                motion_flag=bool(row["motion_flag"]),
            )
            for score, row in zip(dwt_eval_frame["dwt_ml_signal_quality_score"].tolist(), dwt_eval_frame.to_dict(orient="records"))
        ]
        dwt_eval_frame["dwt_ml_signal_quality_label"] = [str(row["signal_quality_label"]) for row in dwt_decisions]
        dwt_eval_frame["dwt_ml_validity_flag"] = [bool(row["validity_flag"]) for row in dwt_decisions]
        dwt_eval_frame["dwt_ml_gated_pred_hr_bpm"] = [
            float(pred_hr) if bool(validity_flag) else math.nan
            for pred_hr, validity_flag in zip(
                dwt_eval_frame["ungated_pred_hr_bpm"].tolist(),
                dwt_eval_frame["dwt_ml_validity_flag"].tolist(),
            )
        ]
        dwt_eval_frame["dwt_ml_gated_is_valid"] = [
            bool(validity_flag and ungated_valid)
            for validity_flag, ungated_valid in zip(
                dwt_eval_frame["dwt_ml_validity_flag"].tolist(),
                dwt_eval_frame["ungated_is_valid"].tolist(),
            )
        ]

        eval_frame["dwt_ungated_pred_hr_bpm"] = dwt_eval_frame["ungated_pred_hr_bpm"].tolist()
        eval_frame["dwt_ungated_is_valid"] = dwt_eval_frame["ungated_is_valid"].tolist()
        eval_frame["dwt_quality_target_label"] = dwt_eval_frame["quality_target_label"].tolist()
        eval_frame["dwt_hr_abs_error_bpm"] = dwt_eval_frame["hr_abs_error_bpm"].tolist()
        eval_frame["dwt_ml_signal_quality_score"] = dwt_eval_frame["dwt_ml_signal_quality_score"].tolist()
        eval_frame["dwt_ml_signal_quality_label"] = dwt_eval_frame["dwt_ml_signal_quality_label"].tolist()
        eval_frame["dwt_ml_validity_flag"] = dwt_eval_frame["dwt_ml_validity_flag"].tolist()
        eval_frame["dwt_ml_gated_pred_hr_bpm"] = dwt_eval_frame["dwt_ml_gated_pred_hr_bpm"].tolist()
        eval_frame["dwt_ml_gated_is_valid"] = dwt_eval_frame["dwt_ml_gated_is_valid"].tolist()

        dwt_eval_selected_row = dwt_eval_report_sweep.loc[
            np.isclose(dwt_eval_report_sweep["threshold"].to_numpy(dtype=float), dwt_selected_threshold)
        ]
        dwt_eval_selected_summary = dwt_eval_selected_row.iloc[0].to_dict() if not dwt_eval_selected_row.empty else {}
        dwt_final_train_summary = summarize_threshold_selection(
            dwt_fine_train_sweep if not dwt_fine_train_sweep.empty else dwt_coarse_train_sweep
        )
        dwt_best_train_rmse_pool = dwt_fine_train_sweep if not dwt_fine_train_sweep.empty else dwt_coarse_train_sweep
        dwt_feasible_train_rmse_pool = dwt_best_train_rmse_pool.loc[
            dwt_best_train_rmse_pool["is_feasible_retention"].astype(bool)
        ].copy()
        if dwt_feasible_train_rmse_pool.empty:
            dwt_feasible_train_rmse_pool = dwt_best_train_rmse_pool.copy()
        dwt_best_train_rmse_row = dwt_feasible_train_rmse_pool.sort_values(
            by=["rmse", "retention_ratio", "f1", "threshold"],
            ascending=[True, False, False, True],
        ).iloc[0].to_dict()
        dwt_fine_status = summarize_operating_point_status(
            dwt_fine_train_sweep if not dwt_fine_train_sweep.empty else dwt_coarse_train_sweep,
            selected_threshold=dwt_selected_threshold,
            stability_mae_tolerance=float(stage3_cfg["ml"].get("stability_mae_tolerance", 0.10)),
            stable_min_threshold_count=int(stage3_cfg["ml"].get("stable_min_threshold_count", 3)),
        )
        dwt_eval_beats_rule = dwt_eval_report_sweep.loc[
            dwt_eval_report_sweep["mae"].notna()
            & (dwt_eval_report_sweep["mae"].to_numpy(dtype=float) < test_rule_mae)
        ].copy()
        if dwt_eval_beats_rule.empty:
            dwt_best_eval_rule_tradeoff = {}
        else:
            dwt_best_eval_rule_tradeoff = dwt_eval_beats_rule.sort_values(
                by=["retention_ratio", "mae", "threshold"],
                ascending=[False, True, True],
            ).iloc[0].to_dict()
        dwt_operating_points_frame = _build_operating_points_frame(
            branch="dwt_ml",
            selected_threshold=dwt_selected_threshold,
            selection_mode="fine_refined" if not dwt_fine_train_sweep.empty else "coarse_only",
            retention_floor=float(stage3_cfg["ml"]["min_retention_ratio"]),
            final_train_summary=dwt_final_train_summary,
            eval_selected_summary=dwt_eval_selected_summary,
            test_rule_mae=test_rule_mae,
            fine_status=dwt_fine_status,
            best_train_rmse_row=dwt_best_train_rmse_row,
            best_eval_rule_tradeoff=dwt_best_eval_rule_tradeoff,
        )

    eval_selected_row = eval_report_sweep.loc[
        np.isclose(eval_report_sweep["threshold"].to_numpy(dtype=float), selected_threshold)
    ]
    eval_selected_summary = eval_selected_row.iloc[0].to_dict() if not eval_selected_row.empty else {}

    ungated_valid_count = int(eval_frame["ungated_is_valid"].fillna(False).astype(bool).sum())
    dwt_quality_frame = None
    if "dwt_quality_target_label" in eval_frame.columns and "dwt_ml_signal_quality_label" in eval_frame.columns:
        dwt_quality_frame = eval_frame.copy()
        dwt_quality_frame["quality_target_label"] = dwt_quality_frame["dwt_quality_target_label"]
    metrics_rows = [
        _summarize_quality_method(eval_frame, label_col="rule_signal_quality_label", method="stage3_rule_baseline"),
        _summarize_quality_method(
            eval_frame,
            label_col="ml_signal_quality_label",
            method="stage3_ml_logreg",
            selected_threshold=selected_threshold,
        ),
        _summarize_quality_method(
            eval_frame,
            label_col="motion_refined_quality_label",
            method="stage3_motion_refined",
            selected_threshold=motion_refined_threshold,
        ),
        _summarize_quality_method(
            dwt_quality_frame if dwt_quality_frame is not None else eval_frame.iloc[0:0].copy(),
            label_col="dwt_ml_signal_quality_label",
            method="stage3_dwt_ml",
            selected_threshold=dwt_selected_threshold if np.isfinite(dwt_selected_threshold) else None,
        ) if "dwt_ml_signal_quality_label" in eval_frame.columns else None,
        summarize_robust_hr_policy_behavior(eval_frame),
        _summarize_hr_method(
            eval_frame,
            pred_col="ungated_pred_hr_bpm",
            valid_col="ungated_is_valid",
            method="ungated_stage1_frequency",
            ungated_valid_count=ungated_valid_count,
        ),
        _summarize_hr_method(
            eval_frame,
            pred_col="rule_gated_pred_hr_bpm",
            valid_col="rule_gated_is_valid",
            method="gated_stage3_rule",
            ungated_valid_count=ungated_valid_count,
        ),
        _summarize_hr_method(
            eval_frame,
            pred_col="ml_gated_pred_hr_bpm",
            valid_col="ml_gated_is_valid",
            method="gated_stage3_ml_logreg",
            ungated_valid_count=ungated_valid_count,
            selected_threshold=selected_threshold,
        ),
        _summarize_hr_method(
            eval_frame,
            pred_col="motion_refined_gated_pred_hr_bpm",
            valid_col="motion_refined_gated_is_valid",
            method="gated_stage3_motion_refined",
            ungated_valid_count=ungated_valid_count,
            selected_threshold=motion_refined_threshold,
        ),
        _summarize_hr_method(
            eval_frame,
            pred_col="dwt_ml_gated_pred_hr_bpm",
            valid_col="dwt_ml_gated_is_valid",
            method="gated_stage3_dwt_ml",
            ungated_valid_count=ungated_valid_count,
            selected_threshold=dwt_selected_threshold if np.isfinite(dwt_selected_threshold) else None,
        ) if "dwt_ml_gated_pred_hr_bpm" in eval_frame.columns else None,
        _summarize_hr_method(
            eval_frame,
            pred_col="robust_hr_bpm",
            valid_col="robust_hr_is_valid",
            method="robust_stage3c2_policy",
            ungated_valid_count=ungated_valid_count,
        ),
    ]
    metrics_frame = pd.DataFrame([row for row in metrics_rows if row is not None])

    final_train_summary = summarize_threshold_selection(fine_train_sweep if not fine_train_sweep.empty else coarse_train_sweep)
    best_train_rmse_pool = (
        fine_train_sweep if not fine_train_sweep.empty else coarse_train_sweep
    )
    feasible_train_rmse_pool = best_train_rmse_pool.loc[best_train_rmse_pool["is_feasible_retention"].astype(bool)].copy()
    if feasible_train_rmse_pool.empty:
        feasible_train_rmse_pool = best_train_rmse_pool.copy()
    best_train_rmse_row = feasible_train_rmse_pool.sort_values(
        by=["rmse", "retention_ratio", "f1", "threshold"],
        ascending=[True, False, False, True],
    ).iloc[0].to_dict()
    fine_status = summarize_operating_point_status(
        fine_train_sweep if not fine_train_sweep.empty else coarse_train_sweep,
        selected_threshold=selected_threshold,
        stability_mae_tolerance=float(stage3_cfg["ml"].get("stability_mae_tolerance", 0.10)),
        stable_min_threshold_count=int(stage3_cfg["ml"].get("stable_min_threshold_count", 3)),
    )
    test_rule_mae = float(metrics_frame.loc[metrics_frame["method"] == "gated_stage3_rule", "mae"].iloc[0])
    eval_beats_rule = eval_report_sweep.loc[
        eval_report_sweep["mae"].notna()
        & (eval_report_sweep["mae"].to_numpy(dtype=float) < test_rule_mae)
    ].copy()
    if eval_beats_rule.empty:
        best_eval_rule_tradeoff = {}
    else:
        best_eval_rule_tradeoff = eval_beats_rule.sort_values(
            by=["retention_ratio", "mae", "threshold"],
            ascending=[False, True, True],
        ).iloc[0].to_dict()
    operating_points_frame = _build_operating_points_frame(
        branch="raw_ml",
        selected_threshold=selected_threshold,
        selection_mode="fine_refined" if not fine_train_sweep.empty else "coarse_only",
        retention_floor=float(stage3_cfg["ml"]["min_retention_ratio"]),
        final_train_summary=final_train_summary,
        eval_selected_summary=eval_selected_summary,
        test_rule_mae=test_rule_mae,
        fine_status=fine_status,
        best_train_rmse_row=best_train_rmse_row,
        best_eval_rule_tradeoff=best_eval_rule_tradeoff,
    )
    if not dwt_threshold_sweep_frame.empty:
        threshold_sweep_frame = pd.concat([threshold_sweep_frame, dwt_threshold_sweep_frame], ignore_index=True, sort=False)
    if not dwt_operating_points_frame.empty:
        operating_points_frame = pd.concat([operating_points_frame, dwt_operating_points_frame], ignore_index=True, sort=False)

    print("Stage 3 enhancement completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects: {len(eval_subjects)}")
    print(f"Windows generated: {len(eval_frame)}")
    print(f"Selected ML threshold: {selected_threshold:.4f}")
    if np.isfinite(dwt_selected_threshold):
        print(f"Selected DWT ML threshold: {dwt_selected_threshold:.4f}")
    print(f"Operating-point status: {fine_status['operating_point_status']}")
    for method in ("stage3_rule_baseline", "stage3_ml_logreg", "stage3_motion_refined", "stage3_dwt_ml"):
        if method not in metrics_frame["method"].tolist():
            continue
        row = metrics_frame.loc[metrics_frame["method"] == method].iloc[0].to_dict()
        print(f"quality method: {method}")
        for key in ("accuracy", "precision", "recall", "f1", "num_eval_windows"):
            value = row[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    if "robust_stage3c2_policy" in metrics_frame["method"].tolist():
        policy_row = metrics_frame.loc[
            (metrics_frame["method"] == "robust_stage3c2_policy") & (metrics_frame["task"] == "policy_summary")
        ].iloc[0].to_dict()
        print("policy summary: robust_stage3c2_policy")
        for key in (
            "frequency_fraction",
            "beat_fallback_fraction",
            "hold_previous_fraction",
            "reject_fraction",
            "avg_abs_jump_bpm",
            "hold_count",
            "subject_boundary_reset_count",
            "fallback_insufficient_count",
        ):
            value = policy_row.get(key, math.nan)
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    for method in ("ungated_stage1_frequency", "gated_stage3_rule", "gated_stage3_ml_logreg", "gated_stage3_motion_refined", "gated_stage3_dwt_ml", "robust_stage3c2_policy"):
        method_subset = metrics_frame.loc[
            (metrics_frame["method"] == method) & (metrics_frame["task"] == "hr_comparison")
        ]
        if method_subset.empty:
            continue
        row = method_subset.iloc[0].to_dict()
        print(f"hr method: {method}")
        for key in ("mae", "rmse", "mape", "pearson_r", "num_valid_windows", "retention_ratio"):
            value = row[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / f"{dataset_cfg['name']}_stage3_enhanced_predictions.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage3_enhanced_metrics.csv"
        threshold_sweep_path = output_dir / f"{dataset_cfg['name']}_stage3_enhanced_threshold_sweep.csv"
        operating_points_path = output_dir / f"{dataset_cfg['name']}_stage3_enhanced_operating_points.csv"
        eval_frame.to_csv(predictions_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        if bool(stage3_cfg["ml"].get("save_threshold_analysis", True)):
            threshold_sweep_frame.to_csv(threshold_sweep_path, index=False)
            operating_points_frame.to_csv(operating_points_path, index=False)
        print(f"Saved predictions to: {predictions_path}")
        print(f"Saved metrics to: {metrics_path}")
        if bool(stage3_cfg["ml"].get("save_threshold_analysis", True)):
            print(f"Saved threshold sweep to: {threshold_sweep_path}")
            print(f"Saved operating points to: {operating_points_path}")


if __name__ == "__main__":
    main()
