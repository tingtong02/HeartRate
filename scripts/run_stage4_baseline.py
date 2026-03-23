from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.preprocess import build_window_samples
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage1_hr import (
    estimate_hr_frequency_stage1,
    estimate_hr_time_stage1,
    fuse_hr_estimates,
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
from heart_rate_cnn.stage4_events import (
    EVENT_TYPES,
    PREDICTION_COLUMNS,
    build_stage4_event_predictions,
    summarize_stage4_event_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 4A quality-gated rule-based HR event baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage4.yaml", help="Stage 4 eval config path.")
    return parser.parse_args()


def make_loader(dataset_name: str, root_dir: str):
    if dataset_name == "ppg_dalia":
        return PPGDaliaLoader(root_dir)
    if dataset_name == "wesad":
        return WESADLoader(root_dir)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _build_stage3_source_rows(
    *,
    loader,
    subject_ids: list[str],
    preprocess_cfg: dict,
    eval_cfg: dict,
    stage1_cfg: dict,
    stage3_cfg: dict,
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
    return rows


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


def _build_quality_aware_source_frames(
    *,
    loader,
    train_subjects: list[str],
    eval_subjects: list[str],
    preprocess_cfg: dict,
    eval_cfg: dict,
    stage1_cfg: dict,
    stage3_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    train_frame = pd.DataFrame(
        _build_stage3_source_rows(
            loader=loader,
            subject_ids=train_subjects,
            preprocess_cfg=preprocess_cfg,
            eval_cfg=eval_cfg,
            stage1_cfg=stage1_cfg,
            stage3_cfg=stage3_cfg,
        )
    )
    eval_frame = pd.DataFrame(
        _build_stage3_source_rows(
            loader=loader,
            subject_ids=eval_subjects,
            preprocess_cfg=preprocess_cfg,
            eval_cfg=eval_cfg,
            stage1_cfg=stage1_cfg,
            stage3_cfg=stage3_cfg,
        )
    )

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

    return train_frame, eval_frame, selected_threshold


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage1_cfg = config["stage1"]
    stage3_cfg = config["stage3"]
    stage4_cfg = config["stage4"]
    output_cfg = config["output"]

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

    train_frame, eval_frame, selected_threshold = _build_quality_aware_source_frames(
        loader=loader,
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
    )

    train_predictions = build_stage4_event_predictions(train_frame, split_name="train", config=stage4_cfg)
    eval_predictions = build_stage4_event_predictions(eval_frame, split_name="eval", config=stage4_cfg)
    predictions_frame = pd.concat([train_predictions, eval_predictions], ignore_index=True, sort=False)
    metrics_frame = summarize_stage4_event_metrics(predictions_frame)

    required_columns = set(PREDICTION_COLUMNS)
    missing_prediction_columns = sorted(required_columns - set(predictions_frame.columns))
    if missing_prediction_columns:
        raise RuntimeError(f"Stage 4 predictions are missing required columns: {missing_prediction_columns}")

    print("Stage 4 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects: {len(eval_subjects)}")
    print(f"Stage 3 ML threshold reused for Stage 4 defaults: {selected_threshold:.2f}")
    for split_name in ("train", "eval"):
        split_metrics = metrics_frame.loc[
            (metrics_frame["split"] == split_name) & (metrics_frame["event_type"].isin(EVENT_TYPES))
        ].copy()
        print(f"split: {split_name}")
        for _, row in split_metrics.iterrows():
            print(
                "  "
                f"{row['event_type']}: "
                f"f1={row['f1']:.4f}, "
                f"precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"pred_events={int(row['num_pred_events'])}, "
                f"eval_events={int(row['num_eval_events'])}"
            )

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / f"{dataset_cfg['name']}_stage4_event_predictions.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage4_event_metrics.csv"
        predictions_frame.to_csv(predictions_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved predictions to: {predictions_path}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
