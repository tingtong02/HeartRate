from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.metrics import compute_hr_metrics
from heart_rate_cnn.preprocess import build_window_samples
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage1_hr import (
    estimate_hr_frequency_stage1,
    estimate_hr_time_stage1,
    fuse_hr_estimates,
)
from heart_rate_cnn.stage3_quality import (
    apply_ml_quality_decision,
    apply_rule_based_quality_decision,
    build_quality_target,
    compute_binary_classification_summary,
    extract_quality_features,
    fit_quality_logistic_regression,
    predict_quality_logistic_regression,
    select_best_ml_threshold,
)


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


def _build_stage3_rows(
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
                    "rule_gated_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"])
                    if bool(rule_row["validity_flag"])
                    else math.nan,
                    "rule_gated_is_valid": bool(window.is_valid and rule_row["validity_flag"]),
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
        )
    )

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

    threshold_summary = select_best_ml_threshold(
        train_frame,
        score_col="ml_signal_quality_score",
        pred_col="ungated_pred_hr_bpm",
        valid_col="ungated_is_valid",
        threshold_grid=[float(value) for value in stage3_cfg["ml"]["threshold_grid"]],
        min_retention_ratio=float(stage3_cfg["ml"]["min_retention_ratio"]),
    )
    selected_threshold = float(threshold_summary["selected_threshold"])

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

    ungated_valid_count = int(eval_frame["ungated_is_valid"].fillna(False).astype(bool).sum())
    metrics_rows = [
        _summarize_quality_method(eval_frame, label_col="rule_signal_quality_label", method="stage3_rule_baseline"),
        _summarize_quality_method(
            eval_frame,
            label_col="ml_signal_quality_label",
            method="stage3_ml_logreg",
            selected_threshold=selected_threshold,
        ),
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
    ]
    metrics_frame = pd.DataFrame(metrics_rows)

    print("Stage 3 enhancement completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects: {len(eval_subjects)}")
    print(f"Windows generated: {len(eval_frame)}")
    print(f"Selected ML threshold: {selected_threshold:.4f}")
    for method in ("stage3_rule_baseline", "stage3_ml_logreg"):
        row = metrics_frame.loc[metrics_frame["method"] == method].iloc[0].to_dict()
        print(f"quality method: {method}")
        for key in ("accuracy", "precision", "recall", "f1", "num_eval_windows"):
            value = row[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    for method in ("ungated_stage1_frequency", "gated_stage3_rule", "gated_stage3_ml_logreg"):
        row = metrics_frame.loc[metrics_frame["method"] == method].iloc[0].to_dict()
        print(f"hr method: {method}")
        for key in ("mae", "rmse", "mape", "pearson_r", "num_valid_windows", "retention_ratio"):
            value = row[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / f"{dataset_cfg['name']}_stage3_enhanced_predictions.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage3_enhanced_metrics.csv"
        eval_frame.to_csv(predictions_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved predictions to: {predictions_path}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
