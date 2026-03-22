from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
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
    apply_rule_based_quality_decision,
    build_quality_target,
    compute_binary_classification_summary,
    extract_quality_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 3 round-1 rule-based quality gating baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage3.yaml", help="Stage 3 eval config path.")
    return parser.parse_args()


def make_loader(dataset_name: str, root_dir: str):
    if dataset_name == "ppg_dalia":
        return PPGDaliaLoader(root_dir)
    if dataset_name == "wesad":
        return WESADLoader(root_dir)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _summarize_hr_method(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    valid_col: str,
    method: str,
    ungated_valid_count: int,
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
    eval_subjects = split.test_subjects if split.test_subjects else split.train_subjects
    hr_band_bpm = tuple(float(value) for value in eval_cfg["hr_band_bpm"])

    rows: list[dict[str, float | int | str | bool | None]] = []
    for subject_id in eval_subjects:
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
            decision_row = apply_rule_based_quality_decision(
                window_is_valid=bool(window.is_valid),
                features=feature_row,
                config=stage3_cfg["rule"],
            )
            gated_pred_hr_bpm = (
                float(freq_result["freq_pred_hr_bpm"])
                if decision_row["validity_flag"] and np.isfinite(float(freq_result["freq_pred_hr_bpm"]))
                else math.nan
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
                    **decision_row,
                    "ungated_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"]),
                    "ungated_is_valid": bool(window.is_valid and freq_result["freq_is_valid"]),
                    "gated_pred_hr_bpm": gated_pred_hr_bpm,
                    "gated_is_valid": bool(window.is_valid and decision_row["validity_flag"]),
                }
            )

    frame = pd.DataFrame(rows)
    classification_frame = frame.loc[frame["quality_target_label"].isin(["good", "poor"])].copy()
    classification_metrics = compute_binary_classification_summary(
        classification_frame["quality_target_label"].tolist(),
        classification_frame["signal_quality_label"].tolist(),
    )

    ungated_valid_count = int(frame["ungated_is_valid"].fillna(False).astype(bool).sum())
    metrics_rows = [
        {
            "task": "quality_classification",
            "method": "stage3_rule_baseline",
            "accuracy": classification_metrics["accuracy"],
            "precision": classification_metrics["precision"],
            "recall": classification_metrics["recall"],
            "f1": classification_metrics["f1"],
            "num_eval_windows": classification_metrics["num_eval_windows"],
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "pearson_r": math.nan,
            "num_valid_windows": math.nan,
            "retention_ratio": math.nan,
        },
        _summarize_hr_method(
            frame,
            pred_col="ungated_pred_hr_bpm",
            valid_col="ungated_is_valid",
            method="ungated_stage1_frequency",
            ungated_valid_count=ungated_valid_count,
        ),
        _summarize_hr_method(
            frame,
            pred_col="gated_pred_hr_bpm",
            valid_col="gated_is_valid",
            method="gated_stage3_rule",
            ungated_valid_count=ungated_valid_count,
        ),
    ]
    metrics_frame = pd.DataFrame(metrics_rows)

    print("Stage 3 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Subjects evaluated: {len(eval_subjects)}")
    print(f"Windows generated: {len(frame)}")
    print("task: quality_classification")
    for key in ("accuracy", "precision", "recall", "f1", "num_eval_windows"):
        value = classification_metrics[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    for method in ("ungated_stage1_frequency", "gated_stage3_rule"):
        row = metrics_frame.loc[metrics_frame["method"] == method].iloc[0].to_dict()
        print(f"method: {method}")
        for key in ("mae", "rmse", "mape", "pearson_r", "num_valid_windows", "retention_ratio"):
            value = row[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / f"{dataset_cfg['name']}_stage3_predictions.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage3_metrics.csv"
        frame.to_csv(predictions_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved predictions to: {predictions_path}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
