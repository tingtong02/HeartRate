from __future__ import annotations

import argparse
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
    compute_time_domain_prv_features,
    detect_beats_in_window,
    detect_reference_beats_in_window,
    evaluate_beat_detection,
    extract_ibi_from_beats,
    extract_matched_ibi_pairs_ms,
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


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage2_cfg = config["stage2"]
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

    beat_rows: list[dict[str, float | str | int]] = []
    feature_rows: list[dict[str, float | str | int]] = []
    all_ref_ibi_ms: list[float] = []
    all_pred_ibi_ms: list[float] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for subject_id in eval_subjects:
        record = loader.load_subject(subject_id)
        windows = build_analysis_windows(
            record=record,
            target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
            analysis_window_seconds=float(stage2_cfg["analysis_window_seconds"]),
            analysis_step_seconds=float(stage2_cfg["analysis_step_seconds"]),
        )
        for window in windows:
            pred_beats = detect_beats_in_window(window["ppg_window"], window["ppg_fs"], stage2_cfg["beat"])
            ref_beats = detect_reference_beats_in_window(window["ecg_window"], window["ecg_fs"])
            beat_eval = evaluate_beat_detection(
                pred_beats,
                ref_beats,
                pred_fs=window["ppg_fs"],
                ref_fs=window["ecg_fs"],
                tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
            )
            total_tp += int(beat_eval["tp"])
            total_fp += int(beat_eval["fp"])
            total_fn += int(beat_eval["fn"])

            pred_ibi_s = extract_ibi_from_beats(pred_beats, window["ppg_fs"])
            ref_ibi_s = extract_ibi_from_beats(ref_beats, window["ecg_fs"])
            pred_clean = clean_ibi_series(pred_ibi_s, stage2_cfg["ibi"])
            ref_clean = clean_ibi_series(ref_ibi_s, stage2_cfg["ibi"])

            ref_ibi_ms, pred_ibi_ms = extract_matched_ibi_pairs_ms(
                pred_beats,
                ref_beats,
                pred_fs=window["ppg_fs"],
                ref_fs=window["ecg_fs"],
                tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
            )
            if ref_ibi_ms.size and pred_ibi_ms.size:
                all_ref_ibi_ms.extend(ref_ibi_ms.tolist())
                all_pred_ibi_ms.extend(pred_ibi_ms.tolist())
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
            ref_features = compute_time_domain_prv_features(
                ref_clean["ibi_clean_s"],
                num_beats=len(ref_beats),
                num_ibi_raw=len(ref_ibi_s),
                num_ibi_clean=len(ref_clean["ibi_clean_s"]),
            )

            beat_rows.append(
                {
                    "dataset": window["dataset"],
                    "subject_id": window["subject_id"],
                    "analysis_window_index": window["analysis_window_index"],
                    "start_time_s": window["start_time_s"],
                    "duration_s": window["duration_s"],
                    "num_pred_beats": float(len(pred_beats)),
                    "num_ref_beats": float(len(ref_beats)),
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

    beat_frame = pd.DataFrame(beat_rows)
    feature_frame = pd.DataFrame(feature_rows)

    beat_summary = compute_precision_recall_f1(total_tp, total_fp, total_fn)
    beat_summary_row = {
        "task": "beat_detection",
        "metric_group": "summary",
        "precision": beat_summary["precision"],
        "recall": beat_summary["recall"],
        "f1": beat_summary["f1"],
        "beat_count_error": float(beat_frame["beat_count_error"].mean()) if not beat_frame.empty else math.nan,
    }
    ibi_summary = compute_ibi_error_metrics(np.asarray(all_ref_ibi_ms), np.asarray(all_pred_ibi_ms))
    ibi_summary_row = {
        "task": "ibi_error",
        "metric_group": "summary",
        "ibi_mae_ms": ibi_summary["ibi_mae_ms"],
        "ibi_rmse_ms": ibi_summary["ibi_rmse_ms"],
        "num_valid_ibi_pairs": ibi_summary["num_valid_ibi_pairs"],
    }
    feature_summary = summarize_feature_metrics(
        feature_frame,
        feature_names=FEATURE_NAMES,
        ref_prefix="ref_",
        pred_prefix="pred_",
    )
    feature_summary["task"] = "feature_comparison"
    feature_summary["metric_group"] = "summary"

    metrics_frame = pd.concat(
        [
            pd.DataFrame([beat_summary_row]),
            pd.DataFrame([ibi_summary_row]),
            feature_summary,
        ],
        ignore_index=True,
        sort=False,
    )

    print("Stage 2 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Subjects evaluated: {len(eval_subjects)}")
    print(f"Analysis windows: {len(beat_frame)}")
    print("task: beat_detection")
    for key in ("precision", "recall", "f1", "beat_count_error"):
        value = beat_summary_row[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")
    print("task: ibi_error")
    for key in ("ibi_mae_ms", "ibi_rmse_ms", "num_valid_ibi_pairs"):
        value = ibi_summary_row[key]
        print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        beats_path = output_dir / f"{dataset_cfg['name']}_stage2_beats.csv"
        features_path = output_dir / f"{dataset_cfg['name']}_stage2_features.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage2_metrics.csv"
        beat_frame.to_csv(beats_path, index=False)
        feature_frame.to_csv(features_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved beats to: {beats_path}")
        print(f"Saved features to: {features_path}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
