from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from heart_rate_cnn.baseline_hr import estimate_hr_frequency_domain
from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.metrics import summarize_method_metrics
from heart_rate_cnn.preprocess import build_window_samples
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage1_hr import (
    estimate_hr_frequency_stage1,
    estimate_hr_time_stage1,
    fuse_hr_estimates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 1 HR comparison baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage1.yaml", help="Stage 1 eval config path.")
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
    stage1_cfg = config["stage1"]
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
            stage0_pred = estimate_hr_frequency_domain(window.ppg, window.ppg_fs, hr_band_bpm=hr_band_bpm)
            freq_result = estimate_hr_frequency_stage1(window.ppg, window.ppg_fs, hr_band_bpm, stage1_cfg["frequency"])
            time_result = estimate_hr_time_stage1(window.ppg, window.ppg_fs, hr_band_bpm, stage1_cfg["time"])
            fusion_result = fuse_hr_estimates(
                freq_result,
                time_result,
                agreement_threshold_bpm=float(stage1_cfg["fusion"]["agreement_threshold_bpm"]),
                conflict_threshold_bpm=float(stage1_cfg["fusion"]["conflict_threshold_bpm"]),
            )

            rows.append(
                {
                    "dataset": window.dataset,
                    "subject_id": window.subject_id,
                    "window_index": window.window_index,
                    "start_time_s": window.start_time_s,
                    "duration_s": window.duration_s,
                    "ref_hr_bpm": window.ref_hr_bpm,
                    "stage0_pred_hr_bpm": stage0_pred,
                    "freq_pred_hr_bpm": freq_result["freq_pred_hr_bpm"],
                    "time_pred_hr_bpm": time_result["time_pred_hr_bpm"],
                    "fusion_pred_hr_bpm": fusion_result["fusion_pred_hr_bpm"],
                    "freq_confidence": freq_result["freq_confidence"],
                    "time_confidence": time_result["time_confidence"],
                    "fusion_confidence": fusion_result["fusion_confidence"],
                    "fusion_source": fusion_result["fusion_source"],
                    "stage0_is_valid": bool(window.is_valid and math.isfinite(stage0_pred)),
                    "freq_is_valid": bool(window.is_valid and freq_result["freq_is_valid"]),
                    "time_is_valid": bool(window.is_valid and time_result["time_is_valid"]),
                    "fusion_is_valid": bool(window.is_valid and fusion_result["fusion_is_valid"]),
                }
            )

    frame = pd.DataFrame(rows)
    metrics_frame = summarize_method_metrics(
        frame,
        method_specs={
            "stage0_baseline": {"pred_col": "stage0_pred_hr_bpm", "valid_col": "stage0_is_valid"},
            "stage1_frequency": {"pred_col": "freq_pred_hr_bpm", "valid_col": "freq_is_valid"},
            "stage1_time": {"pred_col": "time_pred_hr_bpm", "valid_col": "time_is_valid"},
            "stage1_fusion": {"pred_col": "fusion_pred_hr_bpm", "valid_col": "fusion_is_valid"},
        },
        ref_col="ref_hr_bpm",
    )

    print("Stage 1 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Subjects evaluated: {len(eval_subjects)}")
    print(f"Windows generated: {len(frame)}")
    for row in metrics_frame.to_dict(orient="records"):
        print(f"method: {row['method']}")
        for key in ("mae", "rmse", "mape", "pearson_r", "num_valid_windows"):
            value = row[key]
            print(f"  {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"  {key}: {value}")

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = output_dir / f"{dataset_cfg['name']}_stage1_predictions.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage1_metrics.csv"
        frame.to_csv(predictions_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved predictions to: {predictions_path}")
        print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
