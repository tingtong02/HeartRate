from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from heart_rate_cnn.baseline_hr import predict_windows
from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.metrics import compute_hr_metrics
from heart_rate_cnn.preprocess import build_window_samples
from heart_rate_cnn.split import train_test_subject_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 0 frequency-domain HR baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_baseline.yaml", help="Evaluation config path.")
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

    all_windows = []
    for subject_id in eval_subjects:
        record = loader.load_subject(subject_id)
        all_windows.extend(
            build_window_samples(
                record=record,
                target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
                window_seconds=float(preprocess_cfg["window_seconds"]),
                step_seconds=float(preprocess_cfg["step_seconds"]),
            )
        )

    hr_band_bpm = tuple(float(value) for value in eval_cfg["hr_band_bpm"])
    predictions = predict_windows(all_windows, hr_band_bpm=hr_band_bpm)
    frame = pd.DataFrame(predictions)
    valid_frame = frame[frame["is_valid"] & frame["ref_hr_bpm"].notna() & frame["pred_hr_bpm"].notna()].copy()
    metrics = compute_hr_metrics(
        valid_frame["ref_hr_bpm"].to_numpy(),
        valid_frame["pred_hr_bpm"].to_numpy(),
    )

    print("Stage 0 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Subjects evaluated: {len(eval_subjects)}")
    print(f"Windows generated: {len(frame)}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"{key}: {value}")

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{dataset_cfg['name']}_stage0_predictions.csv"
        frame.to_csv(output_path, index=False)
        print(f"Saved predictions to: {output_path}")


if __name__ == "__main__":
    main()
