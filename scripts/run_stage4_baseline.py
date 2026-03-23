from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage4_events import (
    EVENT_TYPES,
    PREDICTION_COLUMNS,
    build_stage4_event_predictions,
    summarize_stage4_event_metrics,
)
from heart_rate_cnn.stage4_features import build_quality_aware_source_frames, make_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 4A quality-gated rule-based HR event baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage4.yaml", help="Stage 4 eval config path.")
    return parser.parse_args()

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

    train_frame, eval_frame, _, _, selected_threshold = build_quality_aware_source_frames(
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
