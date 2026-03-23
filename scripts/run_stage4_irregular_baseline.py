from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage4_features import (
    build_quality_aware_source_frames,
    build_stage4_shared_feature_frame,
    make_loader,
)
from heart_rate_cnn.stage4_irregular import (
    DEFAULT_MODEL_NAME,
    PREDICTION_COLUMNS,
    RULE_BASELINE_NAME,
    build_irregular_proxy_labels,
    build_rule_baseline_candidates,
    build_screening_predictions,
    fit_hist_gbdt_irregular_classifier,
    predict_hist_gbdt_irregular_scores,
    summarize_stage4_irregular_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 4B quality-gated irregular pulse screening baseline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument(
        "--eval-config",
        default="configs/eval/hr_stage4_irregular.yaml",
        help="Stage 4 irregular eval config path.",
    )
    return parser.parse_args()


def _build_method_predictions(
    *,
    feature_frame: pd.DataFrame,
    model_name: str,
    stage4_irregular_cfg: dict,
) -> pd.DataFrame:
    if model_name == RULE_BASELINE_NAME:
        scores, candidate_reasons, candidate_counts = build_rule_baseline_candidates(
            feature_frame,
            config=stage4_irregular_cfg.get("rule_baseline", {}),
        )
        threshold = float(stage4_irregular_cfg["model"]["threshold"])
        return build_screening_predictions(
            feature_frame,
            model_name=model_name,
            scores=scores,
            threshold=threshold,
            candidate_reasons=candidate_reasons,
            candidate_indicator_counts=candidate_counts,
            quality_gate_config=stage4_irregular_cfg.get("quality_gate", {}),
        )
    raise ValueError(f"Unsupported irregular screening method: {model_name}")


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage1_cfg = config["stage1"]
    stage3_cfg = config["stage3"]
    stage4_shared_cfg = config["stage4_shared"]
    stage4_irregular_cfg = config["stage4_irregular"]
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

    train_source_frame, eval_source_frame, _, _, selected_threshold = build_quality_aware_source_frames(
        loader=loader,
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
    )

    train_feature_frame = build_stage4_shared_feature_frame(
        loader=loader,
        subject_ids=train_subjects,
        split_name="train",
        preprocess_cfg=preprocess_cfg,
        stage3_cfg=stage3_cfg,
        stage4_shared_cfg=stage4_shared_cfg,
        source_frame=train_source_frame,
    )
    eval_feature_frame = build_stage4_shared_feature_frame(
        loader=loader,
        subject_ids=eval_subjects,
        split_name="eval",
        preprocess_cfg=preprocess_cfg,
        stage3_cfg=stage3_cfg,
        stage4_shared_cfg=stage4_shared_cfg,
        source_frame=eval_source_frame,
    )
    train_feature_frame = build_irregular_proxy_labels(train_feature_frame, config=stage4_irregular_cfg)
    eval_feature_frame = build_irregular_proxy_labels(eval_feature_frame, config=stage4_irregular_cfg)

    default_model_cfg = {**stage4_irregular_cfg.get("model", {}), "random_seed": int(eval_cfg["random_seed"])}
    model = fit_hist_gbdt_irregular_classifier(train_feature_frame, config=default_model_cfg)
    score_threshold = float(default_model_cfg.get("threshold", 0.50))
    train_scores = predict_hist_gbdt_irregular_scores(model, train_feature_frame)
    eval_scores = predict_hist_gbdt_irregular_scores(model, eval_feature_frame)

    train_model_predictions = build_screening_predictions(
        train_feature_frame,
        model_name=str(default_model_cfg.get("default_method", DEFAULT_MODEL_NAME)),
        scores=train_scores,
        threshold=score_threshold,
        candidate_reasons=[
            "score_threshold_met" if score >= score_threshold else "score_below_threshold"
            for score in train_scores.tolist()
        ],
        candidate_indicator_counts=[0 for _ in range(train_feature_frame.shape[0])],
        quality_gate_config=stage4_irregular_cfg.get("quality_gate", {}),
    )
    eval_model_predictions = build_screening_predictions(
        eval_feature_frame,
        model_name=str(default_model_cfg.get("default_method", DEFAULT_MODEL_NAME)),
        scores=eval_scores,
        threshold=score_threshold,
        candidate_reasons=[
            "score_threshold_met" if score >= score_threshold else "score_below_threshold"
            for score in eval_scores.tolist()
        ],
        candidate_indicator_counts=[0 for _ in range(eval_feature_frame.shape[0])],
        quality_gate_config=stage4_irregular_cfg.get("quality_gate", {}),
    )

    baseline_name = str(default_model_cfg.get("baseline_method", RULE_BASELINE_NAME))
    train_rule_predictions = _build_method_predictions(
        feature_frame=train_feature_frame,
        model_name=baseline_name,
        stage4_irregular_cfg=stage4_irregular_cfg,
    )
    eval_rule_predictions = _build_method_predictions(
        feature_frame=eval_feature_frame,
        model_name=baseline_name,
        stage4_irregular_cfg=stage4_irregular_cfg,
    )

    predictions_frame = pd.concat(
        [
            train_model_predictions,
            eval_model_predictions,
            train_rule_predictions,
            eval_rule_predictions,
        ],
        ignore_index=True,
        sort=False,
    )
    metrics_frame = summarize_stage4_irregular_metrics(predictions_frame)

    missing_prediction_columns = sorted(set(PREDICTION_COLUMNS) - set(predictions_frame.columns))
    if missing_prediction_columns:
        raise RuntimeError(f"Stage 4 irregular predictions are missing required columns: {missing_prediction_columns}")

    print("Stage 4 irregular screening baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects: {len(eval_subjects)}")
    print(f"Stage 3 ML threshold reused upstream: {selected_threshold:.2f}")
    for split_name in ("train", "eval"):
        split_metrics = metrics_frame.loc[metrics_frame["split"] == split_name].copy()
        print(f"split: {split_name}")
        for _, row in split_metrics.iterrows():
            print(
                "  "
                f"{row['method']}: "
                f"f1={row['f1']:.4f}, "
                f"precision={row['precision']:.4f}, "
                f"recall={row['recall']:.4f}, "
                f"positives={int(row['num_positive_predictions'])}, "
                f"targets={int(row['num_positive_targets'])}"
            )

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = str(dataset_cfg["name"])
        predictions_path = output_dir / f"{dataset_name}_stage4_irregular_predictions.csv"
        metrics_path = output_dir / f"{dataset_name}_stage4_irregular_metrics.csv"
        predictions_frame.to_csv(predictions_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved predictions to {predictions_path}")
        print(f"Saved metrics to {metrics_path}")

        if output_cfg.get("save_feature_frame", False):
            feature_frame_path = output_dir / f"{dataset_name}_stage4_feature_frame.csv"
            combined_feature_frame = pd.concat([train_feature_frame, eval_feature_frame], ignore_index=True, sort=False)
            combined_feature_frame.to_csv(feature_frame_path, index=False)
            print(f"Saved feature frame to {feature_frame_path}")


if __name__ == "__main__":
    main()
