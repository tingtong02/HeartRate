from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage4_anomaly import (
    DEFAULT_MODEL_NAME as ANOMALY_MODEL_NAME,
    PREDICTION_COLUMNS as ANOMALY_PREDICTION_COLUMNS,
    build_anomaly_predictions,
    fit_isolation_forest_anomaly_model,
    summarize_stage4_anomaly_metrics,
)
from heart_rate_cnn.stage4_events import (
    PREDICTION_COLUMNS as EVENT_PREDICTION_COLUMNS,
    build_stage4_event_predictions,
    summarize_stage4_event_metrics,
)
from heart_rate_cnn.stage4_features import (
    STAGE4_IDENTITY_COLUMNS,
    build_quality_aware_source_frames,
    build_stage4_shared_feature_frame,
    make_loader,
)
from heart_rate_cnn.stage4_full import (
    FULL_METRIC_COLUMNS,
    FULL_PREDICTION_COLUMNS,
    build_stage4_full_predictions,
    collapse_stage4_event_predictions,
    summarize_stage4_full_metrics,
)
from heart_rate_cnn.stage4_irregular import (
    DEFAULT_MODEL_NAME as IRREGULAR_MODEL_NAME,
    PREDICTION_COLUMNS as IRREGULAR_PREDICTION_COLUMNS,
    RULE_BASELINE_NAME,
    build_irregular_proxy_labels,
    build_rule_baseline_candidates,
    build_screening_predictions,
    fit_hist_gbdt_irregular_classifier,
    predict_hist_gbdt_irregular_scores,
    summarize_stage4_irregular_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final unified Stage 4 pipeline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage4_full.yaml", help="Stage 4 full eval config path.")
    return parser.parse_args()


def _build_irregular_predictions(
    *,
    feature_frame: pd.DataFrame,
    model,
    stage4_irregular_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_cfg = stage4_irregular_cfg.get("model", {})
    threshold = float(model_cfg.get("threshold", 0.50))
    scores = predict_hist_gbdt_irregular_scores(model, feature_frame)
    default_predictions = build_screening_predictions(
        feature_frame,
        model_name=str(model_cfg.get("default_method", IRREGULAR_MODEL_NAME)),
        scores=scores,
        threshold=threshold,
        candidate_reasons=[
            "score_threshold_met" if score >= threshold else "score_below_threshold"
            for score in scores.tolist()
        ],
        candidate_indicator_counts=[0 for _ in range(feature_frame.shape[0])],
        quality_gate_config=stage4_irregular_cfg.get("quality_gate", {}),
    )

    baseline_name = str(model_cfg.get("baseline_method", RULE_BASELINE_NAME))
    rule_scores, rule_reasons, rule_counts = build_rule_baseline_candidates(
        feature_frame,
        config=stage4_irregular_cfg.get("rule_baseline", {}),
    )
    rule_predictions = build_screening_predictions(
        feature_frame,
        model_name=baseline_name,
        scores=rule_scores,
        threshold=threshold,
        candidate_reasons=rule_reasons,
        candidate_indicator_counts=rule_counts,
        quality_gate_config=stage4_irregular_cfg.get("quality_gate", {}),
    )
    return default_predictions, rule_predictions


def _build_anomaly_base_frame(
    *,
    feature_frame: pd.DataFrame,
    event_summary: pd.DataFrame,
    irregular_default_predictions: pd.DataFrame,
) -> pd.DataFrame:
    irregular_subset = irregular_default_predictions.loc[
        :,
        [
            *STAGE4_IDENTITY_COLUMNS,
            "quality_gate_passed",
            "quality_gate_reason",
            "support_sufficient_flag",
        ],
    ].copy()
    anomaly_base = feature_frame.merge(event_summary, on=list(STAGE4_IDENTITY_COLUMNS), how="left", validate="one_to_one")
    anomaly_base = anomaly_base.merge(irregular_subset, on=list(STAGE4_IDENTITY_COLUMNS), how="left", validate="one_to_one")
    if "screening_proxy_target" not in anomaly_base.columns:
        anomaly_base["screening_proxy_target"] = False
    if "proxy_label_support_flag" not in anomaly_base.columns:
        anomaly_base["proxy_label_support_flag"] = False
    anomaly_base["proxy_hr_event_target_any"] = anomaly_base["proxy_hr_event_target_any"].fillna(False).astype(bool)
    anomaly_base["screening_proxy_target"] = anomaly_base["screening_proxy_target"].fillna(False).astype(bool)
    anomaly_base["proxy_label_support_flag"] = anomaly_base["proxy_label_support_flag"].fillna(False).astype(bool)
    anomaly_base["quality_gate_passed"] = anomaly_base["quality_gate_passed"].fillna(False).astype(bool)
    anomaly_base["quality_gate_reason"] = anomaly_base["quality_gate_reason"].fillna("").astype(str)
    anomaly_base["support_sufficient_flag"] = anomaly_base["support_sufficient_flag"].fillna(False).astype(bool)
    anomaly_base["proxy_hr_event_support_flag"] = anomaly_base["window_is_valid"].astype(bool) & anomaly_base["ref_hr_bpm"].notna()
    anomaly_base["proxy_abnormal_target"] = anomaly_base["proxy_hr_event_target_any"].astype(bool) | anomaly_base["screening_proxy_target"].astype(bool)
    anomaly_base["proxy_abnormal_support_flag"] = anomaly_base["proxy_hr_event_support_flag"].astype(bool) & anomaly_base["proxy_label_support_flag"].astype(bool)
    return anomaly_base


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage1_cfg = config["stage1"]
    stage3_cfg = config["stage3"]
    stage4_event_cfg = config["stage4"]
    stage4_shared_cfg = config["stage4_shared"]
    stage4_irregular_cfg = config["stage4_irregular"]
    stage4_anomaly_cfg = config["stage4_anomaly"]
    stage4_full_cfg = config["stage4_full"]
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

    train_event_predictions = build_stage4_event_predictions(train_source_frame, split_name="train", config=stage4_event_cfg)
    eval_event_predictions = build_stage4_event_predictions(eval_source_frame, split_name="eval", config=stage4_event_cfg)
    event_predictions = pd.concat([train_event_predictions, eval_event_predictions], ignore_index=True, sort=False)
    event_metrics = summarize_stage4_event_metrics(event_predictions)
    event_summary = collapse_stage4_event_predictions(event_predictions)

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

    irregular_model_cfg = {**stage4_irregular_cfg.get("model", {}), "random_seed": int(eval_cfg["random_seed"])}
    irregular_model = fit_hist_gbdt_irregular_classifier(train_feature_frame, config=irregular_model_cfg)
    train_irregular_default, train_irregular_rule = _build_irregular_predictions(
        feature_frame=train_feature_frame,
        model=irregular_model,
        stage4_irregular_cfg=stage4_irregular_cfg,
    )
    eval_irregular_default, eval_irregular_rule = _build_irregular_predictions(
        feature_frame=eval_feature_frame,
        model=irregular_model,
        stage4_irregular_cfg=stage4_irregular_cfg,
    )
    irregular_predictions = pd.concat(
        [train_irregular_default, eval_irregular_default, train_irregular_rule, eval_irregular_rule],
        ignore_index=True,
        sort=False,
    )
    irregular_metrics = summarize_stage4_irregular_metrics(irregular_predictions)

    combined_feature_frame = pd.concat([train_feature_frame, eval_feature_frame], ignore_index=True, sort=False)
    default_irregular_predictions = irregular_predictions.loc[
        irregular_predictions["model_name"] == str(stage4_irregular_cfg["model"].get("default_method", IRREGULAR_MODEL_NAME))
    ].copy()
    anomaly_base = _build_anomaly_base_frame(
        feature_frame=combined_feature_frame,
        event_summary=event_summary,
        irregular_default_predictions=default_irregular_predictions,
    )
    anomaly_model, anomaly_reference_mask = fit_isolation_forest_anomaly_model(
        anomaly_base,
        config={**stage4_anomaly_cfg, **stage4_anomaly_cfg.get("fit_reference", {}), "random_seed": int(eval_cfg["random_seed"])},
    )
    anomaly_predictions = build_anomaly_predictions(
        anomaly_base,
        model=anomaly_model,
        fit_reference_mask=anomaly_reference_mask,
        config=stage4_anomaly_cfg,
    )
    anomaly_metrics = summarize_stage4_anomaly_metrics(anomaly_predictions)

    full_predictions = build_stage4_full_predictions(
        feature_frame=combined_feature_frame,
        event_summary=event_summary,
        irregular_predictions=default_irregular_predictions,
        anomaly_predictions=anomaly_predictions,
        config=stage4_full_cfg,
    )
    full_metrics = summarize_stage4_full_metrics(
        full_predictions=full_predictions,
        event_metrics=event_metrics,
        irregular_metrics=irregular_metrics,
        anomaly_metrics=anomaly_metrics,
    )

    required_output_sets = [
        ("event", event_predictions, EVENT_PREDICTION_COLUMNS),
        ("irregular", irregular_predictions, IRREGULAR_PREDICTION_COLUMNS),
        ("anomaly", anomaly_predictions, ANOMALY_PREDICTION_COLUMNS),
        ("full", full_predictions, FULL_PREDICTION_COLUMNS),
    ]
    for output_name, frame, required_columns in required_output_sets:
        missing = sorted(set(required_columns) - set(frame.columns))
        if missing:
            raise RuntimeError(f"Stage 4 {output_name} predictions are missing required columns: {missing}")
    if sorted(set(FULL_METRIC_COLUMNS) - set(full_metrics.columns)):
        missing_metrics = sorted(set(FULL_METRIC_COLUMNS) - set(full_metrics.columns))
        raise RuntimeError(f"Stage 4 full metrics are missing required columns: {missing_metrics}")

    print("Final Stage 4 pipeline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Eval subjects: {len(eval_subjects)}")
    print(f"Stage 3 ML threshold reused upstream: {selected_threshold:.2f}")
    eval_stage3 = full_metrics.loc[
        (full_metrics["metric_group"] == "stage3_comparison")
        & (full_metrics["split"] == "eval")
        & (full_metrics["method"] == "stage3_quality_only")
    ].copy()
    eval_stage4 = full_metrics.loc[
        (full_metrics["metric_group"] == "stage3_comparison")
        & (full_metrics["split"] == "eval")
        & (full_metrics["method"] == "stage4_full_default")
    ].copy()
    eval_anomaly = full_metrics.loc[
        (full_metrics["metric_group"] == "stage3_comparison")
        & (full_metrics["split"] == "eval")
        & (full_metrics["method"] == "stage4_anomaly_default")
    ].copy()
    if not eval_stage3.empty:
        row = eval_stage3.iloc[0]
        print(
            "  "
            f"Stage 3 baseline: auprc={row['auprc']:.4f}, auroc={row['auroc']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}"
        )
    if not eval_stage4.empty:
        row = eval_stage4.iloc[0]
        print(
            "  "
            f"Stage 4 unified: auprc={row['auprc']:.4f}, auroc={row['auroc']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}"
        )
    if not eval_anomaly.empty:
        row = eval_anomaly.iloc[0]
        print(
            "  "
            f"Stage 4 anomaly: auprc={row['auprc']:.4f}, auroc={row['auroc']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}"
        )

    if output_cfg.get("save_csv", False):
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = str(dataset_cfg["name"])
        output_map = {
            f"{dataset_name}_stage4_event_predictions.csv": event_predictions,
            f"{dataset_name}_stage4_event_metrics.csv": event_metrics,
            f"{dataset_name}_stage4_irregular_predictions.csv": irregular_predictions,
            f"{dataset_name}_stage4_irregular_metrics.csv": irregular_metrics,
            f"{dataset_name}_stage4_anomaly_predictions.csv": anomaly_predictions,
            f"{dataset_name}_stage4_anomaly_metrics.csv": anomaly_metrics,
            f"{dataset_name}_stage4_full_predictions.csv": full_predictions,
            f"{dataset_name}_stage4_full_metrics.csv": full_metrics,
        }
        for filename, frame in output_map.items():
            path = output_dir / filename
            frame.to_csv(path, index=False)
            print(f"Saved {filename}")
        if output_cfg.get("save_feature_frame", False):
            feature_path = output_dir / f"{dataset_name}_stage4_feature_frame.csv"
            combined_feature_frame.to_csv(feature_path, index=False)
            print(f"Saved {feature_path.name}")


if __name__ == "__main__":
    main()
