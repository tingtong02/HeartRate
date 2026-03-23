from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage4_features import make_loader, resolve_stage4_output_dir
from heart_rate_cnn.stage5_multitask import build_stage5_multitask_predictions, summarize_stage5_metrics
from heart_rate_cnn.stage5_respiration import (
    _require_torch,
    build_stage4_default_context_frame,
    collect_stage5_window_seconds,
    fit_stage5_resp_cnn,
    predict_stage5_respiration,
    prepare_stage5_window_package,
    run_stage5_tuning,
    torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final Stage 5 respiration and multitask pipeline.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage5.yaml", help="Stage 5 eval config path.")
    parser.add_argument("--cnn-config", default="configs/eval/hr_stage5_cnn.yaml", help="Stage 5 CNN overlay config path.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild Stage 4 and Stage 5 caches.")
    parser.add_argument("--output-scope", choices=("canonical", "validation"), help="Override output scope.")
    parser.add_argument("--output-label", help="Override output label for validation outputs.")
    return parser.parse_args()


def _apply_runtime_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = config
    if args.rebuild_cache:
        updated = merge_dicts(updated, {"cache": {"rebuild": True}, "stage5": {"cache": {"rebuild": True}}})
    if args.output_scope is not None:
        updated = merge_dicts(updated, {"output": {"scope": str(args.output_scope)}})
    if args.output_label is not None:
        updated = merge_dicts(updated, {"output": {"label": str(args.output_label)}})
    return updated


def _print_package_status(name: str, package: dict) -> None:
    cache_path = str(package.get("cache_path", ""))
    location_text = f" ({cache_path})" if cache_path else ""
    print(f"{name}: {package.get('cache_status', 'unknown')} in {float(package.get('elapsed_seconds', 0.0)):.2f}s{location_text}")


def _resolve_model_dir(output_cfg: dict, stage5_cfg: dict) -> Path:
    scope = str(output_cfg.get("scope", "canonical"))
    if scope == "canonical":
        return Path(str(stage5_cfg.get("model_dir", "outputs/models/stage5"))).expanduser()
    return resolve_stage4_output_dir(output_cfg) / "models" / "stage5"


def main() -> None:
    _require_torch()
    overall_start_time = time.perf_counter()
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))
    config = merge_dicts(config, load_yaml(args.cnn_config))
    config = _apply_runtime_overrides(config, args)

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage1_cfg = config["stage1"]
    stage3_cfg = config["stage3"]
    stage4_cfg = config["stage4"]
    stage4_shared_cfg = config["stage4_shared"]
    stage4_irregular_cfg = config["stage4_irregular"]
    stage4_anomaly_cfg = config["stage4_anomaly"]
    stage4_full_cfg = config["stage4_full"]
    stage4_cache_cfg = config.get("cache", {})
    stage5_cfg = config["stage5"]
    stage5_cache_cfg = stage5_cfg.get("cache", {})
    output_cfg = config["output"]

    loader = make_loader(dataset_cfg["name"], dataset_cfg["root_dir"])
    subjects = loader.list_subjects()
    if dataset_cfg.get("subject_include"):
        allowed = set(dataset_cfg["subject_include"])
        subjects = [subject for subject in subjects if subject in allowed]
    if not subjects:
        raise RuntimeError("No subjects available for Stage 5 evaluation.")

    split = train_test_subject_split(
        subjects,
        test_size=float(eval_cfg["test_size"]),
        random_seed=int(eval_cfg["random_seed"]),
    )
    train_subjects = split.train_subjects
    eval_subjects = split.test_subjects if split.test_subjects else split.train_subjects

    output_dir = resolve_stage4_output_dir(output_cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage4_full_frame, source_package, feature_package = build_stage4_default_context_frame(
        loader=loader,
        dataset_name=str(dataset_cfg["name"]),
        root_dir=str(dataset_cfg["root_dir"]),
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
        stage4_cfg=stage4_cfg,
        stage4_shared_cfg=stage4_shared_cfg,
        stage4_irregular_cfg=stage4_irregular_cfg,
        stage4_anomaly_cfg=stage4_anomaly_cfg,
        stage4_full_cfg=stage4_full_cfg,
        stage4_cache_cfg=stage4_cache_cfg,
    )

    packages_by_window: dict[float, dict] = {}
    for window_seconds in collect_stage5_window_seconds(stage5_cfg):
        window_stage5_cfg = merge_dicts(stage5_cfg, {"window_seconds": float(window_seconds)})
        package = prepare_stage5_window_package(
            loader=loader,
            dataset_name=str(dataset_cfg["name"]),
            root_dir=str(dataset_cfg["root_dir"]),
            train_subjects=train_subjects,
            eval_subjects=eval_subjects,
            preprocess_cfg=preprocess_cfg,
            eval_cfg=eval_cfg,
            stage1_cfg=stage1_cfg,
            stage3_cfg=stage3_cfg,
            stage4_cfg=stage4_cfg,
            stage4_shared_cfg=stage4_shared_cfg,
            stage4_irregular_cfg=stage4_irregular_cfg,
            stage4_anomaly_cfg=stage4_anomaly_cfg,
            stage4_full_cfg=stage4_full_cfg,
            stage4_cache_cfg=stage4_cache_cfg,
            stage5_cfg=window_stage5_cfg,
            stage5_cache_cfg=stage5_cache_cfg,
            stage4_full_frame=stage4_full_frame,
            stage4_source_package=source_package,
            stage4_feature_package=feature_package,
        )
        packages_by_window[float(window_seconds)] = package

    tuning_results, best_candidate = run_stage5_tuning(
        packages_by_window,
        train_subjects=train_subjects,
        eval_cfg=eval_cfg,
        stage5_cfg=stage5_cfg,
        output_dir=output_dir,
        dataset_name=str(dataset_cfg["name"]),
    )
    final_candidate = {
        **best_candidate,
        "max_epochs": int(stage5_cfg.get("tuning", {}).get("final_max_epochs", 20)),
        "patience": int(stage5_cfg.get("tuning", {}).get("final_patience", 6)),
    }
    best_package = packages_by_window[float(final_candidate["window_seconds"])]
    model_bundle = fit_stage5_resp_cnn(
        best_package,
        candidate_cfg=final_candidate,
        train_subjects=train_subjects,
        random_seed=int(eval_cfg["random_seed"]),
    )
    stage5_predictions = predict_stage5_respiration(
        best_package,
        model_bundle=model_bundle,
        candidate_cfg=final_candidate,
    )
    multitask_predictions = build_stage5_multitask_predictions(
        stage5_predictions,
        resp_validity_threshold=float(final_candidate["resp_validity_threshold"]),
    )
    stage5_metrics = summarize_stage5_metrics(multitask_predictions)

    predictions_path = output_dir / f"{dataset_cfg['name']}_stage5_predictions.csv"
    metrics_path = output_dir / f"{dataset_cfg['name']}_stage5_metrics.csv"
    multitask_predictions.to_csv(predictions_path, index=False)
    stage5_metrics.to_csv(metrics_path, index=False)

    model_dir = _resolve_model_dir(output_cfg, stage5_cfg)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / f"{dataset_cfg['name']}_{stage5_cfg['model_name']}_best.pt"
    if torch is None:
        raise ImportError("PyTorch is required to save the Stage 5 checkpoint.")
    torch.save(model_bundle, checkpoint_path)
    config_path = model_dir / f"{dataset_cfg['name']}_{stage5_cfg['model_name']}_best_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(final_candidate, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")

    print("Stage 5 full pipeline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {train_subjects}")
    print(f"Eval subjects: {eval_subjects}")
    print(f"Resolved output directory: {output_dir}")
    _print_package_status("Stage 4 source package", source_package)
    _print_package_status("Stage 4 feature package", feature_package)
    for window_seconds, package in sorted(packages_by_window.items()):
        _print_package_status(f"Stage 5 window package ({window_seconds:.0f}s)", package)
    print(f"Selected Stage 5 candidate: {json.dumps(final_candidate, sort_keys=True, default=float)}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Checkpoint: {checkpoint_path}")
    eval_rows = stage5_metrics.loc[stage5_metrics["split"] == "eval"].copy()
    for _, row in eval_rows.iterrows():
        print(
            f"{row['method']} [{row['subset']}]: "
            f"MAE={float(row['resp_mae_bpm']):.3f}, "
            f"RMSE={float(row['resp_rmse_bpm']):.3f}, "
            f"r={float(row['resp_pearson_r']):.3f}, "
            f"within_3={float(row['within_3_bpm_rate']):.3f}"
        )
    print(f"Runtime: {time.perf_counter() - overall_start_time:.2f}s")


if __name__ == "__main__":
    main()
