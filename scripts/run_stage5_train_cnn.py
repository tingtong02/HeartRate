from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage4_features import make_loader, resolve_stage4_output_dir
from heart_rate_cnn.stage5_respiration import (
    _require_torch,
    build_stage4_default_context_frame,
    collect_stage5_window_seconds,
    fit_stage5_resp_cnn,
    prepare_stage5_window_package,
    run_stage5_tuning,
    torch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and train the default Stage 5 respiration CNN.")
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
        raise RuntimeError("No subjects available for Stage 5 training.")

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
        packages_by_window[float(window_seconds)] = prepare_stage5_window_package(
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
    model_bundle = fit_stage5_resp_cnn(
        packages_by_window[float(final_candidate["window_seconds"])],
        candidate_cfg=final_candidate,
        train_subjects=train_subjects,
        random_seed=int(eval_cfg["random_seed"]),
    )
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

    print("Stage 5 CNN training completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {train_subjects}")
    print(f"Eval subjects: {eval_subjects}")
    print(f"Tuning rows: {int(tuning_results.shape[0])}")
    print(f"Selected window seconds: {float(final_candidate['window_seconds']):.0f}")
    print(f"Selected channels: {final_candidate['channel_set']}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Training runtime: {time.perf_counter() - overall_start_time:.2f}s")


if __name__ == "__main__":
    main()
