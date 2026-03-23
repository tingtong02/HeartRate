from __future__ import annotations

import argparse
import time

from heart_rate_cnn.config import load_yaml, merge_dicts
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.stage4_features import (
    make_loader,
    prepare_quality_aware_source_package,
    prepare_stage4_feature_package,
    resolve_stage4_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reusable Stage 4 source and feature packages.")
    parser.add_argument("--config", default="configs/base.yaml", help="Base config path.")
    parser.add_argument("--dataset-config", required=True, help="Dataset-specific config path.")
    parser.add_argument("--eval-config", default="configs/eval/hr_stage4_full.yaml", help="Stage 4 eval config path.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild Stage 4 source and feature caches.")
    parser.add_argument("--output-scope", choices=("canonical", "validation"), help="Override Stage 4 output scope.")
    parser.add_argument("--output-label", help="Override Stage 4 output label for validation outputs.")
    return parser.parse_args()


def _apply_runtime_overrides(config: dict, args: argparse.Namespace) -> dict:
    updated = config
    if args.rebuild_cache:
        updated = merge_dicts(updated, {"cache": {"rebuild": True}})
    if args.output_scope is not None:
        updated = merge_dicts(updated, {"output": {"scope": str(args.output_scope)}})
    if args.output_label is not None:
        updated = merge_dicts(updated, {"output": {"label": str(args.output_label)}})
    return updated


def _print_package_status(package_name: str, package: dict) -> None:
    cache_path = str(package.get("cache_path", ""))
    manifest_path = str(package.get("manifest_path", ""))
    location_text = f" ({cache_path})" if cache_path else ""
    print(
        f"{package_name}: {package.get('cache_status', 'unknown')} "
        f"in {float(package.get('elapsed_seconds', 0.0)):.2f}s{location_text}"
    )
    if manifest_path:
        print(f"  manifest: {manifest_path}")


def main() -> None:
    overall_start_time = time.perf_counter()
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))
    config = _apply_runtime_overrides(config, args)

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage1_cfg = config["stage1"]
    stage3_cfg = config["stage3"]
    stage4_shared_cfg = config["stage4_shared"]
    cache_cfg = config.get("cache", {})
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

    source_package = prepare_quality_aware_source_package(
        loader=loader,
        dataset_name=str(dataset_cfg["name"]),
        root_dir=str(dataset_cfg["root_dir"]),
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
        cache_cfg=cache_cfg,
    )
    feature_package = prepare_stage4_feature_package(
        loader=loader,
        dataset_name=str(dataset_cfg["name"]),
        root_dir=str(dataset_cfg["root_dir"]),
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        stage3_cfg=stage3_cfg,
        stage4_shared_cfg=stage4_shared_cfg,
        source_package=source_package,
        cache_cfg=cache_cfg,
    )

    print("Stage 4 source preparation completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Train subjects: {train_subjects}")
    print(f"Eval subjects: {eval_subjects}")
    print(f"Output scope: {output_cfg.get('scope', 'canonical')}")
    if str(output_cfg.get("scope", "canonical")) == "validation":
        print(f"Output label: {output_cfg.get('label', '')}")
    print(f"Resolved output directory: {resolve_stage4_output_dir(output_cfg)}")
    print(f"Stage 3 ML threshold in source package: {float(source_package['selected_threshold']):.2f}")
    _print_package_status("Stage 4 source package", source_package)
    _print_package_status("Stage 4 feature package", feature_package)
    print(f"Preparation runtime: {time.perf_counter() - overall_start_time:.2f}s")


if __name__ == "__main__":
    main()
