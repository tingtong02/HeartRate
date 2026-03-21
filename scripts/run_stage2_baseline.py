from __future__ import annotations

import argparse
import copy
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
    extract_matched_ibi_pairs_with_indices_ms,
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

FEATURE_HIGHLIGHTS = ["mean_ibi_ms", "median_ibi_ms", "sdnn_ms", "rmssd_ms", "ibi_cv"]


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


def build_variant_configs(stage2_cfg: dict) -> tuple[dict[str, dict], dict]:
    shared_beat = {
        "bandpass_low_hz": float(stage2_cfg["beat"].get("bandpass_low_hz", 0.6)),
        "bandpass_high_hz": float(stage2_cfg["beat"].get("bandpass_high_hz", 3.5)),
        "bandpass_order": int(stage2_cfg["beat"].get("bandpass_order", 3)),
        "smooth_window_seconds": float(stage2_cfg["beat"].get("smooth_window_seconds", 0.20)),
        "smooth_polyorder": int(stage2_cfg["beat"].get("smooth_polyorder", 2)),
        "extra_smoothing": bool(stage2_cfg["beat"].get("extra_smoothing", True)),
        "hr_max_bpm": float(stage2_cfg["beat"].get("hr_max_bpm", 180.0)),
        "min_prominence": 0.05,
        "prominence_scale": 0.35,
        "min_width_seconds": 0.08,
        "refine_radius_seconds": 0.08,
        "refractory_scale": 1.5,
    }
    baseline_beat = {**shared_beat, "variant_mode": "baseline"}
    enhanced_beat = copy.deepcopy(stage2_cfg["beat"])
    enhanced_beat["variant_mode"] = "enhanced"

    shared_ibi = {
        "min_ibi_s": float(stage2_cfg["ibi"].get("min_ibi_s", 0.33)),
        "max_ibi_s": float(stage2_cfg["ibi"].get("max_ibi_s", 1.5)),
        "local_median_radius": int(stage2_cfg["ibi"].get("local_median_radius", 2)),
        "max_deviation_ratio": 0.30,
        "min_clean_ibi": int(stage2_cfg["ibi"].get("min_clean_ibi", 3)),
    }
    baseline_ibi = {**shared_ibi, "variant_mode": "baseline"}
    enhanced_ibi = copy.deepcopy(stage2_cfg["ibi"])
    enhanced_ibi["variant_mode"] = "enhanced"

    return (
        {
            "baseline": {"beat": baseline_beat, "ibi": baseline_ibi},
            "enhanced": {"beat": enhanced_beat, "ibi": enhanced_ibi},
        },
        baseline_ibi,
    )


def summarize_error_cases(beat_frame: pd.DataFrame, max_cases_per_variant: int) -> pd.DataFrame:
    error_columns = [
        "variant",
        "subject_id",
        "analysis_window_index",
        "num_pred_beats",
        "num_ref_beats",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "beat_count_error",
        "ibi_mae_ms",
        "ibi_rmse_ms",
        "pred_num_ibi_clean",
        "ref_num_ibi_clean",
    ]
    frames: list[pd.DataFrame] = []
    for variant in sorted(beat_frame["variant"].unique()):
        variant_frame = beat_frame.loc[beat_frame["variant"] == variant, error_columns].copy()
        if variant_frame.empty:
            continue
        variant_frame["sort_f1"] = variant_frame["f1"].fillna(-1.0)
        variant_frame["sort_ibi_rmse"] = variant_frame["ibi_rmse_ms"].fillna(-1.0)
        variant_frame = variant_frame.sort_values(
            by=["sort_f1", "sort_ibi_rmse"],
            ascending=[True, False],
        ).head(max_cases_per_variant)
        frames.append(variant_frame.drop(columns=["sort_f1", "sort_ibi_rmse"]))
    if not frames:
        return pd.DataFrame(columns=error_columns)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    config = merge_dicts(load_yaml(args.config), load_yaml(args.dataset_config))
    config = merge_dicts(config, load_yaml(args.eval_config))

    dataset_cfg = config["dataset"]
    preprocess_cfg = config["preprocess"]
    eval_cfg = config["eval"]
    stage2_cfg = config["stage2"]
    output_cfg = config["output"]
    variant_cfgs, reference_ibi_cfg = build_variant_configs(stage2_cfg)
    debug_cfg = stage2_cfg.get("debug", {})

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
    variant_state: dict[str, dict[str, list[float] | int]] = {
        variant: {
            "all_ref_ibi_ms": [],
            "all_pred_ibi_ms": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for variant in variant_cfgs
    }

    for subject_id in eval_subjects:
        record = loader.load_subject(subject_id)
        windows = build_analysis_windows(
            record=record,
            target_ppg_fs=float(preprocess_cfg["target_ppg_fs"]),
            analysis_window_seconds=float(stage2_cfg["analysis_window_seconds"]),
            analysis_step_seconds=float(stage2_cfg["analysis_step_seconds"]),
        )
        for window in windows:
            ref_beats = detect_reference_beats_in_window(window["ecg_window"], window["ecg_fs"])
            ref_ibi_s = extract_ibi_from_beats(ref_beats, window["ecg_fs"])
            ref_clean = clean_ibi_series(ref_ibi_s, reference_ibi_cfg)
            ref_features = compute_time_domain_prv_features(
                ref_clean["ibi_clean_s"],
                num_beats=len(ref_beats),
                num_ibi_raw=len(ref_ibi_s),
                num_ibi_clean=len(ref_clean["ibi_clean_s"]),
            )

            for variant, variant_cfg in variant_cfgs.items():
                pred_beats = detect_beats_in_window(window["ppg_window"], window["ppg_fs"], variant_cfg["beat"])
                beat_eval = evaluate_beat_detection(
                    pred_beats,
                    ref_beats,
                    pred_fs=window["ppg_fs"],
                    ref_fs=window["ecg_fs"],
                    tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                )
                variant_state[variant]["tp"] = int(variant_state[variant]["tp"]) + int(beat_eval["tp"])
                variant_state[variant]["fp"] = int(variant_state[variant]["fp"]) + int(beat_eval["fp"])
                variant_state[variant]["fn"] = int(variant_state[variant]["fn"]) + int(beat_eval["fn"])

                pred_ibi_s = extract_ibi_from_beats(pred_beats, window["ppg_fs"])
                pred_clean = clean_ibi_series(pred_ibi_s, variant_cfg["ibi"])
                ref_ibi_ms, pred_ibi_ms, ref_pair_indices, pred_pair_indices = extract_matched_ibi_pairs_with_indices_ms(
                    pred_beats,
                    ref_beats,
                    pred_fs=window["ppg_fs"],
                    ref_fs=window["ecg_fs"],
                    tolerance_seconds=float(stage2_cfg["matching"]["tolerance_seconds"]),
                )
                if ref_ibi_ms.size and pred_ibi_ms.size:
                    valid_pair_mask = (
                        ref_clean["ibi_mask"][ref_pair_indices].astype(bool)
                        & pred_clean["ibi_mask"][pred_pair_indices].astype(bool)
                    )
                    ref_ibi_ms = ref_ibi_ms[valid_pair_mask]
                    pred_ibi_ms = pred_ibi_ms[valid_pair_mask]
                if ref_ibi_ms.size and pred_ibi_ms.size:
                    variant_state[variant]["all_ref_ibi_ms"].extend(ref_ibi_ms.tolist())
                    variant_state[variant]["all_pred_ibi_ms"].extend(pred_ibi_ms.tolist())
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

                beat_rows.append(
                    {
                        "variant": variant,
                        "dataset": window["dataset"],
                        "subject_id": window["subject_id"],
                        "analysis_window_index": window["analysis_window_index"],
                        "start_time_s": window["start_time_s"],
                        "duration_s": window["duration_s"],
                        "num_pred_beats": float(len(pred_beats)),
                        "num_ref_beats": float(len(ref_beats)),
                        "pred_num_ibi_clean": float(len(pred_clean["ibi_clean_s"])),
                        "ref_num_ibi_clean": float(len(ref_clean["ibi_clean_s"])),
                        "pred_ibi_removed_ratio": float(pred_clean["ibi_removed_ratio"]) if not math.isnan(float(pred_clean["ibi_removed_ratio"])) else math.nan,
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
                    "variant": variant,
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

    metrics_parts: list[pd.DataFrame] = []
    for variant in variant_cfgs:
        variant_beat_frame = beat_frame.loc[beat_frame["variant"] == variant].copy()
        variant_feature_frame = feature_frame.loc[feature_frame["variant"] == variant].copy()
        beat_summary = compute_precision_recall_f1(
            int(variant_state[variant]["tp"]),
            int(variant_state[variant]["fp"]),
            int(variant_state[variant]["fn"]),
        )
        beat_summary_row = {
            "variant": variant,
            "task": "beat_detection",
            "metric_group": "summary",
            "precision": beat_summary["precision"],
            "recall": beat_summary["recall"],
            "f1": beat_summary["f1"],
            "beat_count_error": float(variant_beat_frame["beat_count_error"].mean()) if not variant_beat_frame.empty else math.nan,
        }
        ibi_summary = compute_ibi_error_metrics(
            np.asarray(variant_state[variant]["all_ref_ibi_ms"], dtype=float),
            np.asarray(variant_state[variant]["all_pred_ibi_ms"], dtype=float),
        )
        ibi_summary_row = {
            "variant": variant,
            "task": "ibi_error",
            "metric_group": "summary",
            "ibi_mae_ms": ibi_summary["ibi_mae_ms"],
            "ibi_rmse_ms": ibi_summary["ibi_rmse_ms"],
            "num_valid_ibi_pairs": ibi_summary["num_valid_ibi_pairs"],
        }
        feature_summary = summarize_feature_metrics(
            variant_feature_frame,
            feature_names=FEATURE_NAMES,
            ref_prefix="ref_",
            pred_prefix="pred_",
        )
        feature_summary["variant"] = variant
        feature_summary["task"] = "feature_comparison"
        feature_summary["metric_group"] = "summary"
        metrics_parts.extend([pd.DataFrame([beat_summary_row]), pd.DataFrame([ibi_summary_row]), feature_summary])

    metrics_frame = pd.concat(metrics_parts, ignore_index=True, sort=False)

    print("Stage 2 baseline completed.")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"Subjects evaluated: {len(eval_subjects)}")
    print(f"Analysis windows: {int(len(beat_frame) / max(len(variant_cfgs), 1))}")
    for variant in variant_cfgs:
        variant_metrics = metrics_frame.loc[
            (metrics_frame["variant"] == variant) & (metrics_frame["metric_group"] == "summary")
        ].copy()
        beat_summary_row = variant_metrics.loc[variant_metrics["task"] == "beat_detection"].iloc[0].to_dict()
        ibi_summary_row = variant_metrics.loc[variant_metrics["task"] == "ibi_error"].iloc[0].to_dict()
        print(f"variant: {variant}")
        print("  task: beat_detection")
        for key in ("precision", "recall", "f1", "beat_count_error"):
            value = beat_summary_row[key]
            print(f"    {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"    {key}: {value}")
        print("  task: ibi_error")
        for key in ("ibi_mae_ms", "ibi_rmse_ms", "num_valid_ibi_pairs"):
            value = ibi_summary_row[key]
            print(f"    {key}: {value:.4f}" if isinstance(value, float) and not math.isnan(value) else f"    {key}: {value}")
        feature_subset = variant_metrics.loc[
            (variant_metrics["task"] == "feature_comparison")
            & (variant_metrics["feature"].isin(FEATURE_HIGHLIGHTS))
        ]
        print("  task: feature_comparison")
        for _, row in feature_subset.iterrows():
            mae = row["mae"]
            pearson_r = row["pearson_r"]
            print(
                f"    {row['feature']}: mae={mae:.4f}, pearson_r={pearson_r:.4f}"
                if isinstance(mae, float) and not math.isnan(mae)
                else f"    {row['feature']}: mae={mae}, pearson_r={pearson_r}"
            )

    save_csv = bool(output_cfg.get("save_csv", False))
    save_error_cases = bool(debug_cfg.get("save_error_cases", False))
    if save_csv or save_error_cases:
        output_dir = Path(output_cfg["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

    if save_csv:
        beats_path = output_dir / f"{dataset_cfg['name']}_stage2_beats.csv"
        features_path = output_dir / f"{dataset_cfg['name']}_stage2_features.csv"
        metrics_path = output_dir / f"{dataset_cfg['name']}_stage2_metrics.csv"
        beat_frame.to_csv(beats_path, index=False)
        feature_frame.to_csv(features_path, index=False)
        metrics_frame.to_csv(metrics_path, index=False)
        print(f"Saved beats to: {beats_path}")
        print(f"Saved features to: {features_path}")
        print(f"Saved metrics to: {metrics_path}")

    if save_error_cases:
        error_frame = summarize_error_cases(beat_frame, max_cases_per_variant=int(debug_cfg.get("max_cases_per_variant", 10)))
        error_path = output_dir / f"{dataset_cfg['name']}_stage2_error_cases.csv"
        error_frame.to_csv(error_path, index=False)
        print(f"Saved error cases to: {error_path}")


if __name__ == "__main__":
    main()
