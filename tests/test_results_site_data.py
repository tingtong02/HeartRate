from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from heart_rate_cnn.results_site import build_results_site_data, build_artifact_inventory, classify_validation_label


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_classify_validation_label() -> None:
    assert classify_validation_label("bounded_medium6_seed42") == "validation"
    assert classify_validation_label("fusion_balanced_v1") == "analysis-only"
    assert classify_validation_label("custom_analysis_probe") == "analysis-only"
    assert classify_validation_label("quick_sanity_check") == "validation"


def test_artifact_inventory_separates_scopes(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    _write_csv(outputs_root / "ppg_dalia_stage4_full_metrics.csv", [{"metric_group": "stage3_comparison"}])
    _write_csv(outputs_root / "validation" / "bounded_medium6_seed42" / "ppg_dalia_stage4_full_metrics.csv", [{"metric_group": "stage3_comparison"}])
    _write_csv(outputs_root / "validation" / "fusion_balanced_v1" / "ppg_dalia_stage4_full_metrics.csv", [{"metric_group": "stage3_comparison"}])
    (outputs_root / "cache" / "stage4" / "ppg_dalia").mkdir(parents=True, exist_ok=True)
    (outputs_root / "cache" / "stage4" / "ppg_dalia" / "manifest.json").write_text('{"package_name":"x"}', encoding="utf-8")

    inventory = build_artifact_inventory(outputs_root)
    scope_by_path = {artifact["path"]: artifact["scope"] for artifact in inventory}

    assert scope_by_path["ppg_dalia_stage4_full_metrics.csv"] == "canonical"
    assert scope_by_path["validation/bounded_medium6_seed42/ppg_dalia_stage4_full_metrics.csv"] == "validation"
    assert scope_by_path["validation/fusion_balanced_v1/ppg_dalia_stage4_full_metrics.csv"] == "analysis-only"
    assert scope_by_path["cache/stage4/ppg_dalia/manifest.json"] == "cache"


def test_build_results_site_data_exports_expected_json(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    outputs_root = repo_root / "outputs"

    for dataset_name in ("ppg_dalia", "wesad"):
        _write_csv(
            outputs_root / f"{dataset_name}_stage1_metrics.csv",
            [
                {"method": "stage0_baseline", "mae": 20.0, "rmse": 30.0, "pearson_r": 0.3, "num_valid_windows": 100},
                {"method": "stage1_frequency", "mae": 10.0, "rmse": 20.0, "pearson_r": 0.7, "num_valid_windows": 80},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage2_metrics.csv",
            [
                {"variant": "baseline", "task": "beat_detection", "metric_group": "summary", "precision": 0.5, "recall": 0.4, "f1": 0.44, "beat_count_error": 10.0},
                {"variant": "enhanced", "task": "ibi_error", "metric_group": "summary", "ibi_mae_ms": 40.0, "ibi_rmse_ms": 60.0, "num_valid_ibi_pairs": 50},
                {"variant": "enhanced", "task": "feature_comparison", "metric_group": "summary", "feature": "mean_hr_bpm_from_ibi", "mae": 5.0, "pearson_r": 0.8},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage3_metrics.csv",
            [
                {"task": "quality_classification", "method": "stage3_rule_baseline", "accuracy": 0.9, "precision": 0.8, "recall": 0.9, "f1": 0.85, "num_eval_windows": 100},
                {"task": "hr_comparison", "method": "ungated_stage1_frequency", "mae": 10.0, "rmse": 20.0, "pearson_r": 0.7, "num_valid_windows": 80, "retention_ratio": 1.0},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage3_enhanced_metrics.csv",
            [
                {"task": "hr_comparison", "method": "gated_stage3_ml_logreg", "mae": 8.0, "rmse": 16.0, "pearson_r": 0.8, "num_valid_windows": 75, "retention_ratio": 0.95, "output_fraction": 0.4, "selected_threshold": 0.5},
                {"task": "policy_summary", "method": "robust_stage3c2_policy", "output_fraction": 0.5, "retention_ratio": 1.1, "frequency_fraction": 0.3, "beat_fallback_fraction": 0.1, "hold_previous_fraction": 0.01, "reject_fraction": 0.5},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage3_enhanced_threshold_sweep.csv",
            [
                {"split": "train_select", "sweep_stage": "coarse", "threshold": 0.5, "retention_ratio": 0.95, "num_valid_windows": 75, "mae": 8.0, "rmse": 16.0, "pearson_r": 0.8, "accuracy": 0.9, "precision": 0.8, "recall": 0.9, "f1": 0.85, "num_eval_windows": 100, "is_feasible_retention": True, "branch": "raw_ml"},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage3_enhanced_operating_points.csv",
            [
                {"branch": "raw_ml", "summary_type": "selected_operating_point", "threshold": 0.5, "split_used": "train_select", "selection_mode": "fine_refined", "retention_floor": 0.95, "train_retention_ratio": 0.95, "train_mae": 8.0, "train_f1": 0.85, "test_retention_ratio": 0.94, "test_mae": 8.2, "test_rmse": 16.5, "test_rule_mae": 9.0, "stable_threshold_count": 2, "selected_threshold_rank": 1, "operating_point_status": "selected"},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage3_enhanced_policy_sweep.csv",
            [
                {"profile_name": "baseline", "split": "eval", "task": "policy_sweep", "method": "robust_stage3c2_policy", "num_eval_windows": 100, "num_valid_windows": 55, "output_fraction": 0.55, "retention_ratio": 1.05, "mae": 9.0, "rmse": 18.0, "mape": 8.0, "pearson_r": 0.7, "frequency_fraction": 0.4, "beat_fallback_fraction": 0.1, "hold_previous_fraction": 0.01, "reject_fraction": 0.45, "avg_abs_jump_bpm": 3.0, "hold_count": 5, "fallback_insufficient_count": 20, "direct_quality_threshold": 0.55, "direct_jump_guard_bpm": 20.0, "fallback_min_beats": 4, "fallback_min_clean_ibi": 3, "fallback_min_quality_kept_ratio": 0.35, "hold_quality_floor": 0.45, "hold_jump_guard_bpm": 12.0, "is_feasible": True, "is_refined_selected": False, "is_baseline_profile": True},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage4_event_metrics.csv",
            [
                {"task": "event_detection", "method": "stage4_rule_events_v1", "event_type": "all_events", "split": "eval", "num_eval_windows": 100, "num_eval_events": 10, "precision": 0.8, "recall": 0.1, "f1": 0.18, "accuracy": 0.9},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage4_irregular_metrics.csv",
            [
                {"task": "irregular_pulse_screening", "method": "hist_gbdt_irregular", "split": "eval", "num_eval_windows": 100, "num_positive_targets": 40, "num_positive_predictions": 20, "accuracy": 0.7, "precision": 0.6, "recall": 0.3, "f1": 0.4, "auroc": 0.75, "auprc": 0.65, "selected_hr_valid_fraction": 0.5, "quality_gate_pass_fraction": 0.4, "support_sufficient_fraction": 0.45, "suppressed_positive_count": 15, "valid_prediction_fraction": 0.4},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage4_anomaly_metrics.csv",
            [
                {"task": "anomaly_scoring", "method": "isolation_forest_anomaly", "split": "eval", "num_eval_windows": 100, "num_positive_targets": 55, "num_positive_predictions": 5, "accuracy": 0.45, "precision": 0.8, "recall": 0.07, "f1": 0.13, "auroc": 0.61 if dataset_name == 'ppg_dalia' else 0.57, "auprc": 0.69 if dataset_name == 'ppg_dalia' else 0.60, "selected_hr_valid_fraction": 0.4, "quality_gate_pass_fraction": 0.3, "support_sufficient_fraction": 0.4, "suppressed_candidate_count": 20, "valid_prediction_fraction": 0.3, "fit_reference_fraction": 0.1},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage4_full_metrics.csv",
            [
                {"task": "stage4_full", "metric_group": "stage3_comparison", "method": "stage3_quality_only", "subgroup": "score_and_flag", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 100, "num_positive_targets": 50, "num_positive_predictions": 50, "accuracy": 0.6, "precision": 0.6, "recall": 0.7, "f1": 0.64, "auroc": 0.56, "auprc": 0.68 if dataset_name == 'ppg_dalia' else 0.61, "alert_rate": 0.5, "quality_gate_pass_fraction": 0.4, "valid_fraction": 0.4, "proxy_abnormal_rate": 0.6},
                {"task": "stage4_full", "metric_group": "stage3_comparison", "method": "stage4_irregular_default", "subgroup": "score_and_flag", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 100, "num_positive_targets": 50, "num_positive_predictions": 20, "accuracy": 0.62, "precision": 0.55, "recall": 0.2, "f1": 0.29, "auroc": 0.45 if dataset_name == 'ppg_dalia' else 0.61, "auprc": 0.63 if dataset_name == 'ppg_dalia' else 0.67, "alert_rate": 0.2, "quality_gate_pass_fraction": 0.4, "valid_fraction": 0.4, "proxy_abnormal_rate": 0.6},
                {"task": "stage4_full", "metric_group": "stage3_comparison", "method": "stage4_anomaly_default", "subgroup": "score_and_flag", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 100, "num_positive_targets": 50, "num_positive_predictions": 5, "accuracy": 0.45, "precision": 0.8, "recall": 0.07, "f1": 0.13, "auroc": 0.61 if dataset_name == 'ppg_dalia' else 0.57, "auprc": 0.69 if dataset_name == 'ppg_dalia' else 0.60, "alert_rate": 0.05, "quality_gate_pass_fraction": 0.4, "valid_fraction": 0.4, "proxy_abnormal_rate": 0.6},
                {"task": "stage4_full", "metric_group": "stage3_comparison", "method": "stage4_full_default", "subgroup": "score_and_flag", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 100, "num_positive_targets": 50, "num_positive_predictions": 25, "accuracy": 0.58, "precision": 0.66, "recall": 0.21, "f1": 0.32, "auroc": 0.47 if dataset_name == 'ppg_dalia' else 0.49, "auprc": 0.66 if dataset_name == 'ppg_dalia' else 0.60, "alert_rate": 0.25, "quality_gate_pass_fraction": 0.4, "valid_fraction": 0.4, "proxy_abnormal_rate": 0.6},
                {"task": "stage4_full", "metric_group": "stratification", "method": "stage4_stratification", "subgroup": "normal_valid", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 20, "proxy_abnormal_rate": 0.2, "alert_rate": 0.0},
                {"task": "stage4_full", "metric_group": "stratification", "method": "stage4_stratification", "subgroup": "multi_flag_valid", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 10, "proxy_abnormal_rate": 0.8, "alert_rate": 1.0},
                {"task": "stage4_full", "metric_group": "unified", "method": "stage4_full_default", "split": "eval", "target_name": "proxy_abnormal_union", "num_eval_windows": 100, "num_positive_targets": 50, "num_positive_predictions": 25, "accuracy": 0.58, "precision": 0.66, "recall": 0.21, "f1": 0.32, "auroc": 0.47 if dataset_name == 'ppg_dalia' else 0.49, "auprc": 0.66 if dataset_name == 'ppg_dalia' else 0.60, "alert_rate": 0.25, "quality_gate_pass_fraction": 0.4, "valid_fraction": 0.4, "proxy_abnormal_rate": 0.6},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage4_event_predictions.csv",
            [
                {"split": "eval", "dataset": dataset_name, "event_type": "tachycardia_event", "event_validity_flag": True, "episode_id": 1},
                {"split": "eval", "dataset": dataset_name, "event_type": "tachycardia_event", "event_validity_flag": True, "episode_id": 1},
                {"split": "eval", "dataset": dataset_name, "event_type": "abrupt_change_event", "event_validity_flag": True, "episode_id": 2},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage4_full_predictions.csv",
            [
                {"split": "eval", "dataset": dataset_name, "subject_id": "S1", "window_index": 0, "start_time_s": 0.0, "duration_s": 8.0, "selected_hr_bpm": 70.0, "selected_hr_is_valid": True, "ml_signal_quality_score": 0.8, "quality_gate_passed": True, "quality_gate_reason": "pass", "hr_event_flag": False, "hr_event_type_summary": "", "irregular_pulse_flag": True, "irregular_pulse_score": 0.7, "anomaly_flag": False, "anomaly_score": 0.4, "stage4_suspicion_flag": True, "stage4_suspicion_score": 0.7, "stage4_suspicion_type_summary": "irregular", "stage4_reason_code": "irregular_pulse_suspicion", "proxy_abnormal_target": True, "proxy_abnormal_support_flag": True},
                {"split": "eval", "dataset": dataset_name, "subject_id": "S1", "window_index": 1, "start_time_s": 2.0, "duration_s": 8.0, "selected_hr_bpm": 72.0, "selected_hr_is_valid": True, "ml_signal_quality_score": 0.9, "quality_gate_passed": True, "quality_gate_reason": "pass", "hr_event_flag": False, "hr_event_type_summary": "", "irregular_pulse_flag": False, "irregular_pulse_score": 0.2, "anomaly_flag": True, "anomaly_score": 0.9, "stage4_suspicion_flag": True, "stage4_suspicion_score": 0.9, "stage4_suspicion_type_summary": "anomaly", "stage4_reason_code": "anomaly_suspicion", "proxy_abnormal_target": False, "proxy_abnormal_support_flag": True},
            ],
        )
        baseline_mae = 9.0 if dataset_name == "ppg_dalia" else 6.5
        cnn_mae = 2.4 if dataset_name == "ppg_dalia" else 2.9
        _write_csv(
            outputs_root / f"{dataset_name}_stage5_metrics.csv",
            [
                {"task": "stage5_respiration", "method": "resp_surrogate_fusion_baseline", "split": "eval", "subset": "high_quality_ref_valid", "num_eval_windows": 100, "resp_mae_bpm": baseline_mae, "resp_rmse_bpm": 10.0, "resp_pearson_r": 0.1, "within_3_bpm_rate": 0.1, "resp_valid_auroc": None, "resp_valid_auprc": None, "hr_selected_hr_match_rate": 1.0, "stage4_suspicion_match_rate": 1.0},
                {"task": "stage5_respiration", "method": "stage5_resp_multitask_cnn_v1", "split": "eval", "subset": "high_quality_ref_valid", "num_eval_windows": 100, "resp_mae_bpm": cnn_mae, "resp_rmse_bpm": 3.5, "resp_pearson_r": 0.6, "within_3_bpm_rate": 0.7, "resp_valid_auroc": 0.8, "resp_valid_auprc": 0.7, "hr_selected_hr_match_rate": 1.0, "stage4_suspicion_match_rate": 1.0},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage5_tuning_results.csv",
            [
                {"phase": "A", "dataset": dataset_name, "window_seconds": 32.0, "channel_set": "ppg_acc", "base_width": 32, "dropout": 0.1, "learning_rate": 0.001, "batch_size": 64, "weight_decay": 0.0, "kernel_sizes": "[7,5,5]", "huber_delta": 1.0, "rr_loss_weight": 1.0, "validity_loss_weight": 0.25, "max_epochs": 12, "patience": 4, "resp_validity_threshold": 0.5, "torch_num_threads": 8, "best_epoch": 3, "high_quality_resp_mae_bpm": 2.8, "high_quality_resp_rmse_bpm": 3.5, "high_quality_resp_pearson_r": 0.5, "predicted_valid_resp_mae_bpm": 2.7, "predicted_valid_resp_rmse_bpm": 3.4, "predicted_valid_resp_pearson_r": 0.5, "predicted_valid_coverage": 0.8},
            ],
        )
        _write_csv(
            outputs_root / f"{dataset_name}_stage5_predictions.csv",
            [
                {"split": "eval", "dataset": dataset_name, "subject_id": "S1", "window_index": 0, "start_time_s": 0.0, "duration_s": 32.0, "resp_rate_ref_bpm": 18.0, "resp_rate_ref_valid_flag": True, "resp_reference_reason": "reference_valid", "resp_rate_baseline_bpm": 10.0, "resp_rate_pred_bpm": 17.5, "resp_confidence": 0.8, "resp_validity_flag": True, "selected_hr_bpm": 70.0, "selected_hr_is_valid": True, "ml_signal_quality_score": 0.8, "motion_flag": False, "validity_flag": True, "hr_event_flag": False, "irregular_pulse_flag": False, "anomaly_score": 0.3, "stage4_suspicion_flag": False, "stage4_suspicion_score": 0.2, "stage4_suspicion_type_summary": ""},
                {"split": "eval", "dataset": dataset_name, "subject_id": "S1", "window_index": 1, "start_time_s": 4.0, "duration_s": 32.0, "resp_rate_ref_bpm": 20.0, "resp_rate_ref_valid_flag": True, "resp_reference_reason": "reference_valid", "resp_rate_baseline_bpm": 12.0, "resp_rate_pred_bpm": 20.5, "resp_confidence": 0.9, "resp_validity_flag": True, "selected_hr_bpm": 72.0, "selected_hr_is_valid": True, "ml_signal_quality_score": 0.85, "motion_flag": False, "validity_flag": True, "hr_event_flag": False, "irregular_pulse_flag": True, "anomaly_score": 0.6, "stage4_suspicion_flag": True, "stage4_suspicion_score": 0.7, "stage4_suspicion_type_summary": "irregular"},
            ],
        )
        model_dir = outputs_root / "models" / "stage5"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"{dataset_name}_stage5_resp_multitask_cnn_v1_best_config.json").write_text(
            json.dumps({"window_seconds": 32.0, "channel_set": "ppg_acc", "base_width": 32, "resp_validity_threshold": 0.5}),
            encoding="utf-8",
        )
        (model_dir / f"{dataset_name}_stage5_resp_multitask_cnn_v1_best.pt").write_bytes(b"FAKE")

    _write_csv(
        outputs_root / "validation" / "bounded_medium6_seed42" / "ppg_dalia_stage4_full_metrics.csv",
        [{"task": "stage4_full", "metric_group": "stage3_comparison", "method": "stage3_quality_only", "split": "eval", "target_name": "proxy_abnormal_union", "auprc": 0.5, "auroc": 0.6}],
    )
    _write_csv(
        outputs_root / "validation" / "fusion_balanced_v1_canonical" / "wesad_stage4_full_metrics.csv",
        [{"task": "stage4_full", "metric_group": "stage3_comparison", "method": "stage4_full_default", "split": "eval", "target_name": "proxy_abnormal_union", "auprc": 0.55, "auroc": 0.5}],
    )
    cache_manifest = outputs_root / "cache" / "stage4" / "ppg_dalia" / "source" / "sample.json"
    cache_manifest.parent.mkdir(parents=True, exist_ok=True)
    cache_manifest.write_text(json.dumps({"package_name": "quality_aware_source_package", "dataset_name": "ppg_dalia"}), encoding="utf-8")

    output_dir = repo_root / "web" / "public" / "data"
    summary = build_results_site_data(repo_root=repo_root, output_dir=output_dir)

    assert summary["artifact_count"] > 0
    manifest = json.loads((output_dir / "site_manifest.json").read_text(encoding="utf-8"))
    overview = json.loads((output_dir / "overview_summary.json").read_text(encoding="utf-8"))
    stage4 = json.loads((output_dir / "stage_metrics" / "stage4.json").read_text(encoding="utf-8"))
    experiments = json.loads((output_dir / "experiments" / "experiments.json").read_text(encoding="utf-8"))
    timeline = json.loads((output_dir / "stage_timelines" / "stage4" / "ppg_dalia" / "ppg_dalia__eval__S1.json").read_text(encoding="utf-8"))

    assert manifest["site"]["default_scope"] == "canonical"
    assert any(item["scope"] == "validation" for item in experiments["labels"])
    assert any(item["scope"] == "analysis-only" for item in experiments["labels"])
    assert overview["datasets"]["ppg_dalia"]["stage1_best"]["method"] == "stage1_frequency"
    assert stage4["datasets"]["wesad"]["conclusion"]["strongest_stage4_standalone_method"] == "stage4_irregular_default"
    assert timeline["subject_id"] == "S1"
    assert timeline["rows"][0]["quality_gate_passed"] is True
