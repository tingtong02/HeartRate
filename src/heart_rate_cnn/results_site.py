from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DATASET_LABELS: dict[str, str] = {
    "ppg_dalia": "PPG-DaLiA",
    "wesad": "WESAD",
}

DEFAULT_SITE_OUTPUT_DIR = Path("web/public/data")
OUTPUT_ROOT_DIR = Path("outputs")

KNOWN_SCOPE_BY_LABEL: dict[str, str] = {
    "bounded_medium6_seed42": "validation",
    "fusion_balanced_v1": "analysis-only",
    "fusion_balanced_v1_canonical": "analysis-only",
    "stage5_tuning": "analysis-only",
}

STAGE4_TIMELINE_COLUMNS: tuple[str, ...] = (
    "window_index",
    "start_time_s",
    "duration_s",
    "selected_hr_bpm",
    "selected_hr_is_valid",
    "ml_signal_quality_score",
    "quality_gate_passed",
    "quality_gate_reason",
    "hr_event_flag",
    "hr_event_type_summary",
    "irregular_pulse_flag",
    "irregular_pulse_score",
    "anomaly_flag",
    "anomaly_score",
    "stage4_suspicion_flag",
    "stage4_suspicion_score",
    "stage4_suspicion_type_summary",
    "stage4_reason_code",
    "proxy_abnormal_target",
    "proxy_abnormal_support_flag",
)

STAGE5_TIMELINE_COLUMNS: tuple[str, ...] = (
    "window_index",
    "start_time_s",
    "duration_s",
    "resp_rate_ref_bpm",
    "resp_rate_ref_valid_flag",
    "resp_reference_reason",
    "resp_rate_baseline_bpm",
    "resp_rate_pred_bpm",
    "resp_confidence",
    "resp_validity_flag",
    "selected_hr_bpm",
    "selected_hr_is_valid",
    "ml_signal_quality_score",
    "motion_flag",
    "validity_flag",
    "hr_event_flag",
    "irregular_pulse_flag",
    "anomaly_score",
    "stage4_suspicion_flag",
    "stage4_suspicion_score",
    "stage4_suspicion_type_summary",
)

REFERENCE_DOCS: tuple[dict[str, str], ...] = (
    {"label": "README", "path": "README.md"},
    {"label": "Project Tasks", "path": "docs/PROJECT_TASKS.md"},
    {"label": "Stage 3 Runbook", "path": "docs/STAGE3_RUNBOOK.md"},
    {"label": "Stage 4 Runbook", "path": "docs/STAGE4_RUNBOOK.md"},
    {"label": "Stage 5 Runbook", "path": "docs/STAGE5_RUNBOOK.md"},
    {"label": "Usage Guide", "path": "docs/USAGE_GUIDE.md"},
    {"label": "Handoff", "path": "docs/HANDOFF.md"},
)

REFERENCE_SCRIPTS: tuple[dict[str, str], ...] = (
    {"label": "Stage 4 Source Prep", "path": "scripts/prepare_stage4_sources.py"},
    {"label": "Stage 4 Full", "path": "scripts/run_stage4_full.py"},
    {"label": "Stage 5 Source Prep", "path": "scripts/prepare_stage5_sources.py"},
    {"label": "Stage 5 Train", "path": "scripts/run_stage5_train_cnn.py"},
    {"label": "Stage 5 Full", "path": "scripts/run_stage5_full.py"},
    {"label": "Results Site Export", "path": "scripts/build_results_site_data.py"},
)

REFERENCE_CONFIGS: tuple[dict[str, str], ...] = (
    {"label": "Base Config", "path": "configs/base.yaml"},
    {"label": "Stage 3 Enhanced", "path": "configs/eval/hr_stage3_enhanced.yaml"},
    {"label": "Stage 4 Full", "path": "configs/eval/hr_stage4_full.yaml"},
    {"label": "Stage 5 Eval", "path": "configs/eval/hr_stage5.yaml"},
    {"label": "Stage 5 CNN", "path": "configs/eval/hr_stage5_cnn.yaml"},
)


@dataclass(frozen=True)
class ArtifactInfo:
    path: str
    scope: str
    label: str
    dataset: str
    stage: str
    artifact_type: str
    size_bytes: int
    modified_at_utc: str


def classify_validation_label(label: str) -> str:
    normalized = str(label).strip()
    if normalized in KNOWN_SCOPE_BY_LABEL:
        return KNOWN_SCOPE_BY_LABEL[normalized]
    if normalized.startswith("fusion_") or "analysis" in normalized:
        return "analysis-only"
    return "validation"


def _safe_float(value: Any, digits: int = 6) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, int):
        return int(value)
    if pd.isna(value):
        return None
    return value


def _frame_records(frame: pd.DataFrame, columns: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    subset = frame if columns is None else frame.loc[:, [column for column in columns if column in frame.columns]]
    records: list[dict[str, Any]] = []
    for raw_record in subset.to_dict(orient="records"):
        records.append({str(key): _json_safe(value) for key, value in raw_record.items()})
    return records


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path, usecols: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(usecols or []))
    return pd.read_csv(path, usecols=list(usecols) if usecols is not None else None)


def _parse_output_filename(file_name: str) -> tuple[str, str, str]:
    stage_match = re.match(r"(?P<dataset>ppg_dalia|wesad)_(?P<stage>stage[0-5])_(?P<artifact>.+)\.csv$", file_name)
    if stage_match:
        return (
            str(stage_match.group("dataset")),
            str(stage_match.group("stage")),
            str(stage_match.group("artifact")),
        )
    model_match = re.match(r"(?P<dataset>ppg_dalia|wesad)_(?P<artifact>stage5_.+)\.(pt|json)$", file_name)
    if model_match:
        return str(model_match.group("dataset")), "stage5", str(model_match.group("artifact"))
    return "", "", ""


def _artifact_from_path(path: Path, outputs_root: Path) -> ArtifactInfo:
    relative_path = path.relative_to(outputs_root)
    dataset, stage, artifact_type = _parse_output_filename(path.name)
    label = ""
    scope = "canonical"

    if relative_path.parts[0] == "validation" and len(relative_path.parts) >= 2:
        label = str(relative_path.parts[1])
        scope = classify_validation_label(label)
    elif relative_path.parts[0] == "cache":
        scope = "cache"
        if len(relative_path.parts) >= 3 and relative_path.parts[2] in DATASET_LABELS:
            dataset = str(relative_path.parts[2])
        if len(relative_path.parts) >= 2:
            stage = str(relative_path.parts[1])
        artifact_type = "manifest" if path.suffix == ".json" else "cache_blob"
    elif relative_path.parts[0] == "models":
        scope = "canonical"
        if not dataset and len(relative_path.parts) >= 2:
            dataset, stage, artifact_type = _parse_output_filename(path.name)
        artifact_type = "model_config" if path.suffix == ".json" else "model_checkpoint"
    elif path.name.endswith("_stage5_tuning_results.csv"):
        scope = "analysis-only"
        label = "stage5_tuning"

    stat_result = path.stat()
    modified_at_utc = datetime.fromtimestamp(stat_result.st_mtime, timezone.utc).isoformat()
    return ArtifactInfo(
        path=str(relative_path),
        scope=scope,
        label=label,
        dataset=dataset,
        stage=stage,
        artifact_type=artifact_type,
        size_bytes=int(stat_result.st_size),
        modified_at_utc=modified_at_utc,
    )


def build_artifact_inventory(outputs_root: Path) -> list[dict[str, Any]]:
    if not outputs_root.exists():
        return []
    inventory: list[dict[str, Any]] = []
    for path in sorted(outputs_root.rglob("*")):
        if not path.is_file():
            continue
        inventory.append(_json_safe(_artifact_from_path(path, outputs_root).__dict__))
    return inventory


def _best_stage1_method(stage1_metrics: pd.DataFrame) -> dict[str, Any]:
    if stage1_metrics.empty or "mae" not in stage1_metrics.columns:
        return {}
    ordered = stage1_metrics.sort_values(by=["mae", "rmse", "method"], ascending=[True, True, True]).reset_index(drop=True)
    row = ordered.iloc[0]
    return {
        "method": str(row["method"]),
        "mae": _safe_float(row.get("mae")),
        "rmse": _safe_float(row.get("rmse")),
        "pearson_r": _safe_float(row.get("pearson_r")),
        "num_valid_windows": _safe_float(row.get("num_valid_windows")),
    }


def _select_stage4_rows(stage4_full_metrics: pd.DataFrame, split_name: str) -> pd.DataFrame:
    frame = stage4_full_metrics.copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"] == split_name].copy()
    return frame.reset_index(drop=True)


def _best_stage4_conclusion(stage4_full_metrics: pd.DataFrame) -> dict[str, Any]:
    eval_rows = _select_stage4_rows(stage4_full_metrics, "eval")
    comparison_rows = eval_rows.loc[eval_rows["metric_group"] == "stage3_comparison"].copy()
    if comparison_rows.empty:
        return {}

    stage3_row = comparison_rows.loc[comparison_rows["method"] == "stage3_quality_only"].copy()
    irregular_row = comparison_rows.loc[comparison_rows["method"] == "stage4_irregular_default"].copy()
    anomaly_row = comparison_rows.loc[comparison_rows["method"] == "stage4_anomaly_default"].copy()
    unified_row = comparison_rows.loc[comparison_rows["method"] == "stage4_full_default"].copy()

    strongest = ""
    strongest_label = ""
    if not irregular_row.empty and not anomaly_row.empty:
        strongest = (
            "stage4_irregular_default"
            if float(irregular_row["auprc"].iloc[0]) >= float(anomaly_row["auprc"].iloc[0])
            else "stage4_anomaly_default"
        )
    elif not irregular_row.empty:
        strongest = "stage4_irregular_default"
    elif not anomaly_row.empty:
        strongest = "stage4_anomaly_default"

    strongest_label = {
        "stage4_irregular_default": "irregular screening",
        "stage4_anomaly_default": "anomaly scoring",
    }.get(strongest, "")

    unified_beats_stage3 = False
    if not stage3_row.empty and not unified_row.empty:
        stage3_auprc = float(stage3_row["auprc"].iloc[0])
        stage3_auroc = float(stage3_row["auroc"].iloc[0])
        unified_auprc = float(unified_row["auprc"].iloc[0])
        unified_auroc = float(unified_row["auroc"].iloc[0])
        unified_beats_stage3 = unified_auprc > stage3_auprc and unified_auroc > stage3_auroc

    return {
        "strongest_stage4_standalone_method": strongest,
        "strongest_stage4_standalone_label": strongest_label,
        "stage3_quality_only": _frame_records(stage3_row)[0] if not stage3_row.empty else None,
        "stage4_irregular_default": _frame_records(irregular_row)[0] if not irregular_row.empty else None,
        "stage4_anomaly_default": _frame_records(anomaly_row)[0] if not anomaly_row.empty else None,
        "stage4_full_default": _frame_records(unified_row)[0] if not unified_row.empty else None,
        "unified_beats_stage3": unified_beats_stage3,
    }


def _stage5_summary(stage5_metrics: pd.DataFrame) -> dict[str, Any]:
    eval_rows = stage5_metrics.loc[stage5_metrics["split"] == "eval"].copy() if "split" in stage5_metrics.columns else stage5_metrics.copy()
    high_quality = eval_rows.loc[eval_rows["subset"] == "high_quality_ref_valid"].copy()
    baseline = high_quality.loc[high_quality["method"] == "resp_surrogate_fusion_baseline"].copy()
    cnn = high_quality.loc[high_quality["method"] == "stage5_resp_multitask_cnn_v1"].copy()
    if baseline.empty or cnn.empty:
        return {}
    baseline_mae = float(baseline["resp_mae_bpm"].iloc[0])
    cnn_mae = float(cnn["resp_mae_bpm"].iloc[0])
    mae_reduction_pct = ((baseline_mae - cnn_mae) / baseline_mae * 100.0) if baseline_mae > 0 else math.nan
    return {
        "baseline": _frame_records(baseline)[0],
        "cnn": _frame_records(cnn)[0],
        "mae_reduction_pct": _safe_float(mae_reduction_pct),
        "cnn_beats_baseline": bool(cnn_mae < baseline_mae),
    }


def build_overview_summary(outputs_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_LABELS:
        stage1_metrics = _read_csv(outputs_root / f"{dataset_name}_stage1_metrics.csv")
        stage3_metrics = _read_csv(outputs_root / f"{dataset_name}_stage3_enhanced_metrics.csv")
        stage4_metrics = _read_csv(outputs_root / f"{dataset_name}_stage4_full_metrics.csv")
        stage5_metrics = _read_csv(outputs_root / f"{dataset_name}_stage5_metrics.csv")
        datasets[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "stage1_best": _best_stage1_method(stage1_metrics),
            "stage3_default": _frame_records(
                stage3_metrics.loc[stage3_metrics["method"] == "gated_stage3_ml_logreg"].copy()
            ),
            "stage4_conclusion": _best_stage4_conclusion(stage4_metrics),
            "stage5_conclusion": _stage5_summary(stage5_metrics),
        }

    return {
        "project_title": "HeartRate_CNN",
        "summary": "CPU-first public-dataset PPG physiological analysis framework spanning Stage 0–5.",
        "supported_datasets": [
            {"id": dataset_name, "label": dataset_label}
            for dataset_name, dataset_label in DATASET_LABELS.items()
        ],
        "default_scope": "canonical",
        "default_dataset": "ppg_dalia",
        "stage_cards": [
            {"stage": "Stage 0", "status": "implemented", "default_path": "foundation", "focus": "loading, alignment, ECG-backed references"},
            {"stage": "Stage 1", "status": "implemented", "default_path": "stage1_frequency", "focus": "window-level heart-rate estimation"},
            {"stage": "Stage 2", "status": "implemented", "default_path": "enhanced", "focus": "beat / IBI / PRV foundation"},
            {"stage": "Stage 3", "status": "implemented", "default_path": "gated_stage3_ml_logreg", "focus": "quality-aware HR gating"},
            {"stage": "Stage 4", "status": "implemented", "default_path": "stage4_full", "focus": "event / irregular / anomaly layer"},
            {"stage": "Stage 5", "status": "implemented", "default_path": "stage5_resp_multitask_cnn_v1", "focus": "CNN respiration and multitask interface"},
        ],
        "default_paths": [
            {"stage": "Stage 1", "name": "stage1_frequency"},
            {"stage": "Stage 2", "name": "enhanced"},
            {"stage": "Stage 3", "name": "gated_stage3_ml_logreg"},
            {"stage": "Stage 3 prototype", "name": "robust_stage3c2_policy"},
            {"stage": "Stage 4B", "name": "hist_gbdt_irregular"},
            {"stage": "Stage 4C", "name": "isolation_forest_anomaly"},
            {"stage": "Stage 5", "name": "stage5_resp_multitask_cnn_v1"},
        ],
        "evidence_banners": [
            "Stage 1 default = stage1_frequency",
            "Stage 3 default = gated_stage3_ml_logreg",
            "Stage 4 unified suspiciousness is not yet a better ranking baseline than Stage 3-only quality suspiciousness",
            "Stage 5 CNN respiration clearly beats the classical surrogate baseline on canonical outputs",
        ],
        "best_supported_conclusions": [
            "Stage 4 adds a richer and more auditable suspicious-segment layer beyond Stage 3-only validity gating.",
            "Stage 4 proxy labels are repository-specific and non-clinical.",
            "Stage 4 unified suspiciousness remains more useful for interpretation and stratification than as the strongest ranking signal.",
            "Stage 5 uses chest Resp references already present in PPG-DaLiA and WESAD.",
            "Stage 5 carries forward the HR pipeline unchanged, so it does not materially degrade existing HR outputs.",
        ],
        "datasets": datasets,
    }


def build_stage1_metrics(outputs_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_LABELS:
        metrics = _read_csv(outputs_root / f"{dataset_name}_stage1_metrics.csv")
        datasets[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "rows": _frame_records(metrics),
            "best_method": _best_stage1_method(metrics),
        }
    return {"datasets": datasets}


def build_stage2_metrics(outputs_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_LABELS:
        metrics = _read_csv(outputs_root / f"{dataset_name}_stage2_metrics.csv")
        beat_summary = metrics.loc[metrics["task"].isin(["beat_detection", "ibi_error"])].copy()
        feature_rows = metrics.loc[
            (metrics["task"] == "feature_comparison")
            & metrics["feature"].isin(
                ["num_beats", "mean_hr_bpm_from_ibi", "mean_ibi_ms", "sdnn_ms", "rmssd_ms", "pnn50", "ibi_cv"]
            )
        ].copy()
        operating_points = metrics.loc[metrics["operating_point_role"].notna()].copy() if "operating_point_role" in metrics.columns else pd.DataFrame()
        datasets[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "beat_summary": _frame_records(beat_summary),
            "feature_summary": _frame_records(feature_rows),
            "operating_points": _frame_records(operating_points),
        }
    return {"datasets": datasets}


def build_stage3_metrics(outputs_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_LABELS:
        base_metrics = _read_csv(outputs_root / f"{dataset_name}_stage3_metrics.csv")
        enhanced_metrics = _read_csv(outputs_root / f"{dataset_name}_stage3_enhanced_metrics.csv")
        threshold_sweep = _read_csv(outputs_root / f"{dataset_name}_stage3_enhanced_threshold_sweep.csv")
        operating_points = _read_csv(outputs_root / f"{dataset_name}_stage3_enhanced_operating_points.csv")
        policy_sweep = _read_csv(outputs_root / f"{dataset_name}_stage3_enhanced_policy_sweep.csv")
        datasets[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "base_metrics": _frame_records(base_metrics),
            "enhanced_metrics": _frame_records(enhanced_metrics),
            "threshold_sweep": _frame_records(threshold_sweep),
            "operating_points": _frame_records(operating_points),
            "policy_sweep": _frame_records(policy_sweep),
        }
    return {"datasets": datasets}


def _build_stage4_episode_counts(event_predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if event_predictions.empty:
        return []
    valid = event_predictions.loc[event_predictions["event_validity_flag"].astype(bool)].copy()
    if valid.empty:
        return []
    grouped = (
        valid.groupby(["split", "dataset", "event_type"], sort=False)["episode_id"]
        .nunique(dropna=True)
        .reset_index(name="num_valid_episodes")
    )
    return _frame_records(grouped)


def build_stage4_metrics(outputs_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_LABELS:
        event_metrics = _read_csv(outputs_root / f"{dataset_name}_stage4_event_metrics.csv")
        irregular_metrics = _read_csv(outputs_root / f"{dataset_name}_stage4_irregular_metrics.csv")
        anomaly_metrics = _read_csv(outputs_root / f"{dataset_name}_stage4_anomaly_metrics.csv")
        full_metrics = _read_csv(outputs_root / f"{dataset_name}_stage4_full_metrics.csv")
        event_predictions = _read_csv(
            outputs_root / f"{dataset_name}_stage4_event_predictions.csv",
            usecols=["split", "dataset", "event_type", "event_validity_flag", "episode_id"],
        )
        datasets[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "event_metrics": _frame_records(event_metrics),
            "event_episode_counts": _build_stage4_episode_counts(event_predictions),
            "irregular_metrics": _frame_records(irregular_metrics),
            "anomaly_metrics": _frame_records(anomaly_metrics),
            "full_metrics": _frame_records(full_metrics),
            "comparison_rows": _frame_records(full_metrics.loc[full_metrics["metric_group"] == "stage3_comparison"].copy()),
            "stratification_rows": _frame_records(full_metrics.loc[full_metrics["metric_group"] == "stratification"].copy()),
            "conclusion": _best_stage4_conclusion(full_metrics),
        }
    return {"datasets": datasets}


def _sample_frame(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if frame.empty or frame.shape[0] <= max_points:
        return frame.copy()
    step = max(1, frame.shape[0] // max_points)
    sampled = frame.iloc[::step].copy()
    if sampled.shape[0] > max_points:
        sampled = sampled.iloc[:max_points].copy()
    return sampled.reset_index(drop=True)


def build_stage5_metrics(outputs_root: Path) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset_name in DATASET_LABELS:
        metrics = _read_csv(outputs_root / f"{dataset_name}_stage5_metrics.csv")
        tuning = _read_csv(outputs_root / f"{dataset_name}_stage5_tuning_results.csv")
        predictions = _read_csv(
            outputs_root / f"{dataset_name}_stage5_predictions.csv",
            usecols=[
                "split",
                "resp_rate_ref_bpm",
                "resp_rate_ref_valid_flag",
                "resp_rate_baseline_bpm",
                "resp_rate_pred_bpm",
                "resp_confidence",
                "validity_flag",
            ],
        )
        eval_predictions = predictions.loc[
            (predictions["split"] == "eval")
            & predictions["resp_rate_ref_valid_flag"].astype(bool)
        ].copy()
        high_quality = eval_predictions.loc[eval_predictions["validity_flag"].astype(bool)].copy()
        high_quality["baseline_abs_error"] = (high_quality["resp_rate_baseline_bpm"] - high_quality["resp_rate_ref_bpm"]).abs()
        high_quality["cnn_abs_error"] = (high_quality["resp_rate_pred_bpm"] - high_quality["resp_rate_ref_bpm"]).abs()
        scatter_sample = _sample_frame(
            high_quality.loc[:, ["resp_rate_ref_bpm", "resp_rate_baseline_bpm", "resp_rate_pred_bpm", "resp_confidence", "baseline_abs_error", "cnn_abs_error"]],
            max_points=1000,
        )

        config_path = outputs_root / "models" / "stage5" / f"{dataset_name}_stage5_resp_multitask_cnn_v1_best_config.json"
        best_config = {}
        if config_path.exists():
            best_config = json.loads(config_path.read_text(encoding="utf-8"))

        datasets[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "metrics": _frame_records(metrics),
            "tuning_rows": _frame_records(tuning),
            "best_config": _json_safe(best_config),
            "comparison": _stage5_summary(metrics),
            "scatter_sample": _frame_records(scatter_sample),
        }
    return {"datasets": datasets}


def build_experiment_data(outputs_root: Path) -> dict[str, Any]:
    validation_root = outputs_root / "validation"
    labels: list[dict[str, Any]] = []
    for label_dir in sorted(validation_root.iterdir()) if validation_root.exists() else []:
        if not label_dir.is_dir():
            continue
        label = str(label_dir.name)
        scope = classify_validation_label(label)
        datasets: dict[str, Any] = {}
        for dataset_name in DATASET_LABELS:
            stage4_full_metrics = _read_csv(label_dir / f"{dataset_name}_stage4_full_metrics.csv")
            if stage4_full_metrics.empty:
                continue
            comparison_rows = stage4_full_metrics.loc[
                (stage4_full_metrics["metric_group"] == "stage3_comparison")
                & (stage4_full_metrics["split"] == "eval")
            ].copy()
            datasets[dataset_name] = {
                "label": DATASET_LABELS[dataset_name],
                "comparison_rows": _frame_records(comparison_rows),
            }
        labels.append({"label": label, "scope": scope, "datasets": datasets})

    stage5_tuning = {}
    for dataset_name in DATASET_LABELS:
        tuning = _read_csv(outputs_root / f"{dataset_name}_stage5_tuning_results.csv")
        if tuning.empty:
            continue
        ordered = tuning.sort_values(
            by=[
                "high_quality_resp_mae_bpm",
                "high_quality_resp_rmse_bpm",
                "high_quality_resp_pearson_r",
                "window_seconds",
            ],
            ascending=[True, True, False, True],
        ).reset_index(drop=True)
        stage5_tuning[dataset_name] = {
            "label": DATASET_LABELS[dataset_name],
            "top_rows": _frame_records(ordered.head(20)),
            "all_rows": _frame_records(tuning),
        }

    return {"labels": labels, "stage5_tuning": stage5_tuning}


def _timeline_file_name(dataset_name: str, split_name: str, subject_id: str) -> str:
    return f"{dataset_name}__{split_name}__{subject_id}.json"


def export_subject_timelines(outputs_root: Path, site_output_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"stage4": {}, "stage5": {}}
    for stage_name, columns, suffix in (
        ("stage4", STAGE4_TIMELINE_COLUMNS, "stage4_full_predictions.csv"),
        ("stage5", STAGE5_TIMELINE_COLUMNS, "stage5_predictions.csv"),
    ):
        for dataset_name in DATASET_LABELS:
            prediction_path = outputs_root / f"{dataset_name}_{suffix}"
            identity_columns = ["split", "dataset", "subject_id"]
            frame = _read_csv(prediction_path, usecols=identity_columns + list(columns))
            entries: list[dict[str, Any]] = []
            if frame.empty:
                result[stage_name][dataset_name] = entries
                continue
            for (split_name, subject_id), subject_frame in frame.groupby(["split", "subject_id"], sort=False):
                ordered = subject_frame.sort_values(by=["start_time_s", "window_index"]).reset_index(drop=True)
                file_name = _timeline_file_name(dataset_name, str(split_name), str(subject_id))
                relative_path = Path("stage_timelines") / stage_name / dataset_name / file_name
                payload = {
                    "dataset": dataset_name,
                    "subject_id": str(subject_id),
                    "split": str(split_name),
                    "rows": _frame_records(ordered, columns=list(columns)),
                }
                _write_json(site_output_dir / relative_path, payload)
                entries.append(
                    {
                        "subject_id": str(subject_id),
                        "split": str(split_name),
                        "path": str(relative_path).replace("\\", "/"),
                        "num_rows": int(ordered.shape[0]),
                    }
                )
            result[stage_name][dataset_name] = entries
    return result


def build_site_manifest(
    *,
    outputs_root: Path,
    artifact_inventory: list[dict[str, Any]],
    timelines: dict[str, Any],
) -> dict[str, Any]:
    validation_labels = sorted(
        {
            artifact["label"]
            for artifact in artifact_inventory
            if artifact["scope"] in {"validation", "analysis-only"} and artifact["label"]
        }
    )
    by_scope: dict[str, int] = {}
    for artifact in artifact_inventory:
        scope = str(artifact["scope"])
        by_scope[scope] = by_scope.get(scope, 0) + 1
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "site": {
            "title": "HeartRate_CNN Results Dashboard",
            "default_dataset": "ppg_dalia",
            "default_scope": "canonical",
            "pages": [
                "Overview",
                "HR & Quality",
                "Suspicious Segments",
                "Respiration & Multitask",
                "Experiments",
                "Artifacts & Reproducibility",
            ],
        },
        "datasets": [{"id": dataset_name, "label": dataset_label} for dataset_name, dataset_label in DATASET_LABELS.items()],
        "scopes": [
            {"id": "canonical", "label": "Canonical"},
            {"id": "validation", "label": "Validation"},
            {"id": "analysis-only", "label": "Analysis Only"},
        ],
        "validation_labels": [
            {"label": label, "scope": classify_validation_label(label)}
            for label in validation_labels
        ],
        "timeline_index": timelines,
        "artifact_summary": {
            "total_count": len(artifact_inventory),
            "by_scope": by_scope,
        },
        "reference_docs": list(REFERENCE_DOCS),
        "reference_scripts": list(REFERENCE_SCRIPTS),
        "reference_configs": list(REFERENCE_CONFIGS),
        "output_roots": {
            "canonical": str(outputs_root),
            "validation": str(outputs_root / "validation"),
            "cache": str(outputs_root / "cache"),
        },
    }


def build_results_site_data(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    site_output_dir = (output_dir or DEFAULT_SITE_OUTPUT_DIR)
    if not site_output_dir.is_absolute():
        site_output_dir = repo_root / site_output_dir
    outputs_root = repo_root / OUTPUT_ROOT_DIR

    _clear_output_dir(site_output_dir)

    artifact_inventory = build_artifact_inventory(outputs_root)
    overview_summary = build_overview_summary(outputs_root)
    stage1_metrics = build_stage1_metrics(outputs_root)
    stage2_metrics = build_stage2_metrics(outputs_root)
    stage3_metrics = build_stage3_metrics(outputs_root)
    stage4_metrics = build_stage4_metrics(outputs_root)
    stage5_metrics = build_stage5_metrics(outputs_root)
    experiments = build_experiment_data(outputs_root)
    timelines = export_subject_timelines(outputs_root, site_output_dir)
    site_manifest = build_site_manifest(outputs_root=outputs_root, artifact_inventory=artifact_inventory, timelines=timelines)

    _write_json(site_output_dir / "site_manifest.json", site_manifest)
    _write_json(site_output_dir / "overview_summary.json", overview_summary)
    _write_json(site_output_dir / "stage_metrics" / "stage1.json", stage1_metrics)
    _write_json(site_output_dir / "stage_metrics" / "stage2.json", stage2_metrics)
    _write_json(site_output_dir / "stage_metrics" / "stage3.json", stage3_metrics)
    _write_json(site_output_dir / "stage_metrics" / "stage4.json", stage4_metrics)
    _write_json(site_output_dir / "stage_metrics" / "stage5.json", stage5_metrics)
    _write_json(site_output_dir / "experiments" / "experiments.json", experiments)
    _write_json(site_output_dir / "artifacts" / "artifact_inventory.json", {"artifacts": artifact_inventory})

    return {
        "output_dir": str(site_output_dir),
        "artifact_count": len(artifact_inventory),
        "timeline_subject_counts": {
            stage_name: {dataset: len(entries) for dataset, entries in stage_entries.items()}
            for stage_name, stage_entries in timelines.items()
        },
    }
