from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from joblib import Parallel, delayed
from scipy import signal

from heart_rate_cnn.metrics import compute_hr_metrics
from heart_rate_cnn.preprocess import (
    normalize_signal,
    preprocess_ppg_stage1,
    resample_signal,
    trim_record_to_common_duration,
)
from heart_rate_cnn.stage1_hr import estimate_hr_frequency_stage1
from heart_rate_cnn.stage2_beat import (
    clean_ibi_series,
    compute_beat_quality_proxy,
    compute_time_domain_prv_features,
    detect_beats_in_window,
    extract_ibi_from_beats,
    preprocess_ppg_for_beats,
)
from heart_rate_cnn.stage4_anomaly import build_anomaly_predictions, fit_isolation_forest_anomaly_model
from heart_rate_cnn.stage4_events import build_stage4_event_predictions
from heart_rate_cnn.stage4_features import (
    STAGE4_IDENTITY_COLUMNS,
    make_loader,
    prepare_quality_aware_source_package,
    prepare_stage4_feature_package,
    resolve_stage4_output_dir,
)
from heart_rate_cnn.stage4_full import build_stage4_full_predictions, collapse_stage4_event_predictions
from heart_rate_cnn.stage4_irregular import (
    DEFAULT_MODEL_NAME as IRREGULAR_MODEL_NAME,
    build_irregular_proxy_labels,
    build_screening_predictions,
    fit_hist_gbdt_irregular_classifier,
    predict_hist_gbdt_irregular_scores,
)
from heart_rate_cnn.stage5_multitask import aggregate_stage4_context_to_windows
from heart_rate_cnn.types import SubjectRecord

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - exercised only when torch is absent
    torch = None
    nn = None
    DataLoader = None
    Dataset = object


DEFAULT_MODEL_NAME = "stage5_resp_multitask_cnn_v1"
BASELINE_METHOD_NAME = "resp_surrogate_fusion_baseline"

STAGE5_TIME_SERIES_CHANNELS: tuple[str, ...] = (
    "ppg",
    "acc_mag",
    "riav",
    "rifv",
    "ribv",
)

STAGE5_SCALAR_COLUMNS: tuple[str, ...] = (
    "resp_rate_baseline_bpm",
    "resp_baseline_confidence",
    "ml_signal_quality_score",
    "stage4_suspicion_score",
    "num_beats",
    "num_ibi_clean",
    "mean_ibi_ms",
    "rmssd_ms",
    "sdnn_ms",
    "pnn50",
    "support_sufficient_flag",
    "selected_hr_is_valid",
)


def _require_torch() -> None:
    if torch is None or nn is None or DataLoader is None:
        raise ImportError("PyTorch is required for Stage 5. Install torch in HeartRate_env before running Stage 5.")


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return bool(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def _stage5_cache_defaults(cache_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cache_cfg or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "cache_dir": str(cfg.get("cache_dir", "outputs/cache/stage5")),
        "rebuild": bool(cfg.get("rebuild", False)),
        "schema_version": str(cfg.get("schema_version", "stage5_v1")),
    }


def collect_stage5_window_seconds(stage5_cfg: dict[str, Any]) -> list[float]:
    tuning_cfg = stage5_cfg.get("tuning", {})
    candidates = [float(stage5_cfg.get("window_seconds", 32.0))]
    candidates.extend(float(value) for value in tuning_cfg.get("window_seconds_candidates", [32.0, 48.0]))
    unique_values = sorted({float(value) for value in candidates})
    return unique_values


def _stage5_cache_artifact_paths(
    *,
    cache_cfg: dict[str, Any],
    dataset_name: str,
    package_subdir: str,
    cache_key: str,
) -> tuple[Path, Path]:
    cache_root = Path(str(cache_cfg["cache_dir"])).expanduser()
    package_dir = cache_root / str(dataset_name) / str(package_subdir)
    return package_dir / f"{cache_key}.joblib", package_dir / f"{cache_key}.json"


def _read_cache_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _write_cache_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _attach_cache_metadata(
    package: dict[str, Any],
    *,
    cache_key: str,
    config_hash: str,
    cache_status: str,
    cache_path: Path | None,
    manifest_path: Path | None,
    elapsed_seconds: float,
    manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(package)
    enriched["cache_key"] = cache_key
    enriched["config_hash"] = config_hash
    enriched["cache_status"] = cache_status
    enriched["cache_path"] = str(cache_path) if cache_path is not None else ""
    enriched["manifest_path"] = str(manifest_path) if manifest_path is not None else ""
    enriched["elapsed_seconds"] = float(elapsed_seconds)
    if manifest is not None:
        enriched["manifest"] = manifest
    return enriched


def _load_or_build_stage5_package(
    *,
    dataset_name: str,
    package_subdir: str,
    package_name: str,
    cache_cfg: dict[str, Any] | None,
    cache_payload: dict[str, Any],
    build_fn,
    manifest_builder,
) -> dict[str, Any]:
    normalized_cache_cfg = _stage5_cache_defaults(cache_cfg)
    config_hash = _stable_hash(cache_payload)
    cache_key = config_hash
    artifact_path: Path | None = None
    manifest_path: Path | None = None

    if normalized_cache_cfg["enabled"]:
        artifact_path, manifest_path = _stage5_cache_artifact_paths(
            cache_cfg=normalized_cache_cfg,
            dataset_name=dataset_name,
            package_subdir=package_subdir,
            cache_key=cache_key,
        )
        if artifact_path.exists() and not normalized_cache_cfg["rebuild"]:
            start_time = time.perf_counter()
            package = joblib_load(artifact_path)
            elapsed_seconds = time.perf_counter() - start_time
            manifest = _read_cache_manifest(manifest_path)
            return _attach_cache_metadata(
                package,
                cache_key=cache_key,
                config_hash=config_hash,
                cache_status="reused",
                cache_path=artifact_path,
                manifest_path=manifest_path,
                elapsed_seconds=elapsed_seconds,
                manifest=manifest,
            )

    start_time = time.perf_counter()
    package = build_fn()
    elapsed_seconds = time.perf_counter() - start_time
    manifest = manifest_builder(package)
    manifest["package_name"] = package_name
    manifest["cache_key"] = cache_key
    manifest["config_hash"] = config_hash
    manifest["schema_version"] = str(normalized_cache_cfg["schema_version"])
    manifest["built_at_utc"] = datetime.now(timezone.utc).isoformat()

    if normalized_cache_cfg["enabled"] and artifact_path is not None and manifest_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(package, artifact_path)
        _write_cache_manifest(manifest_path, manifest)
        cache_status = "built"
    else:
        cache_status = "disabled"
        artifact_path = None
        manifest_path = None

    return _attach_cache_metadata(
        package,
        cache_key=cache_key,
        config_hash=config_hash,
        cache_status=cache_status,
        cache_path=artifact_path,
        manifest_path=manifest_path,
        elapsed_seconds=elapsed_seconds,
        manifest=manifest,
    )


def bandpass_resp_signal(samples: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 2) -> np.ndarray:
    signal_array = np.asarray(samples, dtype=float).reshape(-1)
    if signal_array.size < 8:
        return signal_array.copy()
    nyquist = 0.5 * fs
    if low_hz <= 0 or high_hz <= low_hz or high_hz >= nyquist:
        return signal_array.copy()
    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, signal_array)
    except ValueError:
        filtered = signal.sosfilt(sos, signal_array)
    return np.asarray(filtered, dtype=float)


def _estimate_rr_from_spectrum(resp_signal: np.ndarray, fs: float, cfg: dict[str, Any]) -> tuple[float, float]:
    signal_array = np.asarray(resp_signal, dtype=float).reshape(-1)
    if signal_array.size < max(int(round(fs * 8.0)), 16):
        return math.nan, math.nan
    min_rr_bpm = float(cfg.get("min_rr_bpm", 6.0))
    max_rr_bpm = float(cfg.get("max_rr_bpm", 42.0))
    min_hz = min_rr_bpm / 60.0
    max_hz = max_rr_bpm / 60.0
    nperseg = min(signal_array.size, int(round(float(cfg.get("welch_seconds", 24.0)) * fs)))
    nperseg = max(nperseg, min(signal_array.size, 32))
    freqs, power = signal.welch(signal_array, fs=fs, nperseg=nperseg, nfft=max(512, 2 ** int(math.ceil(math.log2(max(nperseg, 32))))))
    band_mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(band_mask):
        return math.nan, math.nan
    band_freqs = freqs[band_mask]
    band_power = power[band_mask]
    if band_power.size == 0 or np.allclose(band_power, 0.0):
        return math.nan, math.nan
    peak_index = int(np.argmax(band_power))
    peak_power = float(band_power[peak_index])
    total_band_power = float(np.sum(band_power))
    peak_fraction = peak_power / total_band_power if total_band_power > 0 else math.nan
    return float(band_freqs[peak_index] * 60.0), peak_fraction


def _estimate_rr_from_breath_peaks(resp_signal: np.ndarray, fs: float, cfg: dict[str, Any]) -> tuple[float, int]:
    signal_array = normalize_signal(resp_signal)
    if signal_array.size < max(int(round(fs * 8.0)), 16):
        return math.nan, 0
    max_rr_bpm = float(cfg.get("max_rr_bpm", 42.0))
    min_distance = max(int(round(fs * 60.0 / max_rr_bpm * 0.8)), 1)
    prominence = float(cfg.get("peak_prominence_scale", 0.15)) * max(np.std(signal_array), 1e-6)
    peaks, _ = signal.find_peaks(signal_array, distance=min_distance, prominence=prominence)
    if peaks.size < 3:
        return math.nan, int(peaks.size)
    ibi = np.diff(peaks) / fs
    ibi = ibi[ibi > 0]
    if ibi.size < 2:
        return math.nan, int(peaks.size)
    return float(60.0 / np.mean(ibi)), int(peaks.size)


def estimate_reference_rr_from_resp_window(
    resp_window: np.ndarray,
    resp_fs: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config.get("reference", config)
    target_fs = float(cfg.get("target_resp_fs", 25.0))
    resampled = resample_signal(np.asarray(resp_window, dtype=float).reshape(-1), resp_fs, target_fs)
    filtered = bandpass_resp_signal(
        normalize_signal(resampled),
        fs=target_fs,
        low_hz=float(cfg.get("bandpass_low_hz", 0.08)),
        high_hz=float(cfg.get("bandpass_high_hz", 0.70)),
        order=int(cfg.get("bandpass_order", 2)),
    )
    spectral_rr_bpm, peak_fraction = _estimate_rr_from_spectrum(filtered, target_fs, cfg)
    peak_rr_bpm, num_breaths = _estimate_rr_from_breath_peaks(filtered, target_fs, cfg)
    min_breaths = int(cfg.get("min_breaths", 3))
    max_rr_delta_bpm = float(cfg.get("max_rr_disagreement_bpm", 3.0))
    min_peak_fraction = float(cfg.get("min_peak_power_fraction", 0.15))

    if not np.isfinite(spectral_rr_bpm):
        return {
            "resp_rate_ref_bpm": math.nan,
            "resp_rate_ref_valid_flag": False,
            "resp_reference_reason": "spectral_rr_unavailable",
            "resp_reference_num_breaths": int(num_breaths),
            "resp_reference_peak_fraction": peak_fraction,
            "resp_reference_spectral_rr_bpm": spectral_rr_bpm,
            "resp_reference_peak_rr_bpm": peak_rr_bpm,
        }
    if not np.isfinite(peak_rr_bpm):
        return {
            "resp_rate_ref_bpm": math.nan,
            "resp_rate_ref_valid_flag": False,
            "resp_reference_reason": "peak_rr_unavailable",
            "resp_reference_num_breaths": int(num_breaths),
            "resp_reference_peak_fraction": peak_fraction,
            "resp_reference_spectral_rr_bpm": spectral_rr_bpm,
            "resp_reference_peak_rr_bpm": peak_rr_bpm,
        }
    if int(num_breaths) < min_breaths:
        return {
            "resp_rate_ref_bpm": math.nan,
            "resp_rate_ref_valid_flag": False,
            "resp_reference_reason": "insufficient_breaths",
            "resp_reference_num_breaths": int(num_breaths),
            "resp_reference_peak_fraction": peak_fraction,
            "resp_reference_spectral_rr_bpm": spectral_rr_bpm,
            "resp_reference_peak_rr_bpm": peak_rr_bpm,
        }
    if not np.isfinite(peak_fraction) or peak_fraction < min_peak_fraction:
        return {
            "resp_rate_ref_bpm": math.nan,
            "resp_rate_ref_valid_flag": False,
            "resp_reference_reason": "weak_resp_spectrum",
            "resp_reference_num_breaths": int(num_breaths),
            "resp_reference_peak_fraction": peak_fraction,
            "resp_reference_spectral_rr_bpm": spectral_rr_bpm,
            "resp_reference_peak_rr_bpm": peak_rr_bpm,
        }
    if abs(float(spectral_rr_bpm) - float(peak_rr_bpm)) > max_rr_delta_bpm:
        return {
            "resp_rate_ref_bpm": math.nan,
            "resp_rate_ref_valid_flag": False,
            "resp_reference_reason": "spectral_peak_disagreement",
            "resp_reference_num_breaths": int(num_breaths),
            "resp_reference_peak_fraction": peak_fraction,
            "resp_reference_spectral_rr_bpm": spectral_rr_bpm,
            "resp_reference_peak_rr_bpm": peak_rr_bpm,
        }

    ref_rr_bpm = 0.5 * (float(spectral_rr_bpm) + float(peak_rr_bpm))
    return {
        "resp_rate_ref_bpm": ref_rr_bpm,
        "resp_rate_ref_valid_flag": True,
        "resp_reference_reason": "reference_valid",
        "resp_reference_num_breaths": int(num_breaths),
        "resp_reference_peak_fraction": peak_fraction,
        "resp_reference_spectral_rr_bpm": spectral_rr_bpm,
        "resp_reference_peak_rr_bpm": peak_rr_bpm,
    }


def _interpolate_to_grid(times_s: np.ndarray, values: np.ndarray, grid_times_s: np.ndarray) -> np.ndarray:
    if times_s.size < 2 or values.size < 2:
        return np.zeros_like(grid_times_s, dtype=float)
    order = np.argsort(times_s)
    sorted_times = np.asarray(times_s, dtype=float)[order]
    sorted_values = np.asarray(values, dtype=float)[order]
    unique_times, unique_indices = np.unique(sorted_times, return_index=True)
    if unique_times.size < 2:
        return np.zeros_like(grid_times_s, dtype=float)
    unique_values = sorted_values[unique_indices]
    interpolated = np.interp(grid_times_s, unique_times, unique_values, left=unique_values[0], right=unique_values[-1])
    return np.asarray(interpolated, dtype=float)


def _local_trough_values(signal_window: np.ndarray, beat_indices: np.ndarray) -> np.ndarray:
    samples = np.asarray(signal_window, dtype=float).reshape(-1)
    beats = np.asarray(beat_indices, dtype=int)
    if beats.size == 0:
        return np.array([], dtype=float)
    trough_values: list[float] = []
    for index, beat_index in enumerate(beats):
        left_bound = 0 if index == 0 else int(round(0.5 * (beats[index - 1] + beat_index)))
        right_bound = samples.size if index == beats.size - 1 else int(round(0.5 * (beat_index + beats[index + 1])))
        if right_bound <= left_bound:
            trough_values.append(float(samples[beat_index]))
            continue
        trough_values.append(float(np.min(samples[left_bound:right_bound])))
    return np.asarray(trough_values, dtype=float)


def _estimate_rr_from_surrogate_series(series_values: np.ndarray, fs: float, cfg: dict[str, Any]) -> tuple[float, float]:
    filtered = bandpass_resp_signal(
        normalize_signal(series_values),
        fs=fs,
        low_hz=float(cfg.get("bandpass_low_hz", 0.08)),
        high_hz=float(cfg.get("bandpass_high_hz", 0.70)),
        order=int(cfg.get("bandpass_order", 2)),
    )
    rr_bpm, peak_fraction = _estimate_rr_from_spectrum(filtered, fs, cfg)
    return rr_bpm, peak_fraction


def build_respiration_surrogate_features(
    ppg_window: np.ndarray,
    fs: float,
    input_fs: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    baseline_cfg = config.get("baseline", config)
    beat_cfg = config.get("beat", {"variant_mode": "enhanced"})
    ibi_cfg = config.get("ibi", {"variant_mode": "enhanced", "min_clean_ibi": 3})
    quality_cfg = config.get("beat_quality", {"good_score_threshold": 0.55})

    grid_times_s = np.arange(np.asarray(ppg_window).size / fs, step=1.0 / input_fs, dtype=float)
    beats = detect_beats_in_window(ppg_window, fs=fs, config=beat_cfg)
    processed = preprocess_ppg_for_beats(ppg_window, fs=fs, config=beat_cfg, mode="refine")
    beat_quality = compute_beat_quality_proxy(
        ppg_window,
        beats,
        fs=fs,
        beat_config=beat_cfg,
        ibi_config=ibi_cfg,
        quality_config=quality_cfg,
    )
    keep_beats = beat_quality["beat_is_kept_by_quality"].astype(bool) if beat_quality["beat_is_kept_by_quality"].size else np.zeros(beats.size, dtype=bool)
    selected_beats = beats[keep_beats] if keep_beats.sum() >= int(baseline_cfg.get("min_quality_beats", 4)) else beats
    beat_times_s = selected_beats.astype(float) / fs

    ibi_s = extract_ibi_from_beats(selected_beats, fs=fs)
    clean_result = clean_ibi_series(ibi_s, ibi_cfg)
    clean_ibi = np.asarray(clean_result["ibi_clean_s"], dtype=float)
    prv = compute_time_domain_prv_features(
        clean_ibi,
        num_beats=int(selected_beats.size),
        num_ibi_raw=int(ibi_s.size),
        num_ibi_clean=int(clean_ibi.size),
    )

    peak_values = np.asarray(processed[selected_beats], dtype=float) if selected_beats.size else np.array([], dtype=float)
    trough_values = _local_trough_values(processed, selected_beats)
    rifv_times = beat_times_s[1:] if beat_times_s.size >= 2 else np.array([], dtype=float)
    rifv_values = ibi_s.astype(float)

    riav_wave = _interpolate_to_grid(beat_times_s, peak_values, grid_times_s)
    ribv_wave = _interpolate_to_grid(beat_times_s, trough_values, grid_times_s)
    rifv_wave = _interpolate_to_grid(rifv_times, rifv_values, grid_times_s)

    candidates: list[tuple[float, float]] = []
    rr_riav, conf_riav = _estimate_rr_from_surrogate_series(riav_wave, input_fs, baseline_cfg)
    if np.isfinite(rr_riav) and np.isfinite(conf_riav):
        candidates.append((rr_riav, conf_riav))
    rr_rifv, conf_rifv = _estimate_rr_from_surrogate_series(rifv_wave, input_fs, baseline_cfg)
    if np.isfinite(rr_rifv) and np.isfinite(conf_rifv):
        candidates.append((rr_rifv, conf_rifv))
    rr_ribv, conf_ribv = _estimate_rr_from_surrogate_series(ribv_wave, input_fs, baseline_cfg)
    if np.isfinite(rr_ribv) and np.isfinite(conf_ribv):
        candidates.append((rr_ribv, conf_ribv))

    if candidates:
        weights = np.asarray([max(confidence, 1e-6) for _, confidence in candidates], dtype=float)
        rr_values = np.asarray([rr_bpm for rr_bpm, _ in candidates], dtype=float)
        baseline_rr_bpm = float(np.average(rr_values, weights=weights))
        baseline_confidence = float(np.clip(np.mean(weights), 0.0, 1.0))
        baseline_reason = "surrogate_fusion_valid"
    else:
        baseline_rr_bpm = math.nan
        baseline_confidence = 0.0
        baseline_reason = "surrogate_fusion_unavailable"

    beat_positions_s = (selected_beats.astype(float) / fs).tolist()
    ibi_series_ms = (clean_ibi * 1000.0).tolist()
    support_sufficient = bool(int(selected_beats.size) >= int(baseline_cfg.get("min_beats", 4)) and int(clean_ibi.size) >= int(baseline_cfg.get("min_clean_ibi", 3)))

    return {
        "riav_waveform": normalize_signal(riav_wave).astype(np.float32),
        "rifv_waveform": normalize_signal(rifv_wave).astype(np.float32),
        "ribv_waveform": normalize_signal(ribv_wave).astype(np.float32),
        "resp_rate_baseline_bpm": baseline_rr_bpm,
        "resp_baseline_confidence": baseline_confidence,
        "resp_baseline_reason": baseline_reason,
        "num_beats": float(selected_beats.size),
        "num_ibi_raw": float(ibi_s.size),
        "num_ibi_clean": float(clean_ibi.size),
        "support_sufficient_flag": support_sufficient,
        "beat_positions_s": beat_positions_s,
        "ibi_series_ms": ibi_series_ms,
        **prv,
    }


def _build_default_irregular_predictions(
    feature_frame: pd.DataFrame,
    stage4_irregular_cfg: dict[str, Any],
) -> pd.DataFrame:
    model_cfg = stage4_irregular_cfg.get("model", {})
    threshold = float(model_cfg.get("threshold", 0.50))
    model = fit_hist_gbdt_irregular_classifier(feature_frame.loc[feature_frame["split"] == "train"].copy(), config=model_cfg)
    scores = predict_hist_gbdt_irregular_scores(model, feature_frame)
    return build_screening_predictions(
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


def _build_anomaly_base_frame(
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


def build_stage4_default_context_frame(
    *,
    loader,
    dataset_name: str,
    root_dir: str,
    train_subjects: list[str],
    eval_subjects: list[str],
    preprocess_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    stage1_cfg: dict[str, Any],
    stage3_cfg: dict[str, Any],
    stage4_cfg: dict[str, Any],
    stage4_shared_cfg: dict[str, Any],
    stage4_irregular_cfg: dict[str, Any],
    stage4_anomaly_cfg: dict[str, Any],
    stage4_full_cfg: dict[str, Any],
    stage4_cache_cfg: dict[str, Any] | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    source_package = prepare_quality_aware_source_package(
        loader=loader,
        dataset_name=dataset_name,
        root_dir=root_dir,
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        eval_cfg=eval_cfg,
        stage1_cfg=stage1_cfg,
        stage3_cfg=stage3_cfg,
        cache_cfg=stage4_cache_cfg,
    )
    train_source_frame = source_package["train_source_frame"]
    eval_source_frame = source_package["eval_source_frame"]
    train_event_predictions = build_stage4_event_predictions(train_source_frame, split_name="train", config=stage4_cfg)
    eval_event_predictions = build_stage4_event_predictions(eval_source_frame, split_name="eval", config=stage4_cfg)
    event_predictions = pd.concat([train_event_predictions, eval_event_predictions], ignore_index=True, sort=False)
    event_summary = collapse_stage4_event_predictions(event_predictions)

    feature_package = prepare_stage4_feature_package(
        loader=loader,
        dataset_name=dataset_name,
        root_dir=root_dir,
        train_subjects=train_subjects,
        eval_subjects=eval_subjects,
        preprocess_cfg=preprocess_cfg,
        stage3_cfg=stage3_cfg,
        stage4_shared_cfg=stage4_shared_cfg,
        source_package=source_package,
        cache_cfg=stage4_cache_cfg,
    )
    train_feature_frame = build_irregular_proxy_labels(feature_package["train_feature_frame"].copy(), config=stage4_irregular_cfg)
    eval_feature_frame = build_irregular_proxy_labels(feature_package["eval_feature_frame"].copy(), config=stage4_irregular_cfg)
    combined_feature_frame = pd.concat([train_feature_frame, eval_feature_frame], ignore_index=True, sort=False)
    irregular_default_predictions = _build_default_irregular_predictions(combined_feature_frame, stage4_irregular_cfg)
    anomaly_base = _build_anomaly_base_frame(combined_feature_frame, event_summary, irregular_default_predictions)
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
    full_predictions = build_stage4_full_predictions(
        feature_frame=combined_feature_frame,
        event_summary=event_summary,
        irregular_predictions=irregular_default_predictions,
        anomaly_predictions=anomaly_predictions,
        config=stage4_full_cfg,
    )
    full_predictions = full_predictions.merge(
        combined_feature_frame.loc[:, [*STAGE4_IDENTITY_COLUMNS, "motion_flag"]].copy(),
        on=list(STAGE4_IDENTITY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    full_predictions["motion_flag"] = full_predictions["motion_flag"].fillna(False).astype(bool)
    return full_predictions, source_package, feature_package


def _stage5_identity_frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(STAGE4_IDENTITY_COLUMNS))
    return pd.DataFrame(
        [{column_name: row[column_name] for column_name in STAGE4_IDENTITY_COLUMNS} for row in rows]
    )


def _normalize_channel(samples: np.ndarray) -> np.ndarray:
    array = normalize_signal(np.asarray(samples, dtype=float).reshape(-1))
    return array.astype(np.float32)


def _build_stage5_split_from_scratch(
    *,
    loader,
    dataset_name: str,
    root_dir: str,
    split_name: str,
    subject_ids: list[str],
    stage4_full_frame: pd.DataFrame,
    stage5_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    stage4_subset = stage4_full_frame.loc[stage4_full_frame["split"] == split_name].copy()
    stage4_by_subject = {
        str(subject_id): group.reset_index(drop=True)
        for subject_id, group in stage4_subset.groupby("subject_id", sort=False)
    }
    window_seconds = float(stage5_cfg["window_seconds"])
    step_seconds = float(stage5_cfg["step_seconds"])
    input_fs = float(stage5_cfg["input_fs"])
    beat_ppg_fs = float(stage5_cfg.get("beat_target_ppg_fs", 64.0))

    parallel_jobs = int(stage5_cfg.get("parallel_jobs", 0))
    if parallel_jobs <= 0:
        parallel_jobs = max(1, min(len(subject_ids), min(16, os.cpu_count() or 1)))

    def _build_subject(subject_id: str) -> tuple[pd.DataFrame, np.ndarray]:
        subject_loader = make_loader(dataset_name, root_dir)
        record = trim_record_to_common_duration(subject_loader.load_subject(subject_id))
        if record.resp is None or record.resp_fs is None:
            raise RuntimeError(f"Stage 5 requires chest Resp for subject {subject_id}.")

        ppg_input = resample_signal(record.ppg, record.ppg_fs, input_fs)
        ppg_for_beats = resample_signal(record.ppg, record.ppg_fs, beat_ppg_fs)
        if record.acc is not None and record.acc_fs is not None:
            acc_input = resample_signal(record.acc, record.acc_fs, input_fs)
            acc_mag = np.linalg.norm(np.asarray(acc_input, dtype=float), axis=1)
        else:
            acc_mag = np.zeros(ppg_input.shape[0], dtype=float)
        resp_signal = np.asarray(record.resp, dtype=float).reshape(-1)

        total = int(ppg_input.shape[0])
        window_size = int(round(window_seconds * input_fs))
        step_size = int(round(step_seconds * input_fs))
        if window_size <= 0 or step_size <= 0:
            raise ValueError("Stage 5 window and step sizes must be positive.")
        subject_rows: list[dict[str, Any]] = []
        subject_series_list: list[np.ndarray] = []

        for window_index, start_idx in enumerate(range(0, total - window_size + 1, step_size)):
            end_idx = start_idx + window_size
            start_time_s = start_idx / input_fs
            end_time_s = end_idx / input_fs
            beat_start = int(round(start_time_s * beat_ppg_fs))
            beat_end = int(round(end_time_s * beat_ppg_fs))
            resp_start = int(round(start_time_s * record.resp_fs))
            resp_end = int(round(end_time_s * record.resp_fs))

            ppg_window = np.asarray(ppg_input[start_idx:end_idx], dtype=float).reshape(-1)
            acc_window = np.asarray(acc_mag[start_idx:end_idx], dtype=float).reshape(-1)
            ppg_window_for_beats = np.asarray(ppg_for_beats[beat_start:beat_end], dtype=float).reshape(-1)
            resp_window = np.asarray(resp_signal[resp_start:resp_end], dtype=float).reshape(-1)

            resp_reference = estimate_reference_rr_from_resp_window(resp_window, float(record.resp_fs), stage5_cfg)
            surrogate_features = build_respiration_surrogate_features(
                ppg_window_for_beats,
                fs=beat_ppg_fs,
                input_fs=input_fs,
                config=stage5_cfg,
            )

            row = {
                "split": str(split_name),
                "dataset": str(record.dataset),
                "subject_id": str(record.subject_id),
                "window_index": int(window_index),
                "start_time_s": float(start_time_s),
                "duration_s": float(window_seconds),
                **resp_reference,
                **surrogate_features,
            }
            subject_rows.append(row)
            subject_series_list.append(
                np.stack(
                    [
                        _normalize_channel(ppg_window),
                        _normalize_channel(acc_window),
                        np.asarray(surrogate_features["riav_waveform"], dtype=np.float32),
                        np.asarray(surrogate_features["rifv_waveform"], dtype=np.float32),
                        np.asarray(surrogate_features["ribv_waveform"], dtype=np.float32),
                    ],
                    axis=0,
                )
            )

        metadata_frame = pd.DataFrame(subject_rows)
        stage5_identity_frame = _stage5_identity_frame_from_rows(subject_rows)
        aggregated_stage4 = aggregate_stage4_context_to_windows(
            stage4_by_subject.get(str(subject_id), pd.DataFrame()),
            stage5_identity_frame,
        )
        metadata_frame = metadata_frame.merge(
            aggregated_stage4,
            on=list(STAGE4_IDENTITY_COLUMNS),
            how="left",
            validate="one_to_one",
        )
        if "support_sufficient_flag" not in metadata_frame.columns:
            metadata_frame["support_sufficient_flag"] = False
        metadata_frame["support_sufficient_flag"] = metadata_frame["support_sufficient_flag"].fillna(False).astype(bool)
        subject_time_series = (
            np.stack(subject_series_list, axis=0).astype(np.float32)
            if subject_series_list
            else np.zeros((0, len(STAGE5_TIME_SERIES_CHANNELS), 0), dtype=np.float32)
        )
        return metadata_frame.reset_index(drop=True), subject_time_series

    if parallel_jobs == 1:
        subject_outputs = [_build_subject(str(subject_id)) for subject_id in subject_ids]
    else:
        subject_outputs = Parallel(n_jobs=parallel_jobs, prefer="processes")(
            delayed(_build_subject)(str(subject_id)) for subject_id in subject_ids
        )

    frames = [frame for frame, _ in subject_outputs if not frame.empty]
    arrays = [array for _, array in subject_outputs if array.size > 0]
    metadata_frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    time_series = np.concatenate(arrays, axis=0) if arrays else np.zeros((0, len(STAGE5_TIME_SERIES_CHANNELS), 0), dtype=np.float32)
    return metadata_frame.reset_index(drop=True), time_series


def prepare_stage5_window_package(
    *,
    loader,
    dataset_name: str,
    root_dir: str,
    train_subjects: list[str],
    eval_subjects: list[str],
    preprocess_cfg: dict[str, Any],
    eval_cfg: dict[str, Any],
    stage1_cfg: dict[str, Any],
    stage3_cfg: dict[str, Any],
    stage4_cfg: dict[str, Any],
    stage4_shared_cfg: dict[str, Any],
    stage4_irregular_cfg: dict[str, Any],
    stage4_anomaly_cfg: dict[str, Any],
    stage4_full_cfg: dict[str, Any],
    stage4_cache_cfg: dict[str, Any] | None,
    stage5_cfg: dict[str, Any],
    stage5_cache_cfg: dict[str, Any] | None = None,
    stage4_full_frame: pd.DataFrame | None = None,
    stage4_source_package: dict[str, Any] | None = None,
    stage4_feature_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_subjects_sorted = sorted(str(subject_id) for subject_id in train_subjects)
    eval_subjects_sorted = sorted(str(subject_id) for subject_id in eval_subjects)
    source_package = stage4_source_package
    if source_package is None:
        source_package = prepare_quality_aware_source_package(
            loader=loader,
            dataset_name=dataset_name,
            root_dir=root_dir,
            train_subjects=train_subjects_sorted,
            eval_subjects=eval_subjects_sorted,
            preprocess_cfg=preprocess_cfg,
            eval_cfg=eval_cfg,
            stage1_cfg=stage1_cfg,
            stage3_cfg=stage3_cfg,
            cache_cfg=stage4_cache_cfg,
        )
    feature_package = stage4_feature_package
    if feature_package is None:
        feature_package = prepare_stage4_feature_package(
            loader=loader,
            dataset_name=dataset_name,
            root_dir=root_dir,
            train_subjects=train_subjects_sorted,
            eval_subjects=eval_subjects_sorted,
            preprocess_cfg=preprocess_cfg,
            stage3_cfg=stage3_cfg,
            stage4_shared_cfg=stage4_shared_cfg,
            source_package=source_package,
            cache_cfg=stage4_cache_cfg,
        )

    cache_payload = {
        "schema_version": _stage5_cache_defaults(stage5_cache_cfg)["schema_version"],
        "package_name": "stage5_window_package",
        "dataset_name": str(dataset_name),
        "root_dir": str(root_dir),
        "train_subjects": train_subjects_sorted,
        "eval_subjects": eval_subjects_sorted,
        "preprocess": preprocess_cfg,
        "eval": eval_cfg,
        "stage1": stage1_cfg,
        "stage3": stage3_cfg,
        "stage4": stage4_cfg,
        "stage4_shared": stage4_shared_cfg,
        "stage4_irregular": stage4_irregular_cfg,
        "stage4_anomaly": stage4_anomaly_cfg,
        "stage4_full": stage4_full_cfg,
        "stage5": stage5_cfg,
        "stage4_source_package_key": str(source_package.get("cache_key", "")),
        "stage4_feature_package_key": str(feature_package.get("cache_key", "")),
    }

    def _build_package() -> dict[str, Any]:
        if stage4_full_frame is not None:
            resolved_stage4_full_frame = stage4_full_frame
            built_source_package = source_package
            built_feature_package = feature_package
        else:
            resolved_stage4_full_frame, built_source_package, built_feature_package = build_stage4_default_context_frame(
                loader=loader,
                dataset_name=dataset_name,
                root_dir=root_dir,
                train_subjects=train_subjects_sorted,
                eval_subjects=eval_subjects_sorted,
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
        train_frame, train_timeseries = _build_stage5_split_from_scratch(
            loader=loader,
            dataset_name=dataset_name,
            root_dir=root_dir,
            split_name="train",
            subject_ids=train_subjects_sorted,
            stage4_full_frame=resolved_stage4_full_frame,
            stage5_cfg=stage5_cfg,
        )
        eval_frame, eval_timeseries = _build_stage5_split_from_scratch(
            loader=loader,
            dataset_name=dataset_name,
            root_dir=root_dir,
            split_name="eval",
            subject_ids=eval_subjects_sorted,
            stage4_full_frame=resolved_stage4_full_frame,
            stage5_cfg=stage5_cfg,
        )
        return {
            "train_subjects": train_subjects_sorted,
            "eval_subjects": eval_subjects_sorted,
            "train_frame": train_frame,
            "eval_frame": eval_frame,
            "train_timeseries": train_timeseries,
            "eval_timeseries": eval_timeseries,
            "channel_names": list(STAGE5_TIME_SERIES_CHANNELS),
            "scalar_feature_columns": list(STAGE5_SCALAR_COLUMNS),
            "stage4_source_package_key": str(built_source_package.get("cache_key", "")),
            "stage4_feature_package_key": str(built_feature_package.get("cache_key", "")),
        }

    def _manifest(package: dict[str, Any]) -> dict[str, Any]:
        return {
            "dataset_name": str(dataset_name),
            "root_dir": str(root_dir),
            "train_subjects": list(train_subjects_sorted),
            "eval_subjects": list(eval_subjects_sorted),
            "window_seconds": float(stage5_cfg["window_seconds"]),
            "step_seconds": float(stage5_cfg["step_seconds"]),
            "input_fs": float(stage5_cfg["input_fs"]),
            "row_counts": {
                "train_frame": int(package["train_frame"].shape[0]),
                "eval_frame": int(package["eval_frame"].shape[0]),
            },
            "stage4_source_package_key": str(package["stage4_source_package_key"]),
            "stage4_feature_package_key": str(package["stage4_feature_package_key"]),
        }

    return _load_or_build_stage5_package(
        dataset_name=dataset_name,
        package_subdir="window_package",
        package_name="stage5_window_package",
        cache_cfg=stage5_cache_cfg,
        cache_payload=cache_payload,
        build_fn=_build_package,
        manifest_builder=_manifest,
    )


def _subsample_per_subject(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    max_per_subject: int,
    random_seed: int,
) -> np.ndarray:
    if max_per_subject <= 0:
        return indices.copy()
    rng = np.random.default_rng(int(random_seed))
    selected: list[np.ndarray] = []
    indexed = frame.iloc[indices].copy()
    indexed["_row_index"] = indices
    for _, group in indexed.groupby("subject_id", sort=False):
        subject_indices = group["_row_index"].to_numpy(dtype=int)
        if subject_indices.size <= max_per_subject:
            selected.append(subject_indices)
        else:
            selected.append(np.sort(rng.choice(subject_indices, size=max_per_subject, replace=False)))
    if not selected:
        return np.array([], dtype=int)
    return np.concatenate(selected, axis=0)


def _build_scalar_matrix(frame: pd.DataFrame, scalar_columns: list[str]) -> np.ndarray:
    values = frame.loc[:, scalar_columns].copy()
    for column_name in scalar_columns:
        if values[column_name].dtype == bool:
            values[column_name] = values[column_name].astype(float)
    return values.to_numpy(dtype=float)


@dataclass(slots=True)
class ScalarStandardizer:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std


def _fit_scalar_standardizer(values: np.ndarray) -> ScalarStandardizer:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where((np.isfinite(std)) & (std > 1e-6), std, 1.0)
    return ScalarStandardizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


class _Stage5Dataset(Dataset):
    def __init__(
        self,
        time_series: np.ndarray,
        scalar_features: np.ndarray,
        rr_targets: np.ndarray,
        validity_targets: np.ndarray,
    ) -> None:
        _require_torch()
        self.time_series = torch.as_tensor(time_series, dtype=torch.float32)
        self.scalar_features = torch.as_tensor(np.nan_to_num(scalar_features, nan=0.0), dtype=torch.float32)
        self.rr_targets = torch.as_tensor(rr_targets, dtype=torch.float32)
        self.validity_targets = torch.as_tensor(validity_targets.astype(np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.time_series.shape[0])

    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
        return (
            self.time_series[index],
            self.scalar_features[index],
            self.rr_targets[index],
            self.validity_targets[index],
        )


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Any) -> Any:
        return self.block(inputs)


class Stage5RespMultitaskCNN(nn.Module):
    def __init__(
        self,
        *,
        num_channels: int,
        num_scalar_features: int,
        base_width: int,
        dropout: float,
        kernel_sizes: list[int] | tuple[int, ...],
    ) -> None:
        super().__init__()
        widths = [int(base_width), int(base_width * 2), int(base_width * 4)]
        in_channels = num_channels
        blocks: list[nn.Module] = []
        for out_channels, kernel_size in zip(widths, kernel_sizes):
            blocks.append(_ConvBlock(in_channels, out_channels, int(kernel_size), dropout))
            in_channels = out_channels
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool1d(1))
        self.scalar_mlp = nn.Sequential(
            nn.Linear(num_scalar_features, int(base_width)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        shared_dim = int(widths[-1] + base_width)
        self.shared_mlp = nn.Sequential(
            nn.Linear(shared_dim, int(base_width * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(int(base_width * 4), int(base_width * 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.rr_head = nn.Linear(int(base_width * 2), 1)
        self.validity_head = nn.Linear(int(base_width * 2), 1)

    def forward(self, time_series: Any, scalar_features: Any) -> tuple[Any, Any]:
        encoded = self.encoder(time_series).squeeze(-1)
        scalar_encoded = self.scalar_mlp(scalar_features)
        fused = torch.cat([encoded, scalar_encoded], dim=1)
        shared = self.shared_mlp(fused)
        rr_pred = self.rr_head(shared).squeeze(-1)
        validity_logit = self.validity_head(shared).squeeze(-1)
        return rr_pred, validity_logit


def _candidate_channel_names(candidate_cfg: dict[str, Any]) -> list[str]:
    channel_set = str(candidate_cfg.get("channel_set", "ppg_acc_ri"))
    if channel_set == "ppg_acc":
        return ["ppg", "acc_mag"]
    return ["ppg", "acc_mag", "riav", "rifv", "ribv"]


def _candidate_kernel_sizes(candidate_cfg: dict[str, Any]) -> list[int]:
    kernel_sizes = candidate_cfg.get("kernel_sizes")
    if kernel_sizes is None:
        return [7, 5, 5]
    return [int(value) for value in kernel_sizes]


def _select_channel_indices(channel_names: list[str], selected_names: list[str]) -> list[int]:
    name_to_index = {name: index for index, name in enumerate(channel_names)}
    return [int(name_to_index[name]) for name in selected_names]


def _build_candidate_feature_matrices(
    package: dict[str, Any],
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    channel_names: list[str],
    scalar_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_subset = frame.iloc[indices].reset_index(drop=True)
    time_series = np.asarray(package["train_timeseries"] if frame_subset["split"].iloc[0] == "train" else package["eval_timeseries"], dtype=np.float32)
    series_subset = time_series[indices][:, channel_names, :]
    scalar_values = _build_scalar_matrix(frame_subset, scalar_columns)
    rr_targets = frame_subset["resp_rate_ref_bpm"].to_numpy(dtype=float)
    validity_targets = frame_subset["resp_rate_ref_valid_flag"].astype(bool).to_numpy()
    return series_subset, scalar_values, rr_targets, validity_targets


def _prepare_split_arrays(
    package: dict[str, Any],
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    selected_channel_names: list[str],
    scalar_columns: list[str],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    source_array = np.asarray(package["train_timeseries"] if split_name == "train" else package["eval_timeseries"], dtype=np.float32)
    name_to_index = {name: index for index, name in enumerate(package["channel_names"])}
    channel_indices = [int(name_to_index[name]) for name in selected_channel_names]
    frame_subset = frame.iloc[indices].reset_index(drop=True)
    series_subset = source_array[indices][:, channel_indices, :]
    scalar_values = _build_scalar_matrix(frame_subset, scalar_columns)
    rr_targets = frame_subset["resp_rate_ref_bpm"].to_numpy(dtype=float)
    validity_targets = frame_subset["resp_rate_ref_valid_flag"].astype(bool).to_numpy()
    return series_subset, scalar_values, rr_targets, validity_targets, frame_subset


def _evaluate_predictions(
    frame: pd.DataFrame,
    rr_predictions: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_threshold: float,
) -> dict[str, float]:
    eval_frame = frame.copy().reset_index(drop=True)
    eval_frame["resp_rate_pred_bpm"] = rr_predictions
    eval_frame["resp_confidence"] = confidence
    subset_mask = eval_frame["resp_rate_ref_valid_flag"].astype(bool) & eval_frame["validity_flag"].astype(bool)
    if subset_mask.any():
        metrics = compute_hr_metrics(
            eval_frame.loc[subset_mask, "resp_rate_ref_bpm"].to_numpy(dtype=float),
            eval_frame.loc[subset_mask, "resp_rate_pred_bpm"].to_numpy(dtype=float),
        )
    else:
        metrics = compute_hr_metrics(np.array([], dtype=float), np.array([], dtype=float))
    predicted_valid_mask = subset_mask & (eval_frame["resp_confidence"].to_numpy(dtype=float) >= float(confidence_threshold))
    if predicted_valid_mask.any():
        pred_valid_metrics = compute_hr_metrics(
            eval_frame.loc[predicted_valid_mask, "resp_rate_ref_bpm"].to_numpy(dtype=float),
            eval_frame.loc[predicted_valid_mask, "resp_rate_pred_bpm"].to_numpy(dtype=float),
        )
    else:
        pred_valid_metrics = compute_hr_metrics(np.array([], dtype=float), np.array([], dtype=float))
    coverage = float(np.mean(predicted_valid_mask[subset_mask])) if subset_mask.any() else math.nan
    return {
        "high_quality_resp_mae_bpm": metrics["mae"],
        "high_quality_resp_rmse_bpm": metrics["rmse"],
        "high_quality_resp_pearson_r": metrics["pearson_r"],
        "predicted_valid_resp_mae_bpm": pred_valid_metrics["mae"],
        "predicted_valid_resp_rmse_bpm": pred_valid_metrics["rmse"],
        "predicted_valid_resp_pearson_r": pred_valid_metrics["pearson_r"],
        "predicted_valid_coverage": coverage,
    }


def _train_one_model(
    *,
    train_series: np.ndarray,
    train_scalar: np.ndarray,
    train_rr_targets: np.ndarray,
    train_validity_targets: np.ndarray,
    val_series: np.ndarray,
    val_scalar: np.ndarray,
    val_rr_targets: np.ndarray,
    val_validity_targets: np.ndarray,
    val_frame: pd.DataFrame,
    candidate_cfg: dict[str, Any],
    random_seed: int,
) -> dict[str, Any]:
    _require_torch()
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    torch.set_num_threads(int(candidate_cfg.get("torch_num_threads", max(1, min(16, (torch.get_num_threads() or 1))))))

    rr_train_mask = np.isfinite(train_rr_targets) & train_validity_targets.astype(bool)
    rr_mean = float(np.mean(train_rr_targets[rr_train_mask])) if rr_train_mask.any() else 18.0
    rr_std = float(np.std(train_rr_targets[rr_train_mask])) if rr_train_mask.any() else 5.0
    if not np.isfinite(rr_std) or rr_std < 1e-6:
        rr_std = 1.0

    train_rr_normalized = np.where(rr_train_mask, (train_rr_targets - rr_mean) / rr_std, 0.0).astype(np.float32)
    val_rr_normalized = np.where(np.isfinite(val_rr_targets), (val_rr_targets - rr_mean) / rr_std, 0.0).astype(np.float32)

    scaler = _fit_scalar_standardizer(train_scalar)
    train_scalar_norm = scaler.transform(train_scalar).astype(np.float32)
    val_scalar_norm = scaler.transform(val_scalar).astype(np.float32)

    train_dataset = _Stage5Dataset(train_series, train_scalar_norm, train_rr_normalized, train_validity_targets)
    val_dataset = _Stage5Dataset(val_series, val_scalar_norm, val_rr_normalized, val_validity_targets)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(candidate_cfg["batch_size"]),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=int(candidate_cfg["batch_size"]), shuffle=False, num_workers=0)

    model = Stage5RespMultitaskCNN(
        num_channels=int(train_series.shape[1]),
        num_scalar_features=int(train_scalar_norm.shape[1]),
        base_width=int(candidate_cfg["base_width"]),
        dropout=float(candidate_cfg["dropout"]),
        kernel_sizes=_candidate_kernel_sizes(candidate_cfg),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(candidate_cfg["learning_rate"]),
        weight_decay=float(candidate_cfg["weight_decay"]),
    )
    rr_loss_fn = nn.SmoothL1Loss(beta=float(candidate_cfg.get("huber_delta", 1.0)))
    validity_loss_fn = nn.BCEWithLogitsLoss()
    max_epochs = int(candidate_cfg.get("max_epochs", 12))
    patience = int(candidate_cfg.get("patience", 4))
    rr_weight = float(candidate_cfg.get("rr_loss_weight", 1.0))
    validity_weight = float(candidate_cfg.get("validity_loss_weight", 0.25))

    best_state = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for time_batch, scalar_batch, rr_batch, validity_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            rr_pred, validity_logit = model(time_batch, scalar_batch)
            rr_mask = validity_batch > 0.5
            if bool(rr_mask.any()):
                rr_loss = rr_loss_fn(rr_pred[rr_mask], rr_batch[rr_mask])
            else:
                rr_loss = rr_pred.sum() * 0.0
            validity_loss = validity_loss_fn(validity_logit, validity_batch)
            total_loss = rr_weight * rr_loss + validity_weight * validity_loss
            total_loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_rr_predictions: list[np.ndarray] = []
            val_confidence: list[np.ndarray] = []
            for time_batch, scalar_batch, _, _ in val_loader:
                rr_pred, validity_logit = model(time_batch, scalar_batch)
                val_rr_predictions.append(rr_pred.cpu().numpy())
                val_confidence.append(torch.sigmoid(validity_logit).cpu().numpy())
        val_rr_array = np.concatenate(val_rr_predictions, axis=0) * rr_std + rr_mean if val_rr_predictions else np.array([], dtype=float)
        val_confidence_array = np.concatenate(val_confidence, axis=0) if val_confidence else np.array([], dtype=float)
        threshold = float(candidate_cfg.get("resp_validity_threshold", 0.50))
        metrics = _evaluate_predictions(
            val_frame,
            rr_predictions=np.asarray(val_rr_array, dtype=float),
            confidence=np.asarray(val_confidence_array, dtype=float),
            confidence_threshold=threshold,
        )
        score = metrics["high_quality_resp_mae_bpm"]
        if best_metrics is None or (_safe_float(score, default=math.inf) < _safe_float(best_metrics["high_quality_resp_mae_bpm"], default=math.inf)):
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = metrics
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        best_metrics = {
            "high_quality_resp_mae_bpm": math.nan,
            "high_quality_resp_rmse_bpm": math.nan,
            "high_quality_resp_pearson_r": math.nan,
            "predicted_valid_resp_mae_bpm": math.nan,
            "predicted_valid_resp_rmse_bpm": math.nan,
            "predicted_valid_resp_pearson_r": math.nan,
            "predicted_valid_coverage": math.nan,
        }

    return {
        "model_state_dict": best_state,
        "scalar_mean": scaler.mean,
        "scalar_std": scaler.std,
        "rr_mean": rr_mean,
        "rr_std": rr_std,
        "best_epoch": int(best_epoch),
        **best_metrics,
    }


def _predict_with_bundle(
    *,
    bundle: dict[str, Any],
    time_series: np.ndarray,
    scalar_features: np.ndarray,
    candidate_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    _require_torch()
    scalar_mean = np.asarray(bundle["scalar_mean"], dtype=np.float32)
    scalar_std = np.asarray(bundle["scalar_std"], dtype=np.float32)
    scalar_normalized = ((np.nan_to_num(scalar_features, nan=0.0).astype(np.float32) - scalar_mean) / scalar_std).astype(np.float32)
    model = Stage5RespMultitaskCNN(
        num_channels=int(time_series.shape[1]),
        num_scalar_features=int(scalar_normalized.shape[1]),
        base_width=int(candidate_cfg["base_width"]),
        dropout=float(candidate_cfg["dropout"]),
        kernel_sizes=_candidate_kernel_sizes(candidate_cfg),
    )
    state_dict = {
        key: torch.as_tensor(value) if not isinstance(value, torch.Tensor) else value
        for key, value in bundle["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    dataset = _Stage5Dataset(
        time_series,
        scalar_normalized,
        np.zeros(time_series.shape[0], dtype=np.float32),
        np.zeros(time_series.shape[0], dtype=bool),
    )
    loader = DataLoader(dataset, batch_size=int(candidate_cfg["batch_size"]), shuffle=False, num_workers=0)
    rr_outputs: list[np.ndarray] = []
    conf_outputs: list[np.ndarray] = []
    with torch.no_grad():
        for time_batch, scalar_batch, _, _ in loader:
            rr_pred, validity_logit = model(time_batch, scalar_batch)
            rr_outputs.append(rr_pred.cpu().numpy())
            conf_outputs.append(torch.sigmoid(validity_logit).cpu().numpy())
    rr_pred = np.concatenate(rr_outputs, axis=0) if rr_outputs else np.array([], dtype=float)
    confidence = np.concatenate(conf_outputs, axis=0) if conf_outputs else np.array([], dtype=float)
    rr_pred = rr_pred * float(bundle["rr_std"]) + float(bundle["rr_mean"])
    return np.asarray(rr_pred, dtype=float), np.asarray(confidence, dtype=float)


def _inner_train_val_split(
    train_subjects: list[str],
    eval_cfg: dict[str, Any],
) -> tuple[list[str], list[str]]:
    from heart_rate_cnn.split import train_test_subject_split

    if len(train_subjects) < 3:
        return list(train_subjects), list(train_subjects[-1:])
    inner_split = train_test_subject_split(
        list(train_subjects),
        test_size=float(eval_cfg.get("inner_val_test_size", 0.2)),
        random_seed=int(eval_cfg.get("inner_val_random_seed", eval_cfg.get("random_seed", 42) + 17)),
    )
    inner_train_subjects = inner_split.train_subjects
    inner_val_subjects = inner_split.test_subjects if inner_split.test_subjects else inner_split.train_subjects[-1:]
    return inner_train_subjects, inner_val_subjects


def _select_indices_for_subjects(frame: pd.DataFrame, subject_ids: list[str]) -> np.ndarray:
    allowed = set(str(subject_id) for subject_id in subject_ids)
    return np.flatnonzero(frame["subject_id"].astype(str).isin(allowed).to_numpy())


def run_stage5_tuning(
    package_cache_by_window: dict[float, dict[str, Any]],
    *,
    train_subjects: list[str],
    eval_cfg: dict[str, Any],
    stage5_cfg: dict[str, Any],
    output_dir: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    tuning_cfg = stage5_cfg.get("tuning", {})
    inner_train_subjects, inner_val_subjects = _inner_train_val_split(train_subjects, eval_cfg)
    random_seed = int(eval_cfg.get("random_seed", 42))

    structural_candidates = [
        {"window_seconds": float(window_seconds), "channel_set": str(channel_set)}
        for window_seconds, channel_set in product(
            tuning_cfg.get("window_seconds_candidates", [32.0, 48.0]),
            tuning_cfg.get("channel_sets", ["ppg_acc", "ppg_acc_ri"]),
        )
    ]
    base_candidate_defaults = {
        "base_width": int(tuning_cfg.get("default_base_width", 32)),
        "dropout": float(tuning_cfg.get("default_dropout", 0.10)),
        "learning_rate": float(tuning_cfg.get("default_learning_rate", 1e-3)),
        "batch_size": int(tuning_cfg.get("default_batch_size", 64)),
        "weight_decay": float(tuning_cfg.get("default_weight_decay", 0.0)),
        "kernel_sizes": [int(value) for value in tuning_cfg.get("kernel_sizes", [7, 5, 5])],
        "huber_delta": float(tuning_cfg.get("huber_delta", 1.0)),
        "rr_loss_weight": float(tuning_cfg.get("rr_loss_weight", 1.0)),
        "validity_loss_weight": float(tuning_cfg.get("validity_loss_weight", 0.25)),
        "max_epochs": int(tuning_cfg.get("tuning_max_epochs", 12)),
        "patience": int(tuning_cfg.get("tuning_patience", 4)),
        "resp_validity_threshold": float(tuning_cfg.get("default_resp_validity_threshold", 0.50)),
        "torch_num_threads": int(tuning_cfg.get("torch_num_threads", 8)),
    }

    results: list[dict[str, Any]] = []
    structure_rankings: list[dict[str, Any]] = []
    for structure in structural_candidates:
        package = package_cache_by_window[float(structure["window_seconds"])]
        train_frame = package["train_frame"]
        inner_train_indices = _select_indices_for_subjects(train_frame, inner_train_subjects)
        inner_val_indices = _select_indices_for_subjects(train_frame, inner_val_subjects)
        inner_train_indices = _subsample_per_subject(
            train_frame,
            inner_train_indices,
            max_per_subject=int(tuning_cfg.get("max_tuning_windows_per_subject_train", 256)),
            random_seed=random_seed,
        )
        inner_val_indices = _subsample_per_subject(
            train_frame,
            inner_val_indices,
            max_per_subject=int(tuning_cfg.get("max_tuning_windows_per_subject_val", 128)),
            random_seed=random_seed + 11,
        )
        candidate_cfg = {**base_candidate_defaults, **structure}
        selected_channel_names = _candidate_channel_names(candidate_cfg)
        train_series, train_scalar, train_rr, train_validity, _ = _prepare_split_arrays(
            package,
            train_frame,
            inner_train_indices,
            selected_channel_names=selected_channel_names,
            scalar_columns=list(package["scalar_feature_columns"]),
            split_name="train",
        )
        val_series, val_scalar, val_rr, val_validity, val_frame = _prepare_split_arrays(
            package,
            train_frame,
            inner_val_indices,
            selected_channel_names=selected_channel_names,
            scalar_columns=list(package["scalar_feature_columns"]),
            split_name="train",
        )
        bundle = _train_one_model(
            train_series=train_series,
            train_scalar=train_scalar,
            train_rr_targets=train_rr,
            train_validity_targets=train_validity,
            val_series=val_series,
            val_scalar=val_scalar,
            val_rr_targets=val_rr,
            val_validity_targets=val_validity,
            val_frame=val_frame,
            candidate_cfg=candidate_cfg,
            random_seed=random_seed,
        )
        row = {
            "phase": "A",
            "dataset": dataset_name,
            "window_seconds": structure["window_seconds"],
            "channel_set": structure["channel_set"],
            **base_candidate_defaults,
            **{key: bundle[key] for key in bundle if key not in {"model_state_dict", "scalar_mean", "scalar_std", "rr_mean", "rr_std"}},
        }
        results.append(row)
        structure_rankings.append(row)

    structure_rankings = sorted(
        structure_rankings,
        key=lambda row: (
            _safe_float(row["high_quality_resp_mae_bpm"], default=math.inf),
            _safe_float(row["high_quality_resp_rmse_bpm"], default=math.inf),
            -_safe_float(row["high_quality_resp_pearson_r"], default=-math.inf),
            float(row["window_seconds"]),
            0 if str(row["channel_set"]) == "ppg_acc" else 1,
        ),
    )
    top_structures = structure_rankings[: int(tuning_cfg.get("top_n_structures", 2))]

    hyperparameter_candidates = list(
        product(
            tuning_cfg.get("phase_b_base_widths", [32, 64]),
            tuning_cfg.get("phase_b_dropouts", [0.10, 0.20]),
            tuning_cfg.get("phase_b_learning_rates", [1e-3, 3e-4]),
            tuning_cfg.get("phase_b_batch_sizes", [64, 128]),
            tuning_cfg.get("phase_b_weight_decays", [0.0, 1e-4]),
        )
    )
    for structure in top_structures:
        package = package_cache_by_window[float(structure["window_seconds"])]
        train_frame = package["train_frame"]
        inner_train_indices = _select_indices_for_subjects(train_frame, inner_train_subjects)
        inner_val_indices = _select_indices_for_subjects(train_frame, inner_val_subjects)
        inner_train_indices = _subsample_per_subject(
            train_frame,
            inner_train_indices,
            max_per_subject=int(tuning_cfg.get("max_tuning_windows_per_subject_train", 256)),
            random_seed=random_seed,
        )
        inner_val_indices = _subsample_per_subject(
            train_frame,
            inner_val_indices,
            max_per_subject=int(tuning_cfg.get("max_tuning_windows_per_subject_val", 128)),
            random_seed=random_seed + 11,
        )

        for base_width, dropout, learning_rate, batch_size, weight_decay in hyperparameter_candidates:
            candidate_cfg = {
                **base_candidate_defaults,
                "window_seconds": float(structure["window_seconds"]),
                "channel_set": str(structure["channel_set"]),
                "base_width": int(base_width),
                "dropout": float(dropout),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "weight_decay": float(weight_decay),
            }
            selected_channel_names = _candidate_channel_names(candidate_cfg)
            train_series, train_scalar, train_rr, train_validity, _ = _prepare_split_arrays(
                package,
                train_frame,
                inner_train_indices,
                selected_channel_names=selected_channel_names,
                scalar_columns=list(package["scalar_feature_columns"]),
                split_name="train",
            )
            val_series, val_scalar, val_rr, val_validity, val_frame = _prepare_split_arrays(
                package,
                train_frame,
                inner_val_indices,
                selected_channel_names=selected_channel_names,
                scalar_columns=list(package["scalar_feature_columns"]),
                split_name="train",
            )
            bundle = _train_one_model(
                train_series=train_series,
                train_scalar=train_scalar,
                train_rr_targets=train_rr,
                train_validity_targets=train_validity,
                val_series=val_series,
                val_scalar=val_scalar,
                val_rr_targets=val_rr,
                val_validity_targets=val_validity,
                val_frame=val_frame,
                candidate_cfg=candidate_cfg,
                random_seed=random_seed,
            )
            results.append(
                {
                    "phase": "B",
                    "dataset": dataset_name,
                    **candidate_cfg,
                    **{key: bundle[key] for key in bundle if key not in {"model_state_dict", "scalar_mean", "scalar_std", "rr_mean", "rr_std"}},
                }
            )

    phase_b_rows = [row for row in results if row["phase"] == "B"]
    ranked_phase_b = sorted(
        phase_b_rows,
        key=lambda row: (
            _safe_float(row["high_quality_resp_mae_bpm"], default=math.inf),
            _safe_float(row["high_quality_resp_rmse_bpm"], default=math.inf),
            -_safe_float(row["high_quality_resp_pearson_r"], default=-math.inf),
            int(row["base_width"]),
            0 if str(row["channel_set"]) == "ppg_acc" else 1,
        ),
    )
    if not ranked_phase_b:
        raise RuntimeError("Stage 5 tuning produced no Phase B candidates.")
    top_phase_b = ranked_phase_b[0]

    threshold_rows: list[dict[str, Any]] = []
    best_threshold = float(top_phase_b["resp_validity_threshold"])
    best_threshold_score = math.inf
    for threshold in [float(value) for value in tuning_cfg.get("phase_c_resp_validity_thresholds", [0.4, 0.5, 0.6])]:
        threshold_row = dict(top_phase_b)
        threshold_row["phase"] = "C"
        threshold_row["resp_validity_threshold"] = float(threshold)
        package = package_cache_by_window[float(top_phase_b["window_seconds"])]
        train_frame = package["train_frame"]
        inner_train_indices = _select_indices_for_subjects(train_frame, inner_train_subjects)
        inner_val_indices = _select_indices_for_subjects(train_frame, inner_val_subjects)
        inner_train_indices = _subsample_per_subject(
            train_frame,
            inner_train_indices,
            max_per_subject=int(tuning_cfg.get("max_tuning_windows_per_subject_train", 256)),
            random_seed=random_seed,
        )
        inner_val_indices = _subsample_per_subject(
            train_frame,
            inner_val_indices,
            max_per_subject=int(tuning_cfg.get("max_tuning_windows_per_subject_val", 128)),
            random_seed=random_seed + 11,
        )
        selected_channel_names = _candidate_channel_names(threshold_row)
        train_series, train_scalar, train_rr, train_validity, _ = _prepare_split_arrays(
            package,
            train_frame,
            inner_train_indices,
            selected_channel_names=selected_channel_names,
            scalar_columns=list(package["scalar_feature_columns"]),
            split_name="train",
        )
        val_series, val_scalar, val_rr, val_validity, val_frame = _prepare_split_arrays(
            package,
            train_frame,
            inner_val_indices,
            selected_channel_names=selected_channel_names,
            scalar_columns=list(package["scalar_feature_columns"]),
            split_name="train",
        )
        bundle = _train_one_model(
            train_series=train_series,
            train_scalar=train_scalar,
            train_rr_targets=train_rr,
            train_validity_targets=train_validity,
            val_series=val_series,
            val_scalar=val_scalar,
            val_rr_targets=val_rr,
            val_validity_targets=val_validity,
            val_frame=val_frame,
            candidate_cfg=threshold_row,
            random_seed=random_seed,
        )
        threshold_row.update(
            {key: bundle[key] for key in bundle if key not in {"model_state_dict", "scalar_mean", "scalar_std", "rr_mean", "rr_std"}}
        )
        threshold_rows.append(threshold_row)
        score = _safe_float(threshold_row["predicted_valid_resp_mae_bpm"], default=math.inf)
        coverage = _safe_float(threshold_row["predicted_valid_coverage"], default=0.0)
        if coverage >= float(tuning_cfg.get("min_predicted_valid_coverage", 0.30)) and score < best_threshold_score:
            best_threshold_score = score
            best_threshold = float(threshold)

    results.extend(threshold_rows)
    tuning_results = pd.DataFrame(results)
    tuning_results.to_csv(output_dir / f"{dataset_name}_stage5_tuning_results.csv", index=False)

    best_candidate = dict(top_phase_b)
    best_candidate["resp_validity_threshold"] = best_threshold
    return tuning_results, best_candidate


def fit_stage5_resp_cnn(
    package: dict[str, Any],
    *,
    candidate_cfg: dict[str, Any],
    train_subjects: list[str],
    random_seed: int,
) -> dict[str, Any]:
    train_frame = package["train_frame"]
    train_indices = _select_indices_for_subjects(train_frame, train_subjects)
    selected_channel_names = _candidate_channel_names(candidate_cfg)
    train_series, train_scalar, train_rr, train_validity, train_frame_subset = _prepare_split_arrays(
        package,
        train_frame,
        train_indices,
        selected_channel_names=selected_channel_names,
        scalar_columns=list(package["scalar_feature_columns"]),
        split_name="train",
    )
    bundle = _train_one_model(
        train_series=train_series,
        train_scalar=train_scalar,
        train_rr_targets=train_rr,
        train_validity_targets=train_validity,
        val_series=train_series,
        val_scalar=train_scalar,
        val_rr_targets=train_rr,
        val_validity_targets=train_validity,
        val_frame=train_frame_subset,
        candidate_cfg={**candidate_cfg, "max_epochs": int(max(5, candidate_cfg.get("best_epoch", candidate_cfg.get("max_epochs", 20))))},
        random_seed=random_seed,
    )
    return {**bundle, "selected_channel_names": selected_channel_names}


def predict_stage5_respiration(
    package: dict[str, Any],
    *,
    model_bundle: dict[str, Any],
    candidate_cfg: dict[str, Any],
) -> pd.DataFrame:
    selected_channel_names = list(model_bundle["selected_channel_names"])
    scalar_columns = list(package["scalar_feature_columns"])
    predictions_frames: list[pd.DataFrame] = []
    for split_name, frame, series in (
        ("train", package["train_frame"], package["train_timeseries"]),
        ("eval", package["eval_frame"], package["eval_timeseries"]),
    ):
        indices = np.arange(frame.shape[0], dtype=int)
        _, scalar_values, _, _, frame_subset = _prepare_split_arrays(
            package,
            frame,
            indices,
            selected_channel_names=selected_channel_names,
            scalar_columns=scalar_columns,
            split_name=split_name,
        )
        channel_indices = _select_channel_indices(list(package["channel_names"]), selected_channel_names)
        rr_pred, confidence = _predict_with_bundle(
            bundle=model_bundle,
            time_series=np.asarray(series, dtype=np.float32)[:, channel_indices, :],
            scalar_features=scalar_values,
            candidate_cfg=candidate_cfg,
        )
        split_predictions = frame_subset.copy()
        split_predictions["resp_rate_pred_bpm"] = rr_pred
        split_predictions["resp_confidence"] = confidence
        split_predictions["resp_validity_flag"] = (
            split_predictions["resp_confidence"].to_numpy(dtype=float) >= float(candidate_cfg["resp_validity_threshold"])
        ) & split_predictions["validity_flag"].astype(bool).to_numpy()
        predictions_frames.append(split_predictions)
    return pd.concat(predictions_frames, ignore_index=True, sort=False)
