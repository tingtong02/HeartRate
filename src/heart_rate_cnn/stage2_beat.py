from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from heart_rate_cnn.preprocess import (
    detect_ecg_peaks,
    preprocess_ppg_stage1,
    resample_signal,
    trim_record_to_common_duration,
)
from heart_rate_cnn.types import SubjectRecord


def _variant_mode(config: dict[str, Any] | None) -> str:
    cfg = config or {}
    return str(cfg.get("variant_mode", "enhanced"))


def preprocess_ppg_for_beats(
    samples: np.ndarray,
    fs: float,
    config: dict[str, Any] | None = None,
    *,
    mode: str = "coarse",
) -> np.ndarray:
    cfg = config or {}
    if mode == "refine":
        smooth_window_seconds = float(cfg.get("refine_smooth_window_seconds", cfg.get("smooth_window_seconds", 0.12)))
        extra_smoothing = bool(cfg.get("refine_extra_smoothing", False))
    else:
        smooth_window_seconds = float(cfg.get("smooth_window_seconds", 0.20))
        extra_smoothing = bool(cfg.get("extra_smoothing", True))
    return preprocess_ppg_stage1(
        samples,
        fs=fs,
        bandpass_low_hz=float(cfg.get("bandpass_low_hz", 0.6)),
        bandpass_high_hz=float(cfg.get("bandpass_high_hz", 3.5)),
        bandpass_order=int(cfg.get("bandpass_order", 3)),
        smooth_window_seconds=smooth_window_seconds,
        smooth_polyorder=int(cfg.get("smooth_polyorder", 2)),
        extra_smoothing=extra_smoothing,
    )


def refine_beats_in_window(
    ppg_window: np.ndarray,
    peak_indices: np.ndarray,
    fs: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = config or {}
    radius = max(int(round(float(cfg.get("refine_radius_seconds", 0.08)) * fs)), 1)
    signal = np.asarray(ppg_window, dtype=float).reshape(-1)
    if peak_indices.size == 0:
        return np.array([], dtype=int)

    refined: list[int] = []
    for peak_index in np.asarray(peak_indices, dtype=int):
        start = max(0, peak_index - radius)
        end = min(signal.size, peak_index + radius + 1)
        if end <= start:
            continue
        local_index = int(np.argmax(signal[start:end]))
        refined.append(start + local_index)
    if not refined:
        return np.array([], dtype=int)
    return np.array(sorted(set(refined)), dtype=int)


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    if np.allclose(array, array[0]):
        return np.ones_like(array, dtype=float)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if maximum - minimum <= 1e-8:
        return np.ones_like(array, dtype=float)
    return (array - minimum) / (maximum - minimum)


def _estimate_prominence_threshold(signal: np.ndarray, config: dict[str, Any]) -> float:
    mad = np.median(np.abs(signal - np.median(signal))) + 1e-8
    percentile_high = float(np.percentile(signal, float(config.get("prominence_percentile_high", 90.0))))
    percentile_low = float(np.percentile(signal, float(config.get("prominence_percentile_low", 60.0))))
    percentile_span = max(percentile_high - percentile_low, 0.0)
    return max(
        float(config.get("min_prominence", 0.05)),
        float(config.get("prominence_scale", 0.35)) * mad,
        float(config.get("prominence_percentile_scale", 0.25)) * percentile_span,
    )


def _peak_width_bounds(fs: float, config: dict[str, Any]) -> tuple[int, int | None]:
    min_width = max(int(round(float(config.get("min_width_seconds", 0.08)) * fs)), 1)
    max_width_seconds = float(config.get("max_width_seconds", 0.45))
    max_width = max(int(round(max_width_seconds * fs)), min_width + 1)
    if max_width <= min_width:
        return min_width, None
    return min_width, max_width


def _compute_peak_quality_scores(
    signal: np.ndarray,
    peak_indices: np.ndarray,
    fs: float,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = config or {}
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size == 0:
        return np.array([], dtype=float)

    support_radius = max(int(round(float(cfg.get("support_window_seconds", 0.10)) * fs)), 1)
    heights = signal[peaks]
    local_prominences: list[float] = []
    local_supports: list[float] = []
    for peak in peaks:
        left_start = max(0, peak - support_radius)
        right_end = min(signal.size, peak + support_radius + 1)
        left_segment = signal[left_start : peak + 1]
        right_segment = signal[peak:right_end]
        left_min = float(np.min(left_segment)) if left_segment.size else float(signal[peak])
        right_min = float(np.min(right_segment)) if right_segment.size else float(signal[peak])
        left_mean = float(np.mean(left_segment)) if left_segment.size else float(signal[peak])
        right_mean = float(np.mean(right_segment)) if right_segment.size else float(signal[peak])
        local_prominences.append(min(float(signal[peak]) - left_min, float(signal[peak]) - right_min))
        local_supports.append(min(float(signal[peak]) - left_mean, float(signal[peak]) - right_mean))

    height_score = _normalize_scores(heights)
    prominence_score = _normalize_scores(np.asarray(local_prominences, dtype=float))
    support_score = _normalize_scores(np.asarray(local_supports, dtype=float))
    return 0.45 * height_score + 0.35 * prominence_score + 0.20 * support_score


def _prune_close_beats(
    peak_indices: np.ndarray,
    quality_scores: np.ndarray,
    min_distance: int,
) -> np.ndarray:
    peaks = np.asarray(peak_indices, dtype=int)
    if peaks.size <= 1:
        return peaks

    scores = np.asarray(quality_scores, dtype=float)
    ranked = peaks[np.argsort(scores)[::-1]]
    kept: list[int] = []
    for peak in ranked:
        if all(abs(peak - existing) > min_distance for existing in kept):
            kept.append(int(peak))
    return np.array(sorted(kept), dtype=int)


def _detect_beats_baseline(ppg_window: np.ndarray, fs: float, config: dict[str, Any]) -> np.ndarray:
    processed = preprocess_ppg_for_beats(ppg_window, fs=fs, config=config, mode="coarse")
    if processed.size < 8:
        return np.array([], dtype=int)

    hr_max_bpm = float(config.get("hr_max_bpm", 180.0))
    min_distance = max(int(round(fs * 60.0 / hr_max_bpm)), 1)
    min_width = max(int(round(float(config.get("min_width_seconds", 0.08)) * fs)), 1)
    mad = np.median(np.abs(processed - np.median(processed))) + 1e-8
    prominence = max(float(config.get("prominence_scale", 0.35)) * mad, float(config.get("min_prominence", 0.05)))

    peaks, _ = scipy_signal.find_peaks(
        processed,
        distance=min_distance,
        prominence=prominence,
        width=min_width,
    )
    refined = refine_beats_in_window(processed, peaks, fs=fs, config=config)
    prune_distance = max(int(round(min_distance * float(config.get("refractory_scale", 1.5)))), min_distance)
    scores = _compute_peak_quality_scores(processed, refined, fs=fs, config=config)
    return _prune_close_beats(refined, scores, prune_distance)


def _filter_low_quality_sparse_beats(
    peak_indices: np.ndarray,
    quality_scores: np.ndarray,
    fs: float,
    config: dict[str, Any],
) -> np.ndarray:
    if not bool(config.get("drop_sparse_low_quality", False)):
        return np.asarray(peak_indices, dtype=int)

    peaks = np.asarray(peak_indices, dtype=int)
    scores = np.asarray(quality_scores, dtype=float)
    if peaks.size <= 2:
        return peaks

    hr_min_bpm = float(config.get("hr_min_bpm", 45.0))
    if hr_min_bpm <= 0:
        return peaks

    soft_max_gap = fs * 60.0 / hr_min_bpm
    isolation_scale = float(config.get("isolated_gap_scale", 1.15))
    low_quality_ratio = float(config.get("low_quality_ratio", 0.55))
    median_score = float(np.median(scores)) if scores.size else 0.0
    if median_score <= 1e-8:
        return peaks

    keep_mask = np.ones(peaks.size, dtype=bool)
    for index, peak in enumerate(peaks):
        left_gap = float("inf") if index == 0 else float(peak - peaks[index - 1])
        right_gap = float("inf") if index == peaks.size - 1 else float(peaks[index + 1] - peak)
        is_sparse = left_gap > soft_max_gap * isolation_scale and right_gap > soft_max_gap * isolation_scale
        is_low_quality = scores[index] < median_score * low_quality_ratio
        if is_sparse and is_low_quality:
            keep_mask[index] = False
    kept = peaks[keep_mask]
    return kept if kept.size >= 2 else peaks


def _detect_beats_enhanced(ppg_window: np.ndarray, fs: float, config: dict[str, Any]) -> np.ndarray:
    cfg = config or {}
    coarse_signal = preprocess_ppg_for_beats(ppg_window, fs=fs, config=cfg, mode="coarse")
    refine_signal = preprocess_ppg_for_beats(ppg_window, fs=fs, config=cfg, mode="refine")
    if coarse_signal.size < 8:
        return np.array([], dtype=int)

    hr_max_bpm = float(cfg.get("hr_max_bpm", 180.0))
    min_distance = max(int(round(fs * 60.0 / hr_max_bpm)), 1)
    width_bounds = _peak_width_bounds(fs, cfg)
    prominence = _estimate_prominence_threshold(coarse_signal, cfg)

    peaks, _ = scipy_signal.find_peaks(
        coarse_signal,
        distance=min_distance,
        prominence=prominence,
        width=width_bounds,
    )
    if peaks.size == 0:
        return np.array([], dtype=int)

    refined = refine_beats_in_window(refine_signal, peaks, fs=fs, config=cfg)
    quality_scores = _compute_peak_quality_scores(refine_signal, refined, fs=fs, config=cfg)
    prune_distance = max(int(round(min_distance * float(cfg.get("refractory_scale", 1.25)))), min_distance)
    pruned = _prune_close_beats(refined, quality_scores, prune_distance)
    if bool(cfg.get("use_baseline_recall_safeguard", True)):
        fallback = _detect_beats_baseline(ppg_window, fs=fs, config=cfg)
        if fallback.size > pruned.size:
            merged = np.array(sorted(set(pruned.tolist()) | set(fallback.tolist())), dtype=int)
            merged_scores = _compute_peak_quality_scores(refine_signal, merged, fs=fs, config=cfg)
            pruned = _prune_close_beats(merged, merged_scores, prune_distance)
    pruned_scores = _compute_peak_quality_scores(refine_signal, pruned, fs=fs, config=cfg)
    return _filter_low_quality_sparse_beats(pruned, pruned_scores, fs=fs, config=cfg)


def detect_beats_in_window(ppg_window: np.ndarray, fs: float, config: dict[str, Any] | None = None) -> np.ndarray:
    cfg = config or {}
    if _variant_mode(cfg) == "baseline":
        return _detect_beats_baseline(ppg_window, fs=fs, config=cfg)
    return _detect_beats_enhanced(ppg_window, fs=fs, config=cfg)


def extract_ibi_from_beats(beat_indices: np.ndarray, fs: float) -> np.ndarray:
    beats = np.asarray(beat_indices, dtype=float)
    if beats.size < 2:
        return np.array([], dtype=float)
    return np.diff(beats) / fs


def _empty_clean_ibi_result() -> dict[str, np.ndarray | bool | float]:
    return {
        "ibi_clean_s": np.array([], dtype=float),
        "ibi_mask": np.array([], dtype=bool),
        "ibi_is_valid": False,
        "num_ibi_removed": 0.0,
        "ibi_removed_ratio": math.nan,
    }


def _clean_ibi_series_baseline(ibi_s: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray | bool | float]:
    cfg = config or {}
    ibi = np.asarray(ibi_s, dtype=float)
    if ibi.size == 0:
        return _empty_clean_ibi_result()

    min_ibi_s = float(cfg.get("min_ibi_s", 0.33))
    max_ibi_s = float(cfg.get("max_ibi_s", 1.5))
    local_radius = int(cfg.get("local_median_radius", 2))
    deviation_ratio = float(cfg.get("max_deviation_ratio", 0.30))
    min_clean_ibi = int(cfg.get("min_clean_ibi", 3))

    initial_mask = np.isfinite(ibi) & (ibi > 0.0) & (ibi >= min_ibi_s) & (ibi <= max_ibi_s)
    clean_mask = initial_mask.copy()
    for index in range(ibi.size):
        if not initial_mask[index]:
            clean_mask[index] = False
            continue
        start = max(0, index - local_radius)
        end = min(ibi.size, index + local_radius + 1)
        neighborhood = ibi[start:end][initial_mask[start:end]]
        if neighborhood.size < 2:
            continue
        local_median = float(np.median(neighborhood))
        if local_median <= 1e-8:
            continue
        if abs(ibi[index] - local_median) / local_median > deviation_ratio:
            clean_mask[index] = False

    ibi_clean = ibi[clean_mask]
    return {
        "ibi_clean_s": ibi_clean,
        "ibi_mask": clean_mask,
        "ibi_is_valid": bool(ibi_clean.size >= min_clean_ibi),
        "num_ibi_removed": float(ibi.size - ibi_clean.size),
        "ibi_removed_ratio": float((ibi.size - ibi_clean.size) / ibi.size),
    }


def _clean_ibi_series_enhanced(ibi_s: np.ndarray, config: dict[str, Any]) -> dict[str, np.ndarray | bool | float]:
    cfg = config or {}
    ibi = np.asarray(ibi_s, dtype=float)
    if ibi.size == 0:
        return _empty_clean_ibi_result()

    min_ibi_s = float(cfg.get("min_ibi_s", 0.33))
    max_ibi_s = float(cfg.get("max_ibi_s", 1.5))
    local_radius = int(cfg.get("local_median_radius", 2))
    base_deviation_ratio = float(cfg.get("max_deviation_ratio", 0.30))
    jump_ratio = float(cfg.get("adjacent_jump_ratio", 0.22))
    jump_anchor_ratio = float(cfg.get("jump_anchor_ratio", 0.12))
    short_series_threshold = int(cfg.get("short_series_threshold", 5))
    min_clean_ibi = int(cfg.get("min_clean_ibi", 3))

    initial_mask = np.isfinite(ibi) & (ibi > 0.0) & (ibi >= min_ibi_s) & (ibi <= max_ibi_s)
    clean_mask = initial_mask.copy()
    valid_count = int(np.sum(initial_mask))
    effective_deviation_ratio = base_deviation_ratio * (1.35 if valid_count <= short_series_threshold else 1.0)

    for index in range(ibi.size):
        if not initial_mask[index]:
            clean_mask[index] = False
            continue

        start = max(0, index - local_radius)
        end = min(ibi.size, index + local_radius + 1)
        neighborhood_indices = np.arange(start, end)
        valid_neighbors = initial_mask[start:end]
        neighborhood = ibi[start:end][valid_neighbors]
        if neighborhood.size == 0:
            continue

        local_median = float(np.median(neighborhood))
        if local_median <= 1e-8:
            continue

        deviation = abs(ibi[index] - local_median) / local_median
        if deviation > effective_deviation_ratio:
            clean_mask[index] = False
            continue

        if neighborhood.size < 3:
            continue

        neighbor_values = neighborhood[~np.isclose(neighborhood, ibi[index], atol=1e-10)]
        if neighbor_values.size == 0:
            continue
        neighbor_center = float(np.median(neighbor_values))
        if neighbor_center <= 1e-8:
            continue

        jump_values: list[float] = []
        if index > 0 and initial_mask[index - 1]:
            jump_values.append(abs(ibi[index] - ibi[index - 1]) / neighbor_center)
        if index < ibi.size - 1 and initial_mask[index + 1]:
            jump_values.append(abs(ibi[index] - ibi[index + 1]) / neighbor_center)
        if not jump_values:
            continue

        if max(jump_values) > jump_ratio and deviation > jump_anchor_ratio:
            clean_mask[index] = False

    ibi_clean = ibi[clean_mask]
    return {
        "ibi_clean_s": ibi_clean,
        "ibi_mask": clean_mask,
        "ibi_is_valid": bool(ibi_clean.size >= min_clean_ibi),
        "num_ibi_removed": float(ibi.size - ibi_clean.size),
        "ibi_removed_ratio": float((ibi.size - ibi_clean.size) / ibi.size),
    }


def clean_ibi_series(ibi_s: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, np.ndarray | bool | float]:
    cfg = config or {}
    if _variant_mode(cfg) == "baseline":
        return _clean_ibi_series_baseline(ibi_s, cfg)
    return _clean_ibi_series_enhanced(ibi_s, cfg)


def compute_time_domain_prv_features(
    ibi_s_clean: np.ndarray,
    *,
    num_beats: int,
    num_ibi_raw: int,
    num_ibi_clean: int,
) -> dict[str, float]:
    ibi = np.asarray(ibi_s_clean, dtype=float)
    features: dict[str, float] = {
        "num_beats": float(num_beats),
        "num_ibi_raw": float(num_ibi_raw),
        "num_ibi_clean": float(num_ibi_clean),
        "mean_ibi_ms": math.nan,
        "median_ibi_ms": math.nan,
        "mean_hr_bpm_from_ibi": math.nan,
        "sdnn_ms": math.nan,
        "rmssd_ms": math.nan,
        "pnn50": math.nan,
        "ibi_cv": math.nan,
    }
    if ibi.size == 0:
        return features

    ibi_ms = ibi * 1000.0
    features["mean_ibi_ms"] = float(np.mean(ibi_ms))
    features["median_ibi_ms"] = float(np.median(ibi_ms))
    features["mean_hr_bpm_from_ibi"] = float(60.0 / np.mean(ibi))
    if ibi.size >= 2:
        features["sdnn_ms"] = float(np.std(ibi_ms, ddof=1))
        diff_ms = np.diff(ibi_ms)
        if diff_ms.size >= 1:
            features["rmssd_ms"] = float(np.sqrt(np.mean(diff_ms**2)))
            features["pnn50"] = float(np.mean(np.abs(diff_ms) > 50.0))
        mean_ibi = float(np.mean(ibi))
        if mean_ibi > 1e-8:
            features["ibi_cv"] = float(np.std(ibi, ddof=1) / mean_ibi)
    return features


def match_beats_by_tolerance(pred_beats_s: np.ndarray, ref_beats_s: np.ndarray, tolerance_seconds: float) -> list[tuple[int, int]]:
    pred = np.asarray(pred_beats_s, dtype=float)
    ref = np.asarray(ref_beats_s, dtype=float)
    matches: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < pred.size and j < ref.size:
        diff = pred[i] - ref[j]
        if abs(diff) <= tolerance_seconds:
            matches.append((i, j))
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
    return matches


def evaluate_beat_detection(
    pred_beats: np.ndarray,
    ref_beats: np.ndarray,
    pred_fs: float,
    ref_fs: float,
    tolerance_seconds: float = 0.15,
) -> dict[str, float | int | list[tuple[int, int]]]:
    pred_s = np.asarray(pred_beats, dtype=float) / pred_fs
    ref_s = np.asarray(ref_beats, dtype=float) / ref_fs
    matches = match_beats_by_tolerance(pred_s, ref_s, tolerance_seconds=tolerance_seconds)
    tp = len(matches)
    fp = int(pred_s.size - tp)
    fn = int(ref_s.size - tp)
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else math.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else math.nan
    if math.isnan(precision) or math.isnan(recall) or (precision + recall) == 0:
        f1 = math.nan
    else:
        f1 = float(2.0 * precision * recall / (precision + recall))
    beat_count_error = float(abs(pred_s.size - ref_s.size))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "beat_count_error": beat_count_error,
        "matches": matches,
    }


def extract_matched_ibi_pairs_ms(
    pred_beats: np.ndarray,
    ref_beats: np.ndarray,
    pred_fs: float,
    ref_fs: float,
    tolerance_seconds: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    pred_beats = np.asarray(pred_beats, dtype=float)
    ref_beats = np.asarray(ref_beats, dtype=float)
    matches = match_beats_by_tolerance(pred_beats / pred_fs, ref_beats / ref_fs, tolerance_seconds=tolerance_seconds)
    if len(matches) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    pred_pairs: list[float] = []
    ref_pairs: list[float] = []
    for (pred_i0, ref_i0), (pred_i1, ref_i1) in zip(matches[:-1], matches[1:]):
        if pred_i1 != pred_i0 + 1 or ref_i1 != ref_i0 + 1:
            continue
        pred_ibi_ms = float((pred_beats[pred_i1] - pred_beats[pred_i0]) / pred_fs * 1000.0)
        ref_ibi_ms = float((ref_beats[ref_i1] - ref_beats[ref_i0]) / ref_fs * 1000.0)
        pred_pairs.append(pred_ibi_ms)
        ref_pairs.append(ref_ibi_ms)
    return np.asarray(ref_pairs, dtype=float), np.asarray(pred_pairs, dtype=float)


def extract_matched_ibi_pairs_with_indices_ms(
    pred_beats: np.ndarray,
    ref_beats: np.ndarray,
    pred_fs: float,
    ref_fs: float,
    tolerance_seconds: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_beats = np.asarray(pred_beats, dtype=float)
    ref_beats = np.asarray(ref_beats, dtype=float)
    matches = match_beats_by_tolerance(pred_beats / pred_fs, ref_beats / ref_fs, tolerance_seconds=tolerance_seconds)
    if len(matches) < 2:
        empty = np.array([], dtype=float)
        empty_idx = np.array([], dtype=int)
        return empty, empty, empty_idx, empty_idx

    pred_pairs: list[float] = []
    ref_pairs: list[float] = []
    pred_indices: list[int] = []
    ref_indices: list[int] = []
    for (pred_i0, ref_i0), (pred_i1, ref_i1) in zip(matches[:-1], matches[1:]):
        if pred_i1 != pred_i0 + 1 or ref_i1 != ref_i0 + 1:
            continue
        pred_ibi_ms = float((pred_beats[pred_i1] - pred_beats[pred_i0]) / pred_fs * 1000.0)
        ref_ibi_ms = float((ref_beats[ref_i1] - ref_beats[ref_i0]) / ref_fs * 1000.0)
        pred_pairs.append(pred_ibi_ms)
        ref_pairs.append(ref_ibi_ms)
        pred_indices.append(int(pred_i0))
        ref_indices.append(int(ref_i0))
    return (
        np.asarray(ref_pairs, dtype=float),
        np.asarray(pred_pairs, dtype=float),
        np.asarray(ref_indices, dtype=int),
        np.asarray(pred_indices, dtype=int),
    )


def build_analysis_windows(
    record: SubjectRecord,
    target_ppg_fs: float,
    analysis_window_seconds: float,
    analysis_step_seconds: float,
) -> list[dict[str, Any]]:
    record = trim_record_to_common_duration(record)
    ppg_resampled = resample_signal(record.ppg, record.ppg_fs, target_ppg_fs)
    window_size = int(round(analysis_window_seconds * target_ppg_fs))
    step_size = int(round(analysis_step_seconds * target_ppg_fs))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Analysis window and step sizes must be positive.")

    windows: list[dict[str, Any]] = []
    for window_index, start_idx in enumerate(range(0, ppg_resampled.size - window_size + 1, step_size)):
        end_idx = start_idx + window_size
        start_time_s = start_idx / target_ppg_fs
        end_time_s = end_idx / target_ppg_fs
        ecg_start = int(round(start_time_s * record.ecg_fs))
        ecg_end = int(round(end_time_s * record.ecg_fs))
        windows.append(
            {
                "dataset": record.dataset,
                "subject_id": record.subject_id,
                "analysis_window_index": window_index,
                "start_time_s": start_time_s,
                "duration_s": analysis_window_seconds,
                "ppg_window": ppg_resampled[start_idx:end_idx],
                "ppg_fs": float(target_ppg_fs),
                "ecg_window": record.ecg[ecg_start:ecg_end],
                "ecg_fs": float(record.ecg_fs),
            }
        )
    return windows


def detect_reference_beats_in_window(ecg_window: np.ndarray, ecg_fs: float) -> np.ndarray:
    return detect_ecg_peaks(ecg_window, ecg_fs)
