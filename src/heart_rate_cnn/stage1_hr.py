from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal

from heart_rate_cnn.preprocess import normalize_signal, preprocess_ppg_stage1, smooth_signal_savgol


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def estimate_hr_frequency_stage1(
    ppg_window: np.ndarray,
    fs: float,
    hr_band_bpm: tuple[float, float] = (40.0, 180.0),
    config: dict[str, Any] | None = None,
) -> dict[str, float | bool]:
    cfg = config or {}
    processed = preprocess_ppg_stage1(
        ppg_window,
        fs=fs,
        bandpass_low_hz=float(cfg.get("bandpass_low_hz", 0.6)),
        bandpass_high_hz=float(cfg.get("bandpass_high_hz", 3.5)),
        bandpass_order=int(cfg.get("bandpass_order", 3)),
        smooth_window_seconds=float(cfg.get("smooth_window_seconds", 0.20)),
        smooth_polyorder=int(cfg.get("smooth_polyorder", 2)),
        extra_smoothing=bool(cfg.get("extra_smoothing", False)),
    )

    if processed.size < 8:
        return {
            "freq_pred_hr_bpm": float("nan"),
            "freq_confidence": 0.0,
            "freq_peak_ratio": 0.0,
            "freq_is_valid": False,
        }

    nperseg = min(processed.size, int(round(float(cfg.get("welch_nperseg_seconds", 8.0)) * fs)))
    nperseg = max(nperseg, min(processed.size, 32))
    nfft = max(int(cfg.get("nfft_min", 2048)), int(2 ** np.ceil(np.log2(processed.size))))
    frequencies, power = signal.welch(processed, fs=fs, nperseg=nperseg, nfft=nfft)

    low_hz, high_hz = hr_band_bpm[0] / 60.0, hr_band_bpm[1] / 60.0
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        return {
            "freq_pred_hr_bpm": float("nan"),
            "freq_confidence": 0.0,
            "freq_peak_ratio": 0.0,
            "freq_is_valid": False,
        }

    band_freqs = frequencies[band_mask]
    band_power = power[band_mask]
    if band_power.size == 0 or np.allclose(band_power, 0.0):
        return {
            "freq_pred_hr_bpm": float("nan"),
            "freq_confidence": 0.0,
            "freq_peak_ratio": 0.0,
            "freq_is_valid": False,
        }

    peak_indices, _ = signal.find_peaks(band_power)
    if peak_indices.size == 0:
        peak_indices = np.array([int(np.argmax(band_power))], dtype=int)

    ranked = peak_indices[np.argsort(band_power[peak_indices])[::-1]]
    main_idx = int(ranked[0])
    second_power = float(band_power[ranked[1]]) if ranked.size > 1 else float(np.mean(band_power))
    main_power = float(band_power[main_idx])
    pred_hr_bpm = float(band_freqs[main_idx] * 60.0)

    band_power_sum = float(np.sum(band_power)) + 1e-8
    peak_ratio = main_power / max(second_power, 1e-8)
    peak_power_fraction = main_power / band_power_sum

    edge_margin_bpm = float(cfg.get("edge_margin_bpm", 5.0))
    distance_to_edge_bpm = min(pred_hr_bpm - hr_band_bpm[0], hr_band_bpm[1] - pred_hr_bpm)
    edge_score = _clip01(distance_to_edge_bpm / max(edge_margin_bpm, 1e-8))
    ratio_score = _clip01((peak_ratio - 1.0) / 4.0)
    power_score = _clip01((peak_power_fraction - 0.10) / 0.35)
    confidence = _clip01(0.5 * ratio_score + 0.35 * power_score + 0.15 * edge_score)

    min_peak_power_fraction = float(cfg.get("min_peak_power_fraction", 0.10))
    is_valid = bool(np.isfinite(pred_hr_bpm) and peak_power_fraction >= min_peak_power_fraction)
    return {
        "freq_pred_hr_bpm": pred_hr_bpm,
        "freq_confidence": confidence if is_valid else 0.0,
        "freq_peak_ratio": float(peak_ratio),
        "freq_is_valid": is_valid,
    }


def estimate_hr_time_stage1(
    ppg_window: np.ndarray,
    fs: float,
    hr_band_bpm: tuple[float, float] = (40.0, 180.0),
    config: dict[str, Any] | None = None,
) -> dict[str, float | int | bool]:
    cfg = config or {}
    processed = preprocess_ppg_stage1(
        ppg_window,
        fs=fs,
        bandpass_low_hz=float(cfg.get("bandpass_low_hz", 0.6)),
        bandpass_high_hz=float(cfg.get("bandpass_high_hz", 3.5)),
        bandpass_order=int(cfg.get("bandpass_order", 3)),
        smooth_window_seconds=float(cfg.get("smooth_window_seconds", 0.20)),
        smooth_polyorder=int(cfg.get("smooth_polyorder", 2)),
        extra_smoothing=bool(cfg.get("extra_smoothing", True)),
    )
    peak_signal = normalize_signal(np.abs(signal.hilbert(processed)))
    peak_signal = smooth_signal_savgol(
        peak_signal,
        fs=fs,
        window_seconds=float(cfg.get("envelope_smooth_window_seconds", 0.12)),
        polyorder=int(cfg.get("smooth_polyorder", 2)),
    )

    if peak_signal.size < 8:
        return {
            "time_pred_hr_bpm": float("nan"),
            "time_confidence": 0.0,
            "time_num_peaks": 0,
            "time_is_valid": False,
        }

    hr_min_bpm, hr_max_bpm = hr_band_bpm
    min_distance = max(int(round(fs * 60.0 / hr_max_bpm)), 1)
    min_width = max(int(round(float(cfg.get("min_width_seconds", 0.08)) * fs)), 1)
    mad = np.median(np.abs(peak_signal - np.median(peak_signal))) + 1e-8
    prominence = max(float(cfg.get("prominence_scale", 0.35)) * mad, float(cfg.get("min_prominence", 0.05)))
    peaks, properties = signal.find_peaks(
        peak_signal,
        distance=min_distance,
        prominence=prominence,
        width=min_width,
    )

    if peaks.size < int(cfg.get("min_peaks", 3)):
        return {
            "time_pred_hr_bpm": float("nan"),
            "time_confidence": 0.0,
            "time_num_peaks": int(peaks.size),
            "time_is_valid": False,
        }

    intervals_s = np.diff(peaks) / fs
    valid_intervals_s = intervals_s[(intervals_s > 0) & (60.0 / intervals_s >= hr_min_bpm) & (60.0 / intervals_s <= hr_max_bpm)]
    if valid_intervals_s.size < 2:
        return {
            "time_pred_hr_bpm": float("nan"),
            "time_confidence": 0.0,
            "time_num_peaks": int(peaks.size),
            "time_is_valid": False,
        }

    pred_hr_bpm = float(60.0 / np.median(valid_intervals_s))
    interval_cv = float(np.std(valid_intervals_s) / (np.mean(valid_intervals_s) + 1e-8))
    prominence_values = np.asarray(properties.get("prominences", np.array([], dtype=float)), dtype=float)
    prominence_median = float(np.median(prominence_values)) if prominence_values.size else 0.0

    regularity_score = _clip01(1.0 - interval_cv / 0.25)
    count_score = _clip01((peaks.size - 2) / 4.0)
    prominence_score = _clip01(prominence_median / 1.5)
    confidence = _clip01(0.4 * regularity_score + 0.35 * count_score + 0.25 * prominence_score)

    return {
        "time_pred_hr_bpm": pred_hr_bpm,
        "time_confidence": confidence,
        "time_num_peaks": int(peaks.size),
        "time_is_valid": bool(np.isfinite(pred_hr_bpm)),
    }


def fuse_hr_estimates(
    freq_result: dict[str, float | bool],
    time_result: dict[str, float | int | bool],
    agreement_threshold_bpm: float = 6.0,
    conflict_threshold_bpm: float = 12.0,
) -> dict[str, float | str | bool]:
    freq_valid = bool(freq_result.get("freq_is_valid", False)) and np.isfinite(float(freq_result.get("freq_pred_hr_bpm", math.nan)))
    time_valid = bool(time_result.get("time_is_valid", False)) and np.isfinite(float(time_result.get("time_pred_hr_bpm", math.nan)))

    if not freq_valid and not time_valid:
        return {
            "fusion_pred_hr_bpm": float("nan"),
            "fusion_confidence": 0.0,
            "fusion_source": "none",
            "fusion_is_valid": False,
        }

    if freq_valid and not time_valid:
        return {
            "fusion_pred_hr_bpm": float(freq_result["freq_pred_hr_bpm"]),
            "fusion_confidence": float(freq_result["freq_confidence"]),
            "fusion_source": "frequency",
            "fusion_is_valid": True,
        }

    if time_valid and not freq_valid:
        return {
            "fusion_pred_hr_bpm": float(time_result["time_pred_hr_bpm"]),
            "fusion_confidence": float(time_result["time_confidence"]),
            "fusion_source": "time",
            "fusion_is_valid": True,
        }

    freq_hr = float(freq_result["freq_pred_hr_bpm"])
    time_hr = float(time_result["time_pred_hr_bpm"])
    freq_conf = float(freq_result["freq_confidence"])
    time_conf = float(time_result["time_confidence"])
    diff_bpm = abs(freq_hr - time_hr)

    if diff_bpm <= agreement_threshold_bpm:
        total_conf = max(freq_conf + time_conf, 1e-8)
        blended_hr = (freq_hr * freq_conf + time_hr * time_conf) / total_conf
        blended_conf = _clip01(max(freq_conf, time_conf) + 0.15)
        return {
            "fusion_pred_hr_bpm": float(blended_hr),
            "fusion_confidence": blended_conf,
            "fusion_source": "blended",
            "fusion_is_valid": True,
        }

    if diff_bpm <= conflict_threshold_bpm:
        if freq_conf >= time_conf:
            return {
                "fusion_pred_hr_bpm": freq_hr,
                "fusion_confidence": freq_conf,
                "fusion_source": "frequency",
                "fusion_is_valid": True,
            }
        return {
            "fusion_pred_hr_bpm": time_hr,
            "fusion_confidence": time_conf,
            "fusion_source": "time",
            "fusion_is_valid": True,
        }

    time_num_peaks = int(time_result.get("time_num_peaks", 0))
    if time_conf >= freq_conf + 0.2 and time_num_peaks >= 4:
        return {
            "fusion_pred_hr_bpm": time_hr,
            "fusion_confidence": time_conf,
            "fusion_source": "time",
            "fusion_is_valid": True,
        }

    return {
        "fusion_pred_hr_bpm": freq_hr,
        "fusion_confidence": freq_conf,
        "fusion_source": "frequency",
        "fusion_is_valid": True,
    }
