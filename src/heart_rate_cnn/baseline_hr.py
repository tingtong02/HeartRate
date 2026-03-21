from __future__ import annotations

import numpy as np
from scipy import signal

from heart_rate_cnn.types import WindowSample


def estimate_hr_frequency_domain(
    ppg_window: np.ndarray,
    fs: float,
    hr_band_bpm: tuple[float, float] = (40.0, 180.0),
) -> float:
    samples = np.asarray(ppg_window, dtype=float).reshape(-1)
    if samples.size < 4:
        return float("nan")

    samples = signal.detrend(samples - np.mean(samples))
    nperseg = samples.size
    nfft = max(2048, int(2 ** np.ceil(np.log2(samples.size))))
    frequencies, power = signal.welch(samples, fs=fs, nperseg=nperseg, nfft=nfft)
    low_hz, high_hz = hr_band_bpm[0] / 60.0, hr_band_bpm[1] / 60.0
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(mask):
        return float("nan")
    band_freqs = frequencies[mask]
    band_power = power[mask]
    dominant_idx = int(np.argmax(band_power))
    return float(band_freqs[dominant_idx] * 60.0)


def predict_windows(
    windows: list[WindowSample],
    hr_band_bpm: tuple[float, float] = (40.0, 180.0),
) -> list[dict[str, float | int | str | bool | None]]:
    predictions: list[dict[str, float | int | str | bool | None]] = []
    for window in windows:
        pred_hr = estimate_hr_frequency_domain(window.ppg, window.ppg_fs, hr_band_bpm)
        predictions.append(
            {
                "dataset": window.dataset,
                "subject_id": window.subject_id,
                "window_index": window.window_index,
                "start_time_s": window.start_time_s,
                "duration_s": window.duration_s,
                "ref_hr_bpm": window.ref_hr_bpm,
                "pred_hr_bpm": pred_hr,
                "is_valid": window.is_valid and np.isfinite(pred_hr),
            }
        )
    return predictions
