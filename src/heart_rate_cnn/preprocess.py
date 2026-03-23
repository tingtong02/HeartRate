from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import signal

from heart_rate_cnn.types import SubjectRecord, WindowSample


def resample_signal(samples: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    if np.isclose(original_fs, target_fs):
        return np.asarray(samples, dtype=float).copy()
    array = np.asarray(samples, dtype=float)
    target_length = int(round(array.shape[0] * target_fs / original_fs))
    if target_length <= 1:
        raise ValueError("Resampling would produce fewer than 2 samples.")
    return signal.resample(array, target_length, axis=0)


def bandpass_filter_ppg(
    samples: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 3,
) -> np.ndarray:
    array = np.asarray(samples, dtype=float).reshape(-1)
    nyquist = 0.5 * fs
    if array.size < 8 or low_hz <= 0 or high_hz <= low_hz or high_hz >= nyquist:
        return array.copy()

    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    try:
        return signal.sosfiltfilt(sos, array)
    except ValueError:
        return signal.sosfilt(sos, array)


def normalize_signal(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=float).reshape(-1)
    centered = array - np.mean(array)
    scale = np.std(centered)
    if scale <= 1e-8:
        return centered
    return centered / scale


def smooth_signal_savgol(
    samples: np.ndarray,
    fs: float,
    window_seconds: float,
    polyorder: int = 2,
) -> np.ndarray:
    array = np.asarray(samples, dtype=float).reshape(-1)
    if array.size < 5:
        return array.copy()

    window_length = max(int(round(window_seconds * fs)), polyorder + 2)
    if window_length % 2 == 0:
        window_length += 1
    if window_length >= array.size:
        window_length = array.size - 1 if array.size % 2 == 0 else array.size
    if window_length <= polyorder or window_length < 3:
        return array.copy()
    return signal.savgol_filter(array, window_length=window_length, polyorder=polyorder)


def dwt_denoise_ppg(
    samples: np.ndarray,
    *,
    wavelet: str = "db4",
    max_level: int = 4,
    threshold_mode: str = "soft",
    threshold_scale: float = 1.0,
) -> np.ndarray:
    array = np.asarray(samples, dtype=float).reshape(-1)
    if array.size < 8:
        return array.copy()
    if max_level < 1:
        return array.copy()

    import pywt

    wavelet_obj = pywt.Wavelet(wavelet)
    usable_level = pywt.dwt_max_level(array.size, wavelet_obj.dec_len)
    level = min(int(max_level), int(usable_level))
    if level < 1:
        return array.copy()

    coeffs = pywt.wavedec(array, wavelet_obj, level=level, mode="symmetric")
    detail_coeffs = coeffs[1:]
    if not detail_coeffs:
        return array.copy()

    finest_detail = np.asarray(detail_coeffs[-1], dtype=float)
    sigma = float(np.median(np.abs(finest_detail)) / 0.6745) if finest_detail.size else 0.0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return array.copy()

    threshold = float(threshold_scale) * sigma * np.sqrt(2.0 * np.log(max(array.size, 2)))
    denoised_coeffs = [coeffs[0]]
    denoised_coeffs.extend(pywt.threshold(detail, value=threshold, mode=threshold_mode) for detail in detail_coeffs)
    reconstructed = pywt.waverec(denoised_coeffs, wavelet_obj, mode="symmetric")
    if reconstructed.size > array.size:
        reconstructed = reconstructed[: array.size]
    elif reconstructed.size < array.size:
        reconstructed = np.pad(reconstructed, (0, array.size - reconstructed.size), mode="edge")
    return np.asarray(reconstructed, dtype=float)


def preprocess_ppg_stage1(
    samples: np.ndarray,
    fs: float,
    *,
    bandpass_low_hz: float = 0.6,
    bandpass_high_hz: float = 3.5,
    bandpass_order: int = 3,
    smooth_window_seconds: float = 0.20,
    smooth_polyorder: int = 2,
    extra_smoothing: bool = False,
) -> np.ndarray:
    array = np.asarray(samples, dtype=float).reshape(-1)
    if array.size < 4:
        return array.copy()

    processed = signal.detrend(array - np.mean(array))
    processed = bandpass_filter_ppg(
        processed,
        fs=fs,
        low_hz=bandpass_low_hz,
        high_hz=bandpass_high_hz,
        order=bandpass_order,
    )
    processed = normalize_signal(processed)
    processed = smooth_signal_savgol(
        processed,
        fs=fs,
        window_seconds=smooth_window_seconds,
        polyorder=smooth_polyorder,
    )
    if extra_smoothing:
        processed = smooth_signal_savgol(
            processed,
            fs=fs,
            window_seconds=max(smooth_window_seconds * 1.5, smooth_window_seconds + 0.05),
            polyorder=smooth_polyorder,
        )
    return processed


def trim_record_to_common_duration(record: SubjectRecord) -> SubjectRecord:
    durations = [record.ppg.shape[0] / record.ppg_fs, record.ecg.shape[0] / record.ecg_fs]
    if record.resp is not None and record.resp_fs is not None:
        durations.append(record.resp.shape[0] / record.resp_fs)
    if record.acc is not None and record.acc_fs is not None:
        durations.append(record.acc.shape[0] / record.acc_fs)
    common_duration_s = min(durations)

    ppg_samples = int(np.floor(common_duration_s * record.ppg_fs))
    ecg_samples = int(np.floor(common_duration_s * record.ecg_fs))
    resp = None
    resp_fs = record.resp_fs
    if record.resp is not None and record.resp_fs is not None:
        resp_samples = int(np.floor(common_duration_s * record.resp_fs))
        resp = record.resp[:resp_samples]
    acc = None
    acc_fs = record.acc_fs
    if record.acc is not None and record.acc_fs is not None:
        acc_samples = int(np.floor(common_duration_s * record.acc_fs))
        acc = record.acc[:acc_samples]

    metadata = dict(record.metadata)
    metadata["common_duration_s"] = common_duration_s
    return SubjectRecord(
        dataset=record.dataset,
        subject_id=record.subject_id,
        ppg=record.ppg[:ppg_samples],
        ppg_fs=record.ppg_fs,
        ecg=record.ecg[:ecg_samples],
        ecg_fs=record.ecg_fs,
        resp=resp,
        resp_fs=resp_fs if resp is not None else None,
        acc=acc,
        acc_fs=acc_fs if acc is not None else None,
        metadata=metadata,
    )


def detect_ecg_peaks(ecg: np.ndarray, ecg_fs: float) -> np.ndarray:
    ecg = np.asarray(ecg, dtype=float).reshape(-1)
    if ecg.size < 3:
        return np.array([], dtype=int)

    normalized = ecg - np.median(ecg)
    scale = np.std(normalized)
    if scale > 0:
        normalized = normalized / scale

    min_distance = max(int(0.3 * ecg_fs), 1)
    prominence = max(np.std(normalized) * 0.5, 0.3)
    peaks, _ = signal.find_peaks(normalized, distance=min_distance, prominence=prominence)
    if peaks.size >= 2:
        return peaks

    fallback, _ = signal.find_peaks(np.abs(normalized), distance=min_distance, prominence=prominence)
    return fallback


def compute_window_reference_hr(
    peak_indices: np.ndarray,
    ecg_fs: float,
    start_time_s: float,
    end_time_s: float,
) -> tuple[float | None, bool]:
    peak_times_s = peak_indices / ecg_fs
    mask = (peak_times_s >= start_time_s) & (peak_times_s <= end_time_s)
    window_peaks = peak_times_s[mask]
    if window_peaks.size < 2:
        return None, False
    rr = np.diff(window_peaks)
    rr = rr[rr > 0]
    if rr.size == 0:
        return None, False
    ref_hr_bpm = 60.0 / float(np.mean(rr))
    return ref_hr_bpm, True


def build_window_samples(
    record: SubjectRecord,
    target_ppg_fs: float,
    window_seconds: float,
    step_seconds: float,
) -> list[WindowSample]:
    record = trim_record_to_common_duration(record)
    ppg_resampled = resample_signal(record.ppg, record.ppg_fs, target_ppg_fs)
    acc_resampled = None
    if record.acc is not None and record.acc_fs is not None:
        acc_resampled = resample_signal(record.acc, record.acc_fs, target_ppg_fs)

    peak_indices = detect_ecg_peaks(record.ecg, record.ecg_fs)
    window_size = int(round(window_seconds * target_ppg_fs))
    step_size = int(round(step_seconds * target_ppg_fs))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("Window and step sizes must be positive.")

    samples: list[WindowSample] = []
    total = ppg_resampled.shape[0]
    for window_index, start_idx in enumerate(range(0, total - window_size + 1, step_size)):
        end_idx = start_idx + window_size
        start_time_s = start_idx / target_ppg_fs
        end_time_s = end_idx / target_ppg_fs
        ref_hr_bpm, is_valid = compute_window_reference_hr(
            peak_indices=peak_indices,
            ecg_fs=record.ecg_fs,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        acc_window = None if acc_resampled is None else acc_resampled[start_idx:end_idx]
        samples.append(
            WindowSample(
                dataset=record.dataset,
                subject_id=record.subject_id,
                window_index=window_index,
                start_time_s=start_time_s,
                duration_s=window_seconds,
                ppg=ppg_resampled[start_idx:end_idx],
                ppg_fs=target_ppg_fs,
                acc=acc_window,
                ref_hr_bpm=ref_hr_bpm,
                is_valid=is_valid,
            )
        )
    return samples


def flatten_window_samples(subject_windows: Iterable[list[WindowSample]]) -> list[WindowSample]:
    flattened: list[WindowSample] = []
    for windows in subject_windows:
        flattened.extend(windows)
    return flattened
