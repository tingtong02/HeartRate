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


def trim_record_to_common_duration(record: SubjectRecord) -> SubjectRecord:
    durations = [record.ppg.shape[0] / record.ppg_fs, record.ecg.shape[0] / record.ecg_fs]
    if record.acc is not None and record.acc_fs is not None:
        durations.append(record.acc.shape[0] / record.acc_fs)
    common_duration_s = min(durations)

    ppg_samples = int(np.floor(common_duration_s * record.ppg_fs))
    ecg_samples = int(np.floor(common_duration_s * record.ecg_fs))
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
