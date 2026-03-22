from __future__ import annotations

import numpy as np

from heart_rate_cnn.preprocess import build_window_samples, dwt_denoise_ppg, resample_signal
from heart_rate_cnn.split import train_test_subject_split
from heart_rate_cnn.types import SubjectRecord


def _make_record(subject_id: str, bpm: float = 72.0) -> SubjectRecord:
    ppg_fs = 64.0
    ecg_fs = 256.0
    duration_s = 20.0

    time_ppg = np.arange(0.0, duration_s, 1.0 / ppg_fs)
    time_ecg = np.arange(0.0, duration_s, 1.0 / ecg_fs)
    ppg = np.sin(2 * np.pi * (bpm / 60.0) * time_ppg)
    ecg = np.zeros_like(time_ecg)
    for beat_time in np.arange(0.5, duration_s, 60.0 / bpm):
        center = int(round(beat_time * ecg_fs))
        if 1 <= center < ecg.size - 1:
            ecg[center - 1 : center + 2] = [0.8, 1.2, 0.8]

    return SubjectRecord(
        dataset="synthetic",
        subject_id=subject_id,
        ppg=ppg,
        ppg_fs=ppg_fs,
        ecg=ecg,
        ecg_fs=ecg_fs,
        metadata={},
    )


def test_resample_signal_changes_length() -> None:
    signal = np.arange(64, dtype=float)
    resampled = resample_signal(signal, original_fs=64.0, target_fs=32.0)
    assert resampled.shape == (32,)


def test_build_window_samples_marks_valid_windows() -> None:
    record = _make_record("SYN001")
    windows = build_window_samples(record, target_ppg_fs=64.0, window_seconds=8.0, step_seconds=2.0)
    assert len(windows) == 7
    assert any(window.is_valid for window in windows)
    assert all(window.ppg.shape == (512,) for window in windows)


def test_subject_split_has_no_overlap() -> None:
    split = train_test_subject_split(["S1", "S2", "S3", "S4"], test_size=0.5, random_seed=7)
    assert set(split.train_subjects).isdisjoint(split.test_subjects)
    assert set(split.train_subjects + split.test_subjects) == {"S1", "S2", "S3", "S4"}


def test_dwt_denoise_ppg_keeps_length_and_finite_values() -> None:
    time = np.arange(0.0, 8.0, 1.0 / 64.0)
    signal = np.sin(2 * np.pi * 1.2 * time) + 0.15 * np.random.default_rng(7).normal(size=time.shape[0])
    denoised = dwt_denoise_ppg(signal, wavelet="db4", max_level=4, threshold_mode="soft", threshold_scale=1.0)
    assert denoised.shape == signal.shape
    assert np.all(np.isfinite(denoised))


def test_dwt_denoise_ppg_reduces_high_frequency_energy() -> None:
    time = np.arange(0.0, 8.0, 1.0 / 64.0)
    clean = np.sin(2 * np.pi * 1.2 * time)
    noisy = clean + 0.2 * np.sin(2 * np.pi * 12.0 * time)
    denoised = dwt_denoise_ppg(noisy, wavelet="db4", max_level=4, threshold_mode="soft", threshold_scale=1.0)
    noisy_diff_std = float(np.std(np.diff(noisy)))
    denoised_diff_std = float(np.std(np.diff(denoised)))
    assert denoised_diff_std < noisy_diff_std
