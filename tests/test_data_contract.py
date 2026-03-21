from __future__ import annotations

import pickle

import numpy as np

from heart_rate_cnn.data import PPGDaliaLoader, WESADLoader
from heart_rate_cnn.preprocess import build_window_samples


def _write_subject_pickle(root, subject_id: str, payload: dict) -> None:
    subject_dir = root / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    with (subject_dir / f"{subject_id}.pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def _make_payload() -> dict:
    seconds = 12
    ppg_fs = 64
    ecg_fs = 256
    acc_fs = 32

    time_ppg = np.arange(0.0, seconds, 1.0 / ppg_fs)
    time_ecg = np.arange(0.0, seconds, 1.0 / ecg_fs)
    time_acc = np.arange(0.0, seconds, 1.0 / acc_fs)

    heart_hz = 75.0 / 60.0
    ppg = np.sin(2 * np.pi * heart_hz * time_ppg)
    ecg = np.zeros_like(time_ecg)
    beat_times = np.arange(0.5, seconds, 1.0 / heart_hz)
    for beat_time in beat_times:
        center = int(round(beat_time * ecg_fs))
        if 1 <= center < len(ecg) - 1:
            ecg[center - 1 : center + 2] = [0.8, 1.2, 0.8]
    acc = np.stack(
        [
            np.sin(2 * np.pi * 0.3 * time_acc),
            np.cos(2 * np.pi * 0.3 * time_acc),
            np.zeros_like(time_acc),
        ],
        axis=1,
    )

    return {
        "ppg_fs": ppg_fs,
        "ecg_fs": ecg_fs,
        "acc_fs": acc_fs,
        "signal": {
            "wrist": {"BVP": ppg, "ACC": acc},
            "chest": {"ECG": ecg},
        }
    }


def test_ppg_dalia_loader_and_window_contract(tmp_path) -> None:
    _write_subject_pickle(tmp_path, "S1", _make_payload())
    loader = PPGDaliaLoader(tmp_path)

    subject_ids = loader.list_subjects()
    assert subject_ids == ["S1"]

    record = loader.load_subject("S1")
    assert record.dataset == "ppg_dalia"
    assert record.ppg_fs == 64.0
    assert record.ecg_fs == 256.0
    assert record.acc_fs == 32.0

    windows = build_window_samples(record, target_ppg_fs=64.0, window_seconds=8.0, step_seconds=2.0)
    assert windows
    first = windows[0]
    assert first.dataset == "ppg_dalia"
    assert first.subject_id == "S1"
    assert first.ppg.shape == (512,)
    assert first.acc is not None
    assert first.acc.shape == (512, 3)
    assert first.duration_s == 8.0


def test_wesad_loader_reads_official_like_pickle(tmp_path) -> None:
    payload = _make_payload()
    payload["label"] = np.zeros(10)
    _write_subject_pickle(tmp_path, "S2", payload)
    loader = WESADLoader(tmp_path)

    record = loader.load_subject("S2")
    assert record.dataset == "wesad"
    assert record.metadata["label"] is not None
    assert record.ppg.ndim == 1
    assert record.ecg.ndim == 1
