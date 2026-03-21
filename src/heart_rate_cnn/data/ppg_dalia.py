from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from heart_rate_cnn.data.base import BaseLoader
from heart_rate_cnn.types import SubjectRecord


DEFAULT_PPG_FS = 64.0
DEFAULT_ACC_FS = 32.0
DEFAULT_ECG_FS = 700.0


class PPGDaliaLoader(BaseLoader):
    def list_subjects(self) -> list[str]:
        self.validate_root()
        return sorted(
            path.name
            for path in self.root_dir.iterdir()
            if path.is_dir() and path.name.startswith("S")
        )

    def load_subject(self, subject_id: str) -> SubjectRecord:
        subject_file = self._resolve_subject_file(subject_id)
        payload = self._load_pickle(subject_file)
        signal = self._require_mapping(payload, "signal")
        wrist = self._require_mapping(signal, "wrist")
        chest = self._require_mapping(signal, "chest")

        ppg = self._as_array(self._require_key(wrist, "BVP"), "wrist.BVP")
        ecg = self._as_array(self._require_key(chest, "ECG"), "chest.ECG")
        acc_value = wrist.get("ACC")
        acc = None if acc_value is None else self._as_array(acc_value, "wrist.ACC")

        return SubjectRecord(
            dataset="ppg_dalia",
            subject_id=subject_id,
            ppg=ppg,
            ppg_fs=float(payload.get("ppg_fs", DEFAULT_PPG_FS)),
            ecg=ecg,
            ecg_fs=float(payload.get("ecg_fs", DEFAULT_ECG_FS)),
            acc=acc,
            acc_fs=float(payload.get("acc_fs", DEFAULT_ACC_FS)) if acc is not None else None,
            metadata={
                "source_file": str(subject_file),
                "activity": payload.get("activity"),
            },
        )

    def _resolve_subject_file(self, subject_id: str) -> Path:
        subject_dir = self.root_dir / subject_id
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject directory not found: {subject_dir}")
        candidates = [
            subject_dir / f"{subject_id}.pkl",
            subject_dir / f"{subject_id}.pickle",
        ]
        candidates.extend(sorted(subject_dir.glob("*.pkl")))
        candidates.extend(sorted(subject_dir.glob("*.pickle")))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No pickle file found for subject {subject_id} in {subject_dir}")

    @staticmethod
    def _load_pickle(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict payload in {path}, got {type(payload)!r}")
        return payload

    @staticmethod
    def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
        value = payload.get(key)
        if not isinstance(value, dict):
            raise KeyError(f"Expected mapping for key '{key}'")
        return value

    @staticmethod
    def _require_key(payload: dict[str, Any], key: str) -> Any:
        if key not in payload:
            raise KeyError(f"Missing required key '{key}'")
        return payload[key]

    @staticmethod
    def _as_array(value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 2 and 1 in array.shape:
            array = array.reshape(-1)
        if array.ndim == 0:
            raise ValueError(f"{name} must be at least 1D.")
        return array
