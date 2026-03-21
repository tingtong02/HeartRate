from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


ArrayLike = np.ndarray


@dataclass(slots=True)
class SubjectRecord:
    dataset: str
    subject_id: str
    ppg: ArrayLike
    ppg_fs: float
    ecg: ArrayLike
    ecg_fs: float
    acc: ArrayLike | None = None
    acc_fs: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WindowSample:
    dataset: str
    subject_id: str
    window_index: int
    start_time_s: float
    duration_s: float
    ppg: ArrayLike
    ppg_fs: float
    acc: ArrayLike | None
    ref_hr_bpm: float | None
    is_valid: bool
