from __future__ import annotations

import math

import numpy as np


def compute_hr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.size == 0:
        return {
            "mae": math.nan,
            "rmse": math.nan,
            "mape": math.nan,
            "pearson_r": math.nan,
            "num_valid_windows": 0.0,
        }

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    safe_true = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs(error) / np.abs(safe_true)) * 100.0)

    if y_true.size < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        pearson_r = math.nan
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "pearson_r": pearson_r,
        "num_valid_windows": float(y_true.size),
    }
