from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from heart_rate_cnn.stage5_respiration import (
    STAGE5_SCALAR_COLUMNS,
    Stage5RespMultitaskCNN,
    _train_one_model,
    build_respiration_surrogate_features,
    collect_stage5_window_seconds,
    estimate_reference_rr_from_resp_window,
    prepare_stage5_window_package,
)
from heart_rate_cnn.types import SubjectRecord


def _stage5_cfg() -> dict:
    return {
        "window_seconds": 32.0,
        "step_seconds": 4.0,
        "input_fs": 16.0,
        "beat_target_ppg_fs": 64.0,
        "reference": {
            "target_resp_fs": 25.0,
            "bandpass_low_hz": 0.08,
            "bandpass_high_hz": 0.70,
            "bandpass_order": 2,
            "min_rr_bpm": 6.0,
            "max_rr_bpm": 42.0,
            "welch_seconds": 24.0,
            "peak_prominence_scale": 0.15,
            "min_breaths": 3,
            "max_rr_disagreement_bpm": 3.0,
            "min_peak_power_fraction": 0.10,
        },
        "baseline": {
            "bandpass_low_hz": 0.08,
            "bandpass_high_hz": 0.70,
            "bandpass_order": 2,
            "min_rr_bpm": 6.0,
            "max_rr_bpm": 42.0,
            "welch_seconds": 24.0,
            "min_quality_beats": 4,
            "min_beats": 4,
            "min_clean_ibi": 3,
        },
        "beat": {
            "bandpass_low_hz": 0.6,
            "bandpass_high_hz": 3.5,
            "bandpass_order": 3,
            "smooth_window_seconds": 0.20,
            "refine_smooth_window_seconds": 0.10,
            "smooth_polyorder": 2,
            "extra_smoothing": True,
            "refine_extra_smoothing": False,
            "hr_min_bpm": 45.0,
            "hr_max_bpm": 180.0,
            "min_prominence": 0.05,
            "prominence_scale": 0.35,
            "prominence_percentile_low": 60.0,
            "prominence_percentile_high": 90.0,
            "prominence_percentile_scale": 0.18,
            "min_width_seconds": 0.08,
            "max_width_seconds": 0.45,
            "refine_radius_seconds": 0.08,
            "support_window_seconds": 0.10,
            "refractory_scale": 1.15,
            "isolated_gap_scale": 1.15,
            "low_quality_ratio": 0.55,
            "drop_sparse_low_quality": False,
            "use_baseline_recall_safeguard": True,
        },
        "ibi": {
            "min_ibi_s": 0.33,
            "max_ibi_s": 1.5,
            "local_median_radius": 2,
            "max_deviation_ratio": 0.25,
            "adjacent_jump_ratio": 0.22,
            "jump_anchor_ratio": 0.12,
            "short_series_threshold": 5,
            "min_clean_ibi": 3,
        },
        "beat_quality": {
            "enabled": True,
            "good_score_threshold": 0.55,
            "plausibility_margin_s": 0.08,
            "jump_good_ratio": 0.08,
            "jump_bad_ratio": 0.25,
            "crowding_good_scale": 1.10,
            "missing_ibi_score": 0.50,
            "weights": {
                "base_peak_quality": 0.60,
                "ibi_plausibility": 0.20,
                "ibi_stability": 0.10,
                "crowding": 0.05,
                "clean_pair_bonus": 0.05,
            },
        },
        "tuning": {
            "window_seconds_candidates": [32.0, 48.0],
        },
    }


def _make_ppg(duration_s: float = 32.0, fs: float = 64.0) -> np.ndarray:
    t = np.arange(int(duration_s * fs), dtype=float) / fs
    heart_rate_hz = 1.2 + 0.05 * np.sin(2.0 * np.pi * 0.12 * t)
    phase = 2.0 * np.pi * np.cumsum(heart_rate_hz) / fs
    amplitude = 1.0 + 0.25 * np.sin(2.0 * np.pi * 0.25 * t)
    ppg = amplitude * np.maximum(np.sin(phase), -0.3) + 0.05 * np.sin(2.0 * np.pi * 2.4 * t)
    return ppg.astype(float)


def test_subject_record_respiration_fields_are_backward_compatible() -> None:
    record = SubjectRecord(
        dataset="synthetic",
        subject_id="S1",
        ppg=np.zeros(16, dtype=float),
        ppg_fs=64.0,
        ecg=np.zeros(32, dtype=float),
        ecg_fs=128.0,
    )
    assert record.resp is None
    assert record.resp_fs is None


def test_collect_stage5_window_seconds_combines_default_and_tuning_values() -> None:
    values = collect_stage5_window_seconds(_stage5_cfg())
    assert values == [32.0, 48.0]


def test_estimate_reference_rr_from_resp_window_returns_valid_rate_for_clean_signal() -> None:
    fs = 50.0
    t = np.arange(int(60.0 * fs), dtype=float) / fs
    resp = np.sin(2.0 * np.pi * 0.25 * t)
    result = estimate_reference_rr_from_resp_window(resp, fs, _stage5_cfg())
    assert result["resp_rate_ref_valid_flag"] is True
    assert abs(float(result["resp_rate_ref_bpm"]) - 15.0) < 1.0
    assert result["resp_reference_reason"] == "reference_valid"


def test_estimate_reference_rr_from_resp_window_rejects_uninformative_signal() -> None:
    resp = np.zeros(3000, dtype=float)
    result = estimate_reference_rr_from_resp_window(resp, 50.0, _stage5_cfg())
    assert result["resp_rate_ref_valid_flag"] is False
    assert result["resp_reference_reason"] != "reference_valid"
    assert math.isnan(float(result["resp_rate_ref_bpm"]))


def test_build_respiration_surrogate_features_returns_interpretable_outputs() -> None:
    features = build_respiration_surrogate_features(
        _make_ppg(),
        fs=64.0,
        input_fs=16.0,
        config=_stage5_cfg(),
    )
    assert set(["riav_waveform", "rifv_waveform", "ribv_waveform", "resp_rate_baseline_bpm"]).issubset(features)
    assert features["riav_waveform"].ndim == 1
    assert features["riav_waveform"].size == int(32.0 * 16.0)
    assert isinstance(features["beat_positions_s"], list)
    assert isinstance(features["ibi_series_ms"], list)
    assert "support_sufficient_flag" in features


def test_stage5_cnn_forward_pass_has_expected_shapes() -> None:
    model = Stage5RespMultitaskCNN(
        num_channels=5,
        num_scalar_features=len(STAGE5_SCALAR_COLUMNS),
        base_width=16,
        dropout=0.1,
        kernel_sizes=[7, 5, 5],
    )
    time_series = np.random.default_rng(42).normal(size=(4, 5, 512)).astype(np.float32)
    scalar_features = np.random.default_rng(43).normal(size=(4, len(STAGE5_SCALAR_COLUMNS))).astype(np.float32)
    import torch

    rr_pred, validity_logit = model(torch.as_tensor(time_series), torch.as_tensor(scalar_features))
    assert tuple(rr_pred.shape) == (4,)
    assert tuple(validity_logit.shape) == (4,)


def test_train_one_model_runs_and_returns_finite_metrics() -> None:
    rng = np.random.default_rng(123)
    train_series = rng.normal(size=(24, 5, 128)).astype(np.float32)
    val_series = rng.normal(size=(8, 5, 128)).astype(np.float32)
    train_scalar = rng.normal(size=(24, len(STAGE5_SCALAR_COLUMNS))).astype(np.float32)
    val_scalar = rng.normal(size=(8, len(STAGE5_SCALAR_COLUMNS))).astype(np.float32)
    train_rr = np.linspace(12.0, 24.0, 24).astype(np.float32)
    val_rr = np.linspace(13.0, 23.0, 8).astype(np.float32)
    train_scalar[:, 0] = train_rr
    val_scalar[:, 0] = val_rr
    train_validity = np.ones(24, dtype=bool)
    val_validity = np.ones(8, dtype=bool)
    val_frame = pd.DataFrame(
        {
            "resp_rate_ref_bpm": val_rr,
            "resp_rate_ref_valid_flag": True,
            "validity_flag": True,
        }
    )
    bundle = _train_one_model(
        train_series=train_series,
        train_scalar=train_scalar,
        train_rr_targets=train_rr,
        train_validity_targets=train_validity,
        val_series=val_series,
        val_scalar=val_scalar,
        val_rr_targets=val_rr,
        val_validity_targets=val_validity,
        val_frame=val_frame,
        candidate_cfg={
            "base_width": 16,
            "dropout": 0.1,
            "learning_rate": 1e-2,
            "weight_decay": 0.0,
            "batch_size": 8,
            "max_epochs": 6,
            "patience": 3,
            "kernel_sizes": [7, 5, 5],
            "resp_validity_threshold": 0.5,
            "torch_num_threads": 2,
        },
        random_seed=123,
    )
    assert np.isfinite(float(bundle["high_quality_resp_mae_bpm"]))
    assert int(bundle["best_epoch"]) >= 1


def test_prepare_stage5_window_package_reuses_cache_and_invalidates(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = {"count": 0}

    def fake_split_builder(*, split_name: str, subject_ids: list[str], stage5_cfg: dict, **_) -> tuple[pd.DataFrame, np.ndarray]:
        call_counter["count"] += 1
        frame = pd.DataFrame(
            [
                {
                    "split": split_name,
                    "dataset": "synthetic",
                    "subject_id": subject_ids[0],
                    "window_index": 0,
                    "start_time_s": 0.0,
                    "duration_s": float(stage5_cfg["window_seconds"]),
                    "resp_rate_ref_bpm": 15.0,
                    "resp_rate_ref_valid_flag": True,
                    "resp_reference_reason": "reference_valid",
                    "resp_rate_baseline_bpm": 15.5,
                    "resp_baseline_confidence": 0.7,
                    "num_beats": 6.0,
                    "num_ibi_clean": 5.0,
                    "mean_ibi_ms": 800.0,
                    "rmssd_ms": 20.0,
                    "sdnn_ms": 18.0,
                    "pnn50": 0.05,
                    "support_sufficient_flag": True,
                    "beat_positions_s": [0.5, 1.3],
                    "ibi_series_ms": [780.0, 820.0],
                    "selected_hr_bpm": 70.0,
                    "selected_hr_source": "robust_stage3c2_policy",
                    "selected_hr_is_valid": True,
                    "ml_signal_quality_score": 0.9,
                    "motion_flag": False,
                    "validity_flag": True,
                    "hr_event_flag": False,
                    "irregular_pulse_flag": False,
                    "anomaly_score": 0.1,
                    "stage4_suspicion_flag": False,
                    "stage4_suspicion_score": 0.1,
                    "stage4_suspicion_type_summary": "",
                }
            ]
        )
        return frame, np.ones((1, 5, 64), dtype=np.float32)

    monkeypatch.setattr("heart_rate_cnn.stage5_respiration._build_stage5_split_from_scratch", fake_split_builder)

    stage4_source_package = {"cache_key": "s4_source"}
    stage4_feature_package = {"cache_key": "s4_feature"}
    stage4_full_frame = pd.DataFrame()
    cache_cfg = {"enabled": True, "cache_dir": str(tmp_path), "rebuild": False, "schema_version": "stage5_test"}

    kwargs = dict(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"target_ppg_fs": 64},
        eval_cfg={"random_seed": 42},
        stage1_cfg={"frequency": {"nfft_min": 256}},
        stage3_cfg={},
        stage4_cfg={},
        stage4_shared_cfg={},
        stage4_irregular_cfg={},
        stage4_anomaly_cfg={},
        stage4_full_cfg={},
        stage4_cache_cfg={"enabled": False},
        stage5_cfg=_stage5_cfg(),
        stage5_cache_cfg=cache_cfg,
        stage4_full_frame=stage4_full_frame,
        stage4_source_package=stage4_source_package,
        stage4_feature_package=stage4_feature_package,
    )
    package_1 = prepare_stage5_window_package(**kwargs)
    package_2 = prepare_stage5_window_package(**kwargs)
    package_3 = prepare_stage5_window_package(
        **{**kwargs, "stage5_cfg": {**_stage5_cfg(), "window_seconds": 48.0}}
    )

    assert call_counter["count"] == 4
    assert package_1["cache_status"] == "built"
    assert package_2["cache_status"] == "reused"
    assert package_1["cache_key"] != package_3["cache_key"]

    with open(package_1["manifest_path"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["stage4_source_package_key"] == "s4_source"
    assert manifest["schema_version"] == "stage5_test"
