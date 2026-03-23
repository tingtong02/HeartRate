from __future__ import annotations

import json

import pandas as pd
import pytest

from heart_rate_cnn.stage4_features import (
    prepare_quality_aware_source_package,
    prepare_stage4_feature_package,
    resolve_stage4_output_dir,
)


def _source_package_stub(*, train_subjects: list[str], eval_subjects: list[str], **_) -> dict:
    train_frame = pd.DataFrame(
        [
            {
                "split": "train",
                "dataset": "synthetic",
                "subject_id": train_subjects[0],
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 8.0,
                "value": 1.0,
            }
        ]
    )
    eval_frame = pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": eval_subjects[0],
                "window_index": 1,
                "start_time_s": 8.0,
                "duration_s": 8.0,
                "value": 2.0,
            }
        ]
    )
    return {
        "train_subjects": list(train_subjects),
        "eval_subjects": list(eval_subjects),
        "selected_threshold": 0.64,
        "train_source_frame": train_frame,
        "eval_source_frame": eval_frame,
        "train_window_identity_frame": train_frame.loc[:, ["split", "dataset", "subject_id", "window_index", "start_time_s", "duration_s"]].copy(),
        "eval_window_identity_frame": eval_frame.loc[:, ["split", "dataset", "subject_id", "window_index", "start_time_s", "duration_s"]].copy(),
    }


def _feature_package_stub(*, train_subjects: list[str], eval_subjects: list[str], stage4_shared_cfg: dict, source_package: dict, **_) -> dict:
    train_frame = pd.DataFrame(
        [
            {
                "split": "train",
                "dataset": "synthetic",
                "subject_id": train_subjects[0],
                "window_index": 0,
                "start_time_s": 0.0,
                "duration_s": 8.0,
                "selected_hr_source": stage4_shared_cfg["selected_hr_source"],
                "value": 3.0,
            }
        ]
    )
    eval_frame = pd.DataFrame(
        [
            {
                "split": "eval",
                "dataset": "synthetic",
                "subject_id": eval_subjects[0],
                "window_index": 1,
                "start_time_s": 8.0,
                "duration_s": 8.0,
                "selected_hr_source": stage4_shared_cfg["selected_hr_source"],
                "value": 4.0,
            }
        ]
    )
    return {
        "train_subjects": list(train_subjects),
        "eval_subjects": list(eval_subjects),
        "selected_hr_source": str(stage4_shared_cfg["selected_hr_source"]),
        "source_package_key": str(source_package["cache_key"]),
        "train_feature_frame": train_frame,
        "eval_feature_frame": eval_frame,
    }


def test_prepare_quality_aware_source_package_reuses_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = {"count": 0}

    def fake_builder(**kwargs):
        call_counter["count"] += 1
        return _source_package_stub(**kwargs)

    monkeypatch.setattr("heart_rate_cnn.stage4_features._build_quality_aware_source_package_from_scratch", fake_builder)

    cache_cfg = {"enabled": True, "cache_dir": str(tmp_path), "rebuild": False, "schema_version": "stage4_test"}
    package_1 = prepare_quality_aware_source_package(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"window_seconds": 8.0},
        eval_cfg={"random_seed": 42},
        stage1_cfg={"frequency": {"nfft_min": 256}},
        stage3_cfg={"ml": {"threshold_grid": [0.5]}},
        cache_cfg=cache_cfg,
    )
    package_2 = prepare_quality_aware_source_package(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"window_seconds": 8.0},
        eval_cfg={"random_seed": 42},
        stage1_cfg={"frequency": {"nfft_min": 256}},
        stage3_cfg={"ml": {"threshold_grid": [0.5]}},
        cache_cfg=cache_cfg,
    )

    assert call_counter["count"] == 1
    assert package_1["cache_status"] == "built"
    assert package_2["cache_status"] == "reused"
    pd.testing.assert_frame_equal(package_1["train_source_frame"], package_2["train_source_frame"])

    with open(package_1["manifest_path"], "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["train_subjects"] == ["S1"]
    assert manifest["eval_subjects"] == ["S2"]
    assert manifest["selected_threshold"] == 0.64
    assert manifest["schema_version"] == "stage4_test"


def test_prepare_quality_aware_source_package_invalidates_on_config_change(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = {"count": 0}

    def fake_builder(**kwargs):
        call_counter["count"] += 1
        return _source_package_stub(**kwargs)

    monkeypatch.setattr("heart_rate_cnn.stage4_features._build_quality_aware_source_package_from_scratch", fake_builder)

    cache_cfg = {"enabled": True, "cache_dir": str(tmp_path), "rebuild": False, "schema_version": "stage4_test"}
    package_1 = prepare_quality_aware_source_package(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"window_seconds": 8.0},
        eval_cfg={"random_seed": 42},
        stage1_cfg={"frequency": {"nfft_min": 256}},
        stage3_cfg={"ml": {"threshold_grid": [0.5]}},
        cache_cfg=cache_cfg,
    )
    package_2 = prepare_quality_aware_source_package(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"window_seconds": 10.0},
        eval_cfg={"random_seed": 42},
        stage1_cfg={"frequency": {"nfft_min": 256}},
        stage3_cfg={"ml": {"threshold_grid": [0.5]}},
        cache_cfg=cache_cfg,
    )

    assert call_counter["count"] == 2
    assert package_1["cache_key"] != package_2["cache_key"]


def test_prepare_stage4_feature_package_reuses_cache(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    call_counter = {"count": 0}

    def fake_builder(**kwargs):
        call_counter["count"] += 1
        return _feature_package_stub(**kwargs)

    monkeypatch.setattr("heart_rate_cnn.stage4_features._build_stage4_feature_package_from_scratch", fake_builder)

    source_package = _source_package_stub(train_subjects=["S1"], eval_subjects=["S2"])
    source_package["cache_key"] = "source_package_key"
    cache_cfg = {"enabled": True, "cache_dir": str(tmp_path), "rebuild": False, "schema_version": "stage4_test"}

    package_1 = prepare_stage4_feature_package(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"window_seconds": 8.0},
        stage3_cfg={"robust_hr_policy": {"fallback_variant_mode": "enhanced"}},
        stage4_shared_cfg={"selected_hr_source": "robust_stage3c2_policy"},
        source_package=source_package,
        cache_cfg=cache_cfg,
    )
    package_2 = prepare_stage4_feature_package(
        loader=object(),
        dataset_name="synthetic",
        root_dir="/tmp/synthetic",
        train_subjects=["S1"],
        eval_subjects=["S2"],
        preprocess_cfg={"window_seconds": 8.0},
        stage3_cfg={"robust_hr_policy": {"fallback_variant_mode": "enhanced"}},
        stage4_shared_cfg={"selected_hr_source": "robust_stage3c2_policy"},
        source_package=source_package,
        cache_cfg=cache_cfg,
    )

    assert call_counter["count"] == 1
    assert package_1["cache_status"] == "built"
    assert package_2["cache_status"] == "reused"
    pd.testing.assert_frame_equal(package_1["train_feature_frame"], package_2["train_feature_frame"])


def test_resolve_stage4_output_dir_supports_canonical_and_validation() -> None:
    assert resolve_stage4_output_dir({"output_dir": "outputs", "scope": "canonical"}).as_posix().endswith("outputs")
    validation_path = resolve_stage4_output_dir(
        {"output_dir": "outputs", "scope": "validation", "label": "bounded_medium6_seed42", "validation_subdir": "validation"}
    )
    assert validation_path.as_posix().endswith("outputs/validation/bounded_medium6_seed42")


def test_resolve_stage4_output_dir_requires_label_for_validation() -> None:
    with pytest.raises(ValueError, match="output.label"):
        resolve_stage4_output_dir({"output_dir": "outputs", "scope": "validation", "label": ""})
