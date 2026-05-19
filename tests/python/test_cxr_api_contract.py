from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import pytest


def test_cxr_module_imports_and_validates_arguments() -> None:
    cxr = _import_or_skip("medkit_rs.cxr")

    assert cxr.__all__ == ["Dataset", "DataLoader", "datasets"]
    assert hasattr(cxr, "Dataset")
    assert hasattr(cxr, "DataLoader")

    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        cxr.Dataset("unused-cache", batch_size=0)
    with pytest.raises(ValueError, match="read_mode must be"):
        cxr.Dataset("unused-cache", read_mode="resident")

    with pytest.raises(TypeError, match="expects a medkit_rs.cxr.Dataset"):
        cxr.DataLoader(object())

    dataset = cxr.Dataset("unused-cache")

    with pytest.raises(ValueError, match="must use num_workers=0"):
        cxr.DataLoader(dataset, num_workers=1)

    with pytest.raises(ValueError, match="drop_last=True"):
        cxr.DataLoader(dataset, drop_last=True)

    with pytest.raises(ValueError, match="persistent_workers"):
        cxr.DataLoader(dataset, persistent_workers=True)


def test_package_root_exports_wheel_ready_api_only() -> None:
    medkit = _import_or_skip("medkit_rs")
    dataset = _import_or_skip("medkit_rs.dataset")

    assert "MedkitFfiBatchIterableDataset" not in medkit.__all__
    assert not hasattr(medkit, "MedkitFfiBatchIterableDataset")
    assert hasattr(dataset, "MedkitFfiBatchIterableDataset")


@pytest.mark.cxr_fixture
def test_cxr_dataloader_contract_against_cli_cache_fixture() -> None:
    cxr = _import_or_skip("medkit_rs.cxr")
    torch = _import_or_skip("torch")
    _import_or_skip("medkit_rs._native")

    cache_dir, split, summary = _cli_cache_fixture()
    split_summary = summary["splits"][split]
    sample_count = int(split_summary["samples"])
    if sample_count <= 0:
        pytest.skip(f"CXR cache split {split!r} has no samples")

    targets = list(summary.get("targets", []))
    if not targets:
        pytest.skip("CXR cache fixture has no targets")

    length = min(sample_count, 2)
    batch_size = length
    dataset = cxr.Dataset(
        cache_dir,
        split=split,
        length=length,
        batch_size=1,
        shuffle=False,
        seed=0,
        pin_memory=False,
        prefetch=True,
        prefetch_depth=3,
        read_workers=1,
        read_mode="mmap",
        include_metadata=True,
    )
    loader = cxr.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=13,
        pin_memory=False,
        prefetch=False,
    )

    report = loader.report_metadata()
    assert report["dataset"] == "medkit_rs.cxr.Dataset"
    assert report["cache_dir"] == str(cache_dir)
    assert report["split"] == split
    assert report["batch_size"] == batch_size
    assert report["shuffle"] is True
    assert report["pin_memory"] is False
    assert report["prefetch"] is False
    assert report["prefetch_depth"] == 0
    assert report["read_workers"] == 0
    assert report["read_mode"] == "mmap"
    assert report["include_metadata"] is True
    assert report["worker_mode"] == "single_process"
    assert report["num_workers"] == 0
    assert report["num_samples"] == length
    assert report["num_batches"] == 1
    assert report["targets"] == targets
    assert report["label_policy"]["uncertain"] == "ignore"
    assert report["label_policy"]["missing"] == "ignore"
    assert tuple(report["image_shape"]) == tuple(split_summary["shape"])
    assert report["cache_schema_version"] == summary.get("cache_schema_version")
    assert report["report_schema_version"] == summary.get("report_schema_version")
    assert report["transform_fingerprint"] == summary.get("transform_fingerprint")
    assert report["source_manifest_checksum"] == summary.get("source_manifest_checksum")

    assert loader.dataset.targets == targets
    assert tuple(loader.dataset.image_shape) == tuple(split_summary["shape"])

    batch = next(iter(loader))
    assert {
        "image",
        "labels",
        "mask",
        "metadata",
        "sample_id",
        "patient_id",
        "study_id",
        "image_id",
        "image_path",
    }.issubset(batch)

    _, channels, height, width = split_summary["shape"]
    expected_image_shape = (length, channels, height, width)
    expected_label_shape = (length, len(targets))
    assert tuple(batch["image"].shape) == expected_image_shape
    assert tuple(batch["labels"].shape) == expected_label_shape
    assert tuple(batch["mask"].shape) == expected_label_shape
    assert batch["image"].dtype == torch.float32
    assert batch["labels"].dtype == torch.float32
    assert batch["mask"].dtype == torch.float32
    assert batch["image"].is_contiguous()
    assert batch["labels"].is_contiguous()
    assert batch["mask"].is_contiguous()
    assert len(batch["metadata"]["sample_id"]) == length
    assert len(batch["sample_id"]) == length
    assert batch["metadata"]["sample_id"] == batch["sample_id"]
    assert batch["metadata"]["patient_id"] == batch["patient_id"]
    assert batch["metadata"]["study_id"] == batch["study_id"]
    assert batch["metadata"]["image_id"] == batch["image_id"]
    assert batch["metadata"]["image_path"] == batch["image_path"]


def _import_or_skip(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as error:
        pytest.skip(f"could not import {module!r}: {error}")


def _cli_cache_fixture() -> tuple[Path, str, dict[str, Any]]:
    raw_cache_dir = os.environ.get("MEDKIT_CXR_CACHE_DIR")
    if not raw_cache_dir:
        pytest.skip(
            "set MEDKIT_CXR_CACHE_DIR to a `medkit cxr cache` fixture directory"
        )

    cache_dir = Path(raw_cache_dir).expanduser()
    if not cache_dir.exists():
        pytest.skip(f"MEDKIT_CXR_CACHE_DIR does not exist: {cache_dir}")

    summary_path = cache_dir / "cache-metadata.json"
    if not summary_path.exists():
        pytest.skip(f"CXR CLI cache metadata not found: {summary_path}")

    summary = json.loads(summary_path.read_text())
    splits = summary.get("splits", {})
    split = os.environ.get("MEDKIT_CXR_SPLIT", "train")
    if split not in splits:
        pytest.skip(f"CXR cache split {split!r} not found in {summary_path}")

    return cache_dir, split, summary
