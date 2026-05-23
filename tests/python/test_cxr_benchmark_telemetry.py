from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    REPO_ROOT
    / "crates"
    / "medkit-benchmarks"
    / "scripts"
    / "cxr_classification_benchmark.py"
)
MATRIX_PATH = (
    REPO_ROOT
    / "crates"
    / "medkit-benchmarks"
    / "scripts"
    / "modal_cxr_parallel_matrix.py"
)
MODAL_CLASSIFICATION_PATH = (
    REPO_ROOT
    / "crates"
    / "medkit-benchmarks"
    / "scripts"
    / "modal_cxr_classification.py"
)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("cxr_classification_benchmark", BENCHMARK_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_matrix_module():
    spec = importlib.util.spec_from_file_location("modal_cxr_parallel_matrix", MATRIX_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smaps_full_parser_detects_headers_with_device_colons_and_cache_buckets():
    benchmark = load_benchmark_module()
    raw = """
7f0000000000-7f0000100000 r--s 00000000 00:42 11 /cache/cxr/cache-224/train-images.float32.dat
Size:               1024 kB
Rss:                 512 kB
Pss:                 256 kB
Private_Clean:        64 kB
Private_Dirty:        16 kB
Locked:                0 kB
7f0000100000-7f0000200000 r--s 00000000 00:42 12 /cache/cxr/cache-224/train-labels.float32.dat
Size:               1024 kB
Rss:                 256 kB
Pss:                 128 kB
Private_Clean:         8 kB
Private_Dirty:         4 kB
7f0000200000-7f0000300000 r--s 00000000 00:42 14 /cache/cxr/cache with spaces/train-metadata.jsonl (deleted)
Size:               1024 kB
Rss:                 128 kB
Pss:                  64 kB
Private_Clean:         4 kB
Private_Dirty:         2 kB
7f0000200000-7f0000300000 rw-p 00000000 00:00 0 [heap]
Size:               1024 kB
Rss:                 128 kB
Pss:                  64 kB
Private_Clean:         0 kB
Private_Dirty:        64 kB
7f0000300000-7f0000400000 r-xp 00000000 08:01 13 /usr/lib/libc.so
Size:               1024 kB
Rss:                 128 kB
Pss:                  32 kB
Private_Clean:         0 kB
Private_Dirty:         0 kB
""".strip()
    categories = {
        benchmark.normalize_smaps_pathname(
            "/cache/cxr/cache-224/train-images.float32.dat"
        ): "cache_images",
        benchmark.normalize_smaps_pathname(
            "/cache/cxr/cache-224/train-labels.float32.dat"
        ): "cache_labels",
        benchmark.normalize_smaps_pathname(
            "/cache/cxr/cache with spaces/train-metadata.jsonl"
        ): "metadata",
    }

    report = benchmark.parse_smaps_full_memory(raw, categories)

    assert report["smaps_pss_cache_images_mb"] == 0.25
    assert report["smaps_pss_cache_labels_mb"] == 0.125
    assert report["smaps_pss_metadata_mb"] == 0.0625
    assert report["smaps_pss_other_file_mb"] == 0.03125
    assert report["smaps_pss_heap_mb"] == 0.0625
    assert report["smaps_pss_file_mb"] == 0.46875
    assert report["smaps_pss_anon_mb"] == 0.0625
    assert report["smaps_uss_mb"] == (64 + 16 + 8 + 4 + 4 + 2 + 0 + 64) / 1024.0


def test_smaps_cache_path_categories_reads_cache_metadata(tmp_path):
    benchmark = load_benchmark_module()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cache-metadata.json").write_text(
        """
{
  "splits": {
    "train": {
      "images_path": "train-images.float32.dat",
      "labels_path": "train-labels.float32.dat",
      "masks_path": "train-masks.float32.dat",
      "metadata_path": "train-metadata.jsonl"
    }
  }
}
""".strip()
        + "\n"
    )

    categories = benchmark.smaps_cache_path_categories({"cache_dir": str(cache_dir)})

    assert categories[benchmark.normalize_smaps_pathname(str(cache_dir / "train-images.float32.dat"))] == "cache_images"
    assert categories[benchmark.normalize_smaps_pathname(str(cache_dir / "train-labels.float32.dat"))] == "cache_labels"
    assert categories[benchmark.normalize_smaps_pathname(str(cache_dir / "train-masks.float32.dat"))] == "cache_masks"
    assert categories[benchmark.normalize_smaps_pathname(str(cache_dir / "train-metadata.jsonl"))] == "metadata"


def test_profile_summary_reconciles_samples_and_step_time():
    benchmark = load_benchmark_module()
    records = [
        {
            "samples": 32,
            "data_wait_ms": 1.0,
            "h2d_ms": 0.5,
            "batch_prepare_ms": 0.6,
            "batch_prepare_wall_ms": 0.7,
            "zero_grad_wall_ms": 0.1,
            "forward_ms": 2.0,
            "backward_ms": 3.0,
            "optimizer_ms": 0.75,
            "prefetch_maintenance_wall_ms": 0.2,
            "total_step_ms": 10.0,
            "accounted_step_ms": 6.65,
            "residual_step_ms": 3.35,
            "residual_step_ms_signed": 3.35,
            "residual_step_percent": 33.5,
        },
        {
            "samples": 32,
            "data_wait_ms": 2.0,
            "h2d_ms": 0.75,
            "batch_prepare_ms": 0.9,
            "batch_prepare_wall_ms": 1.0,
            "zero_grad_wall_ms": 0.2,
            "forward_ms": 3.0,
            "backward_ms": 4.0,
            "optimizer_ms": 1.0,
            "prefetch_maintenance_wall_ms": 0.3,
            "total_step_ms": 15.0,
            "accounted_step_ms": 9.4,
            "residual_step_ms": 5.6,
            "residual_step_ms_signed": 5.6,
            "residual_step_percent": 5.6 * 100.0 / 15.0,
        },
    ]

    summary = benchmark.summarize_profile_records(records)

    assert summary["profiled_batches"] == 2
    assert summary["profiled_samples"] == 64
    assert summary["profile_data_wait_total_ms"] == 3.0
    assert summary["profile_total_step_ms"] == 25.0
    assert summary["profile_train_samples_per_s"] == 2560.0
    assert summary["profile_end_to_end_ms"] == 28.0
    assert summary["profile_end_to_end_samples_per_s"] == 1000.0 * 64 / 28.0
    assert summary["profile_data_wait_ms_mean"] == 1.5
    assert summary["profile_batch_prepare_ms_mean"] == 0.75
    assert summary["profile_batch_prepare_wall_ms_p50"] == 0.7
    assert summary["profile_prefetch_maintenance_wall_ms_mean"] == 0.25
    assert summary["profile_residual_step_ms_p95"] == 5.6
    assert summary["profile_phase_budget_ms_per_batch"]["batch_prepare_ms"] == 0.75
    assert summary["profile_phase_budget_ms_per_batch"]["residual_step_ms_signed"] == 4.475
    assert summary["profile_step_accounted_percent"] == (6.65 + 9.4) * 100.0 / 25.0
    assert summary["profile_residual_step_signed_percent"] == (3.35 + 5.6) * 100.0 / 25.0
    assert summary["profile_step_reconciled_percent"] == 100.0


def test_training_ground_truth_report_compares_raw_and_medkit():
    benchmark = load_benchmark_module()
    reports = {
        "loader": {
            "pytorch_raw": {
                "status": "ok",
                "samples_per_second": 1000.0,
                "memory": _memory_report(),
            },
            "medkit_native_prefetch_pinned": {
                "status": "ok",
                "samples_per_second": 2000.0,
                "memory": {**_memory_report(), "smaps_pss_mb": 120.0},
            },
        },
        "gpu": {
            "pytorch_raw": {
                "status": "ok",
                "samples_per_second": 250.0,
                "data_wait_percent": 2.0,
                "samples": 128,
                "batches": 2,
                "memory": _memory_report(),
            },
            "medkit_native_prefetch_pinned": {
                "status": "ok",
                "samples_per_second": 375.0,
                "data_wait_percent": 0.25,
                "samples": 128,
                "batches": 2,
                "train_native_prefetch_read_ms_per_batch": 1.5,
                "memory": {**_memory_report(), "smaps_pss_mb": 125.0},
            },
        },
        "profile": {
            "pytorch_raw": {
                "status": "ok",
                "records": [{"timing_scope": "mixed_cuda_events_and_wall"}],
                "summary": {
                    "profile_end_to_end_samples_per_s": 240.0,
                    "profile_phase_budget_ms_per_batch": {
                        "batch_prepare_ms": 20.0,
                        "backward_ms": 50.0,
                    },
                    "profile_phase_budget_end_to_end_percent": {
                        "batch_prepare_ms": 20.0,
                        "backward_ms": 50.0,
                    },
                },
            },
            "medkit_native_prefetch_pinned": {
                "status": "ok",
                "records": [{"timing_scope": "mixed_cuda_events_and_wall"}],
                "summary": {
                    "profile_end_to_end_samples_per_s": 360.0,
                    "profile_phase_budget_ms_per_batch": {
                        "batch_prepare_ms": 2.0,
                        "backward_ms": 50.0,
                    },
                    "profile_phase_budget_end_to_end_percent": {
                        "batch_prepare_ms": 2.0,
                        "backward_ms": 50.0,
                    },
                },
            },
        },
        "quality": {
            "pytorch_raw": {"status": "ok", "macro_auroc": 0.60, "macro_auprc": 0.10},
            "medkit_native_prefetch_pinned": {
                "status": "ok",
                "macro_auroc": 0.62,
                "macro_auprc": 0.11,
            },
        },
    }

    report = benchmark.training_ground_truth_report(reports)
    medkit = report["baselines"]["medkit_native_prefetch_pinned"]
    comparison = report["comparisons"]["medkit_native_prefetch_pinned:vs:pytorch_raw"]

    assert medkit["profile"]["largest_phase"] == {"phase": "backward_ms", "ms_per_batch": 50.0}
    assert medkit["native_prefetch"]["train_native_prefetch_read_ms_per_batch"] == 1.5
    assert comparison["train_samples_per_second_speedup"] == 1.5
    assert comparison["profile_end_to_end_speedup"] == 1.5
    assert comparison["phase_delta_ms_per_batch"]["batch_prepare_ms"] == -18.0


def test_prediction_pairing_accepts_identical_samples_and_metrics():
    benchmark = load_benchmark_module()
    raw = _prediction_summary(benchmark, "pytorch_raw")
    medkit = _prediction_summary(benchmark, "medkit_native_prefetch_pinned")

    comparison = benchmark.paired_prediction_summary(candidate=medkit, raw=raw)

    assert comparison["paired"] is True
    assert comparison["matched_sample_count"] == 2
    assert comparison["missing_from_raw_count"] == 0
    assert comparison["missing_from_medkit_count"] == 0
    assert comparison["identical_order"] is True
    assert comparison["identical_target_order"] is True
    assert comparison["label_mask_hash_match"] is True
    assert comparison["macro_auroc"]["delta"] == 0.0


def test_prediction_pairing_marks_missing_medkit_sample_unpaired():
    benchmark = load_benchmark_module()
    raw = _prediction_summary(benchmark, "pytorch_raw", sample_ids=["a", "b"])
    medkit = _prediction_summary(
        benchmark,
        "medkit_native_prefetch_pinned",
        sample_ids=["a"],
    )

    comparison = benchmark.paired_prediction_summary(candidate=medkit, raw=raw)

    assert comparison["paired"] is False
    assert comparison["missing_from_medkit_count"] == 1


def test_prediction_pairing_detects_target_order_mismatch():
    benchmark = load_benchmark_module()
    raw = _prediction_summary(benchmark, "pytorch_raw", target_names=["A", "B"])
    medkit = _prediction_summary(
        benchmark,
        "medkit_native_prefetch_pinned",
        target_names=["B", "A"],
    )

    comparison = benchmark.paired_prediction_summary(candidate=medkit, raw=raw)

    assert comparison["paired"] is False
    assert comparison["identical_target_order"] is False


def test_prediction_metric_recompute_rejects_nonfinite_logits():
    benchmark = load_benchmark_module()
    rows = _prediction_rows("pytorch_raw")
    rows[0]["logits"][0] = float("nan")

    try:
        benchmark.metric_report_from_prediction_rows(rows, ["A", "B"])
    except ValueError as error:
        assert "logits contains non-finite values" in str(error)
    else:
        raise AssertionError("non-finite logits were accepted")


def test_train_order_recorder_writes_epoch_dropped_evidence(tmp_path):
    import numpy as np

    benchmark = load_benchmark_module()
    targets = ["Pneumonia", "Edema"]
    records = [
        benchmark.SampleRecord(
            sample_id=sample_id,
            patient_id=f"patient-{sample_id}",
            study_id=f"study-{sample_id}",
            image_id=f"image-{sample_id}",
            image_path=f"/tmp/{sample_id}.png",
            filename=f"{sample_id}.png",
            source_split="hf_train_stream",
            width=320,
            height=320,
            labels=labels,
            split="train",
            sha256=f"sha-{sample_id}",
        )
        for sample_id, labels in [
            ("a", {"Pneumonia": 1, "Edema": 0}),
            ("b", {"Pneumonia": 0, "Edema": 1}),
            ("c", {"Pneumonia": 1, "Edema": 0}),
        ]
    ]
    recorder = benchmark.TrainOrderRecorder(
        baseline="medkit_native_prefetch_pinned",
        targets=targets,
        train_records=records,
        artifact_path=tmp_path / "train-order-medkit.jsonl.gz",
        required=True,
    )
    batch_ab = {
        "image": np.zeros((2, 1, 1, 1), dtype="float32"),
        "labels": np.asarray([[1, 0], [0, 1]], dtype="float32"),
        "mask": np.ones((2, 2), dtype="float32"),
        "sample_id": ["a", "b"],
    }
    recorder.record_batch(
        phase="warmup",
        epoch=None,
        batch_index=0,
        global_batch_index=None,
        batch=batch_ab,
    )
    recorder.record_batch(
        phase="train",
        epoch=0,
        batch_index=0,
        global_batch_index=0,
        batch=batch_ab,
    )
    recorder.record_batch(
        phase="train",
        epoch=1,
        batch_index=0,
        global_batch_index=1,
        batch=batch_ab,
    )

    summary = recorder.write()
    normalized = benchmark.train_order_summary_report(
        report_dir=tmp_path,
        train_order={"medkit_native_prefetch_pinned": summary},
        targets=targets,
        capture_enabled=True,
    )

    assert summary["status"] == "ok"
    assert summary["warmup_batches"] == 1
    assert summary["train_batches"] == 2
    assert summary["train_samples"] == 4
    assert summary["same_train_order_each_epoch"] is True
    assert summary["epoch_summaries"][0]["dropped_sample_ids"] == ["c"]
    assert summary["epoch_summaries"][0]["dropped_target_counts"]["Pneumonia"]["positive"] == 1
    assert normalized["baselines"]["medkit_native_prefetch_pinned"]["artifact_rows"] == 3
    assert (
        normalized["baselines"]["medkit_native_prefetch_pinned"][
            "artifact_recheck_matches_summary"
        ]
        is True
    )


def test_train_order_pairing_detects_nonidentical_schedule():
    benchmark = load_benchmark_module()
    raw = {
        "status": "ok",
        "train_batches": 1,
        "train_samples": 2,
        "same_train_order_each_epoch": False,
        "hashes": {
            "train_sample_order_hash": benchmark.stable_hash(["a", "b"]),
            "train_sample_multiset_hash": benchmark.stable_hash(["a", "b"]),
            "dropped_samples_by_epoch_hash": benchmark.stable_hash(
                [{"epoch": 0, "dropped_sample_ids": ["c"]}]
            ),
            "batch_label_sums_hash": "raw-labels",
        },
        "epoch_summaries": [
            {
                "epoch": 0,
                "sample_order_hash": benchmark.stable_hash(["a", "b"]),
                "dropped_sample_count": 1,
                "dropped_sample_ids": ["c"],
            }
        ],
    }
    medkit = {
        "status": "ok",
        "train_batches": 1,
        "train_samples": 2,
        "same_train_order_each_epoch": True,
        "hashes": {
            "train_sample_order_hash": benchmark.stable_hash(["b", "a"]),
            "train_sample_multiset_hash": benchmark.stable_hash(["a", "b"]),
            "dropped_samples_by_epoch_hash": benchmark.stable_hash(
                [{"epoch": 0, "dropped_sample_ids": ["d"]}]
            ),
            "batch_label_sums_hash": "medkit-labels",
        },
        "epoch_summaries": [
            {
                "epoch": 0,
                "sample_order_hash": benchmark.stable_hash(["b", "a"]),
                "dropped_sample_count": 1,
                "dropped_sample_ids": ["d"],
            }
        ],
    }

    comparison = benchmark.paired_train_order_summary(candidate=medkit, raw=raw)

    assert comparison["paired"] is False
    assert comparison["identical_train_order"] is False
    assert comparison["identical_train_sample_multiset"] is True
    assert comparison["identical_dropped_samples_by_epoch"] is False
    assert comparison["epoch_deltas"]["0"]["candidate_only_dropped_sample_count"] == 1


def test_paired_train_batch_schedule_is_replayable_and_epoch_varying():
    benchmark = load_benchmark_module()
    schedule = benchmark.build_train_batch_schedule(
        train_records=[object() for _ in range(10)],
        batch_size=4,
        seed=17,
        epochs=2,
        warmup_batches=1,
        drop_last_train=True,
        shuffle_block_batches=0,
    )

    assert schedule.iteration_names == ("warmup", "epoch:0", "epoch:1")
    assert [len(batches) for batches in schedule.iteration_batches] == [1, 2, 2]
    assert sum(len(batch) for batch in schedule.iteration_batches[1]) == 8
    assert schedule.iteration_batches[1] != schedule.iteration_batches[2]

    raw_sampler = benchmark.FixedTrainBatchScheduleSampler(schedule)
    medkit_sampler = benchmark.FixedTrainBatchScheduleSampler(schedule)

    assert list(raw_sampler) == list(medkit_sampler)
    assert list(raw_sampler) == list(medkit_sampler)
    assert list(raw_sampler) == list(medkit_sampler)
    assert list(raw_sampler) == []
    assert raw_sampler.report_metadata()["batch_schedule_extra_empty_iterations"] == 1


def test_drop_last_train_only_skips_incomplete_batches():
    benchmark = load_benchmark_module()

    assert benchmark.should_skip_incomplete_train_batch(
        batch_samples=31,
        batch_size=32,
        drop_last_train=True,
    )
    assert not benchmark.should_skip_incomplete_train_batch(
        batch_samples=32,
        batch_size=32,
        drop_last_train=True,
    )
    assert not benchmark.should_skip_incomplete_train_batch(
        batch_samples=31,
        batch_size=32,
        drop_last_train=False,
    )


def test_patient_safe_split_assignment_preserves_manifest_schema(tmp_path):
    benchmark = load_benchmark_module()
    records = [
        benchmark.SampleRecord(
            sample_id=f"sample-{index}",
            patient_id=f"patient-{index}",
            study_id=f"study-{index}",
            image_id=f"image-{index}",
            image_path=f"/tmp/image-{index}.png",
            filename=f"{index:08d}_000.png",
            source_split="hf_train_stream",
            width=320,
            height=320,
            labels={"Pneumonia": index % 2},
            sha256=f"sha-{index}",
        )
        for index in range(3)
    ]

    assigned = benchmark.assign_patient_safe_splits(
        records,
        seed=17,
        max_train=1,
        max_val=1,
        max_test=1,
    )
    assert {record.split for record in assigned} == {"train", "val", "test"}
    assert {record.sha256 for record in assigned} == {"sha-0", "sha-1", "sha-2"}

    manifest = tmp_path / "manifest.jsonl"
    benchmark.write_manifest(manifest, assigned)
    reloaded = benchmark.load_manifest(manifest)

    assert {record.split for record in reloaded} == {"train", "val", "test"}
    assert {record.sha256 for record in reloaded} == {"sha-0", "sha-1", "sha-2"}
    assert benchmark.load_manifest_if_compatible(
        manifest,
        requested_samples=3,
    ) == reloaded
    assert benchmark.load_manifest_if_compatible(
        manifest,
        requested_samples=4,
    ) is None


def test_rsna_localization_boxes_roundtrip_and_report(tmp_path):
    benchmark = load_benchmark_module()
    box = benchmark.rsna_annotation_box(
        {
            "labelId": benchmark.RSNA_CALCULATED_LUNG_OPACITY_LABEL,
            "width": 100,
            "height": 200,
            "data": {"x": 10, "y": 20, "width": 30, "height": 40},
        }
    )
    assert box is not None
    assert box["area_fraction"] == 0.06
    records = [
        benchmark.SampleRecord(
            sample_id="sample-positive",
            patient_id="patient-positive",
            study_id="study-positive",
            image_id="image-positive",
            image_path="/tmp/positive.dcm",
            filename="positive.dcm",
            source_split="rsna_subset_group_train",
            width=100,
            height=200,
            labels={"Pneumonia": 1},
            split="train",
            sha256="sha-positive",
            source_format="dicom",
            localization_boxes=[box],
        ),
        benchmark.SampleRecord(
            sample_id="sample-negative",
            patient_id="patient-negative",
            study_id="study-negative",
            image_id="image-negative",
            image_path="/tmp/negative.dcm",
            filename="negative.dcm",
            source_split="rsna_subset_group_train",
            width=100,
            height=200,
            labels={"Pneumonia": 0},
            split="val",
            sha256="sha-negative",
            source_format="dicom",
        ),
    ]

    manifest = tmp_path / "manifest.jsonl"
    benchmark.write_manifest(manifest, records)
    reloaded = benchmark.load_manifest(manifest)
    report = benchmark.localization_report(reloaded, ["Pneumonia"])

    assert reloaded[0].localization_boxes == [box]
    assert reloaded[1].localization_boxes == []
    assert report["status"] == "ok"
    assert report["overall"]["positive_samples_with_boxes"] == 1
    assert report["overall"]["positive_samples_without_boxes"] == 0
    assert report["overall"]["negative_samples_with_boxes"] == 0
    assert report["overall"]["box_area_fraction"]["median"] == 0.06


def test_cam_localization_metrics_hit_box():
    benchmark = load_benchmark_module()
    numpy = benchmark.import_numpy()
    heatmap = numpy.zeros((10, 10), dtype="float32")
    heatmap[3, 3] = 1.0
    mask = benchmark.box_union_mask(
        boxes=[
            {
                "x1": 2,
                "y1": 2,
                "x2": 5,
                "y2": 5,
                "image_width": 10,
                "image_height": 10,
            }
        ],
        height=10,
        width=10,
    )

    row = benchmark.cam_localization_sample_metrics(
        heatmap=heatmap,
        box_mask=mask,
        sample_id="sample-positive",
        box_count=1,
    )
    report = benchmark.summarize_cam_localization_rows(
        target="Pneumonia",
        rows=[row],
    )

    assert row["top1_hit"] is True
    assert row["top_1pct"]["hit"] is True
    assert row["top_1pct"]["box_coverage"] == pytest.approx(1 / 9)
    assert report["status"] == "ok"
    assert report["method"] == "densenet121_classifier_cam"
    assert report["top1_hit_rate"] == 1.0
    assert report["top_percent"]["top_1pct"]["hit_rate"] == 1.0


def test_gpu_utilization_summary_from_nvidia_smi_values():
    benchmark = load_benchmark_module()
    first = benchmark.parse_nvidia_smi_utilization_values(["12", "4", "1024", "81920", "85.5"])
    second = benchmark.parse_nvidia_smi_utilization_values(["48", "9", "2048", "81920", "140.5"])

    report = benchmark.gpu_utilization_summary(
        samples=[first, second],
        errors=[],
        enabled=True,
        device_index=0,
        interval_seconds=1.0,
        disabled_reason="",
        started_at=10.0,
        stopped_at=12.5,
    )

    assert report["status"] == "ok"
    assert report["sample_count"] == 2
    assert report["duration_seconds"] == 2.5
    assert report["gpu_utilization_percent"]["mean"] == 30.0
    assert report["gpu_utilization_percent"]["median"] == 30.0
    assert report["memory_utilization_percent"]["max"] == 9.0
    assert report["memory_used_mb"]["mean"] == 1536.0
    assert report["power_draw_w"]["mean"] == 113.0


def test_split_report_records_reproducibility_checksums(tmp_path):
    benchmark = load_benchmark_module()
    records = [
        benchmark.SampleRecord(
            sample_id=f"sample-{index}",
            patient_id=f"patient-{index}",
            study_id=f"study-{index}",
            image_id=f"image-{index}",
            image_path=f"/tmp/image-{index}.png",
            filename=f"{index:08d}_000.png",
            source_split="hf_train_stream",
            width=320,
            height=320,
            labels={"Pneumonia": index % 2},
            split="train" if index < 2 else "val",
            sha256=f"sha-{index}",
        )
        for index in range(3)
    ]

    report = benchmark.write_split_file(tmp_path / "splits.json", records)

    assert report["split_checksum"]
    assert report["patient_split_checksum"]
    assert report["counts"] == {"train": 2, "val": 1, "test": 0}


def test_quality_gate_and_balanced_loss_helpers_reject_weak_quality():
    benchmark = load_benchmark_module()
    records = [
        benchmark.SampleRecord(
            sample_id=f"sample-{index}",
            patient_id=f"patient-{index}",
            study_id=f"study-{index}",
            image_id=f"image-{index}",
            image_path=f"/tmp/image-{index}.png",
            filename=f"{index:08d}_000.png",
            source_split="hf_train_stream",
            width=320,
            height=320,
            labels={"Pneumonia": 1 if index == 0 else 0},
            split="train",
            sha256=f"sha-{index}",
        )
        for index in range(4)
    ]

    assert benchmark.class_pos_weight_values(records, ["Pneumonia"]) == [3.0]
    assert benchmark.class_pos_weight_values(records, ["Pneumonia"], cap=2.0) == [2.0]

    report = benchmark.quality_gate_report(
        quality={
            "pytorch_raw": {
                "status": "ok",
                "samples": 16,
                "metric_target_count": 0,
                "macro_auroc": None,
                "macro_auprc": None,
                "prediction_capture": {"enabled": True, "status": "ok"},
                "metric_recompute_matches_predictions": True,
            }
        },
        train_order={},
        validation={
            "split_audit": {
                "patient_overlap_count": 0,
                "study_overlap_count": 0,
                "duplicate_hash_overlap_count": 0,
            }
        },
        run_metadata={
            "quality_gate": True,
            "quality_min_eval_samples": 32,
            "quality_min_metric_targets": 1,
            "quality_min_macro_auroc": 0.5,
            "quality_min_macro_auprc": 0.1,
        },
    )

    assert report["status"] == "failed"
    assert any("eval samples" in error for error in report["errors"])
    assert any("metric targets" in error for error in report["errors"])


def test_threshold_report_includes_max_f1_operating_point():
    benchmark = load_benchmark_module()
    numpy = benchmark.import_numpy()

    report = benchmark.threshold_report(
        numpy.array([[1.0], [1.0], [0.0], [0.0]], dtype="float32"),
        numpy.array([[0.9], [0.8], [0.4], [0.1]], dtype="float32"),
        numpy.ones((4, 1), dtype="float32"),
        ["Finding"],
    )

    max_f1 = report["targets"]["Finding"]["max_f1"]
    assert max_f1["threshold"] == 0.800000011920929
    assert max_f1["precision"] == 1.0
    assert max_f1["sensitivity"] == 1.0
    assert max_f1["f1"] == 1.0


def test_profile_report_disabled_without_requested_batches():
    benchmark = load_benchmark_module()

    report = benchmark.profile_report_for_baseline(
        baseline="pytorch_raw",
        requested_batches=0,
        records=[],
    )

    assert report["status"] == "disabled"
    assert report["summary"] == {}


def test_run_summary_consistency_accepts_matching_provenance_and_rejects_drift():
    benchmark = load_benchmark_module()
    run_metadata = {
        "run_id": "run-1",
        "dataset_requested": "example/cxr",
        "dataset_loaded": "example/cxr",
        "targets": ["Pneumonia"],
        "baselines": ["pytorch_raw"],
        "image_size": 224,
        "cache_image_size": 224,
        "cache_dtype": "float32",
        "batch_size": 64,
        "drop_last_train": True,
        "workers": 8,
        "prefetch_depth": 2,
        "prefetch_read_workers": 4,
        "shuffle_block_batches": 0,
        "gpu_prefetch_batches": 0,
        "gpu_prefetch_reuse_buffers": False,
        "sync_every_step": True,
        "channels_last": False,
        "torch_compile": False,
        "torch_compile_mode": "default",
        "learning_rate": 1.0e-4,
        "amp_dtype": "auto",
        "loss_pos_weight": "none",
        "quality_gate": False,
        "quality_min_eval_samples": 0,
        "quality_min_metric_targets": 0,
        "quality_min_macro_auroc": 0.0,
        "quality_min_macro_auprc": 0.0,
        "eval_predictions": False,
        "train_order_evidence": False,
        "paired_train_order": False,
        "read_mode": "mmap",
        "include_metadata": False,
        "profile_batches": 2,
        "loader_batches": 64,
        "warmup_batches": 4,
        "max_train_batches": 0,
        "max_eval_batches": 1,
        "seed": 17,
    }
    manifest_summary = {
        "dataset_loaded": "example/cxr",
        "samples": 4,
        "targets": ["Pneumonia"],
    }
    split_report = {"counts": {"train": 2, "val": 1, "test": 1}}
    cache_report = {
        "cache_dir": "/cache/cxr",
        "cache_reused": False,
        "dtype": "float32",
        "image_size": 224,
        "transform_fingerprint": "transform-hash",
        "source_manifest_checksum": "manifest-hash",
        "splits": {
            "train": {"samples": 2},
            "val": {"samples": 1},
            "test": {"samples": 1},
        },
    }
    profile_records = [
        {
            "samples": 64,
            "data_wait_ms": 1.0,
            "h2d_ms": 0.5,
            "h2d_timing_mode": benchmark.H2D_TIMING_DIRECT_COPY,
            "forward_ms": 2.0,
            "backward_ms": 3.0,
            "optimizer_ms": 0.75,
            "total_step_ms": 10.0,
        },
        {
            "samples": 64,
            "data_wait_ms": 2.0,
            "h2d_ms": 0.75,
            "h2d_timing_mode": benchmark.H2D_TIMING_DIRECT_COPY,
            "forward_ms": 3.0,
            "backward_ms": 4.0,
            "optimizer_ms": 1.0,
            "total_step_ms": 15.0,
        },
    ]
    profile_summary = benchmark.summarize_profile_records(profile_records)
    profile_summary["profile_artifact_path"] = "step-profile.json"
    reports = {
        "loader": {"pytorch_raw": {"status": "ok", "samples_per_second": 123.4567}},
        "gpu": {"pytorch_raw": {"status": "ok", "samples_per_second": 234.5678}},
        "quality": {"pytorch_raw": {"status": "ok", "macro_auroc": 0.912345}},
        "profile": {
            "pytorch_raw": {
                "status": "ok",
                "records": profile_records,
                "summary": profile_summary,
            }
        },
        "thresholds": {},
        "predictions": {},
        "train_order": {},
        "localization_eval": {},
    }
    environment = {
        "git_commit": "abc123",
        "git_status_short": "",
        "run_metadata": run_metadata,
    }
    provenance = benchmark.build_run_provenance(
        args=argparse.Namespace(),
        run_id="run-1",
        run_metadata=run_metadata,
        manifest_summary=manifest_summary,
        split_report=split_report,
        cache_report=cache_report,
        environment=environment,
        argv=["cxr_classification_benchmark.py", "--run-id", "run-1"],
    )
    summary = {
        "run_id": "run-1",
        "report_dir": "/reports/run-1",
        "dataset_loaded": "example/cxr",
        "samples": 4,
        "targets": ["Pneumonia"],
        "device": "cuda:0",
        "loader_samples_per_second": {"pytorch_raw": 123.457},
        "train_samples_per_second": {"pytorch_raw": 234.568},
        "quality_macro_auroc": {"pytorch_raw": 0.91234},
        "quality_gate": {"status": "recorded", "enabled": False, "errors": []},
        "profile": {"pytorch_raw": profile_summary},
        "memory": benchmark.memory_summary(reports),
        "predictions": {},
        "localization_eval": {},
        "provenance": provenance,
    }

    consistency = benchmark.validate_run_summary_consistency(
        summary=summary,
        run_metadata=run_metadata,
        manifest_summary=manifest_summary,
        split_report=split_report,
        cache_report=cache_report,
        reports=reports,
        environment=environment,
    )

    assert consistency["status"] == "ok"

    drifted = copy.deepcopy(summary)
    drifted["provenance"]["cache_dtype"] = "uint8"
    errors = benchmark.run_summary_consistency_errors(
        summary=drifted,
        run_metadata=run_metadata,
        manifest_summary=manifest_summary,
        split_report=split_report,
        cache_report=cache_report,
        reports=reports,
        environment=environment,
    )
    assert any("provenance.cache_dtype mismatch" in error for error in errors)


def test_gate_presets_build_same_batch_raw_and_medkit_rows_on_one_gpu_type():
    matrix = load_matrix_module()

    h100_args = matrix.parse_args(
        ["--gate", "h100-512-b32", "--batch-id", "repeat-h100", "--dry-run"]
    )
    h100_rows = matrix.rows_for_args(h100_args)
    h100_run_ids = [matrix.run_id_for("repeat-h100", row) for row in h100_rows]

    assert h100_args.modal_gpu == "H100"
    assert h100_args.image_size == 512
    assert h100_args.batch_size == 32
    assert h100_args.profile_batches == 128
    assert h100_args.shuffle_block_batches == 0
    assert h100_args.gpu_prefetch_batches == 0
    assert h100_args.gpu_prefetch_reuse_buffers is False
    assert h100_args.sync_every_step is True
    assert h100_args.repeats == 3
    assert h100_args.fail_fast is True
    assert len(h100_rows) == 9
    assert h100_run_ids[:3] == [
        "repeat-h100-r01-pytorch-raw-float32-mmap",
        "repeat-h100-r01-medkit-native-prefetch-pinned-float32-stream",
        "repeat-h100-r01-medkit-native-prefetch-pinned-uint8-stream",
    ]
    assert h100_run_ids[-3:] == [
        "repeat-h100-r03-pytorch-raw-float32-mmap",
        "repeat-h100-r03-medkit-native-prefetch-pinned-float32-stream",
        "repeat-h100-r03-medkit-native-prefetch-pinned-uint8-stream",
    ]
    assert {
        row.baseline for row in h100_rows
    } == {"pytorch_raw", "medkit_native_prefetch_pinned"}
    for row in h100_rows:
        command = matrix.command_with_env(
            h100_args,
            matrix.build_command(h100_args, run_id=matrix.run_id_for("repeat-h100", row), row=row),
        )
        assert command[0] == "MEDKIT_MODAL_GPU=H100"
        assert "--shuffle-block-batches" in command
        assert "--gpu-prefetch-batches" in command
        assert "--no-gpu-prefetch-reuse-buffers" in command
        assert "--sync-every-step" in command
        assert "--no-channels-last" in command
        assert "--no-torch-compile" in command
        assert command[command.index("--torch-compile-mode") + 1] == "default"

    l4_args = matrix.parse_args(
        [
            "--gate",
            "l4-224-b64",
            "--batch-id",
            "repeat-l4",
            "--shuffle-block-batches",
            "8",
            "--gpu-prefetch-batches",
            "2",
            "--gpu-prefetch-reuse-buffers",
            "--repeats",
            "1",
        ]
    )
    l4_rows = matrix.rows_for_args(l4_args)

    assert l4_args.modal_gpu == "L4"
    assert l4_args.image_size == 224
    assert l4_args.batch_size == 64
    assert l4_args.profile_batches == 64
    assert l4_args.shuffle_block_batches == 8
    assert l4_args.gpu_prefetch_batches == 2
    assert l4_args.gpu_prefetch_reuse_buffers is True
    assert l4_args.sync_every_step is True
    assert l4_args.repeats == 1
    assert [matrix.run_id_for("repeat-l4", row) for row in l4_rows] == [
        "repeat-l4-pytorch-raw-float32-mmap",
        "repeat-l4-medkit-native-prefetch-pinned-float32-stream",
    ]

    quality_args = matrix.parse_args(
        ["--gate", "l4-quality-224-b64", "--batch-id", "quality-l4"]
    )
    quality_rows = matrix.rows_for_args(quality_args)

    assert quality_args.modal_gpu == "L4"
    assert quality_args.epochs == 2
    assert quality_args.max_eval_batches == 0
    assert quality_args.loss_pos_weight == "balanced"
    assert quality_args.loss_pos_weight_cap == 10.0
    assert quality_args.quality_gate is True
    assert quality_args.quality_min_eval_samples == 900
    assert quality_args.quality_min_metric_targets == 5
    assert quality_args.quality_min_macro_auroc == 0.55
    assert quality_args.quality_min_macro_auprc == 0.10
    quality_command = matrix.build_command(
        quality_args,
        run_id=matrix.run_id_for("quality-l4", quality_rows[0]),
        row=quality_rows[0],
    )
    assert "--quality-gate" in quality_command
    assert "--train-order-evidence" in quality_command
    assert "--paired-train-order" in quality_command
    assert "--loss-pos-weight" in quality_command
    assert "balanced" in quality_command
    assert "--loss-pos-weight-cap" in quality_command
    assert "10.0" in quality_command
    assert "--no-channels-last" in quality_command
    assert "--no-torch-compile" in quality_command

    shared = matrix.shared_data_paths("quality-l4")
    assert shared["manifest"] == "/cache/results/cxr/quality-l4-prepare-data/manifest.jsonl"
    assert shared["splits"] == "/cache/results/cxr/quality-l4-prepare-data/splits.json"
    assert matrix.auto_shared_data_required(quality_args) is True
    quality_args.manifest = shared["manifest"]
    quality_args.splits = shared["splits"]
    quality_command = matrix.build_command(
        quality_args,
        run_id=matrix.run_id_for("quality-l4", quality_rows[0]),
        row=quality_rows[0],
    )
    assert quality_command[quality_command.index("--manifest") + 1] == shared["manifest"]
    assert quality_command[quality_command.index("--splits") + 1] == shared["splits"]
    prepare_command = matrix.build_prepare_command(
        matrix.parse_args(["--gate", "l4-quality-224-b64", "--batch-id", "quality-l4"]),
        run_id="quality-l4-prepare-data",
    )
    assert "--prepare-only" in prepare_command
    assert "--manifest" not in prepare_command
    assert "--splits" not in prepare_command


def test_matrix_modal_cli_command_can_be_overridden(monkeypatch):
    matrix = load_matrix_module()
    monkeypatch.setenv("MEDKIT_MODAL_CLI", "uvx --python 3.11 modal")
    args = matrix.parse_args(["--gate", "l4-224-b64", "--batch-id", "repeat-l4"])
    row = matrix.rows_for_args(args)[0]

    command = matrix.build_command(args, run_id=matrix.run_id_for("repeat-l4", row), row=row)

    assert command[:5] == ["uvx", "--python", "3.11", "modal", "run"]


def test_modal_cxr_wrapper_exposes_sync_policy():
    text = MODAL_CLASSIFICATION_PATH.read_text()

    assert "sync_every_step: bool = True" in text
    assert '"--sync-every-step" if sync_every_step else "--no-sync-every-step"' in text
    assert "sync_every_step=sync_every_step" in text
    assert "channels_last: bool = False" in text
    assert '"--channels-last" if channels_last else "--no-channels-last"' in text
    assert "channels_last=channels_last" in text
    assert "torch_compile: bool = False" in text
    assert '"--torch-compile" if torch_compile else "--no-torch-compile"' in text
    assert "--torch-compile-mode" in text
    assert "torch_compile=torch_compile" in text
    assert "torch_compile_mode=torch_compile_mode" in text
    assert "learning_rate: float = 1.0e-4" in text
    assert "--learning-rate" in text
    assert "learning_rate=learning_rate" in text
    assert 'amp_dtype: str = "auto"' in text
    assert "--amp-dtype" in text
    assert "amp_dtype=amp_dtype" in text
    assert 'model_init: str = "random"' in text
    assert "--model-init" in text
    assert "model_init=model_init" in text
    assert 'loss_kind: str = "bce"' in text
    assert "--loss-kind" in text
    assert "loss_kind=loss_kind" in text
    assert "loss_pos_weight_cap: float = 0.0" in text
    assert "--loss-pos-weight-cap" in text
    assert "loss_pos_weight_cap=loss_pos_weight_cap" in text
    assert "focal_gamma: float = 2.0" in text
    assert "--focal-gamma" in text
    assert "focal_gamma=focal_gamma" in text
    assert "focal_alpha: float = 0.0" in text
    assert "--focal-alpha" in text
    assert "focal_alpha=focal_alpha" in text
    assert "gpu_prefetch_reuse_buffers: bool = False" in text
    assert "--gpu-prefetch-reuse-buffers" in text
    assert "gpu_prefetch_reuse_buffers=gpu_prefetch_reuse_buffers" in text
    assert "train_order_evidence: bool | None = None" in text
    assert "--train-order-evidence" in text
    assert "train_order_evidence=train_order_evidence" in text
    assert "paired_train_order: bool | None = None" in text
    assert "--paired-train-order" in text
    assert "paired_train_order=paired_train_order" in text
    assert "volume.reload()" in text
    assert "manifest: str = \"\"" in text
    assert "--manifest" in text
    assert "prepare_only: bool = False" in text
    assert "--prepare-only" in text


def test_native_prefetch_loader_factory_passes_block_shuffle(monkeypatch, tmp_path):
    benchmark = load_benchmark_module()
    captured: dict[str, object] = {}

    class FakeDataLoader:
        def __init__(self, dataset, **kwargs):
            self.dataset = dataset
            self.kwargs = kwargs

    class FakeDataset:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.kwargs = kwargs
            self.shuffle_block_batches = kwargs["shuffle_block_batches"]
            self.native_prefetch_stats = {
                "batches": 2,
                "indexed_batches": 2,
                "indexed_runs": 5,
                "read_micros": 1200,
                "scatter_micros": 800,
            }

        def report_metadata(self):
            return {
                "worker_mode": "fake_actual_dataset_report",
                "read_mode": self.kwargs["read_mode"],
                "include_metadata": self.kwargs["include_metadata"],
                "shuffle_block_batches": self.kwargs["shuffle_block_batches"],
                "prefetch_depth": self.kwargs["prefetch_depth"],
                "prefetch_read_workers": self.kwargs["read_workers"],
                "native_prefetch_stats": dict(self.native_prefetch_stats),
            }

    fake_torch = types.SimpleNamespace(
        utils=types.SimpleNamespace(
            data=types.SimpleNamespace(DataLoader=FakeDataLoader)
        )
    )
    fake_medkit = types.SimpleNamespace(MedkitCxrNativePrefetchDataset=FakeDataset)

    monkeypatch.setattr(benchmark, "import_torch", lambda: fake_torch)
    monkeypatch.setattr(benchmark, "import_numpy", lambda: types.SimpleNamespace())
    monkeypatch.setattr(benchmark, "import_medkit_rs", lambda: fake_medkit)
    monkeypatch.setattr(benchmark, "cache_normalization", lambda _cache_dir: (0.5, 0.25))
    monkeypatch.setattr(benchmark, "cache_dtype_from_metadata", lambda _cache_dir: "float32")

    factory = benchmark.make_loader_factory(
        baseline="medkit_native_prefetch_pinned",
        records=[],
        targets=["Pneumonia"],
        cache_dir=tmp_path,
        webdataset_dir=tmp_path / "webdataset",
        image_size=224,
        batch_size=64,
        workers=4,
        prefetch_depth=2,
        prefetch_read_workers=4,
        shuffle_block_batches=8,
        read_mode="stream",
        include_metadata=True,
        drop_last_train=True,
        seed=17,
    )

    loader = factory("train", shuffle=True)

    assert isinstance(loader, FakeDataLoader)
    assert loader.dataset.shuffle_block_batches == 8
    assert captured["shuffle_block_batches"] == 8
    assert captured["shuffle"] is True
    assert captured["read_mode"] == "stream"
    assert captured["prefetch_depth"] == 2
    assert captured["read_workers"] == 4
    assert captured["include_metadata"] is True
    assert captured["drop_last"] is True
    assert loader.report_metadata()["shuffle_block_batches"] == 8
    assert loader.report_metadata()["worker_mode"] == "fake_actual_dataset_report"
    assert loader.report_metadata()["prefetch_read_workers"] == 4
    assert loader.report_metadata()["native_prefetch_stats"]["indexed_runs"] == 5
    loader.dataset.native_prefetch_stats["indexed_runs"] = 7
    assert loader.report_metadata()["native_prefetch_stats"]["indexed_runs"] == 7


def test_native_prefetch_timing_fields_summarize_pipeline_stats():
    benchmark = load_benchmark_module()

    fields = benchmark.native_prefetch_timing_fields(
        {
            "native_prefetch_stats": {
                "batches": 4,
                "indexed_batches": 4,
                "indexed_runs": 12,
                "read_micros": 8000,
                "scatter_micros": 4000,
                "slot_count": 2,
                "preallocated_batch_buffers": 2,
                "buffer_reuse_enabled": True,
                "pin_memory": True,
            }
        },
        batches=4,
        elapsed_ms=120.0,
        prefix="train",
    )

    assert fields["train_native_prefetch_runs_per_batch"] == 3.0
    assert fields["train_native_prefetch_read_ms_per_batch"] == 2.0
    assert fields["train_native_prefetch_scatter_ms_per_batch"] == 1.0
    assert fields["train_native_prefetch_read_scatter_ms_per_batch"] == 3.0
    assert fields["train_native_prefetch_read_scatter_percent"] == 10.0
    assert fields["train_native_prefetch_slot_count"] == 2.0
    assert fields["train_native_prefetch_preallocated_batch_buffers"] == 2.0
    assert fields["train_native_prefetch_buffer_reuse_enabled"] is True
    assert fields["train_native_prefetch_pin_memory"] is True


def test_matrix_pipeline_validation_rejects_policy_drift():
    matrix = load_matrix_module()
    errors: list[str] = []

    matrix.validate_pipeline_request_metadata(
        errors=errors,
        context="gpu",
        pipeline={
            "native_prefetch": True,
            "shuffle_block_batches": 0,
            "prefetch_depth": 1,
            "prefetch_read_workers": 4,
        },
        metadata={
            "shuffle_block_batches": 8,
            "prefetch_depth": 2,
            "prefetch_read_workers": 4,
        },
    )

    assert any("shuffle_block_batches 0 != expected 8" in error for error in errors)
    assert any("prefetch_depth 1 != expected 2" in error for error in errors)


def test_matrix_row_validation_requires_summary_consistency_and_provenance():
    benchmark = load_benchmark_module()
    matrix = load_matrix_module()
    baseline = "pytorch_raw"
    run_id = "batch-pytorch-raw-float32-mmap"
    row = matrix.Row(
        name="pytorch-raw-float32-mmap",
        baseline=baseline,
        cache_dtype="float32",
        read_mode="mmap",
        purpose="test row",
    )
    active = types.SimpleNamespace(row=row, run_id=run_id)
    records = [
        {
            "samples": 32,
            "data_wait_ms": 1.0,
            "h2d_ms": 0.5,
            "h2d_timing_mode": benchmark.H2D_TIMING_DIRECT_COPY,
            "forward_ms": 2.0,
            "backward_ms": 3.0,
            "optimizer_ms": 0.75,
            "total_step_ms": 10.0,
        }
        for _ in range(20)
    ]
    profile_summary = benchmark.summarize_profile_records(records)
    profile_summary["profile_artifact_path"] = "step-profile.json"
    gpu_row = {
        "status": "ok",
        "samples_per_second": 222.222,
        "memory": _memory_report(),
        **profile_summary,
    }
    loader_row = {"status": "ok", "samples_per_second": 111.111, "memory": _memory_report()}
    environment = {
        "run_metadata": {
            "run_id": run_id,
            "cache_dtype": "float32",
            "read_mode": "mmap",
            "profile_batches": 20,
            "image_size": 224,
            "cache_image_size": 224,
            "batch_size": 32,
            "workers": 8,
            "prefetch_depth": 2,
            "prefetch_read_workers": 4,
            "shuffle_block_batches": 0,
            "gpu_prefetch_batches": 0,
            "gpu_prefetch_reuse_buffers": False,
            "sync_every_step": True,
            "channels_last": False,
            "torch_compile": False,
            "torch_compile_mode": "default",
            "learning_rate": 1.0e-4,
            "amp_dtype": "auto",
            "model_init": "random",
            "loss_kind": "bce",
            "loss_pos_weight": "none",
            "loss_pos_weight_cap": 0.0,
            "focal_gamma": 2.0,
            "focal_alpha": 0.0,
            "quality_gate": False,
            "quality_min_eval_samples": 0,
            "quality_min_metric_targets": 0,
            "quality_min_macro_auroc": 0.0,
            "quality_min_macro_auprc": 0.0,
            "eval_predictions": False,
            "train_order_evidence": False,
            "paired_train_order": False,
            "seed": 17,
        }
    }
    run_summary = {
        "run_id": run_id,
        "loader_samples_per_second": {baseline: 111.111},
        "train_samples_per_second": {baseline: 222.222},
        "provenance": {
            "run_id": run_id,
            "dataset_loaded": "example/cxr",
            "samples": 128,
            "targets": ["Pneumonia"],
            "baselines": [baseline],
            "image_size": 224,
            "cache_image_size": 224,
            "cache_dtype": "float32",
            "batch_size": 32,
            "drop_last_train": True,
            "workers": 8,
            "prefetch_depth": 2,
            "prefetch_read_workers": 4,
            "shuffle_block_batches": 0,
            "gpu_prefetch_batches": 0,
            "gpu_prefetch_reuse_buffers": False,
            "sync_every_step": True,
            "channels_last": False,
            "torch_compile": False,
            "torch_compile_mode": "default",
            "learning_rate": 1.0e-4,
            "amp_dtype": "auto",
            "model_init": "random",
            "loss_kind": "bce",
            "loss_pos_weight": "none",
            "loss_pos_weight_cap": 0.0,
            "focal_gamma": 2.0,
            "focal_alpha": 0.0,
            "quality_gate": False,
            "quality_min_eval_samples": 0,
            "quality_min_metric_targets": 0,
            "quality_min_macro_auroc": 0.0,
            "quality_min_macro_auprc": 0.0,
            "eval_predictions": False,
            "train_order_evidence": False,
            "paired_train_order": False,
            "read_mode": "mmap",
            "include_metadata": False,
            "profile_batches": 20,
            "seed": 17,
            "cache": {},
            "artifacts": {
                "summary_consistency": "summary-consistency.json",
                "step_profile": "step-profile.json",
            },
        },
    }
    summary_consistency = {"status": "ok", "run_id": run_id, "errors": []}
    profile = {
        baseline: {
            "status": "ok",
            "records": records,
            "summary": profile_summary,
        }
    }

    errors = matrix.validate_row_artifacts(
        active=active,
        returncode=0,
        modal_result={
            "status": "ok",
            "artifacts": {
                "run-summary.json": run_summary,
                "summary-consistency.json": summary_consistency,
            },
        },
        run_summary=run_summary,
        loader={baseline: loader_row},
        gpu={baseline: gpu_row},
        profile=profile,
        quality={baseline: {"status": "ok"}},
        quality_gate={"status": "recorded", "enabled": False, "errors": []},
        environment=environment,
        summary_consistency=summary_consistency,
    )

    assert errors == []

    bad_consistency = {"status": "failed", "run_id": run_id, "errors": ["drift"]}
    errors = matrix.validate_row_artifacts(
        active=active,
        returncode=0,
        modal_result={"status": "ok", "artifacts": {}},
        run_summary=run_summary,
        loader={baseline: loader_row},
        gpu={baseline: gpu_row},
        profile=profile,
        quality={baseline: {"status": "ok"}},
        quality_gate={"status": "failed", "enabled": True, "errors": ["drift"]},
        environment=environment,
        summary_consistency=bad_consistency,
    )

    assert any("summary-consistency status" in error for error in errors)


def test_matrix_repeat_summary_aggregates_three_repeat_metrics():
    benchmark = load_benchmark_module()
    matrix = load_matrix_module()
    results = []
    for repeat_index, samples_per_second in enumerate([350.0, 360.0, 370.0]):
        baseline = "medkit_native_prefetch_pinned"
        results.append(
            {
                "run_id": f"batch-r0{repeat_index + 1}-medkit-native-prefetch-pinned-float32-stream",
                "status": "ok",
                "baseline": baseline,
                "cache_dtype": "float32",
                "read_mode": "stream",
                "repeat_index": repeat_index,
                "repeat_count": 3,
                "gpu": {
                    baseline: {
                        "samples_per_second": samples_per_second,
                        "data_wait_percent": 0.25,
                        "train_native_prefetch_read_ms_per_batch": 1.0 + repeat_index,
                        "train_native_prefetch_scatter_ms_per_batch": 0.5,
                        "train_native_prefetch_read_scatter_ms_per_batch": 1.5 + repeat_index,
                        "train_native_prefetch_read_scatter_percent": 2.0 + repeat_index,
                        "train_native_prefetch_runs_per_batch": 3.0,
                        "memory": {
                            **_memory_report(),
                            "smaps_pss_mb": 5700.0 + repeat_index,
                            "smaps_pss_cache_images_mb": 0.0,
                        },
                    }
                },
                "loader": {baseline: {"samples_per_second": 5000.0 + repeat_index}},
                "profile": {
                    baseline: {
                        "summary": {
                            "profile_end_to_end_samples_per_s": samples_per_second - 1.0,
                            "profile_batch_prepare_ms_mean": 1.0 + repeat_index,
                            "profile_residual_step_ms_mean": 0.25 + repeat_index,
                            "profile_prefetch_maintenance_wall_ms_mean": 0.1,
                        }
                    }
                },
                "predictions": {
                    "baselines": {
                        baseline: _prediction_summary(benchmark, baseline),
                    }
                },
            }
        )
        results.append(
            {
                "run_id": f"batch-r0{repeat_index + 1}-pytorch-raw-float32-mmap",
                "status": "ok",
                "baseline": "pytorch_raw",
                "cache_dtype": "float32",
                "read_mode": "mmap",
                "repeat_index": repeat_index,
                "repeat_count": 3,
                "gpu": {
                    "pytorch_raw": {
                        "samples_per_second": 180.0,
                        "data_wait_percent": 1.0,
                        "memory": _memory_report(),
                    }
                },
                "loader": {"pytorch_raw": {"samples_per_second": 1000.0}},
                "profile": {
                    "pytorch_raw": {
                        "summary": {
                            "profile_end_to_end_samples_per_s": 179.0
                        }
                    }
                },
                "predictions": {
                    "baselines": {
                        "pytorch_raw": _prediction_summary(benchmark, "pytorch_raw"),
                    }
                },
            }
        )

    summary = matrix.summarize_repeats(results, running=[], pending=[])
    group = summary["groups"]["medkit_native_prefetch_pinned:float32:stream"]
    comparison = summary["comparisons"][
        "medkit_native_prefetch_pinned:float32:stream:vs:pytorch_raw:float32:mmap"
    ]

    assert summary["status"] == "ok"
    assert group["status"] == "ok"
    assert group["expected_repeats"] == 3
    assert group["ok_repeats"] == 3
    assert group["metrics"]["train_samples_per_second"]["mean"] == 360.0
    assert group["metrics"]["train_native_prefetch_read_ms_per_batch"]["mean"] == 2.0
    assert group["metrics"]["train_native_prefetch_runs_per_batch"]["mean"] == 3.0
    assert group["metrics"]["profile_end_to_end_samples_per_second"]["count"] == 3
    assert group["metrics"]["profile_batch_prepare_ms"]["mean"] == 2.0
    assert group["metrics"]["profile_residual_step_ms"]["mean"] == 1.25
    assert abs(group["metrics"]["profile_prefetch_maintenance_wall_ms"]["mean"] - 0.1) < 1e-12
    assert comparison["train_samples_per_second_speedup"] == 2.0
    prediction_comparison = summary["prediction_comparisons"][
        "medkit_native_prefetch_pinned:float32:stream:r01:vs:pytorch_raw:float32:mmap:r01"
    ]
    assert prediction_comparison["paired"] is True
    assert prediction_comparison["missing_from_medkit_count"] == 0


def test_matrix_prediction_validation_requires_quality_artifact(tmp_path):
    matrix = load_matrix_module()
    errors: list[str] = []

    matrix.validate_prediction_artifacts(
        errors=errors,
        baseline="medkit_native_prefetch_pinned",
        row_dir=tmp_path,
        quality_row={"prediction_capture": {"enabled": True}, "metric_recompute_matches_predictions": True},
        predictions={},
        quality_gate_enabled=True,
    )

    assert any("missing eval-predictions-summary" in error for error in errors)

    artifact_name = "eval-predictions-medkit_native_prefetch_pinned.jsonl.gz"
    (tmp_path / artifact_name).write_bytes(b"placeholder")
    errors = []
    matrix.validate_prediction_artifacts(
        errors=errors,
        baseline="medkit_native_prefetch_pinned",
        row_dir=tmp_path,
        quality_row={"prediction_capture": {"enabled": True}, "metric_recompute_matches_predictions": True},
        predictions={
            "baselines": {
                "medkit_native_prefetch_pinned": {
                    "enabled": True,
                    "status": "ok",
                    "artifact_path": artifact_name,
                    "metric_recompute_matches_quality": True,
                    "metric_recompute_matches_artifact": True,
                }
            }
        },
        quality_gate_enabled=True,
    )

    assert errors == []


def test_h100_promotion_readiness_rejects_noisy_raw_comparator():
    matrix = load_matrix_module()
    repeat_summary = {
        "groups": {
            "pytorch_raw:float32:mmap": _repeat_group(
                baseline="pytorch_raw",
                cv_percent=16.31,
                classification="reject",
            ),
            "medkit_native_prefetch_pinned:float32:stream": _repeat_group(
                baseline="medkit_native_prefetch_pinned",
                cv_percent=0.56,
                classification="ok",
            ),
        }
    }

    readiness = matrix.promotion_readiness_report(
        repeat_summary,
        batch_id="h100-noisy",
        modal_gpu="H100",
    )

    candidate = readiness["candidates"]["medkit_native_prefetch_pinned:float32:stream"]
    assert readiness["status"] == "rejected"
    assert candidate["status"] == "rejected"
    assert candidate["speedup_denominator"]["group_key"] == "pytorch_raw:float32:mmap"
    assert candidate["speedup_denominator"]["batch_id"] == "h100-noisy"
    assert any("raw comparator" in reason for reason in candidate["reasons"])


def test_h100_promotion_readiness_accepts_stable_raw_and_records_thresholds():
    matrix = load_matrix_module()
    repeat_summary = {
        "groups": {
            "pytorch_raw:float32:mmap": _repeat_group(
                baseline="pytorch_raw",
                cv_percent=1.09,
                classification="ok",
            ),
            "medkit_native_prefetch_pinned:float32:stream": _repeat_group(
                baseline="medkit_native_prefetch_pinned",
                cv_percent=0.56,
                classification="ok",
            ),
        }
    }

    readiness = matrix.promotion_readiness_report(
        repeat_summary,
        batch_id="h100-stable",
        modal_gpu="H100",
    )

    assert readiness["status"] == "eligible"
    assert readiness["thresholds"]["H100"]["train_samples_per_second_cv_ok_percent"] == 3.0
    assert readiness["thresholds"]["H100"]["train_samples_per_second_cv_warn_percent"] == 5.0
    assert readiness["thresholds"]["L4"]["train_samples_per_second_cv_ok_percent"] == 3.0
    denominator = readiness["candidates"][
        "medkit_native_prefetch_pinned:float32:stream"
    ]["speedup_denominator"]
    assert denominator == {
        "batch_id": "h100-stable",
        "group_key": "pytorch_raw:float32:mmap",
        "source": "same_batch",
        "stability": repeat_summary["groups"]["pytorch_raw:float32:mmap"]["stability"],
    }


def test_h100_medkit_only_requires_external_comparator():
    matrix = load_matrix_module()
    medkit_only = {
        "groups": {
            "medkit_native_prefetch_pinned:float32:stream": _repeat_group(
                baseline="medkit_native_prefetch_pinned",
                cv_percent=0.56,
                classification="ok",
            ),
        }
    }
    rejected = matrix.promotion_readiness_report(
        medkit_only,
        batch_id="h100-medkit-only",
        modal_gpu="H100",
    )
    assert rejected["status"] == "rejected"

    comparator = {
        "groups": {
            "pytorch_raw:float32:mmap": _repeat_group(
                baseline="pytorch_raw",
                cv_percent=1.09,
                classification="ok",
            ),
        }
    }
    accepted = matrix.promotion_readiness_report(
        medkit_only,
        batch_id="h100-medkit-only",
        modal_gpu="H100",
        comparator_summary=comparator,
        comparator_batch_id="h100-rawcontrol",
    )
    denominator = accepted["candidates"][
        "medkit_native_prefetch_pinned:float32:stream"
    ]["speedup_denominator"]
    assert accepted["status"] == "eligible"
    assert denominator["source"] == "external_comparator_batch"
    assert denominator["batch_id"] == "h100-rawcontrol"


def test_h100_raw_only_batch_is_comparator_ready():
    matrix = load_matrix_module()
    repeat_summary = {
        "groups": {
            "pytorch_raw:float32:mmap": _repeat_group(
                baseline="pytorch_raw",
                cv_percent=1.09,
                classification="ok",
            ),
        }
    }

    readiness = matrix.promotion_readiness_report(
        repeat_summary,
        batch_id="h100-rawcontrol",
        modal_gpu="H100",
    )

    assert readiness["status"] == "comparator_ready"
    assert readiness["raw_comparators"]["pytorch_raw:float32:mmap"]["status"] == "eligible"


def _prediction_rows(
    baseline: str,
    *,
    sample_ids: list[str] | None = None,
    target_names: list[str] | None = None,
) -> list[dict[str, object]]:
    sample_ids = sample_ids or ["a", "b"]
    target_names = target_names or ["A", "B"]
    base_rows = [
        {
            "labels": [0.0, 1.0],
            "label_mask": [1.0, 1.0],
            "logits": [-2.0, 2.0],
            "probabilities": [0.1, 0.9],
        },
        {
            "labels": [1.0, 0.0],
            "label_mask": [1.0, 1.0],
            "logits": [2.0, -2.0],
            "probabilities": [0.9, 0.1],
        },
    ]
    rows = []
    for index, sample_id in enumerate(sample_ids):
        values = base_rows[index % len(base_rows)]
        rows.append(
            {
                "schema_version": 1,
                "baseline": baseline,
                "eval_index": index,
                "sample_id": sample_id,
                "patient_id": f"patient-{sample_id}",
                "study_id": f"study-{sample_id}",
                "image_id": f"image-{sample_id}",
                "source_path": f"/images/{sample_id}.png",
                "sample_hash": f"sha-{sample_id}",
                "target_names": target_names,
                "labels": values["labels"],
                "label_mask": values["label_mask"],
                "logits": values["logits"],
                "probabilities": values["probabilities"],
                "thresholds": [0.5 for _target in target_names],
                "predictions": [0, 1] if index % len(base_rows) == 0 else [1, 0],
            }
        )
    return rows


def _prediction_summary(
    benchmark,
    baseline: str,
    *,
    sample_ids: list[str] | None = None,
    target_names: list[str] | None = None,
) -> dict[str, object]:
    target_names = target_names or ["A", "B"]
    rows = _prediction_rows(baseline, sample_ids=sample_ids, target_names=target_names)
    metrics = {
        "samples": len(rows),
        "macro_auroc": 1.0,
        "macro_auprc": 1.0,
        "metric_target_count": len(target_names),
        "targets": {
            target: {
                "auroc": 1.0,
                "auprc": 1.0,
                "valid_samples": len(rows),
                "positives": 1,
                "negatives": max(len(rows) - 1, 0),
            }
            for target in target_names
        },
    }
    return {
        "status": "ok",
        "enabled": True,
        "baseline": baseline,
        "artifact_path": f"eval-predictions-{baseline}.jsonl.gz",
        "artifact_sha256": "sha256",
        "samples": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "target_names": target_names,
        "hashes": benchmark.eval_prediction_hashes(rows, target_names),
        "metric_recompute": metrics,
        "metric_recompute_matches_quality": True,
        "metric_recompute_matches_artifact": True,
    }


def _repeat_group(*, baseline: str, cv_percent: float, classification: str) -> dict[str, object]:
    return {
        "baseline": baseline,
        "status": "ok",
        "metrics": {
            "train_samples_per_second": {
                "count": 3,
                "mean": 100.0,
                "cv_percent": cv_percent,
            }
        },
        "stability": {
            "metric": "train_samples_per_second",
            "classification": classification,
            "cv_percent": cv_percent,
            "count": 3,
            "thresholds": {
                "train_samples_per_second_cv_ok_percent": 3.0,
                "train_samples_per_second_cv_warn_percent": 5.0,
            },
        },
    }


def _memory_report() -> dict[str, object]:
    return {
        "psutil_pss_mb": 100.0,
        "psutil_uss_mb": 90.0,
        "smaps_pss_mb": 100.0,
        "smaps_uss_mb": 90.0,
        "smaps_pss_file_mb": 10.0,
        "smaps_pss_anon_mb": 80.0,
        "smaps_pss_cache_images_mb": 0.0,
        "sources": ["resource.getrusage", "psutil.Process.memory_full_info", "/proc/self/smaps"],
    }
