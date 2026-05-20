from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import types
from pathlib import Path


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
            "forward_ms": 2.0,
            "backward_ms": 3.0,
            "optimizer_ms": 0.75,
            "total_step_ms": 10.0,
        },
        {
            "samples": 32,
            "data_wait_ms": 2.0,
            "h2d_ms": 0.75,
            "forward_ms": 3.0,
            "backward_ms": 4.0,
            "optimizer_ms": 1.0,
            "total_step_ms": 15.0,
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
        "profile": {"pytorch_raw": profile_summary},
        "memory": benchmark.memory_summary(reports),
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
    assert h100_run_ids == [
        "repeat-h100-pytorch-raw-float32-mmap",
        "repeat-h100-medkit-native-prefetch-pinned-float32-stream",
        "repeat-h100-medkit-native-prefetch-pinned-uint8-stream",
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
        ]
    )
    l4_rows = matrix.rows_for_args(l4_args)

    assert l4_args.modal_gpu == "L4"
    assert l4_args.image_size == 224
    assert l4_args.batch_size == 64
    assert l4_args.profile_batches == 64
    assert l4_args.shuffle_block_batches == 8
    assert l4_args.gpu_prefetch_batches == 2
    assert [matrix.run_id_for("repeat-l4", row) for row in l4_rows] == [
        "repeat-l4-pytorch-raw-float32-mmap",
        "repeat-l4-medkit-native-prefetch-pinned-float32-stream",
    ]


def test_matrix_modal_cli_command_can_be_overridden(monkeypatch):
    matrix = load_matrix_module()
    monkeypatch.setenv("MEDKIT_MODAL_CLI", "uvx --python 3.11 modal")
    args = matrix.parse_args(["--gate", "l4-224-b64", "--batch-id", "repeat-l4"])
    row = matrix.rows_for_args(args)[0]

    command = matrix.build_command(args, run_id=matrix.run_id_for("repeat-l4", row), row=row)

    assert command[:5] == ["uvx", "--python", "3.11", "modal", "run"]


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
        "memory": {},
        **profile_summary,
    }
    loader_row = {"status": "ok", "samples_per_second": 111.111, "memory": {}}
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
        environment=environment,
        summary_consistency=bad_consistency,
    )

    assert any("summary-consistency status" in error for error in errors)
