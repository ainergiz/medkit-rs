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

    report = benchmark.quality_gate_report(
        quality={
            "pytorch_raw": {
                "status": "ok",
                "samples": 16,
                "metric_target_count": 0,
                "macro_auroc": None,
                "macro_auprc": None,
            }
        },
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
        "sync_every_step": True,
        "loss_pos_weight": "none",
        "quality_gate": False,
        "quality_min_eval_samples": 0,
        "quality_min_metric_targets": 0,
        "quality_min_macro_auroc": 0.0,
        "quality_min_macro_auprc": 0.0,
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
        "quality_gate": {"status": "recorded", "enabled": False, "errors": []},
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
        assert "--sync-every-step" in command

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
    assert quality_args.quality_gate is True
    assert quality_args.quality_min_eval_samples == 900
    assert quality_args.quality_min_metric_targets == 3
    quality_command = matrix.build_command(
        quality_args,
        run_id=matrix.run_id_for("quality-l4", quality_rows[0]),
        row=quality_rows[0],
    )
    assert "--quality-gate" in quality_command
    assert "--loss-pos-weight" in quality_command
    assert "balanced" in quality_command


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

        def report_metadata(self):
            return {
                "worker_mode": "fake_actual_dataset_report",
                "read_mode": self.kwargs["read_mode"],
                "include_metadata": self.kwargs["include_metadata"],
                "shuffle_block_batches": self.kwargs["shuffle_block_batches"],
                "prefetch_depth": self.kwargs["prefetch_depth"],
                "prefetch_read_workers": self.kwargs["read_workers"],
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
    assert loader.report_metadata()["shuffle_block_batches"] == 8
    assert loader.report_metadata()["worker_mode"] == "fake_actual_dataset_report"
    assert loader.report_metadata()["prefetch_read_workers"] == 4


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
            "sync_every_step": True,
            "loss_pos_weight": "none",
            "quality_gate": False,
            "quality_min_eval_samples": 0,
            "quality_min_metric_targets": 0,
            "quality_min_macro_auroc": 0.0,
            "quality_min_macro_auprc": 0.0,
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
            "sync_every_step": True,
            "loss_pos_weight": "none",
            "quality_gate": False,
            "quality_min_eval_samples": 0,
            "quality_min_metric_targets": 0,
            "quality_min_macro_auroc": 0.0,
            "quality_min_macro_auprc": 0.0,
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
                            "profile_end_to_end_samples_per_s": samples_per_second - 1.0
                        }
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
    assert group["metrics"]["profile_end_to_end_samples_per_second"]["count"] == 3
    assert comparison["train_samples_per_second_speedup"] == 2.0


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
