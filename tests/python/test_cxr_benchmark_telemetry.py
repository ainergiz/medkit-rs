from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = (
    REPO_ROOT
    / "crates"
    / "medkit-benchmarks"
    / "scripts"
    / "cxr_classification_benchmark.py"
)


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("cxr_classification_benchmark", BENCHMARK_PATH)
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


def test_profile_report_disabled_without_requested_batches():
    benchmark = load_benchmark_module()

    report = benchmark.profile_report_for_baseline(
        baseline="pytorch_raw",
        requested_batches=0,
        records=[],
    )

    assert report["status"] == "disabled"
    assert report["summary"] == {}
