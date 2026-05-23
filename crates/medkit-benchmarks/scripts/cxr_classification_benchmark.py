"""End-to-end CXR classification data-pipeline benchmark.

This harness intentionally keeps the model ordinary and spends its complexity
budget on the data contract:

* materialize a real CXR image subset into a manifest;
* create patient-safe splits from filename-derived NIH patient ids;
* build a deterministic float32 cache;
* compare raw PyTorch, MONAI, and medkit-style cached tensor loaders;
* run the same training/evaluation loop for each pipeline;
* emit a report directory that can be inspected or summarized outside git.

The current implementation is a Python benchmark harness. It is not the final
Rust CLI surface. The report records that limitation explicitly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATASET = "arudaev/chest-xray-14-320"
RSNA_PNEUMONIA_DATASET = "rsna-pneumonia-2018"
DATASET_FALLBACKS = ("HlexNC/chest-xray-14-320",)
DEFAULT_TARGETS = (
    "Pneumonia",
    "Consolidation",
    "Edema",
    "Atelectasis",
    "Effusion",
    "Cardiomegaly",
    "No Finding",
)
ALL_NIH_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
)
SMAPS_HEADER_RE = re.compile(r"^[0-9a-fA-F]+-[0-9a-fA-F]+\s")
H2D_TIMING_DIRECT_COPY = "direct_copy_completion_elapsed"
H2D_TIMING_CUDA_PREFETCH_STREAM = "cuda_prefetch_stream_elapsed"
RSNA_CALCULATED_LUNG_OPACITY_LABEL = "L_v8n"
RSNA_CALCULATED_NORMAL_LABEL = "L_o8w"
RSNA_CALCULATED_NO_OPACITY_LABEL = "L_yd0"


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    patient_id: str
    study_id: str
    image_id: str
    image_path: str
    filename: str
    source_split: str
    width: int
    height: int
    labels: dict[str, int | None]
    split: str = ""
    sha256: str = ""
    source_format: str = "png"
    modality: str = "CR"
    view_position: str = "unknown"
    label_source: str = "nih_chestxray14_nlp_labels"
    localization_boxes: list[dict[str, Any]] | None = None


@dataclass
class DevicePrefetchBatch:
    image: Any
    labels: Any
    mask: Any
    samples: int
    tensor_bytes: int
    h2d_bytes: int
    h2d_start: Any
    h2d_end: Any
    ready_event: Any
    data_wait_seconds: float
    slot_index: int | None = None
    sample_ids: list[str] | None = None
    split: str = ""
    sha256: str = ""


@dataclass
class EvaluationOutputs:
    y_true: Any
    y_score: Any
    y_mask: Any
    y_logits: Any
    samples: list[dict[str, Any]]
    localization: dict[str, Any] | None = None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real CXR classification benchmark and emit report artifacts."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--rsna-root",
        type=Path,
        default=Path("/cache/cxr/datasets/rsna-pneumonia-2018"),
        help="Root containing RSNA Pneumonia raw/ and extracted/ directories.",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("data/cxr-benchmark"))
    parser.add_argument("--report-dir", type=Path, default=Path("target/reports/cxr"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--cache-image-size", type=int, default=0)
    parser.add_argument("--cache-dtype", choices=("float32", "float16", "uint8"), default="float32")
    parser.add_argument(
        "--cache-build-workers",
        type=int,
        default=1,
        help="Threads used by the Python cache builder. 1 preserves the historical serial path.",
    )
    parser.add_argument(
        "--cache-key-mode",
        choices=("legacy", "content"),
        default="legacy",
        help=(
            "legacy uses cache-{size}-{dtype}; content includes manifest, transform, "
            "and target fingerprints so subset/full caches cannot overwrite each other."
        ),
    )
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--max-train", type=int, default=4096)
    parser.add_argument("--max-val", type=int, default=1024)
    parser.add_argument("--max-test", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=0)
    parser.add_argument(
        "--drop-last-train",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip incomplete training batches so every model step uses the requested batch size.",
    )
    parser.add_argument("--loader-batches", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=0,
        help="Record per-step timing for this many post-warmup training batches.",
    )
    parser.add_argument("--prefetch-depth", type=int, default=1)
    parser.add_argument("--prefetch-read-workers", type=int, default=1)
    parser.add_argument(
        "--shuffle-block-batches",
        type=int,
        default=0,
        help=(
            "For medkit native CXR loaders, shuffle blocks of this many batches "
            "instead of individual samples. 0 keeps full random shuffle."
        ),
    )
    parser.add_argument(
        "--gpu-prefetch-batches",
        type=int,
        default=0,
        help=(
            "Opt-in CUDA stream prefetch depth for training batches. 0 uses the "
            "direct per-step H2D copy path."
        ),
    )
    parser.add_argument(
        "--gpu-prefetch-reuse-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Experimental: copy CUDA-prefetched batches into a fixed ring of GPU "
            "buffers instead of allocating new device tensors each batch."
        ),
    )
    parser.add_argument(
        "--sync-every-step",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Synchronize CUDA after every train step for deterministic step timing. "
            "Disable to measure a realistic asynchronous training loop; profiling "
            "still forces synchronization."
        ),
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use PyTorch channels-last memory format for the model and image tensors.",
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the PyTorch model before warmup/training and record compile provenance.",
    )
    parser.add_argument("--torch-compile-mode", default="default")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.0e-4,
        help="AdamW learning rate used for training.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "float16", "bfloat16", "disabled"),
        default="auto",
        help=(
            "CUDA autocast dtype policy. auto preserves PyTorch's default "
            "CUDA autocast dtype, disabled turns autocast and GradScaler off."
        ),
    )
    parser.add_argument(
        "--model-init",
        choices=("random", "imagenet"),
        default="random",
        help="DenseNet121 initialization policy. imagenet adapts RGB conv0 weights to grayscale.",
    )
    parser.add_argument("--read-mode", choices=("mmap", "stream"), default="mmap")
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--baselines",
        default="pytorch_raw,monai_raw,medkit_cached_mmap,medkit_pinned_prefetch",
    )
    parser.add_argument("--uncertain", default="ignore")
    parser.add_argument(
        "--loss-pos-weight",
        choices=("none", "balanced"),
        default="none",
        help="Use train-split positive class weights in BCE when set to balanced.",
    )
    parser.add_argument(
        "--loss-pos-weight-cap",
        type=float,
        default=0.0,
        help="Optional cap for balanced positive class weights. 0 disables capping.",
    )
    parser.add_argument("--loss-kind", choices=("bce", "focal"), default="bce")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.0,
        help="Optional focal alpha in (0, 1). 0 disables alpha weighting.",
    )
    parser.add_argument(
        "--quality-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail the run if quality coverage/metric requirements are not met.",
    )
    parser.add_argument("--quality-min-eval-samples", type=int, default=0)
    parser.add_argument("--quality-min-metric-targets", type=int, default=0)
    parser.add_argument("--quality-min-macro-auroc", type=float, default=0.0)
    parser.add_argument("--quality-min-macro-auprc", type=float, default=0.0)
    parser.add_argument(
        "--eval-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Emit eval-predictions JSONL artifacts. Defaults to enabled for "
            "quality-gated runs and disabled otherwise."
        ),
    )
    parser.add_argument(
        "--train-order-evidence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Emit train-order JSONL artifacts with per-batch sample ids and "
            "label composition. Defaults to enabled for quality-gated runs."
        ),
    )
    parser.add_argument(
        "--paired-train-order",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use one deterministic warmup/epoch train batch schedule for every "
            "baseline. Defaults to enabled for quality-gated runs."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--allow-destructive-cache",
        action="store_true",
        help=(
            "Allow --force-cache to remove an existing non-smoke cache. Without this, "
            "--force-cache may only build missing caches or smoke caches."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Materialize, split, cache, and emit reproducibility artifacts without training.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = min(args.max_samples, 512)
        args.max_train = min(args.max_train, 320)
        args.max_val = min(args.max_val, 96)
        args.max_test = min(args.max_test, 96)
        args.loader_batches = min(args.loader_batches, 8)
        args.max_train_batches = args.max_train_batches or 8
        args.max_eval_batches = args.max_eval_batches or 4
        args.epochs = min(args.epochs, 1)
    if args.force_rematerialize:
        args.force_cache = True
    if args.cache_build_workers < 1:
        raise ValueError("--cache-build-workers must be >= 1")
    if args.loss_pos_weight_cap < 0.0:
        raise ValueError("--loss-pos-weight-cap must be >= 0")
    if args.focal_gamma < 0.0:
        raise ValueError("--focal-gamma must be >= 0")
    if args.focal_alpha < 0.0 or args.focal_alpha >= 1.0:
        raise ValueError("--focal-alpha must be >= 0 and < 1")
    if args.eval_predictions is None:
        args.eval_predictions = bool(args.quality_gate)
    if args.paired_train_order is None:
        args.paired_train_order = bool(args.quality_gate)
    if args.paired_train_order and args.train_order_evidence is None:
        args.train_order_evidence = True
    if args.train_order_evidence is None:
        args.train_order_evidence = bool(args.quality_gate)
    if args.train_order_evidence:
        args.include_metadata = True

    started = time.strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or (
        f"nih-cxr14-320-size{args.image_size}-n{args.max_samples}-"
        f"batch{args.batch_size}-{started}"
    )
    report_dir = args.report_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    commands_path = report_dir / "commands.txt"
    commands_path.write_text(" ".join(shell_quote(part) for part in sys.argv) + "\n")

    random.seed(args.seed)
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    missing_targets = [target for target in targets if target not in ALL_NIH_LABELS]
    if missing_targets:
        raise ValueError(f"Targets are not available in NIH ChestX-ray14: {missing_targets}")
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]

    cache_size = args.cache_image_size or args.image_size
    dataset_work_dir = (
        args.work_dir / safe_artifact_name(args.dataset)
        if args.dataset == RSNA_PNEUMONIA_DATASET
        else args.work_dir
    )
    data_dir = dataset_work_dir / "materialized"
    manifest_path = dataset_work_dir / "manifest.jsonl"
    split_path = dataset_work_dir / "splits.json"
    webdataset_dir = dataset_work_dir / f"webdataset-{cache_size}"
    dataset_metadata = dataset_metadata_for_args(args)

    run_metadata = {
        "run_id": run_id,
        "dataset_requested": args.dataset,
        "manifest_requested": str(args.manifest) if args.manifest else None,
        "splits_requested": str(args.splits) if args.splits else None,
        "plan_requested": str(args.plan) if args.plan else None,
        "dataset_kind": dataset_metadata["dataset_kind"],
        "primary_plan_dataset": dataset_metadata["primary_plan_dataset"],
        "dataset_deviation": dataset_metadata["dataset_deviation"],
        "dataset_root": str(args.rsna_root) if args.dataset == RSNA_PNEUMONIA_DATASET else None,
        "dataset_work_dir": str(dataset_work_dir),
        "targets": targets,
        "uncertain_policy": args.uncertain,
        "missing_policy": "mask_missing",
        "model_init": args.model_init,
        "loss_kind": args.loss_kind,
        "loss_pos_weight": args.loss_pos_weight,
        "loss_pos_weight_cap": args.loss_pos_weight_cap,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "quality_gate": args.quality_gate,
        "quality_min_eval_samples": args.quality_min_eval_samples,
        "quality_min_metric_targets": args.quality_min_metric_targets,
        "quality_min_macro_auroc": args.quality_min_macro_auroc,
        "quality_min_macro_auprc": args.quality_min_macro_auprc,
        "eval_predictions": args.eval_predictions,
        "train_order_evidence": args.train_order_evidence,
        "paired_train_order": args.paired_train_order,
        "prepare_only": args.prepare_only,
        "image_size": args.image_size,
        "cache_image_size": cache_size,
        "cache_dtype": args.cache_dtype,
        "cache_build_workers": args.cache_build_workers,
        "cache_key_mode": args.cache_key_mode,
        "allow_destructive_cache": args.allow_destructive_cache,
        "batch_size": args.batch_size,
        "drop_last_train": args.drop_last_train,
        "workers": args.workers,
        "max_samples": args.max_samples,
        "max_train": args.max_train,
        "max_val": args.max_val,
        "max_test": args.max_test,
        "loader_batches": args.loader_batches,
        "warmup_batches": args.warmup_batches,
        "max_train_batches": args.max_train_batches,
        "max_eval_batches": args.max_eval_batches,
        "prefetch_depth": args.prefetch_depth,
        "prefetch_read_workers": args.prefetch_read_workers,
        "shuffle_block_batches": args.shuffle_block_batches,
        "gpu_prefetch_batches": args.gpu_prefetch_batches,
        "gpu_prefetch_reuse_buffers": args.gpu_prefetch_reuse_buffers,
        "sync_every_step": args.sync_every_step,
        "channels_last": args.channels_last,
        "torch_compile": args.torch_compile,
        "torch_compile_mode": args.torch_compile_mode,
        "learning_rate": args.learning_rate,
        "amp_dtype": args.amp_dtype,
        "profile_batches": args.profile_batches,
        "read_mode": args.read_mode,
        "include_metadata": args.include_metadata,
        "epochs": args.epochs,
        "baselines": baselines,
        "seed": args.seed,
    }

    materialize_start = time.perf_counter()
    records: list[SampleRecord] | None = None
    if args.manifest:
        records = load_manifest_if_compatible(
            args.manifest,
            requested_samples=args.max_samples,
        )
        if records is None:
            raise ValueError(
                f"Requested manifest {args.manifest} is not compatible with "
                f"--max-samples={args.max_samples}"
            )
        dataset_name = args.dataset
    elif not args.force_rematerialize and manifest_path.exists():
        records = load_manifest_if_compatible(
            manifest_path,
            requested_samples=args.max_samples,
        )
    if records is None:
        if args.dataset == RSNA_PNEUMONIA_DATASET:
            records, dataset_name = materialize_rsna_pneumonia(
                root=args.rsna_root,
                manifest_path=manifest_path,
                max_samples=args.max_samples,
                seed=args.seed,
                targets=targets,
            )
        else:
            records, dataset_name = materialize_hf_subset(
                dataset_name=args.dataset,
                fallback_names=DATASET_FALLBACKS,
                out_dir=data_dir,
                manifest_path=manifest_path,
                max_samples=args.max_samples,
                seed=args.seed,
                targets=targets,
            )
    else:
        dataset_name = args.dataset
    materialize_seconds = time.perf_counter() - materialize_start
    run_metadata["dataset_loaded"] = dataset_name
    run_metadata["materialize_seconds"] = materialize_seconds

    split_start = time.perf_counter()
    if args.splits:
        records = apply_requested_splits(records, args.splits)
    else:
        records = assign_patient_safe_splits(
            records,
            seed=args.seed,
            max_train=args.max_train,
            max_val=args.max_val,
            max_test=args.max_test,
        )
    write_manifest(manifest_path, records)
    shutil.copyfile(manifest_path, report_dir / "manifest.jsonl")
    split_report = write_split_file(split_path, records)
    split_seconds = time.perf_counter() - split_start
    write_json(report_dir / "splits.json", split_report)

    manifest_summary = build_manifest_summary(records, targets, run_metadata)
    manifest_summary["manifest_build_seconds"] = materialize_seconds
    manifest_summary["split_build_seconds"] = split_seconds
    write_json(report_dir / "manifest-summary.json", manifest_summary)

    validation = validate_records(records, targets)
    enforce_dataset_validation(validation)
    write_validation(report_dir / "validation.md", validation, run_metadata)
    write_json(report_dir / "split-audit.json", split_report | validation["split_audit"])
    label_balance = label_balance_report(records, targets)
    write_json(report_dir / "label-balance.json", label_balance)
    localization = localization_report(records, targets)
    write_json(report_dir / "localization-report.json", localization)
    train_batch_schedule = (
        build_train_batch_schedule(
            train_records=[record for record in records if record.split == "train"],
            batch_size=args.batch_size,
            seed=args.seed,
            epochs=args.epochs,
            warmup_batches=args.warmup_batches,
            drop_last_train=args.drop_last_train,
            shuffle_block_batches=args.shuffle_block_batches,
        )
        if args.paired_train_order
        else None
    )
    train_schedule_report = train_batch_schedule_report(train_batch_schedule)
    write_json(report_dir / "train-schedule-summary.json", train_schedule_report)

    cache_dir = cache_dir_for_run(
        dataset_work_dir=dataset_work_dir,
        image_size=cache_size,
        cache_dtype=args.cache_dtype,
        cache_key_mode=args.cache_key_mode,
        records=records,
        targets=targets,
    )
    run_metadata["cache_dir"] = str(cache_dir)
    run_metadata["cache_identity"] = cache_identity_report(
        records=records,
        targets=targets,
        image_size=cache_size,
        cache_dtype=args.cache_dtype,
    )
    cache_start = time.perf_counter()
    cache_metadata_path = cache_dir / "cache-metadata.json"
    enforce_destructive_cache_guard(
        force_cache=args.force_cache,
        allow_destructive_cache=args.allow_destructive_cache,
        smoke=args.smoke,
        cache_dir=cache_dir,
        cache_metadata_path=cache_metadata_path,
    )
    rebuild_cache = args.force_cache or not cache_metadata_path.exists()
    if not rebuild_cache:
        existing_cache = load_json(cache_metadata_path)
        rebuild_cache = not cache_matches_run(
            existing_cache,
            records=records,
            targets=targets,
            image_size=cache_size,
            cache_dtype=args.cache_dtype,
        )
    if rebuild_cache:
        cache_report = build_cache(
            records=records,
            targets=targets,
            cache_dir=cache_dir,
            image_size=cache_size,
            cache_dtype=args.cache_dtype,
            cache_build_workers=args.cache_build_workers,
            cache_key_mode=args.cache_key_mode,
            seed=args.seed,
        )
    else:
        cache_report = load_json(cache_metadata_path)
        cache_report["cache_reused"] = True
    cache_report["cache_stage_seconds"] = time.perf_counter() - cache_start
    write_json(report_dir / "cache-report.json", cache_report)

    if "webdataset" in baselines:
        webdataset_start = time.perf_counter()
        webdataset_metadata_path = webdataset_dir / "webdataset-metadata.json"
        if args.force_cache or not webdataset_metadata_path.exists():
            webdataset_report = build_webdataset_shards(
                records=records,
                targets=targets,
                shard_dir=webdataset_dir,
                shard_size=512,
                seed=args.seed,
            )
        else:
            webdataset_report = load_json(webdataset_metadata_path)
            webdataset_report["cache_reused"] = True
        webdataset_report["cache_stage_seconds"] = time.perf_counter() - webdataset_start
        write_json(report_dir / "webdataset-report.json", webdataset_report)

    env_report = environment_report(run_metadata)
    write_json(report_dir / "environment.json", env_report)
    if args.prepare_only:
        provenance = build_run_provenance(
            args=args,
            run_id=run_id,
            run_metadata=run_metadata,
            manifest_summary=manifest_summary,
            split_report=split_report,
            cache_report=cache_report,
            environment=env_report,
        )
        summary = {
            "run_id": run_id,
            "report_dir": str(report_dir),
            "dataset_loaded": dataset_name,
            "samples": manifest_summary["samples"],
            "targets": targets,
            "prepare_only": True,
            "manifest": "manifest.jsonl",
            "splits": "splits.json",
            "localization": localization,
            "cache_report": cache_report,
            "provenance": provenance,
        }
        write_json(report_dir / "run-summary.json", summary)
        write_json(
            report_dir / "summary-consistency.json",
            {"status": "ok", "errors": [], "warnings": []},
        )
        if args.out:
            write_json(args.out, summary)
        print(json.dumps(summary, indent=2))
        return 0

    torch = import_torch()
    device = choose_device(torch, args.device)
    reports = run_all_baselines(
        args=args,
        torch=torch,
        records=records,
        targets=targets,
        baselines=baselines,
        cache_dir=cache_dir,
        webdataset_dir=webdataset_dir,
        image_size=args.image_size,
        device=device,
        report_dir=report_dir,
        capture_eval_predictions=args.eval_predictions,
        capture_train_order=args.train_order_evidence,
        train_batch_schedule=train_batch_schedule,
    )

    write_json(report_dir / "loader-throughput.json", reports["loader"])
    write_json(report_dir / "gpu-throughput.json", reports["gpu"])
    if any(report.get("status") != "disabled" for report in reports["profile"].values()):
        write_json(report_dir / "step-profile.json", reports["profile"])
    write_json(report_dir / "model-quality.json", reports["quality"])
    write_json(report_dir / "threshold-report.json", reports["thresholds"])
    write_json(report_dir / "localization-eval.json", reports["localization_eval"])
    write_json(report_dir / "eval-predictions-summary.json", reports["predictions"])
    write_json(report_dir / "train-order-summary.json", reports["train_order"])
    quality_gate = quality_gate_report(
        quality=reports["quality"],
        train_order=reports["train_order"],
        validation=validation,
        run_metadata=run_metadata,
    )
    write_json(report_dir / "quality-gate.json", quality_gate)
    memory = memory_summary(reports)
    write_json(report_dir / "memory-summary.json", memory)
    ground_truth = training_ground_truth_report(reports)
    write_json(report_dir / "training-ground-truth.json", ground_truth)
    write_json(report_dir / "subgroup-report.json", subgroup_report(records, reports["quality"]))
    provenance = build_run_provenance(
        args=args,
        run_id=run_id,
        run_metadata=run_metadata,
        manifest_summary=manifest_summary,
        split_report=split_report,
        cache_report=cache_report,
        environment=env_report,
    )

    summary = {
        "run_id": run_id,
        "report_dir": str(report_dir),
        "dataset_loaded": dataset_name,
        "samples": manifest_summary["samples"],
        "targets": targets,
        "device": str(device),
        "loader_samples_per_second": {
            name: round(report["samples_per_second"], 3)
            for name, report in reports["loader"].items()
            if "samples_per_second" in report
        },
        "train_samples_per_second": {
            name: round(report["samples_per_second"], 3)
            for name, report in reports["gpu"].items()
            if "samples_per_second" in report
        },
        "quality_macro_auroc": {
            name: round(report.get("macro_auroc", float("nan")), 5)
            for name, report in reports["quality"].items()
            if report.get("status") == "ok"
        },
        "quality_gate": quality_gate,
        "profile": {
            name: report.get("summary")
            for name, report in reports["profile"].items()
            if report.get("status") == "ok"
        },
        "speed_claims": ground_truth.get("speed_claims", {}),
        "memory": memory,
        "ground_truth": ground_truth,
        "predictions": reports["predictions"],
        "train_order": reports["train_order"],
        "localization_eval": reports["localization_eval"],
        "train_schedule": train_schedule_report,
        "localization": localization,
        "provenance": provenance,
    }
    write_json(report_dir / "run-summary.json", summary)
    consistency = validate_run_summary_consistency(
        summary=summary,
        run_metadata=run_metadata,
        manifest_summary=manifest_summary,
        split_report=split_report,
        cache_report=cache_report,
        reports=reports,
        environment=env_report,
    )
    write_json(report_dir / "summary-consistency.json", consistency)
    if consistency["status"] != "ok":
        raise RuntimeError(
            "Run summary/provenance consistency failed: "
            + "; ".join(consistency["errors"])
        )
    if quality_gate["status"] == "failed":
        raise RuntimeError(
            "Quality gate failed: " + "; ".join(quality_gate.get("errors", []))
        )
    if args.out:
        write_json(args.out, summary)
    print(json.dumps(summary, indent=2))
    return 0


def dataset_metadata_for_args(args: argparse.Namespace) -> dict[str, str]:
    if args.dataset == RSNA_PNEUMONIA_DATASET:
        return {
            "dataset_kind": "RSNA Pneumonia Detection Challenge 2018 adjudicated DICOM dataset",
            "primary_plan_dataset": "RSNA Pneumonia Detection Challenge",
            "dataset_deviation": (
                "This run uses the RSNA 2018 adjudicated pneumonia challenge DICOMs "
                "and MD.ai annotations persisted on the Modal volume."
            ),
        }
    return {
        "dataset_kind": "NIH ChestX-ray14 320px Hugging Face parquet subset",
        "primary_plan_dataset": "MIMIC-CXR-JPG",
        "dataset_deviation": (
            "No local credentialed MIMIC-CXR-JPG data was available. This run uses "
            "a public NIH ChestX-ray14 320px dataset so the pipeline can execute "
            "against real CXR images."
        ),
    }


def materialize_hf_subset(
    *,
    dataset_name: str,
    fallback_names: Sequence[str],
    out_dir: Path,
    manifest_path: Path,
    max_samples: int,
    seed: int,
    targets: Sequence[str],
) -> tuple[list[SampleRecord], str]:
    datasets = import_datasets()
    pillow = import_pillow()
    Image = pillow["Image"]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    names = (dataset_name, *fallback_names)
    dataset = None
    loaded_name = ""
    for name in names:
        try:
            dataset = datasets.load_dataset(name, split="train", streaming=True)
            loaded_name = name
            break
        except Exception as error:  # pragma: no cover - depends on network state.
            errors.append(f"{name}: {error}")
    if dataset is None:
        raise RuntimeError("Could not load any CXR dataset candidate:\n" + "\n".join(errors))

    rng = random.Random(seed)
    target_seen = {target: 0 for target in targets}
    records: list[SampleRecord] = []
    # Streaming datasets are ordered by shard. A bounded shuffle keeps the run
    # reproducible while avoiding a full 8 GB materialization before benchmarking.
    iterator = dataset.shuffle(seed=seed, buffer_size=min(10_000, max(max_samples * 4, 1024)))
    for index, row in enumerate(iterator):
        if len(records) >= max_samples:
            break
        image_obj = row.get("image")
        labels_value = str(row.get("labels", ""))
        filename = str(row.get("filename", f"hf-{index:08d}.png"))
        if not filename:
            filename = f"hf-{index:08d}.png"
        labels = parse_labels(labels_value, targets)
        label_set = parse_label_set(labels_value)
        for target in targets:
            if target in label_set:
                target_seen[target] += 1

        patient_id = filename.split("_", 1)[0] or f"patient-{index:08d}"
        study_id = patient_id
        image_id = filename.rsplit(".", 1)[0]
        sample_id = f"{patient_id}/{image_id}"
        image = image_to_pil(image_obj, Image).convert("L")
        width, height = image.size
        # Jitter the nested path slightly so filesystem scans are less toy-like.
        bucket = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:2]
        image_rel = Path("images") / bucket / filename
        image_path = out_dir / image_rel
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
        sha256 = hash_file(image_path)
        records.append(
            SampleRecord(
                sample_id=sample_id,
                patient_id=patient_id,
                study_id=study_id,
                image_id=image_id,
                image_path=str(image_path),
                filename=filename,
                source_split="hf_train_stream",
                width=width,
                height=height,
                labels=labels,
                sha256=sha256,
            )
        )

    rng.shuffle(records)
    if not records:
        raise RuntimeError(f"No records were materialized from {loaded_name}")
    write_manifest(manifest_path, records)
    label_counts_path = out_dir / "materialized-label-counts.json"
    write_json(label_counts_path, target_seen)
    return records, loaded_name


def materialize_rsna_pneumonia(
    *,
    root: Path,
    manifest_path: Path,
    max_samples: int,
    seed: int,
    targets: Sequence[str],
) -> tuple[list[SampleRecord], str]:
    raw_dir = root / "raw"
    extracted_dir = ensure_rsna_pneumonia_extracted(root)
    annotations_path = raw_dir / "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json"
    mappings_path = raw_dir / "pneumonia-challenge-dataset-mappings_2018.json"
    if not annotations_path.exists():
        raise FileNotFoundError(f"RSNA annotations not found: {annotations_path}")
    if not mappings_path.exists():
        raise FileNotFoundError(f"RSNA NIH mapping not found: {mappings_path}")

    annotations = load_json(annotations_path)
    mappings = load_json(mappings_path)
    final_labels_by_sop: dict[str, set[str]] = {}
    lung_opacity_boxes_by_sop: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations.get("datasets", [{}])[0].get("annotations", []):
        label_id = str(annotation.get("labelId") or "")
        if label_id not in {
            RSNA_CALCULATED_LUNG_OPACITY_LABEL,
            RSNA_CALCULATED_NORMAL_LABEL,
            RSNA_CALCULATED_NO_OPACITY_LABEL,
        }:
            continue
        sop_uid = str(annotation.get("SOPInstanceUID") or "")
        if sop_uid:
            final_labels_by_sop.setdefault(sop_uid, set()).add(label_id)
            if label_id == RSNA_CALCULATED_LUNG_OPACITY_LABEL:
                box = rsna_annotation_box(annotation)
                if box is not None:
                    lung_opacity_boxes_by_sop.setdefault(sop_uid, []).append(box)

    dicom_by_sop = {path.stem: path for path in extracted_dir.rglob("*.dcm")}
    records: list[SampleRecord] = []
    skipped: dict[str, int] = {
        "missing_final_label": 0,
        "missing_dicom": 0,
        "unknown_label": 0,
    }
    for row in mappings:
        sop_uid = str(row.get("SOPInstanceUID") or "")
        final_labels = final_labels_by_sop.get(sop_uid)
        if not final_labels:
            skipped["missing_final_label"] += 1
            continue
        dicom_path = dicom_by_sop.get(sop_uid)
        if dicom_path is None:
            skipped["missing_dicom"] += 1
            continue
        pneumonia_value = rsna_pneumonia_label(final_labels)
        if pneumonia_value is None:
            skipped["unknown_label"] += 1
            continue
        img_id = str(row.get("img_id") or f"{sop_uid}.png")
        patient_id = img_id.split("_", 1)[0] or str(row.get("studyId") or sop_uid)
        study_id = str(row.get("StudyInstanceUID") or row.get("studyId") or sop_uid)
        subset_img_id = str(row.get("subset_img_id") or sop_uid)
        orig_labels = {str(label) for label in (row.get("orig_labels") or [])}
        labels = rsna_labels_for_targets(
            targets=targets,
            pneumonia_value=pneumonia_value,
            final_labels=final_labels,
            orig_labels=orig_labels,
        )
        records.append(
            SampleRecord(
                sample_id=f"{patient_id}/{subset_img_id}",
                patient_id=patient_id,
                study_id=study_id,
                image_id=sop_uid,
                image_path=str(dicom_path),
                filename=f"{sop_uid}.dcm",
                source_split=f"rsna_subset_group_{row.get('subset_group', 'unknown')}",
                width=1024,
                height=1024,
                labels=labels,
                sha256=stable_hash({"dataset": RSNA_PNEUMONIA_DATASET, "sop": sop_uid}),
                source_format="dicom",
                modality="CR",
                view_position="unknown",
                label_source="rsna_2018_adjudicated_mdai_calculated_labels",
                localization_boxes=lung_opacity_boxes_by_sop.get(sop_uid, []),
            )
        )

    if not records:
        raise RuntimeError(
            "No RSNA records were materialized; skipped="
            + json.dumps(skipped, sort_keys=True)
        )
    rng = random.Random(seed)
    rng.shuffle(records)
    if max_samples > 0:
        records = records[:max_samples]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, records)
    write_json(
        manifest_path.parent / "rsna-pneumonia-materialize-report.json",
        {
            "dataset": RSNA_PNEUMONIA_DATASET,
            "root": str(root),
            "records": len(records),
            "skipped": skipped,
            "targets": list(targets),
            "final_label_ids": {
                "pneumonia_positive": RSNA_CALCULATED_LUNG_OPACITY_LABEL,
                "normal": RSNA_CALCULATED_NORMAL_LABEL,
                "no_lung_opacity_not_normal": RSNA_CALCULATED_NO_OPACITY_LABEL,
            },
        },
    )
    return records, RSNA_PNEUMONIA_DATASET


def rsna_annotation_box(annotation: dict[str, Any]) -> dict[str, Any] | None:
    data = annotation.get("data")
    if not isinstance(data, dict):
        return None
    try:
        image_width = float(annotation.get("width") or 1024)
        image_height = float(annotation.get("height") or 1024)
        x = float(data.get("x"))
        y = float(data.get("y"))
        width = float(data.get("width"))
        height = float(data.get("height"))
    except (TypeError, ValueError):
        return None
    if image_width <= 0 or image_height <= 0 or width <= 0 or height <= 0:
        return None
    x1 = max(0.0, min(x, image_width))
    y1 = max(0.0, min(y, image_height))
    x2 = max(0.0, min(x + width, image_width))
    y2 = max(0.0, min(y + height, image_height))
    clamped_width = x2 - x1
    clamped_height = y2 - y1
    if clamped_width <= 0 or clamped_height <= 0:
        return None
    area_fraction = (clamped_width * clamped_height) / (image_width * image_height)
    return {
        "label": "lung_opacity",
        "label_id": RSNA_CALCULATED_LUNG_OPACITY_LABEL,
        "x": x1,
        "y": y1,
        "width": clamped_width,
        "height": clamped_height,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "area_fraction": area_fraction,
        "width_fraction": clamped_width / image_width,
        "height_fraction": clamped_height / image_height,
        "center_x_fraction": ((x1 + x2) / 2.0) / image_width,
        "center_y_fraction": ((y1 + y2) / 2.0) / image_height,
    }


def ensure_rsna_pneumonia_extracted(root: Path) -> Path:
    raw_zip = root / "raw" / "pneumonia-challenge-dataset-adjudicated-kaggle_2018.zip"
    extracted_dir = root / "extracted"
    marker_path = extracted_dir / ".rsna-extracted.json"
    if marker_path.exists():
        marker = load_json(marker_path)
        if int(marker.get("dicom_count", 0)) == 30000:
            return extracted_dir
    current_count = sum(1 for _path in extracted_dir.rglob("*.dcm")) if extracted_dir.exists() else 0
    if current_count == 30000:
        write_json(marker_path, {"dicom_count": current_count, "source_zip": str(raw_zip)})
        return extracted_dir
    if not raw_zip.exists():
        raise FileNotFoundError(f"RSNA image zip not found: {raw_zip}")
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip) as archive:
        archive.extractall(extracted_dir)
    dicom_count = sum(1 for _path in extracted_dir.rglob("*.dcm"))
    if dicom_count != 30000:
        raise RuntimeError(f"Expected 30000 RSNA DICOMs after extraction, found {dicom_count}")
    write_json(marker_path, {"dicom_count": dicom_count, "source_zip": str(raw_zip)})
    return extracted_dir


def rsna_pneumonia_label(final_labels: set[str]) -> int | None:
    if RSNA_CALCULATED_LUNG_OPACITY_LABEL in final_labels:
        return 1
    if (
        RSNA_CALCULATED_NORMAL_LABEL in final_labels
        or RSNA_CALCULATED_NO_OPACITY_LABEL in final_labels
    ):
        return 0
    return None


def rsna_labels_for_targets(
    *,
    targets: Sequence[str],
    pneumonia_value: int,
    final_labels: set[str],
    orig_labels: set[str],
) -> dict[str, int | None]:
    labels: dict[str, int | None] = {}
    for target in targets:
        if target == "Pneumonia":
            labels[target] = pneumonia_value
        elif target == "No Finding":
            labels[target] = 1 if RSNA_CALCULATED_NORMAL_LABEL in final_labels else 0
        elif target in ALL_NIH_LABELS:
            labels[target] = 1 if target in orig_labels else 0
        else:
            labels[target] = None
    return labels


def image_to_pil(image_obj: Any, Image: Any) -> Any:
    if hasattr(image_obj, "convert"):
        return image_obj
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            from io import BytesIO

            return Image.open(BytesIO(image_obj["bytes"]))
        if image_obj.get("path"):
            return Image.open(image_obj["path"])
    raise TypeError(f"Unsupported HF image payload: {type(image_obj)!r}")


def parse_label_set(labels_value: str) -> set[str]:
    labels_value = labels_value.strip()
    if not labels_value:
        return set()
    return {part.strip() for part in labels_value.split("|") if part.strip()}


def parse_labels(labels_value: str, targets: Sequence[str]) -> dict[str, int | None]:
    label_set = parse_label_set(labels_value)
    labels: dict[str, int | None] = {}
    for target in targets:
        labels[target] = 1 if target in label_set else 0
    return labels


def assign_patient_safe_splits(
    records: list[SampleRecord],
    *,
    seed: int,
    max_train: int,
    max_val: int,
    max_test: int,
) -> list[SampleRecord]:
    by_patient: dict[str, list[SampleRecord]] = {}
    for record in records:
        by_patient.setdefault(record.patient_id, []).append(record)

    groups = [
        sorted(by_patient[patient_id], key=lambda record: record.sample_id)
        for patient_id in sorted(by_patient)
    ]
    rng = random.Random(seed)
    rng.shuffle(groups)
    caps = {"train": max_train, "val": max_val, "test": max_test}
    assigned_counts = {"train": 0, "val": 0, "test": 0}
    desired_order = ("train", "val", "test")
    assigned: list[SampleRecord] = []
    for group in groups:
        possible = [
            split
            for split in desired_order
            if caps[split] <= 0 or assigned_counts[split] + len(group) <= caps[split]
        ]
        if not possible:
            continue
        # Fill train first, then val/test, but keep the resulting patient groups
        # intact. This gives deterministic, patient-safe bounded subsets.
        split = min(
            possible,
            key=lambda name: assigned_counts[name] / max(caps[name], 1)
            if caps[name] > 0
            else assigned_counts[name],
        )
        for record in group:
            assigned.append(replace_record(record, split=split))
        assigned_counts[split] += len(group)
        if all(caps[name] > 0 and assigned_counts[name] >= caps[name] for name in desired_order):
            break
    return assigned


def apply_requested_splits(
    records: Sequence[SampleRecord], splits_path: Path
) -> list[SampleRecord]:
    raw = json.loads(splits_path.read_text())
    split_lists = raw.get("splits")
    if not isinstance(split_lists, dict):
        raise ValueError(f"Split file {splits_path} must contain a `splits` object")

    split_by_sample: dict[str, str] = {}
    for split in ("train", "val", "test"):
        values = split_lists.get(split, [])
        if not isinstance(values, list):
            raise ValueError(f"Split file {splits_path} has non-list `{split}` entries")
        for sample_id in values:
            sample_id = str(sample_id)
            if sample_id in split_by_sample:
                raise ValueError(f"Sample {sample_id!r} appears in more than one split")
            split_by_sample[sample_id] = split

    record_ids = {record.sample_id for record in records}
    missing = sorted(split_by_sample.keys() - record_ids)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Split file {splits_path} references {len(missing)} unknown samples: {preview}"
        )

    unsplit = sorted(record_ids - split_by_sample.keys())
    if unsplit:
        preview = ", ".join(unsplit[:5])
        raise ValueError(
            f"Split file {splits_path} omits {len(unsplit)} samples: {preview}"
        )

    return [
        replace_record(record, split=split_by_sample[record.sample_id])
        for record in records
    ]


def replace_record(record: SampleRecord, **changes: Any) -> SampleRecord:
    data = record.__dict__.copy()
    data.update(changes)
    return SampleRecord(**data)


def write_split_file(path: Path, records: Sequence[SampleRecord]) -> dict[str, Any]:
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    patients: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for record in records:
        if record.split in splits:
            splits[record.split].append(record.sample_id)
            patients[record.split].add(record.patient_id)
    report = {
        "counts": {name: len(values) for name, values in splits.items()},
        "patient_counts": {name: len(values) for name, values in patients.items()},
        "splits": splits,
        "split_checksum": stable_hash(splits),
        "patient_split_checksum": stable_hash(
            {name: sorted(values) for name, values in patients.items()}
        ),
    }
    write_json(path, report)
    return report


def build_manifest_summary(
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    by_split: dict[str, int] = {}
    view_counts: dict[str, int] = {}
    source_format_counts: dict[str, int] = {}
    target_counts = {
        target: {"positive": 0, "negative": 0, "missing": 0, "uncertain": 0}
        for target in targets
    }
    for record in records:
        by_split[record.split] = by_split.get(record.split, 0) + 1
        view_counts[record.view_position] = view_counts.get(record.view_position, 0) + 1
        source_format_counts[record.source_format] = (
            source_format_counts.get(record.source_format, 0) + 1
        )
        for target in targets:
            value = record.labels.get(target)
            if value == 1:
                target_counts[target]["positive"] += 1
            elif value == 0:
                target_counts[target]["negative"] += 1
            elif value == -1:
                target_counts[target]["uncertain"] += 1
            else:
                target_counts[target]["missing"] += 1
    return {
        "dataset": run_metadata["dataset_kind"],
        "dataset_loaded": run_metadata.get("dataset_loaded", run_metadata["dataset_requested"]),
        "dataset_deviation": run_metadata["dataset_deviation"],
        "samples": len(records),
        "splits": by_split,
        "patients": len({record.patient_id for record in records}),
        "studies": len({record.study_id for record in records}),
        "images": len(records),
        "targets": list(targets),
        "target_counts": target_counts,
        "view_counts": view_counts,
        "source_format": (
            "mixed" if len(source_format_counts) > 1 else next(iter(source_format_counts), "unknown")
        ),
        "source_format_counts": source_format_counts,
        "metadata_limitations": [
            "Dataset-specific metadata is normalized into the benchmark SampleRecord schema.",
            "View position may be unknown when the source dataset does not expose it.",
            "Uncertain labels are masked only when the adapter emits None/-1 labels.",
        ],
    }


def validate_records(records: Sequence[SampleRecord], targets: Sequence[str]) -> dict[str, Any]:
    unreadable: list[str] = []
    missing_labels: list[str] = []
    for record in records:
        if not Path(record.image_path).exists():
            unreadable.append(record.sample_id)
        for target in targets:
            if target not in record.labels:
                missing_labels.append(f"{record.sample_id}:{target}")
    patient_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    hash_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    study_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for record in records:
        patient_by_split.setdefault(record.split, set()).add(record.patient_id)
        study_by_split.setdefault(record.split, set()).add(record.study_id)
        hash_by_split.setdefault(record.split, set()).add(record.sha256)

    patient_overlap = overlap_report(patient_by_split)
    study_overlap = overlap_report(study_by_split)
    duplicate_hash_overlap = overlap_report(hash_by_split)
    return {
        "readable_images": len(records) - len(unreadable),
        "unreadable_images": unreadable,
        "missing_labels": missing_labels,
        "split_audit": {
            "patient_overlap": patient_overlap,
            "study_overlap": study_overlap,
            "duplicate_hash_overlap": duplicate_hash_overlap,
            "patient_overlap_count": sum(len(values) for values in patient_overlap.values()),
            "study_overlap_count": sum(len(values) for values in study_overlap.values()),
            "duplicate_hash_overlap_count": sum(
                len(values) for values in duplicate_hash_overlap.values()
            ),
        },
    }


def enforce_dataset_validation(validation: dict[str, Any]) -> None:
    errors: list[str] = []
    if validation.get("unreadable_images"):
        errors.append(f"unreadable images: {len(validation['unreadable_images'])}")
    if validation.get("missing_labels"):
        errors.append(f"missing labels: {len(validation['missing_labels'])}")
    split_audit = validation.get("split_audit") or {}
    for field in (
        "patient_overlap_count",
        "study_overlap_count",
        "duplicate_hash_overlap_count",
    ):
        count = int(split_audit.get(field) or 0)
        if count:
            errors.append(f"{field}: {count}")
    if errors:
        raise RuntimeError("CXR validation failed: " + "; ".join(errors))


def label_balance_report(
    records: Sequence[SampleRecord],
    targets: Sequence[str],
) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        by_split[split] = {
            "samples": len(split_records),
            "targets": label_counts_for_records(split_records, targets),
        }
    return {
        "status": "ok",
        "targets": list(targets),
        "splits": by_split,
    }


def localization_report(
    records: Sequence[SampleRecord],
    targets: Sequence[str],
) -> dict[str, Any]:
    target = "Pneumonia" if "Pneumonia" in targets else targets[0] if targets else ""
    overall = localization_split_report(records, target)
    by_split = {
        split: localization_split_report(
            [record for record in records if record.split == split],
            target,
        )
        for split in ("train", "val", "test")
    }
    status = "ok" if overall["total_boxes"] > 0 else "not_available"
    return {
        "schema_version": 1,
        "status": status,
        "target": target,
        "box_source": "rsna_2018_adjudicated_mdai_lung_opacity_boxes",
        "box_label_id": RSNA_CALCULATED_LUNG_OPACITY_LABEL,
        "overall": overall,
        "splits": by_split,
        "notes": [
            "These are annotation/readiness metrics, not detector mAP.",
            (
                "Image-level classifier localization needs a CAM/heatmap or detection head "
                "before box-hit metrics are meaningful."
            ),
        ],
    }


def localization_split_report(
    records: Sequence[SampleRecord],
    target: str,
) -> dict[str, Any]:
    positive_records = [record for record in records if record.labels.get(target) == 1]
    negative_records = [record for record in records if record.labels.get(target) == 0]
    boxed_records = [record for record in records if record.localization_boxes]
    positive_boxed_records = [
        record for record in positive_records if record.localization_boxes
    ]
    negative_boxed_records = [
        record for record in negative_records if record.localization_boxes
    ]
    boxes = [box for record in records for box in (record.localization_boxes or [])]
    positive_box_counts = [
        len(record.localization_boxes or []) for record in positive_records
    ]
    multi_box_positive = sum(1 for count in positive_box_counts if count > 1)
    area_values = numeric_box_values(boxes, "area_fraction")
    width_values = numeric_box_values(boxes, "width_fraction")
    height_values = numeric_box_values(boxes, "height_fraction")
    center_x_values = numeric_box_values(boxes, "center_x_fraction")
    center_y_values = numeric_box_values(boxes, "center_y_fraction")
    return {
        "samples": len(records),
        "positive_samples": len(positive_records),
        "negative_samples": len(negative_records),
        "target_prevalence": safe_ratio(
            len(positive_records),
            len(positive_records) + len(negative_records),
        ),
        "samples_with_boxes": len(boxed_records),
        "positive_samples_with_boxes": len(positive_boxed_records),
        "positive_samples_without_boxes": len(positive_records) - len(positive_boxed_records),
        "negative_samples_with_boxes": len(negative_boxed_records),
        "box_positive_coverage": safe_ratio(len(positive_boxed_records), len(positive_records)),
        "total_boxes": len(boxes),
        "boxes_per_positive_sample": safe_ratio(len(boxes), len(positive_records)),
        "boxes_per_positive_boxed_sample": safe_ratio(len(boxes), len(positive_boxed_records)),
        "multi_box_positive_samples": multi_box_positive,
        "multi_box_positive_fraction": safe_ratio(multi_box_positive, len(positive_records)),
        "box_area_fraction": numeric_distribution(area_values),
        "box_width_fraction": numeric_distribution(width_values),
        "box_height_fraction": numeric_distribution(height_values),
        "box_center_x_fraction": numeric_distribution(center_x_values),
        "box_center_y_fraction": numeric_distribution(center_y_values),
        "box_area_bins": box_area_bins(area_values),
    }


def numeric_box_values(boxes: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for box in boxes:
        try:
            value = float(box.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def numeric_distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "p75": percentile(ordered, 0.75),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def percentile(ordered_values: Sequence[float], p: float) -> float | None:
    if not ordered_values:
        return None
    if len(ordered_values) == 1:
        return float(ordered_values[0])
    position = (len(ordered_values) - 1) * p
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered_values[lower])
    weight = position - lower
    return float(ordered_values[lower] * (1.0 - weight) + ordered_values[upper] * weight)


def box_area_bins(area_values: Sequence[float]) -> dict[str, Any]:
    small = sum(1 for value in area_values if value < 0.02)
    medium = sum(1 for value in area_values if 0.02 <= value < 0.10)
    large = sum(1 for value in area_values if value >= 0.10)
    total = len(area_values)
    return {
        "small_lt_2pct": small,
        "medium_2pct_to_10pct": medium,
        "large_gte_10pct": large,
        "small_fraction": safe_ratio(small, total),
        "medium_fraction": safe_ratio(medium, total),
        "large_fraction": safe_ratio(large, total),
    }


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def label_counts_for_records(
    records: Sequence[SampleRecord],
    targets: Sequence[str],
) -> dict[str, dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for target in targets:
        positive = 0
        negative = 0
        missing = 0
        uncertain = 0
        for record in records:
            value = record.labels.get(target)
            if value == 1:
                positive += 1
            elif value == 0:
                negative += 1
            elif value == -1:
                uncertain += 1
            else:
                missing += 1
        valid = positive + negative
        counts[target] = {
            "positive": positive,
            "negative": negative,
            "uncertain": uncertain,
            "missing": missing,
            "valid": valid,
            "prevalence": positive / valid if valid else None,
        }
    return counts


def overlap_report(values_by_split: dict[str, set[str]]) -> dict[str, list[str]]:
    names = sorted(values_by_split)
    overlaps: dict[str, list[str]] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(values_by_split[left] & values_by_split[right])
            if overlap:
                overlaps[f"{left}:{right}"] = overlap[:100]
    return overlaps


def write_validation(path: Path, validation: dict[str, Any], run_metadata: dict[str, Any]) -> None:
    split_audit = validation["split_audit"]
    lines = [
        "# CXR Validation Report",
        "",
        f"Dataset: {run_metadata['dataset_kind']}",
        "",
        "## Deviation",
        "",
        run_metadata["dataset_deviation"],
        "",
        "## Checks",
        "",
        f"- readable images: {validation['readable_images']}",
        f"- unreadable images: {len(validation['unreadable_images'])}",
        f"- missing labels: {len(validation['missing_labels'])}",
        f"- patient overlap count: {split_audit['patient_overlap_count']}",
        f"- study overlap count: {split_audit['study_overlap_count']}",
        f"- duplicate image hash overlap count: {split_audit['duplicate_hash_overlap_count']}",
        "",
        "## Limitations",
        "",
        "- This public NIH export has no view-position metadata.",
        "- This public NIH export has no explicit uncertainty labels.",
        "- Patient ids are derived from NIH filenames.",
    ]
    path.write_text("\n".join(lines) + "\n")


def build_cache(
    *,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    cache_dir: Path,
    image_size: int,
    cache_dtype: str,
    cache_build_workers: int,
    cache_key_mode: str,
    seed: int,
) -> dict[str, Any]:
    numpy = import_numpy()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = cache_transform_plan(image_size=image_size, cache_dtype=cache_dtype)
    transform_hash = stable_hash(transform)

    build_start = time.perf_counter()
    train_records = [record for record in records if record.split == "train"]
    mean_std_start = time.perf_counter()
    mean, std = estimate_mean_std(train_records, image_size)
    mean_std_seconds = time.perf_counter() - mean_std_start
    split_reports: dict[str, Any] = {}
    failed: list[str] = []
    for split in ("train", "val", "test"):
        split_start = time.perf_counter()
        split_records = [record for record in records if record.split == split]
        images_path = cache_dir / f"{split}-images.{cache_dtype}.dat"
        labels_path = cache_dir / f"{split}-labels.float32.dat"
        masks_path = cache_dir / f"{split}-masks.float32.dat"
        metadata_path = cache_dir / f"{split}-metadata.jsonl"
        shape = (len(split_records), 1, image_size, image_size)
        images = numpy.memmap(images_path, dtype=cache_dtype, mode="w+", shape=shape)
        labels = numpy.zeros((len(split_records), len(targets)), dtype="float32")
        masks = numpy.zeros((len(split_records), len(targets)), dtype="float32")
        samples_start = time.perf_counter()
        payloads = cache_sample_payloads(
            records=split_records,
            targets=targets,
            image_size=image_size,
            cache_dtype=cache_dtype,
            mean=mean,
            std=std,
            workers=cache_build_workers,
        )
        samples_seconds = time.perf_counter() - samples_start
        metadata_start = time.perf_counter()
        with metadata_path.open("w", encoding="utf-8") as handle:
            for index, payload in enumerate(payloads):
                error = payload.get("error")
                if error:
                    failed.append(str(error))
                    continue
                images[index, 0, :, :] = payload["image"]
                labels[index, :] = payload["labels"]
                masks[index, :] = payload["mask"]
                handle.write(str(payload["metadata_json"]) + "\n")
        metadata_seconds = time.perf_counter() - metadata_start
        flush_start = time.perf_counter()
        images.flush()
        image_flush_seconds = time.perf_counter() - flush_start
        labels_masks_start = time.perf_counter()
        labels.tofile(labels_path)
        masks.tofile(masks_path)
        labels_masks_seconds = time.perf_counter() - labels_masks_start
        hash_start = time.perf_counter()
        images_sha256 = hash_file(images_path)
        labels_sha256 = hash_file(labels_path)
        masks_sha256 = hash_file(masks_path)
        metadata_sha256 = hash_file(metadata_path)
        hash_seconds = time.perf_counter() - hash_start
        split_reports[split] = {
            "samples": len(split_records),
            "images_path": str(images_path),
            "images_sha256": images_sha256,
            "labels_path": str(labels_path),
            "labels_sha256": labels_sha256,
            "masks_path": str(masks_path),
            "masks_sha256": masks_sha256,
            "metadata_path": str(metadata_path),
            "metadata_sha256": metadata_sha256,
            "shape": list(shape),
            "image_bytes": images_path.stat().st_size if images_path.exists() else 0,
            "build_seconds": time.perf_counter() - split_start,
            "sample_payload_seconds": samples_seconds,
            "metadata_write_seconds": metadata_seconds,
            "image_flush_seconds": image_flush_seconds,
            "labels_masks_write_seconds": labels_masks_seconds,
            "hash_seconds": hash_seconds,
        }

    report = {
        "cache_schema_version": 1,
        "report_schema_version": 1,
        "cache_dir": str(cache_dir),
        "cache_kind": f"medkit_rust_compatible_mmap_{cache_dtype}",
        "cache_key_mode": cache_key_mode,
        "cache_identity": cache_identity_report(
            records=records,
            targets=targets,
            image_size=image_size,
            cache_dtype=cache_dtype,
        ),
        "cache_reused": False,
        "channels": 1,
        "dtype": cache_dtype,
        "image_size": image_size,
        "targets": list(targets),
        "label_policy": {
            "positive": "label=1 mask=1",
            "negative": "label=0 mask=1",
            "uncertain": "ignore",
            "missing": "ignore",
            "loss_mask": "uncertain and missing labels are masked from loss",
        },
        "transform_plan": transform,
        "transform_plan_hash": transform_hash,
        "transform_fingerprint": transform_hash,
        "source_manifest_checksum": manifest_checksum(records),
        "split_names": ["train", "val", "test"],
        "image_size_policy": {
            "channels": 1,
            "height": image_size,
            "width": image_size,
            "dtype": cache_dtype,
            "transform": "decode grayscale, resize square, normalize dataset mean/std",
        },
        "normalization": {"mean": mean, "std": std},
        "cache_build_workers": cache_build_workers,
        "mean_std_seconds": mean_std_seconds,
        "build_seconds": time.perf_counter() - build_start,
        "failed_samples": failed,
        "filtered_samples": [],
        "splits": split_reports,
        "cache_size_bytes": directory_size(cache_dir),
        "seed": seed,
        "limitations": [
            "This cache is still materialized by the Python benchmark harness.",
            "The file layout is compatible with the Rust CxrCacheReader: raw float32 image, label, mask, and metadata files.",
            "The Rust `medkit cxr cache` CLI exists, but this Modal benchmark still needs a Rust-owned cache-build path.",
        ],
    }
    write_json(cache_dir / "cache-metadata.json", report)
    return report


def cache_sample_payloads(
    *,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    image_size: int,
    cache_dtype: str,
    mean: float,
    std: float,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1 or len(records) <= 1:
        return [
            cache_sample_payload(
                record=record,
                targets=targets,
                image_size=image_size,
                cache_dtype=cache_dtype,
                mean=mean,
                std=std,
            )
            for record in records
        ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                cache_sample_payload,
                record=record,
                targets=targets,
                image_size=image_size,
                cache_dtype=cache_dtype,
                mean=mean,
                std=std,
            )
            for record in records
        ]
        return [future.result() for future in futures]


def cache_sample_payload(
    *,
    record: SampleRecord,
    targets: Sequence[str],
    image_size: int,
    cache_dtype: str,
    mean: float,
    std: float,
) -> dict[str, Any]:
    numpy = import_numpy()
    try:
        if cache_dtype == "uint8":
            image = load_resized_grayscale(record.image_path, image_size)
        else:
            image = preprocess_image_to_numpy(
                record.image_path,
                image_size=image_size,
                mean=mean,
                std=std,
            ).astype(cache_dtype)
        labels = numpy.zeros((len(targets),), dtype="float32")
        mask = numpy.zeros((len(targets),), dtype="float32")
        for target_index, target in enumerate(targets):
            value = record.labels.get(target)
            if value is None or value == -1:
                mask[target_index] = 0.0
                labels[target_index] = 0.0
            else:
                mask[target_index] = 1.0
                labels[target_index] = float(value)
        return {
            "image": image,
            "labels": labels,
            "mask": mask,
            "metadata_json": json.dumps(record_to_json(record), sort_keys=True),
            "error": "",
        }
    except Exception as error:
        return {
            "image": None,
            "labels": numpy.zeros((len(targets),), dtype="float32"),
            "mask": numpy.zeros((len(targets),), dtype="float32"),
            "metadata_json": "",
            "error": f"{record.sample_id}: {error}",
        }


def cache_transform_plan(*, image_size: int, cache_dtype: str) -> dict[str, Any]:
    return {
        "name": f"cxr-{image_size}",
        "channels": 1,
        "size": [image_size, image_size],
        "dtype": cache_dtype,
        "operations": [
            "decode_png_or_jpeg",
            "convert_grayscale",
            "resize_square_area",
            "store_uint8_pixels" if cache_dtype == "uint8" else "scale_0_1",
            "defer_normalize_to_reader" if cache_dtype == "uint8" else "normalize_train_mean_std",
        ],
    }


def cache_identity_report(
    *,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    image_size: int,
    cache_dtype: str,
) -> dict[str, Any]:
    transform_fingerprint = stable_hash(
        cache_transform_plan(image_size=image_size, cache_dtype=cache_dtype)
    )
    source_manifest_checksum = manifest_checksum(records)
    target_fingerprint = stable_hash({"targets": list(targets)})
    return {
        "schema_version": 1,
        "image_size": image_size,
        "cache_dtype": cache_dtype,
        "source_manifest_checksum": source_manifest_checksum,
        "transform_fingerprint": transform_fingerprint,
        "target_fingerprint": target_fingerprint,
        "content_key": (
            f"cache-{image_size}-{cache_dtype}-"
            f"m{source_manifest_checksum[:12]}-"
            f"x{transform_fingerprint[:12]}-"
            f"y{target_fingerprint[:12]}"
        ),
    }


def cache_dir_for_run(
    *,
    dataset_work_dir: Path,
    image_size: int,
    cache_dtype: str,
    cache_key_mode: str,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
) -> Path:
    if cache_key_mode == "legacy":
        return dataset_work_dir / f"cache-{image_size}-{cache_dtype}"
    if cache_key_mode != "content":
        raise ValueError(f"unsupported cache key mode: {cache_key_mode}")
    return dataset_work_dir / "caches" / cache_identity_report(
        records=records,
        targets=targets,
        image_size=image_size,
        cache_dtype=cache_dtype,
    )["content_key"]


def enforce_destructive_cache_guard(
    *,
    force_cache: bool,
    allow_destructive_cache: bool,
    smoke: bool,
    cache_dir: Path,
    cache_metadata_path: Path,
) -> None:
    if not force_cache or allow_destructive_cache or smoke or not cache_metadata_path.exists():
        return
    raise ValueError(
        "--force-cache would rebuild an existing non-smoke cache at "
        f"{cache_dir}. Pass --allow-destructive-cache for intentional rebuilds, "
        "or omit --force-cache to reuse a matching cache."
    )


def cache_matches_run(
    cache_report: dict[str, Any],
    *,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    image_size: int,
    cache_dtype: str,
) -> bool:
    if int(cache_report.get("cache_schema_version", 0)) != 1:
        return False
    if cache_report.get("cache_kind") != f"medkit_rust_compatible_mmap_{cache_dtype}":
        return False
    if str(cache_report.get("dtype", "float32")) != cache_dtype:
        return False
    if int(cache_report.get("image_size", 0)) != image_size:
        return False
    if list(cache_report.get("targets", [])) != list(targets):
        return False
    if cache_report.get("source_manifest_checksum") != manifest_checksum(records):
        return False
    splits = cache_report.get("splits")
    if not isinstance(splits, dict):
        return False

    counts = {name: 0 for name in ("train", "val", "test")}
    for record in records:
        if record.split in counts:
            counts[record.split] += 1

    for split, expected_count in counts.items():
        split_info = splits.get(split)
        if not isinstance(split_info, dict):
            return False
        if int(split_info.get("samples", -1)) != expected_count:
            return False
        expected_shape = [expected_count, 1, image_size, image_size]
        if list(split_info.get("shape", [])) != expected_shape:
            return False
    return True


def manifest_checksum(records: Sequence[SampleRecord]) -> str:
    return stable_hash([record_to_json(record) for record in records])


def build_webdataset_shards(
    *,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    shard_dir: Path,
    shard_size: int,
    seed: int,
) -> dict[str, Any]:
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    build_start = time.perf_counter()
    split_reports: dict[str, Any] = {}
    failures: list[str] = []
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        shard_paths: list[str] = []
        sample_count = 0
        for shard_index, start in enumerate(range(0, len(split_records), shard_size)):
            shard_records = split_records[start : start + shard_size]
            tar_path = shard_dir / f"{split}-{shard_index:05d}.tar"
            with tarfile.open(tar_path, "w") as tar:
                for local_index, record in enumerate(shard_records):
                    key = f"{split}-{start + local_index:08d}"
                    try:
                        image_bytes = Path(record.image_path).read_bytes()
                    except Exception as error:
                        failures.append(f"{record.sample_id}: {error}")
                        continue
                    labels, mask = labels_to_plain_lists(record, targets)
                    metadata = {
                        "sample_id": record.sample_id,
                        "patient_id": record.patient_id,
                        "study_id": record.study_id,
                        "image_id": record.image_id,
                        "source_path": record.image_path,
                        "sample_hash": record.sha256,
                        "view_position": "unknown",
                        "labels": labels,
                        "mask": mask,
                    }
                    add_tar_bytes(tar, f"{key}.png", image_bytes)
                    add_tar_bytes(
                        tar,
                        f"{key}.json",
                        json.dumps(metadata, sort_keys=True).encode("utf-8"),
                    )
                    sample_count += 1
            shard_paths.append(str(tar_path))
        split_reports[split] = {
            "samples": sample_count,
            "shards": shard_paths,
            "shard_count": len(shard_paths),
            "bytes": sum(Path(path).stat().st_size for path in shard_paths),
        }

    report = {
        "cache_kind": "webdataset_raw_image_tar",
        "cache_reused": False,
        "seed": seed,
        "targets": list(targets),
        "shard_size": shard_size,
        "splits": split_reports,
        "build_seconds": time.perf_counter() - build_start,
        "cache_size_bytes": directory_size(shard_dir),
        "failed_samples": failures,
        "limitations": [
            "WebDataset shards store raw materialized images and labels; resize and normalization still run in the loader.",
        ],
    }
    write_json(shard_dir / "webdataset-metadata.json", report)
    return report


def add_tar_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, BytesIO(data))


def labels_to_plain_lists(record: SampleRecord, targets: Sequence[str]) -> tuple[list[float], list[float]]:
    labels: list[float] = []
    mask: list[float] = []
    for target in targets:
        value = record.labels.get(target)
        if value is None or value == -1:
            labels.append(0.0)
            mask.append(0.0)
        else:
            labels.append(float(value))
            mask.append(1.0)
    return labels, mask


@dataclass(frozen=True)
class TrainBatchSchedule:
    batch_size: int
    seed: int
    epochs: int
    warmup_batches: int
    drop_last_train: bool
    shuffle_block_batches: int
    train_sample_count: int
    iteration_names: tuple[str, ...]
    iteration_batches: tuple[tuple[tuple[int, ...], ...], ...]

    def batches_for_iteration(self, iteration_index: int) -> tuple[tuple[int, ...], ...]:
        if iteration_index < 0 or iteration_index >= len(self.iteration_batches):
            raise RuntimeError(
                "paired train batch schedule exhausted: "
                f"requested iterator {iteration_index}, available {len(self.iteration_batches)}"
            )
        return self.iteration_batches[iteration_index]

    def summary(self) -> dict[str, Any]:
        iteration_summaries = []
        for name, batches in zip(self.iteration_names, self.iteration_batches, strict=True):
            sample_ids = [index for batch in batches for index in batch]
            iteration_summaries.append(
                {
                    "name": name,
                    "batches": len(batches),
                    "samples": len(sample_ids),
                    "sample_order_hash": stable_hash(sample_ids),
                    "sample_multiset_hash": stable_hash(sorted(sample_ids)),
                    "dropped_sample_count": self.train_sample_count - len(set(sample_ids)),
                    "dropped_sample_indices": sorted(set(range(self.train_sample_count)) - set(sample_ids)),
                }
            )
        return {
            "schema_version": 1,
            "enabled": True,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epochs": self.epochs,
            "warmup_batches": self.warmup_batches,
            "drop_last_train": self.drop_last_train,
            "shuffle_block_batches": self.shuffle_block_batches,
            "train_sample_count": self.train_sample_count,
            "iteration_count": len(self.iteration_batches),
            "iteration_names": list(self.iteration_names),
            "hashes": {
                "schedule_hash": stable_hash(
                    [
                        {"name": name, "batches": batches}
                        for name, batches in zip(
                            self.iteration_names,
                            self.iteration_batches,
                            strict=True,
                        )
                    ]
                ),
                "train_epoch_order_hashes": [
                    stable_hash([index for batch in batches for index in batch])
                    for name, batches in zip(
                        self.iteration_names,
                        self.iteration_batches,
                        strict=True,
                    )
                    if name.startswith("epoch:")
                ],
            },
            "iterations": iteration_summaries,
        }


class FixedBatchSampler:
    def __init__(self, batches: Sequence[Sequence[int]]) -> None:
        self.batches = tuple(tuple(int(index) for index in batch) for batch in batches)

    def __iter__(self) -> Iterable[list[int]]:
        return iter([list(batch) for batch in self.batches])

    def __len__(self) -> int:
        return len(self.batches)


class FixedTrainBatchScheduleSampler:
    def __init__(self, schedule: TrainBatchSchedule) -> None:
        self.schedule = schedule
        self.iteration_index = 0
        self.extra_empty_iterations = 0

    def __iter__(self) -> Iterable[list[int]]:
        if self.iteration_index >= len(self.schedule.iteration_batches):
            self.iteration_index += 1
            self.extra_empty_iterations += 1
            return iter(())
        batches = self.schedule.batches_for_iteration(self.iteration_index)
        self.iteration_index += 1
        return iter([list(batch) for batch in batches])

    def report_metadata(self) -> dict[str, Any]:
        summary = self.schedule.summary()
        return {
            "paired_train_order": True,
            "batch_schedule": "fixed_by_iteration",
            "batch_schedule_hash": (summary.get("hashes") or {}).get("schedule_hash"),
            "batch_schedule_iteration_count": summary.get("iteration_count"),
            "batch_schedule_iteration_names": summary.get("iteration_names"),
            "batch_schedule_current_iteration": self.iteration_index,
            "batch_schedule_extra_empty_iterations": self.extra_empty_iterations,
        }


class ScheduledTorchMapLoader:
    """Build a fresh DataLoader for each fixed train-schedule iteration."""

    def __init__(
        self,
        *,
        torch: Any,
        dataset: Any,
        schedule: TrainBatchSchedule,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        metadata: dict[str, Any],
    ) -> None:
        self.torch = torch
        self.dataset = dataset
        self.schedule = schedule
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.metadata = dict(metadata)
        self.iteration_index = 0
        self.extra_empty_iterations = 0

    def __iter__(self) -> Iterable[Any]:
        if self.iteration_index >= len(self.schedule.iteration_batches):
            self.iteration_index += 1
            self.extra_empty_iterations += 1
            return iter(())
        batches = self.schedule.batches_for_iteration(self.iteration_index)
        self.iteration_index += 1
        loader = self.torch.utils.data.DataLoader(
            self.dataset,
            batch_sampler=FixedBatchSampler(batches),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
        return iter(loader)

    def __len__(self) -> int:
        if not self.schedule.iteration_batches:
            return 0
        index = min(self.iteration_index, len(self.schedule.iteration_batches) - 1)
        return len(self.schedule.iteration_batches[index])

    def report_metadata(self) -> dict[str, Any]:
        summary = self.schedule.summary()
        return {
            **self.metadata,
            "paired_train_order": True,
            "batch_schedule": "fixed_by_iteration",
            "batch_schedule_hash": (summary.get("hashes") or {}).get("schedule_hash"),
            "batch_schedule_iteration_count": summary.get("iteration_count"),
            "batch_schedule_iteration_names": summary.get("iteration_names"),
            "batch_schedule_current_iteration": self.iteration_index,
            "batch_schedule_extra_empty_iterations": self.extra_empty_iterations,
        }


def build_train_batch_schedule(
    *,
    train_records: Sequence[SampleRecord],
    batch_size: int,
    seed: int,
    epochs: int,
    warmup_batches: int,
    drop_last_train: bool,
    shuffle_block_batches: int,
) -> TrainBatchSchedule:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    if warmup_batches < 0:
        raise ValueError("warmup_batches must be non-negative")
    if shuffle_block_batches < 0:
        raise ValueError("shuffle_block_batches must be non-negative")
    names: list[str] = []
    iterations: list[tuple[tuple[int, ...], ...]] = []
    if warmup_batches > 0:
        warmup = train_schedule_batches_for_phase(
            sample_count=len(train_records),
            batch_size=batch_size,
            seed=seed,
            phase="warmup",
            epoch=None,
            drop_last_train=drop_last_train,
            shuffle_block_batches=shuffle_block_batches,
        )
        names.append("warmup")
        iterations.append(tuple(warmup[:warmup_batches]))
    for epoch in range(epochs):
        names.append(f"epoch:{epoch}")
        iterations.append(
            tuple(
                train_schedule_batches_for_phase(
                    sample_count=len(train_records),
                    batch_size=batch_size,
                    seed=seed,
                    phase="train",
                    epoch=epoch,
                    drop_last_train=drop_last_train,
                    shuffle_block_batches=shuffle_block_batches,
                )
            )
        )
    return TrainBatchSchedule(
        batch_size=batch_size,
        seed=seed,
        epochs=epochs,
        warmup_batches=warmup_batches,
        drop_last_train=drop_last_train,
        shuffle_block_batches=shuffle_block_batches,
        train_sample_count=len(train_records),
        iteration_names=tuple(names),
        iteration_batches=tuple(iterations),
    )


def train_schedule_batches_for_phase(
    *,
    sample_count: int,
    batch_size: int,
    seed: int,
    phase: str,
    epoch: int | None,
    drop_last_train: bool,
    shuffle_block_batches: int,
) -> list[tuple[int, ...]]:
    order = list(range(sample_count))
    rng = random.Random(train_schedule_rng_seed(seed=seed, phase=phase, epoch=epoch))
    if shuffle_block_batches <= 0:
        rng.shuffle(order)
    else:
        block_size = batch_size * shuffle_block_batches
        blocks = [order[start : start + block_size] for start in range(0, len(order), block_size)]
        rng.shuffle(blocks)
        order = [index for block in blocks for index in block]
    batches: list[tuple[int, ...]] = []
    for start in range(0, len(order), batch_size):
        batch = tuple(order[start : start + batch_size])
        if drop_last_train and len(batch) != batch_size:
            continue
        batches.append(batch)
    return batches


def train_schedule_rng_seed(*, seed: int, phase: str, epoch: int | None) -> int:
    payload = {"seed": seed, "phase": phase, "epoch": epoch}
    return int(stable_hash(payload)[:16], 16)


def train_batch_schedule_report(schedule: TrainBatchSchedule | None) -> dict[str, Any]:
    if schedule is None:
        return {"schema_version": 1, "enabled": False, "status": "disabled"}
    report = schedule.summary()
    report["status"] = "ok"
    return report


def estimate_mean_std(records: Sequence[SampleRecord], image_size: int) -> tuple[float, float]:
    numpy = import_numpy()
    if not records:
        return 0.5, 0.25
    stride = max(1, len(records) // 512)
    sums = 0.0
    sq_sums = 0.0
    count = 0
    for record in records[::stride]:
        array = load_resized_grayscale(record.image_path, image_size).astype("float32") / 255.0
        sums += float(array.sum())
        sq_sums += float((array * array).sum())
        count += int(array.size)
    mean = sums / max(count, 1)
    variance = max(sq_sums / max(count, 1) - mean * mean, 1.0e-6)
    std = float(math.sqrt(variance))
    return float(mean), float(max(std, 1.0e-3))


def preprocess_image_to_numpy(path: str, *, image_size: int, mean: float, std: float) -> Any:
    array = load_resized_grayscale(path, image_size).astype("float32") / 255.0
    return (array - mean) / std


def load_resized_grayscale(path: str, image_size: int) -> Any:
    if Path(path).suffix.lower() in {".dcm", ".dicom", ".ima"}:
        return load_resized_dicom_grayscale(path, image_size)
    pillow = import_pillow()
    Image = pillow["Image"]
    image = Image.open(path).convert("L")
    return resize_pil_to_array(image, image_size)


def load_resized_dicom_grayscale(path: str, image_size: int) -> Any:
    numpy = import_numpy()
    pillow = import_pillow()
    Image = pillow["Image"]
    pydicom = import_pydicom()
    dataset = pydicom.dcmread(path)
    array = dataset.pixel_array.astype("float32")
    if array.ndim == 3:
        array = array[..., 0]
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    array = array * slope + intercept
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    if photometric == "MONOCHROME1":
        array = float(array.max()) + float(array.min()) - array
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        scaled = numpy.zeros(array.shape, dtype="uint8")
    else:
        scaled = numpy.clip((array - low) / (high - low) * 255.0, 0.0, 255.0).astype("uint8")
    image = Image.fromarray(scaled, mode="L")
    return resize_pil_to_array(image, image_size)


def resize_pil_to_array(image: Any, image_size: int) -> Any:
    numpy = import_numpy()
    pillow = import_pillow()
    Image = pillow["Image"]
    resample = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((image_size, image_size), resample=resample)
    return numpy.asarray(image)


def run_all_baselines(
    *,
    args: argparse.Namespace,
    torch: Any,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    baselines: Sequence[str],
    cache_dir: Path,
    webdataset_dir: Path,
    image_size: int,
    device: Any,
    report_dir: Path,
    capture_eval_predictions: bool,
    capture_train_order: bool,
    train_batch_schedule: TrainBatchSchedule | None,
) -> dict[str, dict[str, Any]]:
    loss_pos_weight_values = (
        class_pos_weight_values(
            records,
            targets,
            cap=args.loss_pos_weight_cap if args.loss_pos_weight_cap > 0.0 else None,
        )
        if args.loss_pos_weight == "balanced"
        else None
    )
    reports: dict[str, dict[str, Any]] = {
        "loader": {},
        "gpu": {},
        "profile": {},
        "quality": {},
        "thresholds": {},
        "predictions": {},
        "train_order": {},
        "localization_eval": {},
    }
    eval_records = [record for record in records if record.split == "val"]
    train_records = [record for record in records if record.split == "train"]
    for baseline in baselines:
        try:
            loader_factory = make_loader_factory(
                baseline=baseline,
                records=records,
                targets=targets,
                cache_dir=cache_dir,
                webdataset_dir=webdataset_dir,
                image_size=image_size,
                batch_size=args.batch_size,
                workers=args.workers,
                prefetch_depth=args.prefetch_depth,
                prefetch_read_workers=args.prefetch_read_workers,
                shuffle_block_batches=args.shuffle_block_batches,
                read_mode=args.read_mode,
                include_metadata=args.include_metadata,
                drop_last_train=args.drop_last_train,
                seed=args.seed,
                train_batch_schedule=train_batch_schedule,
            )
        except Exception as error:
            unavailable = {"status": "unavailable", "reason": str(error)}
            reports["loader"][baseline] = unavailable
            reports["gpu"][baseline] = unavailable
            reports["quality"][baseline] = unavailable
            reports["thresholds"][baseline] = unavailable
            reports["predictions"][baseline] = prediction_capture_disabled_report(
                baseline=baseline,
                reason=str(error),
            )
            reports["train_order"][baseline] = train_order_capture_disabled_report(
                baseline=baseline,
                reason=str(error),
            )
            reports["localization_eval"][baseline] = localization_eval_disabled_report(
                baseline=baseline,
                reason=str(error),
            )
            continue

        try:
            train_loader = loader_factory("train", shuffle=False)
            val_loader = loader_factory("val", shuffle=False)
            reports["loader"][baseline] = benchmark_loader(
                train_loader,
                max_batches=args.loader_batches,
                baseline=baseline,
            )
            prediction_artifact_path = (
                report_dir / f"eval-predictions-{safe_artifact_name(baseline)}.jsonl.gz"
                if capture_eval_predictions
                else None
            )
            train_order_artifact_path = (
                report_dir / f"train-order-{safe_artifact_name(baseline)}.jsonl.gz"
                if capture_train_order
                else None
            )
            (
                train_report,
                quality_report,
                threshold_report,
                profile_report,
                prediction_report,
                train_order_report,
            ) = train_and_evaluate(
                torch=torch,
                baseline=baseline,
                train_loader=loader_factory("train", shuffle=True),
                val_loader=val_loader,
                train_records=train_records,
                eval_records=eval_records,
                targets=targets,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                drop_last_train=args.drop_last_train,
                warmup_batches=args.warmup_batches,
                profile_batches=args.profile_batches,
                gpu_prefetch_batches=args.gpu_prefetch_batches,
                gpu_prefetch_reuse_buffers=args.gpu_prefetch_reuse_buffers,
                sync_every_step=args.sync_every_step,
                channels_last=args.channels_last,
                torch_compile=args.torch_compile,
                torch_compile_mode=args.torch_compile_mode,
                learning_rate=args.learning_rate,
                amp_dtype=args.amp_dtype,
                model_init=args.model_init,
                loss_kind=args.loss_kind,
                loss_pos_weight_values=loss_pos_weight_values,
                loss_pos_weight_mode=args.loss_pos_weight,
                loss_pos_weight_cap=args.loss_pos_weight_cap,
                focal_gamma=args.focal_gamma,
                focal_alpha=args.focal_alpha,
                seed=args.seed,
                prediction_artifact_path=prediction_artifact_path,
                train_order_artifact_path=train_order_artifact_path,
            )
            reports["gpu"][baseline] = train_report
            reports["profile"][baseline] = profile_report
            reports["quality"][baseline] = quality_report
            reports["thresholds"][baseline] = threshold_report
            reports["predictions"][baseline] = prediction_report
            reports["train_order"][baseline] = train_order_report
            reports["localization_eval"][baseline] = quality_report.get(
                "localization",
                localization_eval_disabled_report(
                    baseline=baseline,
                    reason="localization evaluation not emitted",
                ),
            )
        except Exception as error:
            failed = {
                "status": "failed",
                "baseline": baseline,
                "reason": f"{type(error).__name__}: {error}",
            }
            reports["loader"].setdefault(baseline, failed)
            reports["gpu"][baseline] = failed
            reports["profile"][baseline] = failed
            reports["quality"][baseline] = failed
            reports["thresholds"][baseline] = failed
            reports["predictions"][baseline] = failed
            reports["train_order"][baseline] = failed
            reports["localization_eval"][baseline] = failed
    reports["predictions"] = prediction_summary_report(
        report_dir=report_dir,
        predictions=reports["predictions"],
        quality=reports["quality"],
        targets=targets,
        capture_enabled=capture_eval_predictions,
    )
    reports["train_order"] = train_order_summary_report(
        report_dir=report_dir,
        train_order=reports["train_order"],
        targets=targets,
        capture_enabled=capture_train_order,
    )
    return reports


def with_report_metadata(
    loader: Any,
    metadata: dict[str, Any],
    *,
    dataset: Any | None = None,
) -> Any:
    if dataset is None:
        loader.report_metadata = lambda metadata=metadata: dict(metadata)
    else:
        loader.report_metadata = (
            lambda metadata=metadata, dataset=dataset: metadata_from_dataset_report(
                dataset, metadata
            )
        )
    return loader


def metadata_from_dataset_report(dataset: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(fallback)
    report_metadata = getattr(dataset, "report_metadata", None)
    if callable(report_metadata):
        report = report_metadata()
        if isinstance(report, dict):
            metadata.update(report)
    for key in (
        "baseline",
        "cache_dtype",
        "dataset_api",
        "loader_api",
        "dropin_api",
        "native_prefetch_threads",
    ):
        if key in fallback:
            metadata[key] = fallback[key]
    return metadata


def make_loader_factory(
    *,
    baseline: str,
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    cache_dir: Path,
    webdataset_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    prefetch_depth: int,
    prefetch_read_workers: int,
    shuffle_block_batches: int,
    read_mode: str,
    include_metadata: bool,
    drop_last_train: bool,
    seed: int,
    train_batch_schedule: TrainBatchSchedule | None = None,
) -> Any:
    torch = import_torch()
    numpy = import_numpy()
    mean, std = cache_normalization(cache_dir)

    def drop_last_for_split(split: str) -> bool:
        return bool(drop_last_train and split == "train")

    def use_train_batch_schedule(split: str, shuffle: bool) -> bool:
        return train_batch_schedule is not None and split == "train" and shuffle

    def train_batch_schedule_iterations(split: str, shuffle: bool) -> Any:
        if not use_train_batch_schedule(split, shuffle):
            return None
        assert train_batch_schedule is not None
        return train_batch_schedule.iteration_batches

    def train_batch_schedule_metadata(split: str, shuffle: bool) -> dict[str, Any]:
        if not use_train_batch_schedule(split, shuffle):
            return {"paired_train_order": False}
        assert train_batch_schedule is not None
        report = train_batch_schedule.summary()
        return {
            "paired_train_order": True,
            "batch_schedule": "fixed_by_iteration",
            "batch_schedule_hash": (report.get("hashes") or {}).get("schedule_hash"),
            "batch_schedule_iteration_count": report.get("iteration_count"),
            "batch_schedule_iteration_names": report.get("iteration_names"),
        }

    def make_torch_map_loader(
        dataset: Any,
        *,
        split: str,
        shuffle: bool,
        baseline_name: str,
        num_workers: int,
        pin_memory: bool,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        if use_train_batch_schedule(split, shuffle):
            assert train_batch_schedule is not None
            scheduled_num_workers = num_workers
            return ScheduledTorchMapLoader(
                torch=torch,
                dataset=dataset,
                schedule=train_batch_schedule,
                num_workers=scheduled_num_workers,
                pin_memory=pin_memory,
                persistent_workers=False,
                metadata={
                    "baseline": baseline_name,
                    "batch_size": batch_size,
                    "cache_dtype": cache_dtype_from_metadata(cache_dir),
                    "read_mode": read_mode,
                    "shuffle_block_batches": shuffle_block_batches,
                    "shuffle": shuffle,
                    "drop_last": drop_last_for_split(split),
                    "requested_num_workers": num_workers,
                    "num_workers": scheduled_num_workers,
                    "pin_memory": pin_memory,
                    "worker_mode": (
                        "paired_schedule_pytorch_workers"
                        if scheduled_num_workers > 0
                        else "paired_schedule_single_process"
                    ),
                    "native_prefetch": False,
                    **(metadata or {}),
                },
            )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last_for_split(split),
        )
        if metadata is not None:
            return with_report_metadata(loader, metadata)
        return loader

    if baseline == "pytorch_raw":
        dataset_by_split = {
            split: RawCxrDataset(
                [record for record in records if record.split == split],
                targets,
                image_size=image_size,
                mean=mean,
                std=std,
                backend="pytorch_raw",
            )
            for split in ("train", "val", "test")
        }

        def make_raw_loader(split: str, shuffle: bool = False) -> Any:
            return make_torch_map_loader(
                dataset_by_split[split],
                split=split,
                shuffle=shuffle,
                baseline_name=baseline,
                num_workers=workers,
                pin_memory=False,
            )

        return make_raw_loader
    if baseline == "monai_raw":
        monai = import_monai()
        monai_transforms = make_monai_transform(
            monai=monai,
            image_size=image_size,
            mean=mean,
            std=std,
        )
        dataset_by_split = {
            split: monai.data.Dataset(
                data=monai_rows(
                    [record for record in records if record.split == split],
                    targets,
                    numpy=numpy,
                ),
                transform=monai_transforms,
            )
            for split in ("train", "val", "test")
        }

        def make_monai_loader(split: str, shuffle: bool = False) -> Any:
            return make_torch_map_loader(
                dataset_by_split[split],
                split=split,
                shuffle=shuffle,
                baseline_name=baseline,
                num_workers=workers,
                pin_memory=False,
            )

        return make_monai_loader
    if baseline == "torchxrayvision":
        xrv = import_torchxrayvision()
        torchvision = import_torchvision()
        xrv_transform = torchvision.transforms.Compose(
            [
                xrv.datasets.XRayCenterCrop(),
                xrv.datasets.XRayResizer(image_size),
            ]
        )
        dataset_by_split = {
            split: TorchXRayVisionCxrDataset(
                [record for record in records if record.split == split],
                targets,
                transform=xrv_transform,
                xrv=xrv,
            )
            for split in ("train", "val", "test")
        }

        def make_torchxrayvision_loader(split: str, shuffle: bool = False) -> Any:
            return make_torch_map_loader(
                dataset_by_split[split],
                split=split,
                shuffle=shuffle,
                baseline_name=baseline,
                num_workers=workers,
                pin_memory=False,
            )

        return make_torchxrayvision_loader
    if baseline in {"medkit_cached_mmap", "medkit_pinned_prefetch", "medkit_cached_resident"}:
        resident = baseline == "medkit_cached_resident"
        pin_memory = baseline == "medkit_pinned_prefetch"
        dataset_by_split = {
            split: CachedCxrDataset(
                cache_dir=cache_dir,
                split=split,
                targets=targets,
                numpy=numpy,
                resident=resident,
            )
            for split in ("train", "val", "test")
        }

        def make_cached_loader(split: str, shuffle: bool = False) -> Any:
            return make_torch_map_loader(
                dataset_by_split[split],
                split=split,
                shuffle=shuffle,
                baseline_name=baseline,
                num_workers=0 if resident else workers,
                pin_memory=pin_memory,
                metadata={
                    "baseline": baseline,
                    "cache_dir": str(cache_dir),
                    "cache_dtype": cache_dtype_from_metadata(cache_dir),
                    "batch_size": batch_size,
                    "worker_mode": "resident" if resident else "pytorch_workers",
                    "num_workers": 0 if resident else workers,
                    "pin_memory": pin_memory,
                    "shuffle": shuffle,
                    "native_prefetch": False,
                },
            )

        return make_cached_loader
    if baseline in {"medkit_dropin_cxr", "medkit_dropin_cxr_pinned"}:
        medkit_rs = import_medkit_rs()
        pin_memory = baseline == "medkit_dropin_cxr_pinned"

        def make_dropin_loader(split: str, shuffle: bool = False) -> Any:
            dataset = medkit_rs.cxr.Dataset(
                cache_dir=cache_dir,
                split=split,
                read_mode=read_mode,
                include_metadata=include_metadata,
                shuffle_block_batches=shuffle_block_batches,
                drop_last=drop_last_for_split(split),
            )
            loader = medkit_rs.cxr.DataLoader(
                dataset,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=shuffle,
                seed=seed,
                prefetch=True,
                prefetch_depth=prefetch_depth,
                read_workers=prefetch_read_workers,
                read_mode=read_mode,
                include_metadata=include_metadata,
                shuffle_block_batches=shuffle_block_batches,
                drop_last=drop_last_for_split(split),
            )
            return with_report_metadata(
                loader,
                {
                    "baseline": baseline,
                    "cache_dir": str(cache_dir),
                    "cache_dtype": cache_dtype_from_metadata(cache_dir),
                    "dataset_api": "medkit_rs.cxr.Dataset",
                    "loader_api": "medkit_rs.cxr.DataLoader",
                    "batch_size": batch_size,
                    "worker_mode": "rust_thread_prefetch",
                    "num_workers": 0,
                    "pin_memory": pin_memory,
                    "prefetch_depth": prefetch_depth,
                    "prefetch_read_workers": prefetch_read_workers,
                    "read_mode": read_mode,
                    "include_metadata": include_metadata,
                    "shuffle_block_batches": shuffle_block_batches,
                    "shuffle": shuffle,
                    "native_prefetch": True,
                    "dropin_api": True,
                },
            )

        return make_dropin_loader
    if baseline in {"medkit_native_cxr", "medkit_native_cxr_pinned"}:
        medkit_rs = import_medkit_rs()
        pin_memory = baseline == "medkit_native_cxr_pinned"

        def make_native_loader(split: str, shuffle: bool = False) -> Any:
            dataset = medkit_rs.MedkitCxrNativeBatchIterableDataset(
                cache_dir=cache_dir,
                split=split,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=shuffle,
                seed=seed,
                read_mode=read_mode,
                include_metadata=include_metadata,
                shuffle_block_batches=shuffle_block_batches,
                drop_last=drop_last_for_split(split),
                batch_indices_by_iteration=train_batch_schedule_iterations(split, shuffle),
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            metadata = metadata_from_dataset_report(
                dataset,
                {
                    "baseline": baseline,
                    "cache_dir": str(cache_dir),
                    "cache_dtype": cache_dtype_from_metadata(cache_dir),
                    "batch_size": batch_size,
                    "worker_mode": "single_process",
                    "num_workers": 0,
                    "pin_memory": pin_memory,
                    "read_mode": read_mode,
                    "include_metadata": include_metadata,
                    "shuffle_block_batches": shuffle_block_batches,
                    "shuffle": shuffle,
                    "native_prefetch": False,
                    **train_batch_schedule_metadata(split, shuffle),
                },
            )
            return with_report_metadata(loader, metadata, dataset=dataset)

        return make_native_loader
    if baseline in {"medkit_native_prefetch", "medkit_native_prefetch_pinned"}:
        medkit_rs = import_medkit_rs()
        pin_memory = baseline == "medkit_native_prefetch_pinned"

        def make_native_prefetch_loader(split: str, shuffle: bool = False) -> Any:
            dataset = medkit_rs.MedkitCxrNativePrefetchDataset(
                cache_dir=cache_dir,
                split=split,
                batch_size=batch_size,
                pin_memory=pin_memory,
                shuffle=shuffle,
                seed=seed,
                prefetch_depth=prefetch_depth,
                read_workers=prefetch_read_workers,
                read_mode=read_mode,
                include_metadata=include_metadata,
                shuffle_block_batches=shuffle_block_batches,
                drop_last=drop_last_for_split(split),
                batch_indices_by_iteration=train_batch_schedule_iterations(split, shuffle),
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            metadata = metadata_from_dataset_report(
                dataset,
                {
                    "baseline": baseline,
                    "cache_dir": str(cache_dir),
                    "cache_dtype": cache_dtype_from_metadata(cache_dir),
                    "batch_size": batch_size,
                    "worker_mode": "rust_thread_prefetch",
                    "num_workers": 0,
                    "pin_memory": pin_memory,
                    "prefetch_depth": prefetch_depth,
                    "prefetch_read_workers": prefetch_read_workers,
                    "read_mode": read_mode,
                    "include_metadata": include_metadata,
                    "shuffle_block_batches": shuffle_block_batches,
                    "shuffle": shuffle,
                    "native_prefetch": True,
                    "native_prefetch_threads": 1,
                    **train_batch_schedule_metadata(split, shuffle),
                },
            )
            return with_report_metadata(loader, metadata, dataset=dataset)

        return make_native_prefetch_loader
    if baseline == "webdataset":
        webdataset_report = load_json(webdataset_dir / "webdataset-metadata.json")

        def make_webdataset_loader(split: str, shuffle: bool = False) -> Any:
            dataset = make_webdataset_dataset(
                webdataset_report=webdataset_report,
                split=split,
                shuffle=shuffle,
                image_size=image_size,
                mean=mean,
                std=std,
            )
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=workers,
                pin_memory=False,
                persistent_workers=workers > 0,
                drop_last=drop_last_for_split(split),
            )

        return make_webdataset_loader
    if baseline == "dali":
        import_dali()
        dataset_by_split = {
            split: DaliCxrLoader(
                [record for record in records if record.split == split],
                targets,
                image_size=image_size,
                mean=mean,
                std=std,
                batch_size=batch_size,
                workers=workers,
                seed=seed,
            )
            for split in ("train", "val", "test")
        }
        return lambda split, shuffle=False: dataset_by_split[split].with_shuffle(shuffle)
    raise ValueError(f"Unknown baseline: {baseline}")


def make_monai_transform(*, monai: Any, image_size: int, mean: float, std: float) -> Any:
    transforms = monai.transforms
    return transforms.Compose(
        [
            transforms.LoadImaged(keys="image", image_only=True),
            transforms.EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
            transforms.Resized(keys="image", spatial_size=(image_size, image_size), mode="bilinear"),
            transforms.ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
            transforms.NormalizeIntensityd(
                keys="image",
                subtrahend=mean,
                divisor=max(std, 1.0e-3),
            ),
            transforms.ToTensord(keys=("image", "labels", "mask")),
        ]
    )


def make_webdataset_dataset(
    *,
    webdataset_report: dict[str, Any],
    split: str,
    shuffle: bool,
    image_size: int,
    mean: float,
    std: float,
) -> Any:
    webdataset = import_webdataset()
    split_info = webdataset_report["splits"][split]
    urls = split_info["shards"]
    if not urls:
        raise RuntimeError(f"No WebDataset shards for split {split}")
    dataset = webdataset.WebDataset(urls, shardshuffle=shuffle, empty_check=False)
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.to_tuple("png", "json").map(
        lambda sample: webdataset_sample_to_batch(
            sample,
            image_size=image_size,
            mean=mean,
            std=std,
        )
    )


def webdataset_sample_to_batch(
    sample: tuple[Any, Any],
    *,
    image_size: int,
    mean: float,
    std: float,
) -> dict[str, Any]:
    numpy = import_numpy()
    torch = import_torch()
    pillow = import_pillow()
    Image = pillow["Image"]
    image_payload, metadata_payload = sample
    image = Image.open(BytesIO(bytes(image_payload))).convert("L")
    image_array = resize_pil_to_array(image, image_size).astype("float32") / 255.0
    image_array = (image_array - mean) / std
    metadata = parse_webdataset_metadata(metadata_payload)
    return {
        "image": torch.from_numpy(image_array[None, :, :].copy()),
        "labels": torch.from_numpy(numpy.asarray(metadata["labels"], dtype="float32")),
        "mask": torch.from_numpy(numpy.asarray(metadata["mask"], dtype="float32")),
        "patient_id": metadata.get("patient_id", ""),
        "study_id": metadata.get("study_id", ""),
        "image_id": metadata.get("image_id", ""),
        "sample_id": metadata.get("sample_id", ""),
        "source_path": metadata.get("source_path", metadata.get("image_path", "")),
        "sample_hash": metadata.get("sample_hash", metadata.get("sha256", "")),
        "view_position": metadata.get("view_position", "unknown"),
    }


def parse_webdataset_metadata(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, bytes):
        return json.loads(payload.decode("utf-8"))
    if isinstance(payload, str):
        return json.loads(payload)
    raise TypeError(f"Unsupported WebDataset metadata payload: {type(payload)!r}")


def monai_rows(records: Sequence[SampleRecord], targets: Sequence[str], numpy: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        labels, mask = labels_to_arrays(record, targets)
        rows.append(
            {
                "image": record.image_path,
                "labels": labels.astype(numpy.float32),
                "mask": mask.astype(numpy.float32),
                "patient_id": record.patient_id,
                "study_id": record.study_id,
                "image_id": record.image_id,
                "sample_id": record.sample_id,
                "source_path": record.image_path,
                "sample_hash": record.sha256,
                "view_position": record.view_position,
            }
        )
    return rows


class RawCxrDataset:
    def __init__(
        self,
        records: Sequence[SampleRecord],
        targets: Sequence[str],
        *,
        image_size: int,
        mean: float,
        std: float,
        backend: str,
    ) -> None:
        self.records = list(records)
        self.targets = list(targets)
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.backend = backend

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = import_torch()
        record = self.records[index]
        image = preprocess_image_to_numpy(record.image_path, image_size=self.image_size, mean=self.mean, std=self.std)
        labels, mask = labels_to_arrays(record, self.targets)
        return {
            "image": torch.from_numpy(image[None, :, :].copy()),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "patient_id": record.patient_id,
            "study_id": record.study_id,
            "image_id": record.image_id,
            "sample_id": record.sample_id,
            "source_path": record.image_path,
            "sample_hash": record.sha256,
            "view_position": record.view_position,
        }


class CachedCxrDataset:
    def __init__(
        self,
        *,
        cache_dir: Path,
        split: str,
        targets: Sequence[str],
        numpy: Any,
        resident: bool,
    ) -> None:
        metadata = load_json(cache_dir / "cache-metadata.json")
        split_info = metadata["splits"][split]
        self.targets = list(targets)
        self.shape = tuple(split_info["shape"])
        self.image_dtype = str(metadata.get("dtype", "float32"))
        normalization = metadata.get("normalization", {})
        self.mean = float(normalization.get("mean", 0.5))
        self.std = float(normalization.get("std", 0.25))
        images = numpy.memmap(
            split_info["images_path"],
            dtype=self.image_dtype,
            mode="r",
            shape=tuple(self.shape),
        )
        self.images = numpy.asarray(images) if resident else images
        self.labels = load_cache_matrix(
            split_info["labels_path"],
            shape=(self.shape[0], len(targets)),
            numpy=numpy,
            resident=resident,
        )
        self.masks = load_cache_matrix(
            split_info["masks_path"],
            shape=(self.shape[0], len(targets)),
            numpy=numpy,
            resident=resident,
        )
        self.records = [json.loads(line) for line in Path(split_info["metadata_path"]).read_text().splitlines()]

    def __len__(self) -> int:
        return int(self.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch = import_torch()
        record = self.records[index]
        image = self.images[index]
        if self.image_dtype == "uint8":
            image = (image.astype("float32") / 255.0 - self.mean) / self.std
        else:
            image = image.astype("float32", copy=False)
        return {
            "image": torch.from_numpy(image.copy()),
            "labels": torch.from_numpy(self.labels[index].copy()),
            "mask": torch.from_numpy(self.masks[index].copy()),
            "patient_id": record.get("patient_id", ""),
            "study_id": record.get("study_id", ""),
            "image_id": record.get("image_id", ""),
            "sample_id": record.get("sample_id", ""),
            "source_path": record.get("image_path", ""),
            "sample_hash": record.get("sha256", ""),
            "view_position": record.get("view_position", "unknown"),
        }


def load_cache_matrix(
    path: str,
    *,
    shape: tuple[int, int],
    numpy: Any,
    resident: bool,
) -> Any:
    if path.endswith(".npy"):
        array = numpy.load(path)
        return numpy.asarray(array) if resident else array
    matrix = numpy.memmap(path, dtype="float32", mode="r", shape=shape)
    return numpy.asarray(matrix) if resident else matrix


class TorchXRayVisionCxrDataset:
    def __init__(
        self,
        records: Sequence[SampleRecord],
        targets: Sequence[str],
        *,
        transform: Any,
        xrv: Any,
    ) -> None:
        self.records = list(records)
        self.targets = list(targets)
        self.transform = transform
        self.xrv = xrv

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        numpy = import_numpy()
        torch = import_torch()
        pillow = import_pillow()
        Image = pillow["Image"]
        record = self.records[index]
        image = Image.open(record.image_path).convert("L")
        array = numpy.asarray(image).astype("float32")
        array = self.xrv.datasets.normalize(array, 255)
        if array.ndim == 2:
            array = array[None, :, :]
        array = self.transform(array).astype("float32")
        labels, mask = labels_to_arrays(record, self.targets)
        return {
            "image": torch.from_numpy(array.copy()),
            "labels": torch.from_numpy(labels),
            "mask": torch.from_numpy(mask),
            "patient_id": record.patient_id,
            "study_id": record.study_id,
            "image_id": record.image_id,
            "sample_id": record.sample_id,
            "source_path": record.image_path,
            "sample_hash": record.sha256,
            "view_position": record.view_position,
        }


class DaliCxrLoader:
    """Iterable wrapper over a one-epoch DALI CXR file-reader pipeline."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        targets: Sequence[str],
        *,
        image_size: int,
        mean: float,
        std: float,
        batch_size: int,
        workers: int,
        seed: int,
        shuffle: bool = False,
    ) -> None:
        numpy = import_numpy()
        torch = import_torch()
        self.records = list(records)
        self.targets = list(targets)
        self.image_size = image_size
        self.mean = mean
        self.std = max(std, 1.0e-3)
        self.batch_size = batch_size
        self.workers = max(1, workers)
        self.seed = seed
        self.shuffle = shuffle
        self.files = [str(Path(record.image_path)) for record in self.records]
        self.sample_indices = list(range(len(self.records)))
        labels: list[Any] = []
        masks: list[Any] = []
        for record in self.records:
            label, mask = labels_to_arrays(record, self.targets)
            labels.append(label)
            masks.append(mask)
        if labels:
            self.labels = torch.from_numpy(numpy.stack(labels).astype("float32", copy=False))
            self.masks = torch.from_numpy(numpy.stack(masks).astype("float32", copy=False))
        else:
            self.labels = torch.zeros((0, len(self.targets)), dtype=torch.float32)
            self.masks = torch.zeros((0, len(self.targets)), dtype=torch.float32)
        self.patient_ids = [record.patient_id for record in self.records]
        self.study_ids = [record.study_id for record in self.records]
        self.image_ids = [record.image_id for record in self.records]
        self.sample_ids = [record.sample_id for record in self.records]
        self.source_paths = [record.image_path for record in self.records]
        self.sample_hashes = [record.sha256 for record in self.records]
        self.view_positions = ["unknown" for _record in self.records]
        self.pipeline_mode = "not_built"
        self.pipeline_warnings: list[str] = []
        self.requested_modes = parse_dali_modes()

    def with_shuffle(self, shuffle: bool) -> "DaliCxrLoader":
        return DaliCxrLoader(
            self.records,
            self.targets,
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
            batch_size=self.batch_size,
            workers=self.workers,
            seed=self.seed,
            shuffle=shuffle,
        )

    def __iter__(self) -> Any:
        if not self.records:
            return iter(())
        dali = import_dali()
        pipe = self.build_pipeline(dali)
        iterator = make_dali_iterator(dali, pipe)
        return DaliCxrIterator(
            iterator,
            labels=self.labels,
            masks=self.masks,
            patient_ids=self.patient_ids,
            study_ids=self.study_ids,
            image_ids=self.image_ids,
            sample_ids=self.sample_ids,
            source_paths=self.source_paths,
            sample_hashes=self.sample_hashes,
            view_positions=self.view_positions,
        )

    def __len__(self) -> int:
        return math.ceil(len(self.records) / max(self.batch_size, 1))

    def build_pipeline(self, dali: dict[str, Any]) -> Any:
        errors: list[str] = []
        for mode in self.requested_modes:
            pipe = self.make_pipeline(dali, mode)
            try:
                pipe.build()
            except Exception as error:
                errors.append(f"{mode}: {type(error).__name__}: {error}")
                continue
            self.pipeline_mode = dali_pipeline_mode_name(mode)
            self.pipeline_warnings = errors
            return pipe
        self.pipeline_mode = "failed"
        self.pipeline_warnings = errors
        raise RuntimeError("DALI pipeline build failed for all decode modes: " + " | ".join(errors))

    def make_pipeline(self, dali: dict[str, Any], mode: str) -> Any:
        fn = dali["fn"]
        types = dali["types"]
        pipeline_def = dali["pipeline_def"]

        files = list(self.files)
        sample_indices = list(self.sample_indices)
        image_size = self.image_size
        mean = self.mean * 255.0
        std = self.std * 255.0
        shuffle = self.shuffle
        seed = self.seed

        @pipeline_def
        def cxr_file_pipeline() -> Any:
            encoded, index = fn.readers.file(
                files=files,
                labels=sample_indices,
                random_shuffle=shuffle,
                seed=seed,
                name="Reader",
            )
            if mode == "mixed":
                image = fn.decoders.image(encoded, device="mixed", output_type=types.GRAY)
            elif mode == "cpu_gpu":
                image = fn.decoders.image(encoded, device="cpu", output_type=types.GRAY)
                image = image.gpu()
            else:
                image = fn.decoders.image(encoded, device="cpu", output_type=types.GRAY)
            image = fn.resize(
                image,
                resize_x=image_size,
                resize_y=image_size,
                interp_type=types.INTERP_LINEAR,
            )
            image = fn.crop_mirror_normalize(
                image,
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=[mean],
                std=[std],
            )
            return image, index

        return cxr_file_pipeline(
            batch_size=self.batch_size,
            num_threads=self.workers,
            device_id=0,
            seed=seed,
        )

    def report_metadata(self) -> dict[str, Any]:
        return {
            "pipeline_mode": self.pipeline_mode,
            "warnings": self.pipeline_warnings,
            "image_read": "nvidia.dali.fn.readers.file",
            "requested_modes": self.requested_modes,
            "mode_notes": {
                "mixed": "GPU/nvJPEG decode plus GPU resize/normalize",
                "cpu_gpu": "CPU decode followed by GPU resize/normalize",
                "cpu": "CPU decode, resize, and normalize",
            },
        }


class DaliCxrIterator:
    def __init__(
        self,
        iterator: Any,
        *,
        labels: Any,
        masks: Any,
        patient_ids: Sequence[str],
        study_ids: Sequence[str],
        image_ids: Sequence[str],
        sample_ids: Sequence[str],
        source_paths: Sequence[str],
        sample_hashes: Sequence[str],
        view_positions: Sequence[str],
    ) -> None:
        self.iterator = iterator
        self.labels = labels
        self.masks = masks
        self.patient_ids = list(patient_ids)
        self.study_ids = list(study_ids)
        self.image_ids = list(image_ids)
        self.sample_ids = list(sample_ids)
        self.source_paths = list(source_paths)
        self.sample_hashes = list(sample_hashes)
        self.view_positions = list(view_positions)

    def __iter__(self) -> "DaliCxrIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        batch = next(self.iterator)
        if isinstance(batch, list):
            batch = batch[0]
        image = batch["image"]
        indices = batch["index"].detach().cpu().long().view(-1)
        labels = self.labels.index_select(0, indices)
        masks = self.masks.index_select(0, indices)
        index_list = [int(index) for index in indices.tolist()]
        return {
            "image": image,
            "labels": labels,
            "mask": masks,
            "patient_id": [self.patient_ids[index] for index in index_list],
            "study_id": [self.study_ids[index] for index in index_list],
            "image_id": [self.image_ids[index] for index in index_list],
            "sample_id": [self.sample_ids[index] for index in index_list],
            "source_path": [self.source_paths[index] for index in index_list],
            "sample_hash": [self.sample_hashes[index] for index in index_list],
            "view_position": [self.view_positions[index] for index in index_list],
        }


def make_dali_iterator(dali: dict[str, Any], pipe: Any) -> Any:
    iterator_type = dali["DALIGenericIterator"]
    last_batch_policy = dali.get("LastBatchPolicy")
    if last_batch_policy is not None:
        return iterator_type(
            [pipe],
            ["image", "index"],
            reader_name="Reader",
            auto_reset=False,
            last_batch_policy=last_batch_policy.PARTIAL,
        )
    return iterator_type(
        [pipe],
        ["image", "index"],
        size=-1,
        reader_name="Reader",
        auto_reset=False,
        fill_last_batch=False,
    )


def parse_dali_modes() -> list[str]:
    raw = os.environ.get("MEDKIT_DALI_MODES", "cpu")
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"mixed", "cpu_gpu", "cpu"}
    invalid = [mode for mode in modes if mode not in allowed]
    if invalid:
        raise RuntimeError(
            f"Invalid MEDKIT_DALI_MODES entries {invalid}; expected a comma-separated subset of {sorted(allowed)}"
        )
    return modes or ["cpu"]


def dali_pipeline_mode_name(mode: str) -> str:
    if mode == "mixed":
        return "dali_readers_file_mixed_gpu_decode_resize_normalize"
    if mode == "cpu_gpu":
        return "dali_readers_file_cpu_decode_gpu_resize_normalize"
    return "dali_readers_file_cpu_decode_resize_normalize"


def labels_to_arrays(record: SampleRecord, targets: Sequence[str]) -> tuple[Any, Any]:
    numpy = import_numpy()
    labels = numpy.zeros((len(targets),), dtype="float32")
    mask = numpy.zeros((len(targets),), dtype="float32")
    for index, target in enumerate(targets):
        value = record.labels.get(target)
        if value is None or value == -1:
            labels[index] = 0.0
            mask[index] = 0.0
        else:
            labels[index] = float(value)
            mask[index] = 1.0
    return labels, mask


def benchmark_loader(loader: Any, *, max_batches: int, baseline: str) -> dict[str, Any]:
    start = time.perf_counter()
    first_batch_seconds = None
    samples = 0
    checksum = 0.0
    batch_count = 0
    max_batch_tensor_bytes = 0
    for batch in loader:
        now = time.perf_counter()
        if first_batch_seconds is None:
            first_batch_seconds = now - start
        samples += int(batch["image"].shape[0])
        checksum += float(batch["labels"].sum().item())
        max_batch_tensor_bytes = max(max_batch_tensor_bytes, batch_tensor_bytes(batch))
        batch_count += 1
        if max_batches > 0 and batch_count >= max_batches:
            break
    elapsed = time.perf_counter() - start
    report = {
        "status": "ok",
        "baseline": baseline,
        "samples": samples,
        "batches": batch_count,
        "time_to_first_batch_ms": (first_batch_seconds or 0.0) * 1000.0,
        "iter_ms": elapsed * 1000.0,
        "samples_per_second": samples / max(elapsed, sys.float_info.epsilon),
        "peak_rss_mb": peak_rss_mb(),
        "batch_checksum": checksum,
    }
    if hasattr(loader, "report_metadata"):
        pipeline = loader.report_metadata()
        report["pipeline"] = pipeline
        report.update(
            native_prefetch_timing_fields(
                pipeline,
                batches=batch_count,
                elapsed_ms=elapsed * 1000.0,
                prefix="loader",
            )
        )
    report["memory"] = memory_snapshot(
        pipeline=report.get("pipeline"),
        max_batch_tensor_bytes=max_batch_tensor_bytes,
    )
    return report


def train_and_evaluate(
    *,
    torch: Any,
    baseline: str,
    train_loader: Any,
    val_loader: Any,
    train_records: Sequence[SampleRecord],
    eval_records: Sequence[SampleRecord],
    targets: Sequence[str],
    device: Any,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_eval_batches: int,
    drop_last_train: bool,
    warmup_batches: int,
    profile_batches: int,
    gpu_prefetch_batches: int,
    gpu_prefetch_reuse_buffers: bool,
    sync_every_step: bool,
    channels_last: bool,
    torch_compile: bool,
    torch_compile_mode: str,
    learning_rate: float,
    amp_dtype: str,
    model_init: str,
    loss_kind: str,
    loss_pos_weight_values: Sequence[float] | None,
    loss_pos_weight_mode: str,
    loss_pos_weight_cap: float,
    focal_gamma: float,
    focal_alpha: float,
    seed: int,
    prediction_artifact_path: Path | None,
    train_order_artifact_path: Path | None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    set_torch_seed(torch, seed)
    channels_last_active = bool(channels_last)
    model, model_info = make_model(torch, len(targets), model_init=model_init)
    if channels_last_active:
        model = model.to(device=device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    torch_compile_status = "disabled"
    torch_compile_setup_seconds = 0.0
    if torch_compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile requested but this PyTorch build does not expose torch.compile")
        compile_start = time.perf_counter()
        try:
            compile_mode = str(torch_compile_mode or "default")
            if compile_mode == "default":
                model = torch.compile(model)
            else:
                model = torch.compile(model, mode=compile_mode)
            torch_compile_status = "active"
        except Exception as error:
            torch_compile_setup_seconds = time.perf_counter() - compile_start
            raise RuntimeError(
                f"torch.compile failed with mode {torch_compile_mode!r}: {error}"
            ) from error
        torch_compile_setup_seconds = time.perf_counter() - compile_start
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    autocast_dtype = resolve_cuda_amp_dtype(torch, amp_dtype)
    autocast_enabled = device.type == "cuda" and amp_dtype != "disabled"
    grad_scaler_enabled = device.type == "cuda" and amp_dtype in ("auto", "float16")
    scaler = torch.cuda.amp.GradScaler(enabled=grad_scaler_enabled)
    pos_weight = (
        torch.tensor(list(loss_pos_weight_values), dtype=torch.float32, device=device)
        if loss_pos_weight_values is not None
        else None
    )
    gpu_prefetch_active = device.type == "cuda" and gpu_prefetch_batches > 0
    gpu_prefetch_reuse_buffers_active = gpu_prefetch_active and gpu_prefetch_reuse_buffers
    h2d_timing_mode = (
        H2D_TIMING_CUDA_PREFETCH_STREAM if gpu_prefetch_active else H2D_TIMING_DIRECT_COPY
    )
    sync_every_step_effective = sync_every_step or profile_batches > 0
    train_order_recorder = (
        TrainOrderRecorder(
            baseline=baseline,
            targets=targets,
            train_records=train_records,
            artifact_path=train_order_artifact_path,
            required=True,
        )
        if train_order_artifact_path is not None
        else None
    )
    warmup_seconds = 0.0
    if warmup_batches > 0:
        warmup_start = time.perf_counter()
        run_warmup_steps(
            torch=torch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            batches=warmup_batches,
            autocast_enabled=autocast_enabled,
            channels_last=channels_last_active,
            autocast_dtype=autocast_dtype,
            pos_weight=pos_weight,
            loss_kind=loss_kind,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            train_order_recorder=train_order_recorder,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        warmup_seconds = time.perf_counter() - warmup_start
    losses: list[float] = []
    deferred_losses: list[Any] = []
    data_wait_seconds = 0.0
    step_seconds = 0.0
    h2d_bytes = 0
    max_batch_tensor_bytes = 0
    profile_records: list[dict[str, Any]] = []
    samples = 0
    batches = 0
    skipped_incomplete_batches = 0
    skipped_incomplete_samples = 0
    gpu_prefetch_buffer_allocations = 0
    gpu_prefetch_buffer_copies = 0
    gpu_prefetch_buffer_shape_misses = 0
    channels_last_batches = 0
    channels_last_checked_batches = 0
    gpu_utilization_sampler = NvidiaSmiUtilizationSampler.for_device(device)
    train_start = time.perf_counter()
    gpu_utilization_sampler.start()
    model.train()
    for _epoch in range(epochs):
        iterator = iter(train_loader)
        epoch_batches = 0
        prefetcher = (
            CudaBatchPrefetcher(
                torch=torch,
                loader_iterator=iterator,
                device=device,
                batch_size=batch_size,
                drop_last_train=drop_last_train,
                depth=gpu_prefetch_batches,
                reuse_buffers=gpu_prefetch_reuse_buffers,
                channels_last=channels_last_active,
            )
            if gpu_prefetch_active
            else None
        )
        while True:
            prefetched: DevicePrefetchBatch | None = None
            if prefetcher is not None:
                prefetched = prefetcher.pop()
                if prefetched is None:
                    skipped_incomplete_batches += prefetcher.skipped_incomplete_batches
                    skipped_incomplete_samples += prefetcher.skipped_incomplete_samples
                    break
                batch = None
                batch_samples = prefetched.samples
                wait_seconds = prefetched.data_wait_seconds
                data_wait_seconds += wait_seconds
            else:
                wait_start = time.perf_counter()
                try:
                    batch = next(iterator)
                except StopIteration:
                    break
                wait_seconds = time.perf_counter() - wait_start
                data_wait_seconds += wait_seconds
                batch_samples = batch_sample_count(batch)
                if should_skip_incomplete_train_batch(
                    batch_samples=batch_samples,
                    batch_size=batch_size,
                    drop_last_train=drop_last_train,
                ):
                    skipped_incomplete_batches += 1
                    skipped_incomplete_samples += batch_samples
                    continue
            if train_order_recorder is not None:
                train_order_recorder.record_batch(
                    phase="train",
                    epoch=_epoch,
                    batch_index=epoch_batches,
                    global_batch_index=batches,
                    batch=prefetched if prefetched is not None else batch,
                    sample_count=batch_samples,
                )
            step_start = time.perf_counter()
            profile_this_batch = profile_batches > 0 and len(profile_records) < profile_batches
            cuda_profile = profile_this_batch and device.type == "cuda"
            h2d_start = h2d_end = None
            batch_prepare_start = batch_prepare_end = None
            forward_start = forward_end = None
            backward_start = backward_end = None
            optimizer_start = optimizer_end = None

            batch_prepare_wall_start = time.perf_counter()
            if cuda_profile:
                batch_prepare_start = torch.cuda.Event(enable_timing=True)
                batch_prepare_end = torch.cuda.Event(enable_timing=True)
                batch_prepare_start.record()
            if prefetched is not None:
                max_batch_tensor_bytes = max(max_batch_tensor_bytes, prefetched.tensor_bytes)
                torch.cuda.current_stream(device).wait_event(prefetched.ready_event)
                image = prefetched.image.float()
                labels = prefetched.labels.float()
                mask = prefetched.mask.float()
                h2d_bytes += prefetched.h2d_bytes
                h2d_wall_ms = 0.0
            else:
                max_batch_tensor_bytes = max(max_batch_tensor_bytes, batch_tensor_bytes(batch))
                if cuda_profile:
                    h2d_start = torch.cuda.Event(enable_timing=True)
                    h2d_end = torch.cuda.Event(enable_timing=True)
                    h2d_start.record()
                image = image_to_float_on_device(
                    torch,
                    batch["image"],
                    device,
                    channels_last=channels_last_active,
                )
                labels = batch["labels"].to(device, non_blocking=True).float()
                mask = batch["mask"].to(device, non_blocking=True).float()
                if cuda_profile:
                    h2d_end.record()
                h2d_bytes += (
                    batch["image"].numel() * 4
                    + batch["labels"].numel() * 4
                    + batch["mask"].numel() * 4
                )
            if channels_last_active:
                channels_last_checked_batches += 1
                if image.is_contiguous(memory_format=torch.channels_last):
                    channels_last_batches += 1
            if cuda_profile:
                batch_prepare_end.record()
            batch_prepare_wall_ms = (time.perf_counter() - batch_prepare_wall_start) * 1000.0
            if prefetched is None:
                h2d_wall_ms = batch_prepare_wall_ms
            zero_grad_wall_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            zero_grad_wall_ms = (time.perf_counter() - zero_grad_wall_start) * 1000.0
            forward_wall_start = time.perf_counter()
            if cuda_profile:
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
            with torch.cuda.amp.autocast(
                **autocast_kwargs(enabled=autocast_enabled, dtype=autocast_dtype)
            ):
                logits = model(image)
                loss = masked_multilabel_loss(
                    torch=torch,
                    logits=logits,
                    labels=labels,
                    mask=mask,
                    pos_weight=pos_weight,
                    loss_kind=loss_kind,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
            if cuda_profile:
                forward_end.record()
            forward_wall_ms = (time.perf_counter() - forward_wall_start) * 1000.0
            backward_wall_start = time.perf_counter()
            if cuda_profile:
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_end = torch.cuda.Event(enable_timing=True)
                backward_start.record()
            scaler.scale(loss).backward()
            if cuda_profile:
                backward_end.record()
            backward_wall_ms = (time.perf_counter() - backward_wall_start) * 1000.0
            optimizer_wall_start = time.perf_counter()
            if cuda_profile:
                optimizer_start = torch.cuda.Event(enable_timing=True)
                optimizer_end = torch.cuda.Event(enable_timing=True)
                optimizer_start.record()
            scaler.step(optimizer)
            scaler.update()
            if cuda_profile:
                optimizer_end.record()
            optimizer_wall_ms = (time.perf_counter() - optimizer_wall_start) * 1000.0
            prefetch_maintenance_wall_ms = 0.0
            if prefetcher is not None:
                prefetch_maintenance_start = time.perf_counter()
                prefetcher.release(prefetched)
                prefetcher.fill()
                prefetch_maintenance_wall_ms = (
                    time.perf_counter() - prefetch_maintenance_start
                ) * 1000.0
            if device.type == "cuda" and sync_every_step_effective:
                torch.cuda.synchronize(device)
            step_elapsed = time.perf_counter() - step_start
            step_seconds += step_elapsed
            detached_loss = loss.detach()
            if device.type == "cuda" and not sync_every_step_effective:
                deferred_losses.append(detached_loss)
            else:
                losses.append(float(detached_loss.cpu().item()))
            samples += batch_samples
            if profile_this_batch:
                if cuda_profile:
                    h2d_ms = (
                        float(prefetched.h2d_start.elapsed_time(prefetched.h2d_end))
                        if prefetched is not None
                        else float(h2d_start.elapsed_time(h2d_end))
                    )
                    batch_prepare_ms = float(
                        batch_prepare_start.elapsed_time(batch_prepare_end)
                    )
                    forward_ms = float(forward_start.elapsed_time(forward_end))
                    backward_ms = float(backward_start.elapsed_time(backward_end))
                    optimizer_ms = float(optimizer_start.elapsed_time(optimizer_end))
                else:
                    h2d_ms = h2d_wall_ms
                    batch_prepare_ms = batch_prepare_wall_ms
                    forward_ms = forward_wall_ms
                    backward_ms = backward_wall_ms
                    optimizer_ms = optimizer_wall_ms
                accounted_step_ms = (
                    batch_prepare_ms
                    + zero_grad_wall_ms
                    + forward_ms
                    + backward_ms
                    + optimizer_ms
                    + prefetch_maintenance_wall_ms
                )
                step_elapsed_ms = step_elapsed * 1000.0
                residual_step_ms_signed = step_elapsed_ms - accounted_step_ms
                profile_records.append(
                    {
                        "batch_index": batches,
                        "samples": batch_samples,
                        "data_wait_ms": wait_seconds * 1000.0,
                        "h2d_ms": h2d_ms,
                        "h2d_timing_mode": h2d_timing_mode,
                        "batch_prepare_ms": batch_prepare_ms,
                        "batch_prepare_wall_ms": batch_prepare_wall_ms,
                        "zero_grad_wall_ms": zero_grad_wall_ms,
                        "forward_ms": forward_ms,
                        "backward_ms": backward_ms,
                        "optimizer_ms": optimizer_ms,
                        "prefetch_maintenance_wall_ms": prefetch_maintenance_wall_ms,
                        "total_step_ms": step_elapsed_ms,
                        "accounted_step_ms": accounted_step_ms,
                        "residual_step_ms": max(0.0, residual_step_ms_signed),
                        "residual_step_ms_signed": residual_step_ms_signed,
                        "residual_step_percent": (
                            100.0 * residual_step_ms_signed / step_elapsed_ms
                            if step_elapsed_ms > 0.0
                            else 0.0
                        ),
                        "timing_scope": (
                            "mixed_cuda_events_and_wall"
                            if cuda_profile
                            else "wall_clock_only"
                        ),
                        "sync_every_step_effective": sync_every_step_effective,
                    }
                )
            batches += 1
            epoch_batches += 1
            if max_train_batches > 0 and batches >= max_train_batches:
                if prefetcher is not None:
                    skipped_incomplete_batches += prefetcher.skipped_incomplete_batches
                    skipped_incomplete_samples += prefetcher.skipped_incomplete_samples
                break
        if prefetcher is not None:
            stats = prefetcher.stats()
            gpu_prefetch_buffer_allocations += stats["buffer_allocations"]
            gpu_prefetch_buffer_copies += stats["buffer_copies"]
            gpu_prefetch_buffer_shape_misses += stats["buffer_shape_misses"]
        if max_train_batches > 0 and batches >= max_train_batches:
            break
    if device.type == "cuda" and not sync_every_step_effective:
        torch.cuda.synchronize(device)
        losses.extend(float(loss.cpu().item()) for loss in deferred_losses)
    total_seconds = time.perf_counter() - train_start
    gpu_utilization = gpu_utilization_sampler.stop()
    evaluation = evaluate_model(
        torch=torch,
        model=model,
        loader=val_loader,
        device=device,
        max_batches=max_eval_batches,
        channels_last=channels_last_active,
        fallback_records=eval_records,
        targets=targets,
    )
    validate_eval_arrays(
        y_true=evaluation.y_true,
        y_score=evaluation.y_score,
        y_mask=evaluation.y_mask,
        y_logits=evaluation.y_logits,
        targets=targets,
    )
    quality = metric_report(evaluation.y_true, evaluation.y_score, evaluation.y_mask, targets)
    quality["status"] = "ok"
    quality["baseline"] = baseline
    if evaluation.localization is not None:
        evaluation.localization.setdefault("baseline", baseline)
        quality["localization"] = evaluation.localization
    thresholds = threshold_report(evaluation.y_true, evaluation.y_score, evaluation.y_mask, targets)
    thresholds["status"] = "ok"
    thresholds["baseline"] = baseline
    if prediction_artifact_path is not None:
        prediction_report = write_eval_predictions_artifact(
            path=prediction_artifact_path,
            baseline=baseline,
            targets=targets,
            evaluation=evaluation,
            quality=quality,
        )
        quality["prediction_capture"] = {
            "enabled": True,
            "status": prediction_report.get("status"),
            "artifact_path": prediction_report.get("artifact_path"),
        }
        quality["prediction_hashes"] = prediction_report.get("hashes", {})
        quality["prediction_artifact_sha256"] = prediction_report.get("artifact_sha256")
        quality["metric_recompute"] = prediction_report.get("metric_recompute")
        quality["metric_recompute_matches_predictions"] = prediction_report.get(
            "metric_recompute_matches_quality"
        )
    else:
        prediction_report = prediction_capture_disabled_report(
            baseline=baseline,
            reason="eval prediction capture disabled",
        )
        quality["prediction_capture"] = {
            "enabled": False,
            "status": "disabled",
            "reason": prediction_report.get("reason"),
        }
    train_report = {
        "status": "ok",
        "baseline": baseline,
        "device": str(device),
        "samples": samples,
        "batches": batches,
        "epochs_requested": epochs,
        "train_batch_size": batch_size,
        "drop_last_train": drop_last_train,
        "gpu_prefetch_batches": gpu_prefetch_batches,
        "gpu_prefetch_batches_active": gpu_prefetch_active,
        "gpu_prefetch_reuse_buffers": gpu_prefetch_reuse_buffers,
        "gpu_prefetch_reuse_buffers_active": gpu_prefetch_reuse_buffers_active,
        "gpu_prefetch_buffer_allocations": gpu_prefetch_buffer_allocations,
        "gpu_prefetch_buffer_copies": gpu_prefetch_buffer_copies,
        "gpu_prefetch_buffer_shape_misses": gpu_prefetch_buffer_shape_misses,
        "sync_every_step": sync_every_step,
        "sync_every_step_effective": sync_every_step_effective,
        "channels_last_requested": channels_last,
        "channels_last_active": channels_last_active,
        "channels_last_checked_batches": channels_last_checked_batches,
        "channels_last_batches": channels_last_batches,
        "channels_last_all_checked_batches": (
            channels_last_batches == channels_last_checked_batches
            if channels_last_checked_batches
            else None
        ),
        "torch_compile_requested": torch_compile,
        "torch_compile_mode": str(torch_compile_mode or "default"),
        "torch_compile_status": torch_compile_status,
        "torch_compile_setup_ms": torch_compile_setup_seconds * 1000.0,
        "learning_rate": learning_rate,
        "amp_dtype": amp_dtype,
        "amp_torch_dtype": str(autocast_dtype) if autocast_dtype is not None else None,
        "autocast_enabled": autocast_enabled,
        "grad_scaler_enabled": grad_scaler_enabled,
        "warmup_ms": warmup_seconds * 1000.0,
        "model_init": model_init,
        "model_architecture": model_info.get("architecture"),
        "model_pretrained": model_info.get("pretrained"),
        "model_pretrained_weights": model_info.get("pretrained_weights"),
        "loss_kind": loss_kind,
        "loss_pos_weight": loss_pos_weight_mode,
        "loss_pos_weight_cap": loss_pos_weight_cap,
        "loss_pos_weight_values": (
            [float(value) for value in loss_pos_weight_values]
            if loss_pos_weight_values is not None
            else None
        ),
        "focal_gamma": focal_gamma,
        "focal_alpha": focal_alpha,
        "h2d_timing_mode": h2d_timing_mode,
        "skipped_incomplete_batches": skipped_incomplete_batches,
        "skipped_incomplete_samples": skipped_incomplete_samples,
        "train_ms": total_seconds * 1000.0,
        "samples_per_second": samples / max(total_seconds, sys.float_info.epsilon),
        "data_wait_percent": 100.0 * data_wait_seconds / max(total_seconds, sys.float_info.epsilon),
        "model_step_ms": step_seconds * 1000.0,
        "loss_mean": sum(losses) / max(len(losses), 1),
        "loss_last": losses[-1] if losses else None,
        "cuda_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else None
        ),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_utilization": gpu_utilization,
        "h2d_gb_per_second_estimate": (h2d_bytes / (1024.0**3))
        / max(total_seconds, sys.float_info.epsilon),
    }
    if train_order_recorder is not None:
        train_order_report = train_order_recorder.write()
        train_report["train_order_capture"] = {
            "enabled": True,
            "status": train_order_report.get("status"),
            "artifact_path": train_order_report.get("artifact_path"),
            "hashes": train_order_report.get("hashes", {}),
        }
    else:
        train_order_report = train_order_capture_disabled_report(
            baseline=baseline,
            reason="train order evidence disabled",
        )
        train_report["train_order_capture"] = {
            "enabled": False,
            "status": "disabled",
            "reason": train_order_report.get("reason"),
        }
    profile_report = profile_report_for_baseline(
        baseline=baseline,
        requested_batches=profile_batches,
        records=profile_records,
        h2d_timing_mode=h2d_timing_mode,
    )
    if profile_report.get("status") == "ok":
        train_report.update(profile_report["summary"])
    if hasattr(train_loader, "report_metadata"):
        pipeline = train_loader.report_metadata()
        train_report["pipeline"] = pipeline
        train_report.update(
            native_prefetch_timing_fields(
                pipeline,
                batches=batches,
                elapsed_ms=train_report["train_ms"],
                prefix="train",
            )
        )
    train_report["memory"] = memory_snapshot(
        pipeline=train_report.get("pipeline"),
        max_batch_tensor_bytes=max_batch_tensor_bytes,
    )
    return train_report, quality, thresholds, profile_report, prediction_report, train_order_report


class NvidiaSmiUtilizationSampler:
    QUERY_FIELDS = (
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
    )

    def __init__(
        self,
        *,
        enabled: bool,
        device_index: int | None,
        interval_seconds: float = 1.0,
        disabled_reason: str = "",
    ) -> None:
        self.enabled = enabled
        self.device_index = device_index
        self.interval_seconds = max(float(interval_seconds), 0.1)
        self.disabled_reason = disabled_reason
        self._samples: list[dict[str, Any]] = []
        self._errors: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._stopped_at: float | None = None

    @classmethod
    def for_device(cls, device: Any) -> "NvidiaSmiUtilizationSampler":
        if getattr(device, "type", None) != "cuda":
            return cls(
                enabled=False,
                device_index=None,
                disabled_reason=f"device is not cuda: {device}",
            )
        if shutil.which("nvidia-smi") is None:
            return cls(
                enabled=False,
                device_index=cuda_device_index(device),
                disabled_reason="nvidia-smi not found",
            )
        interval = gpu_utilization_interval_seconds()
        return cls(enabled=True, device_index=cuda_device_index(device), interval_seconds=interval)

    def start(self) -> "NvidiaSmiUtilizationSampler":
        self._started_at = time.perf_counter()
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stopped_at = time.perf_counter()
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(2.0, self.interval_seconds + 1.0))
        with self._lock:
            samples = list(self._samples)
            errors = list(self._errors)
        return gpu_utilization_summary(
            samples=samples,
            errors=errors,
            enabled=self.enabled,
            device_index=self.device_index,
            interval_seconds=self.interval_seconds,
            disabled_reason=self.disabled_reason,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                sample = query_nvidia_smi_utilization(self.device_index)
                with self._lock:
                    self._samples.append(sample)
            except Exception as error:
                with self._lock:
                    if len(self._errors) < 5:
                        self._errors.append(f"{type(error).__name__}: {error}")
            self._stop_event.wait(self.interval_seconds)


def cuda_device_index(device: Any) -> int:
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    match = re.search(r"cuda:(\d+)", str(device))
    if match:
        return int(match.group(1))
    return 0


def gpu_utilization_interval_seconds() -> float:
    raw = os.environ.get("MEDKIT_GPU_UTIL_INTERVAL_SECONDS", "1.0")
    try:
        value = float(raw)
    except ValueError:
        return 1.0
    return min(max(value, 0.1), 10.0)


def query_nvidia_smi_utilization(device_index: int | None) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(NvidiaSmiUtilizationSampler.QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    ]
    if device_index is not None:
        command.extend(["--id", str(device_index)])
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=2.0,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout).strip())
    line = next(
        (item.strip() for item in completed.stdout.splitlines() if item.strip()),
        "",
    )
    if not line:
        raise RuntimeError("nvidia-smi returned no utilization rows")
    values = [part.strip() for part in line.split(",")]
    if len(values) < len(NvidiaSmiUtilizationSampler.QUERY_FIELDS):
        raise RuntimeError(f"unexpected nvidia-smi utilization row: {line!r}")
    return parse_nvidia_smi_utilization_values(values)


def parse_nvidia_smi_utilization_values(values: Sequence[str]) -> dict[str, Any]:
    return {
        "gpu_utilization_percent": parse_float_or_none(values[0]),
        "memory_utilization_percent": parse_float_or_none(values[1]),
        "memory_used_mb": parse_float_or_none(values[2]),
        "memory_total_mb": parse_float_or_none(values[3]),
        "power_draw_w": parse_float_or_none(values[4]),
    }


def parse_float_or_none(value: Any) -> float | None:
    text = str(value).strip()
    if text.upper() in {"", "N/A", "[N/A]", "NA"}:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def gpu_utilization_summary(
    *,
    samples: Sequence[dict[str, Any]],
    errors: Sequence[str],
    enabled: bool,
    device_index: int | None,
    interval_seconds: float,
    disabled_reason: str,
    started_at: float | None,
    stopped_at: float | None,
) -> dict[str, Any]:
    duration = (
        stopped_at - started_at
        if started_at is not None and stopped_at is not None
        else None
    )
    status = "ok" if enabled and samples else "unavailable" if enabled else "disabled"
    summary = {
        "schema_version": 1,
        "status": status,
        "enabled": enabled,
        "device_index": device_index,
        "sample_count": len(samples),
        "interval_seconds": interval_seconds,
        "duration_seconds": duration,
        "query": list(NvidiaSmiUtilizationSampler.QUERY_FIELDS),
        "gpu_utilization_percent": numeric_distribution(
            numeric_sample_values(samples, "gpu_utilization_percent")
        ),
        "memory_utilization_percent": numeric_distribution(
            numeric_sample_values(samples, "memory_utilization_percent")
        ),
        "memory_used_mb": numeric_distribution(numeric_sample_values(samples, "memory_used_mb")),
        "memory_total_mb": numeric_distribution(numeric_sample_values(samples, "memory_total_mb")),
        "power_draw_w": numeric_distribution(numeric_sample_values(samples, "power_draw_w")),
    }
    if disabled_reason:
        summary["reason"] = disabled_reason
    if errors:
        summary["errors"] = list(errors)
    return summary


def numeric_sample_values(samples: Sequence[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def native_prefetch_timing_fields(
    pipeline: dict[str, Any] | None,
    *,
    batches: int,
    elapsed_ms: float,
    prefix: str,
) -> dict[str, Any]:
    if not isinstance(pipeline, dict):
        return {}
    stats = pipeline.get("native_prefetch_stats")
    if not isinstance(stats, dict) or not stats:
        return {}
    stats_batches = _stats_float(stats, "batches")
    indexed_batches = _stats_float(stats, "indexed_batches")
    indexed_runs = _stats_float(stats, "indexed_runs")
    read_ms = _stats_float(stats, "read_micros") / 1000.0
    scatter_ms = _stats_float(stats, "scatter_micros") / 1000.0
    read_scatter_ms = read_ms + scatter_ms
    denominator = indexed_batches or stats_batches or float(batches)
    slot_count = _stats_float(stats, "slot_count")
    preallocated_batch_buffers = _stats_float(stats, "preallocated_batch_buffers")
    output: dict[str, Any] = {
        f"{prefix}_native_prefetch_batches": stats_batches,
        f"{prefix}_native_prefetch_indexed_batches": indexed_batches,
        f"{prefix}_native_prefetch_indexed_runs": indexed_runs,
        f"{prefix}_native_prefetch_read_ms": read_ms,
        f"{prefix}_native_prefetch_scatter_ms": scatter_ms,
        f"{prefix}_native_prefetch_read_scatter_ms": read_scatter_ms,
        f"{prefix}_native_prefetch_slot_count": slot_count,
        f"{prefix}_native_prefetch_preallocated_batch_buffers": preallocated_batch_buffers,
        f"{prefix}_native_prefetch_buffer_reuse_enabled": bool(
            stats.get("buffer_reuse_enabled")
        ),
        f"{prefix}_native_prefetch_pin_memory": bool(stats.get("pin_memory")),
    }
    if denominator > 0.0:
        output[f"{prefix}_native_prefetch_runs_per_batch"] = indexed_runs / denominator
        output[f"{prefix}_native_prefetch_read_ms_per_batch"] = read_ms / denominator
        output[f"{prefix}_native_prefetch_scatter_ms_per_batch"] = scatter_ms / denominator
        output[f"{prefix}_native_prefetch_read_scatter_ms_per_batch"] = (
            read_scatter_ms / denominator
        )
    if elapsed_ms > 0.0:
        output[f"{prefix}_native_prefetch_read_scatter_percent"] = (
            100.0 * read_scatter_ms / elapsed_ms
        )
    return output


def _stats_float(stats: dict[str, Any], field: str) -> float:
    value = stats.get(field, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def batch_sample_count(batch: dict[str, Any]) -> int:
    return int(batch["image"].shape[0])


def should_skip_incomplete_train_batch(
    *,
    batch_samples: int,
    batch_size: int,
    drop_last_train: bool,
) -> bool:
    return drop_last_train and batch_size > 0 and batch_samples != batch_size


def class_pos_weight_values(
    records: Sequence[SampleRecord],
    targets: Sequence[str],
    cap: float | None = None,
) -> list[float]:
    train_records = [record for record in records if record.split == "train"]
    weights: list[float] = []
    for target in targets:
        positive = 0
        negative = 0
        for record in train_records:
            value = record.labels.get(target)
            if value == 1:
                positive += 1
            elif value == 0:
                negative += 1
        weight = float(negative / positive) if positive > 0 else 1.0
        if cap is not None and cap > 0.0:
            weight = min(weight, float(cap))
        weights.append(weight)
    return weights


class CudaBatchPrefetcher:
    def __init__(
        self,
        *,
        torch: Any,
        loader_iterator: Any,
        device: Any,
        batch_size: int,
        drop_last_train: bool,
        depth: int,
        reuse_buffers: bool = False,
        channels_last: bool = False,
    ):
        self.torch = torch
        self.loader_iterator = loader_iterator
        self.device = device
        self.batch_size = batch_size
        self.drop_last_train = drop_last_train
        self.depth = max(1, depth)
        self.reuse_buffers = reuse_buffers
        self.channels_last = channels_last
        self.copy_stream = torch.cuda.Stream(device=device)
        self.queue: list[DevicePrefetchBatch] = []
        self.buffer_slots: list[dict[str, Any] | None] = [None] * self.depth
        self.free_slots = list(range(self.depth)) if self.reuse_buffers else []
        self.release_events: list[Any | None] = [None] * self.depth
        self.buffer_allocations = 0
        self.buffer_copies = 0
        self.buffer_shape_misses = 0
        self.exhausted = False
        self.skipped_incomplete_batches = 0
        self.skipped_incomplete_samples = 0
        self.fill()

    def fill(self) -> None:
        while len(self.queue) < self.depth and not self.exhausted:
            if self.reuse_buffers and not self.free_slots:
                break
            wait_start = time.perf_counter()
            try:
                cpu_batch = next(self.loader_iterator)
            except StopIteration:
                self.exhausted = True
                break
            wait_seconds = time.perf_counter() - wait_start
            samples = batch_sample_count(cpu_batch)
            if should_skip_incomplete_train_batch(
                batch_samples=samples,
                batch_size=self.batch_size,
                drop_last_train=self.drop_last_train,
            ):
                self.skipped_incomplete_batches += 1
                self.skipped_incomplete_samples += samples
                continue

            slot_index = self.free_slots.pop(0) if self.reuse_buffers else None
            h2d_start = self.torch.cuda.Event(enable_timing=True)
            h2d_end = self.torch.cuda.Event(enable_timing=True)
            with self.torch.cuda.stream(self.copy_stream):
                if slot_index is not None and self.release_events[slot_index] is not None:
                    self.copy_stream.wait_event(self.release_events[slot_index])
                    self.release_events[slot_index] = None
                h2d_start.record()
                if slot_index is None:
                    image = image_to_float_on_device(
                        self.torch,
                        cpu_batch["image"],
                        self.device,
                        channels_last=self.channels_last,
                    )
                    labels = cpu_batch["labels"].to(self.device, non_blocking=True)
                    mask = cpu_batch["mask"].to(self.device, non_blocking=True)
                else:
                    image = self._copy_to_slot(slot_index, "image", cpu_batch["image"])
                    labels = self._copy_to_slot(slot_index, "labels", cpu_batch["labels"])
                    mask = self._copy_to_slot(slot_index, "mask", cpu_batch["mask"])
                h2d_end.record()
            self.queue.append(
                DevicePrefetchBatch(
                    image=image,
                    labels=labels,
                    mask=mask,
                    samples=samples,
                    tensor_bytes=batch_tensor_bytes(cpu_batch),
                    h2d_bytes=(
                        cpu_batch["image"].numel() * 4
                        + cpu_batch["labels"].numel() * 4
                        + cpu_batch["mask"].numel() * 4
                    ),
                    h2d_start=h2d_start,
                    h2d_end=h2d_end,
                    ready_event=h2d_end,
                    data_wait_seconds=wait_seconds,
                    slot_index=slot_index,
                    sample_ids=[
                        str(value)
                        for value in batch_field_values(cpu_batch, "sample_id", samples)
                        if value not in (None, "")
                    ],
                )
            )

    def pop(self) -> DevicePrefetchBatch | None:
        if not self.queue:
            self.fill()
        if not self.queue:
            return None
        return self.queue.pop(0)

    def release(self, batch: DevicePrefetchBatch | None) -> None:
        if batch is None or batch.slot_index is None:
            return
        release_event = self.torch.cuda.Event(enable_timing=False)
        release_event.record(self.torch.cuda.current_stream(self.device))
        self.release_events[batch.slot_index] = release_event
        self.free_slots.append(batch.slot_index)

    def stats(self) -> dict[str, int]:
        return {
            "buffer_allocations": self.buffer_allocations,
            "buffer_copies": self.buffer_copies,
            "buffer_shape_misses": self.buffer_shape_misses,
        }

    def _copy_to_slot(self, slot_index: int, key: str, source: Any) -> Any:
        slot = self.buffer_slots[slot_index]
        if slot is None:
            slot = {}
            self.buffer_slots[slot_index] = slot
        target = slot.get(key)
        if target is None:
            target = self._empty_slot_tensor(key, source)
            slot[key] = target
            self.buffer_allocations += 1
        elif (
            tuple(target.shape) != tuple(source.shape)
            or target.dtype != source.dtype
        ):
            target = self._empty_slot_tensor(key, source)
            slot[key] = target
            self.buffer_allocations += 1
            self.buffer_shape_misses += 1
        target.copy_(source, non_blocking=True)
        self.buffer_copies += 1
        return target

    def _empty_slot_tensor(self, key: str, source: Any) -> Any:
        if key == "image" and self.channels_last and getattr(source, "dim", lambda: 0)() == 4:
            return self.torch.empty_like(
                source,
                device=self.device,
                memory_format=self.torch.channels_last,
            )
        return self.torch.empty_like(source, device=self.device)


def profile_report_for_baseline(
    *,
    baseline: str,
    requested_batches: int,
    records: list[dict[str, Any]],
    h2d_timing_mode: str = H2D_TIMING_DIRECT_COPY,
) -> dict[str, Any]:
    if requested_batches <= 0:
        return {
            "status": "disabled",
            "baseline": baseline,
            "requested_batches": requested_batches,
            "records": [],
            "summary": {},
    }
    summary = summarize_profile_records(records)
    summary["profile_artifact_path"] = "step-profile.json"
    summary.setdefault("profile_h2d_timing_mode", h2d_timing_mode)
    return {
        "status": "ok" if records else "failed",
        "baseline": baseline,
        "requested_batches": requested_batches,
        "records": records,
        "summary": summary,
    }


def summarize_profile_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    profiled_samples = sum(int(record.get("samples", 0)) for record in records)
    data_wait_total_ms = sum(float(record.get("data_wait_ms", 0.0)) for record in records)
    total_step_ms = sum(float(record.get("total_step_ms", 0.0)) for record in records)
    end_to_end_ms = data_wait_total_ms + total_step_ms
    summary: dict[str, Any] = {
        "profiled_batches": len(records),
        "profiled_samples": profiled_samples,
        "profile_data_wait_total_ms": data_wait_total_ms,
        "profile_total_step_ms": total_step_ms,
        "profile_train_samples_per_s": (
            1000.0 * profiled_samples / total_step_ms if total_step_ms > 0.0 else 0.0
        ),
        "profile_end_to_end_ms": end_to_end_ms,
        "profile_end_to_end_samples_per_s": (
            1000.0 * profiled_samples / end_to_end_ms if end_to_end_ms > 0.0 else 0.0
        ),
    }
    required_profile_fields = {
        "data_wait_ms",
        "h2d_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    }
    for field in (
        "data_wait_ms",
        "h2d_ms",
        "batch_prepare_ms",
        "batch_prepare_wall_ms",
        "zero_grad_wall_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "prefetch_maintenance_wall_ms",
        "accounted_step_ms",
        "residual_step_ms",
        "residual_step_ms_signed",
        "residual_step_percent",
        "total_step_ms",
    ):
        if field not in required_profile_fields and not any(field in record for record in records):
            continue
        values = [float(record.get(field, 0.0)) for record in records]
        stats = profile_stats(values)
        summary[f"profile_{field}_mean"] = stats["mean"]
        summary[f"profile_{field}_p50"] = stats["p50"]
        summary[f"profile_{field}_p95"] = stats["p95"]
    phase_fields = (
        "data_wait_ms",
        "batch_prepare_ms",
        "zero_grad_wall_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
        "prefetch_maintenance_wall_ms",
        "residual_step_ms_signed",
    )
    phase_totals = {
        field: sum(float(record.get(field, 0.0)) for record in records)
        for field in phase_fields
        if any(field in record for record in records)
    }
    if records and phase_totals:
        summary["profile_phase_budget_ms_per_batch"] = {
            field: total / len(records) for field, total in phase_totals.items()
        }
        summary["profile_phase_budget_end_to_end_percent"] = {
            field: 100.0 * total / end_to_end_ms if end_to_end_ms > 0.0 else 0.0
            for field, total in phase_totals.items()
        }
        accounted_step_total_ms = (
            sum(float(record.get("accounted_step_ms", 0.0)) for record in records)
            if any("accounted_step_ms" in record for record in records)
            else sum(
                total
                for field, total in phase_totals.items()
                if field not in {"data_wait_ms", "residual_step_ms_signed"}
            )
        )
        summary["profile_step_accounted_percent"] = (
            100.0 * accounted_step_total_ms / total_step_ms if total_step_ms > 0.0 else 0.0
        )
        residual_signed_total_ms = phase_totals.get("residual_step_ms_signed")
        if residual_signed_total_ms is not None:
            summary["profile_residual_step_signed_total_ms"] = residual_signed_total_ms
            summary["profile_residual_step_signed_percent"] = (
                100.0 * residual_signed_total_ms / total_step_ms
                if total_step_ms > 0.0
                else 0.0
            )
            summary["profile_step_reconciled_percent"] = (
                100.0 * (accounted_step_total_ms + residual_signed_total_ms) / total_step_ms
                if total_step_ms > 0.0
                else 0.0
            )
    h2d_modes = sorted(
        {
            str(record["h2d_timing_mode"])
            for record in records
            if record.get("h2d_timing_mode")
        }
    )
    if len(h2d_modes) == 1:
        summary["profile_h2d_timing_mode"] = h2d_modes[0]
    elif len(h2d_modes) > 1:
        summary["profile_h2d_timing_mode"] = "mixed"
    return summary


def profile_stats(values: list[float]) -> dict[str, float]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(finite_values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile_nearest_rank(ordered, 50.0),
        "p95": percentile_nearest_rank(ordered, 95.0),
    }


def percentile_nearest_rank(ordered_values: list[float], percentile: float) -> float:
    if not ordered_values:
        return 0.0
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered_values)))
    return ordered_values[min(rank, len(ordered_values)) - 1]


def run_warmup_steps(
    *,
    torch: Any,
    model: Any,
    optimizer: Any,
    scaler: Any,
    loader: Any,
    device: Any,
    batches: int,
    autocast_enabled: bool,
    channels_last: bool,
    autocast_dtype: Any | None = None,
    pos_weight: Any | None = None,
    loss_kind: str = "bce",
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.0,
    train_order_recorder: TrainOrderRecorder | None = None,
) -> None:
    model.train()
    for batch_index, batch in enumerate(loader):
        if train_order_recorder is not None:
            train_order_recorder.record_batch(
                phase="warmup",
                epoch=None,
                batch_index=batch_index,
                global_batch_index=None,
                batch=batch,
                sample_count=batch_sample_count(batch),
            )
        image = image_to_float_on_device(
            torch,
            batch["image"],
            device,
            channels_last=channels_last,
        )
        labels = batch["labels"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(
            **autocast_kwargs(enabled=autocast_enabled, dtype=autocast_dtype)
        ):
            logits = model(image)
            loss = masked_multilabel_loss(
                torch=torch,
                logits=logits,
                labels=labels,
                mask=mask,
                pos_weight=pos_weight,
                loss_kind=loss_kind,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_index + 1 >= batches:
            break


def resolve_cuda_amp_dtype(torch: Any, amp_dtype: str) -> Any | None:
    if amp_dtype == "auto" or amp_dtype == "disabled":
        return None
    if amp_dtype == "float16":
        return torch.float16
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype!r}")


def autocast_kwargs(*, enabled: bool, dtype: Any | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"enabled": enabled}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def image_to_float_on_device(torch: Any, image: Any, device: Any, *, channels_last: bool) -> Any:
    if channels_last and getattr(image, "dim", lambda: 0)() == 4:
        return image.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
    return image.to(device=device, dtype=torch.float32, non_blocking=True)


def masked_multilabel_loss(
    *,
    torch: Any,
    logits: Any,
    labels: Any,
    mask: Any,
    pos_weight: Any | None,
    loss_kind: str,
    focal_gamma: float,
    focal_alpha: float,
) -> Any:
    raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        reduction="none",
        pos_weight=pos_weight,
    )
    if loss_kind == "focal":
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * labels + (1.0 - probabilities) * (1.0 - labels)
        raw_loss = raw_loss * (1.0 - p_t).clamp_min(1.0e-6).pow(float(focal_gamma))
        if focal_alpha > 0.0:
            alpha_t = focal_alpha * labels + (1.0 - focal_alpha) * (1.0 - labels)
            raw_loss = raw_loss * alpha_t
    elif loss_kind != "bce":
        raise ValueError(f"Unsupported loss kind: {loss_kind!r}")
    return (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)


def make_model(torch: Any, num_targets: int, *, model_init: str = "random") -> tuple[Any, dict[str, Any]]:
    try:
        torchvision = import_torchvision()
        weights = None
        weights_name = None
        if model_init == "imagenet":
            weights_enum = getattr(torchvision.models, "DenseNet121_Weights", None)
            weights = weights_enum.DEFAULT if weights_enum is not None else "IMAGENET1K_V1"
            weights_name = str(weights)
        elif model_init != "random":
            raise ValueError(f"Unsupported model init: {model_init!r}")
        model = torchvision.models.densenet121(weights=weights)
        conv0 = torch.nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if weights is not None:
            with torch.no_grad():
                conv0.weight.copy_(model.features.conv0.weight.mean(dim=1, keepdim=True))
        model.features.conv0 = conv0
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_targets)
        return model, {
            "architecture": "torchvision.densenet121",
            "model_init": model_init,
            "pretrained": weights is not None,
            "pretrained_weights": weights_name,
        }
    except Exception:
        if model_init != "random":
            raise
        return torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_targets),
        ), {
            "architecture": "fallback_small_cnn",
            "model_init": model_init,
            "pretrained": False,
            "pretrained_weights": None,
        }


def evaluate_model(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int,
    channels_last: bool,
    fallback_records: Sequence[SampleRecord] = (),
    targets: Sequence[str] = (),
) -> EvaluationOutputs:
    numpy = import_numpy()
    model.eval()
    y_true: list[Any] = []
    y_score: list[Any] = []
    y_mask: list[Any] = []
    y_logits: list[Any] = []
    samples: list[dict[str, Any]] = []
    eval_offset = 0
    localization_recorder = CamLocalizationRecorder(
        target="Pneumonia" if "Pneumonia" in targets else "",
        target_index=list(targets).index("Pneumonia") if "Pneumonia" in targets else None,
        eval_records=fallback_records,
    )
    localization_supported = False
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            image = image_to_float_on_device(
                torch,
                batch["image"],
                device,
                channels_last=channels_last,
            )
            cam_result = (
                dense_classifier_logits_and_cam(
                    torch=torch,
                    model=model,
                    image=image,
                    target_index=localization_recorder.target_index,
                )
                if localization_recorder.enabled
                else None
            )
            if cam_result is None:
                logits = model(image)
            else:
                logits, heatmaps = cam_result
                localization_supported = True
                localization_recorder.record_batch(
                    heatmaps=tensor_to_numpy(heatmaps),
                    start_index=eval_offset,
                    count=int(logits.shape[0]),
                )
            logits_np = logits.detach().cpu().numpy()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_score.append(probs)
            y_logits.append(logits_np)
            y_true.append(tensor_to_numpy(batch["labels"]))
            y_mask.append(tensor_to_numpy(batch["mask"]))
            batch_samples = int(probs.shape[0])
            samples.extend(
                prediction_sample_rows(
                    batch=batch,
                    fallback_records=fallback_records,
                    start_index=eval_offset,
                    count=batch_samples,
                )
            )
            eval_offset += batch_samples
            if max_batches > 0 and batch_index + 1 >= max_batches:
                break
    if not y_true:
        empty = numpy.zeros((0, 0), dtype="float32")
        return EvaluationOutputs(
            y_true=empty,
            y_score=empty,
            y_mask=empty,
            y_logits=empty,
            samples=[],
            localization=localization_recorder.report(localization_supported),
        )
    return EvaluationOutputs(
        y_true=numpy.concatenate(y_true),
        y_score=numpy.concatenate(y_score),
        y_mask=numpy.concatenate(y_mask),
        y_logits=numpy.concatenate(y_logits),
        samples=samples,
        localization=localization_recorder.report(localization_supported),
    )


class CamLocalizationRecorder:
    def __init__(
        self,
        *,
        target: str,
        target_index: int | None,
        eval_records: Sequence[SampleRecord],
    ) -> None:
        self.target = target
        self.target_index = target_index
        self.eval_records = list(eval_records)
        self.enabled = (
            target_index is not None
            and bool(target)
            and any(record.localization_boxes for record in eval_records)
        )
        self.rows: list[dict[str, Any]] = []

    def record_batch(self, *, heatmaps: Any, start_index: int, count: int) -> None:
        if not self.enabled:
            return
        numpy = import_numpy()
        heatmaps_np = numpy.asarray(heatmaps, dtype="float64")
        if heatmaps_np.ndim != 3:
            return
        for local_index in range(min(count, heatmaps_np.shape[0])):
            global_index = start_index + local_index
            if global_index >= len(self.eval_records):
                continue
            record = self.eval_records[global_index]
            if record.labels.get(self.target) != 1 or not record.localization_boxes:
                continue
            heatmap = normalize_heatmap(heatmaps_np[local_index])
            box_mask = box_union_mask(
                boxes=record.localization_boxes or [],
                height=int(heatmap.shape[0]),
                width=int(heatmap.shape[1]),
            )
            if box_mask is None or not bool(box_mask.any()):
                continue
            row = cam_localization_sample_metrics(
                heatmap=heatmap,
                box_mask=box_mask,
                sample_id=record.sample_id,
                box_count=len(record.localization_boxes or []),
            )
            self.rows.append(row)

    def report(self, supported: bool) -> dict[str, Any]:
        if not self.enabled:
            return localization_eval_disabled_report(
                baseline="",
                reason="no localization boxes for requested target",
                target=self.target,
            )
        if not supported:
            return localization_eval_disabled_report(
                baseline="",
                reason="model does not expose DenseNet-style classifier CAM features",
                target=self.target,
            )
        return summarize_cam_localization_rows(target=self.target, rows=self.rows)


def dense_classifier_logits_and_cam(
    *,
    torch: Any,
    model: Any,
    image: Any,
    target_index: int | None,
) -> tuple[Any, Any] | None:
    if target_index is None:
        return None
    base_model = getattr(model, "_orig_mod", model)
    features_module = getattr(base_model, "features", None)
    classifier = getattr(base_model, "classifier", None)
    if features_module is None or classifier is None or not hasattr(classifier, "weight"):
        return None
    features = features_module(image)
    activations = torch.nn.functional.relu(features, inplace=False)
    pooled = torch.nn.functional.adaptive_avg_pool2d(activations, (1, 1))
    logits = classifier(torch.flatten(pooled, 1))
    weight = classifier.weight[target_index].reshape(1, -1, 1, 1)
    heatmaps = (activations * weight).sum(dim=1)
    heatmaps = torch.nn.functional.relu(heatmaps, inplace=False)
    heatmaps = torch.nn.functional.interpolate(
        heatmaps.unsqueeze(1),
        size=tuple(image.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return logits, heatmaps


def normalize_heatmap(heatmap: Any) -> Any:
    numpy = import_numpy()
    array = numpy.asarray(heatmap, dtype="float64")
    finite = numpy.isfinite(array)
    if not bool(finite.any()):
        return numpy.zeros_like(array, dtype="float64")
    clean = numpy.where(finite, array, 0.0)
    minimum = float(clean[finite].min())
    maximum = float(clean[finite].max())
    if maximum <= minimum:
        return numpy.zeros_like(clean, dtype="float64")
    return (clean - minimum) / (maximum - minimum)


def box_union_mask(*, boxes: Sequence[dict[str, Any]], height: int, width: int) -> Any | None:
    if height <= 0 or width <= 0:
        return None
    numpy = import_numpy()
    mask = numpy.zeros((height, width), dtype=bool)
    for box in boxes:
        try:
            image_width = float(box.get("image_width") or 1.0)
            image_height = float(box.get("image_height") or 1.0)
            x1 = float(box.get("x1"))
            y1 = float(box.get("y1"))
            x2 = float(box.get("x2"))
            y2 = float(box.get("y2"))
        except (TypeError, ValueError):
            continue
        if image_width <= 0.0 or image_height <= 0.0:
            continue
        left = max(0, min(width - 1, int(math.floor((x1 / image_width) * width))))
        top = max(0, min(height - 1, int(math.floor((y1 / image_height) * height))))
        right = max(left + 1, min(width, int(math.ceil((x2 / image_width) * width))))
        bottom = max(top + 1, min(height, int(math.ceil((y2 / image_height) * height))))
        mask[top:bottom, left:right] = True
    return mask


def cam_localization_sample_metrics(
    *,
    heatmap: Any,
    box_mask: Any,
    sample_id: str,
    box_count: int,
) -> dict[str, Any]:
    numpy = import_numpy()
    flat_index = int(numpy.argmax(heatmap))
    top_y, top_x = numpy.unravel_index(flat_index, heatmap.shape)
    total_heat = float(heatmap.sum())
    box_heat = float(heatmap[box_mask].sum())
    row: dict[str, Any] = {
        "sample_id": sample_id,
        "box_count": int(box_count),
        "top1_hit": bool(box_mask[int(top_y), int(top_x)]),
        "heat_in_box_fraction": (box_heat / total_heat if total_heat > 0.0 else None),
    }
    for fraction in (0.01, 0.05, 0.10, 0.20):
        pred_mask = top_fraction_mask(heatmap, fraction)
        intersection = int(numpy.logical_and(pred_mask, box_mask).sum())
        union = int(numpy.logical_or(pred_mask, box_mask).sum())
        box_area = int(box_mask.sum())
        key = f"top_{int(fraction * 100)}pct"
        row[key] = {
            "hit": intersection > 0,
            "iou": (intersection / union if union else None),
            "box_coverage": (intersection / box_area if box_area else None),
            "predicted_area_fraction": float(pred_mask.mean()),
        }
    return row


def top_fraction_mask(heatmap: Any, fraction: float) -> Any:
    numpy = import_numpy()
    array = numpy.asarray(heatmap, dtype="float64")
    total = int(array.size)
    k = max(1, min(total, int(math.ceil(total * fraction))))
    flat = array.reshape(-1)
    if k >= total:
        return numpy.ones_like(array, dtype=bool)
    indices = numpy.argpartition(flat, total - k)[total - k :]
    mask = numpy.zeros(total, dtype=bool)
    mask[indices] = True
    return mask.reshape(array.shape)


def summarize_cam_localization_rows(*, target: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ok" if rows else "not_available",
        "method": "densenet121_classifier_cam",
        "target": target,
        "evaluated_positive_boxed_samples": len(rows),
        "top1_hit_rate": mean_boolean(row.get("top1_hit") for row in rows),
        "heat_in_box_fraction": numeric_distribution(
            [value for value in (row.get("heat_in_box_fraction") for row in rows) if value is not None]
        ),
        "top_percent": {
            key: summarize_cam_threshold_rows(rows, key)
            for key in ("top_1pct", "top_5pct", "top_10pct", "top_20pct")
        },
        "notes": [
            "Classifier CAM localization is a weakly supervised box-overlap metric.",
            "It is not detector mAP; use it to compare recipes before adding a detection head.",
        ],
    }


def summarize_cam_threshold_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row.get(key) for row in rows if isinstance(row.get(key), dict)]
    return {
        "hit_rate": mean_boolean(value.get("hit") for value in values),
        "iou": numeric_distribution(
            [value.get("iou") for value in values if value.get("iou") is not None]
        ),
        "box_coverage": numeric_distribution(
            [
                value.get("box_coverage")
                for value in values
                if value.get("box_coverage") is not None
            ]
        ),
        "predicted_area_fraction": numeric_distribution(
            [
                value.get("predicted_area_fraction")
                for value in values
                if value.get("predicted_area_fraction") is not None
            ]
        ),
    }


def mean_boolean(values: Iterable[Any]) -> float | None:
    clean = [bool(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(1 for value in clean if value) / len(clean)


def localization_eval_disabled_report(
    *,
    baseline: str,
    reason: str,
    target: str = "Pneumonia",
) -> dict[str, Any]:
    report = {
        "schema_version": 1,
        "status": "disabled",
        "enabled": False,
        "target": target,
        "reason": reason,
    }
    if baseline:
        report["baseline"] = baseline
    return report


def tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "numpy"):
        return value.numpy().copy()
    return import_numpy().asarray(value).copy()


def prediction_sample_rows(
    *,
    batch: dict[str, Any],
    fallback_records: Sequence[SampleRecord],
    start_index: int,
    count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = {
        "sample_id": batch_field_values(batch, "sample_id", count),
        "patient_id": batch_field_values(batch, "patient_id", count),
        "study_id": batch_field_values(batch, "study_id", count),
        "image_id": batch_field_values(batch, "image_id", count),
        "source_path": batch_field_values(batch, "source_path", count),
        "sample_hash": batch_field_values(batch, "sample_hash", count),
    }
    for local_index in range(count):
        global_index = start_index + local_index
        fallback = (
            sample_metadata_from_record(fallback_records[global_index])
            if global_index < len(fallback_records)
            else {}
        )
        row: dict[str, Any] = {"eval_index": global_index}
        for field, values in fields.items():
            value = values[local_index] if local_index < len(values) else None
            if value is None or value == "":
                value = fallback.get(field)
            row[field] = "" if value is None else str(value)
        fallback_sample_id = fallback.get("sample_id")
        if (
            fallback_sample_id
            and fields["sample_id"][local_index] not in (None, "", fallback_sample_id)
        ):
            raise ValueError(
                "eval sample order mismatch: "
                f"batch sample_id {fields['sample_id'][local_index]!r} "
                f"!= fallback {fallback_sample_id!r} at eval index {global_index}"
            )
        rows.append(row)
    return rows


def batch_field_values(batch: dict[str, Any], key: str, count: int) -> list[Any]:
    aliases = {
        "source_path": ("source_path", "image_path"),
        "sample_hash": ("sample_hash", "sha256"),
    }.get(key, (key,))
    candidates: list[Any] = []
    for alias in aliases:
        if alias in batch:
            candidates.append(batch.get(alias))
    metadata = batch.get("metadata")
    if isinstance(metadata, dict):
        for alias in aliases:
            if alias in metadata:
                candidates.append(metadata.get(alias))
    for candidate in candidates:
        values = value_to_list(candidate, count)
        if any(value not in (None, "") for value in values):
            return values
    return [None for _ in range(count)]


def value_to_list(value: Any, count: int) -> list[Any]:
    if value is None:
        return [None for _ in range(count)]
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return [value for _ in range(count)]
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        values = [
            item.decode("utf-8") if isinstance(item, bytes) else item
            for item in value
        ]
        if len(values) < count:
            values.extend([None] * (count - len(values)))
        return values[:count]
    return [value for _ in range(count)]


class TrainOrderRecorder:
    def __init__(
        self,
        *,
        baseline: str,
        targets: Sequence[str],
        train_records: Sequence[SampleRecord],
        artifact_path: Path | None,
        required: bool,
    ) -> None:
        self.baseline = baseline
        self.targets = list(targets)
        self.train_records = list(train_records)
        self.artifact_path = artifact_path
        self.required = required
        self.rows: list[dict[str, Any]] = []

    def record_batch(
        self,
        *,
        phase: str,
        epoch: int | None,
        batch_index: int,
        global_batch_index: int | None,
        batch: Any,
        sample_count: int | None = None,
    ) -> None:
        count = int(sample_count if sample_count is not None else train_batch_sample_count(batch))
        sample_ids = train_batch_sample_ids(batch, count)
        missing_ids = [index for index, value in enumerate(sample_ids) if value in (None, "")]
        if self.required and missing_ids:
            preview = ", ".join(str(index) for index in missing_ids[:5])
            raise ValueError(
                f"train order evidence requires sample_id metadata; missing {len(missing_ids)} "
                f"sample ids in {phase} batch {batch_index} at positions {preview}"
            )
        labels, masks = train_batch_label_arrays(batch, self.targets)
        label_sums, valid_label_sums = train_batch_label_sums(labels, masks, self.targets)
        clean_ids = ["" if value is None else str(value) for value in sample_ids]
        self.rows.append(
            {
                "schema_version": 1,
                "baseline": self.baseline,
                "phase": phase,
                "epoch": epoch,
                "batch_index": int(batch_index),
                "global_batch_index": (
                    int(global_batch_index) if global_batch_index is not None else None
                ),
                "sample_count": count,
                "sample_ids": clean_ids,
                "sample_order_hash": stable_hash(clean_ids),
                "target_names": list(self.targets),
                "label_sums": label_sums,
                "valid_label_sums": valid_label_sums,
            }
        )

    def write(self) -> dict[str, Any]:
        if self.artifact_path is None:
            return train_order_capture_disabled_report(
                baseline=self.baseline,
                reason="train order evidence disabled",
            )
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.artifact_path, "wt", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        return summarize_train_order_rows(
            baseline=self.baseline,
            targets=self.targets,
            train_records=self.train_records,
            rows=self.rows,
            artifact_path=self.artifact_path,
        )


def train_batch_sample_count(batch: Any) -> int:
    if isinstance(batch, DevicePrefetchBatch):
        return int(batch.samples)
    return batch_sample_count(batch)


def train_batch_sample_ids(batch: Any, count: int) -> list[Any]:
    if isinstance(batch, DevicePrefetchBatch):
        values = list(batch.sample_ids or [])
        if len(values) < count:
            values.extend([None] * (count - len(values)))
        return values[:count]
    return batch_field_values(batch, "sample_id", count)


def train_batch_label_arrays(batch: Any, targets: Sequence[str]) -> tuple[Any, Any]:
    numpy = import_numpy()
    if isinstance(batch, DevicePrefetchBatch):
        labels = tensor_to_numpy(batch.labels)
        masks = tensor_to_numpy(batch.mask)
    else:
        labels = tensor_to_numpy(batch["labels"])
        masks = tensor_to_numpy(batch["mask"])
    labels = numpy.asarray(labels, dtype="float64")
    masks = numpy.asarray(masks, dtype="float64")
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    if masks.ndim == 1:
        masks = masks.reshape(1, -1)
    if labels.ndim != 2 or masks.ndim != 2:
        raise ValueError(
            f"train order labels/mask must be 2D, got {labels.shape} and {masks.shape}"
        )
    if labels.shape != masks.shape or labels.shape[1] != len(targets):
        raise ValueError(
            "train order labels/mask target width mismatch: "
            f"labels={labels.shape}, mask={masks.shape}, targets={len(targets)}"
        )
    if not numpy.isfinite(labels).all() or not numpy.isfinite(masks).all():
        raise ValueError("train order labels/mask contain non-finite values")
    return labels, masks


def train_batch_label_sums(
    labels: Any,
    masks: Any,
    targets: Sequence[str],
) -> tuple[dict[str, float], dict[str, float]]:
    positive = (labels * (masks > 0.0)).sum(axis=0)
    valid = (masks > 0.0).sum(axis=0)
    return (
        {target: float(positive[index]) for index, target in enumerate(targets)},
        {target: float(valid[index]) for index, target in enumerate(targets)},
    )


def summarize_train_order_rows(
    *,
    baseline: str,
    targets: Sequence[str],
    train_records: Sequence[SampleRecord],
    rows: Sequence[dict[str, Any]],
    artifact_path: Path,
) -> dict[str, Any]:
    train_rows = [row for row in rows if row.get("phase") == "train"]
    warmup_rows = [row for row in rows if row.get("phase") == "warmup"]
    train_sample_ids = flatten_sample_ids(train_rows)
    warmup_sample_ids = flatten_sample_ids(warmup_rows)
    epoch_summaries = train_order_epoch_summaries(
        rows=train_rows,
        train_records=train_records,
        targets=targets,
    )
    train_epoch_order_hashes = [
        row.get("sample_order_hash")
        for row in epoch_summaries
        if row.get("sample_order_hash") is not None
    ]
    dropped_by_epoch = [
        {
            "epoch": row.get("epoch"),
            "dropped_sample_ids": row.get("dropped_sample_ids", []),
        }
        for row in epoch_summaries
    ]
    return {
        "status": "ok",
        "enabled": True,
        "baseline": baseline,
        "artifact_path": artifact_path.name,
        "artifact_sha256": hash_file(artifact_path),
        "target_names": list(targets),
        "train_universe_samples": len(train_records),
        "train_universe_targets": label_counts_for_records(train_records, targets),
        "warmup_batches": len(warmup_rows),
        "warmup_samples": len(warmup_sample_ids),
        "train_batches": len(train_rows),
        "train_samples": len(train_sample_ids),
        "epoch_summaries": epoch_summaries,
        "same_train_order_each_epoch": (
            len(set(train_epoch_order_hashes)) <= 1 if train_epoch_order_hashes else None
        ),
        "hashes": {
            "warmup_sample_order_hash": stable_hash(warmup_sample_ids),
            "warmup_batch_order_hash": stable_hash(
                [row.get("sample_order_hash") for row in warmup_rows]
            ),
            "train_sample_order_hash": stable_hash(train_sample_ids),
            "train_sample_multiset_hash": stable_hash(sorted(train_sample_ids)),
            "train_batch_order_hash": stable_hash(
                [row.get("sample_order_hash") for row in train_rows]
            ),
            "train_epoch_order_hashes": train_epoch_order_hashes,
            "dropped_samples_by_epoch_hash": stable_hash(dropped_by_epoch),
            "batch_label_sums_hash": stable_hash(
                [
                    {
                        "phase": row.get("phase"),
                        "epoch": row.get("epoch"),
                        "batch_index": row.get("batch_index"),
                        "sample_order_hash": row.get("sample_order_hash"),
                        "label_sums": row.get("label_sums"),
                        "valid_label_sums": row.get("valid_label_sums"),
                    }
                    for row in rows
                ]
            ),
        },
    }


def flatten_sample_ids(rows: Sequence[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for row in rows:
        ids.extend(str(value) for value in row.get("sample_ids", []))
    return ids


def train_order_epoch_summaries(
    *,
    rows: Sequence[dict[str, Any]],
    train_records: Sequence[SampleRecord],
    targets: Sequence[str],
) -> list[dict[str, Any]]:
    expected_ids = [record.sample_id for record in train_records]
    expected_set = set(expected_ids)
    record_by_id = {record.sample_id: record for record in train_records}
    epochs = sorted(
        {
            int(row["epoch"])
            for row in rows
            if row.get("epoch") is not None
        }
    )
    summaries: list[dict[str, Any]] = []
    for epoch in epochs:
        epoch_rows = [row for row in rows if row.get("epoch") == epoch]
        sample_ids = flatten_sample_ids(epoch_rows)
        observed_set = set(sample_ids)
        dropped = sorted(expected_set - observed_set)
        duplicate_ids = sorted(
            sample_id for sample_id in observed_set if sample_ids.count(sample_id) > 1
        )
        dropped_records = [record_by_id[sample_id] for sample_id in dropped if sample_id in record_by_id]
        summaries.append(
            {
                "epoch": epoch,
                "batches": len(epoch_rows),
                "samples": len(sample_ids),
                "unique_samples": len(observed_set),
                "sample_order_hash": stable_hash(sample_ids),
                "sample_set_hash": stable_hash(sorted(sample_ids)),
                "dropped_sample_count": len(dropped),
                "dropped_sample_ids": dropped,
                "dropped_target_counts": label_counts_for_records(dropped_records, targets),
                "duplicate_sample_count": len(duplicate_ids),
                "duplicate_sample_ids": duplicate_ids,
                "label_sums": sum_target_dicts(row.get("label_sums") for row in epoch_rows),
                "valid_label_sums": sum_target_dicts(
                    row.get("valid_label_sums") for row in epoch_rows
                ),
            }
        )
    return summaries


def sum_target_dicts(values: Iterable[Any]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for value in values:
        if not isinstance(value, dict):
            continue
        for key, raw in value.items():
            totals[str(key)] = totals.get(str(key), 0.0) + float(raw or 0.0)
    return totals


def train_order_capture_disabled_report(*, baseline: str, reason: str) -> dict[str, Any]:
    return {
        "status": "disabled",
        "enabled": False,
        "baseline": baseline,
        "reason": reason,
    }


def read_train_order_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        return [json.loads(line) for line in handle if line.strip()]


def train_order_summary_report(
    *,
    report_dir: Path,
    train_order: dict[str, Any],
    targets: Sequence[str],
    capture_enabled: bool,
) -> dict[str, Any]:
    baselines = {
        baseline: normalize_train_order_summary(
            report_dir=report_dir,
            summary=summary,
            targets=targets,
        )
        for baseline, summary in train_order.items()
    }
    return {
        "schema_version": 1,
        "enabled": capture_enabled,
        "baselines": baselines,
        "paired_comparisons": paired_train_order_comparisons(baselines),
    }


def normalize_train_order_summary(
    *,
    report_dir: Path,
    summary: dict[str, Any],
    targets: Sequence[str],
) -> dict[str, Any]:
    if summary.get("status") != "ok":
        return summary
    artifact_path = report_dir / str(summary.get("artifact_path", ""))
    if not artifact_path.exists():
        normalized = dict(summary)
        normalized["status"] = "failed"
        normalized["reason"] = f"train order artifact missing: {artifact_path.name}"
        return normalized
    rows = read_train_order_rows(artifact_path)
    target_orders = {tuple(row.get("target_names") or []) for row in rows}
    if target_orders and target_orders != {tuple(targets)}:
        normalized = dict(summary)
        normalized["status"] = "failed"
        normalized["reason"] = "train order artifact target order mismatch"
        return normalized
    normalized = dict(summary)
    normalized["artifact_rows"] = len(rows)
    normalized["artifact_recheck"] = {
        "train_batches": len([row for row in rows if row.get("phase") == "train"]),
        "warmup_batches": len([row for row in rows if row.get("phase") == "warmup"]),
        "train_sample_order_hash": stable_hash(
            flatten_sample_ids([row for row in rows if row.get("phase") == "train"])
        ),
    }
    normalized["artifact_recheck_matches_summary"] = (
        (normalized.get("hashes") or {}).get("train_sample_order_hash")
        == normalized["artifact_recheck"]["train_sample_order_hash"]
    )
    return normalized


def paired_train_order_comparisons(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = baselines.get("pytorch_raw")
    if not isinstance(raw, dict) or raw.get("status") != "ok":
        return {}
    comparisons: dict[str, Any] = {}
    for baseline, summary in baselines.items():
        if baseline == "pytorch_raw" or summary.get("status") != "ok":
            continue
        comparisons[f"{baseline}:vs:pytorch_raw"] = paired_train_order_summary(
            candidate=summary,
            raw=raw,
        )
    return comparisons


def paired_train_order_summary(
    *,
    candidate: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    candidate_hashes = candidate.get("hashes") or {}
    raw_hashes = raw.get("hashes") or {}
    candidate_epochs = candidate.get("epoch_summaries") or []
    raw_epochs = raw.get("epoch_summaries") or []
    return {
        "paired": (
            candidate_hashes.get("train_sample_order_hash")
            == raw_hashes.get("train_sample_order_hash")
            and candidate_hashes.get("train_sample_multiset_hash")
            == raw_hashes.get("train_sample_multiset_hash")
            and candidate_hashes.get("dropped_samples_by_epoch_hash")
            == raw_hashes.get("dropped_samples_by_epoch_hash")
        ),
        "identical_train_order": (
            candidate_hashes.get("train_sample_order_hash")
            == raw_hashes.get("train_sample_order_hash")
        ),
        "identical_train_sample_multiset": (
            candidate_hashes.get("train_sample_multiset_hash")
            == raw_hashes.get("train_sample_multiset_hash")
        ),
        "identical_dropped_samples_by_epoch": (
            candidate_hashes.get("dropped_samples_by_epoch_hash")
            == raw_hashes.get("dropped_samples_by_epoch_hash")
        ),
        "identical_batch_label_sums": (
            candidate_hashes.get("batch_label_sums_hash")
            == raw_hashes.get("batch_label_sums_hash")
        ),
        "candidate_same_train_order_each_epoch": candidate.get("same_train_order_each_epoch"),
        "raw_same_train_order_each_epoch": raw.get("same_train_order_each_epoch"),
        "candidate_train_batches": candidate.get("train_batches"),
        "raw_train_batches": raw.get("train_batches"),
        "candidate_train_samples": candidate.get("train_samples"),
        "raw_train_samples": raw.get("train_samples"),
        "epoch_deltas": paired_train_order_epoch_deltas(
            candidate_epochs=candidate_epochs,
            raw_epochs=raw_epochs,
        ),
    }


def paired_train_order_epoch_deltas(
    *,
    candidate_epochs: Sequence[dict[str, Any]],
    raw_epochs: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw_by_epoch = {int(row.get("epoch")): row for row in raw_epochs if row.get("epoch") is not None}
    deltas: dict[str, Any] = {}
    for candidate_row in candidate_epochs:
        if candidate_row.get("epoch") is None:
            continue
        epoch = int(candidate_row.get("epoch"))
        raw_row = raw_by_epoch.get(epoch, {})
        candidate_dropped = set(str(value) for value in candidate_row.get("dropped_sample_ids", []))
        raw_dropped = set(str(value) for value in raw_row.get("dropped_sample_ids", []))
        deltas[str(epoch)] = {
            "identical_order": candidate_row.get("sample_order_hash")
            == raw_row.get("sample_order_hash"),
            "candidate_dropped_sample_count": candidate_row.get("dropped_sample_count"),
            "raw_dropped_sample_count": raw_row.get("dropped_sample_count"),
            "shared_dropped_sample_count": len(candidate_dropped & raw_dropped),
            "candidate_only_dropped_sample_count": len(candidate_dropped - raw_dropped),
            "raw_only_dropped_sample_count": len(raw_dropped - candidate_dropped),
            "candidate_dropped_target_counts": candidate_row.get("dropped_target_counts"),
            "raw_dropped_target_counts": raw_row.get("dropped_target_counts"),
        }
    return deltas


def sample_metadata_from_record(record: SampleRecord) -> dict[str, str]:
    return {
        "sample_id": record.sample_id,
        "patient_id": record.patient_id,
        "study_id": record.study_id,
        "image_id": record.image_id,
        "source_path": record.image_path,
        "sample_hash": record.sha256,
    }


def validate_eval_arrays(
    *,
    y_true: Any,
    y_score: Any,
    y_mask: Any,
    y_logits: Any,
    targets: Sequence[str],
) -> None:
    numpy = import_numpy()
    expected_width = len(targets)
    for name, array in {
        "labels": y_true,
        "probabilities": y_score,
        "label_mask": y_mask,
        "logits": y_logits,
    }.items():
        parsed = numpy.asarray(array)
        if parsed.ndim != 2:
            raise ValueError(f"eval {name} must be a 2D array, got shape {parsed.shape}")
        if parsed.shape[1] != expected_width:
            raise ValueError(
                f"eval {name} target width {parsed.shape[1]} != expected {expected_width}"
            )
        if not numpy.isfinite(parsed).all():
            raise ValueError(f"eval {name} contains non-finite values")


def metric_report(y_true: Any, y_score: Any, y_mask: Any, targets: Sequence[str]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"targets": {}, "samples": int(y_true.shape[0])}
    roc_values: list[float] = []
    pr_values: list[float] = []
    sklearn_metrics = import_sklearn_metrics()
    for index, target in enumerate(targets):
        valid = y_mask[:, index] > 0.0
        positives = int(y_true[valid, index].sum()) if valid.any() else 0
        total = int(valid.sum())
        negatives = total - positives
        target_report = {
            "valid_samples": total,
            "positives": positives,
            "negatives": negatives,
            "prevalence": positives / total if total else None,
        }
        if total > 0 and positives > 0 and negatives > 0:
            auroc = float(
                sklearn_metrics.roc_auc_score(y_true[valid, index], y_score[valid, index])
            )
            auprc = float(
                sklearn_metrics.average_precision_score(
                    y_true[valid, index],
                    y_score[valid, index],
                )
            )
            target_report["auroc"] = auroc
            target_report["auprc"] = auprc
            roc_values.append(auroc)
            pr_values.append(auprc)
        else:
            target_report["auroc"] = None
            target_report["auprc"] = None
            target_report["metric_reason"] = "requires at least one positive and one negative"
        metrics["targets"][target] = target_report
    metrics["macro_auroc"] = sum(roc_values) / len(roc_values) if roc_values else None
    metrics["macro_auprc"] = sum(pr_values) / len(pr_values) if pr_values else None
    metrics["metric_target_count"] = len(roc_values)
    return metrics


def quality_gate_report(
    *,
    quality: dict[str, dict[str, Any]],
    train_order: dict[str, Any],
    validation: dict[str, Any],
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    enabled = bool(run_metadata.get("quality_gate"))
    min_eval_samples = int(run_metadata.get("quality_min_eval_samples") or 0)
    min_metric_targets = int(run_metadata.get("quality_min_metric_targets") or 0)
    min_macro_auroc = float(run_metadata.get("quality_min_macro_auroc") or 0.0)
    min_macro_auprc = float(run_metadata.get("quality_min_macro_auprc") or 0.0)
    paired_train_order_required = bool(run_metadata.get("paired_train_order"))
    split_audit = validation.get("split_audit") or {}
    split_checks = {
        "patient_overlap_count": int(split_audit.get("patient_overlap_count") or 0),
        "study_overlap_count": int(split_audit.get("study_overlap_count") or 0),
        "duplicate_hash_overlap_count": int(
            split_audit.get("duplicate_hash_overlap_count") or 0
        ),
    }
    for field, count in split_checks.items():
        if count:
            errors.append(f"{field} must be zero, got {count}")

    baseline_checks: dict[str, Any] = {}
    for baseline, report in quality.items():
        if report.get("status") != "ok":
            errors.append(f"{baseline} quality status is {report.get('status')!r}")
            baseline_checks[baseline] = {"status": report.get("status"), "reason": report.get("reason")}
            continue
        samples = int(report.get("samples") or 0)
        metric_target_count = int(report.get("metric_target_count") or 0)
        macro_auroc = numeric_report_value(report.get("macro_auroc"))
        macro_auprc = numeric_report_value(report.get("macro_auprc"))
        check = {
            "samples": samples,
            "metric_target_count": metric_target_count,
            "macro_auroc": macro_auroc,
            "macro_auprc": macro_auprc,
            "prediction_capture": report.get("prediction_capture"),
            "metric_recompute_matches_predictions": report.get(
                "metric_recompute_matches_predictions"
            ),
        }
        baseline_checks[baseline] = check
        prediction_capture = report.get("prediction_capture") or {}
        if enabled and prediction_capture.get("enabled") is not True:
            errors.append(f"{baseline} prediction capture is not enabled")
        if enabled and prediction_capture.get("status") != "ok":
            errors.append(
                f"{baseline} prediction capture status is {prediction_capture.get('status')!r}"
            )
        if enabled and report.get("metric_recompute_matches_predictions") is not True:
            errors.append(f"{baseline} prediction artifact metric recompute did not match")
        if enabled or min_eval_samples > 0:
            if samples < min_eval_samples:
                errors.append(
                    f"{baseline} eval samples {samples} < required {min_eval_samples}"
                )
        if enabled or min_metric_targets > 0:
            if metric_target_count < min_metric_targets:
                errors.append(
                    f"{baseline} metric targets {metric_target_count} < required {min_metric_targets}"
                )
        if min_macro_auroc > 0.0:
            if macro_auroc is None or macro_auroc < min_macro_auroc:
                errors.append(
                    f"{baseline} macro AUROC {macro_auroc} < required {min_macro_auroc}"
                )
        if min_macro_auprc > 0.0:
            if macro_auprc is None or macro_auprc < min_macro_auprc:
                errors.append(
                    f"{baseline} macro AUPRC {macro_auprc} < required {min_macro_auprc}"
                )
        if not enabled and min_eval_samples <= 0 and min_metric_targets <= 0:
            warnings.append(
                f"{baseline} quality metrics recorded without enforcing coverage thresholds"
            )

    train_order_checks: dict[str, Any] = {}
    if paired_train_order_required and len(quality) > 1:
        paired_comparisons = train_order.get("paired_comparisons") or {}
        raw_present = "pytorch_raw" in quality
        if not raw_present:
            errors.append("paired train order requires pytorch_raw baseline")
        for baseline in quality:
            if baseline == "pytorch_raw":
                continue
            key = f"{baseline}:vs:pytorch_raw"
            comparison = paired_comparisons.get(key)
            train_order_checks[key] = comparison
            if not isinstance(comparison, dict):
                errors.append(f"{key} train order pairing comparison missing")
                continue
            if comparison.get("paired") is not True:
                errors.append(f"{key} train order is not paired")
    elif paired_train_order_required:
        warnings.append(
            "paired train order is enabled for this single-baseline row; "
            "batch-level matrix audit must compare it against pytorch_raw"
        )

    return {
        "status": "failed" if errors else "ok" if enabled else "recorded",
        "enabled": enabled,
        "errors": errors,
        "warnings": warnings,
        "requirements": {
            "min_eval_samples": min_eval_samples,
            "min_metric_targets": min_metric_targets,
            "min_macro_auroc": min_macro_auroc,
            "min_macro_auprc": min_macro_auprc,
            "paired_train_order": paired_train_order_required,
        },
        "split_safety": split_checks,
        "baselines": baseline_checks,
        "train_order": train_order_checks,
    }


def numeric_report_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def threshold_report(y_true: Any, y_score: Any, y_mask: Any, targets: Sequence[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"targets": {}}
    for index, target in enumerate(targets):
        valid = y_mask[:, index] > 0.0
        if not valid.any():
            report["targets"][target] = {"status": "unavailable", "reason": "no valid labels"}
            continue
        scores = y_score[valid, index]
        truth = y_true[valid, index]
        report["targets"][target] = {
            "threshold_0_5": binary_operating_point(truth, scores, 0.5),
            "max_f1": choose_threshold_for_max_f1(truth, scores),
            "fixed_sensitivity_0_8": choose_threshold_for_sensitivity(truth, scores, 0.8),
            "fixed_specificity_0_8": choose_threshold_for_specificity(truth, scores, 0.8),
        }
    return report


def write_eval_predictions_artifact(
    *,
    path: Path,
    baseline: str,
    targets: Sequence[str],
    evaluation: EvaluationOutputs,
    quality: dict[str, Any],
) -> dict[str, Any]:
    validate_eval_arrays(
        y_true=evaluation.y_true,
        y_score=evaluation.y_score,
        y_mask=evaluation.y_mask,
        y_logits=evaluation.y_logits,
        targets=targets,
    )
    rows = eval_prediction_records(
        baseline=baseline,
        targets=targets,
        evaluation=evaluation,
    )
    recomputed = metric_report_from_prediction_rows(rows, targets)
    metric_match = metric_reports_match(quality, recomputed, tolerance=1.0e-6)
    if not metric_match["matches"]:
        raise ValueError(
            "prediction artifact metric recomputation mismatch: "
            + "; ".join(metric_match["errors"])
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    hashes = eval_prediction_hashes(rows, targets)
    return {
        "status": "ok",
        "enabled": True,
        "baseline": baseline,
        "artifact_path": path.name,
        "artifact_sha256": hash_file(path),
        "samples": len(rows),
        "sample_ids": [str(row.get("sample_id", "")) for row in rows],
        "target_names": list(targets),
        "hashes": hashes,
        "metric_recompute": compact_metric_report(recomputed),
        "metric_recompute_matches_quality": True,
        "metric_recompute_tolerance": 1.0e-6,
    }


def eval_prediction_records(
    *,
    baseline: str,
    targets: Sequence[str],
    evaluation: EvaluationOutputs,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thresholds = [0.5 for _target in targets]
    for index, sample in enumerate(evaluation.samples):
        labels = finite_float_list(evaluation.y_true[index], "labels")
        mask = finite_float_list(evaluation.y_mask[index], "label_mask")
        logits = finite_float_list(evaluation.y_logits[index], "logits")
        probabilities = finite_float_list(evaluation.y_score[index], "probabilities")
        if not (
            len(labels)
            == len(mask)
            == len(logits)
            == len(probabilities)
            == len(thresholds)
            == len(targets)
        ):
            raise ValueError(f"prediction row {index} target-width mismatch")
        rows.append(
            {
                "schema_version": 1,
                "baseline": baseline,
                "eval_index": int(sample.get("eval_index", index)),
                "sample_id": str(sample.get("sample_id", "")),
                "patient_id": str(sample.get("patient_id", "")),
                "study_id": str(sample.get("study_id", "")),
                "image_id": str(sample.get("image_id", "")),
                "source_path": str(sample.get("source_path", "")),
                "sample_hash": str(sample.get("sample_hash", "")),
                "target_names": list(targets),
                "labels": labels,
                "label_mask": mask,
                "logits": logits,
                "probabilities": probabilities,
                "thresholds": thresholds,
                "predictions": [
                    int(probability >= threshold)
                    for probability, threshold in zip(probabilities, thresholds)
                ],
            }
        )
    return rows


def finite_float_list(values: Any, field: str) -> list[float]:
    numpy = import_numpy()
    array = numpy.asarray(values, dtype="float64").reshape(-1)
    parsed = [float(value) for value in array.tolist()]
    if not all(math.isfinite(value) for value in parsed):
        raise ValueError(f"{field} contains non-finite values")
    return parsed


def read_eval_prediction_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        return [json.loads(line) for line in handle if line.strip()]


def metric_report_from_prediction_rows(
    rows: Sequence[dict[str, Any]],
    targets: Sequence[str] | None = None,
) -> dict[str, Any]:
    numpy = import_numpy()
    if not rows:
        target_names = list(targets or [])
        empty = numpy.zeros((0, len(target_names)), dtype="float32")
        return metric_report(empty, empty, empty, target_names)
    target_names = list(targets or rows[0].get("target_names") or [])
    labels: list[list[float]] = []
    probabilities: list[list[float]] = []
    masks: list[list[float]] = []
    for index, row in enumerate(rows):
        row_targets = list(row.get("target_names") or [])
        if row_targets != target_names:
            raise ValueError(
                f"prediction row {index} target order mismatch: {row_targets!r} != {target_names!r}"
            )
        row_labels = finite_float_list(row.get("labels", []), "labels")
        row_probs = finite_float_list(row.get("probabilities", []), "probabilities")
        row_masks = finite_float_list(row.get("label_mask", []), "label_mask")
        _ = finite_float_list(row.get("logits", []), "logits")
        if not (
            len(row_labels)
            == len(row_probs)
            == len(row_masks)
            == len(target_names)
        ):
            raise ValueError(f"prediction row {index} target-width mismatch")
        labels.append(row_labels)
        probabilities.append(row_probs)
        masks.append(row_masks)
    return metric_report(
        numpy.asarray(labels, dtype="float32"),
        numpy.asarray(probabilities, dtype="float32"),
        numpy.asarray(masks, dtype="float32"),
        target_names,
    )


def eval_prediction_hashes(
    rows: Sequence[dict[str, Any]],
    targets: Sequence[str],
) -> dict[str, str]:
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    return {
        "eval_sample_order_hash": stable_hash(sample_ids),
        "eval_sample_set_hash": stable_hash(sorted(sample_ids)),
        "target_name_hash": stable_hash(list(targets)),
        "label_mask_hash": stable_hash(
            [
                {
                    "sample_id": row.get("sample_id"),
                    "labels": row.get("labels"),
                    "label_mask": row.get("label_mask"),
                }
                for row in rows
            ]
        ),
        "logits_hash": stable_hash(
            [{"sample_id": row.get("sample_id"), "logits": row.get("logits")} for row in rows]
        ),
        "probability_hash": stable_hash(
            [
                {"sample_id": row.get("sample_id"), "probabilities": row.get("probabilities")}
                for row in rows
            ]
        ),
        "threshold_hash": stable_hash(
            [
                {"sample_id": row.get("sample_id"), "thresholds": row.get("thresholds")}
                for row in rows
            ]
        ),
    }


def compact_metric_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "samples": report.get("samples"),
        "macro_auroc": report.get("macro_auroc"),
        "macro_auprc": report.get("macro_auprc"),
        "metric_target_count": report.get("metric_target_count"),
        "targets": {
            target: {
                "auroc": values.get("auroc"),
                "auprc": values.get("auprc"),
                "valid_samples": values.get("valid_samples"),
                "positives": values.get("positives"),
                "negatives": values.get("negatives"),
            }
            for target, values in (report.get("targets") or {}).items()
            if isinstance(values, dict)
        },
    }


def metric_reports_match(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    errors: list[str] = []
    for field in ("samples", "metric_target_count"):
        if left.get(field) != right.get(field):
            errors.append(f"{field} {left.get(field)!r} != {right.get(field)!r}")
    for field in ("macro_auroc", "macro_auprc"):
        left_value = numeric_report_value(left.get(field))
        right_value = numeric_report_value(right.get(field))
        if left_value is None and right_value is None:
            continue
        if left_value is None or right_value is None or abs(left_value - right_value) > tolerance:
            errors.append(f"{field} {left_value!r} != {right_value!r}")
    left_targets = left.get("targets") or {}
    right_targets = right.get("targets") or {}
    if set(left_targets) != set(right_targets):
        errors.append("target metric names differ")
    for target in sorted(set(left_targets).intersection(right_targets)):
        for field in ("auroc", "auprc"):
            left_value = numeric_report_value((left_targets.get(target) or {}).get(field))
            right_value = numeric_report_value((right_targets.get(target) or {}).get(field))
            if left_value is None and right_value is None:
                continue
            if left_value is None or right_value is None or abs(left_value - right_value) > tolerance:
                errors.append(f"{target} {field} {left_value!r} != {right_value!r}")
    return {"matches": not errors, "errors": errors}


def prediction_capture_disabled_report(*, baseline: str, reason: str) -> dict[str, Any]:
    return {
        "status": "disabled",
        "enabled": False,
        "baseline": baseline,
        "reason": reason,
    }


def prediction_summary_report(
    *,
    report_dir: Path,
    predictions: dict[str, Any],
    quality: dict[str, Any],
    targets: Sequence[str],
    capture_enabled: bool,
) -> dict[str, Any]:
    baselines = {
        baseline: normalize_prediction_summary(
            baseline=baseline,
            report_dir=report_dir,
            summary=summary,
            targets=targets,
        )
        for baseline, summary in predictions.items()
    }
    return {
        "schema_version": 1,
        "enabled": capture_enabled,
        "baselines": baselines,
        "paired_comparisons": paired_prediction_comparisons(baselines),
        "quality_capture": {
            baseline: {
                "quality_status": report.get("status"),
                "prediction_status": baselines.get(baseline, {}).get("status"),
                "metric_recompute_matches_predictions": report.get(
                    "metric_recompute_matches_predictions"
                ),
            }
            for baseline, report in quality.items()
        },
    }


def normalize_prediction_summary(
    *,
    baseline: str,
    report_dir: Path,
    summary: dict[str, Any],
    targets: Sequence[str],
) -> dict[str, Any]:
    if summary.get("status") != "ok":
        return summary
    artifact_path = report_dir / str(summary.get("artifact_path", ""))
    if not artifact_path.exists():
        normalized = dict(summary)
        normalized["status"] = "failed"
        normalized["reason"] = f"prediction artifact missing: {artifact_path.name}"
        return normalized
    rows = read_eval_prediction_rows(artifact_path)
    recomputed = metric_report_from_prediction_rows(rows, targets)
    metric_match = metric_reports_match(
        summary.get("metric_recompute") or {},
        compact_metric_report(recomputed),
        tolerance=1.0e-6,
    )
    if not metric_match["matches"]:
        normalized = dict(summary)
        normalized["status"] = "failed"
        normalized["reason"] = "summary metrics do not match prediction artifact"
        normalized["metric_recompute_errors"] = metric_match["errors"]
        return normalized
    normalized = dict(summary)
    normalized["metric_recompute_matches_artifact"] = True
    normalized.setdefault("sample_ids", [str(row.get("sample_id", "")) for row in rows])
    return normalized


def paired_prediction_comparisons(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = baselines.get("pytorch_raw")
    if not isinstance(raw, dict) or raw.get("status") != "ok":
        return {}
    comparisons: dict[str, Any] = {}
    for baseline, summary in baselines.items():
        if baseline == "pytorch_raw" or summary.get("status") != "ok":
            continue
        comparisons[f"{baseline}:vs:pytorch_raw"] = paired_prediction_summary(
            candidate=summary,
            raw=raw,
        )
    return comparisons


def paired_prediction_summary(
    *,
    candidate: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    candidate_ids = [str(value) for value in candidate.get("sample_ids", [])]
    raw_ids = [str(value) for value in raw.get("sample_ids", [])]
    candidate_set = set(candidate_ids)
    raw_set = set(raw_ids)
    candidate_metrics = candidate.get("metric_recompute") or {}
    raw_metrics = raw.get("metric_recompute") or {}
    target_deltas = target_metric_deltas(candidate_metrics, raw_metrics)
    identical_target_order = candidate.get("target_names") == raw.get("target_names")
    missing_from_raw = sorted(candidate_set - raw_set)
    missing_from_candidate = sorted(raw_set - candidate_set)
    label_mask_hash_match = (
        (candidate.get("hashes") or {}).get("label_mask_hash")
        == (raw.get("hashes") or {}).get("label_mask_hash")
    )
    threshold_hash_match = (
        (candidate.get("hashes") or {}).get("threshold_hash")
        == (raw.get("hashes") or {}).get("threshold_hash")
    )
    paired = (
        not missing_from_raw
        and not missing_from_candidate
        and candidate_ids == raw_ids
        and identical_target_order
        and label_mask_hash_match
    )
    return {
        "paired": paired,
        "matched_sample_count": len(candidate_set.intersection(raw_set)),
        "missing_from_raw_count": len(missing_from_raw),
        "missing_from_candidate_count": len(missing_from_candidate),
        "missing_from_medkit_count": len(missing_from_candidate),
        "identical_order": candidate_ids == raw_ids,
        "identical_target_order": identical_target_order,
        "label_mask_hash_match": label_mask_hash_match,
        "threshold_hash_match": threshold_hash_match,
        "macro_auroc": {
            "raw": raw_metrics.get("macro_auroc"),
            "candidate": candidate_metrics.get("macro_auroc"),
            "delta": delta_values(
                numeric_report_value(candidate_metrics.get("macro_auroc")),
                numeric_report_value(raw_metrics.get("macro_auroc")),
            ),
        },
        "macro_auprc": {
            "raw": raw_metrics.get("macro_auprc"),
            "candidate": candidate_metrics.get("macro_auprc"),
            "delta": delta_values(
                numeric_report_value(candidate_metrics.get("macro_auprc")),
                numeric_report_value(raw_metrics.get("macro_auprc")),
            ),
        },
        "target_metric_deltas": target_deltas,
    }


def target_metric_deltas(candidate_metrics: dict[str, Any], raw_metrics: dict[str, Any]) -> dict[str, Any]:
    candidate_targets = candidate_metrics.get("targets") or {}
    raw_targets = raw_metrics.get("targets") or {}
    deltas: dict[str, Any] = {}
    for target in sorted(set(candidate_targets).intersection(raw_targets)):
        candidate_row = candidate_targets.get(target) or {}
        raw_row = raw_targets.get(target) or {}
        deltas[target] = {
            "auroc_delta": delta_values(
                numeric_report_value(candidate_row.get("auroc")),
                numeric_report_value(raw_row.get("auroc")),
            ),
            "auprc_delta": delta_values(
                numeric_report_value(candidate_row.get("auprc")),
                numeric_report_value(raw_row.get("auprc")),
            ),
        }
    return deltas


def binary_operating_point(y_true: Any, y_score: Any, threshold: float) -> dict[str, Any]:
    numpy = import_numpy()
    pred = y_score >= threshold
    truth = y_true >= 0.5
    tp = int(numpy.logical_and(pred, truth).sum())
    tn = int(numpy.logical_and(~pred, ~truth).sum())
    fp = int(numpy.logical_and(pred, ~truth).sum())
    fn = int(numpy.logical_and(~pred, truth).sum())
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, sys.float_info.epsilon)
    return {
        "threshold": float(threshold),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold_for_max_f1(y_true: Any, y_score: Any) -> dict[str, Any]:
    best = None
    for threshold in sorted(set(float(value) for value in y_score)):
        point = binary_operating_point(y_true, y_score, threshold)
        if best is None or (point["f1"], point["sensitivity"]) > (
            best["f1"],
            best["sensitivity"],
        ):
            best = point
    return best or {"status": "unavailable", "reason": "no score thresholds"}


def choose_threshold_for_sensitivity(y_true: Any, y_score: Any, target: float) -> dict[str, Any]:
    best = None
    for threshold in sorted(set(float(value) for value in y_score)):
        point = binary_operating_point(y_true, y_score, threshold)
        if point["sensitivity"] >= target:
            if best is None or point["specificity"] > best["specificity"]:
                best = point
    return best or {"status": "unavailable", "reason": "target sensitivity not reached"}


def choose_threshold_for_specificity(y_true: Any, y_score: Any, target: float) -> dict[str, Any]:
    best = None
    for threshold in sorted(set(float(value) for value in y_score)):
        point = binary_operating_point(y_true, y_score, threshold)
        if point["specificity"] >= target:
            if best is None or point["sensitivity"] > best["sensitivity"]:
                best = point
    return best or {"status": "unavailable", "reason": "target specificity not reached"}


def subgroup_report(records: Sequence[SampleRecord], quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "partially_unavailable",
        "reason": "The public NIH HF parquet export used in this run does not expose PA/AP view position or demographic/source slices.",
        "available_slices": {
            "source_dataset": {
                "samples": len(records),
                "quality_by_baseline": {
                    name: {
                        "macro_auroc": report.get("macro_auroc"),
                        "macro_auprc": report.get("macro_auprc"),
                    }
                    for name, report in quality.items()
                    if report.get("status") == "ok"
                },
            }
        },
    }


def build_run_provenance(
    *,
    args: argparse.Namespace,
    run_id: str,
    run_metadata: dict[str, Any],
    manifest_summary: dict[str, Any],
    split_report: dict[str, Any],
    cache_report: dict[str, Any],
    environment: dict[str, Any],
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "provenance_schema_version": 1,
        "run_id": run_id,
        "command": [str(part) for part in (argv if argv is not None else sys.argv)],
        "script": str(Path(__file__).resolve()),
        "cwd": os.getcwd(),
        "git_commit": environment.get("git_commit"),
        "git_status_short": environment.get("git_status_short"),
        "modal_gpu": os.environ.get("MEDKIT_MODAL_GPU"),
        "benchmark_uses_local_source": os.environ.get("MEDKIT_BENCHMARK_USE_LOCAL_SOURCE"),
        "dataset_requested": run_metadata.get("dataset_requested"),
        "dataset_loaded": run_metadata.get("dataset_loaded"),
        "samples": manifest_summary.get("samples"),
        "splits": split_report.get("counts", {}),
        "targets": list(run_metadata.get("targets", [])),
        "baselines": list(run_metadata.get("baselines", [])),
        "image_size": run_metadata.get("image_size"),
        "cache_image_size": run_metadata.get("cache_image_size"),
        "cache_dtype": run_metadata.get("cache_dtype"),
        "cache_build_workers": run_metadata.get("cache_build_workers"),
        "cache_key_mode": run_metadata.get("cache_key_mode"),
        "allow_destructive_cache": run_metadata.get("allow_destructive_cache"),
        "batch_size": run_metadata.get("batch_size"),
        "drop_last_train": run_metadata.get("drop_last_train"),
        "workers": run_metadata.get("workers"),
        "prefetch_depth": run_metadata.get("prefetch_depth"),
        "prefetch_read_workers": run_metadata.get("prefetch_read_workers"),
        "shuffle_block_batches": run_metadata.get("shuffle_block_batches"),
        "gpu_prefetch_batches": run_metadata.get("gpu_prefetch_batches"),
        "gpu_prefetch_reuse_buffers": run_metadata.get("gpu_prefetch_reuse_buffers"),
        "sync_every_step": run_metadata.get("sync_every_step"),
        "channels_last": run_metadata.get("channels_last"),
        "torch_compile": run_metadata.get("torch_compile"),
        "torch_compile_mode": run_metadata.get("torch_compile_mode"),
        "learning_rate": run_metadata.get("learning_rate"),
        "amp_dtype": run_metadata.get("amp_dtype"),
        "model_init": run_metadata.get("model_init"),
        "loss_kind": run_metadata.get("loss_kind"),
        "loss_pos_weight": run_metadata.get("loss_pos_weight"),
        "loss_pos_weight_cap": run_metadata.get("loss_pos_weight_cap"),
        "focal_gamma": run_metadata.get("focal_gamma"),
        "focal_alpha": run_metadata.get("focal_alpha"),
        "read_mode": run_metadata.get("read_mode"),
        "include_metadata": run_metadata.get("include_metadata"),
        "quality_gate": run_metadata.get("quality_gate"),
        "quality_min_eval_samples": run_metadata.get("quality_min_eval_samples"),
        "quality_min_metric_targets": run_metadata.get("quality_min_metric_targets"),
        "quality_min_macro_auroc": run_metadata.get("quality_min_macro_auroc"),
        "quality_min_macro_auprc": run_metadata.get("quality_min_macro_auprc"),
        "eval_predictions": run_metadata.get("eval_predictions"),
        "train_order_evidence": run_metadata.get("train_order_evidence"),
        "paired_train_order": run_metadata.get("paired_train_order"),
        "prepare_only": run_metadata.get("prepare_only"),
        "profile_batches": run_metadata.get("profile_batches"),
        "loader_batches": run_metadata.get("loader_batches"),
        "warmup_batches": run_metadata.get("warmup_batches"),
        "max_train_batches": run_metadata.get("max_train_batches"),
        "max_eval_batches": run_metadata.get("max_eval_batches"),
        "seed": run_metadata.get("seed"),
        "cache": {
            "cache_dir": cache_report.get("cache_dir"),
            "cache_reused": cache_report.get("cache_reused"),
            "dtype": cache_report.get("dtype"),
            "image_size": cache_report.get("image_size"),
            "cache_key_mode": cache_report.get("cache_key_mode"),
            "cache_identity": cache_report.get("cache_identity"),
            "transform_fingerprint": cache_report.get("transform_fingerprint"),
            "source_manifest_checksum": cache_report.get("source_manifest_checksum"),
            "split_samples": {
                split: details.get("samples")
                for split, details in (cache_report.get("splits") or {}).items()
                if isinstance(details, dict)
            },
        },
        "artifacts": {
            "run_summary": "run-summary.json",
            "manifest": "manifest.jsonl",
            "splits": "splits.json",
            "summary_consistency": "summary-consistency.json",
            "step_profile": "step-profile.json",
            "environment": "environment.json",
            "training_ground_truth": "training-ground-truth.json",
            "localization_eval": "localization-eval.json",
            "eval_predictions_summary": "eval-predictions-summary.json",
            "train_order_summary": "train-order-summary.json",
            "train_schedule_summary": "train-schedule-summary.json",
        },
    }


def validate_run_summary_consistency(
    *,
    summary: dict[str, Any],
    run_metadata: dict[str, Any],
    manifest_summary: dict[str, Any],
    split_report: dict[str, Any],
    cache_report: dict[str, Any],
    reports: dict[str, dict[str, dict[str, Any]]],
    environment: dict[str, Any],
) -> dict[str, Any]:
    errors = run_summary_consistency_errors(
        summary=summary,
        run_metadata=run_metadata,
        manifest_summary=manifest_summary,
        split_report=split_report,
        cache_report=cache_report,
        reports=reports,
        environment=environment,
    )
    return {
        "status": "ok" if not errors else "failed",
        "run_id": summary.get("run_id"),
        "errors": errors,
        "checks": {
            "summary_matches_provenance": not errors,
            "profile_requested": int(run_metadata.get("profile_batches") or 0),
            "baselines": list(run_metadata.get("baselines", [])),
        },
    }


def run_summary_consistency_errors(
    *,
    summary: dict[str, Any],
    run_metadata: dict[str, Any],
    manifest_summary: dict[str, Any],
    split_report: dict[str, Any],
    cache_report: dict[str, Any],
    reports: dict[str, dict[str, dict[str, Any]]],
    environment: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    provenance = summary.get("provenance")
    if not isinstance(provenance, dict):
        errors.append("run-summary provenance missing")
        provenance = {}

    expect_equal(errors, "summary.run_id", summary.get("run_id"), run_metadata.get("run_id"))
    expect_equal(errors, "provenance.run_id", provenance.get("run_id"), run_metadata.get("run_id"))
    expect_equal(
        errors,
        "summary.dataset_loaded",
        summary.get("dataset_loaded"),
        run_metadata.get("dataset_loaded"),
    )
    expect_equal(
        errors,
        "manifest.dataset_loaded",
        manifest_summary.get("dataset_loaded"),
        run_metadata.get("dataset_loaded"),
    )
    expect_equal(errors, "summary.samples", summary.get("samples"), manifest_summary.get("samples"))
    expect_equal(errors, "provenance.samples", provenance.get("samples"), manifest_summary.get("samples"))
    expect_equal(errors, "summary.targets", summary.get("targets"), list(run_metadata.get("targets", [])))
    expect_equal(errors, "provenance.targets", provenance.get("targets"), list(run_metadata.get("targets", [])))
    expect_equal(
        errors,
        "provenance.baselines",
        provenance.get("baselines"),
        list(run_metadata.get("baselines", [])),
    )

    split_counts = split_report.get("counts") or {}
    if isinstance(split_counts, dict):
        expect_equal(
            errors,
            "manifest split sample total",
            sum_numeric_values(split_counts),
            manifest_summary.get("samples"),
        )
        expect_equal(errors, "provenance.splits", provenance.get("splits"), split_counts)
    else:
        errors.append("split_report.counts missing")

    cache_splits = cache_report.get("splits") or {}
    if isinstance(cache_splits, dict):
        cache_split_samples = {
            split: details.get("samples")
            for split, details in cache_splits.items()
            if isinstance(details, dict)
        }
        expect_equal(errors, "cache split samples", cache_split_samples, split_counts)
        expect_equal(
            errors,
            "provenance.cache.split_samples",
            (provenance.get("cache") or {}).get("split_samples"),
            cache_split_samples,
        )
    else:
        errors.append("cache_report.splits missing")

    for field in (
        "image_size",
        "cache_image_size",
        "cache_dtype",
        "cache_build_workers",
        "cache_key_mode",
        "allow_destructive_cache",
        "batch_size",
        "drop_last_train",
        "workers",
        "prefetch_depth",
        "prefetch_read_workers",
        "shuffle_block_batches",
        "gpu_prefetch_batches",
        "gpu_prefetch_reuse_buffers",
        "sync_every_step",
        "channels_last",
        "torch_compile",
        "torch_compile_mode",
        "learning_rate",
        "amp_dtype",
        "model_init",
        "loss_kind",
        "loss_pos_weight",
        "loss_pos_weight_cap",
        "focal_gamma",
        "focal_alpha",
        "read_mode",
        "include_metadata",
        "quality_gate",
        "quality_min_eval_samples",
        "quality_min_metric_targets",
        "quality_min_macro_auroc",
        "quality_min_macro_auprc",
        "eval_predictions",
        "train_order_evidence",
        "paired_train_order",
        "profile_batches",
        "loader_batches",
        "warmup_batches",
        "max_train_batches",
        "max_eval_batches",
        "seed",
    ):
        expect_equal(errors, f"provenance.{field}", provenance.get(field), run_metadata.get(field))

    cache_provenance = provenance.get("cache") or {}
    expect_equal(errors, "cache.dtype", cache_report.get("dtype"), run_metadata.get("cache_dtype"))
    expect_equal(errors, "cache.image_size", cache_report.get("image_size"), run_metadata.get("cache_image_size"))
    expect_equal(
        errors,
        "cache.cache_key_mode",
        cache_report.get("cache_key_mode"),
        run_metadata.get("cache_key_mode"),
    )
    expect_equal(
        errors,
        "provenance.cache.cache_identity",
        cache_provenance.get("cache_identity"),
        cache_report.get("cache_identity"),
    )
    expect_equal(
        errors,
        "provenance.cache.transform_fingerprint",
        cache_provenance.get("transform_fingerprint"),
        cache_report.get("transform_fingerprint"),
    )
    expect_equal(
        errors,
        "provenance.cache.source_manifest_checksum",
        cache_provenance.get("source_manifest_checksum"),
        cache_report.get("source_manifest_checksum"),
    )

    env_metadata = environment.get("run_metadata")
    if isinstance(env_metadata, dict):
        expect_equal(errors, "environment.run_metadata", env_metadata, run_metadata)
    else:
        errors.append("environment.run_metadata missing")

    expected_loader = rounded_samples_per_second(reports.get("loader", {}), digits=3)
    expected_gpu = rounded_samples_per_second(reports.get("gpu", {}), digits=3)
    expected_quality = rounded_quality_metric(reports.get("quality", {}), "macro_auroc", digits=5)
    expect_equal(
        errors,
        "summary.loader_samples_per_second",
        summary.get("loader_samples_per_second"),
        expected_loader,
    )
    expect_equal(
        errors,
        "summary.train_samples_per_second",
        summary.get("train_samples_per_second"),
        expected_gpu,
    )
    expect_equal(
        errors,
        "summary.quality_macro_auroc",
        summary.get("quality_macro_auroc"),
        expected_quality,
    )
    expect_equal(errors, "summary.memory", summary.get("memory"), memory_summary(reports))
    expect_equal(errors, "summary.predictions", summary.get("predictions"), reports.get("predictions"))
    expect_equal(
        errors,
        "summary.localization_eval",
        summary.get("localization_eval"),
        reports.get("localization_eval"),
    )
    expected_quality_gate = summary.get("quality_gate")
    if not isinstance(expected_quality_gate, dict):
        errors.append("summary.quality_gate missing")

    requested_profile_batches = int(run_metadata.get("profile_batches") or 0)
    expected_profile = {
        name: report.get("summary")
        for name, report in (reports.get("profile") or {}).items()
        if report.get("status") == "ok"
    }
    expect_equal(errors, "summary.profile", summary.get("profile"), expected_profile)
    if requested_profile_batches > 0:
        for baseline in run_metadata.get("baselines", []):
            gpu_report = (reports.get("gpu") or {}).get(baseline, {})
            profile_report = (reports.get("profile") or {}).get(baseline, {})
            if gpu_report.get("status") != "ok":
                continue
            if profile_report.get("status") != "ok":
                errors.append(f"profile report for {baseline!r} is not ok")
                continue
            records = profile_report.get("records")
            profile_summary = profile_report.get("summary")
            if not isinstance(records, list):
                errors.append(f"profile report for {baseline!r} records missing")
                continue
            if not isinstance(profile_summary, dict):
                errors.append(f"profile report for {baseline!r} summary missing")
                continue
            if profile_summary.get("profile_artifact_path") != "step-profile.json":
                errors.append(f"profile report for {baseline!r} missing step-profile artifact path")
            expect_equal(
                errors,
                f"profile profiled_batches for {baseline}",
                profile_summary.get("profiled_batches"),
                len(records),
            )

    return errors


def expect_equal(errors: list[str], label: str, left: Any, right: Any) -> None:
    if left != right:
        errors.append(f"{label} mismatch: {left!r} != {right!r}")


def sum_numeric_values(values: dict[str, Any]) -> int:
    return sum(int(value) for value in values.values())


def rounded_samples_per_second(
    section: dict[str, dict[str, Any]],
    *,
    digits: int,
) -> dict[str, float]:
    return {
        name: round(report["samples_per_second"], digits)
        for name, report in section.items()
        if "samples_per_second" in report
    }


def rounded_quality_metric(
    section: dict[str, dict[str, Any]],
    metric: str,
    *,
    digits: int,
) -> dict[str, float]:
    return {
        name: round(report.get(metric, float("nan")), digits)
        for name, report in section.items()
        if report.get("status") == "ok"
    }


def environment_report(run_metadata: dict[str, Any]) -> dict[str, Any]:
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cwd": os.getcwd(),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": git_output(["git", "status", "--short"]),
        "run_metadata": run_metadata,
        "packages": {},
    }
    for name in (
        "torch",
        "torchvision",
        "torchxrayvision",
        "monai",
        "webdataset",
        "nvidia.dali",
        "nvidia-dali-cuda130",
        "nvidia-dali-cuda120",
        "medkit_rs",
        "datasets",
        "PIL",
        "pydicom",
        "pylibjpeg",
        "pylibjpeg-libjpeg",
        "psutil",
        "numpy",
        "sklearn",
    ):
        report["packages"][name] = package_version(name)
    try:
        torch = import_torch()
        report["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "version": getattr(torch.version, "cuda", None),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as error:
        report["cuda"] = {"available": False, "error": str(error)}
    return report


def choose_device(torch: Any, requested: str) -> Any:
    if requested.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested)
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        return device
    return torch.device("cpu")


def set_torch_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def estimate_h2d_gbps(samples: int, target_count: int, seconds: float) -> float:
    # Filled after training with an image-size independent conservative estimate
    # of image plus label/mask bytes consumed by the GPU loop. Exact H2D event
    # timing belongs in the future Rust/pinned path.
    bytes_per_sample = target_count * 2 * 4
    return (samples * bytes_per_sample / (1024.0**3)) / max(seconds, sys.float_info.epsilon)


def cache_normalization(cache_dir: Path) -> tuple[float, float]:
    metadata = load_json(cache_dir / "cache-metadata.json")
    norm = metadata.get("normalization", {})
    return float(norm.get("mean", 0.5)), float(norm.get("std", 0.25))


def cache_dtype_from_metadata(cache_dir: Path) -> str:
    try:
        metadata = load_json(cache_dir / "cache-metadata.json")
    except (OSError, json.JSONDecodeError):
        return "unknown"
    image_size_policy = metadata.get("image_size_policy", {})
    if not isinstance(image_size_policy, dict):
        image_size_policy = {}
    return str(metadata.get("dtype") or image_size_policy.get("dtype") or "unknown")


def write_manifest(path: Path, records: Sequence[SampleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record_to_json(record), sort_keys=True) + "\n")


def load_manifest(path: Path) -> list[SampleRecord]:
    records = []
    allowed_fields = set(SampleRecord.__dataclass_fields__)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records.append(
            SampleRecord(**{key: value for key, value in row.items() if key in allowed_fields})
        )
    return records


def load_manifest_if_compatible(
    path: Path,
    *,
    requested_samples: int,
) -> list[SampleRecord] | None:
    records = load_manifest(path)
    if requested_samples > 0 and len(records) != requested_samples:
        return None
    return records


def record_to_json(record: SampleRecord) -> dict[str, Any]:
    return {
        "sample_id": record.sample_id,
        "patient_id": record.patient_id,
        "study_id": record.study_id,
        "image_id": record.image_id,
        "image_path": record.image_path,
        "filename": record.filename,
        "source_format": record.source_format,
        "modality": record.modality,
        "view_position": record.view_position,
        "laterality": None,
        "width": record.width,
        "height": record.height,
        "photometric_interpretation": "MONOCHROME2",
        "labels": record.labels,
        "label_source": record.label_source,
        "localization_boxes": record.localization_boxes or [],
        "source_split": record.source_split,
        "split": record.split,
        "sha256": record.sha256,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "baseline"


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def memory_snapshot(
    *,
    pipeline: dict[str, Any] | None = None,
    max_batch_tensor_bytes: int = 0,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "peak_rss_mb": peak_rss_mb(),
        "current_rss_mb": current_rss_mb(),
        "max_batch_tensor_mb": bytes_to_mb(max_batch_tensor_bytes),
        "estimated_pinned_memory_mb": estimate_pinned_memory_mb(
            pipeline=pipeline,
            max_batch_tensor_bytes=max_batch_tensor_bytes,
        ),
        "sources": ["resource.getrusage"],
    }
    full_info = psutil_memory_full_info()
    if full_info:
        snapshot["sources"].append("psutil.Process.memory_full_info")
        snapshot.update(full_info)
    smaps, smaps_source = smaps_memory(smaps_cache_path_categories(pipeline))
    if smaps:
        snapshot["sources"].append(smaps_source)
        snapshot.update(smaps)
    return snapshot


def memory_summary(reports: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    fields = (
        "peak_rss_mb",
        "current_rss_mb",
        "psutil_rss_mb",
        "psutil_pss_mb",
        "psutil_uss_mb",
        "psutil_swap_mb",
        "smaps_pss_mb",
        "smaps_pss_file_mb",
        "smaps_pss_cache_images_mb",
        "smaps_pss_cache_labels_mb",
        "smaps_pss_cache_masks_mb",
        "smaps_pss_metadata_mb",
        "smaps_pss_other_file_mb",
        "smaps_uss_mb",
        "smaps_private_dirty_mb",
        "smaps_private_clean_mb",
        "smaps_locked_mb",
        "max_batch_tensor_mb",
        "estimated_pinned_memory_mb",
    )
    summary: dict[str, Any] = {}
    for section in ("loader", "gpu"):
        section_reports = reports.get(section, {})
        section_summary = {}
        for name, report in section_reports.items():
            memory = report.get("memory")
            if not isinstance(memory, dict):
                continue
            section_summary[name] = {
                field: memory[field] for field in fields if field in memory
            }
        if section_summary:
            summary[section] = section_summary
    return summary


def training_ground_truth_report(reports: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    prediction_baselines = (reports.get("predictions", {}).get("baselines") or {})
    train_order_baselines = (reports.get("train_order", {}).get("baselines") or {})
    baselines = sorted(
        {
            *reports.get("loader", {}).keys(),
            *reports.get("gpu", {}).keys(),
            *reports.get("profile", {}).keys(),
            *reports.get("quality", {}).keys(),
            *prediction_baselines.keys(),
            *train_order_baselines.keys(),
        }
    )
    rows = {
        baseline: training_ground_truth_row(
            baseline=baseline,
            loader=reports.get("loader", {}).get(baseline, {}),
            gpu=reports.get("gpu", {}).get(baseline, {}),
            profile=reports.get("profile", {}).get(baseline, {}),
            quality=reports.get("quality", {}).get(baseline, {}),
            predictions=prediction_baselines.get(baseline, {}),
            train_order=train_order_baselines.get(baseline, {}),
        )
        for baseline in baselines
    }
    raw = rows.get("pytorch_raw")
    comparisons = {
        f"{baseline}:vs:pytorch_raw": compare_ground_truth_rows(row, raw)
        for baseline, row in rows.items()
        if baseline != "pytorch_raw" and raw is not None
    }
    return {
        "schema_version": 1,
        "baselines": rows,
        "comparisons": comparisons,
        "speed_claims": speed_claim_report(rows=rows, comparisons=comparisons),
        "paired_quality": (reports.get("predictions", {}) or {}).get("paired_comparisons", {}),
        "paired_train_order": (reports.get("train_order", {}) or {}).get("paired_comparisons", {}),
    }


def training_ground_truth_row(
    *,
    baseline: str,
    loader: dict[str, Any],
    gpu: dict[str, Any],
    profile: dict[str, Any],
    quality: dict[str, Any],
    predictions: dict[str, Any],
    train_order: dict[str, Any],
) -> dict[str, Any]:
    profile_summary = profile.get("summary") or {}
    memory = gpu.get("memory") or {}
    loader_memory = loader.get("memory") or {}
    phase_budget = profile_summary.get("profile_phase_budget_ms_per_batch") or {}
    phase_percent = profile_summary.get("profile_phase_budget_end_to_end_percent") or {}
    return {
        "baseline": baseline,
        "status": gpu.get("status") or loader.get("status") or profile.get("status"),
        "speed": {
            "train_samples_per_second": gpu.get("samples_per_second"),
            "loader_samples_per_second": loader.get("samples_per_second"),
            "profile_end_to_end_samples_per_second": profile_summary.get(
                "profile_end_to_end_samples_per_s"
            ),
            "profile_train_samples_per_second": profile_summary.get(
                "profile_train_samples_per_s"
            ),
            "data_wait_percent": gpu.get("data_wait_percent"),
        },
        "gpu_utilization": gpu.get("gpu_utilization"),
        "pipeline": pipeline_summary(gpu=gpu, loader=loader),
        "profile": {
            "timing_scope": profile_record_modes(profile, "timing_scope"),
            "h2d_timing_mode": profile_summary.get("profile_h2d_timing_mode"),
            "sync_every_step_effective": profile_record_modes(
                profile, "sync_every_step_effective"
            ),
            "phase_budget_ms_per_batch": phase_budget,
            "phase_budget_end_to_end_percent": phase_percent,
            "largest_phase": largest_phase(phase_budget),
            "step_accounted_percent": profile_summary.get("profile_step_accounted_percent"),
            "step_reconciled_percent": profile_summary.get("profile_step_reconciled_percent"),
            "residual_step_signed_percent": profile_summary.get(
                "profile_residual_step_signed_percent"
            ),
            "profiled_batches": profile_summary.get("profiled_batches"),
            "profiled_samples": profile_summary.get("profiled_samples"),
        },
        "native_prefetch": {
            key: value for key, value in gpu.items() if key.startswith("train_native_prefetch_")
        },
        "memory": {
            "gpu_pss_mb": memory.get("smaps_pss_mb"),
            "gpu_uss_mb": memory.get("smaps_uss_mb"),
            "gpu_file_pss_mb": memory.get("smaps_pss_file_mb"),
            "gpu_anon_pss_mb": memory.get("smaps_pss_anon_mb"),
            "gpu_private_dirty_mb": memory.get("smaps_private_dirty_mb"),
            "gpu_cache_image_pss_mb": memory.get("smaps_pss_cache_images_mb"),
            "gpu_estimated_pinned_memory_mb": memory.get("estimated_pinned_memory_mb"),
            "loader_pss_mb": loader_memory.get("smaps_pss_mb"),
            "loader_cache_image_pss_mb": loader_memory.get("smaps_pss_cache_images_mb"),
            "max_batch_tensor_mb": memory.get("max_batch_tensor_mb"),
            "cuda_peak_allocated_mb": gpu.get("cuda_peak_allocated_mb"),
        },
        "quality": {
            "macro_auroc": quality.get("macro_auroc"),
            "macro_auprc": quality.get("macro_auprc"),
            "samples": quality.get("samples"),
            "metric_target_count": quality.get("metric_target_count"),
        },
        "eval_predictions": {
            "enabled": predictions.get("enabled"),
            "status": predictions.get("status"),
            "artifact_path": predictions.get("artifact_path"),
            "artifact_sha256": predictions.get("artifact_sha256"),
            "hashes": predictions.get("hashes"),
            "metric_recompute": predictions.get("metric_recompute"),
            "metric_recompute_matches_quality": predictions.get(
                "metric_recompute_matches_quality"
            ),
            "metric_recompute_matches_artifact": predictions.get(
                "metric_recompute_matches_artifact"
            ),
        },
        "train_order": {
            "enabled": train_order.get("enabled"),
            "status": train_order.get("status"),
            "artifact_path": train_order.get("artifact_path"),
            "artifact_sha256": train_order.get("artifact_sha256"),
            "hashes": train_order.get("hashes"),
            "warmup_batches": train_order.get("warmup_batches"),
            "train_batches": train_order.get("train_batches"),
            "train_samples": train_order.get("train_samples"),
            "same_train_order_each_epoch": train_order.get("same_train_order_each_epoch"),
            "epoch_summaries": train_order.get("epoch_summaries"),
            "artifact_recheck_matches_summary": train_order.get(
                "artifact_recheck_matches_summary"
            ),
        },
        "training": {
            "samples": gpu.get("samples"),
            "batches": gpu.get("batches"),
            "drop_last_train": gpu.get("drop_last_train"),
            "skipped_incomplete_batches": gpu.get("skipped_incomplete_batches"),
            "loss_mean": gpu.get("loss_mean"),
            "loss_last": gpu.get("loss_last"),
            "loss_pos_weight": gpu.get("loss_pos_weight"),
            "h2d_gb_per_second_estimate": gpu.get("h2d_gb_per_second_estimate"),
        },
    }


def compare_ground_truth_rows(
    candidate: dict[str, Any],
    raw: dict[str, Any],
) -> dict[str, Any]:
    speed_comparison = speed_comparison_report(candidate=candidate, raw=raw)
    return {
        "speed_comparison": speed_comparison,
        "train_samples_per_second_speedup": ratio_values(
            nested_numeric(candidate, "speed", "train_samples_per_second"),
            nested_numeric(raw, "speed", "train_samples_per_second"),
        ),
        "profile_end_to_end_speedup": ratio_values(
            nested_numeric(candidate, "speed", "profile_end_to_end_samples_per_second"),
            nested_numeric(raw, "speed", "profile_end_to_end_samples_per_second"),
        ),
        "gpu_pss_mb_delta": delta_values(
            nested_numeric(candidate, "memory", "gpu_pss_mb"),
            nested_numeric(raw, "memory", "gpu_pss_mb"),
        ),
        "gpu_cache_image_pss_mb_delta": delta_values(
            nested_numeric(candidate, "memory", "gpu_cache_image_pss_mb"),
            nested_numeric(raw, "memory", "gpu_cache_image_pss_mb"),
        ),
        "gpu_utilization_percent_mean_delta": delta_values(
            nested_numeric(candidate, "gpu_utilization", "gpu_utilization_percent", "mean"),
            nested_numeric(raw, "gpu_utilization", "gpu_utilization_percent", "mean"),
        ),
        "gpu_utilization_percent_mean_speedup": ratio_values(
            nested_numeric(candidate, "gpu_utilization", "gpu_utilization_percent", "mean"),
            nested_numeric(raw, "gpu_utilization", "gpu_utilization_percent", "mean"),
        ),
        "memory_utilization_percent_mean_delta": delta_values(
            nested_numeric(candidate, "gpu_utilization", "memory_utilization_percent", "mean"),
            nested_numeric(raw, "gpu_utilization", "memory_utilization_percent", "mean"),
        ),
        "power_draw_w_mean_delta": delta_values(
            nested_numeric(candidate, "gpu_utilization", "power_draw_w", "mean"),
            nested_numeric(raw, "gpu_utilization", "power_draw_w", "mean"),
        ),
        "macro_auroc_delta": delta_values(
            nested_numeric(candidate, "quality", "macro_auroc"),
            nested_numeric(raw, "quality", "macro_auroc"),
        ),
        "macro_auprc_delta": delta_values(
            nested_numeric(candidate, "quality", "macro_auprc"),
            nested_numeric(raw, "quality", "macro_auprc"),
        ),
        "phase_delta_ms_per_batch": phase_budget_deltas(candidate, raw),
        "train_order": paired_train_order_summary(
            candidate=candidate.get("train_order") or {},
            raw=raw.get("train_order") or {},
        )
        if (candidate.get("train_order") or {}).get("status") == "ok"
        and (raw.get("train_order") or {}).get("status") == "ok"
        else {},
    }


def pipeline_summary(*, gpu: dict[str, Any], loader: dict[str, Any]) -> dict[str, Any]:
    pipeline = gpu.get("pipeline") or loader.get("pipeline") or {}
    if not isinstance(pipeline, dict):
        return {}
    keys = (
        "baseline",
        "worker_mode",
        "batch_schedule",
        "paired_train_order",
        "num_workers",
        "requested_num_workers",
        "pin_memory",
        "native_prefetch",
        "prefetch_depth",
        "prefetch_read_workers",
        "read_mode",
        "include_metadata",
    )
    return {key: pipeline.get(key) for key in keys if key in pipeline}


def speed_comparison_report(candidate: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    measured_speedup = ratio_values(
        nested_numeric(candidate, "speed", "train_samples_per_second"),
        nested_numeric(raw, "speed", "train_samples_per_second"),
    )
    raw_pipeline = raw.get("pipeline") or {}
    candidate_pipeline = candidate.get("pipeline") or {}
    reasons: list[str] = []
    headline_eligible = True
    if raw_pipeline.get("worker_mode") == "paired_schedule_single_process":
        headline_eligible = False
        reasons.append(
            "raw PyTorch used the strict paired train-order single-process loader"
        )
    requested_workers = raw_pipeline.get("requested_num_workers")
    actual_workers = raw_pipeline.get("num_workers")
    if requested_workers and actual_workers == 0:
        headline_eligible = False
        reasons.append(
            f"raw PyTorch requested {requested_workers} workers but ran with 0 workers"
        )
    if raw_pipeline.get("paired_train_order") and not candidate_pipeline.get("paired_train_order"):
        headline_eligible = False
        reasons.append("candidate and raw paired-train-order modes differ")
    train_order = paired_train_order_summary(
        candidate=candidate.get("train_order") or {},
        raw=raw.get("train_order") or {},
    )
    if (
        raw_pipeline.get("paired_train_order")
        and candidate_pipeline.get("paired_train_order")
        and train_order
        and not train_order.get("paired")
    ):
        headline_eligible = False
        reasons.append("paired train-order evidence failed")
    return {
        "headline_eligible": headline_eligible,
        "measured_train_samples_per_second_speedup": measured_speedup,
        "headline_train_samples_per_second_speedup": measured_speedup
        if headline_eligible
        else None,
        "denominator_baseline": raw.get("baseline"),
        "denominator_worker_mode": raw_pipeline.get("worker_mode"),
        "candidate_worker_mode": candidate_pipeline.get("worker_mode"),
        "reasons": reasons,
    }


def speed_claim_report(
    *,
    rows: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    claims: dict[str, Any] = {}
    for key, comparison in comparisons.items():
        speed_comparison = comparison.get("speed_comparison") or {}
        claims[key] = {
            "headline_eligible": bool(speed_comparison.get("headline_eligible")),
            "headline_train_samples_per_second_speedup": speed_comparison.get(
                "headline_train_samples_per_second_speedup"
            ),
            "measured_train_samples_per_second_speedup": speed_comparison.get(
                "measured_train_samples_per_second_speedup"
            ),
            "reasons": speed_comparison.get("reasons") or [],
        }
    return {
        "schema_version": 1,
        "raw_baseline": "pytorch_raw" if "pytorch_raw" in rows else None,
        "claims": claims,
    }


def phase_budget_deltas(candidate: dict[str, Any], raw: dict[str, Any]) -> dict[str, float]:
    candidate_phases = candidate.get("profile", {}).get("phase_budget_ms_per_batch") or {}
    deltas: dict[str, float] = {}
    for phase, value in candidate_phases.items():
        delta = delta_values(
            numeric_report_value(value),
            nested_numeric(raw, "profile", "phase_budget_ms_per_batch", phase),
        )
        if delta is not None:
            deltas[str(phase)] = delta
    return deltas


def profile_record_modes(profile: dict[str, Any], field: str) -> list[Any]:
    records = profile.get("records") if isinstance(profile, dict) else None
    if not isinstance(records, list):
        return []
    return sorted(
        {
            record.get(field)
            for record in records
            if isinstance(record, dict) and field in record
        },
        key=str,
    )


def largest_phase(phase_budget: dict[str, Any]) -> dict[str, Any] | None:
    parsed = {
        str(name): numeric_report_value(value)
        for name, value in phase_budget.items()
    }
    parsed = {
        name: value
        for name, value in parsed.items()
        if value is not None and name != "residual_step_ms_signed"
    }
    if not parsed:
        return None
    phase, value = max(parsed.items(), key=lambda item: item[1])
    return {"phase": phase, "ms_per_batch": value}


def nested_numeric(mapping: dict[str, Any], *keys: str) -> float | None:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return numeric_report_value(current)


def ratio_values(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def delta_values(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def current_rss_mb() -> float | None:
    statm = Path("/proc/self/statm")
    if statm.exists():
        try:
            pages = int(statm.read_text().split()[1])
            return bytes_to_mb(pages * os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError):
            pass
    try:
        import psutil  # type: ignore

        return bytes_to_mb(int(psutil.Process().memory_info().rss))
    except Exception:
        return None


def psutil_memory_full_info() -> dict[str, float]:
    try:
        import psutil  # type: ignore

        info = psutil.Process().memory_full_info()
    except Exception:
        return {}
    fields = {
        "psutil_rss_mb": "rss",
        "psutil_pss_mb": "pss",
        "psutil_uss_mb": "uss",
        "psutil_swap_mb": "swap",
    }
    report = {}
    for output_name, attr_name in fields.items():
        value = getattr(info, attr_name, None)
        if value is not None:
            report[output_name] = bytes_to_mb(int(value))
    return report


def smaps_memory(cache_file_categories: dict[str, str] | None = None) -> tuple[dict[str, float], str]:
    rollup = smaps_rollup_memory()
    smaps = smaps_full_memory(cache_file_categories)
    if rollup and smaps:
        combined = dict(rollup)
        for key, value in smaps.items():
            if key.startswith("smaps_pss_cache_") or key in {
                "smaps_pss_file_mb",
                "smaps_pss_anon_mb",
                "smaps_pss_heap_mb",
                "smaps_pss_stack_mb",
                "smaps_pss_dev_mb",
                "smaps_pss_metadata_mb",
                "smaps_pss_other_file_mb",
            }:
                combined[key] = value
        return combined, "/proc/self/smaps_rollup+/proc/self/smaps"
    if rollup:
        return rollup, "/proc/self/smaps_rollup"
    if smaps:
        return smaps, "/proc/self/smaps"
    return {}, ""


def smaps_rollup_memory() -> dict[str, float]:
    path = Path("/proc/self/smaps_rollup")
    if not path.exists():
        return {}
    try:
        raw = path.read_text()
    except OSError:
        return {}
    values_kb: dict[str, int] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            values_kb[key] = int(parts[0])
        except ValueError:
            continue

    private_clean = values_kb.get("Private_Clean", 0)
    private_dirty = values_kb.get("Private_Dirty", 0)
    fields = {
        "smaps_rss_mb": "Rss",
        "smaps_pss_mb": "Pss",
        "smaps_pss_anon_mb": "Pss_Anon",
        "smaps_pss_file_mb": "Pss_File",
        "smaps_pss_shmem_mb": "Pss_Shmem",
        "smaps_private_clean_mb": "Private_Clean",
        "smaps_private_dirty_mb": "Private_Dirty",
        "smaps_shared_clean_mb": "Shared_Clean",
        "smaps_shared_dirty_mb": "Shared_Dirty",
        "smaps_locked_mb": "Locked",
        "smaps_swap_mb": "Swap",
    }
    report = {
        output_name: kib_to_mb(values_kb[field])
        for output_name, field in fields.items()
        if field in values_kb
    }
    report["smaps_uss_mb"] = kib_to_mb(private_clean + private_dirty)
    return report


def smaps_full_memory(cache_file_categories: dict[str, str] | None = None) -> dict[str, float]:
    path = Path("/proc/self/smaps")
    if not path.exists():
        return {}
    try:
        raw = path.read_text()
    except OSError:
        return {}
    return parse_smaps_full_memory(raw, cache_file_categories)


def parse_smaps_full_memory(
    raw: str,
    cache_file_categories: dict[str, str] | None = None,
) -> dict[str, float]:
    totals: dict[str, int] = {}
    pss_buckets_kb: dict[str, int] = {}
    current_bucket = "anon"
    for line in raw.splitlines():
        if not line:
            continue
        if SMAPS_HEADER_RE.match(line):
            current_bucket = smaps_mapping_bucket(line, cache_file_categories or {})
            continue
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            value_kb = int(parts[0])
        except ValueError:
            continue
        totals[key] = totals.get(key, 0) + value_kb
        if key == "Pss":
            pss_buckets_kb[current_bucket] = pss_buckets_kb.get(current_bucket, 0) + value_kb

    private_clean = totals.get("Private_Clean", 0)
    private_dirty = totals.get("Private_Dirty", 0)
    fields = {
        "smaps_rss_mb": "Rss",
        "smaps_pss_mb": "Pss",
        "smaps_private_clean_mb": "Private_Clean",
        "smaps_private_dirty_mb": "Private_Dirty",
        "smaps_shared_clean_mb": "Shared_Clean",
        "smaps_shared_dirty_mb": "Shared_Dirty",
        "smaps_locked_mb": "Locked",
        "smaps_swap_mb": "Swap",
    }
    report = {
        output_name: kib_to_mb(totals[field])
        for output_name, field in fields.items()
        if field in totals
    }
    file_buckets = (
        "cache_images",
        "cache_labels",
        "cache_masks",
        "metadata",
        "other_file",
        "dev",
    )
    anon_buckets = ("anon", "heap", "stack")
    report["smaps_pss_file_mb"] = kib_to_mb(
        sum(pss_buckets_kb.get(bucket, 0) for bucket in file_buckets)
    )
    report["smaps_pss_anon_mb"] = kib_to_mb(
        sum(pss_buckets_kb.get(bucket, 0) for bucket in anon_buckets)
    )
    for bucket, field in {
        "cache_images": "smaps_pss_cache_images_mb",
        "cache_labels": "smaps_pss_cache_labels_mb",
        "cache_masks": "smaps_pss_cache_masks_mb",
        "metadata": "smaps_pss_metadata_mb",
        "other_file": "smaps_pss_other_file_mb",
        "heap": "smaps_pss_heap_mb",
        "stack": "smaps_pss_stack_mb",
        "dev": "smaps_pss_dev_mb",
    }.items():
        report[field] = kib_to_mb(pss_buckets_kb.get(bucket, 0))
    report["smaps_uss_mb"] = kib_to_mb(private_clean + private_dirty)
    return report


def smaps_header_is_file_backed(line: str) -> bool:
    return smaps_mapping_bucket(line, {}) in {
        "other_file",
        "cache_images",
        "cache_labels",
        "cache_masks",
        "metadata",
        "dev",
    }


def smaps_mapping_bucket(line: str, cache_file_categories: dict[str, str]) -> str:
    pathname = smaps_header_pathname(line)
    if not pathname:
        return "anon"
    if pathname == "[heap]":
        return "heap"
    if pathname.startswith("[stack"):
        return "stack"
    if pathname.startswith("[dev:"):
        return "dev"
    if pathname.startswith("["):
        return "anon"
    normalized = normalize_smaps_pathname(pathname)
    category = cache_file_categories.get(normalized)
    if category:
        return category
    if normalized.startswith("/dev/"):
        return "dev"
    return "other_file"


def smaps_header_pathname(line: str) -> str:
    parts = line.split(maxsplit=5)
    if len(parts) < 6:
        return ""
    return parts[5]


def normalize_smaps_pathname(pathname: str) -> str:
    if pathname.endswith(" (deleted)"):
        pathname = pathname[: -len(" (deleted)")]
    try:
        return str(Path(pathname).resolve(strict=False))
    except OSError:
        return pathname


def smaps_cache_path_categories(pipeline: dict[str, Any] | None) -> dict[str, str]:
    if not pipeline:
        return {}
    cache_dir_value = pipeline.get("cache_dir")
    if not cache_dir_value:
        return {}
    cache_dir = Path(str(cache_dir_value))
    metadata_path = cache_dir / "cache-metadata.json"
    try:
        metadata = load_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return {}
    splits = metadata.get("splits")
    if not isinstance(splits, dict):
        return {}
    categories: dict[str, str] = {}
    for split_info in splits.values():
        if not isinstance(split_info, dict):
            continue
        for key, category in {
            "images_path": "cache_images",
            "labels_path": "cache_labels",
            "masks_path": "cache_masks",
            "metadata_path": "metadata",
        }.items():
            value = split_info.get(key)
            if not value:
                continue
            path = Path(str(value))
            if not path.is_absolute():
                path = cache_dir / path
            categories[normalize_smaps_pathname(str(path))] = category
    return categories


def estimate_pinned_memory_mb(
    *,
    pipeline: dict[str, Any] | None,
    max_batch_tensor_bytes: int,
) -> float:
    if not pipeline or not pipeline.get("pin_memory") or max_batch_tensor_bytes <= 0:
        return 0.0
    depth = int(pipeline.get("prefetch_depth") or 1)
    if depth <= 0:
        depth = 1
    return bytes_to_mb(max_batch_tensor_bytes * depth)


def batch_tensor_bytes(batch: Any) -> int:
    if not isinstance(batch, dict):
        return 0
    total = 0
    for key in ("image", "labels", "mask"):
        value = batch.get(key)
        if value is None:
            continue
        numel = getattr(value, "numel", None)
        element_size = getattr(value, "element_size", None)
        if callable(numel) and callable(element_size):
            total += int(numel()) * int(element_size())
    return total


def bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def kib_to_mb(value: int | float) -> float:
    return float(value) / 1024.0


def package_version(name: str) -> str | None:
    try:
        if name in {"nvidia-dali-cuda130", "nvidia-dali-cuda120", "pylibjpeg-libjpeg"}:
            import importlib.metadata

            return importlib.metadata.version(name)
        if name == "nvidia.dali":
            import nvidia.dali as dali  # type: ignore

            version = getattr(dali, "__version__", None)
            if version is not None:
                return str(version)
            import importlib.metadata

            for distribution_name in ("nvidia-dali-cuda130", "nvidia-dali-cuda120"):
                try:
                    return importlib.metadata.version(distribution_name)
                except importlib.metadata.PackageNotFoundError:
                    continue
            return "unknown"
        if name == "PIL":
            import PIL  # type: ignore

            return str(PIL.__version__)
        if name == "sklearn":
            import sklearn  # type: ignore

            return str(sklearn.__version__)
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def git_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def shell_quote(value: str) -> str:
    if not value:
        return "''"
    if all(char.isalnum() or char in "-_./:=," for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def import_numpy() -> Any:
    try:
        import numpy  # type: ignore
    except ImportError as error:
        raise RuntimeError("numpy is required for the CXR benchmark") from error
    return numpy


def import_torch() -> Any:
    try:
        import torch  # type: ignore
    except ImportError as error:
        raise RuntimeError("torch is required for the CXR benchmark") from error
    return torch


def import_medkit_rs() -> Any:
    use_local_source = os.environ.get("MEDKIT_BENCHMARK_USE_LOCAL_SOURCE", "1") != "0"
    source_root = Path(__file__).resolve().parents[3] / "python"
    if use_local_source and source_root.exists() and str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        import medkit_rs  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "medkit_rs is required for medkit_native_cxr; install the PyO3 "
            "extension with `uv sync --dev` or `uv run maturin develop --release`"
        ) from error
    return medkit_rs


def import_torchvision() -> Any:
    try:
        import torchvision  # type: ignore
    except ImportError as error:
        raise RuntimeError("torchvision is required for DenseNet-121") from error
    return torchvision


def import_monai() -> Any:
    try:
        import monai  # type: ignore
    except ImportError as error:
        raise RuntimeError("monai is required for the MONAI baseline") from error
    return monai


def import_torchxrayvision() -> Any:
    try:
        import torchxrayvision as xrv  # type: ignore
    except ImportError as error:
        raise RuntimeError("torchxrayvision is required for the TorchXRayVision baseline") from error
    return xrv


def import_webdataset() -> Any:
    try:
        import webdataset  # type: ignore
    except ImportError as error:
        raise RuntimeError("webdataset is required for the WebDataset baseline") from error
    return webdataset


def import_dali() -> dict[str, Any]:
    try:
        import nvidia.dali.fn as fn  # type: ignore
        import nvidia.dali.types as types  # type: ignore
        from nvidia.dali import pipeline_def  # type: ignore
        from nvidia.dali.plugin.pytorch import DALIGenericIterator  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "NVIDIA DALI is required for the DALI baseline; install "
            "nvidia-dali-cuda130 for CUDA 13 or nvidia-dali-cuda120 for CUDA 12"
        ) from error
    try:
        from nvidia.dali.plugin.pytorch import LastBatchPolicy  # type: ignore
    except ImportError:
        LastBatchPolicy = None
    return {
        "fn": fn,
        "types": types,
        "pipeline_def": pipeline_def,
        "DALIGenericIterator": DALIGenericIterator,
        "LastBatchPolicy": LastBatchPolicy,
    }


def import_datasets() -> Any:
    try:
        import datasets  # type: ignore
    except ImportError as error:
        raise RuntimeError("datasets is required to load the Hugging Face CXR dataset") from error
    return datasets


def import_pillow() -> dict[str, Any]:
    try:
        from PIL import Image  # type: ignore
    except ImportError as error:
        raise RuntimeError("Pillow is required for image preprocessing") from error
    return {"Image": Image}


def import_pydicom() -> Any:
    try:
        import pydicom  # type: ignore
    except ImportError as error:
        raise RuntimeError("pydicom is required for DICOM-backed CXR benchmarks") from error
    return pydicom


def import_sklearn_metrics() -> Any:
    try:
        from sklearn import metrics  # type: ignore
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for AUROC/AUPRC metrics") from error
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
