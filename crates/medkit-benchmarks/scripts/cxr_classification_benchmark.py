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
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_DATASET = "arudaev/chest-xray-14-320"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a real CXR classification benchmark and emit report artifacts."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
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
    data_dir = args.work_dir / "materialized"
    manifest_path = args.work_dir / "manifest.jsonl"
    split_path = args.work_dir / "splits.json"
    cache_dir = args.work_dir / f"cache-{cache_size}-{args.cache_dtype}"
    webdataset_dir = args.work_dir / f"webdataset-{cache_size}"

    run_metadata = {
        "run_id": run_id,
        "dataset_requested": args.dataset,
        "manifest_requested": str(args.manifest) if args.manifest else None,
        "splits_requested": str(args.splits) if args.splits else None,
        "plan_requested": str(args.plan) if args.plan else None,
        "dataset_kind": "NIH ChestX-ray14 320px Hugging Face parquet subset",
        "primary_plan_dataset": "MIMIC-CXR-JPG",
        "dataset_deviation": (
            "No local credentialed MIMIC-CXR-JPG data was available. This run uses "
            "a public NIH ChestX-ray14 320px dataset so the pipeline can execute "
            "against real CXR images."
        ),
        "targets": targets,
        "uncertain_policy": args.uncertain,
        "missing_policy": "mask_missing",
        "loss_pos_weight": args.loss_pos_weight,
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

    cache_start = time.perf_counter()
    cache_metadata_path = cache_dir / "cache-metadata.json"
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
        "memory": memory,
        "ground_truth": ground_truth,
        "predictions": reports["predictions"],
        "train_order": reports["train_order"],
        "train_schedule": train_schedule_report,
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

    groups = list(by_patient.values())
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
    view_counts = {"unknown": len(records)}
    target_counts = {
        target: {"positive": 0, "negative": 0, "missing": 0, "uncertain": 0}
        for target in targets
    }
    for record in records:
        by_split[record.split] = by_split.get(record.split, 0) + 1
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
        "source_format": "png",
        "metadata_limitations": [
            "NIH filename-derived patient ids are used because the HF parquet subset exposes filename, labels, and image only.",
            "View position is not exposed by this dataset export.",
            "Uncertain labels are not exposed by this NIH export.",
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
    seed: int,
) -> dict[str, Any]:
    numpy = import_numpy()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    transform = {
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
    transform_hash = stable_hash(transform)

    train_records = [record for record in records if record.split == "train"]
    mean, std = estimate_mean_std(train_records, image_size)
    build_start = time.perf_counter()
    split_reports: dict[str, Any] = {}
    failed: list[str] = []
    for split in ("train", "val", "test"):
        split_records = [record for record in records if record.split == split]
        images_path = cache_dir / f"{split}-images.{cache_dtype}.dat"
        labels_path = cache_dir / f"{split}-labels.float32.dat"
        masks_path = cache_dir / f"{split}-masks.float32.dat"
        metadata_path = cache_dir / f"{split}-metadata.jsonl"
        shape = (len(split_records), 1, image_size, image_size)
        images = numpy.memmap(images_path, dtype=cache_dtype, mode="w+", shape=shape)
        labels = numpy.zeros((len(split_records), len(targets)), dtype="float32")
        masks = numpy.zeros((len(split_records), len(targets)), dtype="float32")
        with metadata_path.open("w", encoding="utf-8") as handle:
            for index, record in enumerate(split_records):
                try:
                    if cache_dtype == "uint8":
                        images[index, 0, :, :] = load_resized_grayscale(
                            record.image_path, image_size
                        )
                    else:
                        images[index, 0, :, :] = preprocess_image_to_numpy(
                            record.image_path,
                            image_size=image_size,
                            mean=mean,
                            std=std,
                        ).astype(cache_dtype)
                except Exception as error:
                    failed.append(f"{record.sample_id}: {error}")
                    continue
                for target_index, target in enumerate(targets):
                    value = record.labels.get(target)
                    if value is None or value == -1:
                        masks[index, target_index] = 0.0
                        labels[index, target_index] = 0.0
                    else:
                        masks[index, target_index] = 1.0
                        labels[index, target_index] = float(value)
                handle.write(json.dumps(record_to_json(record), sort_keys=True) + "\n")
        images.flush()
        labels.tofile(labels_path)
        masks.tofile(masks_path)
        split_reports[split] = {
            "samples": len(split_records),
            "images_path": str(images_path),
            "images_sha256": hash_file(images_path),
            "labels_path": str(labels_path),
            "labels_sha256": hash_file(labels_path),
            "masks_path": str(masks_path),
            "masks_sha256": hash_file(masks_path),
            "metadata_path": str(metadata_path),
            "metadata_sha256": hash_file(metadata_path),
            "shape": list(shape),
            "image_bytes": images_path.stat().st_size if images_path.exists() else 0,
        }

    report = {
        "cache_schema_version": 1,
        "report_schema_version": 1,
        "cache_dir": str(cache_dir),
        "cache_kind": f"medkit_rust_compatible_mmap_{cache_dtype}",
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

    def __len__(self) -> int:
        index = min(self.iteration_index, max(len(self.schedule.iteration_batches) - 1, 0))
        if not self.schedule.iteration_batches:
            return 0
        return len(self.schedule.iteration_batches[index])

    def report_metadata(self) -> dict[str, Any]:
        summary = self.schedule.summary()
        return {
            "paired_train_order": True,
            "batch_schedule": "fixed_by_iteration",
            "batch_schedule_hash": (summary.get("hashes") or {}).get("schedule_hash"),
            "batch_schedule_iteration_count": summary.get("iteration_count"),
            "batch_schedule_iteration_names": summary.get("iteration_names"),
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
    pillow = import_pillow()
    Image = pillow["Image"]
    image = Image.open(path).convert("L")
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
        class_pos_weight_values(records, targets)
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
                loss_pos_weight_values=loss_pos_weight_values,
                loss_pos_weight_mode=args.loss_pos_weight,
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
            sampler = FixedTrainBatchScheduleSampler(train_batch_schedule)
            # Multiprocess DataLoader prefetch can consume fixed schedule
            # iterations ahead of the training loop. Keep paired train-order
            # runs single-process so the recorded order is the executed order.
            scheduled_num_workers = 0
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=scheduled_num_workers,
                pin_memory=pin_memory,
                persistent_workers=False,
            )
            schedule_metadata = sampler.report_metadata()
            return with_report_metadata(
                loader,
                {
                    "baseline": baseline_name,
                    "batch_size": batch_size,
                    "cache_dtype": cache_dtype_from_metadata(cache_dir),
                    "read_mode": read_mode,
                    "shuffle_block_batches": shuffle_block_batches,
                    "shuffle": shuffle,
                    "drop_last": drop_last_for_split(split),
                    "num_workers": scheduled_num_workers,
                    "requested_num_workers": num_workers,
                    "pin_memory": pin_memory,
                    "worker_mode": "paired_schedule_single_process",
                    "native_prefetch": False,
                    **(metadata or {}),
                    **schedule_metadata,
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
                "view_position": "unknown",
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
            "view_position": "unknown",
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
            "view_position": "unknown",
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
            "view_position": "unknown",
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
    loss_pos_weight_values: Sequence[float] | None,
    loss_pos_weight_mode: str,
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
    model = make_model(torch, len(targets))
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
    train_start = time.perf_counter()
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
                raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
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
    evaluation = evaluate_model(
        torch=torch,
        model=model,
        loader=val_loader,
        device=device,
        max_batches=max_eval_batches,
        channels_last=channels_last_active,
        fallback_records=eval_records,
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
        "loss_pos_weight": loss_pos_weight_mode,
        "loss_pos_weight_values": (
            [float(value) for value in loss_pos_weight_values]
            if loss_pos_weight_values is not None
            else None
        ),
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
    output: dict[str, Any] = {
        f"{prefix}_native_prefetch_batches": stats_batches,
        f"{prefix}_native_prefetch_indexed_batches": indexed_batches,
        f"{prefix}_native_prefetch_indexed_runs": indexed_runs,
        f"{prefix}_native_prefetch_read_ms": read_ms,
        f"{prefix}_native_prefetch_scatter_ms": scatter_ms,
        f"{prefix}_native_prefetch_read_scatter_ms": read_scatter_ms,
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
        weights.append(float(negative / positive) if positive > 0 else 1.0)
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
            raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
                pos_weight=pos_weight,
            )
            loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
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


def make_model(torch: Any, num_targets: int) -> Any:
    try:
        torchvision = import_torchvision()
        model = torchvision.models.densenet121(weights=None)
        model.features.conv0 = torch.nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_targets)
        return model
    except Exception:
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
        )


def evaluate_model(
    *,
    torch: Any,
    model: Any,
    loader: Any,
    device: Any,
    max_batches: int,
    channels_last: bool,
    fallback_records: Sequence[SampleRecord] = (),
) -> EvaluationOutputs:
    numpy = import_numpy()
    model.eval()
    y_true: list[Any] = []
    y_score: list[Any] = []
    y_mask: list[Any] = []
    y_logits: list[Any] = []
    samples: list[dict[str, Any]] = []
    eval_offset = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            image = image_to_float_on_device(
                torch,
                batch["image"],
                device,
                channels_last=channels_last,
            )
            logits = model(image)
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
        )
    return EvaluationOutputs(
        y_true=numpy.concatenate(y_true),
        y_score=numpy.concatenate(y_score),
        y_mask=numpy.concatenate(y_mask),
        y_logits=numpy.concatenate(y_logits),
        samples=samples,
    )


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
    return {
        "threshold": float(threshold),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


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
        "loss_pos_weight": run_metadata.get("loss_pos_weight"),
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
        "loss_pos_weight",
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
        "source_format": "png",
        "modality": "CR",
        "view_position": "unknown",
        "laterality": None,
        "width": record.width,
        "height": record.height,
        "photometric_interpretation": "MONOCHROME2",
        "labels": record.labels,
        "label_source": "nih_chestxray14_nlp_labels",
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
    baselines = sorted(
        {
            *reports.get("loader", {}).keys(),
            *reports.get("gpu", {}).keys(),
            *reports.get("profile", {}).keys(),
            *reports.get("quality", {}).keys(),
            *reports.get("train_order", {}).keys(),
        }
    )
    rows = {
        baseline: training_ground_truth_row(
            baseline=baseline,
            loader=reports.get("loader", {}).get(baseline, {}),
            gpu=reports.get("gpu", {}).get(baseline, {}),
            profile=reports.get("profile", {}).get(baseline, {}),
            quality=reports.get("quality", {}).get(baseline, {}),
            predictions=((reports.get("predictions", {}).get("baselines") or {}).get(baseline, {})),
            train_order=((reports.get("train_order", {}).get("baselines") or {}).get(baseline, {})),
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
    return {
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
        if name in {"nvidia-dali-cuda130", "nvidia-dali-cuda120"}:
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


def import_sklearn_metrics() -> Any:
    try:
        from sklearn import metrics  # type: ignore
    except ImportError as error:
        raise RuntimeError("scikit-learn is required for AUROC/AUPRC metrics") from error
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
