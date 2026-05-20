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
    parser.add_argument("--read-mode", choices=("mmap", "stream"), default="mmap")
    parser.add_argument("--include-metadata", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--baselines",
        default="pytorch_raw,monai_raw,medkit_cached_mmap,medkit_pinned_prefetch",
    )
    parser.add_argument("--uncertain", default="ignore")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rematerialize", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
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
        "image_size": args.image_size,
        "cache_image_size": cache_size,
        "cache_dtype": args.cache_dtype,
        "batch_size": args.batch_size,
        "drop_last_train": args.drop_last_train,
        "workers": args.workers,
        "prefetch_depth": args.prefetch_depth,
        "prefetch_read_workers": args.prefetch_read_workers,
        "profile_batches": args.profile_batches,
        "read_mode": args.read_mode,
        "include_metadata": args.include_metadata,
        "epochs": args.epochs,
        "baselines": baselines,
        "seed": args.seed,
    }

    materialize_start = time.perf_counter()
    if args.force_rematerialize or not manifest_path.exists():
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
        records = load_manifest(manifest_path)
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
    split_report = write_split_file(split_path, records)
    split_seconds = time.perf_counter() - split_start

    manifest_summary = build_manifest_summary(records, targets, run_metadata)
    manifest_summary["manifest_build_seconds"] = materialize_seconds
    manifest_summary["split_build_seconds"] = split_seconds
    write_json(report_dir / "manifest-summary.json", manifest_summary)

    validation = validate_records(records, targets)
    write_validation(report_dir / "validation.md", validation, run_metadata)
    write_json(report_dir / "split-audit.json", split_report | validation["split_audit"])

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
    )

    write_json(report_dir / "loader-throughput.json", reports["loader"])
    write_json(report_dir / "gpu-throughput.json", reports["gpu"])
    if any(report.get("status") != "disabled" for report in reports["profile"].values()):
        write_json(report_dir / "step-profile.json", reports["profile"])
    write_json(report_dir / "model-quality.json", reports["quality"])
    write_json(report_dir / "threshold-report.json", reports["thresholds"])
    memory = memory_summary(reports)
    write_json(report_dir / "memory-summary.json", memory)
    write_json(report_dir / "subgroup-report.json", subgroup_report(records, reports["quality"]))

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
        "profile": {
            name: report.get("summary")
            for name, report in reports["profile"].items()
            if report.get("status") == "ok"
        },
        "memory": memory,
    }
    write_json(report_dir / "run-summary.json", summary)
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
) -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {
        "loader": {},
        "gpu": {},
        "profile": {},
        "quality": {},
        "thresholds": {},
    }
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
                read_mode=args.read_mode,
                include_metadata=args.include_metadata,
                seed=args.seed,
            )
        except Exception as error:
            unavailable = {"status": "unavailable", "reason": str(error)}
            reports["loader"][baseline] = unavailable
            reports["gpu"][baseline] = unavailable
            reports["quality"][baseline] = unavailable
            reports["thresholds"][baseline] = unavailable
            continue

        try:
            train_loader = loader_factory("train", shuffle=False)
            val_loader = loader_factory("val", shuffle=False)
            reports["loader"][baseline] = benchmark_loader(
                train_loader,
                max_batches=args.loader_batches,
                baseline=baseline,
            )
            train_report, quality_report, threshold_report, profile_report = train_and_evaluate(
                torch=torch,
                baseline=baseline,
                train_loader=loader_factory("train", shuffle=True),
                val_loader=val_loader,
                targets=targets,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                max_train_batches=args.max_train_batches,
                max_eval_batches=args.max_eval_batches,
                drop_last_train=args.drop_last_train,
                warmup_batches=args.warmup_batches,
                profile_batches=args.profile_batches,
                seed=args.seed,
            )
            reports["gpu"][baseline] = train_report
            reports["profile"][baseline] = profile_report
            reports["quality"][baseline] = quality_report
            reports["thresholds"][baseline] = threshold_report
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
    return reports


def with_report_metadata(loader: Any, metadata: dict[str, Any]) -> Any:
    loader.report_metadata = lambda metadata=metadata: dict(metadata)
    return loader


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
    read_mode: str,
    include_metadata: bool,
    seed: int,
) -> Any:
    torch = import_torch()
    numpy = import_numpy()
    mean, std = cache_normalization(cache_dir)
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
        return lambda split, shuffle=False: torch.utils.data.DataLoader(
            dataset_by_split[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=False,
            persistent_workers=workers > 0,
        )
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
        return lambda split, shuffle=False: torch.utils.data.DataLoader(
            dataset_by_split[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=False,
            persistent_workers=workers > 0,
        )
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
        return lambda split, shuffle=False: torch.utils.data.DataLoader(
            dataset_by_split[split],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=False,
            persistent_workers=workers > 0,
        )
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
            loader = torch.utils.data.DataLoader(
                dataset_by_split[split],
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0 if resident else workers,
                pin_memory=pin_memory,
                persistent_workers=(not resident and workers > 0),
            )
            return with_report_metadata(
                loader,
                {
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
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            return with_report_metadata(
                loader,
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
                    "shuffle": shuffle,
                    "native_prefetch": False,
                },
            )

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
            )
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
            return with_report_metadata(
                loader,
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
                    "shuffle": shuffle,
                    "native_prefetch": True,
                    "native_prefetch_threads": 1,
                },
            )

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
        "sample_id": metadata.get("sample_id", ""),
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
                "sample_id": record.sample_id,
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
            "sample_id": record.sample_id,
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
            "sample_id": record.get("sample_id", ""),
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
            "sample_id": record.sample_id,
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
        self.sample_ids = [record.sample_id for record in self.records]
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
            sample_ids=self.sample_ids,
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
        sample_ids: Sequence[str],
        view_positions: Sequence[str],
    ) -> None:
        self.iterator = iterator
        self.labels = labels
        self.masks = masks
        self.patient_ids = list(patient_ids)
        self.sample_ids = list(sample_ids)
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
            "sample_id": [self.sample_ids[index] for index in index_list],
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
        report["pipeline"] = loader.report_metadata()
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
    targets: Sequence[str],
    device: Any,
    epochs: int,
    batch_size: int,
    max_train_batches: int,
    max_eval_batches: int,
    drop_last_train: bool,
    warmup_batches: int,
    profile_batches: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    set_torch_seed(torch, seed)
    model = make_model(torch, len(targets)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    autocast_enabled = device.type == "cuda"
    h2d_timing_mode = H2D_TIMING_DIRECT_COPY
    if warmup_batches > 0:
        run_warmup_steps(
            torch=torch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=train_loader,
            device=device,
            batches=warmup_batches,
            autocast_enabled=autocast_enabled,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
    losses: list[float] = []
    data_wait_seconds = 0.0
    step_seconds = 0.0
    h2d_bytes = 0
    max_batch_tensor_bytes = 0
    profile_records: list[dict[str, Any]] = []
    samples = 0
    batches = 0
    skipped_incomplete_batches = 0
    skipped_incomplete_samples = 0
    train_start = time.perf_counter()
    model.train()
    for _epoch in range(epochs):
        iterator = iter(train_loader)
        while True:
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
            step_start = time.perf_counter()
            profile_this_batch = profile_batches > 0 and len(profile_records) < profile_batches
            cuda_profile = profile_this_batch and device.type == "cuda"
            h2d_start = h2d_end = None
            forward_start = forward_end = None
            backward_start = backward_end = None
            optimizer_start = optimizer_end = None

            max_batch_tensor_bytes = max(max_batch_tensor_bytes, batch_tensor_bytes(batch))
            h2d_wall_start = time.perf_counter()
            if cuda_profile:
                h2d_start = torch.cuda.Event(enable_timing=True)
                h2d_end = torch.cuda.Event(enable_timing=True)
                h2d_start.record()
            image = batch["image"].to(device, non_blocking=True).float()
            labels = batch["labels"].to(device, non_blocking=True).float()
            mask = batch["mask"].to(device, non_blocking=True).float()
            if cuda_profile:
                h2d_end.record()
            h2d_wall_ms = (time.perf_counter() - h2d_wall_start) * 1000.0
            h2d_bytes += (
                batch["image"].numel() * 4
                + batch["labels"].numel() * 4
                + batch["mask"].numel() * 4
            )
            optimizer.zero_grad(set_to_none=True)
            forward_wall_start = time.perf_counter()
            if cuda_profile:
                forward_start = torch.cuda.Event(enable_timing=True)
                forward_end = torch.cuda.Event(enable_timing=True)
                forward_start.record()
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                logits = model(image)
                raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
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
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_elapsed = time.perf_counter() - step_start
            step_seconds += step_elapsed
            losses.append(float(loss.detach().cpu().item()))
            samples += batch_samples
            if profile_this_batch:
                if cuda_profile:
                    h2d_ms = float(h2d_start.elapsed_time(h2d_end))
                    forward_ms = float(forward_start.elapsed_time(forward_end))
                    backward_ms = float(backward_start.elapsed_time(backward_end))
                    optimizer_ms = float(optimizer_start.elapsed_time(optimizer_end))
                else:
                    h2d_ms = h2d_wall_ms
                    forward_ms = forward_wall_ms
                    backward_ms = backward_wall_ms
                    optimizer_ms = optimizer_wall_ms
                profile_records.append(
                    {
                        "batch_index": batches,
                        "samples": batch_samples,
                        "data_wait_ms": wait_seconds * 1000.0,
                        "h2d_ms": h2d_ms,
                        "h2d_timing_mode": h2d_timing_mode,
                        "forward_ms": forward_ms,
                        "backward_ms": backward_ms,
                        "optimizer_ms": optimizer_ms,
                        "total_step_ms": step_elapsed * 1000.0,
                    }
                )
            batches += 1
            if max_train_batches > 0 and batches >= max_train_batches:
                break
        if max_train_batches > 0 and batches >= max_train_batches:
            break
    total_seconds = time.perf_counter() - train_start
    y_true, y_score, y_mask = evaluate_model(
        torch=torch,
        model=model,
        loader=val_loader,
        device=device,
        max_batches=max_eval_batches,
    )
    quality = metric_report(y_true, y_score, y_mask, targets)
    quality["status"] = "ok"
    quality["baseline"] = baseline
    thresholds = threshold_report(y_true, y_score, y_mask, targets)
    thresholds["status"] = "ok"
    thresholds["baseline"] = baseline
    train_report = {
        "status": "ok",
        "baseline": baseline,
        "device": str(device),
        "samples": samples,
        "batches": batches,
        "epochs_requested": epochs,
        "train_batch_size": batch_size,
        "drop_last_train": drop_last_train,
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
    profile_report = profile_report_for_baseline(
        baseline=baseline,
        requested_batches=profile_batches,
        records=profile_records,
        h2d_timing_mode=h2d_timing_mode,
    )
    if profile_report.get("status") == "ok":
        train_report.update(profile_report["summary"])
    if hasattr(train_loader, "report_metadata"):
        train_report["pipeline"] = train_loader.report_metadata()
    train_report["memory"] = memory_snapshot(
        pipeline=train_report.get("pipeline"),
        max_batch_tensor_bytes=max_batch_tensor_bytes,
    )
    return train_report, quality, thresholds, profile_report


def batch_sample_count(batch: dict[str, Any]) -> int:
    return int(batch["image"].shape[0])


def should_skip_incomplete_train_batch(
    *,
    batch_samples: int,
    batch_size: int,
    drop_last_train: bool,
) -> bool:
    return drop_last_train and batch_size > 0 and batch_samples != batch_size


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
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ):
        if field not in required_profile_fields and not any(field in record for record in records):
            continue
        values = [float(record.get(field, 0.0)) for record in records]
        stats = profile_stats(values)
        summary[f"profile_{field}_mean"] = stats["mean"]
        summary[f"profile_{field}_p50"] = stats["p50"]
        summary[f"profile_{field}_p95"] = stats["p95"]
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
) -> None:
    model.train()
    for batch_index, batch in enumerate(loader):
        image = batch["image"].to(device, non_blocking=True).float()
        labels = batch["labels"].to(device, non_blocking=True).float()
        mask = batch["mask"].to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            logits = model(image)
            raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1.0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_index + 1 >= batches:
            break


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
) -> tuple[Any, Any, Any]:
    numpy = import_numpy()
    model.eval()
    y_true: list[Any] = []
    y_score: list[Any] = []
    y_mask: list[Any] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            image = batch["image"].to(device, non_blocking=True).float()
            logits = model(image)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_score.append(probs)
            y_true.append(tensor_to_numpy(batch["labels"]))
            y_mask.append(tensor_to_numpy(batch["mask"]))
            if max_batches > 0 and batch_index + 1 >= max_batches:
                break
    if not y_true:
        empty = numpy.zeros((0, 0), dtype="float32")
        return empty, empty, empty
    return numpy.concatenate(y_true), numpy.concatenate(y_score), numpy.concatenate(y_mask)


def tensor_to_numpy(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    if hasattr(value, "numpy"):
        return value.numpy().copy()
    return import_numpy().asarray(value).copy()


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
    return metrics


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
